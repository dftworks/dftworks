#![allow(warnings)]

use super::hartree;
use control::*;
use crystal::Crystal;
use density::*;
use dfttypes::*;
use dwconsts::*;
use ewald::Ewald;
use fftgrid::FFTGrid;
use gvector::*;
use itertools::multizip;
use kpts::KPTS;
use kscf::KSCF;
use lattice::Lattice;
use matrix::Matrix;
use mixing::Mixing;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::*;
use num_traits::Zero;
use pspot::PSPot;
use pwbasis::*;
use pwdensity::*;
use rayon::prelude::*;
use rgtransform::RGTransform;
use std::io::Write;
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;
use xc::*;

const PARALLEL_MIN_LEN: usize = 8192;

#[inline]
fn use_parallel_for_len(len: usize) -> bool {
    len >= PARALLEL_MIN_LEN && rayon::current_num_threads() > 1
}

pub fn compute_v_hartree(pwden: &PWDensity, rhog: &RHOG, vhg: &mut [c64]) {
    if let RHOG::NonSpin(rhog) = rhog {
        hartree::potential(pwden.get_g(), rhog, vhg);
    }
}

#[inline]
pub fn has_external_slab_fields(control: &Control) -> bool {
    control.get_electric_field_2d().abs() > EPS16 || control.get_surface_dipole_correction()
}

#[inline]
fn wrapped_centered_fraction(u: f64, origin: f64) -> f64 {
    let mut t = u - origin;
    t -= t.floor();
    if t >= 0.5 {
        t -= 1.0;
    }
    t
}

#[inline]
fn axis_fraction_from_linear_index(
    axis: ElectricFieldAxis,
    idx: usize,
    n1: usize,
    n2: usize,
    n3: usize,
) -> f64 {
    let i = idx % n1;
    let j = (idx / n1) % n2;
    let k = idx / (n1 * n2);
    match axis {
        ElectricFieldAxis::A => i as f64 / n1 as f64,
        ElectricFieldAxis::B => j as f64 / n2 as f64,
        ElectricFieldAxis::C => k as f64 / n3 as f64,
    }
}

#[inline]
fn axis_fraction_from_position(axis: ElectricFieldAxis, pos: &Vector3f64) -> f64 {
    match axis {
        ElectricFieldAxis::A => pos.x,
        ElectricFieldAxis::B => pos.y,
        ElectricFieldAxis::C => pos.z,
    }
}

#[inline]
fn axis_length_bohr(axis: ElectricFieldAxis, crystal: &Crystal) -> f64 {
    match axis {
        ElectricFieldAxis::A => crystal.get_latt().get_vector_a().norm2(),
        ElectricFieldAxis::B => crystal.get_latt().get_vector_b().norm2(),
        ElectricFieldAxis::C => crystal.get_latt().get_vector_c().norm2(),
    }
}

#[inline]
fn rho_real_total_at(rho_3d: &RHOR, idx: usize) -> f64 {
    match rho_3d {
        RHOR::NonSpin(rho) => rho.as_slice()[idx].re,
        RHOR::Spin(rho_up, rho_dn) => rho_up.as_slice()[idx].re + rho_dn.as_slice()[idx].re,
    }
}

pub fn build_external_slab_potential(
    control: &Control,
    crystal: &Crystal,
    zions: &[f64],
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    rho_3d: &RHOR,
    vext_3d: &mut Array3<c64>,
    vextg: &mut [c64],
) -> bool {
    if !has_external_slab_fields(control) {
        for v in vext_3d.as_mut_slice().iter_mut() {
            *v = c64::zero();
        }
        for v in vextg.iter_mut() {
            *v = c64::zero();
        }
        return false;
    }

    let axis = control.get_electric_field_axis_enum();
    let origin = control.get_electric_field_origin_frac();
    let efield_au = control.get_electric_field_2d();
    let axis_len = axis_length_bohr(axis, crystal);
    let nfft = vext_3d.as_slice().len();
    let [n1, n2, n3] = vext_3d.shape();
    debug_assert_eq!(nfft, n1 * n2 * n3);

    let dvol = crystal.get_latt().volume() / nfft as f64;
    let mut dipole_electron = 0.0;
    for i in 0..nfft {
        let u = axis_fraction_from_linear_index(axis, i, n1, n2, n3);
        let ds = wrapped_centered_fraction(u, origin) * axis_len;
        dipole_electron -= rho_real_total_at(rho_3d, i) * ds * dvol;
    }

    let mut dipole_ion = 0.0;
    debug_assert_eq!(zions.len(), crystal.get_n_atoms());
    for (iat, pos_f) in crystal.get_atom_positions().iter().enumerate() {
        let u = axis_fraction_from_position(axis, pos_f);
        let ds = wrapped_centered_fraction(u, origin) * axis_len;
        dipole_ion += zions.get(iat).copied().unwrap_or(0.0) * ds;
    }

    let dipole_field = if control.get_surface_dipole_correction() {
        -FOURPI * (dipole_electron + dipole_ion) / crystal.get_latt().volume()
    } else {
        0.0
    };

    let mut mean = 0.0;
    for i in 0..nfft {
        let u = axis_fraction_from_linear_index(axis, i, n1, n2, n3);
        let ds = wrapped_centered_fraction(u, origin) * axis_len;
        let v = (efield_au + dipole_field) * ds;
        vext_3d.as_mut_slice()[i] = c64::new(v, 0.0);
        mean += v;
    }
    mean /= nfft as f64;
    for v in vext_3d.as_mut_slice().iter_mut() {
        v.re -= mean;
    }

    rgtrans.r3d_to_g1d(gvec, pwden, vext_3d.as_slice(), vextg);
    true
}

#[inline]
pub fn add_external_potential_to_vlocg(vextg: &[c64], vlocg: &mut [c64]) {
    debug_assert_eq!(vextg.len(), vlocg.len());
    for (vloc, vext) in vlocg.iter_mut().zip(vextg.iter()) {
        *vloc += *vext;
    }
}

pub fn display_parallel_runtime_info() {
    if !dwmpi::is_root() {
        return;
    }

    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let mpi_ranks = dwmpi::get_comm_world_size();

    println!(
        "     {:<width1$} = {:>4} (RAYON_NUM_THREADS={}, host_threads={}, mpi_ranks={})",
        "rayon_threads",
        rayon_threads,
        rayon_env,
        host_threads,
        mpi_ranks,
        width1 = OUT_WIDTH1
    );
}

pub fn validate_hse06_runtime_constraints(control: &Control, kpts: &dyn KPTS) {
    if !matches!(control.get_xc_scheme_enum(), XcScheme::Hse06) {
        return;
    }

    if kpts.get_n_kpts() != 1 {
        panic!("xc_scheme='hse06' currently supports only a single Gamma k-point");
    }

    let k0 = kpts.get_k_frac(0);
    if k0.norm2() > 1.0E-10 {
        panic!(
            "xc_scheme='hse06' currently supports only Gamma point (in.kmesh must map to k=(0,0,0))"
        );
    }

    if dwmpi::is_root() {
        println!(
            "     NOTE: hse06 currently includes screened exact-exchange in the SCF Hamiltonian; force/stress do not include the hybrid exchange term yet."
        );
    }
}

pub fn display_external_field_runtime_note(control: &Control) {
    if !has_external_slab_fields(control) || !dwmpi::is_root() {
        return;
    }

    println!(
        "     NOTE: electric_field_2d/surface_dipole_correction is enabled; force/stress currently exclude explicit ionic external-field terms."
    );
}

// v_xc in r space first and then transform to G space; this will change with the density

pub fn compute_v_e_xc_of_r(
    xc: &dyn XC,
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    vxc_3d: &mut VXCR,
    exc_3d: &mut Array3<c64>,
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        // For NLCC we evaluate XC with the total charge seen by the functional:
        // rho_total = rho_valence + rho_core.
        rho_3d.add_from(rhocore_3d);
    }

    // XC implementation now owns the complete GGA derivative workflow
    // (including gradients/divergences) and directly returns:
    //   - vxc_3d(r): variational XC potential
    //   - exc_3d(r): energy density per particle
    xc.potential_and_energy(gvec, pwden, rgtrans, rho_3d, vxc_3d, exc_3d);

    if let RHOR::NonSpin(rho_3d) = rho_3d {
        // Restore rho_3d to pure valence density for downstream routines.
        rho_3d.substract(rhocore_3d);
    }
}

pub fn compute_v_xc_of_g(
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    vxc_3d: &VXCR,
    vxcg: &mut VXCG,
) {
    let vxc_3d = vxc_3d.as_non_spin().unwrap();
    let vxcg = vxcg.as_non_spin_mut().unwrap();

    rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d.as_slice(), vxcg);
}

// v_xc + v_h + v_psloc in G space

pub fn add_up_v(vpslocg: &[c64], vhg: &[c64], vxcg: &VXCG, vlocg: &mut [c64]) {
    let vxcg = vxcg.as_non_spin().unwrap();
    debug_assert_eq!(vpslocg.len(), vlocg.len());
    debug_assert_eq!(vhg.len(), vlocg.len());
    debug_assert_eq!(vxcg.len(), vlocg.len());

    if use_parallel_for_len(vlocg.len()) {
        vlocg
            .par_iter_mut()
            .zip(vxcg.par_iter())
            .zip(vhg.par_iter())
            .zip(vpslocg.par_iter())
            .for_each(|(((v_loc, v_xc), v_ha), v_psloc)| {
                *v_loc = *v_xc + *v_ha + *v_psloc;
            });
    } else {
        for (v_loc, v_xc, v_ha, v_psloc) in
            multizip((vlocg.iter_mut(), vxcg.iter(), vhg.iter(), vpslocg.iter()))
        {
            *v_loc = *v_xc + *v_ha + *v_psloc;
        }
    }
}

pub fn compute_rho_of_g(
    gvec: &GVector,
    pwden: &PWDensity,
    rgtrans: &RGTransform,
    rho_3d: &mut RHOR,
    rhog_out: &mut [c64],
) {
    if let RHOR::NonSpin(rho_3d) = rho_3d {
        rgtrans.r3d_to_g1d(gvec, pwden, rho_3d.as_slice(), rhog_out);
    }
}

pub fn compute_next_density(
    pwden: &PWDensity,
    mixing: &mut dyn Mixing,
    rhog_out: &[c64],
    rhog_diff: &mut [c64],
    rhog: &mut RHOG,
) {
    if let RHOG::NonSpin(rhog) = rhog {
        debug_assert_eq!(rhog_out.len(), rhog_diff.len());
        debug_assert_eq!(rhog.len(), rhog_diff.len());

        // mix old and new densities to get the density for the next iteration
        if use_parallel_for_len(rhog_diff.len()) {
            rhog_diff
                .par_iter_mut()
                .zip(rhog_out.par_iter())
                .zip(rhog.par_iter())
                .for_each(|((d, out), old)| {
                    *d = *out - *old;
                });
        } else {
            for ipw in 0..rhog_diff.len() {
                rhog_diff[ipw] = rhog_out[ipw] - rhog[ipw];
            }
        }

        mixing.compute_next_density(pwden.get_g(), rhog, rhog_diff);
    }
}

pub fn display_eigen_values(
    verbosity: VerbosityLevel,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    vpwwfc: &[PWBasis],
    vkscf: &VKSCF,
    vkevals: &VKEigenValue,
) {
    if matches!(verbosity, VerbosityLevel::Quiet) {
        return;
    }

    let blatt = crystal.get_latt().reciprocal();

    let t_vkscf = vkscf.as_non_spin().unwrap();
    let t_vkevals = vkevals.as_non_spin().unwrap();

    let rank = dwmpi::get_comm_world_rank();
    let ordered_rank_output = verbosity >= VerbosityLevel::Verbose;

    if ordered_rank_output {
        for irank in 0..dwmpi::get_comm_world_size() {
            dwmpi::barrier(MPI_COMM_WORLD);

            if irank == rank {
                debug_assert_eq!(t_vkscf.len(), t_vkevals.len());
                debug_assert_eq!(t_vkscf.len(), vpwwfc.len());

                for (kscf_k, evals, pwwfc_k) in itertools::multizip((
                    t_vkscf.iter(),
                    t_vkevals.iter(),
                    vpwwfc.iter(),
                )) {
                    let g_ik = kscf_k.get_ik();
                    let k_frac = kpts.get_k_frac(g_ik);
                    let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                    let npw_wfc = pwwfc_k.get_n_plane_waves();

                    print_k_point(g_ik, k_frac, k_cart, npw_wfc);

                    let occ = kscf_k.get_occ();

                    print_eigen_values(evals, occ);
                }

                std::io::stdout().flush();
            }
        }

        dwmpi::barrier(MPI_COMM_WORLD);
    } else if dwmpi::is_root() {
        // Production path: avoid rank-serialized eigenvalue output.
        // Ordered rank-by-rank dumps are available with verbosity=verbose/debug.
        debug_assert_eq!(t_vkscf.len(), t_vkevals.len());
        debug_assert_eq!(t_vkscf.len(), vpwwfc.len());

        for (kscf_k, evals, pwwfc_k) in itertools::multizip((
            t_vkscf.iter(),
            t_vkevals.iter(),
            vpwwfc.iter(),
        )) {
            let g_ik = kscf_k.get_ik();
            let k_frac = kpts.get_k_frac(g_ik);
            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
            let npw_wfc = pwwfc_k.get_n_plane_waves();

            print_k_point(g_ik, k_frac, k_cart, npw_wfc);

            let occ = kscf_k.get_occ();

            print_eigen_values(evals, occ);
        }

        std::io::stdout().flush();
    }
}

pub fn print_eigen_values(v: &[f64], occ: &[f64]) {
    println!();

    for (i, elem) in v.iter().enumerate() {
        println!(
            "       {:<6} {:16.6} {:12.6}",
            i + 1,
            elem * HA_TO_EV,
            occ[i]
        );
    }
}

pub fn print_k_point(ik: usize, xk_frac: Vector3f64, xk_cart: Vector3f64, npw_wfc: usize) {
    println!();

    println!("   kpoint-{} npws = {}", ik + 1, npw_wfc);

    println!(
        "     k_frac = [ {:.8}, {:.8}, {:.8} ]",
        xk_frac.x, xk_frac.y, xk_frac.z
    );

    println!(
        "     k_cart = [ {:.8}, {:.8}, {:.8} ] (1/a0)",
        xk_cart.x, xk_cart.y, xk_cart.z
    );
}

fn transpose_i32_3x3(m: &[[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut out = [[0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    out
}

fn transpose_f64_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
    }
    out
}

fn matmul_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn matvec_3x3(a: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2];
    }
    out
}

fn rotation_cart_from_fractional(latt: &Lattice, rotation_frac: &[[i32; 3]; 3]) -> [[f64; 3]; 3] {
    let mut rot_cart = [[0.0; 3]; 3];

    for col in 0..3 {
        let mut e_cart = [0.0; 3];
        e_cart[col] = 1.0;

        let mut e_frac = [0.0; 3];
        latt.cart_to_frac(&e_cart, &mut e_frac);

        let mut e_frac_rot = [0.0; 3];
        for i in 0..3 {
            e_frac_rot[i] = rotation_frac[i][0] as f64 * e_frac[0]
                + rotation_frac[i][1] as f64 * e_frac[1]
                + rotation_frac[i][2] as f64 * e_frac[2];
        }

        let mut e_cart_rot = [0.0; 3];
        latt.frac_to_cart(&e_frac_rot, &mut e_cart_rot);

        for row in 0..3 {
            rot_cart[row][col] = e_cart_rot[row];
        }
    }

    rot_cart
}

fn matrix_to_array_3x3(m: &Matrix<f64>) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[[i, j]];
        }
    }
    out
}

fn array_3x3_to_matrix(a: &[[f64; 3]; 3], m: &mut Matrix<f64>) {
    for i in 0..3 {
        for j in 0..3 {
            m[[i, j]] = a[i][j];
        }
    }
}

pub(crate) fn project_force_by_symmetry(
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    force: &mut [Vector3f64],
) {
    let natoms = force.len();
    let n_sym = symdrv.get_n_sym_ops();
    if natoms == 0 || n_sym == 0 {
        return;
    }

    let sym_atom = symdrv.get_sym_atom();
    if sym_atom.len() != natoms || sym_atom.iter().any(|row| row.len() != n_sym) {
        return;
    }

    let latt = crystal.get_latt();
    let mut rot_inv_cart = Vec::<[[f64; 3]; 3]>::with_capacity(n_sym);
    for isym in 0..n_sym {
        let rot_inv_frac = transpose_i32_3x3(symdrv.get_rotation(isym));
        rot_inv_cart.push(rotation_cart_from_fractional(latt, &rot_inv_frac));
    }

    let force_raw = force.to_vec();
    for iat in 0..natoms {
        let mut sum = [0.0; 3];
        let mut n_accum = 0usize;

        for isym in 0..n_sym {
            let jat = sym_atom[iat][isym];
            if jat >= natoms {
                continue;
            }

            let fj = [force_raw[jat].x, force_raw[jat].y, force_raw[jat].z];
            let fi = matvec_3x3(&rot_inv_cart[isym], &fj);

            sum[0] += fi[0];
            sum[1] += fi[1];
            sum[2] += fi[2];
            n_accum += 1;
        }

        if n_accum > 0 {
            let inv = 1.0 / n_accum as f64;
            force[iat].x = sum[0] * inv;
            force[iat].y = sum[1] * inv;
            force[iat].z = sum[2] * inv;
        }
    }
}

pub(crate) fn project_stress_by_symmetry(
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    stress: &mut Matrix<f64>,
) {
    let n_sym = symdrv.get_n_sym_ops();
    if n_sym == 0 {
        return;
    }

    let latt = crystal.get_latt();
    let stress_raw = matrix_to_array_3x3(stress);
    let mut stress_proj = [[0.0; 3]; 3];

    for isym in 0..n_sym {
        let rot_cart = rotation_cart_from_fractional(latt, symdrv.get_rotation(isym));
        let rot_cart_t = transpose_f64_3x3(&rot_cart);
        let tmp = matmul_3x3(&rot_cart, &stress_raw);
        let s_rot = matmul_3x3(&tmp, &rot_cart_t);

        for i in 0..3 {
            for j in 0..3 {
                stress_proj[i][j] += s_rot[i][j];
            }
        }
    }

    let inv = 1.0 / n_sym as f64;
    for i in 0..3 {
        for j in 0..3 {
            stress_proj[i][j] *= inv;
        }
    }

    array_3x3_to_matrix(&stress_proj, stress);
    stress.symmetrize();
}

pub(crate) fn finalize_force_by_parts(
    control: &Control,
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    force_total: &mut [Vector3f64],
    force_ewald: &mut [Vector3f64],
    force_loc: &mut [Vector3f64],
    force_vnl: &mut [Vector3f64],
    force_nlcc: &mut [Vector3f64],
) {
    if control.get_symmetry() {
        project_force_by_symmetry(crystal, symdrv, force_loc);
        project_force_by_symmetry(crystal, symdrv, force_vnl);
        project_force_by_symmetry(crystal, symdrv, force_nlcc);
        project_force_by_symmetry(crystal, symdrv, force_ewald);
    }

    for iat in 0..force_total.len() {
        force_total[iat] = force_ewald[iat] + force_loc[iat] + force_vnl[iat] + force_nlcc[iat];
    }

    if dwmpi::is_root() {
        force::display(
            crystal,
            force_total,
            force_ewald,
            force_loc,
            force_vnl,
            force_nlcc,
        );
    }
}

pub(crate) fn finalize_stress_by_parts(
    control: &Control,
    crystal: &Crystal,
    symdrv: &dyn SymmetryDriver,
    stress_total: &mut Matrix<f64>,
    stress_kin: &mut Matrix<f64>,
    stress_hartree: &mut Matrix<f64>,
    stress_xc: &mut Matrix<f64>,
    stress_xc_nlcc: &mut Matrix<f64>,
    stress_loc: &mut Matrix<f64>,
    stress_vnl: &mut Matrix<f64>,
    stress_ewald: &mut Matrix<f64>,
) {
    if control.get_symmetry() {
        project_stress_by_symmetry(crystal, symdrv, stress_kin);
        project_stress_by_symmetry(crystal, symdrv, stress_hartree);
        project_stress_by_symmetry(crystal, symdrv, stress_xc);
        project_stress_by_symmetry(crystal, symdrv, stress_xc_nlcc);
        project_stress_by_symmetry(crystal, symdrv, stress_loc);
        project_stress_by_symmetry(crystal, symdrv, stress_vnl);
        project_stress_by_symmetry(crystal, symdrv, stress_ewald);
    }

    for i in 0..3 {
        for j in 0..3 {
            stress_total[[i, j]] = stress_kin[[i, j]]
                + stress_hartree[[i, j]]
                + stress_xc[[i, j]]
                + stress_xc_nlcc[[i, j]]
                + stress_loc[[i, j]]
                + stress_vnl[[i, j]]
                + stress_ewald[[i, j]];
        }
    }

    if dwmpi::is_root() {
        stress::display_stress_by_parts(
            stress_kin,
            stress_hartree,
            stress_xc,
            stress_xc_nlcc,
            stress_loc,
            stress_vnl,
            stress_ewald,
            stress_total,
        );
    }
}

pub fn compute_force(
    control: &Control,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    pots: &PSPot,
    ewald: &Ewald,
    vkscf: &VKSCF,
    vkevecs: &VKEigenVector,
    rhog: &RHOG,
    vxcg: &VXCG,
    symdrv: &dyn SymmetryDriver,
    force_total: &mut Vec<Vector3f64>,
) {
    let natoms = crystal.get_n_atoms();

    let mut force_loc = vec![Vector3f64::zeros(); natoms];
    let mut force_vnl_local = vec![Vector3f64::zeros(); natoms];
    let mut force_vnl = vec![Vector3f64::zeros(); natoms];
    let mut force_spectral_ws = force::SpectralWorkspace::new();

    force::vpsloc_with_workspace(
        pots,
        crystal,
        gvec,
        pwden,
        &mut force_spectral_ws,
        rhog.as_non_spin().unwrap(),
        &mut force_loc,
    );

    force::vnl(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
        &mut force_vnl_local,
    );

    dwmpi::reduce_slice_sum(
        vector3::as_slice_of_element(&force_vnl_local),
        vector3::as_mut_slice_of_element(&mut force_vnl),
        MPI_COMM_WORLD,
    );

    dwmpi::bcast_slice(
        vector3::as_mut_slice_of_element(&mut force_vnl),
        MPI_COMM_WORLD,
    );

    let mut force_ewald = ewald.get_force().to_vec();

    let mut force_nlcc = vec![Vector3f64::zeros(); natoms];

    force::nlcc_xc_with_workspace(
        pots,
        crystal,
        gvec,
        pwden,
        &mut force_spectral_ws,
        vxcg.as_non_spin().unwrap(),
        &mut force_nlcc,
    );
    finalize_force_by_parts(
        control,
        crystal,
        symdrv,
        force_total.as_mut_slice(),
        force_ewald.as_mut_slice(),
        force_loc.as_mut_slice(),
        force_vnl.as_mut_slice(),
        force_nlcc.as_mut_slice(),
    );
}

pub fn compute_stress(
    control: &Control,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    pots: &PSPot,
    ewald: &Ewald,
    vkscf: &VKSCF,
    vkevecs: &VKEigenVector,
    rhog: &RHOG,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    vxcg: &VXCG,
    vxc_3d: &VXCR,
    exc_3d: &Array3<c64>,
    symdrv: &dyn SymmetryDriver,
    stress_total: &mut Matrix<f64>,
) {
    let mut stress_kin_local = stress::kinetic(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

    let mut stress_kin = Matrix::new(3, 3);

    dwmpi::reduce_slice_sum(
        stress_kin_local.as_slice(),
        stress_kin.as_mut_slice(),
        MPI_COMM_WORLD,
    );

    dwmpi::bcast_slice(stress_kin.as_mut_slice(), MPI_COMM_WORLD);

    let mut stress_vnl_local = stress::vnl(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

    let mut stress_vnl = Matrix::new(3, 3);

    dwmpi::reduce_slice_sum(
        stress_vnl_local.as_slice(),
        stress_vnl.as_mut_slice(),
        MPI_COMM_WORLD,
    );

    dwmpi::bcast_slice(stress_vnl.as_mut_slice(), MPI_COMM_WORLD);

    let mut stress_hartree = stress::hartree(gvec, pwden, rhog.as_non_spin().unwrap());
    let mut stress_spectral_ws = stress::SpectralWorkspace::new();
    let mut stress_loc = Matrix::new(3, 3);
    stress::vpsloc_with_workspace(
        pots,
        crystal,
        gvec,
        pwden,
        &mut stress_spectral_ws,
        rhog.as_non_spin().unwrap(),
        &mut stress_loc,
    );

    let mut stress_xc = stress::xc(
        crystal.get_latt(),
        rho_3d.as_non_spin_mut().unwrap(),
        rhocore_3d,
        vxc_3d.as_non_spin().unwrap(),
        &exc_3d,
    );

    let mut stress_xc_nlcc = Matrix::new(3, 3);
    stress::nlcc_xc_with_workspace(
        pots,
        crystal,
        gvec,
        pwden,
        &mut stress_spectral_ws,
        vxcg.as_non_spin().unwrap(),
        &mut stress_xc_nlcc,
    );

    let mut stress_ewald = ewald.get_stress().clone();
    finalize_stress_by_parts(
        control,
        crystal,
        symdrv,
        stress_total,
        &mut stress_kin,
        &mut stress_hartree,
        &mut stress_xc,
        &mut stress_xc_nlcc,
        &mut stress_loc,
        &mut stress_vnl,
        &mut stress_ewald,
    );
}

pub fn get_n_plane_waves_max(vpwwfc: &[PWBasis]) -> usize {
    let mut npw_max = 0;
    for pwwfc in vpwwfc.iter() {
        let npw = pwwfc.get_n_plane_waves();

        if npw > npw_max {
            npw_max = npw;
        }
    }

    npw_max
}

pub fn solve_eigen_equations(
    rgtrans: &RGTransform,
    vloc_3d: &Array3<c64>,
    eigvalue_epsilon: f64,
    geom_iter: usize,
    scf_iter: usize,
    vkscf: &VKSCF,
    vkevals: &mut VKEigenValue,
    vkevecs: &mut VKEigenVector,
) {
    let t_vkscf = vkscf.as_non_spin().unwrap();
    let t_vkevecs = vkevecs.as_non_spin_mut().unwrap();
    let t_vkevals = vkevals.as_non_spin_mut().unwrap();

    for (ik, kscf) in t_vkscf.iter().enumerate() {
        kscf.run(
            rgtrans,
            vloc_3d,
            eigvalue_epsilon,
            geom_iter,
            scf_iter,
            &mut t_vkevals[ik],
            &mut t_vkevecs[ik],
        );
    }
}

pub fn get_eigvalue_epsilon(
    geom_iter: usize,
    scf_iter: usize,
    control: &Control,
    ntot_elec: f64,
    energy_diff: f64,
    npw_wfc: usize,
) -> f64 {
    let mut eig_epsilon: f64;

    //if control.is_band() {
    if control.get_scf_max_iter() <= 1 {
        eig_epsilon = control.get_eigval_epsilon();
    } else {
        if geom_iter == 1 {
            match scf_iter {
                1 => {
                    eig_epsilon = EPS2 * EV_TO_HA;
                }

                2 => {
                    eig_epsilon = EPS3 * EV_TO_HA;
                }

                3 => {
                    eig_epsilon = EPS4 * EV_TO_HA;
                }

                _ => {
                    eig_epsilon =
                        (EPS3 * EV_TO_HA).min(0.0001 * energy_diff / (1.0_f64).max(ntot_elec));

                    eig_epsilon = eig_epsilon
                        .max(EPS13 * EV_TO_HA)
                        .min(control.get_eigval_epsilon());
                }
            }
        } else {
            match scf_iter {
                1 => {
                    eig_epsilon = EPS2 * EV_TO_HA;
                }

                2 => {
                    eig_epsilon = EPS4 * EV_TO_HA;
                }

                3 => {
                    eig_epsilon = EPS6 * EV_TO_HA;
                }

                _ => {
                    eig_epsilon = (EPS11 * EV_TO_HA)
                        .min(energy_diff / (npw_wfc as f64).powf(1.0) / (1.0_f64).max(ntot_elec));
                }
            }
        }

        eig_epsilon = eig_epsilon.max(EPS16 * EV_TO_HA);
    }

    eig_epsilon
}

pub fn compute_total_energy(
    pwden: &PWDensity,
    crystal: &Crystal,
    rhog: &[c64],
    vkscf: &[KSCF],
    vevals: &[Vec<f64>],
    rho_3d: &mut Array3<c64>,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
    vxc_3d: &Array3<c64>,
    vext_3d: Option<&Array3<c64>>,
    ew_total: f64,
    hubbard_energy: f64,
) -> f64 {
    let latt = crystal.get_latt();

    let etot_hartree = energy::hartree(pwden, latt, rhog);

    // bands energy

    let etot_bands_local = get_bands_energy(vkscf, vevals);

    let mut etot_bands = 0.0;

    dwmpi::reduce_scalar_sum(&etot_bands_local, &mut etot_bands, MPI_COMM_WORLD);
    dwmpi::bcast_scalar(&mut etot_bands, MPI_COMM_WORLD);

    //

    let etot_xc = energy::exc(latt, &rho_3d, &rhocore_3d, &exc_3d);
    let etot_vxc = energy::vxc(
        latt,
        rho_3d.as_slice(),
        rhocore_3d.as_slice(),
        vxc_3d.as_slice(),
    );
    let etot_ext = if let Some(vext_3d) = vext_3d {
        let nfft = vext_3d.as_slice().len();
        let dvol = latt.volume() / nfft as f64;
        let mut sum = 0.0;
        for (rho, vext) in rho_3d.as_slice().iter().zip(vext_3d.as_slice().iter()) {
            sum += rho.re * vext.re;
        }
        sum * dvol
    } else {
        0.0
    };
    let hybrid_exchange = get_hybrid_exchange_energy(vkscf);

    let etot_one = etot_bands - etot_vxc - 2.0 * etot_hartree;

    // `etot_bands` already contains <Vx_hybrid>; subtract E_x^hybrid once to
    // remove the double counting and keep one copy in total energy.
    let etot =
        etot_one + etot_xc + etot_hartree + etot_ext + ew_total + hubbard_energy - hybrid_exchange;

    etot
}

// In QE
// hwf_energy = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband_hwf
// etot       = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband + descf

fn get_bands_energy(vkscf: &[KSCF], vevals: &[Vec<f64>]) -> f64 {
    let etot_bands = energy::band_structure(vkscf, vevals);

    etot_bands
}

fn get_hybrid_exchange_energy(vkscf: &[KSCF]) -> f64 {
    let local = vkscf
        .iter()
        .map(|kscf| kscf.get_hybrid_exchange_energy() * kscf.get_k_weight())
        .sum::<f64>();

    let mut total = 0.0;
    dwmpi::reduce_scalar_sum(&local, &mut total, MPI_COMM_WORLD);
    dwmpi::bcast_scalar(&mut total, MPI_COMM_WORLD);

    total
}
