#![allow(warnings)]

use crate::hartree;
use control::*;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use gvector::*;
use itertools::multizip;
use kpts::KPTS;
use ndarray::*;
use num_traits::Zero;
use pwdensity::*;
use rayon::prelude::*;
use rgtransform::RGTransform;
use types::*;
use types::Vector3f64;
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
        ElectricFieldAxis::A => crystal.get_latt().get_vector_a().norm(),
        ElectricFieldAxis::B => crystal.get_latt().get_vector_b().norm(),
        ElectricFieldAxis::C => crystal.get_latt().get_vector_c().norm(),
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

pub fn validate_hse06_runtime_constraints(
    control: &Control,
    kpts: &dyn KPTS,
) -> Result<(), String> {
    if !matches!(control.get_xc_scheme_enum(), XcScheme::Hse06) {
        return Ok(());
    }

    if kpts.get_n_kpts() != 1 {
        return Err("xc_scheme='hse06' currently supports only a single Gamma k-point".to_string());
    }

    let k0 = kpts.get_k_frac(0);
    if k0.norm() > 1.0E-10 {
        return Err(
            "xc_scheme='hse06' currently supports only Gamma point (in.kmesh must map to k=(0,0,0))"
                .to_string(),
        );
    }

    Ok(())
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
