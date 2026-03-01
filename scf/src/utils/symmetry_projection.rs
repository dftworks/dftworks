#![allow(warnings)]

use control::Control;
use crystal::Crystal;
use dfttypes::*;
use ewald::Ewald;
use gvector::GVector;
use lattice::Lattice;
use matrix::Matrix;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::Array3;
use pspot::PSPot;
use pwdensity::PWDensity;
use symmetry::SymmetryDriver;
use types::c64;
use vector3::Vector3f64;
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
    force_vdw: &mut [Vector3f64],
) {
    if control.get_symmetry() {
        project_force_by_symmetry(crystal, symdrv, force_loc);
        project_force_by_symmetry(crystal, symdrv, force_vnl);
        project_force_by_symmetry(crystal, symdrv, force_nlcc);
        project_force_by_symmetry(crystal, symdrv, force_ewald);
        project_force_by_symmetry(crystal, symdrv, force_vdw);
    }

    for iat in 0..force_total.len() {
        force_total[iat] = force_ewald[iat] + force_loc[iat] + force_vnl[iat] + force_nlcc[iat] + force_vdw[iat];
    }

    if dwmpi::is_root() {
        force::display(
            crystal,
            force_total,
            force_ewald,
            force_loc,
            force_vnl,
            force_nlcc,
            force_vdw,
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
    stress_vdw: &mut Matrix<f64>,
) {
    if control.get_symmetry() {
        project_stress_by_symmetry(crystal, symdrv, stress_kin);
        project_stress_by_symmetry(crystal, symdrv, stress_hartree);
        project_stress_by_symmetry(crystal, symdrv, stress_xc);
        project_stress_by_symmetry(crystal, symdrv, stress_xc_nlcc);
        project_stress_by_symmetry(crystal, symdrv, stress_loc);
        project_stress_by_symmetry(crystal, symdrv, stress_vnl);
        project_stress_by_symmetry(crystal, symdrv, stress_ewald);
        project_stress_by_symmetry(crystal, symdrv, stress_vdw);
    }

    for i in 0..3 {
        for j in 0..3 {
            stress_total[[i, j]] = stress_kin[[i, j]]
                + stress_hartree[[i, j]]
                + stress_xc[[i, j]]
                + stress_xc_nlcc[[i, j]]
                + stress_loc[[i, j]]
                + stress_vnl[[i, j]]
                + stress_ewald[[i, j]]
                + stress_vdw[[i, j]];
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
            stress_vdw,
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

    let mut force_vdw = vec![Vector3f64::zeros(); natoms];
    force::vdw(control, crystal, &mut force_vdw);

    finalize_force_by_parts(
        control,
        crystal,
        symdrv,
        force_total.as_mut_slice(),
        force_ewald.as_mut_slice(),
        force_loc.as_mut_slice(),
        force_vnl.as_mut_slice(),
        force_nlcc.as_mut_slice(),
        force_vdw.as_mut_slice(),
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

    let mut stress_vdw = Matrix::new(3, 3);
    stress::vdw(control, crystal, &mut stress_vdw);

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
        &mut stress_vdw,
    );
}
