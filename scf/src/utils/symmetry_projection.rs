#![allow(warnings)]

use control::Control;
use crystal::Crystal;
use dfttypes::*;
use ewald::Ewald;
use gvector::GVector;
use types::Matrix;
use ndarray::Array3;
use pspot::PSPot;
use pwdensity::PWDensity;
use symmetry::SymmetryDriver;
use types::c64;
use types::Vector3f64;

// Re-export symmetry projection functions from force and stress crates
pub(crate) use force::symmetry::finalize_force_by_parts;
pub(crate) use stress::symmetry::finalize_stress_by_parts;

fn flatten_vec3(v: &[Vector3f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for item in v {
        out.extend_from_slice(item.as_slice());
    }
    out
}

fn scatter_vec3(flat: &[f64], out: &mut [Vector3f64]) {
    debug_assert_eq!(flat.len(), out.len() * 3);
    for (dst, chunk) in out.iter_mut().zip(flat.chunks_exact(3)) {
        dst.x = chunk[0];
        dst.y = chunk[1];
        dst.z = chunk[2];
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

    let force_vnl_local_flat = flatten_vec3(&force_vnl_local);
    let mut force_vnl_flat = vec![0.0; force_vnl_local_flat.len()];
    dwmpi::reduce_slice_sum(
        force_vnl_local_flat.as_slice(),
        force_vnl_flat.as_mut_slice(),
        dwmpi::comm_world(),
    );

    dwmpi::bcast_slice(force_vnl_flat.as_mut_slice(), dwmpi::comm_world());
    scatter_vec3(&force_vnl_flat, &mut force_vnl);

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
        dwmpi::comm_world(),
    );

    dwmpi::bcast_slice(stress_kin.as_mut_slice(), dwmpi::comm_world());

    let mut stress_vnl_local = stress::vnl(
        crystal,
        &vkscf.as_non_spin().unwrap(),
        &vkevecs.as_non_spin().unwrap(),
    );

    let mut stress_vnl = Matrix::new(3, 3);

    dwmpi::reduce_slice_sum(
        stress_vnl_local.as_slice(),
        stress_vnl.as_mut_slice(),
        dwmpi::comm_world(),
    );

    dwmpi::bcast_slice(stress_vnl.as_mut_slice(), dwmpi::comm_world());

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
