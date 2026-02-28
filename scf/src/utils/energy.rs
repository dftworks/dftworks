#![allow(warnings)]

use crystal::Crystal;
use dfttypes::*;
use kscf::KSCF;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::Array3;
use pwdensity::PWDensity;
use types::c64;
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
