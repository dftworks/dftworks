#![allow(warnings)]
use atompsp::AtomPSP;
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use gvector::GVector;
use itertools::multizip;
use kgylm::KGYLM;
use kscf::KSCF;
use lattice::Lattice;
use matrix::Matrix;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use types::c64;
use vector3::Vector3f64;

pub fn kinetic(vkscf: &[KSCF], vevecs: &Vec<Matrix<c64>>) -> f64 {
    let mut etot = 0.0;

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evecs = &vevecs[ik];

        let nband = kscf.get_nbands();
        let occ = kscf.get_occ();

        let kg = kscf.get_pwwfc().get_kg();
        let npw = kscf.get_pwwfc().get_n_plane_waves();

        let mut etot_k = 0.0;

        for ibnd in 0..nband {
            if occ[ibnd] < EPS20 {
                continue;
            }

            let mut etot_band = 0.0;

            let cnk = evecs.get_col(ibnd);

            for i in 0..npw {
                etot_band += (cnk[i] * kg[i]).norm_sqr();
            }

            etot_k += etot_band * occ[ibnd];
        }

        etot += etot_k * kscf.get_k_weight();
    }

    etot *= 0.5;

    etot
}

pub fn vpsloc(crystal: &Crystal, vpslocg: &[c64], rhog: &[c64]) -> f64 {
    let mut etot_vpsloc = c64::zero();

    for (vps, rho) in multizip((vpslocg.iter(), rhog.iter())) {
        etot_vpsloc += vps * rho.conj();
    }

    etot_vpsloc *= crystal.get_latt().volume();

    etot_vpsloc.re
}

pub fn vnl(crystal: &Crystal, vkscf: &[KSCF], vevecs: &Vec<Matrix<c64>>) -> f64 {
    let mut etot = c64::zero();

    for (ik, kscf) in vkscf.iter().enumerate() {
        let mut etot_k = c64::zero();

        let evecs = &vevecs[ik];

        let kgylm = kscf.get_kgylm();

        let kgbeta_all = kscf.get_vnl().get_kgbeta_all();

        for (isp, specie) in crystal.get_unique_species().iter().enumerate() {
            let kgbeta = kgbeta_all.get(specie).unwrap();

            let atom_positions_for_this_specie = crystal.get_atom_positions_of_specie(isp);

            let atompsp = kscf.get_pspot().get_psp(specie);

            etot_k += vnl_of_one_specie_one_k(
                atompsp,
                &atom_positions_for_this_specie,
                kscf.get_gvec(),
                kscf.get_pwwfc(),
                kgbeta,
                kgylm,
                kscf,
                evecs,
            );
        }

        etot += etot_k * kscf.get_k_weight();
    }

    etot.re
}

pub fn vnl_of_one_specie_one_k(
    atpsp: &dyn AtomPSP,
    atom_positions: &Vec<Vector3f64>,
    gvec: &GVector,
    pwwfc: &PWBasis,
    vnlbeta: &Vec<Vec<f64>>,
    kgylm: &KGYLM,
    kscf: &KSCF,
    evecs: &Matrix<c64>,
) -> c64 {
    let gindex = pwwfc.get_gindex();

    let nbeta = atpsp.get_nbeta();

    let npw = pwwfc.get_n_plane_waves();

    let mut sfact: Vec<c64> = vec![c64::zero(); npw];

    let occ = kscf.get_occ();

    let mut etot = c64::zero();

    for ibnd in 0..kscf.get_nbands() {
        if occ[ibnd] < EPS20 {
            continue;
        }

        let cnk = evecs.get_col(ibnd);

        let mut etot_band = c64::zero();

        for atom in atom_positions {
            let sfact = fhkl::compute_structure_factor_for_many_g_one_atom(
                gvec.get_miller(),
                pwwfc.get_gindex(),
                *atom,
            );

            for j in 0..nbeta {
                let l = atpsp.get_lbeta(j);
                let beta = &vnlbeta[j];
                let dfact = atpsp.get_dfact(j);

                for m in utility::get_quant_num_m(l) {
                    let ylm = kgylm.get_data(l, m);

                    let mut beta_kg_cnk = c64::zero();

                    for i in 0..npw {
                        beta_kg_cnk += ylm[i] * beta[i] * sfact[i].conj() * cnk[i];
                    }

                    etot_band += dfact * beta_kg_cnk.norm_sqr();
                }
            }
        }

        etot += etot_band * occ[ibnd];
    }

    etot
}

// Phys. Rev. B 59, 11716 (1999)
// vxc_3d: the xc potential calculated with the sum of valence charge and core charge
// vxc: the potential energy of valence charge experienced in the potential vxc_3d
pub fn vxc(latt: &Lattice, rho_3d: &[c64], _rhocore_3d: &[c64], vxc_3d: &[c64]) -> f64 {
    let mut etot_vxc = c64::zero();

    for (r, v) in multizip((rho_3d.iter(), vxc_3d.iter())) {
        etot_vxc += r * v;
    }

    etot_vxc *= latt.volume() / rho_3d.len() as f64;

    etot_vxc.re
}

pub fn vxc_spin(latt: &Lattice, rho_3d: &RHOR, _rhocore_3d: &[c64], vxc_3d: &VXCR) -> f64 {
    let (rho_3d_up, rho_3d_dn) = rho_3d.as_spin().unwrap();
    let rho_3d_up = rho_3d_up.as_slice();
    let rho_3d_dn = rho_3d_dn.as_slice();

    let (vxc_3d_up, vxc_3d_dn) = vxc_3d.as_spin().unwrap();
    let vxc_3d_up = vxc_3d_up.as_slice();
    let vxc_3d_dn = vxc_3d_dn.as_slice();

    let mut etot_vxc_up = c64::zero();
    let mut etot_vxc_dn = c64::zero();

    for (r, v) in multizip((rho_3d_up.iter(), vxc_3d_up.iter())) {
        etot_vxc_up += r * v;
    }

    etot_vxc_up *= latt.volume() / rho_3d_up.len() as f64;

    for (r, v) in multizip((rho_3d_dn.iter(), vxc_3d_dn.iter())) {
        etot_vxc_dn += r * v;
    }

    etot_vxc_dn *= latt.volume() / rho_3d_dn.len() as f64;

    etot_vxc_up.re + etot_vxc_dn.re
}

// exc: the xc energy of the sum of valence charge and core charge
pub fn exc(
    latt: &Lattice,
    rho_3d: &Array3<c64>,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
) -> f64 {
    let rho = rho_3d.as_slice();
    let rhocore = rhocore_3d.as_slice();
    let exc = exc_3d.as_slice();

    let mut etot_xc = c64::zero();

    for (r, c, e) in multizip((rho.iter(), rhocore.iter(), exc.iter())) {
        etot_xc += (r + c) * e;
    }

    etot_xc *= latt.volume() / rho_3d.len() as f64;

    etot_xc.re
}

pub fn exc_spin(
    latt: &Lattice,
    rho_3d: &RHOR,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
) -> f64 {
    let (rho_up, rho_dn) = rho_3d.as_spin().unwrap();
    let rho_up = rho_up.as_slice();
    let rho_dn = rho_dn.as_slice();

    let mut etot_xc = c64::zero();

    let rhocore = rhocore_3d.as_slice();

    let exc = exc_3d.as_slice();

    for (up, dn, c, e) in multizip((rho_up.iter(), rho_dn.iter(), rhocore.iter(), exc.iter())) {
        etot_xc += (up + dn + c) * e;
    }

    etot_xc *= latt.volume() / rho_up.len() as f64;

    etot_xc.re
}

pub fn hartree(pwden: &PWDensity, latt: &Lattice, rhog: &[c64]) -> f64 {
    let npw_rho = pwden.get_n_plane_waves();

    let g_pwden = pwden.get_g();

    let mut etot_hartree = 0.0f64;

    for i in 1..npw_rho {
        let gg = g_pwden[i] * g_pwden[i];

        etot_hartree += rhog[i].norm_sqr() / gg;
    }

    etot_hartree *= 0.5 * latt.volume() * FOURPI;

    etot_hartree
}

pub fn band_structure(vkscf: &[KSCF], vevals: &Vec<Vec<f64>>) -> f64 {
    let mut etot_bands = 0.0;

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evals = &vevals[ik];

        etot_bands += kscf.get_band_structure_energy(evals) * kscf.get_k_weight();
    }

    etot_bands
}

pub fn output(
    ctrl: &Control,
    etot_bands: f64,
    etot_hartree: f64,
    etot_xc: f64,
    etot_one: f64,
    ew_total: f64,
    etot: f64,
    etot0: f64,
) {
    println!();
    println!("      {:-^30}", " total energy (Ry) ");
    println!();

    println!(
        "      {:<width1$} = {:>width2$.12}",
        "bands",
        etot_bands * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "Hartree (wo G0)",
        etot_hartree * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "E_xc",
        etot_xc * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "one_electron",
        etot_one * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "Ewald",
        ew_total * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "total_energy",
        etot * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "delta_energy",
        (etot - etot0).abs() * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
    println!(
        "      {:<width1$} = {:>width2$.12}",
        "delta_energy_set",
        ctrl.get_energy_epsilon() * HA_TO_RY,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH3
    );
}
