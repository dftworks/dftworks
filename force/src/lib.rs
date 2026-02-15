//#![allow(warnings)]

use atompsp::*;
use crystal::*;
use dwconsts::*;
use gvector::*;
use kgylm::KGYLM;
use kscf::*;
use lattice::*;
use matrix::*;
use num_traits::identities::Zero;
use pspot::*;
use pwbasis::*;
use pwdensity::*;
use types::*;
use vector3::*;

use itertools::multizip;

// Force decomposition utilities.
//
// Total ionic force is assembled from:
// - local ionic term
// - non-local pseudopotential term
// - Ewald ion-ion term
// - NLCC correction term involving v_xc
//
// SCF drivers compute each component separately and add them for diagnostics.

// Phys. Rev. B 41, 7876 (1990)
pub fn nlcc_xc(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    vxcg: &[c64],
    force: &mut [Vector3f64],
) {
    // NLCC force starts from zero and accumulates per atom.
    force.iter_mut().for_each(|x| x.set_zeros());

    let volume = crystal.get_latt().volume();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let gidx = pwden.get_gindex();

    let npw_rho = pwden.get_n_plane_waves();
    let mut atrhocoreg = vec![0.0; npw_rho];

    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();

    let species = crystal.get_atom_species();

    for iat in 0..natoms {
        // Complex accumulator in reciprocal space, converted to real at the end.
        let mut v = Vector3c64::zeros();

        let atom = atom_positions[iat];

        let atpsp = atpsps.get_psp(&species[iat]);

        rhocore_of_g_one_atom(atpsp, pwden, volume, &mut atrhocoreg);

        for i in 1..npw_rho {
            let mill = miller[gidx[i]];

            // let ngd =
            //-TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let ngd = -TWOPI * utility::dot_product_v3i32_v3f64(mill, atom);

            let comm = c64 {
                re: ngd.cos(),
                im: ngd.sin(),
            } * atrhocoreg[i]
                * vxcg[i].conj();

            let gcoord = cart[gidx[i]];

            v.x += I_C64 * gcoord.x * comm * volume;
            v.y += I_C64 * gcoord.y * comm * volume;
            v.z += I_C64 * gcoord.z * comm * volume;
        }

        // Physical force is the real part.
        force[iat].x = v.x.re;
        force[iat].y = v.y.re;
        force[iat].z = v.z.re;
    }
}

pub fn vpsloc(
    atpsps: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    rhog: &[c64],
    force: &mut [Vector3f64],
) {
    // Local ionic force in reciprocal space.
    force.iter_mut().for_each(|x| x.set_zeros());

    let volume = crystal.get_latt().volume();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let gidx = pwden.get_gindex();

    let npw_rho = pwden.get_n_plane_waves();
    let mut vatlocg = vec![0.0; npw_rho];

    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();

    let species = crystal.get_atom_species();

    for iat in 0..natoms {
        // Complex accumulator in reciprocal space, converted to real at the end.
        let mut v = Vector3c64::zeros();

        let atom = atom_positions[iat];

        let atpsp = atpsps.get_psp(&species[iat]);

        vpsloc_of_g_one_atom(atpsp, pwden, volume, &mut vatlocg);

        for i in 1..npw_rho {
            let mill = miller[gidx[i]];

            //let ngd =
            //-TWOPI * (atom.x * mill.x as f64 + atom.y * mill.y as f64 + atom.z * mill.z as f64);

            let ngd = -TWOPI * utility::dot_product_v3i32_v3f64(mill, atom);

            let comm = c64 {
                re: ngd.cos(),
                im: ngd.sin(),
            } * vatlocg[i]
                * rhog[i].conj();

            let gcoord = cart[gidx[i]];

            v.x += I_C64 * gcoord.x * comm * volume;
            v.y += I_C64 * gcoord.y * comm * volume;
            v.z += I_C64 * gcoord.z * comm * volume;
        }

        force[iat].x = v.x.re;
        force[iat].y = v.y.re;
        force[iat].z = v.z.re;
    }
}

pub fn vnl(crystal: &Crystal, vkscf: &[KSCF], vevecs: &[Matrix<c64>], force: &mut [Vector3f64]) {
    // Non-local force is summed over k points and weighted by k-point weights.
    let natoms = crystal.get_n_atoms();
    let atom_positions = crystal.get_atom_positions();
    let atom_species = crystal.get_atom_species();

    for (ik, kscf) in vkscf.iter().enumerate() {
        let evecs = &vevecs[ik];

        let kylms = kscf.get_kgylm();

        let kgbeta_all = kscf.get_vnl().get_kgbeta_all();

        for iat in 0..natoms {
            let atom = atom_positions[iat];
            let atpsp = kscf.get_pspot().get_psp(&atom_species[iat]);

            let kgbeta = kgbeta_all.get(&atom_species[iat]).unwrap();

            let f = vnl_of_one_atom_one_k(
                atpsp,
                atom,
                kscf.get_gvec(),
                kscf.get_pwwfc(),
                kgbeta,
                kylms,
                kscf,
                evecs,
            );

            force[iat].x += f.x * kscf.get_k_weight();
            force[iat].y += f.y * kscf.get_k_weight();
            force[iat].z += f.z * kscf.get_k_weight();

            //println!("ik = {}, iat = {}, f = {:?}", ik, iat, f);
        }
    }
}

pub fn vnl_of_one_atom_one_k(
    atpsp: &dyn AtomPSP,
    atom_position: Vector3f64,
    gvec: &GVector,
    pwwfc: &PWBasis,
    vnlbeta: &[Vec<f64>],
    ylms: &KGYLM,
    kscf: &KSCF,
    evecs: &Matrix<c64>,
) -> Vector3f64 {
    // One-atom/one-k contribution to non-local KB force.
    let npw = pwwfc.get_n_plane_waves();
    let gidx = pwwfc.get_gindex();

    let occs = kscf.get_occ();

    let miller = gvec.get_miller();
    let cart = gvec.get_cart();

    let mut v = Vector3c64::zeros();

    for (ibnd, occ) in occs.iter().enumerate() {
        if *occ < EPS20 {
            continue;
        }

        let cnk = evecs.get_col(ibnd);

        let mut v_band = Vector3c64::zeros();

        let sfact = fhkl::compute_structure_factor_for_many_g_one_atom(miller, gidx, atom_position);

        for (j, beta) in vnlbeta.iter().enumerate() {
            let l = atpsp.get_lbeta(j);

            let dfact = atpsp.get_dfact(j);

            for m in utility::get_quant_num_m(l) {
                let ylm = ylms.get_data(l, m);

                let mut beta_kg_cnk = c64::zero();

                for i in 0..npw {
                    beta_kg_cnk += ylm[i] * beta[i] * sfact[i].conj() * cnk[i];
                }

                for i in 0..npw {
                    let gcoord = cart[gidx[i]];

                    let f = cnk[i].conj() * ylm[i] * beta[i] * sfact[i];

                    v_band.x += I_C64 * gcoord.x * f * beta_kg_cnk * dfact;
                    v_band.y += I_C64 * gcoord.y * f * beta_kg_cnk * dfact;
                    v_band.z += I_C64 * gcoord.z * f * beta_kg_cnk * dfact;
                }
            }
        }

        v.x += v_band.x * occ;
        v.y += v_band.y * occ;
        v.z += v_band.z * occ;
    }

    // Factor 2 accounts for complex-conjugate pair in this formulation.
    Vector3f64 {
        x: 2.0 * v.x.re,
        y: 2.0 * v.y.re,
        z: 2.0 * v.z.re,
    }
}

pub fn get_total(
    force: &mut [Vector3f64],
    force_ew: &[Vector3f64],
    force_vpsloc: &[Vector3f64],
    force_vnl: &[Vector3f64],
) {
    // Historical helper: total = Ewald + local + non-local.
    // (NLCC is added explicitly by SCF code paths that need it.)
    let natoms = force.len();

    for i in 0..natoms {
        force[i].x = force_ew[i].x + force_vpsloc[i].x + force_vnl[i].x;
        force[i].y = force_ew[i].y + force_vpsloc[i].y + force_vnl[i].y;
        force[i].z = force_ew[i].z + force_vpsloc[i].z + force_vnl[i].z;
    }
}

pub fn get_max_force(force: &[Vector3f64]) -> f64 {
    // Infinity norm over Cartesian components and atoms.
    let p_force = vector3::as_slice_of_element(force);

    p_force
        .iter()
        .max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap())
        .unwrap()
        .abs()

    // let mut v_max = 0.0;

    // for v in p_force.iter() {
    //     if v_max < v.abs() {
    //         v_max = v.abs();
    //     }
    // }

    // v_max
}

//fn frac_to_cart(latt: &Lattice, vf: &[Vector3f64], vc: &mut [Vector3f64]) {
//    let a = latt.get_vector_a();
//    let b = latt.get_vector_b();
//    let c = latt.get_vector_c();
//
//    for (tf, tc) in multizip((vf.iter(), vc.iter_mut())) {
//        tc.x = a.x * tf.x + b.x * tf.y + c.x * tf.z;
//        tc.y = a.y * tf.x + b.y * tf.y + c.y * tf.z;
//        tc.z = a.z * tf.x + b.z * tf.y + c.z * tf.z;
//    }
//}

pub fn cart_to_frac(latt: &Lattice, vc: &[Vector3f64], vf: &mut [Vector3f64]) {
    let mut mat = latt.as_matrix().clone();

    mat.inv();

    mat = mat.transpose();

    for (tc, tf) in multizip((vc.iter(), vf.iter_mut())) {
        tf.x = mat[[0, 0]] * tc.x + mat[[1, 0]] * tc.y + mat[[2, 0]] * tc.z;
        tf.y = mat[[0, 1]] * tc.x + mat[[1, 1]] * tc.y + mat[[2, 1]] * tc.z;
        tf.z = mat[[0, 2]] * tc.x + mat[[1, 2]] * tc.y + mat[[2, 2]] * tc.z;
    }
}

fn rhocore_of_g_one_atom(
    atompsp: &dyn AtomPSP,
    pwden: &PWDensity,
    volume: f64,
    rhocoreg: &mut [f64],
) {
    let ffact_rhocore = nlcc::rhocore_of_g_on_shells(atompsp, pwden, volume);

    let npw_rho = pwden.get_n_plane_waves();

    let gshell_index = pwden.get_gshell_index();

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        rhocoreg[i] = ffact_rhocore[ish];
    }
}

fn vpsloc_of_g_one_atom(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64, vlocg: &mut [f64]) {
    let ffact_vloc = vloc::vloc_of_g_on_shells(atompsp, pwden, volume);

    let npw_rho = pwden.get_n_plane_waves();

    let gshell_index = pwden.get_gshell_index();

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        vlocg[i] = ffact_vloc[ish];
    }
}

fn printout_force_atoms(crystal: &Crystal, force: &[Vector3f64]) {
    println!();

    let a = crystal.get_latt().get_vector_a();
    let b = crystal.get_latt().get_vector_b();
    let c = crystal.get_latt().get_vector_c();

    let atom_positions = crystal.get_atom_positions();
    let species = crystal.get_atom_species();

    for (iat, f) in force.iter().enumerate() {
        let atom = &atom_positions[iat];

        let atc_x = a.x * atom.x + b.x * atom.y + c.x * atom.z;
        let atc_y = a.y * atom.x + b.y * atom.y + c.y * atom.z;
        let atc_z = a.z * atom.x + b.z * atom.y + c.z * atom.z;

        println!(
            "    {:<3} {:<4} : {:>16.6} {:>16.6} {:>16.6} {:20.6}  {:20.6}  {:20.6}",
            iat + 1,
            species[iat],
            f.x * FORCE_HA_TO_EV,
            f.y * FORCE_HA_TO_EV,
            f.z * FORCE_HA_TO_EV,
            atc_x * BOHR_TO_ANG,
            atc_y * BOHR_TO_ANG,
            atc_z * BOHR_TO_ANG
        );
    }
}

fn printout_force(crystal: &Crystal, force: &[Vector3f64]) {
    println!();

    let species = crystal.get_atom_species();

    for (iat, f) in force.iter().enumerate() {
        println!(
            "    {:<3} {:<4} : {:>16.6} {:>16.6} {:>16.6}",
            iat + 1,
            species[iat],
            f.x * FORCE_HA_TO_EV,
            f.y * FORCE_HA_TO_EV,
            f.z * FORCE_HA_TO_EV
        );
    }
}

pub fn display(
    crystal: &Crystal,
    force_total: &[Vector3f64],
    force_ewald: &[Vector3f64],
    force_loc: &[Vector3f64],
    force_vnl: &[Vector3f64],
    force_nlcc: &[Vector3f64],
) {
    println!("\n");
    println!(
        "   {:-^64}    {:-^60}",
        " total-force (cartesian) (eV/A) ", " atomic-positions (cartesian) (A) "
    );

    printout_force_atoms(crystal, force_total);

    println!("\n   {:-^64}", " local ");

    printout_force(crystal, force_loc);

    println!("\n   {:-^64}", " non-local ");

    printout_force(crystal, force_vnl);

    println!("\n   {:-^64}", " Ewald ");

    printout_force(crystal, force_ewald);

    println!("\n   {:-^64}", " nlcc ");

    printout_force(crystal, force_nlcc);
}
