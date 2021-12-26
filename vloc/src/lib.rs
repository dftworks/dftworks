use atompsp::AtomPSP;
use crystal::Crystal;
use dwconsts::*;
use fhkl;
use gvector::GVector;
use itertools::multizip;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwdensity::PWDensity;
use types::*;
use vector3::*;

pub fn from_atomic_super_position(
    pspot: &PSPot,
    crystal: &Crystal,
    gvec: &GVector,
    pwden: &PWDensity,
    vpslocg: &mut [c64],
) {
    // 1D Vloc(G)

    atomic_super_position(pspot, crystal, pwden, gvec, vpslocg);
}

fn atomic_super_position(
    atpsps: &PSPot,
    crystal: &Crystal,
    pwden: &PWDensity,
    gvec: &GVector,
    vlocg: &mut [c64],
) {
    let volume = crystal.get_latt().volume();

    let species = crystal.get_unique_species();

    for (isp, sp) in species.iter().enumerate() {
        let atpsp = atpsps.get_psp(sp);

        let atom_positions = crystal.get_atom_positions_of_specie(isp);

        let vlocg_one = atom_super_pos_one_specie(atpsp, &atom_positions, pwden, gvec, volume);

        for (x, y) in multizip((vlocg_one.iter(), vlocg.iter_mut())) {
            *y += *x;
        }
    }
}

fn atom_super_pos_one_specie(
    atompsp: &dyn AtomPSP,
    atom_positions: &[Vector3f64],
    pwden: &PWDensity,
    gvec: &GVector,
    volume: f64,
) -> Vec<c64> {
    let miller = gvec.get_miller();

    let gindex = pwden.get_gindex();

    let gshell_index = pwden.get_gshell_index();

    let npw_rho = pwden.get_n_plane_waves();

    // structure factor

    let sfact = fhkl::compute_structure_factor(miller, gindex, atom_positions);

    // form factor on G shells

    let ffact_vloc = vloc_of_g_on_shells(atompsp, &pwden, volume);

    // form factor on G for rho

    let mut vlocg = vec![c64::zero(); npw_rho];

    for i in 0..npw_rho {
        let ish = gshell_index[i];
        vlocg[i] = ffact_vloc[ish] * sfact[i];
    }

    vlocg
}

pub fn vloc_of_g_on_shells(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64) -> Vec<f64> {
    let gshell = pwden.get_gshell_norms();
    let zion = atompsp.get_zion();
    let vloc = atompsp.get_vloc();
    let rad = atompsp.get_rad();
    let rab = atompsp.get_rab();

    compute_vloc_of_g(zion, vloc, rad, rab, gshell, volume)
}

fn compute_vloc_of_g(
    zion: f64,
    vloc: &[f64],
    rad: &[f64],
    rab: &[f64],
    gshell: &[f64],
    volume: f64,
) -> Vec<f64> {
    let nshell = gshell.len();

    let mut vg = vec![0.0; nshell];

    let mmax = vloc.len();

    let mut work = vec![0.0; mmax];

    // G = 0
    //
    // Here we neglect the divergent part which cancels with the (G = 0) part in Hartree.
    // So in the Hartree, we simply make the G = 0 term zero.
    //
    for i in 0..mmax {
        work[i] = rad[i] * (rad[i] * vloc[i] + zion);
    }

    //vg[0] = integral::simpson_log(&work, rad);
    vg[0] = integral::simpson_rab(&work, rab);

    // G > 0

    for iw in 1..nshell {
        let g = gshell[iw];
        let g2 = g * g;

        for i in 0..mmax {
            let r = rad[i];

            work[i] = (r * vloc[i] + zion * special::erf(r)) * (g * r).sin() / g;
        }

        let vh = zion * (-g2 / 4.0).exp() / g2;

        //vg[iw] = integral::simpson_log(&work, rad) - vh;
        vg[iw] = integral::simpson_rab(&work, rab) - vh;
    }

    vg.iter_mut().for_each(|v| *v *= FOURPI / volume);

    vg
}

// dvloc / d(G^2)
pub fn dvloc_of_g_on_shells(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64) -> Vec<f64> {
    let gshell = pwden.get_gshell_norms();
    let zion = atompsp.get_zion();
    let vloc = atompsp.get_vloc();
    let rad = atompsp.get_rad();
    let rab = atompsp.get_rab();

    compute_dvloc_of_g(zion, vloc, rad, rab, gshell, volume)
}

//dV_{at}^{loc} / d(G^2)
pub fn compute_dvloc_of_g(
    zion: f64,
    vloc: &[f64],
    rad: &[f64],
    rab: &[f64],
    gshell: &[f64],
    volume: f64,
) -> Vec<f64> {
    let nshell = gshell.len();

    let mut vg = vec![0.0; nshell];

    let mmax = vloc.len();

    let mut work = vec![0.0; mmax];

    // G = 0

    vg[0] = 0.0;

    // G > 0

    for iw in 1..nshell {
        let g = gshell[iw];
        let g2 = g * g;
        let g3 = g * g2;

        for i in 0..mmax {
            let r = rad[i];
            let gr = g * r;

            work[i] =
                (r * vloc[i] + zion * special::erf(r)) * (gr * gr.cos() - gr.sin()) / g3 / 2.0;
        }

        let vh = zion * (-g2 / 4.0).exp() / g2 * (0.25 + 1.0 / g2);

        //vg[iw] = integral::simpson_log(&work, rad) + vh;
        vg[iw] = integral::simpson_rab(&work, rab) + vh;
    }

    vg.iter_mut().for_each(|v| *v *= FOURPI / volume);

    vg
}
