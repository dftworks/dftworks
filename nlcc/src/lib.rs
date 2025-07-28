use atompsp::AtomPSP;
use crystal::Crystal;
use dwconsts::*;
use fhkl;
use gvector::GVector;
use ndarray::*;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::*;
use vector3::*;

pub fn from_atomic_super_position(
    pspot: &PSPot,
    crystal: &Crystal,
    rgtrans: &RGTransform,
    gvec: &GVector,
    pwden: &PWDensity,
    rhog: &mut [c64],
    rho_3d: &mut Array3<c64>,
) {
    // 1D Rho(G)

    atomic_super_position(pspot, crystal, pwden, gvec, rhog);

    // 1D Rho(G) -> 3D Rho(r)

    rgtrans.g1d_to_r3d(gvec, pwden, rhog, rho_3d.as_mut_slice());
}

fn atomic_super_position(
    atpsps: &PSPot,
    crystal: &Crystal,
    pwden: &PWDensity,
    gvec: &GVector,
    rhog: &mut [c64],
) {
    let volume = crystal.get_latt().volume();

    let atom_positions = crystal.get_atom_positions();

    let species = crystal.get_unique_species();

    let npw_rho = pwden.get_n_plane_waves();

    for (i, sp) in species.iter().enumerate() {
        let atpsp = atpsps.get_psp(sp);

        if atpsp.get_nlcc() == false {
            continue;
        }

        let atom_indices = crystal.get_atom_indices_of_specie(i);

        let mut atom_positions_for_this_specie = vec![Vector3f64::zeros(); atom_indices.len()];

        for (i, idx) in atom_indices.iter().enumerate() {
            atom_positions_for_this_specie[i] = atom_positions[*idx];
        }

        let rhog_one =
            atom_super_pos_one_specie(atpsp, &atom_positions_for_this_specie, pwden, gvec, volume);

        for i in 0..npw_rho {
            rhog[i] += rhog_one[i];
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

    let ffact_rho = rhocore_of_g_on_shells(atompsp, &pwden, volume);

    // crystal rho of G for rho

    let mut rhog = vec![c64::zero(); npw_rho];

    for i in 0..npw_rho {
        let ish = gshell_index[i];

        rhog[i] = ffact_rho[ish] * sfact[i];
    }

    rhog
}

pub fn rhocore_of_g_on_shells(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64) -> Vec<f64> {
    let gshell = pwden.get_gshell_norms();
    let atrho = atompsp.get_rhocore();
    let rad = atompsp.get_rad();
    let rab = atompsp.get_rab();

    compute_rhocore_of_g(atrho, rad, rab, gshell, volume)
}

//https://blog.cupcakephysics.com/electromagnetism/math%20methods/2014/10/04/the-fourier-transform-of-the-coulomb-potential.html

fn compute_rhocore_of_g(
    rho: &[f64],
    rad: &[f64],
    rab: &[f64],
    gshell: &[f64],
    volume: f64,
) -> Vec<f64> {
    let nshell = gshell.len();

    let mut rhog = vec![0.0; nshell];

    let mmax = rho.len();

    let mut work = vec![0.0; mmax];

    // G = 0

    for i in 0..mmax {
        work[i] = rho[i] * FOURPI * rad[i] * rad[i];
    }
    rhog[0] = integral::simpson_rab(&work, rab);

    // G > 0

    for iw in 1..nshell {
        for i in 0..mmax {
            if rad[i] < EPS8 {
                work[i] = rho[i] * FOURPI * rad[i] * rad[i];
            } else {
                let gr = gshell[iw] * rad[i];
                work[i] = rho[i] * gr.sin() / gr * FOURPI * rad[i] * rad[i];
            }
        }

        rhog[iw] = integral::simpson_rab(&work, rab);
    }

    rhog.iter_mut().for_each(|v| *v /= volume);

    rhog
}

// d(rhocore) / d(G^2)
pub fn drhocore_of_g_on_shells(atompsp: &dyn AtomPSP, pwden: &PWDensity, volume: f64) -> Vec<f64> {
    let gshell = pwden.get_gshell_norms();
    let zion = atompsp.get_zion();
    let rhocore = atompsp.get_rhocore();
    let rad = atompsp.get_rad();
    let rab = atompsp.get_rab();

    compute_drhocore_of_g(zion, rhocore, rad, rab, gshell, volume)
}

//d rho_{at}^{c} / d(G^2)
pub fn compute_drhocore_of_g(
    _zion: f64,
    rhocore: &[f64],
    rad: &[f64],
    rab: &[f64],
    gshell: &[f64],
    volume: f64,
) -> Vec<f64> {
    let nshell = gshell.len();

    let mut vg = vec![0.0; nshell];

    let mmax = rhocore.len();

    let mut work = vec![0.0; mmax];

    // G = 0

    vg[0] = 0.0;

    // G > 0

    for iw in 1..nshell {
        let g = gshell[iw];
        let g2 = g * g;

        for i in 0..mmax {
            let r = rad[i];
            work[i] = (r * rhocore[i]) * ((g * r) * (g * r).cos() - (g * r).sin()) / g2 / g / 2.0;
        }

        vg[iw] = integral::simpson_rab(&work, rab);
    }

    // for v in vg.iter_mut() {
    //     *v *= FOURPI / volume;
    // }

    vg.iter_mut().for_each(|v| *v *= FOURPI / volume);

    vg
}
