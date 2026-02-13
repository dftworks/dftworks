use atompsp::AtomPSP;
use gvector::GVector;
use itertools::multizip;
use kgylm::KGYLM;
use ndarray::Array3;
use num_traits::identities::Zero;
use pwbasis::PWBasis;
use rayon::prelude::*;
use rgtransform::RGTransform;
use types::c64;
use vector3::Vector3f64;

const PARALLEL_MIN_LEN: usize = 8192;

#[inline]
fn use_parallel_for_len(len: usize) -> bool {
    len >= PARALLEL_MIN_LEN && rayon::current_num_threads() > 1
}

pub fn kinetic_on_psi(kin: &[f64], vin: &[c64], vout: &mut [c64]) {
    debug_assert_eq!(kin.len(), vin.len());
    debug_assert_eq!(vin.len(), vout.len());

    if use_parallel_for_len(vout.len()) {
        vout.par_iter_mut()
            .zip(kin.par_iter())
            .zip(vin.par_iter())
            .for_each(|((z, x), y)| {
                *z += (*x) * (*y);
            });
    } else {
        for (x, y, z) in multizip((kin.iter(), vin.iter(), vout.iter_mut())) {
            *z += (*x) * (*y);
        }
    }
}

pub fn vnl_on_psi(
    atpsp: &dyn AtomPSP,
    atom_positions: &[Vector3f64],
    gvec: &GVector,
    pwwfc: &PWBasis,
    vnlbeta: &[Vec<f64>],
    kgylm: &KGYLM,
    vin: &[c64],
    vout: &mut [c64],
) {
    let sfact_by_atom = compute_structure_factors_for_atoms(gvec, pwwfc, atom_positions);
    vnl_on_psi_with_structure_factors(atpsp, &sfact_by_atom, vnlbeta, kgylm, vin, vout);
}

pub fn compute_structure_factors_for_atoms(
    gvec: &GVector,
    pwwfc: &PWBasis,
    atom_positions: &[Vector3f64],
) -> Vec<Vec<c64>> {
    atom_positions
        .iter()
        .map(|atom| {
            fhkl::compute_structure_factor_for_many_g_one_atom(
                gvec.get_miller(),
                pwwfc.get_gindex(),
                *atom,
            )
        })
        .collect()
}

pub fn vnl_on_psi_with_structure_factors(
    atpsp: &dyn AtomPSP,
    sfact_by_atom: &[Vec<c64>],
    vnlbeta: &[Vec<f64>],
    kgylm: &KGYLM,
    vin: &[c64],
    vout: &mut [c64],
) {
    for sfact in sfact_by_atom.iter() {
        apply_vnl_one_atom(atpsp, sfact, vnlbeta, kgylm, vin, vout);
    }
}

fn zddzs_product(v_out: &mut [c64], a: &[f64], b: &[f64], c: &[c64], s: c64) {
    debug_assert_eq!(v_out.len(), a.len());
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(b.len(), c.len());

    if use_parallel_for_len(v_out.len()) {
        v_out
            .par_iter_mut()
            .zip(a.par_iter())
            .zip(b.par_iter())
            .zip(c.par_iter())
            .for_each(|(((vx, ax), bx), cx)| {
                *vx += ax * bx * cx * s;
            });
    } else {
        for (vx, ax, bx, cx) in multizip((v_out.iter_mut(), a.iter(), b.iter(), c.iter())) {
            *vx += ax * bx * cx * s;
        }
    }
}

fn ddcz_dot_product(a: &[f64], b: &[f64], c: &[c64], d: &[c64]) -> c64 {
    let mut sum = c64::zero();

    for (ax, bx, cx, dx) in multizip((a.iter(), b.iter(), c.iter(), d.iter())) {
        sum += ax * bx * cx.conj() * dx;
    }

    sum
}

fn apply_vnl_one_atom(
    atpsp: &dyn AtomPSP,
    sfact: &[c64],
    vnlbeta: &[Vec<f64>],
    kgylm: &KGYLM,
    vin: &[c64],
    vout: &mut [c64],
) {
    for (j, beta) in vnlbeta.iter().enumerate() {
        let l = atpsp.get_lbeta(j);
        let dfact = atpsp.get_dfact(j);

        for m in utility::get_quant_num_m(l) {
            let ylm = kgylm.get_data(l, m);
            let beta_kg_cnk = ddcz_dot_product(ylm, beta, sfact, vin);
            zddzs_product(vout, ylm, beta, sfact, beta_kg_cnk * dfact);
        }
    }
}

pub fn vloc_on_psi(
    gvec: &GVector,
    pwwfc: &PWBasis,
    rgtrans: &RGTransform,
    volume: f64,
    vloc_3d: &Array3<c64>,
    vunkg_3d: &mut Array3<c64>,
    unk_3d: &mut Array3<c64>,
    fft_workspace: &mut Array3<c64>,
    vin: &[c64],
    vout: &mut [c64],
) {
    let [n1, n2, n3] = vloc_3d.shape();
    let fft_linear_index =
        utility::compute_fft_linear_index_map(gvec.get_miller(), pwwfc.get_gindex(), n1, n2, n3);

    vloc_on_psi_with_cached_fft_index(
        rgtrans,
        volume,
        &fft_linear_index,
        vloc_3d,
        vunkg_3d,
        unk_3d,
        fft_workspace,
        vin,
        vout,
    );
}

pub fn vloc_on_psi_with_cached_fft_index(
    rgtrans: &RGTransform,
    volume: f64,
    fft_linear_index: &[usize],
    vloc_3d: &Array3<c64>,
    vunkg_3d: &mut Array3<c64>,
    unk_3d: &mut Array3<c64>,
    fft_workspace: &mut Array3<c64>,
    vin: &[c64],
    vout: &mut [c64],
) {
    // from cnk in G space to get unk in r space

    compute_unk_3d_with_cached_fft_index(
        rgtrans,
        volume,
        fft_linear_index,
        vin,
        unk_3d,
        fft_workspace,
    );

    // (v_xc + v_h + v_psloc)|psi> in r space

    Array3::hadamard_product(vloc_3d, unk_3d, fft_workspace);

    // transform (v_xc + v_h + v_psloc)|psi> from r space  to G space

    rgtrans.r3d_to_g3d(fft_workspace.as_slice(), vunkg_3d.as_mut_slice());

    vunkg_3d.scale(volume.sqrt());

    // from 3d representation of (vxc + vh + vloc)|psi> to get
    // extract the elements corresponding to plane waves in wavefunction basis set

    utility::map_3d_to_1d_with_linear_index(fft_linear_index, vunkg_3d, vout);
}

pub fn compute_unk_3d(
    gvec: &GVector,
    pwwfc: &PWBasis,
    rgtrans: &RGTransform,
    volume: f64,
    v: &[c64],
    unk_3d: &mut Array3<c64>,
    fft_workspace: &mut Array3<c64>,
) {
    let [n1, n2, n3] = fft_workspace.shape();
    let fft_linear_index =
        utility::compute_fft_linear_index_map(gvec.get_miller(), pwwfc.get_gindex(), n1, n2, n3);

    compute_unk_3d_with_cached_fft_index(
        rgtrans,
        volume,
        &fft_linear_index,
        v,
        unk_3d,
        fft_workspace,
    );
}

pub fn compute_unk_3d_with_cached_fft_index(
    rgtrans: &RGTransform,
    volume: f64,
    fft_linear_index: &[usize],
    v: &[c64],
    unk_3d: &mut Array3<c64>,
    fft_workspace: &mut Array3<c64>,
) {
    utility::map_1d_to_3d_with_linear_index(fft_linear_index, v, fft_workspace);

    // cnk(G) -> unk(r)
    rgtrans.g3d_to_r3d(fft_workspace.as_slice(), unk_3d.as_mut_slice());
    unk_3d.scale(1.0 / volume.sqrt());
}
