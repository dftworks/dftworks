use atompsp::AtomPSP;
use gvector::GVector;
use itertools::multizip;
use kgylm::KGYLM;
use ndarray::Array3;
use num_traits::identities::Zero;
use pwbasis::PWBasis;
use rgtransform::RGTransform;
use types::c64;
use vector3::Vector3f64;

pub fn kinetic_on_psi(kin: &[f64], vin: &[c64], vout: &mut [c64]) {
    for (x, y, z) in multizip((kin.iter(), vin.iter(), vout.iter_mut())) {
        *z += (*x) * (*y);
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
    for atom in atom_positions {
        let sfact = fhkl::compute_structure_factor_for_many_g_one_atom(
            gvec.get_miller(),
            pwwfc.get_gindex(),
            *atom,
        );

        for (j, beta) in vnlbeta.iter().enumerate() {
            let l = atpsp.get_lbeta(j);

            let dfact = atpsp.get_dfact(j);

            for m in utility::get_quant_num_m(l) {
                let ylm = kgylm.get_data(l, m);

                //println!("ylm = {:.5?}", ylm);

                //let mut beta_kg_cnk = c64::zero();

                //for i in 0..npw {
                //    beta_kg_cnk += ylm[i] * beta[i] * sfact[i].conj() * vin[i];
                //}

                let beta_kg_cnk = ddcz_dot_product(&ylm, &beta, &sfact, &vin);

                //for i in 0..npw {
                //    vout[i] += ylm[i] * beta[i] * sfact[i] * beta_kg_cnk * dfact;
                //}

                zddzs_product(vout, &ylm, &beta, &sfact, beta_kg_cnk * dfact);
            }
        }
    }
}

fn zddzs_product(v_out: &mut [c64], a: &[f64], b: &[f64], c: &[c64], s: c64) {
    for (vx, ax, bx, cx) in multizip((v_out.iter_mut(), a.iter(), b.iter(), c.iter())) {
        *vx += ax * bx * cx * s;
    }
}

fn ddcz_dot_product(a: &[f64], b: &[f64], c: &[c64], d: &[c64]) -> c64 {
    let mut sum = c64::zero();

    for (ax, bx, cx, dx) in multizip((a.iter(), b.iter(), c.iter(), d.iter())) {
        sum += ax * bx * cx.conj() * dx;
    }

    sum
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

    // from cnk in G space to get unk in r space

    compute_unk_3d(gvec, pwwfc, rgtrans, volume, vin, unk_3d, fft_workspace);

    // (v_xc + v_h + v_psloc)|psi> in r space

    Array3::hadamard_product(vloc_3d, unk_3d, fft_workspace);

    // transform (v_xc + v_h + v_psloc)|psi> from r space  to G space

    rgtrans.r3d_to_g3d(fft_workspace.as_slice(), vunkg_3d.as_mut_slice());

    vunkg_3d.scale(volume.sqrt());

    // from 3d representation of (vxc + vh + vloc)|psi> to get
    // extract the elements corresponding to plane waves in wavefunction basis set

    utility::map_3d_to_1d(
        gvec.get_miller(),
        pwwfc.get_gindex(),
        n1,
        n2,
        n3,
        &vunkg_3d,
        vout,
    );
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

    utility::map_1d_to_3d(
        gvec.get_miller(),
        pwwfc.get_gindex(),
        n1,
        n2,
        n3,
        v,
        fft_workspace,
    );

    // cnk(G) -> unk(r)
    rgtrans.g3d_to_r3d(fft_workspace.as_slice(), unk_3d.as_mut_slice());
    unk_3d.scale(1.0 / volume.sqrt());
}
