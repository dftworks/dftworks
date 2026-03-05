#![allow(warnings)]

use control::SpinScheme;
use dfttypes::*;
use gvector::GVector;
use kscf::KSCF;
use types::Matrix;
use types::MatrixExt;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use types::c64;

pub(crate) fn estimate_kpoint_costs(
    runtime: &crate::RuntimeContext,
    blatt: &lattice::Lattice,
    gvec: &GVector,
) -> Vec<u64> {
    let nkpt = runtime.kpts.get_n_kpts();
    let nband = runtime.control.get_nband() as u64;
    let mut costs = Vec::<u64>::with_capacity(nkpt);

    for ik in 0..nkpt {
        let k_frac = runtime.kpts.get_k_frac(ik);
        let k_cart = runtime.kpts.frac_to_cart(&k_frac, blatt);
        let pwwfc = PWBasis::new(k_cart, ik, runtime.control.get_ecut(), gvec);
        let npw = pwwfc.get_n_plane_waves() as u64;
        costs.push(npw.saturating_mul(nband).max(1));
    }

    costs
}

pub(crate) fn build_kscf_channel<'a>(
    runtime: &crate::RuntimeContext<'a>,
    electronic_ctx: &'a crate::ElectronicStepContext,
    gvec: &'a GVector,
    pwden: &'a PWDensity,
    fft_shape: [usize; 3],
    random_stream_id: u64,
) -> Vec<KSCF<'a>> {
    let mut channel = Vec::<KSCF>::with_capacity(electronic_ctx.k_domain.len());
    let blatt = runtime.crystal.get_latt().reciprocal();

    for slot in electronic_ctx.k_domain.iter() {
        let ik = slot.global_index;
        let ilocal = slot.local_slot;
        let k_frac = runtime.kpts.get_k_frac(ik);
        let k_cart = runtime.kpts.frac_to_cart(&k_frac, &blatt);
        let k_weight = runtime.kpts.get_k_weight(ik);

        let kscf = KSCF::new(
            runtime.control,
            gvec,
            pwden,
            runtime.crystal,
            runtime.pots,
            &electronic_ctx.vpwwfc[ilocal],
            &electronic_ctx.vvnl[ilocal],
            fft_shape,
            ik,
            k_cart,
            k_weight,
            random_stream_id,
        );
        channel.push(kscf);
    }

    channel
}

pub(crate) fn build_spin_kscf_channels<'a>(
    runtime: &crate::RuntimeContext<'a>,
    electronic_ctx: &'a crate::ElectronicStepContext,
    gvec: &'a GVector,
    pwden: &'a PWDensity,
    fft_shape: [usize; 3],
) -> (Vec<KSCF<'a>>, Vec<KSCF<'a>>, usize) {
    let mut shared = Vec::with_capacity(electronic_ctx.k_domain.len());
    let blatt = runtime.crystal.get_latt().reciprocal();

    for slot in electronic_ctx.k_domain.iter() {
        let ik = slot.global_index;
        let ilocal = slot.local_slot;
        let k_frac = runtime.kpts.get_k_frac(ik);
        let k_cart = runtime.kpts.frac_to_cart(&k_frac, &blatt);
        let cache = KSCF::build_shared_cache(
            gvec,
            runtime.crystal,
            runtime.pots,
            &electronic_ctx.vpwwfc[ilocal],
            &electronic_ctx.vvnl[ilocal],
            fft_shape,
            ik,
            k_cart,
        );
        shared.push(cache);
    }

    let saved_bytes = shared.iter().map(|cache| cache.estimated_bytes()).sum::<usize>();

    let mut channel_up = Vec::<KSCF>::with_capacity(electronic_ctx.k_domain.len());
    let mut channel_dn = Vec::<KSCF>::with_capacity(electronic_ctx.k_domain.len());

    for slot in electronic_ctx.k_domain.iter() {
        let ik = slot.global_index;
        let ilocal = slot.local_slot;
        let k_weight = runtime.kpts.get_k_weight(ik);
        let cache = shared[ilocal].clone();

        let kscf_up = KSCF::new_with_shared_cache(
            runtime.control,
            gvec,
            pwden,
            runtime.crystal,
            runtime.pots,
            &electronic_ctx.vpwwfc[ilocal],
            &electronic_ctx.vvnl[ilocal],
            fft_shape,
            ik,
            k_weight,
            cache.clone(),
            0,
        );
        let kscf_dn = KSCF::new_with_shared_cache(
            runtime.control,
            gvec,
            pwden,
            runtime.crystal,
            runtime.pots,
            &electronic_ctx.vpwwfc[ilocal],
            &electronic_ctx.vvnl[ilocal],
            fft_shape,
            ik,
            k_weight,
            cache,
            1,
        );

        channel_up.push(kscf_up);
        channel_dn.push(kscf_dn);
    }

    (channel_up, channel_dn, saved_bytes)
}

pub(crate) fn allocate_eigenvalues(
    spin_scheme: SpinScheme,
    nband: usize,
    nk_local: usize,
) -> Result<VKEigenValue, String> {
    match spin_scheme {
        SpinScheme::NonSpin => Ok(VKEigenValue::NonSpin(vec![vec![0.0; nband]; nk_local])),
        SpinScheme::Spin => Ok(VKEigenValue::Spin(
            vec![vec![0.0; nband]; nk_local],
            vec![vec![0.0; nband]; nk_local],
        )),
        SpinScheme::Ncl => Err(
            "unsupported capability: spin_scheme='ncl' is not implemented in eigen setup"
                .to_string(),
        ),
    }
}

fn allocate_eigenvector_channel(vpwwfc: &[PWBasis], nband: usize) -> Vec<Matrix<c64>> {
    let mut channel = Vec::with_capacity(vpwwfc.len());
    for pwwfc in vpwwfc.iter() {
        channel.push(Matrix::new(pwwfc.get_n_plane_waves(), nband));
    }
    channel
}

pub(crate) fn allocate_eigenvectors(
    spin_scheme: SpinScheme,
    nband: usize,
    vpwwfc: &[PWBasis],
) -> Result<VKEigenVector, String> {
    match spin_scheme {
        SpinScheme::NonSpin => Ok(VKEigenVector::NonSpin(allocate_eigenvector_channel(
            vpwwfc, nband,
        ))),
        SpinScheme::Spin => Ok(VKEigenVector::Spin(
            allocate_eigenvector_channel(vpwwfc, nband),
            allocate_eigenvector_channel(vpwwfc, nband),
        )),
        SpinScheme::Ncl => Err(
            "unsupported capability: spin_scheme='ncl' is not implemented in eigen setup"
                .to_string(),
        ),
    }
}
