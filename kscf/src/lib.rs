#![allow(warnings)]

use atompsp::AtomPSP;
use control::Control;
use crystal::Crystal;
use dwconsts::*;
use eigensolver;
use fftgrid::FFTGrid;
use gvector::GVector;
use kgylm::KGYLM;
use matrix::Matrix;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use smearing;
use smearing::Smearing;
use types::c64;
use utility;
use vector3::*;
use vnl::VNL;

use itertools::multizip;
use std::cell::Cell;

mod hubbard;
mod hybrid;
mod subspace;
use hubbard::HubbardPotential;
use hybrid::HybridPotential;

// Cached non-local data for one species at one k-point.
// Keeping structure factors + projector tables together avoids repeated
// reconstruction inside Hamiltonian applications.
struct NonLocalTerm<'a> {
    atompsp: &'a dyn AtomPSP,
    kgbeta: &'a [Vec<f64>],
    sfact_by_atom: Vec<Vec<c64>>,
}

// Per-k-point SCF worker.
//
// Encapsulates everything required to apply the Kohn-Sham Hamiltonian and
// solve the eigenproblem at one k-point:
// - kinetic diagonal
// - local potential application buffers/layout
// - non-local projector terms
// - occupations and smearing model
pub struct KSCF<'a> {
    control: &'a Control,
    gvec: &'a GVector,
    pwden: &'a PWDensity,
    pspot: &'a PSPot,
    vnl: &'a VNL,
    pwwfc: &'a PWBasis,

    ik: usize,
    kgylm: KGYLM,
    kin: Vec<f64>,
    occ: Vec<f64>,
    smearing: Box<dyn Smearing>,
    volume: f64,
    fft_shape: [usize; 3],
    fft_linear_index: Vec<usize>,
    vnl_terms: Vec<NonLocalTerm<'a>>,
    hybrid: HybridPotential,
    hubbard: HubbardPotential,
    hybrid_exchange_energy: Cell<f64>,

    k_weight: f64,
}

impl<'a> KSCF<'a> {
    pub fn get_pwwfc(&self) -> &PWBasis {
        self.pwwfc
    }

    pub fn get_kgylm(&self) -> &KGYLM {
        &self.kgylm
    }

    pub fn get_vnl(&self) -> &VNL {
        self.vnl
    }

    pub fn get_pspot(&self) -> &PSPot {
        self.pspot
    }

    pub fn get_gvec(&self) -> &GVector {
        self.gvec
    }

    pub fn new(
        control: &'a Control,
        gvec: &'a GVector,
        pwden: &'a PWDensity,
        crystal: &'a Crystal,
        pspot: &'a PSPot,
        pwwfc: &'a PWBasis,
        vnl: &'a VNL,
        fft_shape: [usize; 3],

        ik: usize,
        k_cart: Vector3f64,
        k_weight: f64,
    ) -> KSCF<'a> {
        // Real spherical harmonics Y_lm(k+G) table for this k-point.
        let kgylm = KGYLM::new(k_cart, pspot.get_max_lmax(), gvec, pwwfc);

        // Kinetic diagonal 1/2 |k+G|^2.
        let kin = compute_kinetic_energy(pwwfc.get_kg());

        let occ = vec![0.0; control.get_nband()];

        let smearing = smearing::new(control.get_smearing_scheme());

        let volume = crystal.get_latt().volume();
        // Cache FFT index mapping for fast v_loc application.
        let fft_linear_index = utility::compute_fft_linear_index_map(
            gvec.get_miller(),
            pwwfc.get_gindex(),
            fft_shape[0],
            fft_shape[1],
            fft_shape[2],
        );
        // Precompute non-local projector terms by species.
        let vnl_terms = build_nonlocal_terms(crystal, gvec, pspot, pwwfc, vnl);
        let hybrid = HybridPotential::new(control, pwwfc, pwden);
        let hubbard = HubbardPotential::new(control, crystal, gvec, pspot, pwwfc, &kgylm);

        KSCF {
            control,
            gvec,
            pwden,
            pspot,
            vnl,
            pwwfc,
            ik,
            kgylm,
            kin,
            occ,
            smearing,
            volume,
            fft_shape,
            fft_linear_index,
            vnl_terms,
            hybrid,
            hubbard,
            hybrid_exchange_energy: Cell::new(0.0),
            k_weight,
        }
    }

    pub fn get_ik(&self) -> usize {
        self.ik
    }

    pub fn get_nbands(&self) -> usize {
        self.occ.len()
    }

    pub fn get_k_weight(&self) -> f64 {
        self.k_weight
    }

    pub fn get_band_structure_energy(&self, evals: &[f64]) -> f64 {
        // Occupation-weighted eigenvalue sum for this k-point.
        let mut ebands = 0.0;

        for (ibnd, &occ) in self.occ.iter().enumerate() {
            if occ > EPS20 {
                ebands += evals[ibnd] * occ;
            }
        }

        ebands
    }

    pub fn get_occ(&self) -> &[f64] {
        &self.occ
    }

    pub fn get_total_occ(&self) -> f64 {
        self.occ.iter().sum()
    }

    pub fn hybrid_is_enabled(&self) -> bool {
        self.hybrid.is_enabled()
    }

    pub fn get_hybrid_exchange_energy(&self) -> f64 {
        self.hybrid_exchange_energy.get()
    }

    pub fn hubbard_is_enabled(&self) -> bool {
        self.hubbard.is_enabled()
    }

    pub fn hubbard_u_eff(&self) -> f64 {
        self.hubbard.u_eff()
    }

    pub fn hubbard_n_m(&self) -> usize {
        self.hubbard.n_m()
    }

    pub fn hubbard_n_atoms(&self) -> usize {
        self.hubbard.n_atoms()
    }

    pub fn hubbard_occ_len(&self) -> usize {
        self.hubbard.occ_len()
    }

    pub fn hubbard_set_global_occupation(&mut self, occ_global: &[c64]) {
        self.hubbard.set_global_occupation(occ_global);
    }

    pub fn hubbard_accumulate_occupation(&self, evecs: &Matrix<c64>, out: &mut [c64]) {
        self.hubbard.accumulate_occupation(
            &self.occ,
            self.k_weight,
            self.control.is_spin(),
            evecs,
            out,
        );
    }

    pub fn get_total_valence_occ_below(&self, evals: &[f64], energy_level: f64) -> f64 {
        let mut ntot = 0.0;

        for (i, &ev) in evals.iter().enumerate() {
            if ev <= energy_level {
                ntot += self.occ[i];
            }
        }

        ntot
    }

    pub fn get_total_conduction_occ_between(
        &self,
        evals: &[f64],
        upper_level: f64,
        lower_level: f64,
    ) -> f64 {
        // Counts nominal states in [lower, upper] used by inversion workflows.
        let mut ntot = 0.0;

        let mut occ = 1.0;
        if self.control.is_spin() {
            occ = 2.0;
        } else {
            occ = 1.0;
        }

        for (i, &ev) in evals.iter().enumerate() {
            if ev >= lower_level && ev <= upper_level {
                ntot += occ;
            }
        }

        ntot
    }

    pub fn get_unk(
        &self,
        rgtrans: &RGTransform,
        evecs: &Matrix<c64>,
        volume: f64,
        ib: usize,
        unk: &mut Array3<c64>,
        fft_workspace: &mut Array3<c64>,
    ) {
        // Reconstruct periodic part u_nk(r) from PW coefficients.
        hpsi::compute_unk_3d(
            self.gvec,
            self.pwwfc,
            rgtrans,
            volume,
            evecs.get_col(ib),
            unk,
            fft_workspace,
        );

        // unk
    }

    pub fn run(
        &self,
        rgtrans: &RGTransform,
        vloc_3d: &Array3<c64>,
        eigval_epsilon: f64,
        geom_iter: usize,
        scf_iter: usize,
        evals: &mut [f64],
        evecs: &mut Matrix<c64>,
    ) -> (usize, usize) {
        // Optional randomized initialization in early SCF cycles.
        if scf_iter <= self.control.get_scf_max_iter_rand_wfc() {
            let kg = self.pwwfc.get_kg();

            for ib in 0..self.control.get_nband() {
                let mut evec = evecs.get_mut_col(ib);

                utility::make_normalized_rand_vector(&mut evec);

                for i in 0..evec.len() {
                    evec[i] /= 0.5 * kg[i] * kg[i] + 1.0;
                }

                utility::normalize_vector_c64(&mut evec);
            }
        }

        let npw_wfc = self.pwwfc.get_n_plane_waves();

        // Workspaces reused by Hamiltonian applications.

        let mut vunkg_3d = Array3::<c64>::new(self.fft_shape);
        let mut unk_3d = Array3::<c64>::new(self.fft_shape);
        let mut fft_workspace = Array3::<c64>::new(self.fft_shape);
        let mut hubbard_beta = vec![c64::new(0.0, 0.0); self.hubbard.n_m()];
        let mut hubbard_coeff = vec![c64::new(0.0, 0.0); self.hubbard.n_m()];
        let hybrid_prepared = self.hybrid.prepare(
            rgtrans,
            self.gvec,
            self.pwden,
            self.volume,
            self.fft_shape,
            &self.fft_linear_index,
            evecs,
            &self.occ,
            self.control.is_spin(),
        );
        self.hybrid_exchange_energy
            .set(hybrid_prepared.exchange_energy());
        let mut hybrid_workspace = self
            .hybrid
            .make_workspace(self.fft_shape, self.pwden.get_n_plane_waves());

        //

        // Closure implementing H|psi> for eigensolver backend.
        let mut hamiltonian_on_psi = |vin: &[c64], vout: &mut [c64]| {
            for v in vout.iter_mut() {
                *v = c64::zero();
            }

            // Local potential contribution.

            hpsi::vloc_on_psi_with_cached_fft_index(
                rgtrans,
                self.volume,
                &self.fft_linear_index,
                vloc_3d,
                &mut vunkg_3d,
                &mut unk_3d,
                &mut fft_workspace,
                vin,
                vout,
            );

            // Non-local pseudopotential projector contribution.

            for term in self.vnl_terms.iter() {
                hpsi::vnl_on_psi_with_structure_factors(
                    term.atompsp,
                    &term.sfact_by_atom,
                    term.kgbeta,
                    &self.kgylm,
                    vin,
                    vout,
                );
            }

            // Screened exact-exchange (HSE06 short-range) contribution.
            self.hybrid.apply_on_psi(
                rgtrans,
                self.gvec,
                self.pwden,
                self.volume,
                &self.fft_linear_index,
                &hybrid_prepared,
                &mut hybrid_workspace,
                vin,
                vout,
            );

            // Hubbard (+U) contribution hook.
            self.hubbard
                .apply_on_psi(vin, vout, &mut hubbard_beta, &mut hubbard_coeff);

            // Kinetic contribution (diagonal in PW basis).

            hpsi::kinetic_on_psi(&self.kin, vin, vout);
        };

        // Eigensolver instance (PCG or chosen backend).
        let mut sparse = eigensolver::new(
            self.control.get_eigen_solver(),
            npw_wfc,
            self.control.get_nband(),
        );

        let max_scf_iter_wfc = self.control.get_scf_max_iter_wfc();

        let mut n_cg_loop = 0;

        let mut n_band_converged = 0;

        let mut n_hpsi = 0;

        loop {
            n_cg_loop += 1;

            // Solve subspace eigenproblem / update eigenpairs.

            let (n_band_converged_this_loop, n_hpsi_this_loop) = sparse.compute(
                &mut hamiltonian_on_psi,
                &self.kin,
                evecs,
                evals,
                &self.occ,
                eigval_epsilon,
                max_scf_iter_wfc,
                self.control.get_scf_max_iter(),
            );

            n_band_converged = n_band_converged_this_loop;
            n_hpsi += n_hpsi_this_loop; // total operations of Hamiltonian on the wavefunctions

            // Optional Rayleigh-Ritz rotation pass.
            if need_rayleigh_quotient(self.control, scf_iter, n_cg_loop) {
                let mut t_evecs = Matrix::<c64>::new(npw_wfc, self.control.get_nband());

                t_evecs.assign(evecs);

                subspace::rotate_wfc(&mut hamiltonian_on_psi, &mut t_evecs, evecs, evals);

                // println!("subspace rotation after eigen-solver");
            }

            if time_to_exit_cg(self.control, n_cg_loop, n_band_converged) {
                break;
            }
        }

        // Keep eigenpairs sorted in ascending eigenvalue order.
        sort_eigen_values_and_vectors(evals, evecs);

        (n_band_converged, n_hpsi)
    }

    pub fn compute_occ(&mut self, fermi_level: f64, evals: &[f64]) {
        // Smearing occupation update for this k-point.
        let occupation_scale = if self.control.is_spin() { 1.0 } else { 2.0 };

        for (occ, &ev) in multizip((self.occ.iter_mut(), evals.iter())) {
            *occ = occupation_scale
                * self.smearing.get_occupation_number(
                    fermi_level,
                    self.control.get_temperature(),
                    ev,
                );
        }
    }

    pub fn set_occ_inversion(&mut self, evals: &[f64], vb_level: f64, fermi_level: f64) {
        // Simple valence-band clipping used by occupation inversion path.
        for (occ, &ev) in multizip((self.occ.iter_mut(), evals.iter())) {
            if ev > vb_level {
                *occ = 0.0;
            }
        }
    }
}

fn sort_eigen_values_and_vectors(evals: &mut [f64], evecs: &mut Matrix<c64>) {
    // Stable sort of eigenvalues with synchronized eigenvector columns.
    let sort_idx = utility::argsort(evals);
    if sort_idx.iter().enumerate().all(|(i, &j)| i == j) {
        return;
    }

    let tmp_evals = evals.to_vec();
    let tmp_evecs = evecs.clone();

    for i in 0..evals.len() {
        let j = sort_idx[i];

        evals[i] = tmp_evals[j];

        let pcol = tmp_evecs.get_col(j);

        evecs.set_col(i, pcol);
    }
}

fn build_nonlocal_terms<'a>(
    crystal: &Crystal,
    gvec: &GVector,
    pspot: &'a PSPot,
    pwwfc: &PWBasis,
    vnl: &'a VNL,
) -> Vec<NonLocalTerm<'a>> {
    // Species-local caches of structure factors + projector tables.
    let species = crystal.get_unique_species();
    let kgbeta_all = vnl.get_kgbeta_all();

    species
        .iter()
        .enumerate()
        .map(|(isp, specie)| {
            let atom_positions = crystal.get_atom_positions_of_specie(isp);
            let sfact_by_atom =
                hpsi::compute_structure_factors_for_atoms(gvec, pwwfc, &atom_positions);

            NonLocalTerm {
                atompsp: pspot.get_psp(specie),
                kgbeta: kgbeta_all
                    .get(specie)
                    .unwrap_or_else(|| {
                        panic!("missing nonlocal projector data for specie {specie}")
                    })
                    .as_slice(),
                sfact_by_atom,
            }
        })
        .collect()
}

fn compute_kinetic_energy(kg: &[f64]) -> Vec<f64> {
    // T diagonal in PW basis: 1/2 |k+G|^2.
    kg.iter().map(|x| 0.5 * x * x).collect()
}

fn time_to_exit_cg(control: &Control, n_cg_loop: usize, n_band_converged: usize) -> bool {
    const MAX_CG_SCF: usize = 5;
    const MAX_CG_BAND: usize = 6;

    // Exit criteria for inner eigensolver loops.
    let mut b_exit = false;

    let n_states_bad = control.get_nband() - n_band_converged;

    if n_states_bad == 0 {
        b_exit = true;
    }

    if n_cg_loop >= MAX_CG_SCF {
        if n_states_bad > 0 {
            println!("{} states not converged", n_states_bad);
        }

        b_exit = true;
    }

    b_exit
}

fn need_rayleigh_quotient(control: &Control, scf_iter: usize, n_cg_loop: usize) -> bool {
    // Heuristic toggle for optional subspace rotation.
    let mut b_subspace_diag = false;

    if n_cg_loop > 1 {
        b_subspace_diag = true;
    }

    //false
    b_subspace_diag
}
