#![allow(warnings)]

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
use rgtransform::RGTransform;
use smearing;
use smearing::Smearing;
use types::c64;
use utility;
use vector3::*;
use vnl::VNL;

use itertools::multizip;

mod subspace;

pub struct KSCF<'a> {
    control: &'a Control,
    gvec: &'a GVector,
    pspot: &'a PSPot,
    vnl: &'a VNL,
    pwwfc: &'a PWBasis,

    ik: usize,
    kgylm: KGYLM,
    kin: Vec<f64>,
    occ: Vec<f64>,
    smearing: Box<dyn Smearing>,

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
        pspot: &'a PSPot,
        pwwfc: &'a PWBasis,
        vnl: &'a VNL,

        ik: usize,
        k_cart: Vector3f64,
        k_weight: f64,
    ) -> KSCF<'a> {
        let kgylm = KGYLM::new(k_cart, pspot.get_max_lmax(), gvec, pwwfc);

        let kin = compute_kinetic_energy(pwwfc.get_kg());

        let occ = vec![0.0; control.get_nband()];

        let smearing = smearing::new(control.get_smearing_scheme());

        KSCF {
            control,
            gvec,
            pspot,
            vnl,
            pwwfc,
            ik,
            kgylm,
            kin,
            occ,
            smearing,
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
        crystal: &Crystal,
        fftgrid: &FFTGrid,
        rgtrans: &RGTransform,
        vloc_3d: &Array3<c64>,
        eigval_epsilon: f64,
        geom_iter: usize,
        scf_iter: usize,
        evals: &mut [f64],
        evecs: &mut Matrix<c64>,
    ) -> (usize, usize) {
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

        let volume = crystal.get_latt().volume();

        let npw_wfc = self.pwwfc.get_n_plane_waves();

        let fft_shape = fftgrid.get_size();

        // workspace for hamiltonian_on_psi

        let mut vunkg_3d = Array3::<c64>::new(fft_shape);
        let mut unk_3d = Array3::<c64>::new(fft_shape);
        let mut fft_workspace = Array3::<c64>::new(fft_shape);

        //

        let mut hamiltonian_on_psi = |vin: &[c64], vout: &mut [c64]| {
            for v in vout.iter_mut() {
                *v = c64::zero();
            }

            // compute vloc on |psi>

            hpsi::vloc_on_psi(
                self.gvec,
                self.pwwfc,
                rgtrans,
                volume,
                vloc_3d,
                &mut vunkg_3d,
                &mut unk_3d,
                &mut fft_workspace,
                vin,
                vout,
            );

            // add v_nl |psi> to v_loc |psi>

            let kgbeta_all = self.vnl.get_kgbeta_all();

            for (isp, specie) in crystal.get_unique_species().iter().enumerate() {
                let kgbeta = kgbeta_all.get(specie).unwrap();

                let atompsp = self.pspot.get_psp(specie);
                let atom_positions_for_this_specie = crystal.get_atom_positions_of_specie(isp);

                hpsi::vnl_on_psi(
                    atompsp,
                    &atom_positions_for_this_specie,
                    self.gvec,
                    self.pwwfc,
                    kgbeta,
                    &self.kgylm,
                    vin,
                    vout,
                );
            }

            // add kinetic energy | psi> to v_loc |psi> + v_nl |psi>

            hpsi::kinetic_on_psi(&self.kin, vin, vout);
        };

        // begin: preconditioner

        let mut h_diag = self.kin.clone();

        //h_diag(ig,:) = 1.D0 + g2kin(ig) + SQRT( 1.D0 + ( g2kin(ig) - 1.D0 )**2 )

        // for x in h_diag.iter_mut() {
        //     *x = 1.0 + *x + (1.0 + (*x - 1.0).powf(2.0)).sqrt();
        // }

        // end: preconditioner

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

            // subspace rotation

            if need_rayleigh_quotient(self.control, scf_iter, n_cg_loop) {
                let mut t_evecs = Matrix::<c64>::new(npw_wfc, self.control.get_nband());

                t_evecs.assign(evecs);

                //subspace::rotate_wfc(&mut hamiltonian_on_psi, &mut t_evecs, evecs, evals);

                // println!("subspace rotation before eigen-solver");
            }

            // sparse solver

            let (n_band_converged_this_loop, n_hpsi_this_loop) = sparse.compute(
                &mut hamiltonian_on_psi,
                &h_diag,
                evecs,
                evals,
                &self.occ,
                eigval_epsilon,
                max_scf_iter_wfc,
                self.control.get_scf_max_iter(),
            );

            n_band_converged = n_band_converged_this_loop;
            n_hpsi += n_hpsi_this_loop; // total operations of Hamiltonian on the wavefunctions

            // subspace rotation

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

        sort_eigen_values_and_vectors(evals, evecs);

        (n_band_converged, n_hpsi)
    }

    pub fn compute_occ(&mut self, fermi_level: f64, evals: &[f64]) {
        if self.control.is_spin() {
            for (occ, &ev) in multizip((self.occ.iter_mut(), evals.iter())) {
                *occ = 1.0
                    * self.smearing.get_occupation_number(
                        fermi_level,
                        self.control.get_temperature(),
                        ev,
                    );
            }
        } else {
            for (occ, &ev) in multizip((self.occ.iter_mut(), evals.iter())) {
                *occ = 2.0
                    * self.smearing.get_occupation_number(
                        fermi_level,
                        self.control.get_temperature(),
                        ev,
                    );
            }
        }
    }

    pub fn set_occ_inversion(&mut self, evals: &[f64], vb_level: f64, fermi_level: f64) {
        for (occ, &ev) in multizip((self.occ.iter_mut(), evals.iter())) {
            if ev > vb_level {
                *occ = 0.0;
            }
        }
    }
}

fn sort_eigen_values_and_vectors(evals: &mut [f64], evecs: &mut Matrix<c64>) {
    let sort_idx = utility::argsort(evals);
    let tmp_evals = evals.to_vec();
    let tmp_evecs = evecs.clone();

    for i in 0..evals.len() {
        let j = sort_idx[i];

        evals[i] = tmp_evals[j];

        let pcol = tmp_evecs.get_col(j);

        evecs.set_col(i, pcol);
    }
}

fn compute_kinetic_energy(kg: &[f64]) -> Vec<f64> {
    kg.iter().map(|x| 0.5 * x * x).collect()
}

fn time_to_exit_cg(control: &Control, n_cg_loop: usize, n_band_converged: usize) -> bool {
    const MAX_CG_SCF: usize = 5;
    const MAX_CG_BAND: usize = 6;

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
    let mut b_subspace_diag = false;

    if n_cg_loop > 1 {
        b_subspace_diag = true;
    }

    //false
    b_subspace_diag
}
