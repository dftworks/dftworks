use control::Control;
use crystal::Crystal;
use dwconsts::*;
use gvector::GVector;
use kgylm::KGYLM;
use matrix::Matrix;
use pspot::PSPot;
use pwbasis::PWBasis;
use types::c64;
use vector3::Vector3f64;

fn occ_index(iat: usize, m: usize, mp: usize, n_m: usize) -> usize {
    iat * n_m * n_m + m + mp * n_m
}

fn compute_atomic_wfc_of_kg(
    kg: &[f64],
    l: usize,
    chi: &[f64],
    rad: &[f64],
    rab: &[f64],
    volume: f64,
) -> Vec<f64> {
    assert_eq!(
        chi.len(),
        rad.len(),
        "chi/rad radial-grid mismatch while building Hubbard projectors"
    );
    assert_eq!(
        rad.len(),
        rab.len(),
        "rad/rab radial-grid mismatch while building Hubbard projectors"
    );

    let npw = kg.len();
    let mut chi_kg = vec![0.0; npw];
    let mut work = vec![0.0; rad.len()];
    let prefactor = FOURPI / volume.sqrt();

    for ig in 0..npw {
        for ir in 0..rad.len() {
            let r = rad[ir];
            work[ir] = chi[ir] * r * special::spherical_bessel_jn(l, kg[ig] * r);
        }

        chi_kg[ig] = prefactor * integral::simpson_rab(&work, rab);
    }

    chi_kg
}

fn build_correlated_atom_positions(crystal: &Crystal, specie: &str) -> Vec<Vector3f64> {
    crystal
        .get_atom_species()
        .iter()
        .enumerate()
        .filter_map(|(iat, sp)| {
            if sp == specie {
                Some(crystal.get_atom_positions()[iat])
            } else {
                None
            }
        })
        .collect()
}

pub struct HubbardPotential {
    enabled: bool,
    u_eff: f64,
    n_m: usize,
    projector_by_atom_m: Vec<Vec<Vec<c64>>>,
    d_by_atom: Vec<Matrix<c64>>,
}

impl HubbardPotential {
    pub fn new(
        control: &Control,
        crystal: &Crystal,
        gvec: &GVector,
        pspot: &PSPot,
        pwwfc: &PWBasis,
        kgylm: &KGYLM,
    ) -> Self {
        if !control.get_hubbard_u_enabled() {
            return Self {
                enabled: false,
                u_eff: 0.0,
                n_m: 0,
                projector_by_atom_m: Vec::new(),
                d_by_atom: Vec::new(),
            };
        }

        let specie = control.get_hubbard_species();
        let l = control.get_hubbard_l() as usize;
        let u_eff = control.get_hubbard_u_eff();

        let atom_positions = build_correlated_atom_positions(crystal, specie);
        if atom_positions.is_empty() {
            panic!(
                "hubbard_species='{}' not found in crystal atom list",
                specie
            );
        }

        let atpsp = pspot.get_psp(specie);
        if !atpsp.has_wfc(l) {
            panic!(
                "hubbard channel requires PP_CHI for species='{}', l={} but pseudopotential does not provide it",
                specie, l
            );
        }

        let m_values = utility::get_quant_num_m(l);
        let n_m = m_values.len();
        let npw = pwwfc.get_n_plane_waves();

        let chi_kg = compute_atomic_wfc_of_kg(
            pwwfc.get_kg(),
            l,
            atpsp.get_wfc(l),
            atpsp.get_rad(),
            atpsp.get_rab(),
            crystal.get_latt().volume(),
        );

        let sfact_by_atom = hpsi::compute_structure_factors_for_atoms(gvec, pwwfc, &atom_positions);

        let mut projector_by_atom_m =
            vec![vec![vec![c64::new(0.0, 0.0); npw]; n_m]; sfact_by_atom.len()];

        for (iat, sfact) in sfact_by_atom.iter().enumerate() {
            for (im, &m) in m_values.iter().enumerate() {
                let ylm = kgylm.get_data(l, m);

                for ig in 0..npw {
                    projector_by_atom_m[iat][im][ig] =
                        c64::new(ylm[ig] * chi_kg[ig], 0.0) * sfact[ig];
                }
            }
        }

        let d_by_atom = vec![Matrix::<c64>::new(n_m, n_m); projector_by_atom_m.len()];

        Self {
            enabled: true,
            u_eff,
            n_m,
            projector_by_atom_m,
            d_by_atom,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn n_m(&self) -> usize {
        self.n_m
    }

    pub fn n_atoms(&self) -> usize {
        self.projector_by_atom_m.len()
    }

    pub fn u_eff(&self) -> f64 {
        self.u_eff
    }

    pub fn occ_len(&self) -> usize {
        self.n_atoms() * self.n_m * self.n_m
    }

    pub fn accumulate_occupation(
        &self,
        occ: &[f64],
        k_weight: f64,
        is_spin: bool,
        evecs: &Matrix<c64>,
        out: &mut [c64],
    ) {
        if !self.enabled {
            return;
        }

        assert_eq!(out.len(), self.occ_len());
        let n_m = self.n_m;
        let occ_scale = if is_spin { 1.0 } else { 0.5 };
        let mut beta = vec![c64::new(0.0, 0.0); n_m];

        for (ibnd, &occ_raw) in occ.iter().enumerate() {
            let occ_eff = occ_raw * occ_scale;

            if occ_eff < EPS20 {
                continue;
            }

            let weight = k_weight * occ_eff;
            let cnk = evecs.get_col(ibnd);

            for (iat, projectors_m) in self.projector_by_atom_m.iter().enumerate() {
                for m in 0..n_m {
                    beta[m] = utility::zdot_product(&projectors_m[m], cnk);
                }

                for m in 0..n_m {
                    for mp in 0..n_m {
                        out[occ_index(iat, m, mp, n_m)] += weight * beta[m] * beta[mp].conj();
                    }
                }
            }
        }
    }

    pub fn set_global_occupation(&mut self, occ_global: &[c64]) {
        if !self.enabled {
            return;
        }

        assert_eq!(occ_global.len(), self.occ_len());

        for iat in 0..self.n_atoms() {
            for m in 0..self.n_m {
                for mp in 0..self.n_m {
                    let mut d = -self.u_eff * occ_global[occ_index(iat, m, mp, self.n_m)];
                    if m == mp {
                        d += c64::new(0.5 * self.u_eff, 0.0);
                    }

                    self.d_by_atom[iat][[m, mp]] = d;
                }
            }
        }
    }

    pub fn apply_on_psi(&self, vin: &[c64], vout: &mut [c64], beta: &mut [c64], coeff: &mut [c64]) {
        if !self.enabled {
            return;
        }

        assert_eq!(beta.len(), self.n_m);
        assert_eq!(coeff.len(), self.n_m);

        for (iat, projectors_m) in self.projector_by_atom_m.iter().enumerate() {
            for mp in 0..self.n_m {
                beta[mp] = utility::zdot_product(&projectors_m[mp], vin);
            }

            for m in 0..self.n_m {
                let mut val = c64::new(0.0, 0.0);
                for mp in 0..self.n_m {
                    val += self.d_by_atom[iat][[m, mp]] * beta[mp];
                }

                coeff[m] = val;
            }

            for m in 0..self.n_m {
                if coeff[m].norm_sqr() < EPS30 {
                    continue;
                }

                for ig in 0..vout.len() {
                    vout[ig] += projectors_m[m][ig] * coeff[m];
                }
            }
        }
    }
}
