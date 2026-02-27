#![allow(warnings)]

use super::engine::{run_scf_iteration_engine, ScfIterationAdapter};
use super::hartree;
use super::hubbard;
use super::utils;
use crate::SCF;
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use ewald::Ewald;
use fftgrid::FFTGrid;
use gvector::GVector;
use itertools::multizip;
use kpts::KPTS;
use kscf::KSCF;
use lattice::Lattice;
use matrix::Matrix;
use mpi_sys::MPI_COMM_WORLD;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use symmetry::SymmetryDriver;
use types::c64;
use vector3::Vector3f64;

//use rayon::prelude::*;

pub struct SCFSpin {}

impl SCFSpin {
    pub fn new() -> SCFSpin {
        SCFSpin {}
    }
}

#[inline]
fn sum_spin_channels(rhog_up: &[c64], rhog_dn: &[c64], rhog_tot: &mut [c64]) {
    debug_assert_eq!(rhog_up.len(), rhog_dn.len());
    debug_assert_eq!(rhog_up.len(), rhog_tot.len());

    for i in 0..rhog_tot.len() {
        rhog_tot[i] = rhog_up[i] + rhog_dn[i];
    }
}

// Scratch/work buffers reused across the spin-collinear SCF loop.
// All sizes are derived once from (npw_rho, fft_shape) and reused in-place.
struct SpinScfWorkspace {
    rhog_tot: Vec<c64>,
    rhog_tot_scratch: Vec<c64>,
    vpslocg: Vec<c64>,
    vhg: Vec<c64>,
    vxcg: VXCG,
    vxc_3d: VXCR,
    exc_3d: Array3<c64>,
    vlocg_up: Vec<c64>,
    vlocg_dn: Vec<c64>,
    vloc_3d_up: Array3<c64>,
    vloc_3d_dn: Array3<c64>,
    rhog_out_up: Vec<c64>,
    rhog_out_dn: Vec<c64>,
    rhog_diff: Vec<c64>,
    rhog_total: Vec<c64>,
    rhog_spin: Vec<c64>,
}

impl SpinScfWorkspace {
    fn new(npw_rho: usize, fft_shape: [usize; 3]) -> Self {
        Self {
            rhog_tot: vec![c64::zero(); npw_rho],
            rhog_tot_scratch: vec![c64::zero(); npw_rho],
            vpslocg: vec![c64::zero(); npw_rho],
            vhg: vec![c64::zero(); npw_rho],
            vxcg: VXCG::Spin(vec![c64::zero(); npw_rho], vec![c64::zero(); npw_rho]),
            vxc_3d: VXCR::Spin(Array3::<c64>::new(fft_shape), Array3::<c64>::new(fft_shape)),
            exc_3d: Array3::<c64>::new(fft_shape),
            vlocg_up: vec![c64::zero(); npw_rho],
            vlocg_dn: vec![c64::zero(); npw_rho],
            vloc_3d_up: Array3::<c64>::new(fft_shape),
            vloc_3d_dn: Array3::<c64>::new(fft_shape),
            rhog_out_up: vec![c64::zero(); npw_rho],
            rhog_out_dn: vec![c64::zero(); npw_rho],
            rhog_diff: vec![c64::zero(); npw_rho],
            rhog_total: vec![c64::zero(); npw_rho],
            rhog_spin: vec![c64::zero(); npw_rho],
        }
    }

    fn validate(&self, npw_rho: usize, fft_shape: [usize; 3]) {
        let nfft = fft_shape[0] * fft_shape[1] * fft_shape[2];

        debug_assert_eq!(self.rhog_tot.len(), npw_rho);
        debug_assert_eq!(self.rhog_tot_scratch.len(), npw_rho);
        debug_assert_eq!(self.vpslocg.len(), npw_rho);
        debug_assert_eq!(self.vhg.len(), npw_rho);
        debug_assert_eq!(self.vlocg_up.len(), npw_rho);
        debug_assert_eq!(self.vlocg_dn.len(), npw_rho);
        debug_assert_eq!(self.rhog_out_up.len(), npw_rho);
        debug_assert_eq!(self.rhog_out_dn.len(), npw_rho);
        debug_assert_eq!(self.rhog_diff.len(), npw_rho);
        debug_assert_eq!(self.rhog_total.len(), npw_rho);
        debug_assert_eq!(self.rhog_spin.len(), npw_rho);
        debug_assert_eq!(self.vxcg.as_spin().unwrap().0.len(), npw_rho);
        debug_assert_eq!(self.vxcg.as_spin().unwrap().1.len(), npw_rho);
        debug_assert_eq!(self.vxc_3d.as_spin().unwrap().0.as_slice().len(), nfft);
        debug_assert_eq!(self.vxc_3d.as_spin().unwrap().1.as_slice().len(), nfft);
        debug_assert_eq!(self.exc_3d.as_slice().len(), nfft);
        debug_assert_eq!(self.vloc_3d_up.as_slice().len(), nfft);
        debug_assert_eq!(self.vloc_3d_dn.as_slice().len(), nfft);
    }
}

#[inline]
fn add_core_density_to_spin_channels(rho_3d: &mut RHOR, rhocore_3d: &Array3<c64>, scale: f64) {
    if let RHOR::Spin(rho_3d_up, rho_3d_dn) = rho_3d {
        for i in 0..rhocore_3d.as_slice().len() {
            let core = rhocore_3d.as_slice()[i] * scale;
            rho_3d_up.as_mut_slice()[i] += core;
            rho_3d_dn.as_mut_slice()[i] += core;
        }
    }
}

fn solve_spin_eigen_equations(
    rgtrans: &RGTransform,
    vloc_3d_up: &Array3<c64>,
    vloc_3d_dn: &Array3<c64>,
    eigvalue_epsilon: f64,
    geom_iter: usize,
    scf_iter: usize,
    vkscf: &VKSCF,
    vkevals: &mut VKEigenValue,
    vkevecs: &mut VKEigenVector,
) {
    if let VKEigenValue::Spin(vkevals_up, vkevals_dn) = vkevals {
        if let VKEigenVector::Spin(vkevecs_up, vkevecs_dn) = vkevecs {
            if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
                for (ik, kscf) in vkscf_up.iter().enumerate() {
                    kscf.run(
                        rgtrans,
                        vloc_3d_up,
                        eigvalue_epsilon,
                        geom_iter,
                        scf_iter,
                        &mut vkevals_up[ik],
                        &mut vkevecs_up[ik],
                    );
                }

                for (ik, kscf) in vkscf_dn.iter().enumerate() {
                    kscf.run(
                        rgtrans,
                        vloc_3d_dn,
                        eigvalue_epsilon,
                        geom_iter,
                        scf_iter,
                        &mut vkevals_dn[ik],
                        &mut vkevecs_dn[ik],
                    );
                }
            }
        }
    }
}

struct SpinIterationAdapter<'ctx, 'ks> {
    geom_iter: usize,
    control: &'ctx Control,
    crystal: &'ctx Crystal,
    gvec: &'ctx GVector,
    pwden: &'ctx PWDensity,
    rgtrans: &'ctx RGTransform,
    ewald: &'ctx Ewald,
    ntot_elec: f64,
    npw_wfc_max: usize,
    fft_ntotf64: f64,
    xc: &'ctx dyn xc::XC,
    density_driver: &'ctx dyn density::Density,
    fermi_driver: &'ctx dyn fermilevel::FermiLevel,
    mixing_rho: &'ctx mut dyn mixing::Mixing,
    mixing_spin: &'ctx mut dyn mixing::Mixing,
    vkscf: &'ctx mut VKSCF<'ks>,
    rhog: &'ctx mut RHOG,
    rho_3d: &'ctx mut RHOR,
    rhocore_3d: &'ctx Array3<c64>,
    vkevals: &'ctx mut VKEigenValue,
    vkevecs: &'ctx mut VKEigenVector,
    ws: &'ctx mut SpinScfWorkspace,
    hubbard_energy: f64,
}

impl SpinIterationAdapter<'_, '_> {
    fn refresh_spin_xc_for_iteration(&mut self, core_scale: f64) {
        add_core_density_to_spin_channels(self.rho_3d, self.rhocore_3d, core_scale);
        self.xc.potential_and_energy(
            self.gvec,
            self.pwden,
            self.rgtrans,
            self.rho_3d,
            &mut self.ws.vxc_3d,
            &mut self.ws.exc_3d,
        );
        add_core_density_to_spin_channels(self.rho_3d, self.rhocore_3d, -core_scale);
    }

    fn refresh_vxcg_from_vxc_3d(&mut self) {
        let (vxc_3d_up, vxc_3d_dn) = self.ws.vxc_3d.as_spin().unwrap();
        let (vxcg_up, vxcg_dn) = self.ws.vxcg.as_spin_mut().unwrap();

        self.rgtrans
            .r3d_to_g1d(self.gvec, self.pwden, vxc_3d_up.as_slice(), vxcg_up);
        self.rgtrans
            .r3d_to_g1d(self.gvec, self.pwden, vxc_3d_dn.as_slice(), vxcg_dn);
    }

    fn rebuild_vlocg_from_components(&mut self) {
        let (vxcg_up, vxcg_dn) = self.ws.vxcg.as_spin().unwrap();

        for (v_loc, v_xc, v_ha, v_psloc) in multizip((
            self.ws.vlocg_up.iter_mut(),
            vxcg_up.iter(),
            self.ws.vhg.iter(),
            self.ws.vpslocg.iter(),
        )) {
            *v_loc = *v_xc + *v_ha + *v_psloc;
        }

        for (v_loc, v_xc, v_ha, v_psloc) in multizip((
            self.ws.vlocg_dn.iter_mut(),
            vxcg_dn.iter(),
            self.ws.vhg.iter(),
            self.ws.vpslocg.iter(),
        )) {
            *v_loc = *v_xc + *v_ha + *v_psloc;
        }
    }

    fn rebuild_hartree_from_spin_density(&mut self) {
        if let RHOG::Spin(rhog_up, rhog_dn) = self.rhog {
            sum_spin_channels(rhog_up, rhog_dn, &mut self.ws.rhog_tot);
        }

        hartree::potential(self.pwden.get_g(), &self.ws.rhog_tot, &mut self.ws.vhg);
    }
}

impl ScfIterationAdapter for SpinIterationAdapter<'_, '_> {
    fn prepare_potential_in_rspace(&mut self) {
        self.rgtrans.g1d_to_r3d(
            self.gvec,
            self.pwden,
            &self.ws.vlocg_up,
            self.ws.vloc_3d_up.as_mut_slice(),
        );
        self.rgtrans.g1d_to_r3d(
            self.gvec,
            self.pwden,
            &self.ws.vlocg_dn,
            self.ws.vloc_3d_dn.as_mut_slice(),
        );
    }

    fn solve_eigen_equations(&mut self, scf_iter: usize, energy_diff: f64) -> f64 {
        let eigvalue_epsilon = get_eigvalue_epsilon(
            self.geom_iter,
            scf_iter,
            self.control,
            self.ntot_elec,
            energy_diff,
            self.npw_wfc_max,
        );

        solve_spin_eigen_equations(
            self.rgtrans,
            &self.ws.vloc_3d_up,
            &self.ws.vloc_3d_dn,
            eigvalue_epsilon,
            self.geom_iter,
            scf_iter,
            self.vkscf,
            self.vkevals,
            self.vkevecs,
        );

        eigvalue_epsilon
    }

    fn update_occupations(&mut self) -> f64 {
        let fermi_level =
            self.fermi_driver
                .get_fermi_level(self.vkscf, self.ntot_elec, self.vkevals);
        self.hubbard_energy =
            hubbard::update_hubbard_spin(self.control, self.vkscf, &*self.vkevecs);

        fermi_level
    }

    fn compute_harris_energy(&mut self) -> f64 {
        let (rhog_up, rhog_dn) = self.rhog.as_spin().unwrap();

        compute_total_energy(
            self.pwden,
            self.crystal.get_latt(),
            rhog_up,
            rhog_dn,
            self.vkscf,
            self.vkevals,
            self.rho_3d,
            self.rhocore_3d,
            &self.ws.exc_3d,
            &self.ws.vxc_3d,
            self.ewald.get_energy(),
            self.hubbard_energy,
            &mut self.ws.rhog_tot_scratch,
        )
    }

    fn rebuild_density(&mut self) {
        self.density_driver.compute_charge_density(
            self.vkscf,
            self.rgtrans,
            self.vkevecs,
            self.crystal.get_latt().volume(),
            self.rho_3d,
        );
    }

    fn compute_charge(&mut self) -> f64 {
        if let RHOR::Spin(rho_3d_up, rho_3d_dn) = self.rho_3d {
            let charge_up =
                rho_3d_up.sum().re * self.crystal.get_latt().volume() / self.fft_ntotf64;
            let charge_dn =
                rho_3d_dn.sum().re * self.crystal.get_latt().volume() / self.fft_ntotf64;

            charge_up + charge_dn
        } else {
            0.0
        }
    }

    fn refresh_energy_terms(&mut self) {
        if let RHOR::Spin(rho_3d_up, rho_3d_dn) = self.rho_3d {
            self.rgtrans.r3d_to_g1d(
                self.gvec,
                self.pwden,
                rho_3d_up.as_slice(),
                &mut self.ws.rhog_out_up,
            );
            self.rgtrans.r3d_to_g1d(
                self.gvec,
                self.pwden,
                rho_3d_dn.as_slice(),
                &mut self.ws.rhog_out_dn,
            );
        }

        self.refresh_spin_xc_for_iteration(0.5);
    }

    fn compute_scf_energy(&mut self) -> f64 {
        compute_total_energy(
            self.pwden,
            self.crystal.get_latt(),
            &self.ws.rhog_out_up,
            &self.ws.rhog_out_dn,
            self.vkscf,
            self.vkevals,
            self.rho_3d,
            self.rhocore_3d,
            &self.ws.exc_3d,
            &self.ws.vxc_3d,
            self.ewald.get_energy(),
            self.hubbard_energy,
            &mut self.ws.rhog_tot_scratch,
        )
    }

    fn mix_and_rebuild_potential(&mut self) {
        if let RHOR::Spin(rho_3d_up, rho_3d_dn) = self.rho_3d {
            let rho_3d_up = rho_3d_up.as_mut_slice();
            let rho_3d_dn = rho_3d_dn.as_mut_slice();

            if let RHOG::Spin(rhog_up, rhog_dn) = self.rhog {
                for ipw in 0..self.pwden.get_n_plane_waves() {
                    self.ws.rhog_total[ipw] = rhog_up[ipw] + rhog_dn[ipw];
                    self.ws.rhog_spin[ipw] = rhog_up[ipw] - rhog_dn[ipw];
                }

                for ipw in 0..self.pwden.get_n_plane_waves() {
                    self.ws.rhog_diff[ipw] = (self.ws.rhog_out_up[ipw] + self.ws.rhog_out_dn[ipw])
                        - (rhog_up[ipw] + rhog_dn[ipw]);
                }

                self.mixing_rho.compute_next_density(
                    self.pwden.get_g(),
                    &mut self.ws.rhog_total,
                    &self.ws.rhog_diff,
                );

                for ipw in 0..self.pwden.get_n_plane_waves() {
                    self.ws.rhog_diff[ipw] = (self.ws.rhog_out_up[ipw] - self.ws.rhog_out_dn[ipw])
                        - (rhog_up[ipw] - rhog_dn[ipw]);
                }

                self.mixing_spin.compute_next_density(
                    self.pwden.get_g(),
                    &mut self.ws.rhog_spin,
                    &self.ws.rhog_diff,
                );

                for ipw in 0..self.pwden.get_n_plane_waves() {
                    rhog_up[ipw] = (self.ws.rhog_total[ipw] + self.ws.rhog_spin[ipw]) / 2.0;
                    rhog_dn[ipw] = (self.ws.rhog_total[ipw] - self.ws.rhog_spin[ipw]) / 2.0;
                }

                self.rgtrans
                    .g1d_to_r3d(self.gvec, self.pwden, rhog_up, rho_3d_up);
                self.rgtrans
                    .g1d_to_r3d(self.gvec, self.pwden, rhog_dn, rho_3d_dn);
            }
        }

        self.rebuild_hartree_from_spin_density();
        self.refresh_spin_xc_for_iteration(0.5);
        self.refresh_vxcg_from_vxc_3d();
        self.rebuild_vlocg_from_components();
    }
}

impl SCF for SCFSpin {
    fn run(
        &self,
        geom_iter: usize,
        control: &Control,
        crystal: &Crystal,
        gvec: &GVector,
        pwden: &PWDensity,
        pots: &PSPot,
        rgtrans: &RGTransform,
        kpts: &dyn KPTS,
        ewald: &Ewald,
        vpwwfc: &[PWBasis],
        vkscf: &mut VKSCF,
        rhog: &mut RHOG,
        rho_3d: &mut RHOR,
        rhocore_3d: &Array3<c64>,
        vkevals: &mut VKEigenValue,
        vkevecs: &mut VKEigenVector,
        symdrv: &dyn SymmetryDriver,
        stress_total: &mut Matrix<f64>,
        force_total: &mut Vec<Vector3f64>,
    ) {
        let density_driver = density::new(control.get_spin_scheme_enum());
        utils::validate_hse06_runtime_constraints(control, kpts);

        let blatt = crystal.get_latt().reciprocal();

        //
        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());

        let fft_shape = fftgrid.get_size();

        let nfft = fftgrid.get_ntot();

        let [n1, n2, n3] = fft_shape;

        let npw_rho = pwden.get_n_plane_waves();

        let mut ws = SpinScfWorkspace::new(npw_rho, [n1, n2, n3]);
        ws.validate(npw_rho, [n1, n2, n3]);

        if let RHOG::Spin(rhog_up, rhog_dn) = rhog {
            sum_spin_channels(rhog_up, rhog_dn, &mut ws.rhog_tot);
        }

        // v_psloc in G space; this will not change for a fixed set of ion positions

        vloc::from_atomic_super_position(pots, crystal, gvec, pwden, &mut ws.vpslocg);

        // v_h in G space; this will change with the density

        hartree::potential(pwden.get_g(), &ws.rhog_tot, &mut ws.vhg);

        // v_xc in r space first and then transform to G space; this changes
        // with density every SCF iteration.

        let xc = xc::new(control.get_xc_scheme_enum());

        add_core_density_to_spin_channels(rho_3d, rhocore_3d, 1.0);
        xc.potential_and_energy(gvec, pwden, rgtrans, rho_3d, &mut ws.vxc_3d, &mut ws.exc_3d);
        add_core_density_to_spin_channels(rho_3d, rhocore_3d, -1.0);

        {
            let (vxc_3d_up, vxc_3d_dn) = ws.vxc_3d.as_spin().unwrap();
            let (vxcg_up, vxcg_dn) = ws.vxcg.as_spin_mut().unwrap();

            rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d_up.as_slice(), vxcg_up);
            rgtrans.r3d_to_g1d(gvec, pwden, vxc_3d_dn.as_slice(), vxcg_dn);
        }

        // v_xc + v_h + v_psloc in G space

        {
            let (vxcg_up, vxcg_dn) = ws.vxcg.as_spin().unwrap();

            // spin up

            for (v_loc, v_xc, v_ha, v_psloc) in multizip((
                ws.vlocg_up.iter_mut(),
                vxcg_up.iter(),
                ws.vhg.iter(),
                ws.vpslocg.iter(),
            )) {
                *v_loc = *v_xc + *v_ha + *v_psloc;
            }

            // spin dn

            for (v_loc, v_xc, v_ha, v_psloc) in multizip((
                ws.vlocg_dn.iter_mut(),
                vxcg_dn.iter(),
                ws.vhg.iter(),
                ws.vpslocg.iter(),
            )) {
                *v_loc = *v_xc + *v_ha + *v_psloc;
            }
        }
        //

        // density mixing

        let mut mixing_rho = mixing::new(control);

        let mut mixing_spin = mixing::new(control);

        let ntot_elec = crystal.get_n_total_electrons(pots);
        let fft_ntotf64 = fftgrid.get_ntotf64();

        let npw_wfc_max = get_n_plane_waves_max(&vpwwfc);

        // Reuse one Fermi-level driver across SCF iterations to avoid repeated allocations.
        let fermi_driver = fermilevel::new(control.get_spin_scheme_enum());

        let mut adapter = SpinIterationAdapter {
            geom_iter,
            control,
            crystal,
            gvec,
            pwden,
            rgtrans,
            ewald,
            ntot_elec,
            npw_wfc_max,
            fft_ntotf64,
            xc: xc.as_ref(),
            density_driver: density_driver.as_ref(),
            fermi_driver: fermi_driver.as_ref(),
            mixing_rho: mixing_rho.as_mut(),
            mixing_spin: mixing_spin.as_mut(),
            vkscf,
            rhog,
            rho_3d,
            rhocore_3d,
            vkevals,
            vkevecs,
            ws: &mut ws,
            hubbard_energy: 0.0,
        };

        run_scf_iteration_engine(control, &mut adapter);
        drop(adapter);

        // display eigenvalues

        //let (vkscf_up, vkscf_dn) = utility::get_slice_up_dn(vkscf);
        //let (vkevals_up, vkevals_dn) = utility::get_slice_up_dn(vkevals);

        if dwmpi::is_root() {
            if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
                if let VKEigenValue::Spin(vkevals_up, vkevals_dn) = vkevals {
                    debug_assert_eq!(vkscf_up.len(), vkscf_dn.len());
                    debug_assert_eq!(vkscf_up.len(), vkevals_up.len());
                    debug_assert_eq!(vkscf_up.len(), vkevals_dn.len());
                    debug_assert_eq!(vkscf_up.len(), vpwwfc.len());

                    for (kscf_up_k, kscf_dn_k, evals_up, evals_dn, pwwfc_k) in itertools::multizip(
                        (
                            vkscf_up.iter(),
                            vkscf_dn.iter(),
                            vkevals_up.iter(),
                            vkevals_dn.iter(),
                            vpwwfc.iter(),
                        ),
                    )
                    {
                        let ik_global = kscf_up_k.get_ik();
                        let k_frac = kpts.get_k_frac(ik_global);
                        let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
                        let npw_wfc = pwwfc_k.get_n_plane_waves();

                        print_k_point(ik_global, k_frac, k_cart, npw_wfc);

                        let occ_up = kscf_up_k.get_occ();
                        let occ_dn = kscf_dn_k.get_occ();

                        print_eigen_values(evals_up, occ_up, evals_dn, occ_dn);
                    }
                }
            }
        }

        // force

        let natoms = crystal.get_n_atoms();

        let mut force_loc = vec![Vector3f64::zeros(); natoms];
        let mut force_vnl_local = vec![Vector3f64::zeros(); natoms];
        let mut force_vnl = vec![Vector3f64::zeros(); natoms];
        let mut force_spectral_ws = force::SpectralWorkspace::new();

        force::vpsloc_with_workspace(
            pots,
            crystal,
            gvec,
            pwden,
            &mut force_spectral_ws,
            &ws.rhog_tot,
            &mut force_loc,
        );

        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
            if let VKEigenVector::Spin(vkevecs_up, vkevecs_dn) = vkevecs {
                force::vnl(crystal, &vkscf_up, &vkevecs_up, &mut force_vnl_local);
                force::vnl(crystal, &vkscf_dn, &vkevecs_dn, &mut force_vnl_local);
            }
        }

        dwmpi::reduce_slice_sum(
            vector3::as_slice_of_element(&force_vnl_local),
            vector3::as_mut_slice_of_element(&mut force_vnl),
            MPI_COMM_WORLD,
        );

        dwmpi::bcast_slice(
            vector3::as_mut_slice_of_element(&mut force_vnl),
            MPI_COMM_WORLD,
        );

        let mut force_ewald = ewald.get_force().to_vec();

        let mut force_nlcc = vec![Vector3f64::zeros(); natoms];

        {
            let (vxcg_up, vxcg_dn) = ws.vxcg.as_spin().unwrap();

            let mut force_nlcc_up = vec![Vector3f64::zeros(); natoms];
            let mut force_nlcc_dn = vec![Vector3f64::zeros(); natoms];

            force::nlcc_xc_with_workspace(
                pots,
                crystal,
                gvec,
                pwden,
                &mut force_spectral_ws,
                vxcg_up,
                &mut force_nlcc_up,
            );
            force::nlcc_xc_with_workspace(
                pots,
                crystal,
                gvec,
                pwden,
                &mut force_spectral_ws,
                vxcg_dn,
                &mut force_nlcc_dn,
            );

            for iat in 0..natoms {
                force_nlcc[iat] = (force_nlcc_up[iat] + force_nlcc_dn[iat]) / 2.0;
            }
        }

        utils::finalize_force_by_parts(
            control,
            crystal,
            symdrv,
            force_total.as_mut_slice(),
            force_ewald.as_mut_slice(),
            force_loc.as_mut_slice(),
            force_vnl.as_mut_slice(),
            force_nlcc.as_mut_slice(),
        );

        // stress

        let mut stress_kin_local = Matrix::<f64>::new(3, 3);
        let mut stress_vnl_local = Matrix::<f64>::new(3, 3);

        if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
            if let VKEigenVector::Spin(vkevecs_up, vkevecs_dn) = vkevecs {
                let stress_kin_up = stress::kinetic(crystal, &vkscf_up, &vkevecs_up);
                let stress_kin_dn = stress::kinetic(crystal, &vkscf_dn, &vkevecs_dn);

                stress_kin_local = stress_kin_up + stress_kin_dn;

                let stress_vnl_up = stress::vnl(crystal, &vkscf_up, &vkevecs_up);
                let stress_vnl_dn = stress::vnl(crystal, &vkscf_dn, &vkevecs_dn);

                stress_vnl_local = stress_vnl_up + stress_vnl_dn;
            }
        }

        let mut stress_kin = Matrix::<f64>::new(3, 3);
        let mut stress_vnl = Matrix::<f64>::new(3, 3);

        dwmpi::reduce_slice_sum(
            stress_kin_local.as_slice(),
            stress_kin.as_mut_slice(),
            MPI_COMM_WORLD,
        );
        dwmpi::reduce_slice_sum(
            stress_vnl_local.as_slice(),
            stress_vnl.as_mut_slice(),
            MPI_COMM_WORLD,
        );
        dwmpi::bcast_slice(stress_kin.as_mut_slice(), MPI_COMM_WORLD);
        dwmpi::bcast_slice(stress_vnl.as_mut_slice(), MPI_COMM_WORLD);

        let mut stress_hartree = stress::hartree(gvec, pwden, &ws.rhog_tot);
        let mut stress_xc = stress::xc_spin(
            crystal.get_latt(),
            rho_3d,
            rhocore_3d,
            &ws.vxc_3d,
            &ws.exc_3d,
        );
        let mut stress_spectral_ws = stress::SpectralWorkspace::new();

        let (vxcg_up, vxcg_dn) = ws.vxcg.as_spin().unwrap();

        let mut stress_xc_nlcc_up = Matrix::<f64>::new(3, 3);
        let mut stress_xc_nlcc_dn = Matrix::<f64>::new(3, 3);
        stress::nlcc_xc_with_workspace(
            pots,
            crystal,
            gvec,
            pwden,
            &mut stress_spectral_ws,
            vxcg_up,
            &mut stress_xc_nlcc_up,
        );
        stress::nlcc_xc_with_workspace(
            pots,
            crystal,
            gvec,
            pwden,
            &mut stress_spectral_ws,
            vxcg_dn,
            &mut stress_xc_nlcc_dn,
        );

        let mut stress_xc_nlcc = Matrix::<f64>::new(3, 3);

        for i in 0..3 {
            for j in 0..3 {
                stress_xc_nlcc[[i, j]] =
                    (stress_xc_nlcc_up[[i, j]] + stress_xc_nlcc_dn[[i, j]]) / 2.0;
            }
        }

        let mut stress_loc = Matrix::<f64>::new(3, 3);
        stress::vpsloc_with_workspace(
            pots,
            crystal,
            gvec,
            pwden,
            &mut stress_spectral_ws,
            &ws.rhog_tot,
            &mut stress_loc,
        );
        let mut stress_ewald = ewald.get_stress().clone();

        utils::finalize_stress_by_parts(
            control,
            crystal,
            symdrv,
            stress_total,
            &mut stress_kin,
            &mut stress_hartree,
            &mut stress_xc,
            &mut stress_xc_nlcc,
            &mut stress_loc,
            &mut stress_vnl,
            &mut stress_ewald,
        );
    }
}

pub fn compute_total_energy(
    pwden: &PWDensity,
    latt: &Lattice,
    rhog_up: &[c64],
    rhog_dn: &[c64],
    vkscf: &VKSCF,
    vevals: &VKEigenValue,
    rho_3d: &mut RHOR,
    rhocore_3d: &Array3<c64>,
    exc_3d: &Array3<c64>,
    vxc_3d: &VXCR,
    ew_total: f64,
    hubbard_energy: f64,
    rhog_tot_scratch: &mut [c64],
) -> f64 {
    let npw_rho = pwden.get_n_plane_waves();

    // hartree energy

    debug_assert_eq!(rhog_up.len(), npw_rho);
    debug_assert_eq!(rhog_dn.len(), npw_rho);
    debug_assert_eq!(rhog_tot_scratch.len(), npw_rho);
    sum_spin_channels(rhog_up, rhog_dn, rhog_tot_scratch);

    let etot_hartree = energy::hartree(pwden, latt, rhog_tot_scratch);

    //

    let mut etot_bands_local = 0.0;
    if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
        if let VKEigenValue::Spin(vevals_up, vevals_dn) = vevals {
            let etot_bands_up = get_bands_energy(vkscf_up, vevals_up);
            let etot_bands_dn = get_bands_energy(vkscf_dn, vevals_dn);

            etot_bands_local = etot_bands_up + etot_bands_dn;
        }
    }
    let mut etot_bands = 0.0;
    dwmpi::reduce_scalar_sum(&etot_bands_local, &mut etot_bands, MPI_COMM_WORLD);
    dwmpi::bcast_scalar(&mut etot_bands, MPI_COMM_WORLD);

    let etot_vxc = energy::vxc_spin(latt, rho_3d, rhocore_3d.as_slice(), vxc_3d);

    let etot_xc = energy::exc_spin(latt, &rho_3d, &rhocore_3d, &exc_3d);
    let mut hybrid_exchange_local = 0.0;
    if let VKSCF::Spin(vkscf_up, vkscf_dn) = vkscf {
        hybrid_exchange_local =
            get_hybrid_exchange_energy(vkscf_up) + get_hybrid_exchange_energy(vkscf_dn);
    }
    let mut hybrid_exchange = 0.0;
    dwmpi::reduce_scalar_sum(&hybrid_exchange_local, &mut hybrid_exchange, MPI_COMM_WORLD);
    dwmpi::bcast_scalar(&mut hybrid_exchange, MPI_COMM_WORLD);

    let etot_one = etot_bands - etot_vxc - 2.0 * etot_hartree;

    let etot = etot_one + etot_xc + etot_hartree + ew_total + hubbard_energy - hybrid_exchange;

    etot
}

//hwf_energy = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband_hwf
//etot       = eband + ( etxc - etxcc ) + ewld + ehart + demet + deband + descf

pub fn get_bands_energy(vkscf: &[KSCF], vevals: &Vec<Vec<f64>>) -> f64 {
    let etot_bands = energy::band_structure(vkscf, vevals);

    //mpi::reduce_scalar_sum(&etot_bands_local, &mut etot_bands, MPI_COMM_WORLD);
    //mpi::bcast_scalar(&etot_bands, MPI_COMM_WORLD);

    etot_bands
}

pub fn get_hybrid_exchange_energy(vkscf: &[KSCF]) -> f64 {
    vkscf
        .iter()
        .map(|kscf| kscf.get_hybrid_exchange_energy() * kscf.get_k_weight())
        .sum::<f64>()
}

pub fn get_eigvalue_epsilon(
    geom_iter: usize,
    scf_iter: usize,
    control: &Control,
    ntot_elec: f64,
    energy_diff: f64,
    npw_wfc: usize,
) -> f64 {
    let mut eig_epsilon: f64;

    //if control.is_band() {
    if control.get_scf_max_iter() <= 1 {
        eig_epsilon = control.get_eigval_epsilon();
    } else {
        if geom_iter == 1 {
            match scf_iter {
                1 => {
                    eig_epsilon = EPS2 * EV_TO_HA;
                }

                2 => {
                    eig_epsilon = EPS3 * EV_TO_HA;
                }

                3 => {
                    eig_epsilon = EPS4 * EV_TO_HA;
                }

                _ => {
                    eig_epsilon =
                        (EPS4 * EV_TO_HA).min(0.0001 * energy_diff / (1.0_f64).max(ntot_elec));

                    eig_epsilon = eig_epsilon
                        .max(EPS13 * EV_TO_HA)
                        .min(control.get_eigval_epsilon());
                }
            }
        } else {
            match scf_iter {
                1 => {
                    eig_epsilon = EPS2 * EV_TO_HA;
                }

                2 => {
                    eig_epsilon = EPS4 * EV_TO_HA;
                }

                3 => {
                    eig_epsilon = EPS6 * EV_TO_HA;
                }

                _ => {
                    eig_epsilon = (EPS11 * EV_TO_HA)
                        .min(energy_diff / (npw_wfc as f64).powf(1.0) / (1.0_f64).max(ntot_elec));
                }
            }
        }

        eig_epsilon = eig_epsilon.max(EPS16 * EV_TO_HA);
    }

    eig_epsilon
}

fn get_n_plane_waves_max(vpwwfc: &[PWBasis]) -> usize {
    let mut npw_max = 0;

    for pwwfc in vpwwfc.iter() {
        let npw = pwwfc.get_n_plane_waves();

        if npw > npw_max {
            npw_max = npw;
        }
    }

    npw_max
}

pub fn print_k_point(ik: usize, xk_frac: Vector3f64, xk_cart: Vector3f64, npw_wfc: usize) {
    println!();

    println!("   kpoint-{} npws = {}", ik + 1, npw_wfc);

    println!(
        "     k_frac = [ {:.8}, {:.8}, {:.8} ]",
        xk_frac.x, xk_frac.y, xk_frac.z
    );

    println!(
        "     k_cart = [ {:.8}, {:.8}, {:.8} ] (1/a0)",
        xk_cart.x, xk_cart.y, xk_cart.z
    );
}

pub fn print_eigen_values(v_up: &[f64], occ_up: &[f64], v_dn: &[f64], occ_dn: &[f64]) {
    println!();

    for (i, _elem) in v_up.iter().enumerate() {
        println!(
            "       {:<6} {:16.6} {:12.6} {:16.6} {:12.6}",
            i + 1,
            v_up[i] * HA_TO_EV,
            occ_up[i],
            v_dn[i] * HA_TO_EV,
            occ_dn[i]
        );
    }
}
