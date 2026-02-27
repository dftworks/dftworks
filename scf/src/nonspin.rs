//#![allow(warnings)]

use super::engine::{run_scf_iteration_engine, ScfIterationAdapter};
use super::hubbard;
use super::utils;
use crate::SCF;
use control::Control;
use crystal::Crystal;
use dfttypes::*;
use ewald::Ewald;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use matrix::Matrix;
use ndarray::Array3;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use symmetry::SymmetryDriver;
use types::*;
use vector3::Vector3f64;

//use rayon::prelude::*;

// Non-spin SCF driver:
// - single density channel
// - single local KS potential
// - iterative solve/mix/update loop until energy convergence.
pub struct SCFNonspin {}

impl SCFNonspin {
    pub fn new() -> SCFNonspin {
        SCFNonspin {}
    }
}

// Scratch/work buffers reused across the non-spin SCF loop.
// All sizes are derived once from (npw_rho, fft_shape) and reused in-place.
struct NonSpinScfWorkspace {
    vhg: Vec<c64>,
    vxcg: VXCG,
    vxc_3d: VXCR,
    exc_3d: Array3<c64>,
    vpslocg: Vec<c64>,
    vlocg: Vec<c64>,
    vloc_3d: Array3<c64>,
    rhog_out: Vec<c64>,
    rhog_diff: Vec<c64>,
}

impl NonSpinScfWorkspace {
    fn new(npw_rho: usize, fft_shape: [usize; 3]) -> Self {
        Self {
            vhg: vec![c64::zero(); npw_rho],
            vxcg: VXCG::NonSpin(vec![c64::zero(); npw_rho]),
            vxc_3d: VXCR::NonSpin(Array3::<c64>::new(fft_shape)),
            exc_3d: Array3::<c64>::new(fft_shape),
            vpslocg: vec![c64::zero(); npw_rho],
            vlocg: vec![c64::zero(); npw_rho],
            vloc_3d: Array3::<c64>::new(fft_shape),
            rhog_out: vec![c64::zero(); npw_rho],
            rhog_diff: vec![c64::zero(); npw_rho],
        }
    }

    fn validate(&self, npw_rho: usize, fft_shape: [usize; 3]) {
        let nfft = fft_shape[0] * fft_shape[1] * fft_shape[2];

        debug_assert_eq!(self.vhg.len(), npw_rho);
        debug_assert_eq!(self.vpslocg.len(), npw_rho);
        debug_assert_eq!(self.vlocg.len(), npw_rho);
        debug_assert_eq!(self.rhog_out.len(), npw_rho);
        debug_assert_eq!(self.rhog_diff.len(), npw_rho);
        debug_assert_eq!(self.vxcg.as_non_spin().unwrap().len(), npw_rho);
        debug_assert_eq!(self.vxc_3d.as_non_spin().unwrap().as_slice().len(), nfft);
        debug_assert_eq!(self.exc_3d.as_slice().len(), nfft);
        debug_assert_eq!(self.vloc_3d.as_slice().len(), nfft);
    }
}

struct NonSpinIterationAdapter<'ctx, 'ks> {
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
    mixing: &'ctx mut dyn mixing::Mixing,
    vkscf: &'ctx mut VKSCF<'ks>,
    rhog: &'ctx mut RHOG,
    rho_3d: &'ctx mut RHOR,
    rhocore_3d: &'ctx Array3<c64>,
    vkevals: &'ctx mut VKEigenValue,
    vkevecs: &'ctx mut VKEigenVector,
    ws: &'ctx mut NonSpinScfWorkspace,
    hubbard_energy: f64,
}

impl ScfIterationAdapter for NonSpinIterationAdapter<'_, '_> {
    fn prepare_potential_in_rspace(&mut self) {
        self.rgtrans.g1d_to_r3d(
            self.gvec,
            self.pwden,
            &self.ws.vlocg,
            self.ws.vloc_3d.as_mut_slice(),
        );
    }

    fn solve_eigen_equations(&mut self, scf_iter: usize, energy_diff: f64) -> f64 {
        let eigvalue_epsilon = utils::get_eigvalue_epsilon(
            self.geom_iter,
            scf_iter,
            self.control,
            self.ntot_elec,
            energy_diff,
            self.npw_wfc_max,
        );

        utils::solve_eigen_equations(
            self.rgtrans,
            &self.ws.vloc_3d,
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
            hubbard::update_hubbard_nonspin(self.control, self.vkscf, &*self.vkevecs);

        fermi_level
    }

    fn compute_harris_energy(&mut self) -> f64 {
        utils::compute_total_energy(
            self.pwden,
            self.crystal,
            self.rhog.as_non_spin().unwrap(),
            self.vkscf.as_non_spin().unwrap(),
            self.vkevals.as_non_spin().unwrap(),
            self.rho_3d.as_non_spin_mut().unwrap(),
            self.rhocore_3d,
            &self.ws.exc_3d,
            self.ws.vxc_3d.as_non_spin().unwrap(),
            self.ewald.get_energy(),
            self.hubbard_energy,
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
        self.rho_3d.as_non_spin().unwrap().sum().re * self.crystal.get_latt().volume()
            / self.fft_ntotf64
    }

    fn refresh_energy_terms(&mut self) {
        utils::compute_rho_of_g(
            self.gvec,
            self.pwden,
            self.rgtrans,
            self.rho_3d,
            &mut self.ws.rhog_out,
        );

        utils::compute_v_e_xc_of_r(
            self.xc,
            self.gvec,
            self.pwden,
            self.rgtrans,
            self.rho_3d,
            self.rhocore_3d,
            &mut self.ws.vxc_3d,
            &mut self.ws.exc_3d,
        );
    }

    fn compute_scf_energy(&mut self) -> f64 {
        utils::compute_total_energy(
            self.pwden,
            self.crystal,
            &self.ws.rhog_out,
            self.vkscf.as_non_spin().unwrap(),
            self.vkevals.as_non_spin().unwrap(),
            self.rho_3d.as_non_spin_mut().unwrap(),
            self.rhocore_3d,
            &self.ws.exc_3d,
            self.ws.vxc_3d.as_non_spin().unwrap(),
            self.ewald.get_energy(),
            self.hubbard_energy,
        )
    }

    fn mix_and_rebuild_potential(&mut self) {
        utils::compute_next_density(
            self.pwden,
            self.mixing,
            &self.ws.rhog_out,
            &mut self.ws.rhog_diff,
            self.rhog,
        );

        self.rgtrans.g1d_to_r3d(
            self.gvec,
            self.pwden,
            self.rhog.as_non_spin().unwrap(),
            self.rho_3d.as_non_spin_mut().unwrap().as_mut_slice(),
        );

        utils::compute_v_hartree(self.pwden, self.rhog, &mut self.ws.vhg);

        utils::compute_v_e_xc_of_r(
            self.xc,
            self.gvec,
            self.pwden,
            self.rgtrans,
            self.rho_3d,
            self.rhocore_3d,
            &mut self.ws.vxc_3d,
            &mut self.ws.exc_3d,
        );

        utils::compute_v_xc_of_g(
            self.gvec,
            self.pwden,
            self.rgtrans,
            &self.ws.vxc_3d,
            &mut self.ws.vxcg,
        );

        utils::add_up_v(
            &self.ws.vpslocg,
            &self.ws.vhg,
            &self.ws.vxcg,
            &mut self.ws.vlocg,
        );
    }
}

impl SCF for SCFNonspin {
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
        //println!("");
        //println!("   {:*^120}", " self-consistent field ");

        // Density helper chosen from spin scheme; this resolves to non-spin here.
        let density_driver = density::new(control.get_spin_scheme_enum());
        utils::validate_hse06_runtime_constraints(control, kpts);

        //

        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());

        let fft_shape = fftgrid.get_size();
        let npw_rho = pwden.get_n_plane_waves();

        let mut ws = NonSpinScfWorkspace::new(npw_rho, fft_shape);
        ws.validate(npw_rho, fft_shape);

        let xc = xc::new(control.get_xc_scheme_enum());

        // v_psloc in G space; this will not change for a fixed set of ion positions

        vloc::from_atomic_super_position(pots, crystal, gvec, pwden, &mut ws.vpslocg);

        // v_ha and v_xc change with density

        utils::compute_v_hartree(pwden, rhog, &mut ws.vhg);

        utils::compute_v_e_xc_of_r(
            xc.as_ref(),
            gvec,
            pwden,
            rgtrans,
            rho_3d,
            rhocore_3d,
            &mut ws.vxc_3d,
            &mut ws.exc_3d,
        );

        utils::compute_v_xc_of_g(gvec, pwden, rgtrans, &ws.vxc_3d, &mut ws.vxcg);

        //

        // Total local KS potential in reciprocal space.
        utils::add_up_v(&ws.vpslocg, &ws.vhg, &ws.vxcg, &mut ws.vlocg);

        // density mixing

        let mut mixing = mixing::new(control);

        let ntot_elec = crystal.get_n_total_electrons(pots);
        let fft_ntotf64 = fftgrid.get_ntotf64();

        let npw_wfc_max = utils::get_n_plane_waves_max(vpwwfc);

        let fermi_driver = fermilevel::new(control.get_spin_scheme_enum());

        let mut adapter = NonSpinIterationAdapter {
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
            mixing: mixing.as_mut(),
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

        // after the SCF iterations

        // display eigenvalues

        utils::display_eigen_values(
            control.get_verbosity(),
            crystal,
            kpts,
            vpwwfc,
            vkscf,
            vkevals,
        );

        // force

        utils::compute_force(
            control,
            crystal,
            gvec,
            pwden,
            pots,
            ewald,
            vkscf,
            vkevecs,
            rhog,
            &ws.vxcg,
            symdrv,
            force_total,
        );

        // stress

        utils::compute_stress(
            control,
            crystal,
            gvec,
            pwden,
            pots,
            ewald,
            vkscf,
            vkevecs,
            rhog,
            rho_3d,
            rhocore_3d,
            &ws.vxcg,
            &ws.vxc_3d,
            &ws.exc_3d,
            symdrv,
            stress_total,
        );
    }
}
