// in.ctrl parameters, grouped by topic.
//
// To add an input key:
//   1. add a field (and its default) to the group it belongs to, and
//   2. add one line to the `match` in `Control::set_key` at the end of this file.
// Cross-key checks live in `Control::validate_*` (lib.rs).
//
// Values are stored in internal units (Hartree, bohr); the field comments note
// the input unit when it differs.

use crate::*;
use dwconsts::{EPS6, EV_TO_HA};

// 1 a.u. electric field = Eh/(e*a0) = 5.14220674763e11 V/m = 51.4220674763 V/Ang.
pub(crate) const ELECTRIC_FIELD_AU_TO_VANG: f64 = 51.4220674763;
const ELECTRIC_FIELD_VANG_TO_AU: f64 = 1.0 / ELECTRIC_FIELD_AU_TO_VANG;

// ---- value parsers -------------------------------------------------------

fn parse_bool(value: &str) -> Result<bool, String> {
    value
        .trim()
        .parse::<bool>()
        .map_err(|_| "expected a boolean ('true' or 'false')".to_string())
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| "expected an unsigned integer".to_string())
}

fn parse_i32(value: &str) -> Result<i32, String> {
    value
        .trim()
        .parse::<i32>()
        .map_err(|_| "expected an integer".to_string())
}

fn parse_f64(value: &str) -> Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| "expected a floating-point number".to_string())
}

/// Energy given in eV, stored in Hartree.
fn parse_ev_as_ha(value: &str) -> Result<f64, String> {
    Ok(parse_f64(value)? * EV_TO_HA)
}

fn parse_string(value: &str) -> Result<String, String> {
    Ok(value.trim().to_string())
}

/// `none`/`auto`/`unset` mean "no fixed seed".
fn parse_optional_seed(value: &str) -> Result<Option<u64>, String> {
    let raw = value.trim();
    if raw.eq_ignore_ascii_case("none")
        || raw.eq_ignore_ascii_case("auto")
        || raw.eq_ignore_ascii_case("unset")
    {
        return Ok(None);
    }
    raw.parse::<u64>()
        .map(Some)
        .map_err(|_| "expected a non-negative 64-bit integer".to_string())
}

fn parse_vdw_damping(value: &str) -> Result<String, String> {
    let damping = value.trim().to_lowercase();
    if damping == "zero" || damping == "bj" {
        Ok(damping)
    } else {
        Err("expected one of: zero, bj".to_string())
    }
}

/// Turns an enum's `parse` result into a parser error listing the accepted values.
fn expect_one_of<T>(parsed: Option<T>, accepted: &str) -> Result<T, String> {
    parsed.ok_or_else(|| format!("expected one of: {}", accepted))
}

// ---- groups --------------------------------------------------------------

/// Run mode and global switches.
#[derive(Debug, Clone, PartialEq)]
pub struct RunParams {
    pub task: String,
    pub restart: bool,
    pub symmetry: bool,
    pub random_seed: Option<u64>,
    pub kpoint_schedule: KPointScheduleScheme,
}

impl Default for RunParams {
    fn default() -> Self {
        RunParams {
            task: "scf".to_string(),
            restart: false,
            symmetry: false,
            random_seed: None,
            kpoint_schedule: KPointScheduleScheme::CostAware,
        }
    }
}

/// Plane-wave basis and band count.
#[derive(Debug, Clone, PartialEq)]
pub struct BasisParams {
    /// Input in eV, stored in Hartree.
    pub ecut_wfc: f64,
    /// Input in eV, stored in Hartree. `None` means 4 * ecut_wfc; use `resolved_ecut_rho()`.
    pub ecut_rho: Option<f64>,
    pub nband: usize,
}

impl Default for BasisParams {
    fn default() -> Self {
        BasisParams {
            ecut_wfc: 400.0 * EV_TO_HA,
            ecut_rho: None,
            nband: 0,
        }
    }
}

impl BasisParams {
    /// Density cutoff in Hartree: the explicit `ecut_rho`, else 4 * ecut_wfc.
    pub fn resolved_ecut_rho(&self) -> f64 {
        self.ecut_rho.unwrap_or(4.0 * self.ecut_wfc)
    }
}

/// Physical model: spin treatment, functional, pseudopotentials, k-points, occupations.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelParams {
    pub spin_scheme: SpinScheme,
    pub xc_scheme: XcScheme,
    pub pot_scheme: PotScheme,
    pub kpts_scheme: KptsScheme,
    pub smearing_scheme: SmearingScheme,
    /// K
    pub temperature: f64,
    pub occ_inversion: f64,
}

impl Default for ModelParams {
    fn default() -> Self {
        ModelParams {
            spin_scheme: SpinScheme::NonSpin,
            xc_scheme: XcScheme::LdaPz,
            pot_scheme: PotScheme::Upf,
            kpts_scheme: KptsScheme::Kmesh,
            smearing_scheme: SmearingScheme::Mp2,
            temperature: 0.0,
            occ_inversion: 0.0,
        }
    }
}

/// SCF loop, convergence thresholds, and eigensolver.
#[derive(Debug, Clone, PartialEq)]
pub struct ScfParams {
    pub harris: bool,
    pub min_iter: usize,
    pub max_iter: usize,
    pub max_iter_rand_wfc: usize,
    pub max_iter_wfc: usize,
    pub rho_epsilon: f64,
    /// Input in eV, stored in Hartree.
    pub energy_epsilon: f64,
    /// Input in eV, stored in Hartree.
    pub eigval_epsilon: f64,
    pub eigval_same_epsilon: bool,
    pub eigen_solver: EigenSolverScheme,
    pub davidson_ndim: usize,
}

impl Default for ScfParams {
    fn default() -> Self {
        ScfParams {
            harris: false,
            min_iter: 1,
            max_iter: 60,
            max_iter_rand_wfc: 1,
            max_iter_wfc: 30,
            rho_epsilon: 1.0E-5,
            energy_epsilon: EPS6,
            eigval_epsilon: 1.0E-6,
            eigval_same_epsilon: false,
            eigen_solver: EigenSolverScheme::Pcg,
            davidson_ndim: 6,
        }
    }
}

/// Density mixing.
#[derive(Debug, Clone, PartialEq)]
pub struct MixingParams {
    /// `pulay` (default) or `broyden`.
    pub scheme: MixingScheme,
    /// Weight of the new output density: rho_next = (1-alpha)*rho_in + alpha*rho_out
    /// (the mixers apply rho_in + alpha*(rho_out - rho_in)).
    pub alpha: f64,
    /// Intended Kerker factor alpha * G2/(G2+beta). Parsed but currently not used by
    /// the Pulay or Broyden mixers.
    pub beta: f64,
    pub history_steps: usize,
    pub pulay_metric_weight: f64,
}

impl Default for MixingParams {
    fn default() -> Self {
        MixingParams {
            scheme: MixingScheme::Pulay,
            alpha: 0.7,
            beta: 0.5,
            history_steps: 4,
            pulay_metric_weight: 20.0,
        }
    }
}

/// Geometry optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct GeomOptParams {
    pub cell: bool,
    pub scheme: String,
    pub max_steps: usize,
    pub history_steps: usize,
    pub alpha: f64,
    /// eV / Ang
    pub force_tolerance: f64,
    /// kbar
    pub stress_tolerance: f64,
}

impl Default for GeomOptParams {
    fn default() -> Self {
        GeomOptParams {
            cell: false,
            scheme: "bfgs".to_string(),
            max_steps: 1,
            history_steps: 4,
            alpha: 0.8,
            force_tolerance: 0.01,
            stress_tolerance: 0.05,
        }
    }
}

/// FFT backend policy.
#[derive(Debug, Clone, PartialEq)]
pub struct FftParams {
    pub threads: usize,
    pub planner: FftPlannerScheme,
    pub wisdom_file: String,
}

impl Default for FftParams {
    fn default() -> Self {
        FftParams {
            threads: 1,
            planner: FftPlannerScheme::Estimate,
            wisdom_file: String::new(),
        }
    }
}

/// External slab field and dipole correction.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldParams {
    pub surface_dipole_correction: bool,
    /// Input in V/Ang, stored in Ha/bohr.
    pub electric_field_2d: f64,
    pub electric_field_axis: ElectricFieldAxis,
    pub electric_field_origin_frac: f64,
}

impl Default for FieldParams {
    fn default() -> Self {
        FieldParams {
            surface_dipole_correction: false,
            electric_field_2d: 0.0,
            electric_field_axis: ElectricFieldAxis::C,
            electric_field_origin_frac: 0.5,
        }
    }
}

/// Logging, checkpoints, and provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputParams {
    pub verbosity: VerbosityLevel,
    pub scf_log_format: ScfLogFormat,
    pub scf_log_file: String,
    pub save_rho: bool,
    pub save_wfc: bool,
    pub output_atomic_density: bool,
    pub provenance_manifest: String,
    pub provenance_check: bool,
}

impl Default for OutputParams {
    fn default() -> Self {
        OutputParams {
            verbosity: VerbosityLevel::Normal,
            scf_log_format: ScfLogFormat::None,
            scf_log_file: "out.scf.iter.jsonl".to_string(),
            save_rho: true,
            save_wfc: true,
            output_atomic_density: false,
            provenance_manifest: "run.provenance.json".to_string(),
            provenance_check: false,
        }
    }
}

/// Density of states.
#[derive(Debug, Clone, PartialEq)]
pub struct DosParams {
    pub scheme: String,
    /// Input in eV, stored in Hartree. Default 0.1 eV.
    pub sigma: f64,
    pub ne: usize,
}

impl Default for DosParams {
    fn default() -> Self {
        DosParams {
            scheme: "gauss".to_string(),
            sigma: 0.1 * EV_TO_HA,
            ne: 500,
        }
    }
}

/// Wannier90 export.
#[derive(Debug, Clone, PartialEq)]
pub struct Wannier90Params {
    pub export: bool,
    pub seedname: String,
    /// `None` means nband; use `resolved_num_wann()`.
    pub num_wann: Option<usize>,
    pub num_iter: usize,
}

impl Default for Wannier90Params {
    fn default() -> Self {
        Wannier90Params {
            export: false,
            seedname: "dftworks".to_string(),
            num_wann: None,
            num_iter: 200,
        }
    }
}

impl Wannier90Params {
    /// Number of Wannier functions: the explicit `wannier90_num_wann`, else `nband`.
    pub fn resolved_num_wann(&self, nband: usize) -> usize {
        self.num_wann.unwrap_or(nband)
    }
}

/// DFT+U.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardParams {
    pub enabled: bool,
    pub species: String,
    pub l: i32,
    /// Input in eV, stored in Hartree.
    pub u: f64,
    /// Input in eV, stored in Hartree.
    pub j: f64,
}

impl Default for HubbardParams {
    fn default() -> Self {
        HubbardParams {
            enabled: false,
            species: String::new(),
            l: -1,
            u: 0.0,
            j: 0.0,
        }
    }
}

impl HubbardParams {
    pub fn u_eff(&self) -> f64 {
        self.u - self.j
    }
}

/// HSE06 screened exact exchange.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridParams {
    pub alpha: f64,
    /// bohr^-1
    pub omega: f64,
}

impl Default for HybridParams {
    fn default() -> Self {
        HybridParams {
            alpha: 0.25,
            omega: 0.11,
        }
    }
}

/// DFT-D van der Waals correction.
#[derive(Debug, Clone, PartialEq)]
pub struct VdwParams {
    pub correction: bool,
    /// "zero" or "bj"
    pub damping: String,
}

impl Default for VdwParams {
    fn default() -> Self {
        VdwParams {
            correction: false,
            damping: "bj".to_string(),
        }
    }
}

// ---- in.ctrl key table -----------------------------------------------------

impl Control {
    /// Stores one `key = value` assignment from in.ctrl.
    /// Returns `Ok(false)` if the key is unknown.
    pub(crate) fn set_key(&mut self, key: &str, value: &str) -> Result<bool, String> {
        match key {
            // run
            "task" => self.run.task = value.trim().to_lowercase(),
            "restart" => self.run.restart = parse_bool(value)?,
            "symmetry" => self.run.symmetry = parse_bool(value)?,
            "random_seed" => self.run.random_seed = parse_optional_seed(value)?,
            "kpoint_schedule" => {
                self.run.kpoint_schedule = expect_one_of(
                    KPointScheduleScheme::parse(value),
                    "contiguous, cost_aware (alias: cost-aware), dynamic",
                )?
            }

            // basis
            "ecut_wfc" => self.basis.ecut_wfc = parse_ev_as_ha(value)?,
            "ecut_rho" => self.basis.ecut_rho = Some(parse_ev_as_ha(value)?),
            "nband" => self.basis.nband = parse_usize(value)?,

            // model
            "spin_scheme" => {
                self.model.spin_scheme =
                    expect_one_of(SpinScheme::parse(value), "nonspin, spin, ncl")?
            }
            "xc_scheme" => {
                self.model.xc_scheme =
                    expect_one_of(XcScheme::parse(value), "lda-pz, lsda-pz, pbe, hse06")?
            }
            "pot_scheme" => {
                self.model.pot_scheme = expect_one_of(PotScheme::parse(value), "upf, upf-fr")?
            }
            "kpts_scheme" => {
                self.model.kpts_scheme = expect_one_of(KptsScheme::parse(value), "kmesh, kline")?
            }
            "smearing_scheme" => {
                self.model.smearing_scheme = expect_one_of(
                    SmearingScheme::parse(value),
                    "fd, gs, mp1, mp2, mv (aliases: cold, marzari-vanderbilt)",
                )?
            }
            "temperature" => self.model.temperature = parse_f64(value)?,
            "occ_inversion" => self.model.occ_inversion = parse_f64(value)?,

            // scf
            "scf_harris" => self.scf.harris = parse_bool(value)?,
            "scf_min_iter" => self.scf.min_iter = parse_usize(value)?,
            "scf_max_iter" => self.scf.max_iter = parse_usize(value)?,
            "scf_max_iter_rand_wfc" => self.scf.max_iter_rand_wfc = parse_usize(value)?,
            "scf_max_iter_wfc" => self.scf.max_iter_wfc = parse_usize(value)?,
            "rho_epsilon" => self.scf.rho_epsilon = parse_f64(value)?,
            "energy_epsilon" => self.scf.energy_epsilon = parse_ev_as_ha(value)?,
            "eigval_epsilon" => self.scf.eigval_epsilon = parse_ev_as_ha(value)?,
            "eigval_same_epsilon" => self.scf.eigval_same_epsilon = parse_bool(value)?,
            "eigen_solver" => {
                self.scf.eigen_solver = expect_one_of(
                    EigenSolverScheme::parse(value),
                    "sd, psd, cg, pcg, lobpcg, arpack, davidson",
                )?
            }
            "davidson_ndim" => self.scf.davidson_ndim = parse_usize(value)?,

            // mixing
            "scf_rho_mix_scheme" => {
                self.mixing.scheme = expect_one_of(MixingScheme::parse(value), "pulay, broyden")?
            }
            "scf_rho_mix_alpha" => self.mixing.alpha = parse_f64(value)?,
            "scf_rho_mix_beta" => self.mixing.beta = parse_f64(value)?,
            "scf_rho_mix_history_steps" => self.mixing.history_steps = parse_usize(value)?,
            "scf_rho_mix_pulay_metric_weight" => {
                self.mixing.pulay_metric_weight = parse_f64(value)?
            }

            // geometry optimization
            "geom_optim_cell" => self.geom_opt.cell = parse_bool(value)?,
            "geom_optim_scheme" => self.geom_opt.scheme = parse_string(value)?,
            "geom_optim_max_steps" => self.geom_opt.max_steps = parse_usize(value)?,
            "geom_optim_history_steps" => self.geom_opt.history_steps = parse_usize(value)?,
            "geom_optim_alpha" => self.geom_opt.alpha = parse_f64(value)?,
            "geom_optim_force_tolerance" => self.geom_opt.force_tolerance = parse_f64(value)?,
            "geom_optim_stress_tolerance" => self.geom_opt.stress_tolerance = parse_f64(value)?,

            // fft
            "fft_threads" => self.fft.threads = parse_usize(value)?,
            "fft_planner" => {
                self.fft.planner =
                    expect_one_of(FftPlannerScheme::parse(value), "estimate, measure")?
            }
            "fft_wisdom_file" => self.fft.wisdom_file = parse_string(value)?,

            // external field
            "surface_dipole_correction" => {
                self.field.surface_dipole_correction = parse_bool(value)?
            }
            "electric_field_2d" => {
                self.field.electric_field_2d = parse_f64(value)? * ELECTRIC_FIELD_VANG_TO_AU
            }
            "electric_field_axis" => {
                self.field.electric_field_axis = expect_one_of(
                    ElectricFieldAxis::parse(value),
                    "a, b, c (aliases: x,y,z or 1,2,3)",
                )?
            }
            "electric_field_origin_frac" => {
                self.field.electric_field_origin_frac = parse_f64(value)?
            }

            // output
            "verbosity" => {
                self.output.verbosity = expect_one_of(
                    VerbosityLevel::parse(value),
                    "quiet, normal, verbose, debug",
                )?
            }
            "scf_log_format" => {
                self.output.scf_log_format =
                    expect_one_of(ScfLogFormat::parse(value), "none, jsonl, csv")?
            }
            "scf_log_file" => self.output.scf_log_file = parse_string(value)?,
            "save_rho" => self.output.save_rho = parse_bool(value)?,
            "save_wfc" => self.output.save_wfc = parse_bool(value)?,
            "output_atomic_density" => self.output.output_atomic_density = parse_bool(value)?,
            "provenance_manifest" => self.output.provenance_manifest = parse_string(value)?,
            "provenance_check" => self.output.provenance_check = parse_bool(value)?,

            // dos
            "dos_scheme" => self.dos.scheme = parse_string(value)?,
            "dos_sigma" => self.dos.sigma = parse_ev_as_ha(value)?,
            "dos_ne" => self.dos.ne = parse_usize(value)?,

            // wannier90
            "wannier90_export" => self.wannier90.export = parse_bool(value)?,
            "wannier90_seedname" => self.wannier90.seedname = parse_string(value)?,
            "wannier90_num_wann" => self.wannier90.num_wann = Some(parse_usize(value)?),
            "wannier90_num_iter" => self.wannier90.num_iter = parse_usize(value)?,

            // hubbard
            "hubbard_u_enabled" => self.hubbard.enabled = parse_bool(value)?,
            "hubbard_species" => self.hubbard.species = parse_string(value)?,
            "hubbard_l" => self.hubbard.l = parse_i32(value)?,
            "hubbard_u" => self.hubbard.u = parse_ev_as_ha(value)?,
            "hubbard_j" => self.hubbard.j = parse_ev_as_ha(value)?,

            // hse06
            "hse06_alpha" => self.hybrid.alpha = parse_f64(value)?,
            "hse06_omega" => self.hybrid.omega = parse_f64(value)?,

            // vdw
            "vdw_correction" => self.vdw.correction = parse_bool(value)?,
            "vdw_damping" => self.vdw.damping = parse_vdw_damping(value)?,

            _ => return Ok(false),
        }
        Ok(true)
    }
}
