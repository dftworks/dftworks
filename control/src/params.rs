// Parameter groups for in.ctrl.
//
// Each group is declared with `param_group!`, one line per input key:
//
//     "in_ctrl_key" => field: Type = default, parser;
//
// The macro generates the struct (public fields), its `Default`, the list of
// keys it owns, and the key dispatch used by the parser. Adding an input key
// is therefore a single line here; cross-key validation still lives in
// `Control::validate_*`.
//
// Parsers are `fn(&str) -> Result<T, String>` (plain functions or
// non-capturing closures). Values are stored in internal units (Hartree,
// bohr); the field docs note the input unit when it differs.

use crate::*;
use dwconsts::{EPS6, EV_TO_HA};

// 1 a.u. electric field = Eh/(e*a0) = 5.14220674763e11 V/m = 51.4220674763 V/Ang.
pub(crate) const ELECTRIC_FIELD_AU_TO_VANG: f64 = 51.4220674763;
const ELECTRIC_FIELD_VANG_TO_AU: f64 = 1.0 / ELECTRIC_FIELD_AU_TO_VANG;

macro_rules! param_group {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $(
                $(#[$fmeta:meta])*
                $key:literal => $field:ident : $ty:ty = $default:expr, $parse:expr;
            )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            $(
                $(#[$fmeta])*
                pub $field: $ty,
            )*
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $( $field: $default, )*
                }
            }
        }

        impl $name {
            /// in.ctrl keys owned by this group.
            pub const KEYS: &'static [&'static str] = &[$($key),*];

            /// Returns `None` when `key` does not belong to this group.
            pub(crate) fn set_key(&mut self, key: &str, value: &str) -> Option<Result<(), String>> {
                match key {
                    $(
                        $key => Some(($parse)(value).map(|parsed: $ty| self.$field = parsed)),
                    )*
                    _ => None,
                }
            }
        }
    };
}

// ---- value parsers -------------------------------------------------------

pub(crate) fn parse_bool(value: &str) -> Result<bool, String> {
    value
        .trim()
        .parse::<bool>()
        .map_err(|_| "expected a boolean ('true' or 'false')".to_string())
}

pub(crate) fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| "expected an unsigned integer".to_string())
}

pub(crate) fn parse_i32(value: &str) -> Result<i32, String> {
    value
        .trim()
        .parse::<i32>()
        .map_err(|_| "expected an integer".to_string())
}

pub(crate) fn parse_f64(value: &str) -> Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| "expected a floating-point number".to_string())
}

/// Energy given in eV, stored in Hartree.
pub(crate) fn parse_ev_as_ha(value: &str) -> Result<f64, String> {
    Ok(parse_f64(value)? * EV_TO_HA)
}

pub(crate) fn parse_string(value: &str) -> Result<String, String> {
    Ok(value.trim().to_string())
}

pub(crate) fn parse_lowercase(value: &str) -> Result<String, String> {
    Ok(value.trim().to_lowercase())
}

/// `none`/`auto`/`unset` mean "no fixed seed".
pub(crate) fn parse_optional_seed(value: &str) -> Result<Option<u64>, String> {
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

/// Turns an enum's `parse` result into a parser error listing the accepted values.
fn expect_one_of<T>(parsed: Option<T>, accepted: &str) -> Result<T, String> {
    parsed.ok_or_else(|| format!("expected one of: {}", accepted))
}

// ---- groups --------------------------------------------------------------

param_group! {
    /// Run mode and global switches.
    pub struct RunParams {
        "task" => task: String = "scf".to_string(), parse_lowercase;
        "restart" => restart: bool = false, parse_bool;
        "symmetry" => symmetry: bool = false, parse_bool;
        "random_seed" => random_seed: Option<u64> = None, parse_optional_seed;
        "kpoint_schedule" => kpoint_schedule: KPointScheduleScheme = KPointScheduleScheme::CostAware,
            |v| expect_one_of(KPointScheduleScheme::parse(v), "contiguous, cost_aware (alias: cost-aware), dynamic");
    }
}

param_group! {
    /// Plane-wave basis and band count.
    pub struct BasisParams {
        /// Input in eV, stored in Hartree.
        "ecut_wfc" => ecut_wfc: f64 = 400.0 * EV_TO_HA, parse_ev_as_ha;
        /// Input in eV, stored in Hartree. `None` means 4 * ecut_wfc; use `resolved_ecut_rho()`.
        "ecut_rho" => ecut_rho: Option<f64> = None, |v| parse_ev_as_ha(v).map(Some);
        "nband" => nband: usize = 0, parse_usize;
    }
}

impl BasisParams {
    /// Density cutoff in Hartree: the explicit `ecut_rho`, else 4 * ecut_wfc.
    pub fn resolved_ecut_rho(&self) -> f64 {
        self.ecut_rho.unwrap_or(4.0 * self.ecut_wfc)
    }
}

param_group! {
    /// Physical model: spin treatment, functional, pseudopotentials, k-points, occupations.
    pub struct ModelParams {
        "spin_scheme" => spin_scheme: SpinScheme = SpinScheme::NonSpin,
            |v| expect_one_of(SpinScheme::parse(v), "nonspin, spin, ncl");
        "xc_scheme" => xc_scheme: XcScheme = XcScheme::LdaPz,
            |v| expect_one_of(XcScheme::parse(v), "lda-pz, lsda-pz, pbe, hse06");
        "pot_scheme" => pot_scheme: PotScheme = PotScheme::Upf,
            |v| expect_one_of(PotScheme::parse(v), "upf, upf-fr");
        "kpts_scheme" => kpts_scheme: KptsScheme = KptsScheme::Kmesh,
            |v| expect_one_of(KptsScheme::parse(v), "kmesh, kline");
        "smearing_scheme" => smearing_scheme: SmearingScheme = SmearingScheme::Mp2,
            |v| expect_one_of(SmearingScheme::parse(v), "fd, gs, mp1, mp2, mv (aliases: cold, marzari-vanderbilt)");
        "temperature" => temperature: f64 = 0.0, parse_f64;
        "occ_inversion" => occ_inversion: f64 = 0.0, parse_f64;
    }
}

param_group! {
    /// SCF loop, convergence thresholds, and eigensolver.
    pub struct ScfParams {
        "scf_harris" => harris: bool = false, parse_bool;
        "scf_min_iter" => min_iter: usize = 1, parse_usize;
        "scf_max_iter" => max_iter: usize = 60, parse_usize;
        "scf_max_iter_rand_wfc" => max_iter_rand_wfc: usize = 1, parse_usize;
        "scf_max_iter_wfc" => max_iter_wfc: usize = 30, parse_usize;
        "rho_epsilon" => rho_epsilon: f64 = 1.0E-5, parse_f64;
        /// Input in eV, stored in Hartree.
        "energy_epsilon" => energy_epsilon: f64 = EPS6, parse_ev_as_ha;
        /// Input in eV, stored in Hartree.
        "eigval_epsilon" => eigval_epsilon: f64 = 1.0E-6, parse_ev_as_ha;
        "eigval_same_epsilon" => eigval_same_epsilon: bool = false, parse_bool;
        "eigen_solver" => eigen_solver: EigenSolverScheme = EigenSolverScheme::Pcg,
            |v| expect_one_of(EigenSolverScheme::parse(v), "sd, psd, cg, pcg, lobpcg, arpack, davidson");
        "davidson_ndim" => davidson_ndim: usize = 6, parse_usize;
    }
}

param_group! {
    /// Density mixing.
    pub struct MixingParams {
        /// `pulay` (default) or `broyden`.
        "scf_rho_mix_scheme" => scheme: MixingScheme = MixingScheme::Pulay,
            |v| MixingScheme::parse(v)
                .ok_or_else(|| format!("expected one of: {}", MixingScheme::accepted_values()));
        /// Weight of the new output density: rho_next = (1-alpha)*rho_in + alpha*rho_out
        /// (the mixers apply rho_in + alpha*(rho_out - rho_in)).
        "scf_rho_mix_alpha" => alpha: f64 = 0.7, parse_f64;
        /// Intended Kerker factor alpha * G2/(G2+beta). Parsed but currently not used by
        /// the Pulay or Broyden mixers.
        "scf_rho_mix_beta" => beta: f64 = 0.5, parse_f64;
        "scf_rho_mix_history_steps" => history_steps: usize = 4, parse_usize;
        "scf_rho_mix_pulay_metric_weight" => pulay_metric_weight: f64 = 20.0, parse_f64;
    }
}

param_group! {
    /// Geometry optimization.
    pub struct GeomOptParams {
        "geom_optim_cell" => cell: bool = false, parse_bool;
        "geom_optim_scheme" => scheme: String = "bfgs".to_string(), parse_string;
        "geom_optim_max_steps" => max_steps: usize = 1, parse_usize;
        "geom_optim_history_steps" => history_steps: usize = 4, parse_usize;
        "geom_optim_alpha" => alpha: f64 = 0.8, parse_f64;
        /// eV / Ang
        "geom_optim_force_tolerance" => force_tolerance: f64 = 0.01, parse_f64;
        /// kbar
        "geom_optim_stress_tolerance" => stress_tolerance: f64 = 0.05, parse_f64;
    }
}

param_group! {
    /// FFT backend policy.
    pub struct FftParams {
        "fft_threads" => threads: usize = 1, parse_usize;
        "fft_planner" => planner: FftPlannerScheme = FftPlannerScheme::Estimate,
            |v| expect_one_of(FftPlannerScheme::parse(v), "estimate, measure");
        "fft_wisdom_file" => wisdom_file: String = String::new(), parse_string;
    }
}

param_group! {
    /// External slab field and dipole correction.
    pub struct FieldParams {
        "surface_dipole_correction" => surface_dipole_correction: bool = false, parse_bool;
        /// Input in V/Ang, stored in Ha/bohr.
        "electric_field_2d" => electric_field_2d: f64 = 0.0,
            |v| Ok(parse_f64(v)? * ELECTRIC_FIELD_VANG_TO_AU);
        "electric_field_axis" => electric_field_axis: ElectricFieldAxis = ElectricFieldAxis::C,
            |v| expect_one_of(ElectricFieldAxis::parse(v), "a, b, c (aliases: x,y,z or 1,2,3)");
        "electric_field_origin_frac" => electric_field_origin_frac: f64 = 0.5, parse_f64;
    }
}

param_group! {
    /// Logging, checkpoints, and provenance.
    pub struct OutputParams {
        "verbosity" => verbosity: VerbosityLevel = VerbosityLevel::Normal,
            |v| expect_one_of(VerbosityLevel::parse(v), "quiet, normal, verbose, debug");
        "scf_log_format" => scf_log_format: ScfLogFormat = ScfLogFormat::None,
            |v| expect_one_of(ScfLogFormat::parse(v), "none, jsonl, csv");
        "scf_log_file" => scf_log_file: String = "out.scf.iter.jsonl".to_string(), parse_string;
        "save_rho" => save_rho: bool = true, parse_bool;
        "save_wfc" => save_wfc: bool = true, parse_bool;
        "output_atomic_density" => output_atomic_density: bool = false, parse_bool;
        "provenance_manifest" => provenance_manifest: String = "run.provenance.json".to_string(), parse_string;
        "provenance_check" => provenance_check: bool = false, parse_bool;
    }
}

param_group! {
    /// Density of states.
    pub struct DosParams {
        "dos_scheme" => scheme: String = "gauss".to_string(), parse_string;
        /// Input in eV, stored in Hartree. Default 0.1 eV.
        "dos_sigma" => sigma: f64 = 0.1 * EV_TO_HA, parse_ev_as_ha;
        "dos_ne" => ne: usize = 500, parse_usize;
    }
}

param_group! {
    /// Wannier90 export.
    pub struct Wannier90Params {
        "wannier90_export" => export: bool = false, parse_bool;
        "wannier90_seedname" => seedname: String = "dftworks".to_string(), parse_string;
        /// `None` means nband; use `resolved_num_wann()`.
        "wannier90_num_wann" => num_wann: Option<usize> = None, |v| parse_usize(v).map(Some);
        "wannier90_num_iter" => num_iter: usize = 200, parse_usize;
    }
}

impl Wannier90Params {
    /// Number of Wannier functions: the explicit `wannier90_num_wann`, else `nband`.
    pub fn resolved_num_wann(&self, nband: usize) -> usize {
        self.num_wann.unwrap_or(nband)
    }
}

param_group! {
    /// DFT+U.
    pub struct HubbardParams {
        "hubbard_u_enabled" => enabled: bool = false, parse_bool;
        "hubbard_species" => species: String = String::new(), parse_string;
        "hubbard_l" => l: i32 = -1, parse_i32;
        /// Input in eV, stored in Hartree.
        "hubbard_u" => u: f64 = 0.0, parse_ev_as_ha;
        /// Input in eV, stored in Hartree.
        "hubbard_j" => j: f64 = 0.0, parse_ev_as_ha;
    }
}

impl HubbardParams {
    pub fn u_eff(&self) -> f64 {
        self.u - self.j
    }
}

param_group! {
    /// HSE06 screened exact exchange.
    pub struct HybridParams {
        "hse06_alpha" => alpha: f64 = 0.25, parse_f64;
        /// bohr^-1
        "hse06_omega" => omega: f64 = 0.11, parse_f64;
    }
}

param_group! {
    /// DFT-D van der Waals correction.
    pub struct VdwParams {
        "vdw_correction" => correction: bool = false, parse_bool;
        "vdw_damping" => damping: String = "bj".to_string(),
            |v: &str| {
                let damping = v.trim().to_lowercase();
                if damping == "zero" || damping == "bj" {
                    Ok(damping)
                } else {
                    Err("expected one of: zero, bj".to_string())
                }
            };
    }
}
