use dwconsts::*;

use std::{
    error::Error,
    fmt,
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpinScheme {
    NonSpin,
    Spin,
    Ncl,
}

impl Default for SpinScheme {
    fn default() -> Self {
        SpinScheme::NonSpin
    }
}

impl SpinScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            SpinScheme::NonSpin => "nonspin",
            SpinScheme::Spin => "spin",
            SpinScheme::Ncl => "ncl",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "nonspin" => Some(SpinScheme::NonSpin),
            "spin" => Some(SpinScheme::Spin),
            "ncl" => Some(SpinScheme::Ncl),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XcScheme {
    LdaPz,
    LsdaPz,
    Pbe,
    Hse06,
}

impl Default for XcScheme {
    fn default() -> Self {
        XcScheme::LdaPz
    }
}

impl XcScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            XcScheme::LdaPz => "lda-pz",
            XcScheme::LsdaPz => "lsda-pz",
            XcScheme::Pbe => "pbe",
            XcScheme::Hse06 => "hse06",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "lda-pz" => Some(XcScheme::LdaPz),
            "lsda-pz" => Some(XcScheme::LsdaPz),
            "pbe" => Some(XcScheme::Pbe),
            "hse06" => Some(XcScheme::Hse06),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmearingScheme {
    Fd,
    Gs,
    Mp1,
    Mp2,
}

impl Default for SmearingScheme {
    fn default() -> Self {
        SmearingScheme::Mp2
    }
}

impl SmearingScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            SmearingScheme::Fd => "fd",
            SmearingScheme::Gs => "gs",
            SmearingScheme::Mp1 => "mp1",
            SmearingScheme::Mp2 => "mp2",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "fd" => Some(SmearingScheme::Fd),
            "gs" => Some(SmearingScheme::Gs),
            "mp1" => Some(SmearingScheme::Mp1),
            "mp2" => Some(SmearingScheme::Mp2),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenSolverScheme {
    Sd,
    Psd,
    Cg,
    Pcg,
    Arpack,
    Davidson,
}

impl Default for EigenSolverScheme {
    fn default() -> Self {
        EigenSolverScheme::Pcg
    }
}

impl EigenSolverScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            EigenSolverScheme::Sd => "sd",
            EigenSolverScheme::Psd => "psd",
            EigenSolverScheme::Cg => "cg",
            EigenSolverScheme::Pcg => "pcg",
            EigenSolverScheme::Arpack => "arpack",
            EigenSolverScheme::Davidson => "davidson",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "sd" => Some(EigenSolverScheme::Sd),
            "psd" => Some(EigenSolverScheme::Psd),
            "cg" => Some(EigenSolverScheme::Cg),
            "pcg" => Some(EigenSolverScheme::Pcg),
            "arpack" => Some(EigenSolverScheme::Arpack),
            "davidson" => Some(EigenSolverScheme::Davidson),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PotScheme {
    Upf,
    UpfFr,
}

impl Default for PotScheme {
    fn default() -> Self {
        PotScheme::Upf
    }
}

impl PotScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            PotScheme::Upf => "upf",
            PotScheme::UpfFr => "upf-fr",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "upf" => Some(PotScheme::Upf),
            "upf-fr" => Some(PotScheme::UpfFr),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KptsScheme {
    Kmesh,
    Kline,
}

impl Default for KptsScheme {
    fn default() -> Self {
        KptsScheme::Kmesh
    }
}

impl KptsScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            KptsScheme::Kmesh => "kmesh",
            KptsScheme::Kline => "kline",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "kmesh" => Some(KptsScheme::Kmesh),
            "kline" => Some(KptsScheme::Kline),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Control {
    // Runtime/solver settings parsed from in.ctrl with defaults.
    verbosity: String,
    ecut_wfc: f64,
    ecut_rho: f64,

    geom_optim_cell: bool,
    geom_optim_scheme: String,
    geom_optim_max_steps: usize,
    geom_optim_history_steps: usize,
    geom_optim_alpha: f64,
    geom_optim_force_tolerance: f64,
    geom_optim_stress_tolerance: f64,

    scf_harris: bool,
    scf_max_iter: usize,
    scf_min_iter: usize,
    scf_max_iter_rand_wfc: usize,
    scf_max_iter_wfc: usize,
    random_seed: Option<u64>,
    scf_rho_mix_scheme: String,
    scf_rho_mix_alpha: f64, // old*alpha + new*(1-alpha)
    scf_rho_mix_beta: f64,  // alpha * G2/(G2+beta)
    scf_rho_mix_history_steps: usize,
    scf_rho_mix_pulay_metric_weight: f64,
    rho_epsilon: f64,
    energy_epsilon: f64,

    eigval_epsilon: f64,

    eigval_same_epsilon: bool,

    nband: usize,

    spin_scheme: SpinScheme, // nonspin, spin, ncl
    task: String,            // scf, band
    restart: bool,
    save_rho: bool,
    save_wfc: bool,
    eigen_solver: EigenSolverScheme,

    davidson_ndim: usize,

    temperature: f64,
    smearing_scheme: SmearingScheme,
    xc_scheme: XcScheme,
    pot_scheme: PotScheme,
    output_atomic_density: bool,

    dos_scheme: String,
    dos_sigma: f64,
    dos_ne: usize,

    kpts_scheme: KptsScheme,
    provenance_manifest: String,
    provenance_check: bool,

    symmetry: bool,

    occ_inversion: f64,

    wannier90_export: bool,
    wannier90_seedname: String,
    wannier90_num_wann: usize,
    wannier90_num_iter: usize,

    hubbard_u_enabled: bool,
    hubbard_species: String,
    hubbard_l: i32,
    hubbard_u: f64, // eV in input, stored in Ha
    hubbard_j: f64, // eV in input, stored in Ha

    hse06_alpha: f64,
    hse06_omega: f64, // bohr^-1
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlError {
    pub line: Option<usize>,
    pub key: Option<String>,
    pub message: String,
}

impl ControlError {
    fn io(path: &str, err: std::io::Error) -> Self {
        Self {
            line: None,
            key: None,
            message: format!("failed to read '{}': {}", path, err),
        }
    }

    fn syntax(line: usize, message: impl Into<String>) -> Self {
        Self {
            line: Some(line),
            key: None,
            message: message.into(),
        }
    }

    fn unknown_key(line: usize, key: &str) -> Self {
        Self {
            line: Some(line),
            key: Some(key.to_string()),
            message: "unknown parameter".to_string(),
        }
    }

    fn invalid_value(line: usize, key: &str, value: &str, details: impl Into<String>) -> Self {
        Self {
            line: Some(line),
            key: Some(key.to_string()),
            message: format!("invalid value '{}': {}", value, details.into()),
        }
    }

    fn validation(key: &str, message: impl Into<String>) -> Self {
        Self {
            line: None,
            key: Some(key.to_string()),
            message: message.into(),
        }
    }
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.key.as_deref()) {
            (Some(line), Some(key)) => write!(f, "in.ctrl:{}:{}: {}", line, key, self.message),
            (Some(line), None) => write!(f, "in.ctrl:{}: {}", line, self.message),
            (None, Some(key)) => write!(f, "in.ctrl:{}: {}", key, self.message),
            (None, None) => write!(f, "in.ctrl: {}", self.message),
        }
    }
}

impl Error for ControlError {}

type SetterFn = fn(&mut Control, &str) -> Result<(), String>;

struct KeySpec {
    key: &'static str,
    setter: SetterFn,
}

#[inline]
fn parse_bool_value(value: &str) -> Result<bool, String> {
    value
        .trim()
        .parse::<bool>()
        .map_err(|_| "expected a boolean ('true' or 'false')".to_string())
}

#[inline]
fn parse_usize_value(value: &str) -> Result<usize, String> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| "expected an unsigned integer".to_string())
}

#[inline]
fn parse_i32_value(value: &str) -> Result<i32, String> {
    value
        .trim()
        .parse::<i32>()
        .map_err(|_| "expected an integer".to_string())
}

#[inline]
fn parse_u64_value(value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| "expected a non-negative 64-bit integer".to_string())
}

#[inline]
fn parse_f64_value(value: &str) -> Result<f64, String> {
    value
        .trim()
        .parse::<f64>()
        .map_err(|_| "expected a floating-point number".to_string())
}

macro_rules! set_string_field {
    ($fn_name:ident, $field:ident) => {
        fn $fn_name(control: &mut Control, value: &str) -> Result<(), String> {
            control.$field = value.trim().to_string();
            Ok(())
        }
    };
}

macro_rules! set_bool_field {
    ($fn_name:ident, $field:ident) => {
        fn $fn_name(control: &mut Control, value: &str) -> Result<(), String> {
            control.$field = parse_bool_value(value)?;
            Ok(())
        }
    };
}

macro_rules! set_usize_field {
    ($fn_name:ident, $field:ident) => {
        fn $fn_name(control: &mut Control, value: &str) -> Result<(), String> {
            control.$field = parse_usize_value(value)?;
            Ok(())
        }
    };
}

macro_rules! set_i32_field {
    ($fn_name:ident, $field:ident) => {
        fn $fn_name(control: &mut Control, value: &str) -> Result<(), String> {
            control.$field = parse_i32_value(value)?;
            Ok(())
        }
    };
}

macro_rules! set_f64_field {
    ($fn_name:ident, $field:ident) => {
        fn $fn_name(control: &mut Control, value: &str) -> Result<(), String> {
            control.$field = parse_f64_value(value)?;
            Ok(())
        }
    };
}

set_string_field!(set_verbosity, verbosity);
set_bool_field!(set_scf_harris, scf_harris);
set_usize_field!(set_scf_max_iter_wfc, scf_max_iter_wfc);
set_usize_field!(set_scf_max_iter_rand_wfc, scf_max_iter_rand_wfc);
set_bool_field!(set_output_atomic_density, output_atomic_density);
set_bool_field!(set_geom_optim_cell, geom_optim_cell);
set_string_field!(set_geom_optim_scheme, geom_optim_scheme);
set_usize_field!(set_geom_optim_max_steps, geom_optim_max_steps);
set_usize_field!(set_geom_optim_history_steps, geom_optim_history_steps);
set_f64_field!(set_geom_optim_alpha, geom_optim_alpha);
set_f64_field!(set_geom_optim_force_tolerance, geom_optim_force_tolerance);
set_f64_field!(set_geom_optim_stress_tolerance, geom_optim_stress_tolerance);
set_f64_field!(set_temperature, temperature);
set_usize_field!(set_scf_min_iter, scf_min_iter);
set_usize_field!(set_scf_max_iter, scf_max_iter);
set_usize_field!(set_nband, nband);
set_f64_field!(set_scf_rho_mix_alpha, scf_rho_mix_alpha);
set_f64_field!(set_scf_rho_mix_beta, scf_rho_mix_beta);
set_f64_field!(set_rho_epsilon, rho_epsilon);
set_string_field!(set_scf_rho_mix_scheme, scf_rho_mix_scheme);
set_usize_field!(set_scf_rho_mix_history_steps, scf_rho_mix_history_steps);
set_f64_field!(
    set_scf_rho_mix_pulay_metric_weight,
    scf_rho_mix_pulay_metric_weight
);
set_bool_field!(set_eigval_same_epsilon, eigval_same_epsilon);
set_bool_field!(set_restart, restart);
set_bool_field!(set_save_rho, save_rho);
set_bool_field!(set_save_wfc, save_wfc);
set_usize_field!(set_davidson_ndim, davidson_ndim);
set_string_field!(set_dos_scheme, dos_scheme);
set_usize_field!(set_dos_ne, dos_ne);
set_string_field!(set_provenance_manifest, provenance_manifest);
set_bool_field!(set_provenance_check, provenance_check);
set_f64_field!(set_occ_inversion, occ_inversion);
set_bool_field!(set_wannier90_export, wannier90_export);
set_string_field!(set_wannier90_seedname, wannier90_seedname);
set_usize_field!(set_wannier90_num_wann, wannier90_num_wann);
set_usize_field!(set_wannier90_num_iter, wannier90_num_iter);
set_bool_field!(set_hubbard_u_enabled, hubbard_u_enabled);
set_string_field!(set_hubbard_species, hubbard_species);
set_i32_field!(set_hubbard_l, hubbard_l);
set_f64_field!(set_hse06_alpha, hse06_alpha);
set_f64_field!(set_hse06_omega, hse06_omega);
set_bool_field!(set_symmetry, symmetry);

fn set_ecut_wfc(control: &mut Control, value: &str) -> Result<(), String> {
    control.ecut_wfc = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_ecut_rho(control: &mut Control, value: &str) -> Result<(), String> {
    control.ecut_rho = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_energy_epsilon(control: &mut Control, value: &str) -> Result<(), String> {
    control.energy_epsilon = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_eigval_epsilon(control: &mut Control, value: &str) -> Result<(), String> {
    control.eigval_epsilon = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_dos_sigma(control: &mut Control, value: &str) -> Result<(), String> {
    control.dos_sigma = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_hubbard_u(control: &mut Control, value: &str) -> Result<(), String> {
    control.hubbard_u = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_hubbard_j(control: &mut Control, value: &str) -> Result<(), String> {
    control.hubbard_j = parse_f64_value(value)? * EV_TO_HA;
    Ok(())
}

fn set_spin_scheme(control: &mut Control, value: &str) -> Result<(), String> {
    control.spin_scheme = SpinScheme::parse(value)
        .ok_or_else(|| "expected one of: nonspin, spin, ncl".to_string())?;
    Ok(())
}

fn set_kpts_scheme(control: &mut Control, value: &str) -> Result<(), String> {
    control.kpts_scheme = KptsScheme::parse(value)
        .ok_or_else(|| "expected one of: kmesh, kline".to_string())?;
    Ok(())
}

fn set_pot_scheme(control: &mut Control, value: &str) -> Result<(), String> {
    control.pot_scheme =
        PotScheme::parse(value).ok_or_else(|| "expected one of: upf, upf-fr".to_string())?;
    Ok(())
}

fn set_xc_scheme(control: &mut Control, value: &str) -> Result<(), String> {
    control.xc_scheme = XcScheme::parse(value)
        .ok_or_else(|| "expected one of: lda-pz, lsda-pz, pbe, hse06".to_string())?;
    Ok(())
}

fn set_smearing_scheme(control: &mut Control, value: &str) -> Result<(), String> {
    control.smearing_scheme = SmearingScheme::parse(value)
        .ok_or_else(|| "expected one of: fd, gs, mp1, mp2".to_string())?;
    Ok(())
}

fn set_eigen_solver(control: &mut Control, value: &str) -> Result<(), String> {
    control.eigen_solver = EigenSolverScheme::parse(value)
        .ok_or_else(|| "expected one of: sd, psd, cg, pcg, arpack, davidson".to_string())?;
    Ok(())
}

fn set_task(control: &mut Control, value: &str) -> Result<(), String> {
    control.task = value.trim().to_lowercase();
    Ok(())
}

fn set_random_seed(control: &mut Control, value: &str) -> Result<(), String> {
    let raw = value.trim();
    if raw.eq_ignore_ascii_case("none")
        || raw.eq_ignore_ascii_case("auto")
        || raw.eq_ignore_ascii_case("unset")
    {
        control.random_seed = None;
        return Ok(());
    }
    control.random_seed = Some(parse_u64_value(raw)?);
    Ok(())
}

const CONTROL_KEY_SPECS: &[KeySpec] = &[
    KeySpec {
        key: "verbosity",
        setter: set_verbosity,
    },
    KeySpec {
        key: "scf_harris",
        setter: set_scf_harris,
    },
    KeySpec {
        key: "scf_max_iter_wfc",
        setter: set_scf_max_iter_wfc,
    },
    KeySpec {
        key: "scf_max_iter_rand_wfc",
        setter: set_scf_max_iter_rand_wfc,
    },
    KeySpec {
        key: "random_seed",
        setter: set_random_seed,
    },
    KeySpec {
        key: "pot_scheme",
        setter: set_pot_scheme,
    },
    KeySpec {
        key: "output_atomic_density",
        setter: set_output_atomic_density,
    },
    KeySpec {
        key: "xc_scheme",
        setter: set_xc_scheme,
    },
    KeySpec {
        key: "geom_optim_cell",
        setter: set_geom_optim_cell,
    },
    KeySpec {
        key: "geom_optim_scheme",
        setter: set_geom_optim_scheme,
    },
    KeySpec {
        key: "geom_optim_max_steps",
        setter: set_geom_optim_max_steps,
    },
    KeySpec {
        key: "geom_optim_history_steps",
        setter: set_geom_optim_history_steps,
    },
    KeySpec {
        key: "geom_optim_alpha",
        setter: set_geom_optim_alpha,
    },
    KeySpec {
        key: "geom_optim_force_tolerance",
        setter: set_geom_optim_force_tolerance,
    },
    KeySpec {
        key: "geom_optim_stress_tolerance",
        setter: set_geom_optim_stress_tolerance,
    },
    KeySpec {
        key: "smearing_scheme",
        setter: set_smearing_scheme,
    },
    KeySpec {
        key: "temperature",
        setter: set_temperature,
    },
    KeySpec {
        key: "ecut_wfc",
        setter: set_ecut_wfc,
    },
    KeySpec {
        key: "ecut_rho",
        setter: set_ecut_rho,
    },
    KeySpec {
        key: "scf_min_iter",
        setter: set_scf_min_iter,
    },
    KeySpec {
        key: "scf_max_iter",
        setter: set_scf_max_iter,
    },
    KeySpec {
        key: "nband",
        setter: set_nband,
    },
    KeySpec {
        key: "scf_rho_mix_alpha",
        setter: set_scf_rho_mix_alpha,
    },
    KeySpec {
        key: "scf_rho_mix_beta",
        setter: set_scf_rho_mix_beta,
    },
    KeySpec {
        key: "energy_epsilon",
        setter: set_energy_epsilon,
    },
    KeySpec {
        key: "rho_epsilon",
        setter: set_rho_epsilon,
    },
    KeySpec {
        key: "scf_rho_mix_scheme",
        setter: set_scf_rho_mix_scheme,
    },
    KeySpec {
        key: "scf_rho_mix_history_steps",
        setter: set_scf_rho_mix_history_steps,
    },
    KeySpec {
        key: "scf_rho_mix_pulay_metric_weight",
        setter: set_scf_rho_mix_pulay_metric_weight,
    },
    KeySpec {
        key: "eigval_epsilon",
        setter: set_eigval_epsilon,
    },
    KeySpec {
        key: "eigval_same_epsilon",
        setter: set_eigval_same_epsilon,
    },
    KeySpec {
        key: "spin_scheme",
        setter: set_spin_scheme,
    },
    KeySpec {
        key: "task",
        setter: set_task,
    },
    KeySpec {
        key: "restart",
        setter: set_restart,
    },
    KeySpec {
        key: "save_rho",
        setter: set_save_rho,
    },
    KeySpec {
        key: "save_wfc",
        setter: set_save_wfc,
    },
    KeySpec {
        key: "eigen_solver",
        setter: set_eigen_solver,
    },
    KeySpec {
        key: "davidson_ndim",
        setter: set_davidson_ndim,
    },
    KeySpec {
        key: "dos_scheme",
        setter: set_dos_scheme,
    },
    KeySpec {
        key: "dos_sigma",
        setter: set_dos_sigma,
    },
    KeySpec {
        key: "dos_ne",
        setter: set_dos_ne,
    },
    KeySpec {
        key: "kpts_scheme",
        setter: set_kpts_scheme,
    },
    KeySpec {
        key: "provenance_manifest",
        setter: set_provenance_manifest,
    },
    KeySpec {
        key: "provenance_check",
        setter: set_provenance_check,
    },
    KeySpec {
        key: "occ_inversion",
        setter: set_occ_inversion,
    },
    KeySpec {
        key: "wannier90_export",
        setter: set_wannier90_export,
    },
    KeySpec {
        key: "wannier90_seedname",
        setter: set_wannier90_seedname,
    },
    KeySpec {
        key: "wannier90_num_wann",
        setter: set_wannier90_num_wann,
    },
    KeySpec {
        key: "wannier90_num_iter",
        setter: set_wannier90_num_iter,
    },
    KeySpec {
        key: "hubbard_u_enabled",
        setter: set_hubbard_u_enabled,
    },
    KeySpec {
        key: "hubbard_species",
        setter: set_hubbard_species,
    },
    KeySpec {
        key: "hubbard_l",
        setter: set_hubbard_l,
    },
    KeySpec {
        key: "hubbard_u",
        setter: set_hubbard_u,
    },
    KeySpec {
        key: "hubbard_j",
        setter: set_hubbard_j,
    },
    KeySpec {
        key: "hse06_alpha",
        setter: set_hse06_alpha,
    },
    KeySpec {
        key: "hse06_omega",
        setter: set_hse06_omega,
    },
    KeySpec {
        key: "symmetry",
        setter: set_symmetry,
    },
];

fn find_key_spec(key: &str) -> Option<&'static KeySpec> {
    CONTROL_KEY_SPECS.iter().find(|spec| spec.key == key)
}

fn canonicalize_control_key(key: &str) -> &str {
    match key {
        // Deprecated aliases retained for backward compatibility.
        "ecut" => "ecut_wfc",
        "mixing_scheme" => "scf_rho_mix_scheme",
        _ => key,
    }
}

impl Control {
    pub fn new() -> Control {
        Control::default()
    }

    pub fn get_symmetry(&self) -> bool {
        self.symmetry
    }

    pub fn is_scf_harris(&self) -> bool {
        self.scf_harris
    }

    pub fn get_output_atomic_density(&self) -> bool {
        self.output_atomic_density
    }

    //pub fn is_scf(&self) -> bool {
    //    "scf" == self.get_task()
    //}

    //pub fn is_band(&self) -> bool {
    //    "band" == self.get_task()
    //}

    pub fn is_spin(&self) -> bool {
        matches!(self.spin_scheme, SpinScheme::Spin)
    }

    pub fn is_noncollinear(&self) -> bool {
        matches!(self.spin_scheme, SpinScheme::Ncl)
    }

    pub fn is_collinear(&self) -> bool {
        matches!(self.spin_scheme, SpinScheme::NonSpin | SpinScheme::Spin)
    }

    //pub fn is_nscf(&self) -> bool {
    //    "nscf" == self.get_task()
    //}

    pub fn get_dos_scheme(&self) -> &str {
        &self.dos_scheme
    }

    pub fn get_dos_sigma(&self) -> f64 {
        self.dos_sigma
    }

    pub fn get_dos_ne(&self) -> usize {
        self.dos_ne
    }

    pub fn get_kpts_scheme(&self) -> &str {
        self.kpts_scheme.as_str()
    }

    pub fn get_kpts_scheme_enum(&self) -> KptsScheme {
        self.kpts_scheme
    }

    pub fn get_provenance_manifest(&self) -> &str {
        &self.provenance_manifest
    }

    pub fn get_provenance_check(&self) -> bool {
        self.provenance_check
    }

    pub fn get_scf_min_iter(&self) -> usize {
        self.scf_min_iter
    }

    pub fn get_scf_max_iter_wfc(&self) -> usize {
        self.scf_max_iter_wfc
    }

    pub fn get_scf_max_iter_rand_wfc(&self) -> usize {
        self.scf_max_iter_rand_wfc
    }

    pub fn get_random_seed(&self) -> Option<u64> {
        self.random_seed
    }

    pub fn get_pot_scheme(&self) -> &str {
        self.pot_scheme.as_str()
    }

    pub fn get_pot_scheme_enum(&self) -> PotScheme {
        self.pot_scheme
    }

    pub fn get_xc_scheme(&self) -> &str {
        self.xc_scheme.as_str()
    }

    pub fn get_xc_scheme_enum(&self) -> XcScheme {
        self.xc_scheme
    }

    pub fn get_hse06_alpha(&self) -> f64 {
        self.hse06_alpha
    }

    pub fn get_hse06_omega(&self) -> f64 {
        self.hse06_omega
    }

    pub fn get_geom_optim_cell(&self) -> bool {
        self.geom_optim_cell
    }

    pub fn get_geom_optim_scheme(&self) -> &str {
        &self.geom_optim_scheme
    }

    pub fn get_geom_optim_max_steps(&self) -> usize {
        self.geom_optim_max_steps
    }

    pub fn get_geom_optim_history_steps(&self) -> usize {
        self.geom_optim_history_steps
    }

    pub fn get_geom_optim_alpha(&self) -> f64 {
        self.geom_optim_alpha
    }

    pub fn get_geom_optim_force_tolerance(&self) -> f64 {
        self.geom_optim_force_tolerance
    }

    pub fn get_geom_optim_stress_tolerance(&self) -> f64 {
        self.geom_optim_stress_tolerance
    }

    pub fn get_smearing_scheme(&self) -> &str {
        self.smearing_scheme.as_str()
    }

    pub fn get_smearing_scheme_enum(&self) -> SmearingScheme {
        self.smearing_scheme
    }

    pub fn get_temperature(&self) -> f64 {
        self.temperature
    }

    pub fn get_scf_max_iter(&self) -> usize {
        self.scf_max_iter
    }

    pub fn get_ecut(&self) -> f64 {
        self.ecut_wfc
    }

    pub fn get_ecutrho(&self) -> f64 {
        self.ecut_rho
    }

    pub fn get_nband(&self) -> usize {
        self.nband
    }

    pub fn get_scf_rho_mix_scheme(&self) -> &str {
        &self.scf_rho_mix_scheme
    }

    pub fn get_scf_rho_mix_history_steps(&self) -> usize {
        self.scf_rho_mix_history_steps
    }

    pub fn get_scf_rho_mix_pulay_metric_weight(&self) -> f64 {
        self.scf_rho_mix_pulay_metric_weight
    }

    pub fn get_scf_rho_mix_beta(&self) -> f64 {
        self.scf_rho_mix_beta
    }

    pub fn get_scf_rho_mix_alpha(&self) -> f64 {
        self.scf_rho_mix_alpha
    }

    pub fn get_energy_epsilon(&self) -> f64 {
        self.energy_epsilon
    }

    pub fn get_rho_epsilon(&self) -> f64 {
        self.rho_epsilon
    }

    pub fn get_eigval_epsilon(&self) -> f64 {
        self.eigval_epsilon
    }

    pub fn get_spin_scheme(&self) -> &str {
        self.spin_scheme.as_str()
    }

    pub fn get_spin_scheme_enum(&self) -> SpinScheme {
        self.spin_scheme
    }

    pub fn get_task(&self) -> &str {
        &self.task
    }

    pub fn get_eigen_solver(&self) -> &str {
        self.eigen_solver.as_str()
    }

    pub fn get_eigen_solver_enum(&self) -> EigenSolverScheme {
        self.eigen_solver
    }

    pub fn get_davidson_ndim(&self) -> usize {
        self.davidson_ndim
    }

    // the setting of task will override some parameters read from in.ctrl

    pub fn set_task(&mut self, task: &str) {
        self.task = task.to_string();

        match task {
            "scf" => {
                //self.restart = false;
            }
            "band" => {
                self.restart = true;
                self.save_rho = false;
                self.save_wfc = false;
                //self.scf_max_iter = 1;
            }

            _ => {}
        }
    }

    pub fn get_verbosity(&self) -> &str {
        &self.verbosity
    }

    pub fn get_restart(&self) -> bool {
        self.restart
    }

    pub fn get_save_rho(&self) -> bool {
        self.save_rho
    }

    pub fn get_save_wfc(&self) -> bool {
        self.save_wfc
    }

    pub fn get_occ_inversion(&self) -> f64 {
        self.occ_inversion
    }

    pub fn get_wannier90_export(&self) -> bool {
        self.wannier90_export
    }

    pub fn get_wannier90_seedname(&self) -> &str {
        &self.wannier90_seedname
    }

    pub fn get_wannier90_num_wann(&self) -> usize {
        self.wannier90_num_wann
    }

    pub fn get_wannier90_num_iter(&self) -> usize {
        self.wannier90_num_iter
    }

    pub fn get_hubbard_u_enabled(&self) -> bool {
        self.hubbard_u_enabled
    }

    pub fn get_hubbard_species(&self) -> &str {
        &self.hubbard_species
    }

    pub fn get_hubbard_l(&self) -> i32 {
        self.hubbard_l
    }

    pub fn get_hubbard_u(&self) -> f64 {
        self.hubbard_u
    }

    pub fn get_hubbard_j(&self) -> f64 {
        self.hubbard_j
    }

    pub fn get_hubbard_u_eff(&self) -> f64 {
        self.hubbard_u - self.hubbard_j
    }

    pub fn read_file(&mut self, inpfile: &str) {
        if let Err(err) = self.try_read_file(inpfile) {
            panic!("{}", err);
        }
    }

    pub fn from_file(inpfile: &str) -> Result<Control, ControlError> {
        let mut control = Control::new();
        control.try_read_file(inpfile)?;
        Ok(control)
    }

    pub fn try_read_file(&mut self, inpfile: &str) -> Result<(), ControlError> {
        let lines = Self::read_file_data_to_vec(inpfile)?;
        self.try_read_lines(lines.as_slice())
    }

    fn reset_defaults(&mut self) {
        self.task = "scf".to_string();
        self.spin_scheme = SpinScheme::NonSpin;

        self.geom_optim_cell = false;
        self.geom_optim_scheme = "bfgs".to_string();
        self.geom_optim_max_steps = 1;
        self.geom_optim_history_steps = 4;
        self.geom_optim_alpha = 0.8;
        self.geom_optim_force_tolerance = 0.01; // eV / A
        self.geom_optim_stress_tolerance = 0.05; // kbar

        self.save_rho = true;
        self.save_wfc = true;

        self.ecut_wfc = 400.0 * EV_TO_HA; // eV in in.ctrl, stored as Ha
        self.ecut_rho = 4.0 * self.ecut_wfc;

        self.eigval_same_epsilon = false;
        self.eigval_epsilon = 1.0E-6;
        self.rho_epsilon = 1.0E-5;
        self.energy_epsilon = EPS6;

        self.eigen_solver = EigenSolverScheme::Pcg;
        self.davidson_ndim = 6;

        self.scf_rho_mix_scheme = "simple".to_string();
        self.scf_rho_mix_alpha = 0.7;
        self.scf_rho_mix_beta = 0.5;
        self.scf_rho_mix_history_steps = 4;
        self.scf_rho_mix_pulay_metric_weight = 20.0;

        self.scf_harris = false;
        self.scf_min_iter = 1;
        self.scf_max_iter = 60;
        self.scf_max_iter_rand_wfc = 1;
        self.scf_max_iter_wfc = 30;
        self.random_seed = None;

        self.dos_scheme = "gauss".to_string();
        self.dos_ne = 500;
        self.dos_sigma = 0.1;

        self.kpts_scheme = KptsScheme::Kmesh;
        self.smearing_scheme = SmearingScheme::Mp2;
        self.xc_scheme = XcScheme::LdaPz;
        self.pot_scheme = PotScheme::Upf;
        self.provenance_manifest = "run.provenance.json".to_string();
        self.provenance_check = false;
        self.verbosity = "high".to_string();
        self.occ_inversion = 0.0;

        self.wannier90_export = false;
        self.wannier90_seedname = "dftworks".to_string();
        self.wannier90_num_wann = 0;
        self.wannier90_num_iter = 200;

        self.hubbard_u_enabled = false;
        self.hubbard_species = String::new();
        self.hubbard_l = -1;
        self.hubbard_u = 0.0;
        self.hubbard_j = 0.0;

        self.hse06_alpha = 0.25;
        self.hse06_omega = 0.11;
    }

    fn parse_assignment_line(line: &str, line_no: usize) -> Result<Option<(String, String)>, ControlError> {
        let line = line.split('#').next().unwrap_or(line);
        let line = line.split('!').next().unwrap_or(line);
        let trimmed = line.trim();

        if trimmed.is_empty() {
            return Ok(None);
        }

        let Some((key, value)) = trimmed.split_once('=') else {
            return Err(ControlError::syntax(
                line_no,
                "expected 'key = value' assignment",
            ));
        };

        let key = key.trim().to_lowercase();
        if key.is_empty() {
            return Err(ControlError::syntax(line_no, "parameter name must not be empty"));
        }

        Ok(Some((key, value.trim().to_string())))
    }

    fn try_read_lines(&mut self, lines: &[String]) -> Result<(), ControlError> {
        self.reset_defaults();

        let mut ecut_rho_set = false;
        let mut wannier90_num_wann_set = false;

        for (idx, raw_line) in lines.iter().enumerate() {
            let line_no = idx + 1;
            let Some((raw_key, raw_value)) = Self::parse_assignment_line(raw_line, line_no)? else {
                continue;
            };

            let key = canonicalize_control_key(raw_key.as_str());
            let spec = find_key_spec(key).ok_or_else(|| ControlError::unknown_key(line_no, &raw_key))?;

            (spec.setter)(self, &raw_value)
                .map_err(|details| ControlError::invalid_value(line_no, key, &raw_value, details))?;

            if key == "ecut_rho" {
                ecut_rho_set = true;
            }
            if key == "wannier90_num_wann" {
                wannier90_num_wann_set = true;
            }
        }

        if !ecut_rho_set {
            self.ecut_rho = 4.0 * self.ecut_wfc;
        }
        if !wannier90_num_wann_set {
            self.wannier90_num_wann = self.nband;
        }

        self.validate_semantic_ranges()?;
        self.validate_feature_compatibility()?;

        Ok(())
    }

    fn validate_semantic_ranges(&self) -> Result<(), ControlError> {
        if self.scf_min_iter > self.scf_max_iter {
            return Err(ControlError::validation(
                "scf_min_iter/scf_max_iter",
                "scf_min_iter must be <= scf_max_iter",
            ));
        }

        if self.wannier90_export {
            if self.wannier90_seedname.trim().is_empty() {
                return Err(ControlError::validation(
                    "wannier90_seedname",
                    "must not be empty when wannier90_export=true",
                ));
            }
            if self.wannier90_num_wann == 0 {
                return Err(ControlError::validation(
                    "wannier90_num_wann",
                    "must be > 0 when wannier90_export=true",
                ));
            }
            if self.nband == 0 {
                return Err(ControlError::validation(
                    "nband",
                    "must be > 0 when wannier90_export=true",
                ));
            }
            if self.wannier90_num_wann > self.nband {
                return Err(ControlError::validation(
                    "wannier90_num_wann",
                    format!("{} > nband ({})", self.wannier90_num_wann, self.nband),
                ));
            }
        }

        if self.provenance_manifest.trim().is_empty() {
            return Err(ControlError::validation(
                "provenance_manifest",
                "must not be empty",
            ));
        }

        if !matches!(self.eigen_solver, EigenSolverScheme::Pcg) {
            return Err(ControlError::validation(
                "eigen_solver",
                format!(
                    "eigen_solver='{}' is parsed but not implemented yet; use 'pcg'",
                    self.eigen_solver.as_str()
                ),
            ));
        }

        Ok(())
    }

    fn validate_feature_compatibility(&self) -> Result<(), ControlError> {
        if self.restart && self.is_noncollinear() {
            return Err(ControlError::validation(
                "restart/spin_scheme",
                "restart currently supports only nonspin/spin",
            ));
        }

        if self.hubbard_u_enabled {
            if self.hubbard_species.trim().is_empty() {
                return Err(ControlError::validation(
                    "hubbard_species",
                    "must not be empty when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard_l < 0 {
                return Err(ControlError::validation(
                    "hubbard_l",
                    "must be >= 0 when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard_u < 0.0 {
                return Err(ControlError::validation(
                    "hubbard_u",
                    "must be >= 0 eV when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard_j < 0.0 {
                return Err(ControlError::validation(
                    "hubbard_j",
                    "must be >= 0 eV when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard_u < self.hubbard_j {
                return Err(ControlError::validation(
                    "hubbard_u/hubbard_j",
                    "requires hubbard_u >= hubbard_j",
                ));
            }
            if self.is_noncollinear() {
                return Err(ControlError::validation(
                    "spin_scheme",
                    "hubbard_u_enabled=true currently supports only nonspin/spin",
                ));
            }
        }

        if matches!(self.xc_scheme, XcScheme::Hse06) {
            if self.hse06_alpha < 0.0 || self.hse06_alpha > 1.0 {
                return Err(ControlError::validation(
                    "hse06_alpha",
                    "must be within [0, 1]",
                ));
            }
            if self.hse06_omega <= 0.0 {
                return Err(ControlError::validation(
                    "hse06_omega",
                    "must be > 0 (bohr^-1)",
                ));
            }
            if self.is_noncollinear() {
                return Err(ControlError::validation(
                    "spin_scheme",
                    "xc_scheme='hse06' currently supports only nonspin/spin",
                ));
            }
        }

        Ok(())
    }

    fn read_file_data_to_vec(path: &str) -> Result<Vec<String>, ControlError> {
        let file = File::open(path).map_err(|e| ControlError::io(path, e))?;
        let mut lines = Vec::new();
        for (line_idx, line_res) in BufReader::new(file).lines().enumerate() {
            let line = line_res.map_err(|e| ControlError {
                line: Some(line_idx + 1),
                key: None,
                message: format!("failed to read line: {}", e),
            })?;
            lines.push(line);
        }
        Ok(lines)
    }

    pub fn display(&self) {
        const OUT_WIDTH1: usize = 28;
        const OUT_WIDTH2: usize = 18;

        println!("   {:-^88}", " control parameters ");
        println!();

        println!(
            "   {:<width1$} = {:>width2$}",
            "restart",
            self.get_restart(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        let random_seed_text = self
            .get_random_seed()
            .map(|seed| seed.to_string())
            .unwrap_or_else(|| "auto".to_string());
        println!(
            "   {:<width1$} = {:>width2$}",
            "random_seed",
            random_seed_text,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "provenance_check",
            self.get_provenance_check(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {}",
            "provenance_manifest",
            self.get_provenance_manifest(),
            width1 = OUT_WIDTH1
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "spin_scheme",
            self.get_spin_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "pot_scheme",
            self.get_pot_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} ",
            "eigen_solver",
            self.get_eigen_solver(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.3E} eV",
            "energy_conv_eps",
            self.get_energy_epsilon() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$.3E} eV",
            "eig_conv_eps",
            self.get_eigval_epsilon() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_harris",
            self.scf_harris,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_max_iter",
            self.get_scf_max_iter(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_min_iter",
            self.get_scf_min_iter(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "smearing_scheme",
            self.get_smearing_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$} K",
            "temperature",
            self.get_temperature(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.3} eV",
            "ecut",
            self.get_ecut() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$.3} eV",
            "ecutrho",
            self.get_ecutrho() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "nband",
            self.get_nband(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "wannier90_export",
            self.get_wannier90_export(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "wannier90_seedname",
            self.get_wannier90_seedname(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "wannier90_num_wann",
            self.get_wannier90_num_wann(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "wannier90_num_iter",
            self.get_wannier90_num_iter(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "hubbard_u_enabled",
            self.get_hubbard_u_enabled(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "hubbard_species",
            self.get_hubbard_species(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "hubbard_l",
            self.get_hubbard_l(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.6} eV",
            "hubbard_u",
            self.get_hubbard_u() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.6} eV",
            "hubbard_j",
            self.get_hubbard_j() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$.6} eV",
            "hubbard_u_eff",
            self.get_hubbard_u_eff() * HA_TO_EV,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "hse06_alpha",
            self.get_hse06_alpha(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} bohr^-1",
            "hse06_omega",
            self.get_hse06_omega(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_rho_mix_scheme",
            self.get_scf_rho_mix_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_rho_mix_alpha",
            self.get_scf_rho_mix_alpha(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_rho_mix_beta",
            self.get_scf_rho_mix_beta(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_cell",
            self.get_geom_optim_cell(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_scheme",
            self.get_geom_optim_scheme(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_history_steps",
            self.get_geom_optim_history_steps(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_max_steps",
            self.get_geom_optim_max_steps(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$}",
            "geom_optim_alpha",
            self.get_geom_optim_alpha(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} eV/A",
            "geom_optim_force_tolerance",
            self.get_geom_optim_force_tolerance(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!(
            "   {:<width1$} = {:>width2$} kbar",
            "geom_optim_stress_tolerance",
            self.get_geom_optim_stress_tolerance(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );

        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::{Control, ControlError, KptsScheme, PotScheme, SmearingScheme, SpinScheme, XcScheme};

    fn parse_control(lines: &[&str]) -> Result<Control, ControlError> {
        let mut control = Control::new();
        let lines_owned: Vec<String> = lines.iter().map(|line| (*line).to_string()).collect();
        control.try_read_lines(lines_owned.as_slice())?;
        Ok(control)
    }

    #[test]
    fn test_spin_scheme_parse_accepts_supported_values() {
        assert_eq!(SpinScheme::parse("nonspin"), Some(SpinScheme::NonSpin));
        assert_eq!(SpinScheme::parse("spin"), Some(SpinScheme::Spin));
        assert_eq!(SpinScheme::parse("ncl"), Some(SpinScheme::Ncl));
        assert_eq!(SpinScheme::parse("  Spin "), Some(SpinScheme::Spin));
    }

    #[test]
    fn test_spin_scheme_parse_rejects_unsupported_values() {
        assert_eq!(SpinScheme::parse(""), None);
        assert_eq!(SpinScheme::parse("collinear"), None);
        assert_eq!(SpinScheme::parse("spin-orbit"), None);
        assert_eq!(SpinScheme::parse("foo"), None);
    }

    #[test]
    fn test_parser_reports_unknown_key_with_line_and_key() {
        let err = parse_control(&["unknown_knob = 1"]).unwrap_err();
        assert_eq!(err.line, Some(1));
        assert_eq!(err.key.as_deref(), Some("unknown_knob"));
        assert_eq!(err.message, "unknown parameter");
    }

    #[test]
    fn test_parser_reports_typed_value_error() {
        let err = parse_control(&["scf_max_iter = nope"]).unwrap_err();
        assert_eq!(err.line, Some(1));
        assert_eq!(err.key.as_deref(), Some("scf_max_iter"));
        assert!(err.message.contains("expected an unsigned integer"));
    }

    #[test]
    fn test_parser_accepts_deprecated_aliases() {
        let control =
            parse_control(&["ecut = 320.0", "mixing_scheme = pulay", "nband = 12"]).unwrap();

        let ecut_ev = control.get_ecut() * dwconsts::HA_TO_EV;
        assert!((ecut_ev - 320.0).abs() < 1.0e-12);
        assert!((control.get_ecutrho() - 4.0 * control.get_ecut()).abs() < 1.0e-12);
        assert_eq!(control.get_scf_rho_mix_scheme(), "pulay");
    }

    #[test]
    fn test_validation_rejects_invalid_semantic_range() {
        let err = parse_control(&["scf_min_iter = 5", "scf_max_iter = 4"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("scf_min_iter/scf_max_iter"));
        assert!(err.message.contains("must be <="));
    }

    #[test]
    fn test_validation_rejects_hse06_with_ncl() {
        let err = parse_control(&["xc_scheme = hse06", "spin_scheme = ncl"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("spin_scheme"));
        assert!(err.message.contains("hse06"));
    }

    #[test]
    fn test_validation_rejects_hubbard_incompatible_values() {
        let err = parse_control(&[
            "hubbard_u_enabled = true",
            "hubbard_species = Fe",
            "hubbard_l = 2",
            "hubbard_u = 1.0",
            "hubbard_j = 2.0",
        ])
        .unwrap_err();
        assert_eq!(err.key.as_deref(), Some("hubbard_u/hubbard_j"));
        assert!(err.message.contains("requires hubbard_u >= hubbard_j"));
    }

    #[test]
    fn test_validation_rejects_restart_with_ncl() {
        let err = parse_control(&["restart = true", "spin_scheme = ncl"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("restart/spin_scheme"));
        assert!(err.message.contains("supports only nonspin/spin"));
    }

    #[test]
    fn test_parser_accepts_random_seed_and_provenance_fields() {
        let control = parse_control(&[
            "random_seed = 42",
            "provenance_manifest = provenance/run.json",
            "provenance_check = true",
        ])
        .expect("control parse should succeed");

        assert_eq!(control.get_random_seed(), Some(42));
        assert_eq!(control.get_provenance_manifest(), "provenance/run.json");
        assert!(control.get_provenance_check());
    }

    #[test]
    fn test_validation_rejects_empty_provenance_manifest() {
        let err = parse_control(&["provenance_manifest ="]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("provenance_manifest"));
        assert!(err.message.contains("must not be empty"));
    }

    #[test]
    fn test_parser_exposes_typed_runtime_modes() {
        let control = parse_control(&[
            "kpts_scheme = kline",
            "pot_scheme = upf-fr",
            "smearing_scheme = fd",
            "xc_scheme = pbe",
        ])
        .expect("control parse should succeed");

        assert_eq!(control.get_kpts_scheme_enum(), KptsScheme::Kline);
        assert_eq!(control.get_pot_scheme_enum(), PotScheme::UpfFr);
        assert_eq!(control.get_smearing_scheme_enum(), SmearingScheme::Fd);
        assert_eq!(control.get_xc_scheme_enum(), XcScheme::Pbe);
    }

    #[test]
    fn test_validation_rejects_unimplemented_eigen_solver_modes() {
        let err = parse_control(&["eigen_solver = davidson"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("eigen_solver"));
        assert!(err.message.contains("not implemented yet"));
    }
}
