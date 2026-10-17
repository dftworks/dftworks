use dwconsts::*;

use std::{
    error::Error,
    fmt,
    fs::File,
    io::{BufRead, BufReader},
};

mod params;
pub use params::*;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerbosityLevel {
    Quiet,
    Normal,
    Verbose,
    Debug,
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        VerbosityLevel::Normal
    }
}

impl VerbosityLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            VerbosityLevel::Quiet => "quiet",
            VerbosityLevel::Normal => "normal",
            VerbosityLevel::Verbose => "verbose",
            VerbosityLevel::Debug => "debug",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "quiet" => Some(VerbosityLevel::Quiet),
            "normal" | "medium" => Some(VerbosityLevel::Normal),
            "verbose" | "high" => Some(VerbosityLevel::Verbose),
            "debug" => Some(VerbosityLevel::Debug),
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
    Mv,
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
            SmearingScheme::Mv => "mv",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "fd" => Some(SmearingScheme::Fd),
            "gs" => Some(SmearingScheme::Gs),
            "mp1" => Some(SmearingScheme::Mp1),
            "mp2" => Some(SmearingScheme::Mp2),
            "mv" | "cold" | "marzari-vanderbilt" | "marzari_vanderbilt" => Some(SmearingScheme::Mv),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Only implemented solvers are listed; add a variant together with its
// implementation in the `eigensolver` crate.
pub enum EigenSolverScheme {
    Pcg,
    Lobpcg,
}

impl Default for EigenSolverScheme {
    fn default() -> Self {
        EigenSolverScheme::Pcg
    }
}

impl EigenSolverScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            EigenSolverScheme::Pcg => "pcg",
            EigenSolverScheme::Lobpcg => "lobpcg",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "pcg" => Some(EigenSolverScheme::Pcg),
            "lobpcg" => Some(EigenSolverScheme::Lobpcg),
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

/// Run mode. Both modes currently run the same SCF path; band-structure runs
/// are driven by `kpts_scheme = kline` with an `in.kline` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Scf,
    Band,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Task::Scf => "scf",
            Task::Band => "band",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "scf" => Some(Task::Scf),
            "band" => Some(Task::Band),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomOptScheme {
    Bfgs,
    Diis,
}

impl GeomOptScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            GeomOptScheme::Bfgs => "bfgs",
            GeomOptScheme::Diis => "diis",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "bfgs" => Some(GeomOptScheme::Bfgs),
            "diis" => Some(GeomOptScheme::Diis),
            _ => None,
        }
    }
}

/// DOS broadening shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DosScheme {
    Gauss,
}

impl DosScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            DosScheme::Gauss => "gauss",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "gauss" => Some(DosScheme::Gauss),
            _ => None,
        }
    }
}

/// DFT-D damping function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VdwDampingScheme {
    Zero,
    Bj,
}

impl VdwDampingScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            VdwDampingScheme::Zero => "zero",
            VdwDampingScheme::Bj => "bj",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "zero" => Some(VdwDampingScheme::Zero),
            "bj" => Some(VdwDampingScheme::Bj),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixingScheme {
    Pulay,
    Broyden,
}

impl Default for MixingScheme {
    fn default() -> Self {
        MixingScheme::Pulay
    }
}

impl MixingScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            MixingScheme::Pulay => "pulay",
            MixingScheme::Broyden => "broyden",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "pulay" => Some(MixingScheme::Pulay),
            "broyden" => Some(MixingScheme::Broyden),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftPlannerScheme {
    Estimate,
    Measure,
}

impl Default for FftPlannerScheme {
    fn default() -> Self {
        FftPlannerScheme::Estimate
    }
}

impl FftPlannerScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            FftPlannerScheme::Estimate => "estimate",
            FftPlannerScheme::Measure => "measure",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "estimate" => Some(FftPlannerScheme::Estimate),
            "measure" => Some(FftPlannerScheme::Measure),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScfLogFormat {
    None,
    Jsonl,
    Csv,
}

impl Default for ScfLogFormat {
    fn default() -> Self {
        ScfLogFormat::None
    }
}

impl ScfLogFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ScfLogFormat::None => "none",
            ScfLogFormat::Jsonl => "jsonl",
            ScfLogFormat::Csv => "csv",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "none" => Some(ScfLogFormat::None),
            "jsonl" => Some(ScfLogFormat::Jsonl),
            "csv" => Some(ScfLogFormat::Csv),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElectricFieldAxis {
    A,
    B,
    C,
}

impl Default for ElectricFieldAxis {
    fn default() -> Self {
        ElectricFieldAxis::C
    }
}

impl ElectricFieldAxis {
    pub fn as_str(self) -> &'static str {
        match self {
            ElectricFieldAxis::A => "a",
            ElectricFieldAxis::B => "b",
            ElectricFieldAxis::C => "c",
        }
    }

    pub fn axis_index(self) -> usize {
        match self {
            ElectricFieldAxis::A => 0,
            ElectricFieldAxis::B => 1,
            ElectricFieldAxis::C => 2,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "a" | "x" | "1" => Some(ElectricFieldAxis::A),
            "b" | "y" | "2" => Some(ElectricFieldAxis::B),
            "c" | "z" | "3" => Some(ElectricFieldAxis::C),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KPointScheduleScheme {
    Contiguous,
    CostAware,
    Dynamic,
}

impl Default for KPointScheduleScheme {
    fn default() -> Self {
        KPointScheduleScheme::CostAware
    }
}

impl KPointScheduleScheme {
    pub fn as_str(self) -> &'static str {
        match self {
            KPointScheduleScheme::Contiguous => "contiguous",
            KPointScheduleScheme::CostAware => "cost_aware",
            KPointScheduleScheme::Dynamic => "dynamic",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "contiguous" => Some(KPointScheduleScheme::Contiguous),
            "cost_aware" | "cost-aware" | "costaware" => Some(KPointScheduleScheme::CostAware),
            "dynamic" => Some(KPointScheduleScheme::Dynamic),
            _ => None,
        }
    }
}

/// Runtime/solver settings parsed from in.ctrl, with defaults.
///
/// Settings are grouped by concern (see `params.rs`). New code should take the
/// group it needs (e.g. `control.mixing()`) rather than the whole `Control`.
/// The flat `get_*` accessors below are kept for existing callers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Control {
    run: RunParams,
    basis: BasisParams,
    model: ModelParams,
    scf: ScfParams,
    mixing: MixingParams,
    geom_opt: GeomOptParams,
    fft: FftParams,
    field: FieldParams,
    output: OutputParams,
    dos: DosParams,
    wannier90: Wannier90Params,
    hubbard: HubbardParams,
    hybrid: HybridParams,
    vdw: VdwParams,
}

// Spin schemes that parse and can run. Every other option only lists
// implemented values, so anything that parses is supported; `ncl` parses but
// has no runtime yet. Restart supports the same spin schemes.
const RUNTIME_SPIN_SCHEMES: &[SpinScheme] = &[SpinScheme::NonSpin, SpinScheme::Spin];

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

    // ---- grouped accessors ----

    pub fn run(&self) -> &RunParams {
        &self.run
    }

    pub fn basis(&self) -> &BasisParams {
        &self.basis
    }

    pub fn model(&self) -> &ModelParams {
        &self.model
    }

    pub fn scf(&self) -> &ScfParams {
        &self.scf
    }

    pub fn mixing(&self) -> &MixingParams {
        &self.mixing
    }

    pub fn geom_opt(&self) -> &GeomOptParams {
        &self.geom_opt
    }

    pub fn fft(&self) -> &FftParams {
        &self.fft
    }

    pub fn field(&self) -> &FieldParams {
        &self.field
    }

    pub fn output(&self) -> &OutputParams {
        &self.output
    }

    pub fn dos(&self) -> &DosParams {
        &self.dos
    }

    pub fn wannier90(&self) -> &Wannier90Params {
        &self.wannier90
    }

    pub fn hubbard(&self) -> &HubbardParams {
        &self.hubbard
    }

    pub fn hybrid(&self) -> &HybridParams {
        &self.hybrid
    }

    pub fn vdw(&self) -> &VdwParams {
        &self.vdw
    }

    // ---- flat accessors (kept for existing callers) ----

    pub fn get_symmetry(&self) -> bool {
        self.run.symmetry
    }

    pub fn is_scf_harris(&self) -> bool {
        self.scf.harris
    }

    pub fn get_output_atomic_density(&self) -> bool {
        self.output.output_atomic_density
    }

    pub fn is_spin(&self) -> bool {
        matches!(self.model.spin_scheme, SpinScheme::Spin)
    }

    pub fn is_noncollinear(&self) -> bool {
        matches!(self.model.spin_scheme, SpinScheme::Ncl)
    }

    pub fn is_collinear(&self) -> bool {
        matches!(
            self.model.spin_scheme,
            SpinScheme::NonSpin | SpinScheme::Spin
        )
    }

    pub fn get_dos_scheme(&self) -> DosScheme {
        self.dos.scheme
    }

    pub fn get_dos_sigma(&self) -> f64 {
        self.dos.sigma
    }

    pub fn get_dos_ne(&self) -> usize {
        self.dos.ne
    }

    pub fn get_kpts_scheme(&self) -> &str {
        self.model.kpts_scheme.as_str()
    }

    pub fn get_kpts_scheme_enum(&self) -> KptsScheme {
        self.model.kpts_scheme
    }

    pub fn get_provenance_manifest(&self) -> &str {
        &self.output.provenance_manifest
    }

    pub fn get_provenance_check(&self) -> bool {
        self.output.provenance_check
    }

    pub fn get_scf_min_iter(&self) -> usize {
        self.scf.min_iter
    }

    pub fn get_scf_max_iter_wfc(&self) -> usize {
        self.scf.max_iter_wfc
    }

    pub fn get_scf_max_iter_rand_wfc(&self) -> usize {
        self.scf.max_iter_rand_wfc
    }

    pub fn get_random_seed(&self) -> Option<u64> {
        self.run.random_seed
    }

    pub fn get_pot_scheme(&self) -> &str {
        self.model.pot_scheme.as_str()
    }

    pub fn get_pot_scheme_enum(&self) -> PotScheme {
        self.model.pot_scheme
    }

    pub fn get_xc_scheme(&self) -> &str {
        self.model.xc_scheme.as_str()
    }

    pub fn get_xc_scheme_enum(&self) -> XcScheme {
        self.model.xc_scheme
    }

    pub fn get_hse06_alpha(&self) -> f64 {
        self.hybrid.alpha
    }

    pub fn get_hse06_omega(&self) -> f64 {
        self.hybrid.omega
    }

    pub fn get_geom_optim_cell(&self) -> bool {
        self.geom_opt.cell
    }

    pub fn get_geom_optim_scheme(&self) -> GeomOptScheme {
        self.geom_opt.scheme
    }

    pub fn get_geom_optim_max_steps(&self) -> usize {
        self.geom_opt.max_steps
    }

    pub fn get_geom_optim_history_steps(&self) -> usize {
        self.geom_opt.history_steps
    }

    pub fn get_geom_optim_alpha(&self) -> f64 {
        self.geom_opt.alpha
    }

    pub fn get_geom_optim_force_tolerance(&self) -> f64 {
        self.geom_opt.force_tolerance
    }

    pub fn get_geom_optim_stress_tolerance(&self) -> f64 {
        self.geom_opt.stress_tolerance
    }

    pub fn get_smearing_scheme(&self) -> &str {
        self.model.smearing_scheme.as_str()
    }

    pub fn get_smearing_scheme_enum(&self) -> SmearingScheme {
        self.model.smearing_scheme
    }

    pub fn get_temperature(&self) -> f64 {
        self.model.temperature
    }

    pub fn get_scf_max_iter(&self) -> usize {
        self.scf.max_iter
    }

    pub fn get_ecut(&self) -> f64 {
        self.basis.ecut_wfc
    }

    pub fn get_ecutrho(&self) -> f64 {
        self.basis.resolved_ecut_rho()
    }

    pub fn get_nband(&self) -> usize {
        self.basis.nband
    }

    pub fn get_scf_rho_mix_scheme(&self) -> MixingScheme {
        self.mixing.scheme
    }

    pub fn get_scf_rho_mix_history_steps(&self) -> usize {
        self.mixing.history_steps
    }

    pub fn get_scf_rho_mix_pulay_metric_weight(&self) -> f64 {
        self.mixing.pulay_metric_weight
    }

    pub fn get_scf_rho_mix_beta(&self) -> f64 {
        self.mixing.beta
    }

    pub fn get_scf_rho_mix_alpha(&self) -> f64 {
        self.mixing.alpha
    }

    pub fn get_energy_epsilon(&self) -> f64 {
        self.scf.energy_epsilon
    }

    pub fn get_rho_epsilon(&self) -> f64 {
        self.scf.rho_epsilon
    }

    pub fn get_eigval_epsilon(&self) -> f64 {
        self.scf.eigval_epsilon
    }

    pub fn get_spin_scheme(&self) -> &str {
        self.model.spin_scheme.as_str()
    }

    pub fn get_spin_scheme_enum(&self) -> SpinScheme {
        self.model.spin_scheme
    }

    pub fn get_task(&self) -> Task {
        self.run.task
    }

    pub fn get_surface_dipole_correction(&self) -> bool {
        self.field.surface_dipole_correction
    }

    pub fn get_electric_field_2d(&self) -> f64 {
        self.field.electric_field_2d
    }

    pub fn get_electric_field_2d_vang(&self) -> f64 {
        self.field.electric_field_2d * ELECTRIC_FIELD_AU_TO_VANG
    }

    pub fn get_electric_field_axis(&self) -> &str {
        self.field.electric_field_axis.as_str()
    }

    pub fn get_electric_field_axis_enum(&self) -> ElectricFieldAxis {
        self.field.electric_field_axis
    }

    pub fn get_electric_field_origin_frac(&self) -> f64 {
        self.field.electric_field_origin_frac
    }

    pub fn get_kpoint_schedule(&self) -> &str {
        self.run.kpoint_schedule.as_str()
    }

    pub fn get_kpoint_schedule_enum(&self) -> KPointScheduleScheme {
        self.run.kpoint_schedule
    }

    pub fn get_scf_log_format(&self) -> &str {
        self.output.scf_log_format.as_str()
    }

    pub fn get_scf_log_format_enum(&self) -> ScfLogFormat {
        self.output.scf_log_format
    }

    pub fn get_scf_log_file(&self) -> &str {
        &self.output.scf_log_file
    }

    pub fn get_fft_threads(&self) -> usize {
        self.fft.threads
    }

    pub fn get_fft_planner(&self) -> &str {
        self.fft.planner.as_str()
    }

    pub fn get_fft_planner_enum(&self) -> FftPlannerScheme {
        self.fft.planner
    }

    pub fn get_fft_wisdom_file(&self) -> &str {
        &self.fft.wisdom_file
    }

    pub fn get_eigen_solver(&self) -> &str {
        self.scf.eigen_solver.as_str()
    }

    pub fn get_eigen_solver_enum(&self) -> EigenSolverScheme {
        self.scf.eigen_solver
    }

    pub fn get_davidson_ndim(&self) -> usize {
        self.scf.davidson_ndim
    }

    // the setting of task will override some parameters read from in.ctrl

    pub fn set_task(&mut self, task: Task) {
        self.run.task = task;

        match task {
            Task::Scf => {
                //self.run.restart = false;
            }
            Task::Band => {
                self.run.restart = true;
                self.output.save_rho = false;
                self.output.save_wfc = false;
                //self.scf.max_iter = 1;
            }
        }
    }

    pub fn get_verbosity(&self) -> &str {
        self.output.verbosity.as_str()
    }

    pub fn get_verbosity_enum(&self) -> VerbosityLevel {
        self.output.verbosity
    }

    pub fn verbosity_at_least(&self, level: VerbosityLevel) -> bool {
        self.output.verbosity >= level
    }

    pub fn get_restart(&self) -> bool {
        self.run.restart
    }

    pub fn get_save_rho(&self) -> bool {
        self.output.save_rho
    }

    pub fn get_save_wfc(&self) -> bool {
        self.output.save_wfc
    }

    pub fn get_occ_inversion(&self) -> f64 {
        self.model.occ_inversion
    }

    pub fn get_wannier90_export(&self) -> bool {
        self.wannier90.export
    }

    pub fn get_wannier90_seedname(&self) -> &str {
        &self.wannier90.seedname
    }

    pub fn get_wannier90_num_wann(&self) -> usize {
        self.wannier90.resolved_num_wann(self.basis.nband)
    }

    pub fn get_wannier90_num_iter(&self) -> usize {
        self.wannier90.num_iter
    }

    pub fn get_hubbard_u_enabled(&self) -> bool {
        self.hubbard.enabled
    }

    pub fn get_hubbard_species(&self) -> &str {
        &self.hubbard.species
    }

    pub fn get_hubbard_l(&self) -> i32 {
        self.hubbard.l
    }

    pub fn get_hubbard_u(&self) -> f64 {
        self.hubbard.u
    }

    pub fn get_hubbard_j(&self) -> f64 {
        self.hubbard.j
    }

    pub fn get_hubbard_u_eff(&self) -> f64 {
        self.hubbard.u_eff()
    }

    pub fn get_vdw_correction(&self) -> bool {
        self.vdw.correction
    }

    pub fn get_vdw_damping(&self) -> VdwDampingScheme {
        self.vdw.damping
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
        *self = Control::default();
    }

    fn parse_assignment_line(
        line: &str,
        line_no: usize,
    ) -> Result<Option<(String, String)>, ControlError> {
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
            return Err(ControlError::syntax(
                line_no,
                "parameter name must not be empty",
            ));
        }

        Ok(Some((key, value.trim().to_string())))
    }

    fn try_read_lines(&mut self, lines: &[String]) -> Result<(), ControlError> {
        self.reset_defaults();

        for (idx, raw_line) in lines.iter().enumerate() {
            let line_no = idx + 1;
            let Some((raw_key, raw_value)) = Self::parse_assignment_line(raw_line, line_no)? else {
                continue;
            };

            let key = canonicalize_control_key(raw_key.as_str());
            match self.set_key(key, &raw_value) {
                Ok(true) => {}
                Ok(false) => return Err(ControlError::unknown_key(line_no, &raw_key)),
                Err(details) => {
                    return Err(ControlError::invalid_value(
                        line_no, key, &raw_value, details,
                    ))
                }
            }
        }

        self.validate_semantic_ranges()?;
        self.validate_feature_compatibility()?;

        Ok(())
    }

    fn validate_semantic_ranges(&self) -> Result<(), ControlError> {
        let num_wann = self.get_wannier90_num_wann();

        if self.scf.min_iter > self.scf.max_iter {
            return Err(ControlError::validation(
                "scf_min_iter/scf_max_iter",
                "scf_min_iter must be <= scf_max_iter",
            ));
        }

        if self.wannier90.export {
            if self.wannier90.seedname.trim().is_empty() {
                return Err(ControlError::validation(
                    "wannier90_seedname",
                    "must not be empty when wannier90_export=true",
                ));
            }
            if num_wann == 0 {
                return Err(ControlError::validation(
                    "wannier90_num_wann",
                    "must be > 0 when wannier90_export=true",
                ));
            }
            if self.basis.nband == 0 {
                return Err(ControlError::validation(
                    "nband",
                    "must be > 0 when wannier90_export=true",
                ));
            }
            if num_wann > self.basis.nband {
                return Err(ControlError::validation(
                    "wannier90_num_wann",
                    format!("{} > nband ({})", num_wann, self.basis.nband),
                ));
            }
        }

        if self.output.provenance_manifest.trim().is_empty() {
            return Err(ControlError::validation(
                "provenance_manifest",
                "must not be empty",
            ));
        }

        if self.fft.threads == 0 {
            return Err(ControlError::validation("fft_threads", "must be >= 1"));
        }

        if !matches!(self.output.scf_log_format, ScfLogFormat::None)
            && self.output.scf_log_file.trim().is_empty()
        {
            return Err(ControlError::validation(
                "scf_log_file",
                "must not be empty when scf_log_format is enabled",
            ));
        }

        if self.field.electric_field_origin_frac < 0.0
            || self.field.electric_field_origin_frac >= 1.0
        {
            return Err(ControlError::validation(
                "electric_field_origin_frac",
                "must be within [0, 1)",
            ));
        }

        if (self.field.electric_field_2d.abs() > EPS16 || self.field.surface_dipole_correction)
            && self.run.symmetry
        {
            return Err(ControlError::validation(
                "symmetry/electric_field_2d/surface_dipole_correction",
                "external slab fields currently require symmetry=false",
            ));
        }

        Ok(())
    }

    pub fn validate_capability_matrix(&self) -> Result<(), ControlError> {
        let spin_scheme = self.model.spin_scheme;

        if !RUNTIME_SPIN_SCHEMES.contains(&spin_scheme) {
            let supported = RUNTIME_SPIN_SCHEMES
                .iter()
                .map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ControlError::validation(
                "spin_scheme",
                format!(
                    "spin_scheme='{}' is parsed but runtime support is not enabled yet; supported: {}",
                    spin_scheme.as_str(),
                    supported
                ),
            ));
        }

        Ok(())
    }

    fn validate_feature_compatibility(&self) -> Result<(), ControlError> {
        self.validate_capability_matrix()?;

        if self.hubbard.enabled {
            if self.hubbard.species.trim().is_empty() {
                return Err(ControlError::validation(
                    "hubbard_species",
                    "must not be empty when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard.l < 0 {
                return Err(ControlError::validation(
                    "hubbard_l",
                    "must be >= 0 when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard.u < 0.0 {
                return Err(ControlError::validation(
                    "hubbard_u",
                    "must be >= 0 eV when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard.j < 0.0 {
                return Err(ControlError::validation(
                    "hubbard_j",
                    "must be >= 0 eV when hubbard_u_enabled=true",
                ));
            }
            if self.hubbard.u < self.hubbard.j {
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

        if matches!(self.model.xc_scheme, XcScheme::Hse06) {
            if self.hybrid.alpha < 0.0 || self.hybrid.alpha > 1.0 {
                return Err(ControlError::validation(
                    "hse06_alpha",
                    "must be within [0, 1]",
                ));
            }
            if self.hybrid.omega <= 0.0 {
                return Err(ControlError::validation(
                    "hse06_omega",
                    "must be > 0 (bohr^-1)",
                ));
            }
            if self.is_noncollinear() {
                return Err(ControlError::validation(
                    "spin_scheme",
                    "xc_scheme='hse06' currently supports only nonspin/spin (capability matrix)",
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
        const OUT_WIDTH1: usize = 32;
        const OUT_WIDTH2: usize = 0;

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
            "verbosity",
            self.get_verbosity(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "scf_log_format",
            self.get_scf_log_format(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {}",
            "scf_log_file",
            self.get_scf_log_file(),
            width1 = OUT_WIDTH1
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "fft_threads",
            self.get_fft_threads(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "fft_planner",
            self.get_fft_planner(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        let wisdom_file = if self.get_fft_wisdom_file().trim().is_empty() {
            "(none)"
        } else {
            self.get_fft_wisdom_file()
        };
        println!(
            "   {:<width1$} = {:>width2$}",
            "fft_wisdom_file",
            wisdom_file,
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$.6}",
            "electric_field_2d",
            self.get_electric_field_2d_vang(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "electric_field_axis",
            self.get_electric_field_axis(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$.4}",
            "electric_field_origin",
            self.get_electric_field_origin_frac(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "surface_dipole_correction",
            self.get_surface_dipole_correction(),
            width1 = OUT_WIDTH1,
            width2 = OUT_WIDTH2
        );
        println!(
            "   {:<width1$} = {:>width2$}",
            "kpoint_schedule",
            self.get_kpoint_schedule(),
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
            self.scf.harris,
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
            self.get_scf_rho_mix_scheme().as_str(),
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
            self.get_geom_optim_scheme().as_str(),
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
    use super::{
        Control, ControlError, DosScheme, EigenSolverScheme, ElectricFieldAxis, FftPlannerScheme,
        GeomOptScheme, KPointScheduleScheme, KptsScheme, MixingScheme, PotScheme, ScfLogFormat,
        SmearingScheme, SpinScheme, Task, VdwDampingScheme, VerbosityLevel, XcScheme,
    };

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
        assert_eq!(control.get_scf_rho_mix_scheme(), MixingScheme::Pulay);
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
        assert!(err.message.contains("runtime support is not enabled yet"));
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
        assert_eq!(err.key.as_deref(), Some("spin_scheme"));
        assert!(err.message.contains("runtime support is not enabled yet"));
    }

    #[test]
    fn test_validation_rejects_unsupported_task_mode() {
        let err = parse_control(&["task = relax"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("task"));
        assert!(err.message.contains("expected one of: scf, band"));
    }

    #[test]
    fn test_capability_matrix_accepts_supported_core_combinations() {
        let tasks = ["scf", "band"];
        let spins = ["nonspin", "spin"];
        let xcs = ["lda-pz", "lsda-pz", "pbe", "hse06"];

        for task in tasks.iter() {
            for spin in spins.iter() {
                for xc in xcs.iter() {
                    let lines = vec![
                        format!("task = {}", task),
                        format!("spin_scheme = {}", spin),
                        format!("xc_scheme = {}", xc),
                        "eigen_solver = pcg".to_string(),
                    ];
                    let line_refs = lines.iter().map(|s| s.as_str()).collect::<Vec<_>>();
                    parse_control(line_refs.as_slice())
                        .unwrap_or_else(|e| panic!("unexpected capability rejection: {}", e));
                }
            }
        }
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
            "fft_planner = measure",
        ])
        .expect("control parse should succeed");

        assert_eq!(control.get_kpts_scheme_enum(), KptsScheme::Kline);
        assert_eq!(control.get_pot_scheme_enum(), PotScheme::UpfFr);
        assert_eq!(control.get_smearing_scheme_enum(), SmearingScheme::Fd);
        assert_eq!(control.get_xc_scheme_enum(), XcScheme::Pbe);
        assert_eq!(control.get_fft_planner_enum(), FftPlannerScheme::Measure);
    }

    #[test]
    fn test_parser_accepts_mv_smearing_aliases() {
        let control = parse_control(&["smearing_scheme = cold"]).unwrap();
        assert_eq!(control.get_smearing_scheme_enum(), SmearingScheme::Mv);

        let control = parse_control(&["smearing_scheme = marzari-vanderbilt"]).unwrap();
        assert_eq!(control.get_smearing_scheme_enum(), SmearingScheme::Mv);
    }

    #[test]
    fn test_parser_rejects_unimplemented_eigen_solvers() {
        for name in ["sd", "psd", "cg", "arpack", "davidson"].iter() {
            let line = format!("eigen_solver = {}", name);
            let err = parse_control(&[line.as_str()]).unwrap_err();
            assert_eq!(err.key.as_deref(), Some("eigen_solver"));
            assert!(err.message.contains("expected one of: pcg, lobpcg"));
        }
    }

    #[test]
    fn test_parser_accepts_fft_runtime_policy() {
        let control = parse_control(&[
            "fft_threads = 4",
            "fft_planner = estimate",
            "fft_wisdom_file = cache/fftw.wisdom",
        ])
        .expect("control parse should succeed");

        assert_eq!(control.get_fft_threads(), 4);
        assert_eq!(control.get_fft_planner_enum(), FftPlannerScheme::Estimate);
        assert_eq!(control.get_fft_wisdom_file(), "cache/fftw.wisdom");
    }

    #[test]
    fn test_validation_rejects_zero_fft_threads() {
        let err = parse_control(&["fft_threads = 0"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("fft_threads"));
        assert!(err.message.contains(">= 1"));
    }

    #[test]
    fn test_parser_accepts_surface_field_controls() {
        let control = parse_control(&[
            "electric_field_2d = 0.25",
            "electric_field_axis = b",
            "electric_field_origin_frac = 0.2",
            "surface_dipole_correction = true",
            "symmetry = false",
        ])
        .expect("control parse should succeed");

        assert!((control.get_electric_field_2d_vang() - 0.25).abs() < 1.0e-12);
        assert_eq!(control.get_electric_field_axis_enum(), ElectricFieldAxis::B);
        assert!((control.get_electric_field_origin_frac() - 0.2).abs() < 1.0e-12);
        assert!(control.get_surface_dipole_correction());
    }

    #[test]
    fn test_validation_rejects_invalid_surface_field_origin() {
        let err = parse_control(&["electric_field_origin_frac = 1.2"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("electric_field_origin_frac"));
        assert!(err.message.contains("[0, 1)"));
    }

    #[test]
    fn test_validation_rejects_surface_field_with_symmetry() {
        let err = parse_control(&["electric_field_2d = 0.1", "symmetry = true"]).unwrap_err();
        assert_eq!(
            err.key.as_deref(),
            Some("symmetry/electric_field_2d/surface_dipole_correction")
        );
        assert!(err.message.contains("symmetry=false"));
    }

    #[test]
    fn test_parser_accepts_kpoint_schedule_modes() {
        let control = parse_control(&["kpoint_schedule = dynamic"]).unwrap();
        assert_eq!(
            control.get_kpoint_schedule_enum(),
            KPointScheduleScheme::Dynamic
        );
    }

    #[test]
    fn test_parser_rejects_unknown_kpoint_schedule_modes() {
        let err = parse_control(&["kpoint_schedule = adaptive"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("kpoint_schedule"));
        assert!(err.message.contains("contiguous"));
    }

    #[test]
    fn test_parser_accepts_typed_verbosity_levels() {
        let control = parse_control(&["verbosity = debug"]).unwrap();
        assert_eq!(control.get_verbosity_enum(), VerbosityLevel::Debug);
    }

    #[test]
    fn test_parser_accepts_legacy_verbosity_aliases() {
        let control = parse_control(&["verbosity = high"]).unwrap();
        assert_eq!(control.get_verbosity_enum(), VerbosityLevel::Verbose);
    }

    #[test]
    fn test_parser_rejects_unknown_verbosity_levels() {
        let err = parse_control(&["verbosity = noisy"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("verbosity"));
        assert!(err.message.contains("quiet"));
    }

    #[test]
    fn test_parser_accepts_scf_log_controls() {
        let control = parse_control(&["scf_log_format = csv", "scf_log_file = iter.csv"]).unwrap();
        assert_eq!(control.get_scf_log_format_enum(), ScfLogFormat::Csv);
        assert_eq!(control.get_scf_log_file(), "iter.csv");
    }

    #[test]
    fn test_validation_rejects_empty_scf_log_file_when_enabled() {
        let err = parse_control(&["scf_log_format = jsonl", "scf_log_file ="]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("scf_log_file"));
        assert!(err.message.contains("must not be empty"));
    }

    #[test]
    fn test_parser_accepts_vdw_correction_parameters() {
        let control = parse_control(&["vdw_correction = true", "vdw_damping = zero"]).unwrap();
        assert!(control.get_vdw_correction());
        assert_eq!(control.get_vdw_damping(), VdwDampingScheme::Zero);
    }

    #[test]
    fn test_parser_uses_vdw_defaults() {
        let control = parse_control(&["nband = 10"]).unwrap();
        assert!(!control.get_vdw_correction());
        assert_eq!(control.get_vdw_damping(), VdwDampingScheme::Bj);
    }

    #[test]
    fn test_parser_rejects_invalid_vdw_damping() {
        let err = parse_control(&["vdw_damping = invalid"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("vdw_damping"));
        assert!(err.message.contains("expected one of: zero, bj"));
    }

    #[test]
    fn test_key_cases_match_legacy_key_set() {
        // Keys accepted before parameters were grouped; none may be lost or renamed.
        let legacy = [
            "davidson_ndim",
            "dos_ne",
            "dos_scheme",
            "dos_sigma",
            "ecut_rho",
            "ecut_wfc",
            "eigen_solver",
            "eigval_epsilon",
            "eigval_same_epsilon",
            "electric_field_2d",
            "electric_field_axis",
            "electric_field_origin_frac",
            "energy_epsilon",
            "fft_planner",
            "fft_threads",
            "fft_wisdom_file",
            "geom_optim_alpha",
            "geom_optim_cell",
            "geom_optim_force_tolerance",
            "geom_optim_history_steps",
            "geom_optim_max_steps",
            "geom_optim_scheme",
            "geom_optim_stress_tolerance",
            "hse06_alpha",
            "hse06_omega",
            "hubbard_j",
            "hubbard_l",
            "hubbard_species",
            "hubbard_u",
            "hubbard_u_enabled",
            "kpoint_schedule",
            "kpts_scheme",
            "nband",
            "occ_inversion",
            "output_atomic_density",
            "pot_scheme",
            "provenance_check",
            "provenance_manifest",
            "random_seed",
            "restart",
            "rho_epsilon",
            "save_rho",
            "save_wfc",
            "scf_harris",
            "scf_log_file",
            "scf_log_format",
            "scf_max_iter",
            "scf_max_iter_rand_wfc",
            "scf_max_iter_wfc",
            "scf_min_iter",
            "scf_rho_mix_alpha",
            "scf_rho_mix_beta",
            "scf_rho_mix_history_steps",
            "scf_rho_mix_pulay_metric_weight",
            "scf_rho_mix_scheme",
            "smearing_scheme",
            "spin_scheme",
            "surface_dipole_correction",
            "symmetry",
            "task",
            "temperature",
            "vdw_correction",
            "vdw_damping",
            "verbosity",
            "wannier90_export",
            "wannier90_num_iter",
            "wannier90_num_wann",
            "wannier90_seedname",
            "xc_scheme",
        ];
        let mut covered: Vec<&str> = KEY_CASES.iter().map(|(key, _, _)| *key).collect();
        covered.sort_unstable();
        let mut expected = legacy.to_vec();
        expected.sort_unstable();
        assert_eq!(covered, expected, "add a KEY_CASES entry for each new key");
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1.0e-12 * b.abs().max(1.0)
    }

    fn ev(x: f64) -> f64 {
        x * dwconsts::EV_TO_HA
    }

    type KeyCase = (&'static str, &'static str, fn(&Control) -> bool);

    // dos_scheme has a single accepted value, so its case cannot differ from the default.
    const SINGLE_VALUE_KEYS: &[&str] = &["dos_scheme"];

    // One case per in.ctrl key: a non-default value and a check that the value
    // reached the intended field (in stored units). The check must fail on the
    // defaults, so a key routed to the wrong field is caught.
    const KEY_CASES: &[KeyCase] = &[
        ("task", "band", |c| c.get_task() == Task::Band),
        ("restart", "true", |c| c.get_restart()),
        ("symmetry", "true", |c| c.get_symmetry()),
        ("random_seed", "42", |c| c.get_random_seed() == Some(42)),
        ("kpoint_schedule", "dynamic", |c| {
            c.get_kpoint_schedule_enum() == KPointScheduleScheme::Dynamic
        }),
        ("ecut_wfc", "500", |c| approx(c.get_ecut(), ev(500.0))),
        ("ecut_rho", "3000", |c| approx(c.get_ecutrho(), ev(3000.0))),
        ("nband", "13", |c| c.get_nband() == 13),
        ("spin_scheme", "spin", |c| {
            c.get_spin_scheme_enum() == SpinScheme::Spin
        }),
        ("xc_scheme", "pbe", |c| {
            c.get_xc_scheme_enum() == XcScheme::Pbe
        }),
        ("pot_scheme", "upf-fr", |c| {
            c.get_pot_scheme_enum() == PotScheme::UpfFr
        }),
        ("kpts_scheme", "kline", |c| {
            c.get_kpts_scheme_enum() == KptsScheme::Kline
        }),
        ("smearing_scheme", "fd", |c| {
            c.get_smearing_scheme_enum() == SmearingScheme::Fd
        }),
        ("temperature", "300", |c| approx(c.get_temperature(), 300.0)),
        ("occ_inversion", "0.5", |c| {
            approx(c.get_occ_inversion(), 0.5)
        }),
        ("scf_harris", "true", |c| c.is_scf_harris()),
        ("scf_min_iter", "3", |c| c.get_scf_min_iter() == 3),
        ("scf_max_iter", "77", |c| c.get_scf_max_iter() == 77),
        ("scf_max_iter_rand_wfc", "3", |c| {
            c.get_scf_max_iter_rand_wfc() == 3
        }),
        ("scf_max_iter_wfc", "7", |c| c.get_scf_max_iter_wfc() == 7),
        ("rho_epsilon", "1e-7", |c| {
            approx(c.get_rho_epsilon(), 1.0e-7)
        }),
        ("energy_epsilon", "1e-4", |c| {
            approx(c.get_energy_epsilon(), ev(1.0e-4))
        }),
        ("eigval_epsilon", "1e-3", |c| {
            approx(c.get_eigval_epsilon(), ev(1.0e-3))
        }),
        ("eigval_same_epsilon", "true", |c| {
            c.scf().eigval_same_epsilon
        }),
        ("eigen_solver", "lobpcg", |c| {
            c.get_eigen_solver_enum() == EigenSolverScheme::Lobpcg
        }),
        ("davidson_ndim", "9", |c| c.get_davidson_ndim() == 9),
        ("scf_rho_mix_scheme", "broyden", |c| {
            c.mixing().scheme == MixingScheme::Broyden
        }),
        ("scf_rho_mix_alpha", "0.3", |c| {
            approx(c.get_scf_rho_mix_alpha(), 0.3)
        }),
        ("scf_rho_mix_beta", "0.9", |c| {
            approx(c.get_scf_rho_mix_beta(), 0.9)
        }),
        ("scf_rho_mix_history_steps", "9", |c| {
            c.get_scf_rho_mix_history_steps() == 9
        }),
        ("scf_rho_mix_pulay_metric_weight", "5", |c| {
            approx(c.get_scf_rho_mix_pulay_metric_weight(), 5.0)
        }),
        ("geom_optim_cell", "true", |c| c.get_geom_optim_cell()),
        ("geom_optim_scheme", "diis", |c| {
            c.get_geom_optim_scheme() == GeomOptScheme::Diis
        }),
        ("geom_optim_max_steps", "9", |c| {
            c.get_geom_optim_max_steps() == 9
        }),
        ("geom_optim_history_steps", "7", |c| {
            c.get_geom_optim_history_steps() == 7
        }),
        ("geom_optim_alpha", "0.3", |c| {
            approx(c.get_geom_optim_alpha(), 0.3)
        }),
        ("geom_optim_force_tolerance", "0.02", |c| {
            approx(c.get_geom_optim_force_tolerance(), 0.02)
        }),
        ("geom_optim_stress_tolerance", "0.07", |c| {
            approx(c.get_geom_optim_stress_tolerance(), 0.07)
        }),
        ("fft_threads", "4", |c| c.get_fft_threads() == 4),
        ("fft_planner", "measure", |c| {
            c.get_fft_planner_enum() == FftPlannerScheme::Measure
        }),
        ("fft_wisdom_file", "w.wis", |c| {
            c.get_fft_wisdom_file() == "w.wis"
        }),
        ("surface_dipole_correction", "true", |c| {
            c.get_surface_dipole_correction()
        }),
        ("electric_field_2d", "0.25", |c| {
            approx(c.get_electric_field_2d_vang(), 0.25)
        }),
        ("electric_field_axis", "a", |c| {
            c.get_electric_field_axis_enum() == ElectricFieldAxis::A
        }),
        ("electric_field_origin_frac", "0.2", |c| {
            approx(c.get_electric_field_origin_frac(), 0.2)
        }),
        ("verbosity", "debug", |c| {
            c.get_verbosity_enum() == VerbosityLevel::Debug
        }),
        ("scf_log_format", "csv", |c| {
            c.get_scf_log_format_enum() == ScfLogFormat::Csv
        }),
        ("scf_log_file", "x.log", |c| c.get_scf_log_file() == "x.log"),
        ("save_rho", "false", |c| !c.get_save_rho()),
        ("save_wfc", "false", |c| !c.get_save_wfc()),
        ("output_atomic_density", "true", |c| {
            c.get_output_atomic_density()
        }),
        ("provenance_manifest", "p.json", |c| {
            c.get_provenance_manifest() == "p.json"
        }),
        ("provenance_check", "true", |c| c.get_provenance_check()),
        ("dos_scheme", "GAUSS", |c| {
            c.get_dos_scheme() == DosScheme::Gauss
        }),
        ("dos_sigma", "0.05", |c| approx(c.get_dos_sigma(), ev(0.05))),
        ("dos_ne", "123", |c| c.get_dos_ne() == 123),
        ("wannier90_export", "true", |c| c.get_wannier90_export()),
        ("wannier90_seedname", "si", |c| {
            c.get_wannier90_seedname() == "si"
        }),
        ("wannier90_num_wann", "5", |c| {
            c.get_wannier90_num_wann() == 5
        }),
        ("wannier90_num_iter", "50", |c| {
            c.get_wannier90_num_iter() == 50
        }),
        ("hubbard_u_enabled", "true", |c| c.get_hubbard_u_enabled()),
        ("hubbard_species", "Ni", |c| c.get_hubbard_species() == "Ni"),
        ("hubbard_l", "2", |c| c.get_hubbard_l() == 2),
        ("hubbard_u", "5", |c| approx(c.get_hubbard_u(), ev(5.0))),
        ("hubbard_j", "1", |c| approx(c.get_hubbard_j(), ev(1.0))),
        ("hse06_alpha", "0.3", |c| approx(c.get_hse06_alpha(), 0.3)),
        ("hse06_omega", "0.2", |c| approx(c.get_hse06_omega(), 0.2)),
        ("vdw_correction", "true", |c| c.get_vdw_correction()),
        ("vdw_damping", "zero", |c| {
            c.get_vdw_damping() == VdwDampingScheme::Zero
        }),
    ];

    #[test]
    fn test_every_key_round_trips_to_its_field() {
        let defaults = Control::new();
        for (key, value, check) in KEY_CASES {
            if !SINGLE_VALUE_KEYS.contains(key) {
                assert!(
                    !check(&defaults),
                    "case for '{}' must use a non-default value",
                    key
                );
            }

            let mut control = Control::new();
            match control.set_key(key, value) {
                Ok(true) => {}
                Ok(false) => panic!("key '{}' is not handled", key),
                Err(err) => panic!("key '{}' rejected '{}': {}", key, value, err),
            }
            assert!(check(&control), "key '{}' did not reach its field", key);
        }
        assert_eq!(Control::new().set_key("not_a_key", "1"), Ok(false));
    }

    #[test]
    fn test_ecut_rho_defaults_to_four_times_ecut_wfc() {
        let control = parse_control(&["ecut_wfc = 500"]).unwrap();
        assert!(approx(control.get_ecutrho(), 4.0 * control.get_ecut()));
        assert!(approx(
            Control::new().get_ecutrho(),
            4.0 * Control::new().get_ecut()
        ));

        // An explicit ecut_rho wins regardless of line order.
        for lines in [
            ["ecut_rho = 3000", "ecut_wfc = 500"],
            ["ecut_wfc = 500", "ecut_rho = 3000"],
        ]
        .iter()
        {
            let control = parse_control(lines).unwrap();
            assert!(approx(control.get_ecutrho(), ev(3000.0)));
        }
    }

    #[test]
    fn test_wannier90_num_wann_defaults_to_nband() {
        assert_eq!(
            parse_control(&["nband = 12"])
                .unwrap()
                .get_wannier90_num_wann(),
            12
        );
        let explicit = parse_control(&["nband = 12", "wannier90_num_wann = 5"]).unwrap();
        assert_eq!(explicit.get_wannier90_num_wann(), 5);
        let err = parse_control(&[
            "nband = 4",
            "wannier90_export = true",
            "wannier90_num_wann = 8",
        ])
        .unwrap_err();
        assert_eq!(err.key.as_deref(), Some("wannier90_num_wann"));
    }

    #[test]
    fn test_dos_sigma_default_is_one_tenth_ev() {
        let sigma_ev = Control::new().get_dos_sigma() * dwconsts::HA_TO_EV;
        assert!(approx(sigma_ev, 0.1));
    }

    #[test]
    fn test_new_matches_parsed_defaults() {
        let parsed = parse_control(&[]).unwrap();
        assert_eq!(Control::new(), parsed);
    }

    #[test]
    fn test_reparse_resets_previous_values() {
        let mut control = Control::new();
        let first: Vec<String> = [
            "nband = 12",
            "symmetry = true",
            "temperature = 300",
            "scf_rho_mix_alpha = 0.3",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        control.try_read_lines(&first).unwrap();
        control.try_read_lines(&[]).unwrap();
        assert_eq!(control, Control::new());
    }

    #[test]
    fn test_grouped_accessors_expose_parsed_values() {
        let control = parse_control(&[
            "scf_rho_mix_scheme = broyden",
            "scf_rho_mix_alpha = 0.3",
            "scf_rho_mix_history_steps = 8",
            "hubbard_u = 5.0",
            "hubbard_j = 1.0",
        ])
        .unwrap();
        assert_eq!(control.mixing().scheme, MixingScheme::Broyden);
        assert!((control.mixing().alpha - 0.3).abs() < 1.0e-12);
        assert_eq!(control.mixing().history_steps, 8);
        assert!((control.hubbard().u_eff() * dwconsts::HA_TO_EV - 4.0).abs() < 1.0e-12);
        assert_eq!(control.get_scf_rho_mix_alpha(), control.mixing().alpha);
    }

    #[test]
    fn test_parser_rejects_unknown_mixing_scheme() {
        let err = parse_control(&["scf_rho_mix_scheme = broydn"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("scf_rho_mix_scheme"));
        assert!(err.message.contains("expected one of: pulay, broyden"));

        // "simple" was an undocumented alias for Pulay and is now rejected.
        let err = parse_control(&["scf_rho_mix_scheme = simple"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("scf_rho_mix_scheme"));

        // The deprecated alias goes through the same validation.
        let err = parse_control(&["mixing_scheme = anderson"]).unwrap_err();
        assert_eq!(err.key.as_deref(), Some("scf_rho_mix_scheme"));
    }

    #[test]
    fn test_parser_accepts_mixing_scheme_case_insensitively() {
        let control = parse_control(&["scf_rho_mix_scheme = Broyden"]).unwrap();
        assert_eq!(control.mixing().scheme, MixingScheme::Broyden);
        assert_eq!(control.get_scf_rho_mix_scheme(), MixingScheme::Broyden);
    }

    #[test]
    fn test_mixing_scheme_names_round_trip() {
        for scheme in [MixingScheme::Pulay, MixingScheme::Broyden].iter() {
            assert_eq!(MixingScheme::parse(scheme.as_str()), Some(*scheme));
        }
        // "simple" used to be an undocumented alias for Pulay; it is no longer accepted.
        assert_eq!(MixingScheme::parse("simple"), None);
        assert_eq!(
            parse_control(&[]).unwrap().get_scf_rho_mix_scheme(),
            MixingScheme::Pulay
        );
    }

    #[test]
    fn test_mixing_scheme_error_lists_accepted_names() {
        let err = parse_control(&["scf_rho_mix_scheme = linear"]).unwrap_err();
        assert!(err.message.contains("expected one of: pulay, broyden"));
    }

    #[test]
    fn test_new_scheme_names_round_trip() {
        for v in [Task::Scf, Task::Band].iter() {
            assert_eq!(Task::parse(v.as_str()), Some(*v));
        }
        for v in [GeomOptScheme::Bfgs, GeomOptScheme::Diis].iter() {
            assert_eq!(GeomOptScheme::parse(v.as_str()), Some(*v));
        }
        assert_eq!(DosScheme::parse("gauss"), Some(DosScheme::Gauss));
        for v in [VdwDampingScheme::Zero, VdwDampingScheme::Bj].iter() {
            assert_eq!(VdwDampingScheme::parse(v.as_str()), Some(*v));
        }
        for v in [EigenSolverScheme::Pcg, EigenSolverScheme::Lobpcg].iter() {
            assert_eq!(EigenSolverScheme::parse(v.as_str()), Some(*v));
        }
    }

    #[test]
    fn test_parser_rejects_unknown_named_values() {
        let cases = [
            (
                "geom_optim_scheme = lbfgs",
                "geom_optim_scheme",
                "expected one of: bfgs, diis",
            ),
            (
                "dos_scheme = lorentz",
                "dos_scheme",
                "expected one of: gauss",
            ),
            (
                "vdw_damping = d3",
                "vdw_damping",
                "expected one of: zero, bj",
            ),
        ];
        for (line, key, message) in cases.iter() {
            let err = parse_control(&[*line]).unwrap_err();
            assert_eq!(err.key.as_deref(), Some(*key));
            assert!(err.message.contains(message), "{}", err.message);
        }
    }
}
