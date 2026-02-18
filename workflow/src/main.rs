use std::collections::{HashMap, HashSet};
use std::env;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::{Args, Parser, Subcommand, ValueEnum};
use crystal::Crystal;
use symops::{classify_symmetry, detect_symmetry, CrystalSystem, DetectOptions, Structure};

const STAGE_SCF: &str = "scf";
const STAGE_NSCF: &str = "nscf";
const STAGE_BANDS: &str = "bands";
const STAGE_WANNIER: &str = "wannier";
const STAGE_PIPELINE: &str = "pipeline";
const CANONICAL_STAGES: [&str; 4] = [STAGE_SCF, STAGE_NSCF, STAGE_BANDS, STAGE_WANNIER];

type CliResult<T> = Result<T, String>;

#[derive(Clone, Debug, Parser)]
#[command(name = "dwf")]
struct Cli {
    #[command(subcommand)]
    command: DwfCommand,
}

#[derive(Clone, Debug, Subcommand)]
enum DwfCommand {
    Validate(ValidateArgs),
    Run(RunArgs),
    Properties(PropertiesArgs),
    Status(StatusArgs),
}

#[derive(Clone, Debug, Args)]
struct ValidateArgs {
    case_dir: PathBuf,
    #[arg(long, value_name = "yaml")]
    config: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
struct StatusArgs {
    case_dir: PathBuf,
    #[arg(long, value_name = "yaml")]
    config: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
struct RunArgs {
    stage: RunStageArg,
    case_dir: PathBuf,
    #[arg(long, value_name = "stage:latest|run_dir|latest")]
    from: Option<String>,
    #[arg(long, value_name = "path")]
    pw_bin: Option<PathBuf>,
    #[arg(long, value_name = "path")]
    w90_win_bin: Option<PathBuf>,
    #[arg(long, value_name = "path")]
    w90_amn_bin: Option<PathBuf>,
    #[arg(long, value_name = "path")]
    wannier90_x_bin: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', value_parser = parse_stage_list_item)]
    stages: Option<Vec<String>>,
    #[arg(long, value_name = "yaml")]
    config: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
struct PropertiesArgs {
    run_dir: PathBuf,
    stage: PropertiesStageArg,
    #[arg(long, value_name = "path")]
    log: Option<PathBuf>,
    #[arg(long = "dos-sigma")]
    dos_sigma: Option<f64>,
    #[arg(long = "dos-ne")]
    dos_ne: Option<usize>,
    #[arg(long = "dos-emin")]
    dos_emin: Option<f64>,
    #[arg(long = "dos-emax")]
    dos_emax: Option<f64>,
    #[arg(long = "dos-format")]
    dos_format: Option<DosFormatArg>,
    #[arg(long = "fermi-tol")]
    fermi_tol: Option<f64>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RunStageArg {
    Scf,
    Nscf,
    Bands,
    Wannier,
    Pipeline,
}

impl RunStageArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scf => STAGE_SCF,
            Self::Nscf => STAGE_NSCF,
            Self::Bands => STAGE_BANDS,
            Self::Wannier => STAGE_WANNIER,
            Self::Pipeline => STAGE_PIPELINE,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PropertiesStageArg {
    Scf,
    Nscf,
    Bands,
}

impl PropertiesStageArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scf => STAGE_SCF,
            Self::Nscf => STAGE_NSCF,
            Self::Bands => STAGE_BANDS,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DosFormatArg {
    Dat,
    Csv,
    Json,
}

impl DosFormatArg {
    fn as_output_format(self) -> property::DosOutputFormat {
        match self {
            Self::Dat => property::DosOutputFormat::Dat,
            Self::Csv => property::DosOutputFormat::Csv,
            Self::Json => property::DosOutputFormat::Json,
        }
    }
}

fn parse_stage_list_item(raw: &str) -> Result<String, String> {
    let stage = raw.trim();
    if stage.is_empty() {
        return Err("empty stage in --stages list".to_string());
    }
    Ok(stage.to_string())
}

#[derive(Clone, Debug)]
struct StageInputs {
    files: Vec<(String, PathBuf)>,
}

#[derive(Clone, Debug, Default)]
struct RunOptions {
    from_scf: Option<String>,
    pw_bin: Option<PathBuf>,
    w90_win_bin: Option<PathBuf>,
    w90_amn_bin: Option<PathBuf>,
    wannier90_x_bin: Option<PathBuf>,
    stages: Option<Vec<String>>,
    config_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct CaseConfig {
    case_dir: PathBuf,
    config_file: Option<PathBuf>,
    common_dir: PathBuf,
    stage_dirs: HashMap<String, PathBuf>,
    pipeline_stages: Vec<String>,
    binaries: BinaryYamlConfig,
}

impl CaseConfig {
    fn stage_dir(&self, stage: &str) -> Option<&Path> {
        self.stage_dirs.get(stage).map(|p| p.as_path())
    }

    fn stage_enabled(&self, stage: &str) -> bool {
        self.stage_dirs.contains_key(stage)
    }
}

#[derive(Clone, Debug, Default)]
struct WorkflowYamlConfig {
    common_dir: Option<String>,
    stages: Option<HashMap<String, StageYamlConfig>>,
    pipeline: Option<PipelineYamlConfig>,
    binaries: Option<BinaryYamlConfig>,
}

#[derive(Clone, Debug, Default)]
struct StageYamlConfig {
    dir: Option<String>,
    enabled: Option<bool>,
}

#[derive(Clone, Debug, Default)]
struct PipelineYamlConfig {
    stages: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default)]
struct BinaryYamlConfig {
    pw: Option<String>,
    w90_win: Option<String>,
    w90_amn: Option<String>,
    wannier90_x: Option<String>,
}

#[derive(Clone, Debug)]
struct ResolvedSource {
    stage: Option<String>,
    run_dir: PathBuf,
}

#[derive(Clone, Copy)]
struct SourceCopyOptions {
    rho: bool,
    wfc: bool,
    eig: bool,
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run_cli(cli) {
        eprintln!("error: {}", err);
        std::process::exit(1);
    }
}

fn run_cli(cli: Cli) -> CliResult<()> {
    match cli.command {
        DwfCommand::Validate(args) => run_validate_command(args),
        DwfCommand::Run(args) => run_stage_command(args),
        DwfCommand::Properties(args) => run_properties_command(args),
        DwfCommand::Status(args) => run_status_command(args),
    }
}

fn run_validate_command(args: ValidateArgs) -> CliResult<()> {
    let config = load_case_config(&args.case_dir, args.config.as_deref())?;
    validate_case(&config)
}

fn run_status_command(args: StatusArgs) -> CliResult<()> {
    let config = load_case_config(&args.case_dir, args.config.as_deref())?;
    print_status(&config)
}

fn run_stage_command(args: RunArgs) -> CliResult<()> {
    let opts = RunOptions {
        from_scf: args.from,
        pw_bin: args.pw_bin,
        w90_win_bin: args.w90_win_bin,
        w90_amn_bin: args.w90_amn_bin,
        wannier90_x_bin: args.wannier90_x_bin,
        stages: args.stages,
        config_path: args.config,
    };
    let stage = args.stage.as_str();
    let config = load_case_config(&args.case_dir, opts.config_path.as_deref())?;

    match stage {
        STAGE_SCF | STAGE_NSCF | STAGE_BANDS => {
            run_stage(&config, stage, opts.from_scf, opts.pw_bin)?;
            Ok(())
        }
        STAGE_WANNIER => {
            run_wannier_stage(&config, opts)?;
            Ok(())
        }
        STAGE_PIPELINE => run_pipeline(&config, opts),
        _ => Err(format!("unsupported stage '{}'", stage)),
    }
}

fn run_properties_command(args: PropertiesArgs) -> CliResult<()> {
    let run_dir = args.run_dir;
    if !run_dir.is_dir() {
        return Err(format!(
            "run directory '{}' does not exist",
            run_dir.display()
        ));
    }

    let stage = args.stage.as_str();
    let mut log_path = run_dir.join("out.pw.log");
    let mut options = property::PropertyExportOptions::default();
    if let Some(log) = args.log {
        log_path = if log.is_absolute() {
            log
        } else {
            run_dir.join(log)
        };
    }
    options.dos_sigma_ev = args.dos_sigma;
    options.dos_ne = args.dos_ne;
    options.dos_emin_ev = args.dos_emin;
    options.dos_emax_ev = args.dos_emax;
    if let Some(dos_format) = args.dos_format {
        options.dos_format = dos_format.as_output_format();
    }
    if let Some(fermi_tol) = args.fermi_tol {
        options.fermi_tol_ev = fermi_tol;
    }

    if !log_path.is_file() {
        return Err(format!(
            "stage log '{}' does not exist; pass --log <path> if needed",
            log_path.display()
        ));
    }

    property::export_stage_properties_with_options(&run_dir, stage, &log_path, 0.0, &options)
        .map_err(|err| format!("failed to export properties: {}", err))?;

    println!(
        "Exported properties to '{}'",
        run_dir.join("properties").display()
    );
    Ok(())
}

fn is_run_stage(stage: &str) -> bool {
    matches!(stage, STAGE_SCF | STAGE_NSCF | STAGE_BANDS | STAGE_WANNIER)
}

fn load_case_config(case_dir: &Path, config_path: Option<&Path>) -> CliResult<CaseConfig> {
    if !case_dir.is_dir() {
        return Err(format!(
            "case directory '{}' does not exist",
            case_dir.display()
        ));
    }

    let config_file = resolve_config_file(case_dir, config_path)?;
    let yaml = if let Some(path) = config_file.as_ref() {
        let content = fs::read_to_string(path)
            .map_err(|err| format!("failed to read config '{}': {}", path.display(), err))?;
        parse_workflow_yaml(&content)
            .map_err(|err| format!("failed to parse YAML config '{}': {}", path.display(), err))?
    } else {
        WorkflowYamlConfig::default()
    };

    let WorkflowYamlConfig {
        common_dir,
        stages,
        pipeline,
        binaries,
    } = yaml;

    let common_dir = resolve_directory_entry(case_dir, common_dir.as_deref().unwrap_or("common"))?;

    let mut stage_cfg = stages.unwrap_or_default();
    for key in stage_cfg.keys() {
        if !is_run_stage(key.as_str()) {
            return Err(format!(
                "unknown stage '{}' in YAML 'stages'; allowed: {},{},{},{}",
                key, STAGE_SCF, STAGE_NSCF, STAGE_BANDS, STAGE_WANNIER
            ));
        }
    }

    let mut stage_dirs = HashMap::new();
    for stage in CANONICAL_STAGES.iter() {
        let cfg = stage_cfg.remove(*stage).unwrap_or_default();
        if cfg.enabled == Some(false) {
            continue;
        }
        let dir_raw = cfg.dir.unwrap_or_else(|| (*stage).to_string());
        let dir_path = resolve_directory_entry(case_dir, &dir_raw)?;
        stage_dirs.insert((*stage).to_string(), dir_path);
    }

    let pipeline_stages = if let Some(pipe_cfg) = pipeline {
        if let Some(stages) = pipe_cfg.stages {
            normalize_pipeline_stages(&stages, &stage_dirs)?
        } else {
            default_pipeline_stages(&stage_dirs)
        }
    } else {
        default_pipeline_stages(&stage_dirs)
    };

    Ok(CaseConfig {
        case_dir: case_dir.to_path_buf(),
        config_file,
        common_dir,
        stage_dirs,
        pipeline_stages,
        binaries: binaries.unwrap_or_default(),
    })
}

fn resolve_config_file(case_dir: &Path, config_path: Option<&Path>) -> CliResult<Option<PathBuf>> {
    if let Some(path) = config_path {
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            case_dir.join(path)
        };

        if !resolved.is_file() {
            return Err(format!(
                "config file '{}' does not exist",
                resolved.display()
            ));
        }

        return Ok(Some(resolved));
    }

    for name in ["dwf.yaml", "workflow.yaml"] {
        let path = case_dir.join(name);
        if path.is_file() {
            return Ok(Some(path));
        }
    }

    Ok(None)
}

#[derive(Clone, Debug)]
struct YamlLine {
    indent: usize,
    line_no: usize,
    text: String,
}

fn parse_workflow_yaml(content: &str) -> CliResult<WorkflowYamlConfig> {
    let lines = tokenize_yaml_lines(content)?;
    let mut cfg = WorkflowYamlConfig::default();

    let mut i = 0usize;
    while i < lines.len() {
        let line = &lines[i];
        if line.indent != 0 {
            return Err(format!("line {}: expected a top-level key", line.line_no));
        }

        let (key, value) = parse_yaml_mapping_line(line)?;
        match key.as_str() {
            "common_dir" => {
                let value = value.ok_or_else(|| {
                    format!("line {}: 'common_dir' requires a value", line.line_no)
                })?;
                cfg.common_dir = Some(parse_yaml_scalar(&value));
                i += 1;
            }
            "stages" => {
                if value.is_some() {
                    return Err(format!(
                        "line {}: 'stages' must be a nested mapping block",
                        line.line_no
                    ));
                }
                let (stages, next) = parse_stages_block(&lines, i + 1, line.indent)?;
                cfg.stages = Some(stages);
                i = next;
            }
            "pipeline" => {
                if value.is_some() {
                    return Err(format!(
                        "line {}: 'pipeline' must be a nested mapping block",
                        line.line_no
                    ));
                }
                let (pipeline, next) = parse_pipeline_block(&lines, i + 1, line.indent)?;
                cfg.pipeline = Some(pipeline);
                i = next;
            }
            "binaries" => {
                if value.is_some() {
                    return Err(format!(
                        "line {}: 'binaries' must be a nested mapping block",
                        line.line_no
                    ));
                }
                let (binaries, next) = parse_binaries_block(&lines, i + 1, line.indent)?;
                cfg.binaries = Some(binaries);
                i = next;
            }
            other => {
                return Err(format!(
                    "line {}: unknown top-level key '{}'",
                    line.line_no, other
                ));
            }
        }
    }

    Ok(cfg)
}

fn tokenize_yaml_lines(content: &str) -> CliResult<Vec<YamlLine>> {
    let mut out = Vec::new();

    for (idx, raw) in content.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_yaml_comment(raw);
        if line.trim().is_empty() {
            continue;
        }

        let leading_ws = line
            .chars()
            .take_while(|c| c.is_whitespace())
            .collect::<String>();
        if leading_ws.contains('\t') {
            return Err(format!(
                "line {}: tab indentation is not supported in dwf YAML",
                line_no
            ));
        }

        let indent = line.chars().take_while(|c| *c == ' ').count();
        let text = line[indent..].trim_end().to_string();
        if text == "---" || text == "..." {
            continue;
        }

        out.push(YamlLine {
            indent,
            line_no,
            text,
        });
    }

    Ok(out)
}

fn strip_yaml_comment(line: &str) -> String {
    let mut out = String::new();
    let mut in_single = false;
    let mut in_double = false;

    for ch in line.chars() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                out.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                out.push(ch);
            }
            '#' if !in_single && !in_double => break,
            _ => out.push(ch),
        }
    }

    out
}

fn parse_yaml_mapping_line(line: &YamlLine) -> CliResult<(String, Option<String>)> {
    let (key, value) = line
        .text
        .split_once(':')
        .ok_or_else(|| format!("line {}: expected 'key: value' mapping", line.line_no))?;
    let key = key.trim();
    if key.is_empty() {
        return Err(format!("line {}: empty YAML key", line.line_no));
    }

    let value = value.trim();
    if value.is_empty() {
        Ok((key.to_string(), None))
    } else {
        Ok((key.to_string(), Some(value.to_string())))
    }
}

fn parse_stages_block(
    lines: &[YamlLine],
    mut i: usize,
    parent_indent: usize,
) -> CliResult<(HashMap<String, StageYamlConfig>, usize)> {
    let mut out = HashMap::new();

    while i < lines.len() {
        let line = &lines[i];
        if line.indent <= parent_indent {
            break;
        }

        let stage_indent = line.indent;
        let (stage_name, value) = parse_yaml_mapping_line(line)?;
        if value.is_some() {
            return Err(format!(
                "line {}: stage '{}' must use a nested mapping",
                line.line_no, stage_name
            ));
        }

        let mut stage_cfg = StageYamlConfig::default();
        i += 1;

        while i < lines.len() && lines[i].indent > stage_indent {
            let item = &lines[i];
            let (key, value) = parse_yaml_mapping_line(item)?;
            let raw_value = value.ok_or_else(|| {
                format!(
                    "line {}: stage key '{}' requires a scalar value",
                    item.line_no, key
                )
            })?;
            match key.as_str() {
                "dir" => {
                    stage_cfg.dir = Some(parse_yaml_scalar(&raw_value));
                }
                "enabled" => {
                    stage_cfg.enabled = Some(parse_yaml_bool(&raw_value, item.line_no)?);
                }
                other => {
                    return Err(format!(
                        "line {}: unknown key '{}' under stage '{}'",
                        item.line_no, other, stage_name
                    ));
                }
            }
            i += 1;
        }

        out.insert(stage_name, stage_cfg);
    }

    Ok((out, i))
}

fn parse_pipeline_block(
    lines: &[YamlLine],
    mut i: usize,
    parent_indent: usize,
) -> CliResult<(PipelineYamlConfig, usize)> {
    let mut pipeline = PipelineYamlConfig::default();

    while i < lines.len() {
        let line = &lines[i];
        if line.indent <= parent_indent {
            break;
        }

        let item_indent = line.indent;
        let (key, value) = parse_yaml_mapping_line(line)?;
        match key.as_str() {
            "stages" => {
                if let Some(raw) = value {
                    let stages = parse_stage_list_scalar(&raw, line.line_no)?;
                    pipeline.stages = Some(stages);
                    i += 1;
                } else {
                    let (stages, next) = parse_yaml_list(lines, i + 1, item_indent)?;
                    pipeline.stages = Some(stages);
                    i = next;
                }
            }
            other => {
                return Err(format!(
                    "line {}: unknown key '{}' under 'pipeline'",
                    line.line_no, other
                ));
            }
        }
    }

    Ok((pipeline, i))
}

fn parse_binaries_block(
    lines: &[YamlLine],
    mut i: usize,
    parent_indent: usize,
) -> CliResult<(BinaryYamlConfig, usize)> {
    let mut binaries = BinaryYamlConfig::default();

    while i < lines.len() {
        let line = &lines[i];
        if line.indent <= parent_indent {
            break;
        }

        let (key, value) = parse_yaml_mapping_line(line)?;
        let raw_value =
            value.ok_or_else(|| format!("line {}: '{}' requires a value", line.line_no, key))?;
        let parsed = parse_yaml_scalar(&raw_value);
        match key.as_str() {
            "pw" => binaries.pw = Some(parsed),
            "w90_win" => binaries.w90_win = Some(parsed),
            "w90_amn" => binaries.w90_amn = Some(parsed),
            "wannier90_x" => binaries.wannier90_x = Some(parsed),
            other => {
                return Err(format!(
                    "line {}: unknown key '{}' under 'binaries'",
                    line.line_no, other
                ));
            }
        }

        i += 1;
    }

    Ok((binaries, i))
}

fn parse_stage_list_scalar(raw: &str, line_no: usize) -> CliResult<Vec<String>> {
    let value = raw.trim();
    if value.starts_with('[') && value.ends_with(']') {
        return parse_inline_yaml_list(value, line_no);
    }

    let single = parse_yaml_scalar(value);
    if single.is_empty() {
        return Err(format!("line {}: empty stage name", line_no));
    }
    Ok(vec![single])
}

fn parse_inline_yaml_list(raw: &str, line_no: usize) -> CliResult<Vec<String>> {
    let trimmed = raw.trim();
    if !(trimmed.starts_with('[') && trimmed.ends_with(']')) {
        return Err(format!(
            "line {}: expected inline list syntax [a, b, c]",
            line_no
        ));
    }

    let inner = trimmed[1..trimmed.len() - 1].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for part in inner.split(',') {
        let value = parse_yaml_scalar(part);
        if value.is_empty() {
            return Err(format!("line {}: empty list item", line_no));
        }
        out.push(value);
    }

    Ok(out)
}

fn parse_yaml_list(
    lines: &[YamlLine],
    mut i: usize,
    parent_indent: usize,
) -> CliResult<(Vec<String>, usize)> {
    let mut items = Vec::new();

    while i < lines.len() {
        let line = &lines[i];
        if line.indent <= parent_indent {
            break;
        }

        let item = line
            .text
            .strip_prefix('-')
            .ok_or_else(|| format!("line {}: expected list item '- value'", line.line_no))?
            .trim();
        if item.is_empty() {
            return Err(format!("line {}: empty list item", line.line_no));
        }
        items.push(parse_yaml_scalar(item));
        i += 1;
    }

    Ok((items, i))
}

fn parse_yaml_scalar(raw: &str) -> String {
    let value = raw.trim();
    if value.len() >= 2 {
        let bytes = value.as_bytes();
        let quoted = (bytes[0] == b'"' && bytes[value.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[value.len() - 1] == b'\'');
        if quoted {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

fn parse_yaml_bool(raw: &str, line_no: usize) -> CliResult<bool> {
    match parse_yaml_scalar(raw).to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" => Ok(true),
        "false" | "no" | "off" => Ok(false),
        other => Err(format!(
            "line {}: invalid boolean value '{}', expected true/false",
            line_no, other
        )),
    }
}

fn resolve_directory_entry(case_dir: &Path, raw: &str) -> CliResult<PathBuf> {
    let value = raw.trim();
    if value.is_empty() {
        return Err("YAML path value must not be empty".to_string());
    }

    let path = PathBuf::from(value);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(case_dir.join(path))
    }
}

fn normalize_pipeline_stages(
    stages: &[String],
    stage_dirs: &HashMap<String, PathBuf>,
) -> CliResult<Vec<String>> {
    if stages.is_empty() {
        return Err("YAML 'pipeline.stages' is empty".to_string());
    }

    let mut out = Vec::new();
    for stage in stages.iter() {
        if !is_run_stage(stage) {
            return Err(format!(
                "invalid stage '{}' in YAML 'pipeline.stages'; allowed: {},{},{},{}",
                stage, STAGE_SCF, STAGE_NSCF, STAGE_BANDS, STAGE_WANNIER
            ));
        }
        if !stage_dirs.contains_key(stage) {
            return Err(format!(
                "stage '{}' is listed in YAML 'pipeline.stages' but disabled in YAML 'stages'",
                stage
            ));
        }
        if !out.contains(stage) {
            out.push(stage.clone());
        }
    }

    Ok(out)
}

fn default_pipeline_stages(stage_dirs: &HashMap<String, PathBuf>) -> Vec<String> {
    CANONICAL_STAGES
        .iter()
        .filter(|stage| {
            stage_dirs
                .get(**stage)
                .map(|path| path.is_dir())
                .unwrap_or(false)
        })
        .map(|stage| (*stage).to_string())
        .collect()
}

fn validate_case(config: &CaseConfig) -> CliResult<()> {
    println!("Case: {}", config.case_dir.display());
    if let Some(path) = config.config_file.as_ref() {
        println!("Config: {}", path.display());
    }

    let mut configured = 0usize;
    let mut total_errors = 0usize;

    for stage in CANONICAL_STAGES.iter() {
        if !config.stage_enabled(stage) {
            println!("  [{}] disabled", stage);
            continue;
        }

        let stage_dir = match config.stage_dir(stage) {
            Some(path) => path,
            None => {
                println!("  [{}] disabled", stage);
                continue;
            }
        };

        if !stage_dir.is_dir() {
            println!(
                "  [{}] not configured (missing {})",
                stage,
                stage_dir.display()
            );
            continue;
        }

        configured += 1;
        let errors = validate_stage_inputs(config, stage)?;
        if errors.is_empty() {
            println!("  [{}] ok", stage);
        } else {
            println!("  [{}] invalid", stage);
            for err in errors.iter() {
                println!("    - {}", err);
            }
            total_errors += errors.len();
        }
    }

    if configured == 0 {
        return Err(format!(
            "no configured stages found under '{}'; create at least one stage directory",
            config.case_dir.display()
        ));
    }

    if total_errors > 0 {
        return Err(format!("validation failed with {} error(s)", total_errors));
    }

    if !config.pipeline_stages.is_empty() {
        println!(
            "Pipeline default stages: {}",
            config.pipeline_stages.join(",")
        );
    }

    println!("Validation passed.");
    Ok(())
}

fn validate_stage_inputs(config: &CaseConfig, stage: &str) -> CliResult<Vec<String>> {
    let mut errors = Vec::new();
    let required = required_input_files(stage)?;

    let stage_dir = config
        .stage_dir(stage)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| format!("{}/{}", config.case_dir.display(), stage));

    for filename in required.iter() {
        if resolve_input(config, stage, filename).is_none() {
            errors.push(format!(
                "missing '{}' (looked in '{}' and '{}')",
                filename,
                stage_dir,
                config.common_dir.display()
            ));
        }
    }

    if let Some(ctrl_path) = resolve_input(config, stage, "in.ctrl") {
        if spin_file_required(&ctrl_path)? && resolve_input(config, stage, "in.spin").is_none() {
            errors.push(format!(
                "spin run detected from '{}' but missing 'in.spin'",
                ctrl_path.display()
            ));
        }
    }

    if let Some(in_pot_path) = resolve_input(config, stage, "in.pot") {
        let pot_errors = validate_pot_files(config, stage, &in_pot_path)?;
        errors.extend(pot_errors);
    }

    Ok(errors)
}

fn required_input_files(stage: &str) -> CliResult<Vec<&'static str>> {
    match stage {
        STAGE_SCF | STAGE_NSCF | STAGE_WANNIER => {
            Ok(vec!["in.ctrl", "in.kmesh", "in.crystal", "in.pot"])
        }
        STAGE_BANDS => Ok(vec!["in.ctrl", "in.kline", "in.crystal", "in.pot"]),
        _ => Err(format!("unknown stage '{}'", stage)),
    }
}

fn run_stage(
    config: &CaseConfig,
    stage: &str,
    from_scf: Option<String>,
    pw_bin_arg: Option<PathBuf>,
) -> CliResult<PathBuf> {
    ensure_stage_configured(config, stage)?;

    let errors = validate_stage_inputs(config, stage)?;
    if !errors.is_empty() {
        let joined = errors.join("; ");
        return Err(format!("cannot run '{}': {}", stage, joined));
    }

    let run_dir = create_run_dir(&config.case_dir, stage)?;
    write_text(&run_dir.join("status.txt"), "preparing\n")?;
    write_text(&run_dir.join("stage.txt"), &format!("{}\n", stage))?;

    let inputs = collect_stage_inputs(config, stage)?;
    if let Err(err) = copy_inputs(&inputs, &run_dir) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(
            &run_dir.join("exit_code.txt"),
            "prepare-copy-inputs-failed\n",
        );
        return Err(err);
    }
    if let Err(err) = copy_pot_files(config, stage, &run_dir) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "prepare-copy-pot-failed\n");
        return Err(err);
    }

    if matches!(stage, STAGE_NSCF | STAGE_BANDS) {
        let source = resolve_stage_source(config, stage, from_scf.as_deref())?;
        let copy_opts = SourceCopyOptions {
            rho: true,
            wfc: false,
            eig: false,
        };

        if let Err(err) = copy_source_outputs(&source.run_dir, &run_dir, copy_opts) {
            let _ = write_text(&run_dir.join("status.txt"), "failed\n");
            let _ = write_text(&run_dir.join("exit_code.txt"), "prepare-copy-scf-failed\n");
            return Err(err);
        }

        if let Err(err) = write_source_metadata(&run_dir, &source) {
            let _ = write_text(&run_dir.join("status.txt"), "failed\n");
            let _ = write_text(
                &run_dir.join("exit_code.txt"),
                "prepare-write-source-meta-failed\n",
            );
            return Err(err);
        }
    }

    write_text(&run_dir.join("status.txt"), "running\n")?;

    let pw_bin = resolve_pw_bin(pw_bin_arg, config);
    if let Err(err) = ensure_binary_available(&pw_bin, "pw", "--pw-bin <path>") {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "missing-pw\n");
        return Err(err);
    }

    let log_path = run_dir.join("out.pw.log");
    let (exec_bin, exec_args, command_display) = build_stage_pw_command(stage, &pw_bin);
    let command_line = format!("{} (cwd={})", command_display, run_dir.display());
    write_text(&run_dir.join("command.txt"), &format!("{}\n", command_line))?;

    println!("Running stage '{}' in {}", stage, run_dir.display());
    println!("Command: {}", command_display);

    let log_file = File::create(&log_path).map_err(|err| {
        format!(
            "failed to create log file '{}': {}",
            log_path.display(),
            err
        )
    })?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|err| format!("failed to clone log file handle: {}", err))?;

    let stopwatch = Instant::now();
    let status = Command::new(&exec_bin)
        .current_dir(&run_dir)
        .args(exec_args)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .status();
    let elapsed_wall_seconds = stopwatch.elapsed().as_secs_f64();

    let status = match status {
        Ok(status) => status,
        Err(err) => {
            let _ = write_text(&run_dir.join("status.txt"), "failed\n");
            let _ = write_text(&run_dir.join("exit_code.txt"), "spawn-error\n");
            return Err(format!(
                "failed to start '{}' for stage '{}': {}",
                exec_bin.display(),
                stage,
                err
            ));
        }
    };

    let exit_text = match status.code() {
        Some(code) => code.to_string(),
        None => "terminated-by-signal".to_string(),
    };
    write_text(&run_dir.join("exit_code.txt"), &format!("{}\n", exit_text))?;

    if status.success() {
        if let Err(err) =
            property::export_stage_properties(&run_dir, stage, &log_path, elapsed_wall_seconds)
        {
            eprintln!(
                "warning: failed to export stage properties for '{}': {}",
                stage, err
            );
        }
        match export_symmetry_metadata(&run_dir) {
            Ok(path) => {
                println!("Symmetry: {}", path.display());
            }
            Err(err) => {
                eprintln!(
                    "warning: failed to export symmetry metadata for '{}': {}",
                    stage, err
                );
            }
        }
        write_text(&run_dir.join("status.txt"), "success\n")?;
        println!("Stage '{}' completed successfully.", stage);
        println!("Run directory: {}", run_dir.display());
        println!("Log file: {}", log_path.display());
        println!("Properties: {}", run_dir.join("properties").display());
        Ok(run_dir)
    } else {
        write_text(&run_dir.join("status.txt"), "failed\n")?;
        Err(format!(
            "stage '{}' failed (exit={}); see '{}'",
            stage,
            exit_text,
            log_path.display()
        ))
    }
}

fn run_wannier_stage(config: &CaseConfig, opts: RunOptions) -> CliResult<PathBuf> {
    ensure_stage_configured(config, STAGE_WANNIER)?;

    let errors = validate_stage_inputs(config, STAGE_WANNIER)?;
    if !errors.is_empty() {
        let joined = errors.join("; ");
        return Err(format!("cannot run '{}': {}", STAGE_WANNIER, joined));
    }

    let run_dir = create_run_dir(&config.case_dir, STAGE_WANNIER)?;
    write_text(&run_dir.join("status.txt"), "preparing\n")?;
    write_text(&run_dir.join("stage.txt"), &format!("{}\n", STAGE_WANNIER))?;

    let inputs = collect_stage_inputs(config, STAGE_WANNIER)?;
    if let Err(err) = copy_inputs(&inputs, &run_dir) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(
            &run_dir.join("exit_code.txt"),
            "prepare-copy-inputs-failed\n",
        );
        return Err(err);
    }
    if let Err(err) = copy_pot_files(config, STAGE_WANNIER, &run_dir) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "prepare-copy-pot-failed\n");
        return Err(err);
    }

    let source = resolve_stage_source(config, STAGE_WANNIER, opts.from_scf.as_deref())?;
    let copy_opts = SourceCopyOptions {
        rho: true,
        wfc: true,
        eig: true,
    };

    if let Err(err) = copy_source_outputs(&source.run_dir, &run_dir, copy_opts) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "prepare-copy-scf-failed\n");
        return Err(err);
    }
    if let Err(err) = write_source_metadata(&run_dir, &source) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(
            &run_dir.join("exit_code.txt"),
            "prepare-write-source-meta-failed\n",
        );
        return Err(err);
    }

    write_text(&run_dir.join("status.txt"), "running\n")?;

    let w90_win_bin = resolve_w90_win_bin(opts.w90_win_bin, config);
    let w90_amn_bin = resolve_w90_amn_bin(opts.w90_amn_bin, config);
    let wannier90_x_bin = resolve_wannier90_x_bin(opts.wannier90_x_bin, config);

    if let Err(err) = ensure_binary_available(&w90_win_bin, "w90-win", "--w90-win-bin <path>") {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "missing-w90-win\n");
        return Err(err);
    }
    if let Err(err) = ensure_binary_available(&w90_amn_bin, "w90-amn", "--w90-amn-bin <path>") {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "missing-w90-amn\n");
        return Err(err);
    }
    if let Err(err) =
        ensure_binary_available(&wannier90_x_bin, "wannier90.x", "--wannier90-x-bin <path>")
    {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "missing-wannier90.x\n");
        return Err(err);
    }

    let command_manifest = format!(
        "{}\n{}\n{}\n",
        w90_win_bin.display(),
        w90_amn_bin.display(),
        wannier90_x_bin.display()
    );
    write_text(&run_dir.join("command.txt"), &command_manifest)?;

    println!("Running stage '{}' in {}", STAGE_WANNIER, run_dir.display());
    println!("Command: {}", w90_win_bin.display());
    if let Err(err) = run_logged_command(&run_dir, "out.w90-win.log", &w90_win_bin, &[]) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "w90-win-failed\n");
        return Err(err);
    }

    if let Err(err) = apply_projection_overrides(&run_dir) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(
            &run_dir.join("exit_code.txt"),
            "prepare-apply-projection-failed\n",
        );
        return Err(err);
    }

    println!("Command: {}", w90_amn_bin.display());
    if let Err(err) = run_logged_command(&run_dir, "out.w90-amn.log", &w90_amn_bin, &[]) {
        let _ = write_text(&run_dir.join("status.txt"), "failed\n");
        let _ = write_text(&run_dir.join("exit_code.txt"), "w90-amn-failed\n");
        return Err(err);
    }

    let seeds = discover_seednames_from_win(&run_dir)?;
    for seed in seeds.iter() {
        let eig_path = run_dir.join(format!("{}.eig", seed));
        if !eig_path.is_file() {
            let _ = write_text(&run_dir.join("status.txt"), "failed\n");
            let _ = write_text(&run_dir.join("exit_code.txt"), "missing-eig\n");
            return Err(format!(
                "missing required eig file '{}' before running '{} {}'",
                eig_path.display(),
                wannier90_x_bin.display(),
                seed
            ));
        }
    }

    for seed in seeds.iter() {
        let args = vec![seed.clone()];
        let log_name = format!("out.w90.{}.log", seed);
        println!("Command: {} {}", wannier90_x_bin.display(), seed);
        if let Err(err) = run_logged_command(&run_dir, &log_name, &wannier90_x_bin, &args) {
            let _ = write_text(&run_dir.join("status.txt"), "failed\n");
            let _ = write_text(&run_dir.join("exit_code.txt"), "wannier90-x-failed\n");
            return Err(err);
        }
    }

    match export_symmetry_metadata(&run_dir) {
        Ok(path) => {
            println!("Symmetry: {}", path.display());
        }
        Err(err) => {
            eprintln!(
                "warning: failed to export symmetry metadata for '{}': {}",
                STAGE_WANNIER, err
            );
        }
    }

    write_text(&run_dir.join("status.txt"), "success\n")?;
    write_text(&run_dir.join("exit_code.txt"), "0\n")?;
    println!("Stage '{}' completed successfully.", STAGE_WANNIER);
    println!("Run directory: {}", run_dir.display());
    Ok(run_dir)
}

fn ensure_stage_configured(config: &CaseConfig, stage: &str) -> CliResult<PathBuf> {
    let stage_dir = config
        .stage_dir(stage)
        .ok_or_else(|| format!("stage '{}' is disabled in configuration", stage))?
        .to_path_buf();

    if !stage_dir.is_dir() {
        return Err(format!(
            "stage '{}' is not configured; missing directory '{}'",
            stage,
            stage_dir.display()
        ));
    }

    Ok(stage_dir)
}

fn run_pipeline(config: &CaseConfig, opts: RunOptions) -> CliResult<()> {
    let stage_list = resolve_pipeline_stages(config, opts.stages.as_ref())?;
    if stage_list.is_empty() {
        return Err(format!(
            "no configured pipeline stages in '{}'",
            config.case_dir.display()
        ));
    }

    println!("Pipeline stages: {}", stage_list.join(","));

    let mut completed: HashMap<String, PathBuf> = HashMap::new();

    for stage in stage_list.iter() {
        let from_stage = if opts.from_scf.is_some() {
            opts.from_scf.clone()
        } else {
            infer_pipeline_source(stage, &completed)
        };

        match stage.as_str() {
            STAGE_SCF | STAGE_NSCF | STAGE_BANDS => {
                let run_dir = run_stage(config, stage, from_stage, opts.pw_bin.clone())?;
                completed.insert(stage.clone(), run_dir);
            }
            STAGE_WANNIER => {
                let mut stage_opts = opts.clone();
                stage_opts.from_scf = from_stage;
                let run_dir = run_wannier_stage(config, stage_opts)?;
                completed.insert(stage.clone(), run_dir);
            }
            other => return Err(format!("unsupported pipeline stage '{}'", other)),
        }
    }

    println!("Pipeline completed successfully.");
    Ok(())
}

fn infer_pipeline_source(stage: &str, completed: &HashMap<String, PathBuf>) -> Option<String> {
    match stage {
        STAGE_NSCF | STAGE_BANDS => completed
            .get(STAGE_SCF)
            .map(|path| path.display().to_string()),
        STAGE_WANNIER => completed
            .get(STAGE_NSCF)
            .or_else(|| completed.get(STAGE_SCF))
            .map(|path| path.display().to_string()),
        _ => None,
    }
}

fn resolve_pipeline_stages(
    config: &CaseConfig,
    stages_opt: Option<&Vec<String>>,
) -> CliResult<Vec<String>> {
    if let Some(stages) = stages_opt {
        let mut out = Vec::new();
        for stage in stages.iter() {
            if !is_run_stage(stage) {
                return Err(format!(
                    "invalid stage '{}' in --stages; allowed: {},{},{},{}",
                    stage, STAGE_SCF, STAGE_NSCF, STAGE_BANDS, STAGE_WANNIER
                ));
            }
            if !config.stage_enabled(stage) {
                return Err(format!(
                    "stage '{}' requested in --stages but is disabled in configuration",
                    stage
                ));
            }
            if !out.contains(stage) {
                out.push(stage.clone());
            }
        }
        return Ok(out);
    }

    Ok(config.pipeline_stages.clone())
}

fn build_stage_pw_command(stage: &str, pw_bin: &Path) -> (PathBuf, Vec<String>, String) {
    if stage_uses_resource_metrics(stage) {
        if let Some((time_bin, time_flag)) = resolve_time_wrapper() {
            let command_display = format!("{} {} {}", time_bin, time_flag, pw_bin.display());
            return (
                PathBuf::from(time_bin),
                vec![time_flag.to_string(), pw_bin.display().to_string()],
                command_display,
            );
        }
    }

    (
        pw_bin.to_path_buf(),
        Vec::new(),
        pw_bin.display().to_string(),
    )
}

fn stage_uses_resource_metrics(stage: &str) -> bool {
    matches!(stage, STAGE_SCF | STAGE_NSCF)
}

fn resolve_time_wrapper() -> Option<(&'static str, &'static str)> {
    let time_bin = "/usr/bin/time";
    if !Path::new(time_bin).is_file() {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        Some((time_bin, "-l"))
    }

    #[cfg(not(target_os = "macos"))]
    {
        Some((time_bin, "-v"))
    }
}

fn discover_seednames_from_win(run_dir: &Path) -> CliResult<Vec<String>> {
    let mut seeds = Vec::new();
    let entries = fs::read_dir(run_dir)
        .map_err(|err| format!("failed to read '{}': {}", run_dir.display(), err))?;

    for entry in entries {
        let path = entry
            .map_err(|err| format!("failed to inspect run directory entry: {}", err))?
            .path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|x| x.to_str()) != Some("win") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|x| x.to_str()) {
            seeds.push(stem.to_string());
        }
    }

    seeds.sort();
    seeds.dedup();
    if seeds.is_empty() {
        return Err(format!(
            "no '*.win' files generated in '{}' after w90-win",
            run_dir.display()
        ));
    }

    Ok(seeds)
}

fn run_logged_command(
    run_dir: &Path,
    log_name: &str,
    cmd: &Path,
    args: &[String],
) -> CliResult<()> {
    let log_path = run_dir.join(log_name);
    let log_file = File::create(&log_path).map_err(|err| {
        format!(
            "failed to create log file '{}': {}",
            log_path.display(),
            err
        )
    })?;
    let log_file_err = log_file
        .try_clone()
        .map_err(|err| format!("failed to clone log file handle: {}", err))?;

    let status = Command::new(cmd)
        .current_dir(run_dir)
        .args(args)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .status()
        .map_err(|err| {
            format!(
                "failed to start '{}' with args {:?}: {}",
                cmd.display(),
                args,
                err
            )
        })?;

    if status.success() {
        Ok(())
    } else {
        let exit_text = match status.code() {
            Some(code) => code.to_string(),
            None => "terminated-by-signal".to_string(),
        };
        Err(format!(
            "command '{}' with args {:?} failed (exit={}); see '{}'",
            cmd.display(),
            args,
            exit_text,
            log_path.display()
        ))
    }
}

fn apply_projection_overrides(run_dir: &Path) -> CliResult<()> {
    let proj_path = run_dir.join("in.proj");
    if !proj_path.is_file() {
        return Ok(());
    }

    let projection_lines = read_projection_override_lines(&proj_path)?;
    if projection_lines.is_empty() {
        return Err(format!(
            "'{}' is empty; expected at least one projection line like 'Si1:sp3'",
            proj_path.display()
        ));
    }

    let seeds = discover_seednames_from_win(run_dir)?;
    for seed in seeds.iter() {
        let win_path = run_dir.join(format!("{}.win", seed));
        if !win_path.is_file() {
            return Err(format!(
                "expected win file '{}' was not found after w90-win",
                win_path.display()
            ));
        }
        rewrite_win_projection_block(&win_path, &projection_lines)?;
    }

    Ok(())
}

fn read_projection_override_lines(path: &Path) -> CliResult<Vec<String>> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("failed to read '{}': {}", path.display(), err))?;

    let mut out = Vec::new();
    for raw in content.lines() {
        let line = strip_comments(raw);
        if line.is_empty() {
            continue;
        }
        out.push(line);
    }

    Ok(out)
}

fn rewrite_win_projection_block(win_path: &Path, projection_lines: &[String]) -> CliResult<()> {
    let content = fs::read_to_string(win_path)
        .map_err(|err| format!("failed to read '{}': {}", win_path.display(), err))?;

    let mut out = Vec::new();
    let mut in_block = false;
    let mut found_begin = false;
    let mut found_end = false;

    for line in content.lines() {
        let marker = line.trim();
        if marker.eq_ignore_ascii_case("begin projections") {
            found_begin = true;
            in_block = true;
            out.push(line.to_string());
            for proj in projection_lines.iter() {
                out.push(proj.to_string());
            }
            continue;
        }

        if in_block {
            if marker.eq_ignore_ascii_case("end projections") {
                found_end = true;
                in_block = false;
                out.push(line.to_string());
            }
            continue;
        }

        out.push(line.to_string());
    }

    if !found_begin || !found_end {
        return Err(format!(
            "win file '{}' does not contain a complete 'begin projections ... end projections' block",
            win_path.display()
        ));
    }

    let mut rewritten = out.join("\n");
    rewritten.push('\n');
    write_text(win_path, &rewritten)
}

fn print_status(config: &CaseConfig) -> CliResult<()> {
    println!("Case: {}", config.case_dir.display());
    if let Some(path) = config.config_file.as_ref() {
        println!("Config: {}", path.display());
    }

    for stage in CANONICAL_STAGES.iter() {
        if !config.stage_enabled(stage) {
            println!("  [{}] disabled", stage);
            continue;
        }

        let stage_dir = config
            .stage_dir(stage)
            .ok_or_else(|| format!("internal error: missing stage '{}': no directory", stage))?;

        if !stage_dir.is_dir() {
            println!(
                "  [{}] not configured (missing {})",
                stage,
                stage_dir.display()
            );
            continue;
        }

        let latest = latest_run_dir(&config.case_dir, stage)?;
        if let Some(run_dir) = latest {
            let status =
                read_trimmed(run_dir.join("status.txt")).unwrap_or_else(|| "unknown".to_string());
            println!("  [{}] latest: {}", stage, run_dir.display());
            println!("      status: {}", status);
        } else {
            println!("  [{}] configured, no runs yet", stage);
        }
    }

    if !config.pipeline_stages.is_empty() {
        println!(
            "Pipeline default stages: {}",
            config.pipeline_stages.join(",")
        );
    }

    Ok(())
}

fn collect_stage_inputs(config: &CaseConfig, stage: &str) -> CliResult<StageInputs> {
    let mut files = Vec::new();

    for filename in required_input_files(stage)?.iter() {
        let src = resolve_input(config, stage, filename).ok_or_else(|| {
            format!(
                "missing required input '{}' for stage '{}'",
                filename, stage
            )
        })?;
        files.push(((*filename).to_string(), src));
    }

    if let Some(ctrl_path) = resolve_input(config, stage, "in.ctrl") {
        if spin_file_required(&ctrl_path)? {
            let spin_src = resolve_input(config, stage, "in.spin").ok_or_else(|| {
                format!(
                    "stage '{}' requires 'in.spin' (spin detected in '{}')",
                    stage,
                    ctrl_path.display()
                )
            })?;
            files.push(("in.spin".to_string(), spin_src));
        } else if let Some(spin_src) = resolve_input(config, stage, "in.spin") {
            files.push(("in.spin".to_string(), spin_src));
        }
    }

    if stage == STAGE_WANNIER {
        if let Some(proj_src) = resolve_input(config, stage, "in.proj") {
            files.push(("in.proj".to_string(), proj_src));
        }
    }

    Ok(StageInputs { files })
}

fn resolve_input(config: &CaseConfig, stage: &str, filename: &str) -> Option<PathBuf> {
    if let Some(stage_dir) = config.stage_dir(stage) {
        let stage_path = stage_dir.join(filename);
        if stage_path.is_file() {
            return Some(stage_path);
        }
    }

    let common_path = config.common_dir.join(filename);
    if common_path.is_file() {
        return Some(common_path);
    }

    None
}

fn spin_file_required(ctrl_path: &Path) -> CliResult<bool> {
    let content = fs::read_to_string(ctrl_path)
        .map_err(|err| format!("failed to read '{}': {}", ctrl_path.display(), err))?;

    for raw_line in content.lines() {
        let line = strip_comments(raw_line).to_ascii_lowercase();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("spin_scheme") {
            let normalized = line.replace(' ', "");
            if normalized.contains("spin_scheme=nonspin") {
                return Ok(false);
            }
            if normalized.contains("spin_scheme=spin") {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

fn strip_comments(line: &str) -> String {
    let mut s = line;
    if let Some((head, _)) = s.split_once('!') {
        s = head;
    }
    if let Some((head, _)) = s.split_once('#') {
        s = head;
    }
    s.trim().to_string()
}

fn create_run_dir(case_dir: &Path, stage: &str) -> CliResult<PathBuf> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock error: {}", err))?
        .as_secs();
    let pid = std::process::id();

    for idx in 0..1000u32 {
        let dirname = if idx == 0 {
            format!("{}-{}", ts, pid)
        } else {
            format!("{}-{}-{}", ts, pid, idx)
        };
        let run_dir = case_dir.join("runs").join(stage).join(dirname);
        if run_dir.exists() {
            continue;
        }

        fs::create_dir_all(&run_dir).map_err(|err| {
            format!(
                "failed to create run directory '{}': {}",
                run_dir.display(),
                err
            )
        })?;
        return Ok(run_dir);
    }

    Err(format!(
        "failed to create unique run directory under '{}/runs/{}'",
        case_dir.display(),
        stage
    ))
}

fn copy_inputs(inputs: &StageInputs, run_dir: &Path) -> CliResult<()> {
    for (name, src) in inputs.files.iter() {
        let dst = run_dir.join(name);
        fs::copy(src, &dst).map_err(|err| {
            format!(
                "failed to copy '{}' to '{}': {}",
                src.display(),
                dst.display(),
                err
            )
        })?;
    }
    Ok(())
}

fn validate_pot_files(
    config: &CaseConfig,
    stage: &str,
    in_pot_path: &Path,
) -> CliResult<Vec<String>> {
    let mut errors = Vec::new();
    let entries = parse_in_pot_entries(in_pot_path)?;
    if entries.is_empty() {
        errors.push(format!(
            "input '{}' has no pseudopotential entries",
            in_pot_path.display()
        ));
        return Ok(errors);
    }

    for token in entries.iter() {
        if let Some(reason) = validate_pot_token(token) {
            errors.push(format!(
                "invalid pseudopotential '{}' in '{}': {}",
                token,
                in_pot_path.display(),
                reason
            ));
            continue;
        }

        if resolve_pot_source_file(config, stage, in_pot_path, token).is_none() {
            let stage_hint = config
                .stage_dir(stage)
                .map(|p| p.join("pot").display().to_string())
                .unwrap_or_else(|| format!("{}/{}", config.case_dir.display(), stage));
            errors.push(format!(
                "missing pseudopotential '{}' referenced by '{}' (looked in '{}', '{}', and '{}')",
                token,
                in_pot_path.display(),
                stage_hint,
                config.common_dir.join("pot").display(),
                in_pot_path
                    .parent()
                    .map(|p| p.join("pot").display().to_string())
                    .unwrap_or_else(|| in_pot_path.display().to_string())
            ));
        }
    }

    Ok(errors)
}

fn copy_pot_files(config: &CaseConfig, stage: &str, run_dir: &Path) -> CliResult<()> {
    let in_pot_path = resolve_input(config, stage, "in.pot")
        .ok_or_else(|| format!("missing required input 'in.pot' for stage '{}'", stage))?;
    let entries = parse_in_pot_entries(&in_pot_path)?;
    if entries.is_empty() {
        return Err(format!(
            "input '{}' has no pseudopotential entries",
            in_pot_path.display()
        ));
    }

    let run_pot_dir = run_dir.join("pot");
    fs::create_dir_all(&run_pot_dir).map_err(|err| {
        format!(
            "failed to create run pot directory '{}': {}",
            run_pot_dir.display(),
            err
        )
    })?;

    for token in entries.iter() {
        if let Some(reason) = validate_pot_token(token) {
            return Err(format!(
                "invalid pseudopotential '{}' in '{}': {}",
                token,
                in_pot_path.display(),
                reason
            ));
        }

        let src = resolve_pot_source_file(config, stage, &in_pot_path, token).ok_or_else(|| {
            format!(
                "missing pseudopotential '{}' referenced by '{}'",
                token,
                in_pot_path.display()
            )
        })?;

        let dst = run_pot_dir.join(token);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create pseudopotential directory '{}': {}",
                    parent.display(),
                    err
                )
            })?;
        }
        fs::copy(&src, &dst).map_err(|err| {
            format!(
                "failed to copy pseudopotential '{}' to '{}': {}",
                src.display(),
                dst.display(),
                err
            )
        })?;
    }

    Ok(())
}

fn parse_in_pot_entries(in_pot_path: &Path) -> CliResult<Vec<String>> {
    let content = fs::read_to_string(in_pot_path)
        .map_err(|err| format!("failed to read '{}': {}", in_pot_path.display(), err))?;

    let mut seen = HashSet::new();
    let mut out = Vec::new();

    for (idx, raw_line) in content.lines().enumerate() {
        let line = strip_comments(raw_line);
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 2 {
            return Err(format!(
                "invalid line {} in '{}': expected '<species> <pseudo-file>'",
                idx + 1,
                in_pot_path.display()
            ));
        }
        let token = fields[1].to_string();
        if seen.insert(token.clone()) {
            out.push(token);
        }
    }

    Ok(out)
}

fn validate_pot_token(token: &str) -> Option<&'static str> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Some("empty filename");
    }

    let path = Path::new(trimmed);
    if path.is_absolute() {
        return Some("absolute paths are not supported; use a filename under 'pot/'");
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Some("parent path segments ('..') are not supported; use a filename under 'pot/'");
    }
    if path.components().any(|c| matches!(c, Component::Prefix(_))) {
        return Some("path prefixes are not supported");
    }

    None
}

fn resolve_pot_source_file(
    config: &CaseConfig,
    stage: &str,
    in_pot_path: &Path,
    token: &str,
) -> Option<PathBuf> {
    let token_path = PathBuf::from(token);
    let mut candidates = Vec::new();

    if let Some(stage_dir) = config.stage_dir(stage) {
        candidates.push(stage_dir.join("pot").join(&token_path));
        candidates.push(stage_dir.join(&token_path));
    }

    candidates.push(config.common_dir.join("pot").join(&token_path));
    candidates.push(config.common_dir.join(&token_path));

    if let Some(parent) = in_pot_path.parent() {
        candidates.push(parent.join("pot").join(&token_path));
        candidates.push(parent.join(&token_path));
    }

    let mut seen = HashSet::new();
    for candidate in candidates.into_iter() {
        let key = candidate.display().to_string();
        if !seen.insert(key) {
            continue;
        }
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

fn resolve_stage_source(
    config: &CaseConfig,
    target_stage: &str,
    from_spec: Option<&str>,
) -> CliResult<ResolvedSource> {
    if let Some(spec) = from_spec {
        return resolve_explicit_source(config, target_stage, spec);
    }

    resolve_default_source(config, target_stage)
}

fn resolve_explicit_source(
    config: &CaseConfig,
    target_stage: &str,
    from_spec: &str,
) -> CliResult<ResolvedSource> {
    let spec = from_spec.trim();
    if spec.is_empty() {
        return Err("--from value must not be empty".to_string());
    }

    if spec == "latest" {
        return resolve_default_source(config, target_stage);
    }

    if let Some((stage, selector)) = spec.split_once(':') {
        if selector == "latest" {
            if !is_run_stage(stage) {
                return Err(format!(
                    "invalid --from stage '{}' in '{}'; allowed: {},{},{},{}",
                    stage, spec, STAGE_SCF, STAGE_NSCF, STAGE_BANDS, STAGE_WANNIER
                ));
            }
            return resolve_latest_stage_source(config, stage);
        }
    }

    let mut path = PathBuf::from(spec);
    if !path.is_absolute() && !path.is_dir() {
        let case_relative = config.case_dir.join(spec);
        if case_relative.is_dir() {
            path = case_relative;
        }
    }

    if path.is_dir() {
        Ok(ResolvedSource {
            stage: infer_stage_from_run_dir(&path),
            run_dir: path,
        })
    } else {
        Err(format!("--from path '{}' does not exist", path.display()))
    }
}

fn resolve_default_source(config: &CaseConfig, target_stage: &str) -> CliResult<ResolvedSource> {
    match target_stage {
        STAGE_NSCF | STAGE_BANDS => resolve_latest_stage_source(config, STAGE_SCF),
        STAGE_WANNIER => {
            if let Some(nscf_latest) = latest_run_dir(&config.case_dir, STAGE_NSCF)? {
                return Ok(ResolvedSource {
                    stage: Some(STAGE_NSCF.to_string()),
                    run_dir: nscf_latest,
                });
            }
            resolve_latest_stage_source(config, STAGE_SCF)
        }
        _ => Err(format!(
            "stage '{}' does not consume prior SCF/NSCF outputs",
            target_stage
        )),
    }
}

fn resolve_latest_stage_source(config: &CaseConfig, stage: &str) -> CliResult<ResolvedSource> {
    let latest = latest_run_dir(&config.case_dir, stage)?.ok_or_else(|| {
        format!(
            "no runs found under '{}/runs/{}'",
            config.case_dir.display(),
            stage
        )
    })?;
    Ok(ResolvedSource {
        stage: Some(stage.to_string()),
        run_dir: latest,
    })
}

fn infer_stage_from_run_dir(path: &Path) -> Option<String> {
    let stage = path.parent()?.file_name()?.to_str()?;
    if is_run_stage(stage) {
        Some(stage.to_string())
    } else {
        None
    }
}

fn copy_source_outputs(
    scf_run: &Path,
    target_run: &Path,
    opts: SourceCopyOptions,
) -> CliResult<()> {
    let entries = fs::read_dir(scf_run).map_err(|err| {
        format!(
            "failed to read source run directory '{}': {}",
            scf_run.display(),
            err
        )
    })?;

    let mut copied_rho = 0usize;
    let mut copied_wfc = 0usize;
    let mut copied_eig = 0usize;
    let mut copied_any = 0usize;

    for entry in entries {
        let entry =
            entry.map_err(|err| format!("failed to inspect source output entry: {}", err))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let name = match path.file_name().and_then(|x| x.to_str()) {
            Some(v) => v,
            None => continue,
        };

        let take_rho = opts.rho && name.starts_with("out.scf.rho");
        let take_wfc = opts.wfc && name.starts_with("out.wfc");
        let take_eig = opts.eig && name.ends_with(".eig");
        if !(take_rho || take_wfc || take_eig) {
            continue;
        }

        let dst = target_run.join(name);
        fs::copy(&path, &dst).map_err(|err| {
            format!(
                "failed to copy source output '{}' to '{}': {}",
                path.display(),
                dst.display(),
                err
            )
        })?;

        copied_any += 1;
        if take_rho {
            copied_rho += 1;
        }
        if take_wfc {
            copied_wfc += 1;
        }
        if take_eig {
            copied_eig += 1;
        }
    }

    if copied_any == 0 {
        return Err(format!(
            "no required source files found in '{}'",
            scf_run.display()
        ));
    }

    if opts.rho && copied_rho == 0 {
        return Err(format!(
            "no charge-density file out.scf.rho* found in '{}'",
            scf_run.display()
        ));
    }
    if opts.wfc && copied_wfc == 0 {
        return Err(format!(
            "no wavefunction file out.wfc* found in '{}'",
            scf_run.display()
        ));
    }
    if opts.eig && copied_eig == 0 {
        return Err(format!(
            "no Wannier eigenvalue file '*.eig' found in '{}' (run source with wannier90_export=true)",
            scf_run.display()
        ));
    }

    Ok(())
}

fn write_source_metadata(run_dir: &Path, source: &ResolvedSource) -> CliResult<()> {
    write_text(
        &run_dir.join("from_source_run.txt"),
        &format!("{}\n", source.run_dir.display()),
    )?;

    if let Some(stage) = source.stage.as_ref() {
        write_text(&run_dir.join("from_stage.txt"), &format!("{}\n", stage))?;
        if stage == STAGE_SCF {
            write_text(
                &run_dir.join("from_scf_run.txt"),
                &format!("{}\n", source.run_dir.display()),
            )?;
        }
    }

    Ok(())
}

fn resolve_pw_bin(bin_arg: Option<PathBuf>, config: &CaseConfig) -> PathBuf {
    if let Some(p) = bin_arg {
        return p;
    }

    if let Some(path) = yaml_binary_to_path(&config.case_dir, config.binaries.pw.as_deref()) {
        return path;
    }

    if let Ok(env_bin) = env::var("DW_PW_BIN") {
        if !env_bin.trim().is_empty() {
            return PathBuf::from(env_bin);
        }
    }

    if let Some(found) = resolve_workspace_binary(&config.case_dir, "pw") {
        return found;
    }

    PathBuf::from("pw")
}

fn resolve_w90_win_bin(bin_arg: Option<PathBuf>, config: &CaseConfig) -> PathBuf {
    if let Some(p) = bin_arg {
        return p;
    }

    if let Some(path) = yaml_binary_to_path(&config.case_dir, config.binaries.w90_win.as_deref()) {
        return path;
    }

    if let Ok(env_bin) = env::var("DW_W90_WIN_BIN") {
        if !env_bin.trim().is_empty() {
            return PathBuf::from(env_bin);
        }
    }

    if let Some(found) = resolve_workspace_binary(&config.case_dir, "w90-win") {
        return found;
    }

    PathBuf::from("w90-win")
}

fn resolve_w90_amn_bin(bin_arg: Option<PathBuf>, config: &CaseConfig) -> PathBuf {
    if let Some(p) = bin_arg {
        return p;
    }

    if let Some(path) = yaml_binary_to_path(&config.case_dir, config.binaries.w90_amn.as_deref()) {
        return path;
    }

    if let Ok(env_bin) = env::var("DW_W90_AMN_BIN") {
        if !env_bin.trim().is_empty() {
            return PathBuf::from(env_bin);
        }
    }

    if let Some(found) = resolve_workspace_binary(&config.case_dir, "w90-amn") {
        return found;
    }

    PathBuf::from("w90-amn")
}

fn resolve_wannier90_x_bin(bin_arg: Option<PathBuf>, config: &CaseConfig) -> PathBuf {
    if let Some(p) = bin_arg {
        return p;
    }

    if let Some(path) =
        yaml_binary_to_path(&config.case_dir, config.binaries.wannier90_x.as_deref())
    {
        return path;
    }

    if let Ok(env_bin) = env::var("DW_WANNIER90_X_BIN") {
        if !env_bin.trim().is_empty() {
            return PathBuf::from(env_bin);
        }
    }

    PathBuf::from("wannier90.x")
}

fn yaml_binary_to_path(case_dir: &Path, value: Option<&str>) -> Option<PathBuf> {
    let raw = value?.trim();
    if raw.is_empty() {
        return None;
    }

    let path = PathBuf::from(raw);
    if path.is_absolute() {
        Some(path)
    } else if raw.contains('/') {
        Some(case_dir.join(path))
    } else {
        Some(path)
    }
}

fn ensure_binary_available(bin: &Path, logical: &str, cli_hint: &str) -> CliResult<()> {
    if resolve_binary_path(bin).is_some() {
        return Ok(());
    }

    if logical == "wannier90.x" {
        return Err(format!(
            "required executable '{}' was not found. Install Wannier90 (providing 'wannier90.x') and ensure it is in PATH, or pass {}",
            bin.display(),
            cli_hint
        ));
    }

    Err(format!(
        "required executable '{}' for '{}' was not found. Ensure it is installed and in PATH, or pass {}",
        bin.display(),
        logical,
        cli_hint
    ))
}

fn resolve_binary_path(bin: &Path) -> Option<PathBuf> {
    let has_path_component = bin.components().nth(1).is_some()
        || bin
            .to_string_lossy()
            .chars()
            .any(|ch| ch == '/' || ch == '\\');

    if bin.is_absolute() || has_path_component {
        if bin.is_file() {
            return Some(bin.to_path_buf());
        }
        return None;
    }

    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(bin);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

fn resolve_workspace_binary(case_dir: &Path, bin_name: &str) -> Option<PathBuf> {
    if let Some(ws_root) = find_workspace_root(case_dir) {
        let release_bin = ws_root.join("target").join("release").join(bin_name);
        if release_bin.is_file() {
            return Some(release_bin);
        }

        let debug_bin = ws_root.join("target").join("debug").join(bin_name);
        if debug_bin.is_file() {
            return Some(debug_bin);
        }
    }

    None
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    let mut current = Some(start);
    while let Some(path) = current {
        let cargo_toml = path.join("Cargo.toml");
        if cargo_toml.is_file() {
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return Some(path.to_path_buf());
                }
            }
        }
        current = path.parent();
    }
    None
}

fn export_symmetry_metadata(run_dir: &Path) -> CliResult<PathBuf> {
    let crystal_path = run_dir.join("in.crystal");
    if !crystal_path.is_file() {
        return Err(format!(
            "missing required input '{}' for symmetry export",
            crystal_path.display()
        ));
    }

    let structure = load_structure_for_symmetry(&crystal_path)?;
    let detected = detect_symmetry(&structure, DetectOptions::default())
        .map_err(|err| format!("symmetry detection failed: {}", err))?;
    let class = classify_symmetry(&detected.operations)
        .map_err(|err| format!("symmetry classification failed: {}", err))?;

    let properties_dir = run_dir.join("properties");
    fs::create_dir_all(&properties_dir).map_err(|err| {
        format!(
            "failed to create properties directory '{}': {}",
            properties_dir.display(),
            err
        )
    })?;

    let out_path = properties_dir.join("symmetry_ops.json");
    let content = format_symmetry_json(&detected, &class);
    write_text(&out_path, &content)?;
    Ok(out_path)
}

fn load_structure_for_symmetry(crystal_path: &Path) -> CliResult<Structure> {
    let crystal_path_str = crystal_path.to_string_lossy().to_string();
    let mut crystal = Crystal::new();
    let parse_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crystal.read_file(&crystal_path_str);
    }));
    if parse_result.is_err() {
        return Err(format!(
            "failed to parse crystal structure '{}'",
            crystal_path.display()
        ));
    }

    let lattice = crystal.get_latt().as_2d_array_col_major();
    let positions = crystal
        .get_atom_positions()
        .iter()
        .map(|v| [v.x, v.y, v.z])
        .collect::<Vec<_>>();
    let atom_types = crystal.get_atom_types();

    Ok(Structure {
        lattice,
        positions,
        atom_types,
    })
}

fn format_symmetry_json(
    detected: &symops::DetectedSymmetry,
    class: &symops::SymmetryClassification,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{{");
    let _ = writeln!(out, "  \"schema\": \"symmetry-ops-v1\",");
    let _ = writeln!(
        out,
        "  \"n_candidate_rotations\": {},",
        detected.candidate_rotations
    );
    let _ = writeln!(out, "  \"n_operations\": {},", detected.operations.len());
    let _ = writeln!(out, "  \"classification\": {{");
    let _ = writeln!(
        out,
        "    \"crystal_system\": \"{}\",",
        crystal_system_name(class.crystal_system)
    );
    let _ = writeln!(
        out,
        "    \"point_group_hint\": \"{}\",",
        class.point_group_hint
    );
    let _ = writeln!(out, "    \"has_inversion\": {},", class.has_inversion);
    let _ = writeln!(
        out,
        "    \"n_proper_rotations\": {},",
        class.n_proper_rotations
    );
    let _ = writeln!(
        out,
        "    \"max_rotation_order\": {},",
        class.max_rotation_order
    );
    match class.space_group_number {
        Some(number) => {
            let _ = writeln!(out, "    \"space_group_number\": {}", number);
        }
        None => {
            let _ = writeln!(out, "    \"space_group_number\": null");
        }
    }
    let _ = writeln!(out, "  }},");
    let _ = writeln!(out, "  \"operations\": [");

    for (idx, op) in detected.operations.iter().enumerate() {
        let r = op.rotation();
        let t = op.translation();
        let _ = writeln!(out, "    {{");
        let _ = writeln!(out, "      \"index\": {},", idx);
        let _ = writeln!(out, "      \"determinant\": {},", symops::determinant(*r));
        let _ = writeln!(out, "      \"rotation\": [");
        let _ = writeln!(out, "        [{}, {}, {}],", r[0][0], r[0][1], r[0][2]);
        let _ = writeln!(out, "        [{}, {}, {}],", r[1][0], r[1][1], r[1][2]);
        let _ = writeln!(out, "        [{}, {}, {}]", r[2][0], r[2][1], r[2][2]);
        let _ = writeln!(out, "      ],");
        let _ = writeln!(
            out,
            "      \"translation\": [{:.15}, {:.15}, {:.15}]",
            t[0], t[1], t[2]
        );
        if idx + 1 == detected.operations.len() {
            let _ = writeln!(out, "    }}");
        } else {
            let _ = writeln!(out, "    }},");
        }
    }

    let _ = writeln!(out, "  ]");
    let _ = writeln!(out, "}}");
    out
}

fn crystal_system_name(system: CrystalSystem) -> &'static str {
    match system {
        CrystalSystem::Triclinic => "triclinic",
        CrystalSystem::Monoclinic => "monoclinic",
        CrystalSystem::Orthorhombic => "orthorhombic",
        CrystalSystem::Tetragonal => "tetragonal",
        CrystalSystem::Trigonal => "trigonal",
        CrystalSystem::Hexagonal => "hexagonal",
        CrystalSystem::Cubic => "cubic",
    }
}

fn latest_run_dir(case_dir: &Path, stage: &str) -> CliResult<Option<PathBuf>> {
    let stage_runs_dir = case_dir.join("runs").join(stage);
    if !stage_runs_dir.is_dir() {
        return Ok(None);
    }

    let dirs: Vec<PathBuf> = fs::read_dir(&stage_runs_dir)
        .map_err(|err| format!("failed to read '{}': {}", stage_runs_dir.display(), err))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();

    Ok(dirs.into_iter().max_by_key(run_dir_rank))
}

fn run_dir_rank(path: &PathBuf) -> (u64, String) {
    let name = path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_string();
    let ts = name
        .split('-')
        .next()
        .and_then(|x| x.parse::<u64>().ok())
        .unwrap_or(0u64);

    (ts, name)
}

fn read_trimmed(path: PathBuf) -> Option<String> {
    fs::read_to_string(path).ok().map(|x| x.trim().to_string())
}

fn write_text(path: &Path, content: &str) -> CliResult<()> {
    let mut file = File::create(path)
        .map_err(|err| format!("failed to create '{}': {}", path.display(), err))?;
    file.write_all(content.as_bytes())
        .map_err(|err| format!("failed to write '{}': {}", path.display(), err))?;
    Ok(())
}
