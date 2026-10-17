// Command-line wrapper around pw/src/memory_estimate.rs (shared with `pw --dry-run`).
#[path = "../memory_estimate.rs"]
mod memory_estimate;

use control::{Control, KptsScheme, SpinScheme, XcScheme};
use crystal::Crystal;
use kpts::KPTS;
use memory_estimate::{ComponentBreakdown, EstimateReport};
use pspot::PSPot;
use std::env;
use std::fmt::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct CliOptions {
    case_dir: PathBuf,
    ranks_override: Option<usize>,
    json: bool,
}

#[derive(Debug)]
enum EstimateError {
    Unsupported(String),
    Fatal(String),
}

enum ParseAction {
    Help,
    Run(CliOptions),
}

fn main() {
    let _ = dwmpi::init();
    let exit_code = run_main();
    let _ = dwmpi::barrier(dwmpi::comm_world());
    let _ = dwmpi::finalize();
    std::process::exit(exit_code);
}

fn run_main() -> i32 {
    let action = match parse_cli() {
        Ok(action) => action,
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("error: {}", err);
                print_help();
            }
            return 1;
        }
    };

    match action {
        ParseAction::Help => {
            if dwmpi::is_root() {
                print_help();
            }
            0
        }
        ParseAction::Run(options) => match estimate_case(&options) {
            Ok(report) => {
                if dwmpi::is_root() {
                    if options.json {
                        println!("{}", render_json(&report));
                    } else {
                        memory_estimate::print_human_report(&report);
                    }
                }
                0
            }
            Err(EstimateError::Unsupported(message)) => {
                if dwmpi::is_root() {
                    if options.json {
                        println!("{}", render_status_json("unsupported", &message));
                    } else {
                        eprintln!("status = unsupported");
                        eprintln!("{}", message);
                    }
                }
                2
            }
            Err(EstimateError::Fatal(message)) => {
                if dwmpi::is_root() {
                    if options.json {
                        println!("{}", render_status_json("error", &message));
                    } else {
                        eprintln!("status = error");
                        eprintln!("{}", message);
                    }
                }
                1
            }
        },
    }
}

fn parse_cli() -> Result<ParseAction, String> {
    let args = env::args().collect::<Vec<String>>();
    if args.len() == 1 {
        return Err("missing required arguments".to_string());
    }

    let mut case_dir: Option<PathBuf> = None;
    let mut ranks_override: Option<usize> = None;
    let mut json = false;
    let mut idx = 1usize;

    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Ok(ParseAction::Help),
            "--json" => {
                json = true;
                idx += 1;
            }
            "--case" => {
                if idx + 1 >= args.len() {
                    return Err("expected a path after --case".to_string());
                }
                case_dir = Some(PathBuf::from(args[idx + 1].clone()));
                idx += 2;
            }
            "--ranks" => {
                if idx + 1 >= args.len() {
                    return Err("expected an integer after --ranks".to_string());
                }
                let parsed = args[idx + 1]
                    .parse::<usize>()
                    .map_err(|_| format!("invalid value for --ranks: '{}'", args[idx + 1]))?;
                if parsed == 0 {
                    return Err("invalid value for --ranks: must be >= 1".to_string());
                }
                ranks_override = Some(parsed);
                idx += 2;
            }
            other => {
                return Err(format!("unrecognized argument: '{}'", other));
            }
        }
    }

    let case_dir = case_dir.ok_or_else(|| "missing required argument: --case <dir>".to_string())?;

    Ok(ParseAction::Run(CliOptions {
        case_dir,
        ranks_override,
        json,
    }))
}

fn print_help() {
    println!("Input-driven runtime memory estimator for `pw`.");
    println!();
    println!("Usage:");
    println!("  cargo run -p pw --bin memory_estimate -- --case <dir> [--ranks <N>] [--json]");
    println!();
    println!("Required:");
    println!(
        "  --case <dir>      Directory containing input files (in.ctrl/in.crystal/in.pot/in.kmesh)"
    );
    println!();
    println!("Optional:");
    println!("  --ranks <N>       Override rank count for per-rank/global estimates");
    println!("  --json            Emit machine-readable JSON");
}

fn estimate_case(options: &CliOptions) -> Result<EstimateReport, EstimateError> {
    let case_dir = resolve_case_dir(&options.case_dir)?;
    with_case_dir(case_dir.as_path(), || {
        ensure_file_exists(Path::new("in.ctrl"))?;
        ensure_file_exists(Path::new("in.crystal"))?;
        ensure_file_exists(Path::new("in.pot"))?;

        let control = Control::from_file("in.ctrl")
            .map_err(|err| EstimateError::Fatal(format!("failed to load control file: {}", err)))?;
        control.validate_capability_matrix().map_err(|err| {
            EstimateError::Unsupported(format!("runtime capability preflight failed: {}", err))
        })?;

        let k_input = match control.get_kpts_scheme_enum() {
            KptsScheme::Kmesh => "in.kmesh",
            KptsScheme::Kline => "in.kline",
        };
        ensure_file_exists(Path::new(k_input))?;

        let spin_scheme = control.get_spin_scheme_enum();
        if matches!(spin_scheme, SpinScheme::Ncl) {
            return Err(EstimateError::Unsupported(
                "unsupported capability: spin_scheme='ncl' is not implemented for runtime memory estimate"
                    .to_string(),
            ));
        }

        let mut crystal = Crystal::new();
        crystal
            .try_read_file("in.crystal")
            .map_err(|err| EstimateError::Fatal(format!("failed to load crystal file: {}", err)))?;

        let pots = PSPot::try_new(control.get_pot_scheme_enum()).map_err(|err| {
            EstimateError::Fatal(format!("failed to load pseudopotentials: {}", err))
        })?;
        for species in crystal.get_unique_species().iter() {
            if !pots.contains_species(species) {
                return Err(EstimateError::Fatal(format!(
                    "failed to load pseudopotentials: species '{}' appears in in.crystal but has no mapping in in.pot",
                    species
                )));
            }
        }
        let kpts = kpts::try_new(
            control.get_kpts_scheme_enum(),
            &crystal,
            control.get_symmetry(),
        )
        .map_err(|err| EstimateError::Fatal(format!("failed to initialize k-points: {}", err)))?;

        validate_runtime_constraints(&control, kpts.as_ref())?;

        memory_estimate::estimate(
            &control,
            &crystal,
            &pots,
            kpts.as_ref(),
            options.ranks_override,
            &case_dir.display().to_string(),
        )
        .map_err(EstimateError::Fatal)
    })
}

fn with_case_dir<T>(
    case_dir: &Path,
    f: impl FnOnce() -> Result<T, EstimateError>,
) -> Result<T, EstimateError> {
    let original_dir = env::current_dir().map_err(|err| {
        EstimateError::Fatal(format!("failed to read current directory: {}", err))
    })?;
    env::set_current_dir(case_dir).map_err(|err| {
        EstimateError::Fatal(format!(
            "failed to set current directory to '{}': {}",
            case_dir.display(),
            err
        ))
    })?;

    let result = f();
    let restore_result = env::set_current_dir(original_dir.as_path());

    match (result, restore_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(err), Ok(())) => Err(err),
        (_, Err(err)) => Err(EstimateError::Fatal(format!(
            "failed to restore working directory to '{}': {}",
            original_dir.display(),
            err
        ))),
    }
}

fn resolve_case_dir(case_dir: &Path) -> Result<PathBuf, EstimateError> {
    let absolute = if case_dir.is_absolute() {
        case_dir.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|err| {
                EstimateError::Fatal(format!("failed to read current directory: {}", err))
            })?
            .join(case_dir)
    };

    if !absolute.exists() {
        return Err(EstimateError::Fatal(format!(
            "case directory does not exist: '{}'",
            absolute.display()
        )));
    }
    if !absolute.is_dir() {
        return Err(EstimateError::Fatal(format!(
            "case path is not a directory: '{}'",
            absolute.display()
        )));
    }

    Ok(absolute)
}

fn ensure_file_exists(path: &Path) -> Result<(), EstimateError> {
    if path.exists() && path.is_file() {
        return Ok(());
    }
    Err(EstimateError::Fatal(format!(
        "required input file is missing: '{}'",
        path.display()
    )))
}

fn validate_runtime_constraints(control: &Control, kpts: &dyn KPTS) -> Result<(), EstimateError> {
    if !matches!(control.get_xc_scheme_enum(), XcScheme::Hse06) {
        return Ok(());
    }

    if kpts.get_n_kpts() != 1 {
        return Err(EstimateError::Unsupported(
            "unsupported capability: xc_scheme='hse06' currently requires exactly one k-point"
                .to_string(),
        ));
    }

    let k0 = kpts.get_k_frac(0);
    if k0.norm() > 1.0E-10 {
        return Err(EstimateError::Unsupported(
            "unsupported capability: xc_scheme='hse06' currently requires Gamma-only k=(0,0,0)"
                .to_string(),
        ));
    }

    Ok(())
}

fn render_json(report: &EstimateReport) -> String {
    let mut rank_totals = report
        .rank_estimates
        .iter()
        .map(|rank| rank.components.total())
        .collect::<Vec<usize>>();
    rank_totals.sort_unstable();
    let per_rank_min = rank_totals.first().copied().unwrap_or(0);
    let per_rank_max = rank_totals.last().copied().unwrap_or(0);
    let per_rank_sum = rank_totals
        .iter()
        .copied()
        .fold(0usize, |acc, v| acc.saturating_add(v));
    let per_rank_mean = if report.rank_estimates.is_empty() {
        0usize
    } else {
        per_rank_sum / report.rank_estimates.len()
    };

    let peak_rank = report
        .rank_estimates
        .iter()
        .max_by_key(|rank| rank.components.total())
        .unwrap();

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"status\": \"ok\",\n");
    writeln!(
        &mut out,
        "  \"case_dir\": \"{}\",",
        json_escape(&report.case_dir)
    )
    .unwrap();
    writeln!(&mut out, "  \"task\": \"{}\",", json_escape(&report.task)).unwrap();
    writeln!(
        &mut out,
        "  \"spin_scheme\": \"{}\",",
        report.spin_scheme.as_str()
    )
    .unwrap();
    writeln!(
        &mut out,
        "  \"kpoint_schedule\": \"{}\",",
        report.kpoint_schedule_mode.as_str()
    )
    .unwrap();
    writeln!(&mut out, "  \"mpi_world_size\": {},", report.mpi_world_size).unwrap();
    writeln!(&mut out, "  \"assumed_ranks\": {},", report.ranks).unwrap();
    writeln!(&mut out, "  \"estimated_bytes_total\": {},", per_rank_max).unwrap();
    writeln!(
        &mut out,
        "  \"estimated_bytes_total_per_rank_peak\": {},",
        per_rank_max
    )
    .unwrap();
    writeln!(
        &mut out,
        "  \"estimated_bytes_total_per_rank_min\": {},",
        per_rank_min
    )
    .unwrap();
    writeln!(
        &mut out,
        "  \"estimated_bytes_total_per_rank_mean\": {},",
        per_rank_mean
    )
    .unwrap();
    writeln!(
        &mut out,
        "  \"estimated_bytes_total_global\": {},",
        report.global_components.total()
    )
    .unwrap();
    writeln!(&mut out, "  \"peak_rank\": {},", peak_rank.rank).unwrap();
    writeln!(
        &mut out,
        "  \"peak_rank_local_kpoints\": {},",
        peak_rank.nk_local
    )
    .unwrap();
    out.push_str("  \"dimensions\": {\n");
    writeln!(
        &mut out,
        "    \"fft_shape\": [{}, {}, {}],",
        report.fft_shape[0], report.fft_shape[1], report.fft_shape[2]
    )
    .unwrap();
    writeln!(&mut out, "    \"nfft\": {},", report.nfft).unwrap();
    writeln!(&mut out, "    \"npw_rho\": {},", report.npw_rho).unwrap();
    writeln!(&mut out, "    \"nkpt\": {},", report.nkpt).unwrap();
    writeln!(&mut out, "    \"nband\": {}", report.nband).unwrap();
    out.push_str("  },\n");
    out.push_str("  \"components_per_rank_peak\": ");
    append_component_json(&mut out, peak_rank.components, 2);
    out.push_str(",\n");
    out.push_str("  \"components_global\": ");
    append_component_json(&mut out, report.global_components, 2);
    out.push_str(",\n");
    out.push_str("  \"rank_totals\": [");
    for (idx, rank) in report.rank_estimates.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        write!(&mut out, "{}", rank.components.total()).unwrap();
    }
    out.push_str("],\n");
    out.push_str("  \"rank_components\": [\n");
    for (idx, rank) in report.rank_estimates.iter().enumerate() {
        out.push_str("    {\n");
        writeln!(&mut out, "      \"rank\": {},", rank.rank).unwrap();
        writeln!(&mut out, "      \"local_kpoints\": {},", rank.nk_local).unwrap();
        writeln!(
            &mut out,
            "      \"estimated_bytes_total\": {},",
            rank.components.total()
        )
        .unwrap();
        out.push_str("      \"components\": ");
        append_component_json(&mut out, rank.components, 3);
        out.push_str("\n    }");
        if idx + 1 != report.rank_estimates.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ],\n");
    out.push_str("  \"assumptions\": [");
    for (idx, note) in report.assumptions.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        write!(&mut out, "\"{}\"", json_escape(note)).unwrap();
    }
    out.push_str("]\n");
    out.push_str("}");
    out
}

fn append_component_json(out: &mut String, components: ComponentBreakdown, indent_level: usize) {
    let indent = "  ".repeat(indent_level);
    let inner = "  ".repeat(indent_level + 1);
    out.push_str("{\n");
    writeln!(
        out,
        "{}\"fft_real_space_arrays\": {},",
        inner, components.fft_real_space_arrays
    )
    .unwrap();
    writeln!(
        out,
        "{}\"gvectors_and_plane_wave_bases\": {},",
        inner, components.gvectors_and_plane_wave_bases
    )
    .unwrap();
    writeln!(
        out,
        "{}\"wavefunction_eigen_storage\": {},",
        inner, components.wavefunction_eigen_storage
    )
    .unwrap();
    writeln!(
        out,
        "{}\"density_potential_workspaces\": {},",
        inner, components.density_potential_workspaces
    )
    .unwrap();
    writeln!(
        out,
        "{}\"nonlocal_projector_caches\": {},",
        inner, components.nonlocal_projector_caches
    )
    .unwrap();
    writeln!(
        out,
        "{}\"runtime_process_overhead\": {},",
        inner, components.runtime_process_overhead
    )
    .unwrap();
    writeln!(
        out,
        "{}\"estimated_bytes_total\": {}",
        inner,
        components.total()
    )
    .unwrap();
    write!(out, "{}}}", indent).unwrap();
}

fn json_escape(inp: &str) -> String {
    let mut escaped = String::with_capacity(inp.len());
    for ch in inp.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn render_status_json(status: &str, message: &str) -> String {
    format!(
        "{{\"status\":\"{}\",\"message\":\"{}\"}}",
        json_escape(status),
        json_escape(message)
    )
}
