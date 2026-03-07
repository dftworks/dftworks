#![allow(warnings)]

use control::{Control, KPointScheduleScheme, KptsScheme, SpinScheme, XcScheme};
use crystal::Crystal;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use kpts_distribution::{KPointScheduleMode, KPointSchedulePlan};
use kscf::KSCF;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use std::cmp::{max, min};
use std::env;
use std::fmt::Write;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use types::{c64, Vector3f64, Vector3i32};
use vnl::VNL;

#[derive(Debug, Clone)]
struct CliOptions {
    case_dir: PathBuf,
    ranks_override: Option<usize>,
    json: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct ComponentBreakdown {
    fft_real_space_arrays: usize,
    gvectors_and_plane_wave_bases: usize,
    wavefunction_eigen_storage: usize,
    density_potential_workspaces: usize,
    nonlocal_projector_caches: usize,
    runtime_process_overhead: usize,
}

impl ComponentBreakdown {
    fn total(self) -> usize {
        self.fft_real_space_arrays
            .saturating_add(self.gvectors_and_plane_wave_bases)
            .saturating_add(self.wavefunction_eigen_storage)
            .saturating_add(self.density_potential_workspaces)
            .saturating_add(self.nonlocal_projector_caches)
            .saturating_add(self.runtime_process_overhead)
    }

    fn add_assign(&mut self, other: Self) {
        self.fft_real_space_arrays = self
            .fft_real_space_arrays
            .saturating_add(other.fft_real_space_arrays);
        self.gvectors_and_plane_wave_bases = self
            .gvectors_and_plane_wave_bases
            .saturating_add(other.gvectors_and_plane_wave_bases);
        self.wavefunction_eigen_storage = self
            .wavefunction_eigen_storage
            .saturating_add(other.wavefunction_eigen_storage);
        self.density_potential_workspaces = self
            .density_potential_workspaces
            .saturating_add(other.density_potential_workspaces);
        self.nonlocal_projector_caches = self
            .nonlocal_projector_caches
            .saturating_add(other.nonlocal_projector_caches);
        self.runtime_process_overhead = self
            .runtime_process_overhead
            .saturating_add(other.runtime_process_overhead);
    }
}

#[derive(Debug, Clone)]
struct RankEstimate {
    rank: usize,
    nk_local: usize,
    components: ComponentBreakdown,
}

#[derive(Debug, Clone)]
struct KPointEstimate {
    npw: usize,
    pwbasis_bytes: usize,
    vnl_bytes: usize,
    shared_cache_bytes: usize,
}

#[derive(Debug, Clone)]
struct EstimateReport {
    case_dir: String,
    task: String,
    spin_scheme: SpinScheme,
    kpoint_schedule_mode: KPointScheduleMode,
    mpi_world_size: usize,
    ranks: usize,
    fft_shape: [usize; 3],
    nfft: usize,
    npw_rho: usize,
    nkpt: usize,
    nband: usize,
    rank_estimates: Vec<RankEstimate>,
    global_components: ComponentBreakdown,
    assumptions: Vec<String>,
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

const CALIBRATED_PROCESS_OVERHEAD_BYTES: usize = 32 * 1024 * 1024;

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
                        print_human_report(&report);
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
        crystal.read_file("in.crystal");

        let pots = PSPot::new(control.get_pot_scheme_enum());
        let kpts = kpts::try_new(
            control.get_kpts_scheme_enum(),
            &crystal,
            control.get_symmetry(),
        )
        .map_err(|err| EstimateError::Fatal(format!("failed to initialize k-points: {}", err)))?;

        validate_runtime_constraints(&control, kpts.as_ref())?;

        let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
        let fft_shape = fftgrid.get_size();
        let nfft = fftgrid.get_ntot();

        let gvec = GVector::new(crystal.get_latt(), fft_shape[0], fft_shape[1], fft_shape[2]);
        let pwden = PWDensity::new(control.get_ecutrho(), &gvec);
        let npw_rho = pwden.get_n_plane_waves();
        let blatt = crystal.get_latt().reciprocal();

        let nkpt = kpts.get_n_kpts();
        let nband = control.get_nband();
        let hubbard_n_m = estimate_hubbard_n_m(&control)?;
        let spin_channels = spin_channel_count(spin_scheme);

        let mut kpoint_estimates = Vec::<KPointEstimate>::with_capacity(nkpt);
        let mut kpoint_costs = Vec::<u64>::with_capacity(nkpt);

        for ik in 0..nkpt {
            let k_frac = kpts.get_k_frac(ik);
            let k_cart = kpts.frac_to_cart(&k_frac, &blatt);
            let pwwfc = PWBasis::new(k_cart, ik, control.get_ecut(), &gvec);
            let npw = pwwfc.get_n_plane_waves();

            let cost = npw.saturating_mul(nband).max(1) as u64;
            kpoint_costs.push(cost);

            let pwbasis_bytes = estimate_pwbasis_bytes(npw);
            let vnl = VNL::new(ik, &pots, &pwwfc, &crystal);
            let vnl_bytes = estimate_vnl_bytes(&vnl);

            let shared_cache = KSCF::build_shared_cache(
                &gvec, &crystal, &pots, &pwwfc, &vnl, fft_shape, ik, k_cart,
            );
            let shared_cache_bytes = shared_cache.estimated_bytes();

            kpoint_estimates.push(KPointEstimate {
                npw,
                pwbasis_bytes,
                vnl_bytes,
                shared_cache_bytes,
            });
        }

        let mpi_world_size = max(1, dwmpi::get_comm_world_size() as usize);
        let ranks = options.ranks_override.unwrap_or(mpi_world_size);
        if ranks == 0 {
            return Err(EstimateError::Fatal("rank count must be >= 1".to_string()));
        }

        let kpoint_schedule_mode = to_kpoint_schedule_mode(control.get_kpoint_schedule_enum());
        let schedule_plan =
            KPointSchedulePlan::new_from_costs(&kpoint_costs, ranks, kpoint_schedule_mode);

        let replicated = estimate_replicated_components(spin_scheme, nfft, npw_rho, &gvec, &pwden);

        let mut rank_estimates = Vec::<RankEstimate>::with_capacity(ranks);
        for rank in 0..ranks {
            let domain = schedule_plan.domain_for_rank(rank);
            let mut components = replicated;

            let mut npw_sum_local = 0usize;
            for ik in domain.global_indices().iter().copied() {
                let est = &kpoint_estimates[ik];
                npw_sum_local = npw_sum_local.saturating_add(est.npw);
                components.gvectors_and_plane_wave_bases = components
                    .gvectors_and_plane_wave_bases
                    .saturating_add(est.pwbasis_bytes);
                components.nonlocal_projector_caches = components
                    .nonlocal_projector_caches
                    .saturating_add(est.vnl_bytes.saturating_add(est.shared_cache_bytes));
            }

            let nk_local = domain.len();
            let npw_nband_local = npw_sum_local.saturating_mul(nband);
            let nk_nband_local = nk_local.saturating_mul(nband);

            let eigenvectors_bytes = spin_channels.saturating_mul(bytes_c64(npw_nband_local));
            let eigenvalues_bytes = spin_channels.saturating_mul(bytes_f64(nk_nband_local));
            let occupations_bytes = spin_channels.saturating_mul(bytes_f64(nk_nband_local));

            let kscf_workspace_elements = nk_local
                .saturating_mul(12usize.saturating_mul(nfft))
                .saturating_add(nk_local.saturating_mul(3usize.saturating_mul(npw_rho)))
                .saturating_add(npw_nband_local)
                .saturating_add(nk_local.saturating_mul(2usize.saturating_mul(hubbard_n_m)));
            let kscf_workspace_bytes =
                spin_channels.saturating_mul(bytes_c64(kscf_workspace_elements));

            components.wavefunction_eigen_storage = components
                .wavefunction_eigen_storage
                .saturating_add(eigenvectors_bytes)
                .saturating_add(eigenvalues_bytes)
                .saturating_add(occupations_bytes)
                .saturating_add(kscf_workspace_bytes);

            rank_estimates.push(RankEstimate {
                rank,
                nk_local,
                components,
            });
        }

        let mut global_components = ComponentBreakdown::default();
        for estimate in rank_estimates.iter() {
            global_components.add_assign(estimate.components);
        }

        let assumptions = vec![
            "Estimator models persistent runtime buffers from geometry setup, SCF workspaces, per-k caches, and eigen storage.".to_string(),
            "Totals include duplicated per-rank allocations and rank-local k-point imbalance from the configured schedule.".to_string(),
            format!(
                "A calibrated process/allocator baseline of {} is included per rank.",
                format_bytes(CALIBRATED_PROCESS_OVERHEAD_BYTES)
            ),
            "Short-lived temporaries are not explicitly modeled; practical error is usually within about +/-20% on representative Si ONCV runs.".to_string(),
        ];

        Ok(EstimateReport {
            case_dir: case_dir.display().to_string(),
            task: control.get_task().to_string(),
            spin_scheme,
            kpoint_schedule_mode,
            mpi_world_size,
            ranks,
            fft_shape,
            nfft,
            npw_rho,
            nkpt,
            nband,
            rank_estimates,
            global_components,
            assumptions,
        })
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

fn to_kpoint_schedule_mode(mode: KPointScheduleScheme) -> KPointScheduleMode {
    match mode {
        KPointScheduleScheme::Contiguous => KPointScheduleMode::Contiguous,
        KPointScheduleScheme::CostAware => KPointScheduleMode::CostAware,
        KPointScheduleScheme::Dynamic => KPointScheduleMode::Dynamic,
    }
}

fn spin_channel_count(spin_scheme: SpinScheme) -> usize {
    match spin_scheme {
        SpinScheme::NonSpin => 1,
        SpinScheme::Spin => 2,
        SpinScheme::Ncl => 1,
    }
}

fn estimate_hubbard_n_m(control: &Control) -> Result<usize, EstimateError> {
    if !control.get_hubbard_u_enabled() {
        return Ok(0);
    }
    let l = control.get_hubbard_l();
    if l < 0 {
        return Err(EstimateError::Fatal(format!(
            "invalid hubbard_l={} (must be >= 0)",
            l
        )));
    }
    Ok(2usize.saturating_mul(l as usize).saturating_add(1))
}

fn estimate_replicated_components(
    spin_scheme: SpinScheme,
    nfft: usize,
    npw_rho: usize,
    gvec: &GVector,
    pwden: &PWDensity,
) -> ComponentBreakdown {
    let spin_channels = spin_channel_count(spin_scheme);
    let mut components = ComponentBreakdown::default();

    let rho_3d_channels = spin_channels;
    let rhocore_channels = 1usize;
    components.fft_real_space_arrays = bytes_c64(
        rho_3d_channels
            .saturating_mul(nfft)
            .saturating_add(rhocore_channels.saturating_mul(nfft)),
    );

    components.gvectors_and_plane_wave_bases = bytes_f64(
        gvec.get_cart()
            .len()
            .saturating_mul(size_of::<Vector3f64>() / size_of::<f64>()),
    )
    .saturating_add(bytes_i32(
        gvec.get_miller()
            .len()
            .saturating_mul(size_of::<Vector3i32>() / size_of::<i32>()),
    ))
    .saturating_add(bytes_f64(pwden.get_g().len()))
    .saturating_add(bytes_usize(pwden.get_gindex().len()))
    .saturating_add(bytes_f64(pwden.get_gshell_norms().len()))
    .saturating_add(bytes_usize(pwden.get_gshell_index().len()));

    components.density_potential_workspaces = bytes_c64(spin_channels.saturating_mul(npw_rho))
        .saturating_add(match spin_scheme {
            SpinScheme::NonSpin => {
                let npw_arrays = 7usize;
                let nfft_arrays = 4usize;
                bytes_c64(
                    npw_arrays
                        .saturating_mul(npw_rho)
                        .saturating_add(nfft_arrays.saturating_mul(nfft)),
                )
            }
            SpinScheme::Spin => {
                let npw_arrays = 14usize;
                let nfft_arrays = 6usize;
                bytes_c64(
                    npw_arrays
                        .saturating_mul(npw_rho)
                        .saturating_add(nfft_arrays.saturating_mul(nfft)),
                )
            }
            SpinScheme::Ncl => 0,
        });

    components.runtime_process_overhead = CALIBRATED_PROCESS_OVERHEAD_BYTES;

    components
}

fn estimate_pwbasis_bytes(npw: usize) -> usize {
    bytes_usize(npw).saturating_add(bytes_f64(npw))
}

fn estimate_vnl_bytes(vnl: &VNL) -> usize {
    let mut bytes = 0usize;
    for by_projector in vnl.get_kgbeta_all().values() {
        for beta_values in by_projector.iter() {
            bytes = bytes.saturating_add(bytes_f64(beta_values.len()));
        }
    }
    for by_projector in vnl.get_dkgbeta_all().values() {
        for beta_values in by_projector.iter() {
            bytes = bytes.saturating_add(bytes_f64(beta_values.len()));
        }
    }
    bytes
}

fn bytes_c64(count: usize) -> usize {
    count.saturating_mul(size_of::<c64>())
}

fn bytes_f64(count: usize) -> usize {
    count.saturating_mul(size_of::<f64>())
}

fn bytes_i32(count: usize) -> usize {
    count.saturating_mul(size_of::<i32>())
}

fn bytes_usize(count: usize) -> usize {
    count.saturating_mul(size_of::<usize>())
}

fn format_bytes(bytes: usize) -> String {
    let kib = bytes as f64 / 1024.0;
    let mib = kib / 1024.0;
    let gib = mib / 1024.0;
    if gib >= 1.0 {
        format!("{} ({:.3} GiB)", bytes, gib)
    } else if mib >= 1.0 {
        format!("{} ({:.3} MiB)", bytes, mib)
    } else if kib >= 1.0 {
        format!("{} ({:.3} KiB)", bytes, kib)
    } else {
        format!("{} B", bytes)
    }
}

fn print_human_report(report: &EstimateReport) {
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
    let global_total = report.global_components.total();

    let peak_rank = report
        .rank_estimates
        .iter()
        .max_by_key(|rank| rank.components.total())
        .unwrap();

    println!("memory_estimate");
    println!("  case_dir = {}", report.case_dir);
    println!("  task = {}", report.task);
    println!("  spin_scheme = {}", report.spin_scheme.as_str());
    println!("  mpi_world_size = {}", report.mpi_world_size);
    println!("  assumed_ranks = {}", report.ranks);
    println!(
        "  kpoint_schedule = {}",
        report.kpoint_schedule_mode.as_str()
    );
    println!(
        "  dimensions = fft={}x{}x{}, nfft={}, npw_rho={}, nkpt={}, nband={}",
        report.fft_shape[0],
        report.fft_shape[1],
        report.fft_shape[2],
        report.nfft,
        report.npw_rho,
        report.nkpt,
        report.nband
    );
    println!();
    println!("totals");
    println!("  estimated_bytes_total = {}", per_rank_max);
    println!(
        "  estimated_bytes_total_per_rank_peak = {}",
        format_bytes(per_rank_max)
    );
    println!(
        "  estimated_bytes_total_per_rank_min = {}",
        format_bytes(per_rank_min)
    );
    println!(
        "  estimated_bytes_total_per_rank_mean = {}",
        format_bytes(per_rank_mean)
    );
    println!(
        "  estimated_bytes_total_global = {}",
        format_bytes(global_total)
    );
    println!();
    println!(
        "breakdown (peak rank = {}, local_kpoints = {})",
        peak_rank.rank, peak_rank.nk_local
    );
    print_component_lines(peak_rank.components, "  ");
    println!();
    println!("breakdown (global cluster sum)");
    print_component_lines(report.global_components, "  ");
    println!();
    println!("rank totals");
    for rank in report.rank_estimates.iter() {
        println!(
            "  rank {:>4}: local_kpoints={:>4}, estimated_bytes_total={}",
            rank.rank,
            rank.nk_local,
            format_bytes(rank.components.total())
        );
    }
    println!();
    println!("assumptions");
    for note in report.assumptions.iter() {
        println!("  - {}", note);
    }
}

fn print_component_lines(components: ComponentBreakdown, indent: &str) {
    println!(
        "{}fft_real_space_arrays = {}",
        indent,
        format_bytes(components.fft_real_space_arrays)
    );
    println!(
        "{}gvectors_and_plane_wave_bases = {}",
        indent,
        format_bytes(components.gvectors_and_plane_wave_bases)
    );
    println!(
        "{}wavefunction_eigen_storage = {}",
        indent,
        format_bytes(components.wavefunction_eigen_storage)
    );
    println!(
        "{}density_potential_workspaces = {}",
        indent,
        format_bytes(components.density_potential_workspaces)
    );
    println!(
        "{}nonlocal_projector_caches = {}",
        indent,
        format_bytes(components.nonlocal_projector_caches)
    );
    println!(
        "{}runtime_process_overhead = {}",
        indent,
        format_bytes(components.runtime_process_overhead)
    );
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
