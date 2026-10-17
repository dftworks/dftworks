// Memory estimate for a pw run, computed from the loaded inputs.
//
// Used by `pw --dry-run` and by the `memory_estimate` binary
// (pw/src/bin/memory_estimate.rs includes this file with #[path]), so both
// report the same numbers. The formulas mirror the persistent allocations
// made by pw: FFT grids, G-vectors and plane-wave bases, wavefunctions and
// eigenvalues, per-k-point KSCF workspaces, density/potential workspaces and
// nonlocal projector caches.

use control::{Control, KPointScheduleScheme, SpinScheme, XcScheme};
use crystal::Crystal;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use kpts_distribution::{KPointScheduleMode, KPointSchedulePlan};
use kscf::KSCF;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::PWDensity;
use std::cmp::max;
use std::mem::size_of;
use types::{c64, Vector3f64, Vector3i32};
use vnl::VNL;

#[derive(Debug, Clone, Copy, Default)]
pub struct ComponentBreakdown {
    pub fft_real_space_arrays: usize,
    pub gvectors_and_plane_wave_bases: usize,
    pub wavefunction_eigen_storage: usize,
    pub density_potential_workspaces: usize,
    pub nonlocal_projector_caches: usize,
    pub runtime_process_overhead: usize,
}

impl ComponentBreakdown {
    pub fn total(self) -> usize {
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
pub struct RankEstimate {
    pub rank: usize,
    pub nk_local: usize,
    pub components: ComponentBreakdown,
}

#[derive(Debug, Clone)]
pub struct KPointEstimate {
    npw: usize,
    pwbasis_bytes: usize,
    vnl_bytes: usize,
    shared_cache_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct EstimateReport {
    pub case_dir: String,
    pub task: String,
    pub spin_scheme: SpinScheme,
    pub kpoint_schedule_mode: KPointScheduleMode,
    pub mpi_world_size: usize,
    pub ranks: usize,
    pub fft_shape: [usize; 3],
    pub nfft: usize,
    pub npw_rho: usize,
    pub nkpt: usize,
    pub nband: usize,
    pub rank_estimates: Vec<RankEstimate>,
    pub global_components: ComponentBreakdown,
    pub assumptions: Vec<String>,
}

pub const CALIBRATED_PROCESS_OVERHEAD_BYTES: usize = 32 * 1024 * 1024;

/// Estimates per-rank and global memory for the given inputs.
/// `ranks_override` replaces the MPI world size (used by `memory_estimate --ranks`).
pub fn estimate(
    control: &Control,
    crystal: &Crystal,
    pots: &PSPot,
    kpts: &dyn KPTS,
    ranks_override: Option<usize>,
    case_dir: &str,
) -> Result<EstimateReport, String> {
    let spin_scheme = control.get_spin_scheme_enum();

    let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
    let fft_shape = fftgrid.get_size();
    let nfft = fftgrid.get_ntot();

    let gvec = GVector::new(crystal.get_latt(), fft_shape[0], fft_shape[1], fft_shape[2]);
    let pwden = PWDensity::new(control.get_ecutrho(), &gvec);
    let npw_rho = pwden.get_n_plane_waves();
    let blatt = crystal.get_latt().reciprocal();

    let nkpt = kpts.get_n_kpts();
    let nband = control.get_nband();
    let hubbard_n_m = estimate_hubbard_n_m(control)?;
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
        let vnl = VNL::new(ik, pots, &pwwfc, crystal);
        let vnl_bytes = estimate_vnl_bytes(&vnl);

        let shared_cache =
            KSCF::build_shared_cache(&gvec, crystal, pots, &pwwfc, &vnl, fft_shape, ik, k_cart);
        let shared_cache_bytes = shared_cache.estimated_bytes();

        kpoint_estimates.push(KPointEstimate {
            npw,
            pwbasis_bytes,
            vnl_bytes,
            shared_cache_bytes,
        });
    }

    let mpi_world_size = max(1, dwmpi::get_comm_world_size() as usize);
    let ranks = ranks_override.unwrap_or(mpi_world_size);
    if ranks == 0 {
        return Err("rank count must be >= 1".to_string());
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

        // Per k-point KSCF workspace: 3 FFT-grid arrays, plus the hybrid
        // buffers (9 FFT-grid arrays, 3 density-sized vectors) for HSE06 only.
        let (hybrid_nfft_arrays, hybrid_npw_rho_vectors) =
            if matches!(control.get_xc_scheme_enum(), XcScheme::Hse06) {
                (9usize, 3usize)
            } else {
                (0usize, 0usize)
            };
        let kscf_workspace_elements = nk_local
            .saturating_mul((3 + hybrid_nfft_arrays).saturating_mul(nfft))
            .saturating_add(nk_local.saturating_mul(hybrid_npw_rho_vectors.saturating_mul(npw_rho)))
            .saturating_add(npw_nband_local)
            .saturating_add(nk_local.saturating_mul(2usize.saturating_mul(hubbard_n_m)));
        let kscf_workspace_bytes = spin_channels.saturating_mul(bytes_c64(kscf_workspace_elements));

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
        case_dir: case_dir.to_string(),
        task: control.get_task().as_str().to_string(),
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

fn estimate_hubbard_n_m(control: &Control) -> Result<usize, String> {
    if !control.get_hubbard_u_enabled() {
        return Ok(0);
    }
    let l = control.get_hubbard_l();
    if l < 0 {
        return Err(format!("invalid hubbard_l={} (must be >= 0)", l));
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

pub fn format_bytes(bytes: usize) -> String {
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

pub fn print_human_report(report: &EstimateReport) {
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

pub fn print_component_lines(components: ComponentBreakdown, indent: &str) {
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
