#![allow(warnings)]
use control::{Control, KPointScheduleScheme, SpinScheme};
use crystal::Crystal;
use dfttypes::*;
use dwconsts::*;
use fftgrid::FFTGrid;
use gvector::GVector;
use kpts::KPTS;
use kpts_distribution::{KPointDomain, KPointScheduleMode, KPointSchedulePlan};
use kscf::KSCF;
use ndarray::*;
use num_traits::identities::Zero;
use pspot::PSPot;
use pwbasis::PWBasis;
use pwdensity::*;
use symmetry::SymmetryDriver;
use types::Vector3f64;
use types::*;
use types::*;
use vnl::VNL;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

mod orchestration;
mod provenance;
mod restart;
mod runtime_display;

struct CountingAlloc;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAlloc = CountingAlloc;

static ALLOC_STATS_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_OLD_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_NEW_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

#[inline]
fn alloc_stats_enabled() -> bool {
    ALLOC_STATS_ENABLED.load(Ordering::Relaxed)
}

#[inline]
fn update_peak_live_bytes(candidate: u64) {
    let mut observed = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    while candidate > observed {
        match PEAK_LIVE_BYTES.compare_exchange_weak(
            observed,
            candidate,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => observed = actual,
        }
    }
}

#[inline]
fn add_live_bytes(bytes: u64) {
    let now = LIVE_BYTES
        .fetch_add(bytes, Ordering::Relaxed)
        .saturating_add(bytes);
    update_peak_live_bytes(now);
}

#[inline]
fn sub_live_bytes(bytes: u64) {
    let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_sub(bytes))
    });
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if ptr.is_null() || !alloc_stats_enabled() {
            return ptr;
        }
        let size = layout.size() as u64;
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
        add_live_bytes(size);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if alloc_stats_enabled() {
            let size = layout.size() as u64;
            DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            DEALLOC_BYTES.fetch_add(size, Ordering::Relaxed);
            sub_live_bytes(size);
        }
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if new_ptr.is_null() || !alloc_stats_enabled() {
            return new_ptr;
        }

        let old_size = layout.size() as u64;
        let new_size = new_size as u64;
        REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        REALLOC_OLD_BYTES.fetch_add(old_size, Ordering::Relaxed);
        REALLOC_NEW_BYTES.fetch_add(new_size, Ordering::Relaxed);
        if new_size >= old_size {
            add_live_bytes(new_size - old_size);
        } else {
            sub_live_bytes(old_size - new_size);
        }
        new_ptr
    }
}

fn read_env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let norm = value.trim().to_ascii_lowercase();
            matches!(norm.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

fn format_bytes_human(bytes: u64) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    format!("{} ({:.3} MiB)", bytes, mib)
}

fn print_runtime_allocation_statistics() {
    if !alloc_stats_enabled() || !dwmpi::is_root() {
        return;
    }

    let alloc_calls = ALLOC_CALLS.load(Ordering::Relaxed);
    let alloc_bytes = ALLOC_BYTES.load(Ordering::Relaxed);
    let dealloc_calls = DEALLOC_CALLS.load(Ordering::Relaxed);
    let dealloc_bytes = DEALLOC_BYTES.load(Ordering::Relaxed);
    let realloc_calls = REALLOC_CALLS.load(Ordering::Relaxed);
    let realloc_old_bytes = REALLOC_OLD_BYTES.load(Ordering::Relaxed);
    let realloc_new_bytes = REALLOC_NEW_BYTES.load(Ordering::Relaxed);
    let live_bytes = LIVE_BYTES.load(Ordering::Relaxed);
    let peak_live_bytes = PEAK_LIVE_BYTES.load(Ordering::Relaxed);
    let gross_requested_bytes = alloc_bytes.saturating_add(realloc_new_bytes);
    let gross_freed_bytes = dealloc_bytes.saturating_add(realloc_old_bytes);
    let net_requested_bytes = gross_requested_bytes.saturating_sub(gross_freed_bytes);

    println!();
    println!("   {:-^88}", " runtime memory allocation statistics ");
    println!();
    println!("   {:28}: {:<18}", "alloc_calls", alloc_calls);
    println!("   {:28}: {:<18}", "dealloc_calls", dealloc_calls);
    println!("   {:28}: {:<18}", "realloc_calls", realloc_calls);
    println!(
        "   {:28}: {:<18}",
        "alloc_bytes",
        format_bytes_human(alloc_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "dealloc_bytes",
        format_bytes_human(dealloc_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "realloc_old_bytes",
        format_bytes_human(realloc_old_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "realloc_new_bytes",
        format_bytes_human(realloc_new_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "gross_requested_bytes",
        format_bytes_human(gross_requested_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "gross_freed_bytes",
        format_bytes_human(gross_freed_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "net_requested_bytes",
        format_bytes_human(net_requested_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "live_bytes",
        format_bytes_human(live_bytes)
    );
    println!(
        "   {:28}: {:<18}",
        "peak_live_bytes",
        format_bytes_human(peak_live_bytes)
    );
}

fn shutdown_and_exit(code: i32) -> ! {
    print_runtime_allocation_statistics();
    dwmpi::barrier(dwmpi::comm_world());
    dwmpi::finalize();
    std::process::exit(code);
}

struct GeometryStepContext {
    fftgrid: FFTGrid,
    rgtrans: rgtransform::RGTransform,
    gvec: GVector,
    pwden: PWDensity,
    blatt: lattice::Lattice,
}

impl GeometryStepContext {
    fn new(crystal: &Crystal, ecutrho: f64) -> Self {
        let fftgrid = FFTGrid::new(crystal.get_latt(), ecutrho);
        let [n1, n2, n3] = fftgrid.get_size();
        let rgtrans = rgtransform::RGTransform::new(n1, n2, n3);
        let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);
        let pwden = PWDensity::new(ecutrho, &gvec);
        let blatt = crystal.get_latt().reciprocal();

        Self {
            fftgrid,
            rgtrans,
            gvec,
            pwden,
            blatt,
        }
    }

    fn fft_shape(&self) -> [usize; 3] {
        self.fftgrid.get_size()
    }

    fn fftgrid(&self) -> &FFTGrid {
        &self.fftgrid
    }

    fn rgtrans(&self) -> &rgtransform::RGTransform {
        &self.rgtrans
    }

    fn gvec(&self) -> &GVector {
        &self.gvec
    }

    fn pwden(&self) -> &PWDensity {
        &self.pwden
    }

    fn blatt(&self) -> &lattice::Lattice {
        &self.blatt
    }
}

struct OrchestrationWorkspace {
    rhocoreg: Vec<c64>,
    rhocore_3d: Option<Array3<c64>>,
    atom_positions: Vec<[f64; 3]>,
}

impl OrchestrationWorkspace {
    fn new() -> Self {
        Self {
            rhocoreg: Vec::new(),
            rhocore_3d: None,
            atom_positions: Vec::new(),
        }
    }

    fn ensure_shape(&mut self, npw_rho: usize, fft_shape: [usize; 3]) {
        if self.rhocoreg.len() != npw_rho {
            self.rhocoreg.resize(npw_rho, c64::zero());
        } else {
            self.rhocoreg.fill(c64::zero());
        }

        let needs_rhocore_resize = self
            .rhocore_3d
            .as_ref()
            .map(|rho| rho.shape() != fft_shape)
            .unwrap_or(true);
        if needs_rhocore_resize {
            self.rhocore_3d = Some(Array3::<c64>::new(fft_shape));
        }

        self.rhocore_3d
            .as_mut()
            .expect("orchestration workspace rhocore_3d should be initialized")
            .set_value(c64::zero());
    }

    fn update_atom_positions(&mut self, crystal: &Crystal) {
        let natom = crystal.get_n_atoms();
        if self.atom_positions.len() != natom {
            self.atom_positions = vec![[0.0; 3]; natom];
        }

        let vatoms = crystal.get_atom_positions();
        for (dst, src) in self.atom_positions.iter_mut().zip(vatoms.iter()) {
            dst[0] = src.x;
            dst[1] = src.y;
            dst[2] = src.z;
        }
    }

    fn core_charge_buffers_mut(&mut self) -> (&mut [c64], &mut Array3<c64>) {
        let rhocoreg = self.rhocoreg.as_mut_slice();
        let rhocore_3d = self
            .rhocore_3d
            .as_mut()
            .expect("orchestration workspace rhocore_3d should be initialized");
        (rhocoreg, rhocore_3d)
    }

    fn rhocore_3d(&self) -> &Array3<c64> {
        self.rhocore_3d
            .as_ref()
            .expect("orchestration workspace rhocore_3d should be initialized")
    }

    fn atom_positions(&self) -> &[[f64; 3]] {
        self.atom_positions.as_slice()
    }
}

struct RuntimeContext<'a> {
    control: &'a Control,
    crystal: &'a Crystal,
    pots: &'a PSPot,
    kpts: &'a dyn KPTS,
    spin_scheme: SpinScheme,
}

impl<'a> RuntimeContext<'a> {
    fn new(
        control: &'a Control,
        crystal: &'a Crystal,
        pots: &'a PSPot,
        kpts: &'a dyn KPTS,
        spin_scheme: SpinScheme,
    ) -> Self {
        Self {
            control,
            crystal,
            pots,
            kpts,
            spin_scheme,
        }
    }

    fn kpoint_schedule_mode(&self) -> KPointScheduleMode {
        match self.control.get_kpoint_schedule_enum() {
            KPointScheduleScheme::Contiguous => KPointScheduleMode::Contiguous,
            KPointScheduleScheme::CostAware => KPointScheduleMode::CostAware,
            KPointScheduleScheme::Dynamic => KPointScheduleMode::Dynamic,
        }
    }

    fn checkpoint_meta(&self) -> CheckpointMeta {
        restart::checkpoint_meta_for_run(self.control, self.spin_scheme, self.kpts.get_k_mesh())
    }
}

struct ElectronicStepContext {
    k_schedule_plan: KPointSchedulePlan,
    k_domain: KPointDomain,
    vpwwfc: Vec<PWBasis>,
    vvnl: Vec<VNL>,
}

impl ElectronicStepContext {
    fn build(runtime: &RuntimeContext, blatt: &lattice::Lattice, gvec: &GVector) -> Self {
        let nrank = dwmpi::get_comm_world_size() as usize;
        let k_costs = orchestration::electronic::estimate_kpoint_costs(runtime, blatt, gvec);
        let k_schedule_plan = KPointSchedulePlan::new_from_costs(
            k_costs.as_slice(),
            nrank,
            runtime.kpoint_schedule_mode(),
        );
        let k_domain = k_schedule_plan.domain_for_current_rank();

        if dwmpi::is_root() {
            const OUT_WIDTH1: usize = 32;
            let min_load = k_schedule_plan.min_rank_load();
            let max_load = k_schedule_plan.max_rank_load();
            let avg_load = k_schedule_plan.mean_rank_load();
            let imbalance_pct = if avg_load > 0.0 {
                (k_schedule_plan.imbalance_ratio() - 1.0) * 100.0
            } else {
                0.0
            };
            println!(
                "   {:<width1$} = {} (rank_cost min/avg/max = {}/{:.2}/{}, imbalance={:.2}%)",
                "kpoint_schedule",
                k_schedule_plan.mode().as_str(),
                min_load,
                avg_load,
                max_load,
                imbalance_pct,
                width1 = OUT_WIDTH1
            );
        }

        let mut vpwwfc = Vec::<PWBasis>::with_capacity(k_domain.len());
        for slot in k_domain.iter() {
            let ik = slot.global_index;
            let k_frac = runtime.kpts.get_k_frac(ik);
            let k_cart = runtime.kpts.frac_to_cart(&k_frac, blatt);
            let pwwfc = PWBasis::new(k_cart, ik, runtime.control.get_ecut(), gvec);
            vpwwfc.push(pwwfc);
        }

        let mut vvnl = Vec::<VNL>::with_capacity(k_domain.len());
        for slot in k_domain.iter() {
            let ik = slot.global_index;
            let ilocal = slot.local_slot;
            let vnl = VNL::new(ik, runtime.pots, &vpwwfc[ilocal], runtime.crystal);
            vvnl.push(vnl);
        }

        Self {
            k_schedule_plan,
            k_domain,
            vpwwfc,
            vvnl,
        }
    }

    fn kpoint_domain(&self) -> &KPointDomain {
        &self.k_domain
    }

    fn global_k_indices(&self) -> &[usize] {
        self.k_domain.global_indices()
    }

    fn local_nkpt(&self) -> usize {
        self.k_domain.len()
    }

    fn vpwwfc(&self) -> &[PWBasis] {
        self.vpwwfc.as_slice()
    }

    fn build_scf_state<'a>(
        &'a self,
        runtime: &RuntimeContext<'a>,
        gvec: &'a GVector,
        pwden: &'a PWDensity,
        fft_shape: [usize; 3],
    ) -> Result<(VKSCF<'a>, VKEigenValue, VKEigenVector, usize), String> {
        let nband = runtime.control.get_nband();
        let my_nkpt = self.local_nkpt();

        let (vkscf, spin_cache_saved_bytes_local) =
            match runtime.spin_scheme {
                SpinScheme::NonSpin => {
                    let channel = orchestration::electronic::build_kscf_channel(
                        runtime, self, gvec, pwden, fft_shape, 0,
                    );
                    (VKSCF::NonSpin(channel), 0usize)
                }
                SpinScheme::Spin => {
                    let (channel_up, channel_dn, saved_bytes) =
                        orchestration::electronic::build_spin_kscf_channels(
                            runtime, self, gvec, pwden, fft_shape,
                        );
                    (VKSCF::Spin(channel_up, channel_dn), saved_bytes)
                }
                SpinScheme::Ncl => return Err(
                    "unsupported capability: spin_scheme='ncl' is not implemented in KSCF setup"
                        .to_string(),
                ),
            };

        let vkevals =
            orchestration::electronic::allocate_eigenvalues(runtime.spin_scheme, nband, my_nkpt)?;
        let vkevecs = orchestration::electronic::allocate_eigenvectors(
            runtime.spin_scheme,
            nband,
            &self.vpwwfc,
        )?;

        Ok((vkscf, vkevals, vkevecs, spin_cache_saved_bytes_local))
    }
}

fn main() {
    // Top-level program flow:
    // 1) initialize MPI/runtime and read all inputs
    // 2) build per-geometry numerical context (FFT/G-vectors/k-points)
    // 3) run SCF to obtain energies, forces, and stress
    // 4) optionally perform geometry optimization step
    // 5) export requested artifacts (charge/wfc/eig files)
    // first statement
    dwmpi::init();
    ALLOC_STATS_ENABLED.store(read_env_flag("PW_ALLOC_STATS"), Ordering::Relaxed);

    // start the timer-main

    let stopwatch_main = std::time::Instant::now();

    let bootstrap = match orchestration::bootstrap::load_bootstrap_inputs() {
        Ok(bootstrap) => bootstrap,
        Err(err) => {
            if dwmpi::is_root() {
                eprintln!("{}", err);
            }
            shutdown_and_exit(1);
        }
    };
    let orchestration::bootstrap::BootstrapData {
        control,
        mut crystal,
        pots,
        kpts,
        zions,
        verbosity,
        spin_scheme,
    } = bootstrap;

    let latt0 = crystal.get_latt().clone();
    let geom_optim_mask_cell = crystal.get_cell_mask().clone();

    //

    let mut stress_total = Matrix::<f64>::new(3, 3);
    let mut force_total = vec![Vector3f64::zeros(); crystal.get_n_atoms()];

    // Geometry optimizer state (BFGS/DIIS depending on in.ctrl).

    let mut geom_iter = 1;

    let mut geom_driver = geom::new(
        control.get_geom_optim_scheme(),
        control.get_geom_optim_alpha(),
        control.get_geom_optim_history_steps(),
    );

    // Self-consistent field drivers selected by spin scheme.

    let scf_driver = scf::new(spin_scheme);

    let density_driver = density::new(spin_scheme);
    let mut orchestration_workspace = OrchestrationWorkspace::new();

    // crystal.display();

    loop {
        let mut phase = match orchestration::construction::construct_geometry_phase(
            orchestration::construction::GeometryPhaseInput {
                control: &control,
                crystal: &crystal,
                pots: &pots,
                kpts: kpts.as_ref(),
                zions: &zions,
                spin_scheme,
                geom_iter,
                density_driver: density_driver.as_ref(),
                orchestration_workspace: &mut orchestration_workspace,
            },
        ) {
            Ok(phase) => phase,
            Err(err) => {
                if dwmpi::is_root() {
                    eprintln!("failed to construct geometry phase: {}", err);
                }
                shutdown_and_exit(1);
            }
        };

        let (mut vkscf, mut vkevals, mut vkevecs, spin_cache_saved_bytes_local) =
            match phase.electronic_ctx.build_scf_state(
                &phase.runtime_ctx,
                phase.geom_ctx.gvec(),
                phase.geom_ctx.pwden(),
                phase.geom_ctx.fft_shape(),
            ) {
                Ok(state) => state,
                Err(err) => {
                    if dwmpi::is_root() {
                        eprintln!("failed to initialize SCF state: {}", err);
                    }
                    shutdown_and_exit(1);
                }
            };

        if matches!(spin_scheme, SpinScheme::Spin) {
            let local_saved = spin_cache_saved_bytes_local as f64;
            let mut saved_total = 0.0f64;
            dwmpi::reduce_scalar_sum(&local_saved, &mut saved_total, dwmpi::comm_world());
            dwmpi::bcast_scalar(&mut saved_total, dwmpi::comm_world());
            if dwmpi::is_root() && saved_total > 0.0 {
                println!(
                    "   spin_cache_dedup_saved ~= {:.3} MiB (shared immutable per-k caches)",
                    saved_total / (1024.0 * 1024.0)
                );
            }
        }

        if geom_iter == 1 && control.get_restart() {
            match restart::try_load_wavefunction_checkpoint(
                spin_scheme,
                phase.electronic_ctx.kpoint_domain(),
                phase.geom_ctx.blatt(),
                &phase.expected_checkpoint_meta,
                phase.electronic_ctx.vpwwfc(),
                &mut vkevecs,
            ) {
                Ok(message) => {
                    if dwmpi::is_root() {
                        println!("   {}", message);
                    }
                }
                Err(err) => {
                    if dwmpi::is_root() {
                        println!("   NOTE: {}", err);
                        println!("   NOTE: continue with default wavefunction initialization");
                    }
                }
            }
        }

        // ions optimization

        if dwmpi::is_root() {
            println!("\n   #step: geom-{}\n", geom_iter);
        }

        let rhocore_3d = orchestration_workspace.rhocore_3d();

        scf_driver.run(
            geom_iter,
            &control,
            &crystal,
            phase.geom_ctx.gvec(),
            phase.geom_ctx.pwden(),
            &pots,
            phase.geom_ctx.rgtrans(),
            kpts.as_ref(),
            &phase.ewald,
            phase.electronic_ctx.vpwwfc(),
            &mut vkscf,
            &mut phase.rhog,
            &mut phase.rho_3d,
            rhocore_3d,
            &mut vkevals,
            &mut vkevecs,
            phase.symdrv.as_ref(),
            &mut stress_total,
            &mut force_total,
        );

        if let Err(err) = orchestration::outputs::persist_outputs(
            &control,
            &crystal,
            phase.geom_ctx.blatt(),
            &phase.expected_checkpoint_meta,
            &phase.rho_3d,
            &phase.electronic_ctx,
            &vkevecs,
        ) {
            if dwmpi::is_root() {
                eprintln!("failed to persist checkpoint/output artifacts: {}", err);
            }
            shutdown_and_exit(1);
        }

        let should_exit = orchestration::outputs::evaluate_exit_and_finalize(
            &control,
            geom_iter,
            &stress_total,
            force_total.as_slice(),
            &vkevals,
            &vkevecs,
            &vkscf,
            &phase.electronic_ctx,
        );

        if should_exit {
            break;
        }

        // if not converged, get the atom positions and (lattice vectors if cell is also relaxed) for the next optim iteration
        let geom_optim_mask_ions = vec![Vector3f64::new(1.0, 1.0, 1.0); crystal.get_n_atoms()];

        let mut bcell_move = control.get_geom_optim_cell();

        geom_driver.compute_next_configuration(
            &mut crystal,
            &force_total,
            &stress_total,
            &geom_optim_mask_ions,
            &geom_optim_mask_cell,
            &latt0,
            bcell_move,
        );

        // end: geom relaxation

        geom_iter += 1;
    }

    // computing time statistics
    let elapsed_main_seconds = stopwatch_main.elapsed().as_secs_f64();

    if dwmpi::is_root() {
        println!();
        println!("   {:-^88}", " statistics ");
        println!();

        println!(
            "   {:16}{:5}{:16.2} seconds {:16.2} hours",
            "Total",
            ":",
            elapsed_main_seconds,
            elapsed_main_seconds / 3600.0
        );
    }

    // last statement

    shutdown_and_exit(0);
}

fn post_processing(
    control: &Control,
    vkevals: &VKEigenValue,
    vkevecs: &VKEigenVector,
    vkscf: &VKSCF,
) {
    // total density of states

    // println!("   compute total density of states");

    //dos::compute_total_density_of_states(control, vkevals, vkevecs, vkscf);
}

fn matrix3x3_to_vector3(mat: &Matrix<f64>) -> Vec<Vector3f64> {
    let mut v = vec![Vector3f64::zeros(); 3];

    for i in 0..3 {
        v[i].x = mat[(0, i)];
        v[i].y = mat[(1, i)];
        v[i].z = mat[(2, i)];
    }

    v
}
