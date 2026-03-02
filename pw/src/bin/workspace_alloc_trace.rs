use control::{Control, FftPlannerScheme};
use crystal::Crystal;
use fftgrid::FFTGrid;
use force;
use gvector::GVector;
use matrix::Matrix;
use pspot::PSPot;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use stress;
use types::c64;
use types::Vector3f64;

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        REALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[derive(Clone, Copy)]
struct AllocSnapshot {
    alloc_calls: u64,
    alloc_bytes: u64,
    realloc_calls: u64,
    realloc_bytes: u64,
    dealloc_calls: u64,
}

impl AllocSnapshot {
    fn capture() -> Self {
        Self {
            alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
            alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
            realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
            realloc_bytes: REALLOC_BYTES.load(Ordering::Relaxed),
            dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        }
    }

    fn delta(&self, after: &Self) -> Self {
        Self {
            alloc_calls: after.alloc_calls.saturating_sub(self.alloc_calls),
            alloc_bytes: after.alloc_bytes.saturating_sub(self.alloc_bytes),
            realloc_calls: after.realloc_calls.saturating_sub(self.realloc_calls),
            realloc_bytes: after.realloc_bytes.saturating_sub(self.realloc_bytes),
            dealloc_calls: after.dealloc_calls.saturating_sub(self.dealloc_calls),
        }
    }
}

fn trace_alloc(label: &str, iterations: usize, mut f: impl FnMut()) {
    let before = AllocSnapshot::capture();
    for _ in 0..iterations {
        f();
    }
    let after = AllocSnapshot::capture();
    let delta = before.delta(&after);

    println!(
        "{:<36} iter={:>3} alloc_calls={:>4} realloc_calls={:>4} alloc_bytes={:>8} realloc_bytes={:>8} dealloc_calls={:>4}",
        label,
        iterations,
        delta.alloc_calls,
        delta.realloc_calls,
        delta.alloc_bytes,
        delta.realloc_bytes,
        delta.dealloc_calls
    );
}

fn repo_root_from_manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| panic!("failed to resolve workspace root from CARGO_MANIFEST_DIR"))
        .to_path_buf()
}

fn main() {
    let repo_root = repo_root_from_manifest();
    let case_dir = repo_root.join("test_example/si-oncv/scf");

    std::env::set_current_dir(&case_dir).unwrap_or_else(|e| {
        panic!(
            "failed to set current dir to '{}': {}",
            case_dir.display(),
            e
        )
    });

    let mut control = Control::new();
    control.read_file("in.ctrl");
    let fft_planning_mode = match control.get_fft_planner_enum() {
        FftPlannerScheme::Estimate => dwfft3d::FftPlanningMode::Estimate,
        FftPlannerScheme::Measure => dwfft3d::FftPlanningMode::Measure,
    };
    let fft_wisdom_file = control.get_fft_wisdom_file().trim();
    dwfft3d::configure_runtime(dwfft3d::BackendOptions {
        threads: control.get_fft_threads(),
        planning_mode: fft_planning_mode,
        wisdom_file: if fft_wisdom_file.is_empty() {
            None
        } else {
            Some(fft_wisdom_file.to_string())
        },
    });

    let mut crystal = Crystal::new();
    crystal.read_file("in.crystal");

    let pots = PSPot::new(control.get_pot_scheme_enum());

    let fftgrid = FFTGrid::new(crystal.get_latt(), control.get_ecutrho());
    let [n1, n2, n3] = fftgrid.get_size();
    let nfft = fftgrid.get_ntot();

    let rgtrans = RGTransform::new(n1, n2, n3);
    let gvec = GVector::new(crystal.get_latt(), n1, n2, n3);
    let pwden = PWDensity::new(control.get_ecutrho(), &gvec);
    let npw_rho = pwden.get_n_plane_waves();

    let mut rho_3d = vec![c64::new(0.0, 0.0); nfft];
    for (i, v) in rho_3d.iter_mut().enumerate() {
        let x = (i as f64 + 1.0) / nfft as f64;
        *v = c64::new(0.2 + 0.8 * x, 0.0);
    }

    let mut grad_x = vec![c64::new(0.0, 0.0); nfft];
    let mut grad_y = vec![c64::new(0.0, 0.0); nfft];
    let mut grad_z = vec![c64::new(0.0, 0.0); nfft];
    let mut grad_norm = vec![c64::new(0.0, 0.0); nfft];
    let mut div = vec![c64::new(0.0, 0.0); nfft];

    let mut vec_x = vec![c64::new(0.0, 0.0); nfft];
    let mut vec_y = vec![c64::new(0.0, 0.0); nfft];
    let mut vec_z = vec![c64::new(0.0, 0.0); nfft];
    for i in 0..nfft {
        vec_x[i] = c64::new((i as f64).sin() * 1.0e-3, 0.0);
        vec_y[i] = c64::new((i as f64).cos() * 1.0e-3, 0.0);
        vec_z[i] = c64::new(((i as f64) * 0.5).sin() * 1.0e-3, 0.0);
    }

    let mut rhog = vec![c64::new(0.0, 0.0); npw_rho];
    let mut vxcg = vec![c64::new(0.0, 0.0); npw_rho];
    for i in 0..npw_rho {
        let x = (i as f64 + 1.0) / npw_rho as f64;
        rhog[i] = c64::new(1.0e-3 * x, -5.0e-4 * x);
        vxcg[i] = c64::new(2.0e-3 * x, 3.0e-4 * x);
    }

    let natoms = crystal.get_n_atoms();
    let mut force_out = vec![Vector3f64::zeros(); natoms];
    let mut stress_out = Matrix::<f64>::new(3, 3);

    let mut force_ws = force::SpectralWorkspace::new();
    let mut stress_ws = stress::SpectralWorkspace::new();

    // Warmup creates persistent scratch/caches.
    rgtrans.gradient_r3d(
        &gvec,
        &pwden,
        &rho_3d,
        &mut grad_x,
        &mut grad_y,
        &mut grad_z,
    );
    rgtrans.gradient_norm_r3d(&gvec, &pwden, &rho_3d, &mut grad_norm);
    rgtrans.divergence_r3d(&gvec, &pwden, &vec_x, &vec_y, &vec_z, &mut div);

    force::vpsloc_with_workspace(
        &pots,
        &crystal,
        &gvec,
        &pwden,
        &mut force_ws,
        &rhog,
        &mut force_out,
    );
    force::nlcc_xc_with_workspace(
        &pots,
        &crystal,
        &gvec,
        &pwden,
        &mut force_ws,
        &vxcg,
        &mut force_out,
    );

    stress::vpsloc_with_workspace(
        &pots,
        &crystal,
        &gvec,
        &pwden,
        &mut stress_ws,
        &rhog,
        &mut stress_out,
    );
    stress::nlcc_xc_with_workspace(
        &pots,
        &crystal,
        &gvec,
        &pwden,
        &mut stress_ws,
        &vxcg,
        &mut stress_out,
    );

    println!("Workspace allocation trace (steady-state after warmup)");
    println!("case_dir = {}", case_dir.display());
    println!("nfft = {}, npw_rho = {}, natoms = {}", nfft, npw_rho, natoms);

    trace_alloc("rgtransform.gradient_r3d", 20, || {
        rgtrans.gradient_r3d(
            &gvec,
            &pwden,
            &rho_3d,
            &mut grad_x,
            &mut grad_y,
            &mut grad_z,
        )
    });
    trace_alloc("rgtransform.gradient_norm_r3d", 20, || {
        rgtrans.gradient_norm_r3d(&gvec, &pwden, &rho_3d, &mut grad_norm)
    });
    trace_alloc("rgtransform.divergence_r3d", 20, || {
        rgtrans.divergence_r3d(&gvec, &pwden, &vec_x, &vec_y, &vec_z, &mut div)
    });

    trace_alloc("force.vpsloc_with_workspace", 20, || {
        force::vpsloc_with_workspace(
            &pots,
            &crystal,
            &gvec,
            &pwden,
            &mut force_ws,
            &rhog,
            &mut force_out,
        )
    });
    trace_alloc("force.nlcc_xc_with_workspace", 20, || {
        force::nlcc_xc_with_workspace(
            &pots,
            &crystal,
            &gvec,
            &pwden,
            &mut force_ws,
            &vxcg,
            &mut force_out,
        )
    });
    trace_alloc("stress.vpsloc_with_workspace", 20, || {
        stress::vpsloc_with_workspace(
            &pots,
            &crystal,
            &gvec,
            &pwden,
            &mut stress_ws,
            &rhog,
            &mut stress_out,
        )
    });
    trace_alloc("stress.nlcc_xc_with_workspace", 20, || {
        stress::nlcc_xc_with_workspace(
            &pots,
            &crystal,
            &gvec,
            &pwden,
            &mut stress_ws,
            &vxcg,
            &mut stress_out,
        )
    });
}
