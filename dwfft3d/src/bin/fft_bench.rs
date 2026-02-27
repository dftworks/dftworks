use dwfft3d::{BackendOptions, DWFFT3D, FftPlanningMode};
use num_complex::Complex64 as c64;
use std::env;
use std::time::Instant;

fn parse_env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_planning_mode() -> FftPlanningMode {
    match env::var("DWWORKS_FFT_PLANNER")
        .unwrap_or_else(|_| "estimate".to_string())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "measure" => FftPlanningMode::Measure,
        _ => FftPlanningMode::Estimate,
    }
}

fn run_case(
    n1: usize,
    n2: usize,
    n3: usize,
    threads: usize,
    niter: usize,
    mode: FftPlanningMode,
    wisdom_file: Option<String>,
) {
    dwfft3d::configure_runtime(BackendOptions {
        threads,
        planning_mode: mode,
        wisdom_file,
    });

    let mut in_data = vec![c64::new(0.0, 0.0); n1 * n2 * n3];
    let mut out_data = vec![c64::new(0.0, 0.0); n1 * n2 * n3];

    for (i, v) in in_data.iter_mut().enumerate() {
        *v = c64::new(i as f64, -(i as isize) as f64);
    }

    let plan_start = Instant::now();
    let fft = DWFFT3D::new(n1, n2, n3);
    let plan_elapsed = plan_start.elapsed();

    fft.fft3d(&in_data, &mut out_data);
    let run_start = Instant::now();
    for _ in 0..niter {
        fft.fft3d(&in_data, &mut out_data);
    }
    let run_elapsed = run_start.elapsed();
    let avg_ms = run_elapsed.as_secs_f64() * 1.0e3 / niter as f64;

    println!(
        "mode={} threads={} plan_s={:.6} exec_avg_ms={:.4}",
        mode.as_str(),
        threads,
        plan_elapsed.as_secs_f64(),
        avg_ms
    );
}

fn main() {
    let n1 = parse_env_usize("DWWORKS_FFT_N1", 128);
    let n2 = parse_env_usize("DWWORKS_FFT_N2", 128);
    let n3 = parse_env_usize("DWWORKS_FFT_N3", 128);
    let niter = parse_env_usize("DWWORKS_FFT_ITERS", 10);
    let threads = parse_env_usize("DWWORKS_FFT_THREADS", 1);
    let wisdom_file = env::var("DWWORKS_FFT_WISDOM")
        .ok()
        .and_then(|s| if s.trim().is_empty() { None } else { Some(s) });

    println!(
        "Running {} FFT benchmark: size = {} x {} x {}, iters = {}",
        if cfg!(feature = "gpu") { "GPU" } else { "CPU" },
        n1,
        n2,
        n3,
        niter
    );

    if env::var("DWWORKS_FFT_COMPARE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        run_case(
            n1,
            n2,
            n3,
            threads,
            niter,
            FftPlanningMode::Estimate,
            wisdom_file.clone(),
        );
        run_case(
            n1,
            n2,
            n3,
            threads,
            niter,
            FftPlanningMode::Measure,
            wisdom_file,
        );
    } else {
        run_case(
            n1,
            n2,
            n3,
            threads,
            niter,
            parse_planning_mode(),
            wisdom_file,
        );
    }
}
