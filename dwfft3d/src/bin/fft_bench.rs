use dwfft3d::DWFFT3D;
use num_complex::Complex64 as c64;
use std::time::Instant;

fn main() {
    // Pick a moderately large FFT size for benchmarking
    let n1 = 256;
    let n2 = 256;
    let n3 = 256;
    println!(
        "Running {} FFT benchmark: size = {} x {} x {}",
        if cfg!(feature = "gpu") { "GPU" } else { "CPU" },
        n1, n2, n3
    );

    // Create FFT object
    let fft = DWFFT3D::new(n1, n2, n3);

    // Allocate input and output
    let mut in_data = vec![c64::new(0.0, 0.0); n1 * n2 * n3];
    let mut out_data = vec![c64::new(0.0, 0.0); n1 * n2 * n3];

    // Fill input with some data
    for (i, v) in in_data.iter_mut().enumerate() {
        //*v = c64::new(i as f64, -i as f64);
        *v = c64::new(i as f64, -(i as isize) as f64);
    }

    // Warm-up run (important for GPU to avoid including init time)
    fft.fft3d(&in_data, &mut out_data);

    // Benchmark
    let start = Instant::now();
    fft.fft3d(&in_data, &mut out_data);
    let elapsed = start.elapsed();

    println!("FFT time: {:.6} sec", elapsed.as_secs_f64());
}

