mod diis;
use diis::*;

/// calculate the next input so vin is mutable instead of vout
pub trait OptimizationDriver {
    fn compute_next_input(&mut self, vin: &mut [f64], vout: &[f64], mask: &[f64]);
}

pub fn new(scheme: &str, alpha: f64, nstep: usize) -> Box<dyn OptimizationDriver> {
    let optim: Box<dyn OptimizationDriver>;

    match scheme.to_lowercase().as_str() {
        "diis" => {
            optim = Box::new(DIIS::new(alpha, nstep));
        }

        other => {
            panic!("unsupported optimization scheme '{}'", other);
        }
    }

    optim
}
