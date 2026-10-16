mod pulay;
use pulay::*;

mod broyden;
use broyden::*;

use control::{MixingParams, MixingScheme};
use types::c64;

pub trait Mixing {
    fn compute_next_density(&mut self, gs: &[f64], inp: &mut [c64], out: &[c64]);
}

pub fn new(params: &MixingParams) -> Box<dyn Mixing> {
    match params.scheme {
        MixingScheme::Pulay => Box::new(MixingPulay::new(params)),
        MixingScheme::Broyden => Box::new(MixingBroyden::new(params)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_builds_from_mixing_params_alone() {
        for scheme in MixingScheme::ALL {
            let params = MixingParams {
                scheme: *scheme,
                ..MixingParams::default()
            };
            let mut mixer = new(&params);
            let gs = vec![0.0, 1.0, 2.0];
            let mut inp = vec![c64::new(1.0, 0.0); 3];
            let out = vec![c64::new(1.0, 0.0); 3];
            mixer.compute_next_density(&gs, &mut inp, &out);
            assert!(inp.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        }
    }
}
