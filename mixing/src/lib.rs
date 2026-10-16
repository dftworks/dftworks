mod pulay;
use pulay::*;

mod broyden;
use broyden::*;

use control::MixingParams;
use types::c64;

pub trait Mixing {
    fn compute_next_density(&mut self, gs: &[f64], inp: &mut [c64], out: &[c64]);
}

pub fn new(params: &MixingParams) -> Box<dyn Mixing> {
    let mixing: Box<dyn Mixing>;

    let scheme = params.scheme.as_str();

    match scheme.to_lowercase().as_str() {
        "simple" | "pulay" => {
            mixing = Box::new(MixingPulay::new(params));
        }

        "broyden" => {
            mixing = Box::new(MixingBroyden::new(params));
        }

        other => {
            eprintln!("unknown scf_rho_mix_scheme '{}', fallback to pulay", other);
            mixing = Box::new(MixingPulay::new(params));
        }
    }

    mixing
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_builds_from_mixing_params_alone() {
        let pulay = MixingParams::default();
        let broyden = MixingParams {
            scheme: "broyden".to_string(),
            ..MixingParams::default()
        };

        for params in [pulay, broyden].iter() {
            let mut mixer = new(params);
            let gs = vec![0.0, 1.0, 2.0];
            let mut inp = vec![c64::new(1.0, 0.0); 3];
            let out = vec![c64::new(1.0, 0.0); 3];
            mixer.compute_next_density(&gs, &mut inp, &out);
            assert!(inp.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        }
    }
}
