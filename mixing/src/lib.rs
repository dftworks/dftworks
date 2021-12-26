mod pulay;
use pulay::*;

mod broyden;
use broyden::*;

use control::Control;
use types::c64;

pub trait Mixing {
    fn compute_next_density(&mut self, gs: &[f64], inp: &mut [c64], out: &[c64]);
}

//pub fn new(scheme: &str, alpha: f64, nhistory: usize) -> Box<dyn Mixing> {
pub fn new(control: &Control) -> Box<dyn Mixing> {
    let mixing: Box<dyn Mixing>;

    let scheme = control.get_scf_rho_mix_scheme();

    match scheme {
        "pulay" => {
            mixing = Box::new(MixingPulay::new(control));
        }

        "broyden" => {
            mixing = Box::new(MixingBroyden::new(control));
        }

        &_ => {
            mixing = Box::new(MixingPulay::new(control));
        }
    }

    mixing
}
