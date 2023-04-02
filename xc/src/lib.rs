mod ldapz;
mod lsdapz;
use ldapz::*;
use lsdapz::*;

use dfttypes::*;
use ndarray::*;
use types::*;

pub trait XC {
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        drho: &Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    );
}

pub fn new(xc_scheme: &str) -> Box<dyn XC> {
    let xc: Box<dyn XC>;

    match xc_scheme {
        "lda-pz" => {
            xc = Box::new(XCLDAPZ::new());
        }

        "lsda-pz" => {
            xc = Box::new(XCLSDAPZ::new());
        }

        &_ => {
            println!("{} not implemented", xc_scheme);
            panic!();
        }
    }

    xc
}
