#![allow(warnings)]

mod ldapz;
use ldapz::*;

mod pbe;
use pbe::*;

use dfttypes::*;
use gvector::GVector;
use ndarray::*;
use pwdensity::PWDensity;
use rgtransform::RGTransform;
use types::*;

// Exchange-correlation model interface used by SCF.
//
// Important design choice:
// - We pass `gvec/pwden/rgtrans` into XC so a GGA functional can evaluate
//   gradients/divergences internally and return a variationally consistent
//   real-space potential.
// - This avoids ad-hoc "precomputed |grad rho|" plumbing and keeps the full
//   GGA derivative workflow in one place.
pub trait XC {
    fn potential_and_energy(
        &self,
        gvec: &GVector,
        pwden: &PWDensity,
        rgtrans: &RGTransform,
        rho: &RHOR,
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
            xc = Box::new(XCLDAPZ::new());
        }

        "pbe" => {
            xc = Box::new(XCPBE::new());
        }

        &_ => {
            println!("{} not implemented", xc_scheme);
            panic!();
        }
    }

    xc
}
