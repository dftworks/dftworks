#![allow(warnings)]

use dfttypes::*;
use ndarray::Array3;
use types::c64;

use crate::XC;

pub struct XCPBE {}

impl XCPBE {
    pub fn new() -> XCPBE {
        XCPBE {}
    }
}

impl XC for XCPBE {
    fn potential_and_energy(
        &self,
        rho: &RHOR,
        drho: Option<&DRHOR>,
        vxc: &mut VXCR,
        exc: &mut Array3<c64>,
    ) {
    }
}
