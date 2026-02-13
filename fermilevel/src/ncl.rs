use crate::FermiLevel;
use dfttypes::*;

pub struct FermiLevelNcl {}

impl FermiLevelNcl {
    pub fn new() -> FermiLevelNcl {
        FermiLevelNcl {}
    }
}

impl FermiLevel for FermiLevelNcl {
    fn get_fermi_level(&self, _vkscf: &mut VKSCF, _nelec: f64, _vevals: &VKEigenValue) -> f64 {
        panic!("FermiLevelNcl::get_fermi_level is not implemented yet");
    }

    fn set_occ(
        &self,
        _vkscf: &mut VKSCF,
        _nelec: f64,
        _vevals: &VKEigenValue,
        _fermi_level: f64,
        _occ_inversion: f64,
    ) -> Option<f64> {
        panic!("FermiLevelNcl::set_occ is not implemented yet");
    }
}
