mod upf;
use upf::*;

mod upffr;
use upffr::*;

pub trait AtomPSP {
    fn get_nbeta(&self) -> usize;
    fn get_lbeta(&self, ibeta: usize) -> usize;

    /// for each l, there can be multiple projectors
    fn get_beta(&self, ibeta: usize) -> &[f64];

    /// return a vector because of multiple projectors
    fn get_dfact(&self, ibeta: usize) -> f64;

    /// atomic number
    fn get_zatom(&self) -> f64;

    /// valence electron number
    fn get_zion(&self) -> f64;

    fn get_lloc(&self) -> i32;
    fn get_lmax(&self) -> usize;
    fn get_mmax(&self) -> usize;
    fn get_rad(&self) -> &[f64];
    fn get_rab(&self) -> &[f64];
    fn get_rho(&self) -> &[f64];
    fn get_nlcc(&self) -> bool;
    fn get_rhocore(&self) -> &[f64];
    fn get_wfc(&self, l: usize) -> &[f64];
    fn get_vloc(&self) -> &[f64];
    fn read_file(&mut self, pspfile: &str);

    fn get_nbeta_soc(&self) -> usize;
    fn get_lbeta_soc(&self, ibeta: usize) -> usize;
    fn get_beta_soc(&self, ibeta: usize) -> &[f64];
    fn get_dfact_soc(&self, ibeta: usize) -> f64;
}

pub fn new(scheme: &str) -> Box<dyn AtomPSP> {
    let atompsp: Box<dyn AtomPSP>;

    match scheme {
        "upf" => {
            atompsp = Box::new(AtomPSPUPF::new());
        }

        "upf-fr" => {
            atompsp = Box::new(AtomPSPUPFFR::new());
        }

        &_ => {
            atompsp = Box::new(AtomPSPUPF::new());
        }
    }

    atompsp
}
