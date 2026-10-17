mod upf;
pub use upf::AtomPSPUPF;

mod upffr;
pub use upffr::AtomPSPUPFFR;

use std::str::FromStr;

/// Read-only view of one atom's pseudopotential, whatever the file format.
/// Each format has its own loader, `AtomPSPUPF::from_file` or
/// `AtomPSPUPFFR::from_file`, which returns an error for a bad file.
pub trait AtomPSP {
    fn get_nbeta(&self) -> usize;
    fn get_lbeta(&self, ibeta: usize) -> usize;

    /// for each l, there can be multiple projectors
    fn get_beta(&self, ibeta: usize) -> &[f64];

    /// return a vector because of multiple projectors
    fn get_dfact(&self, ibeta: usize) -> f64;

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
    fn has_wfc(&self, l: usize) -> bool;
    fn get_wfc_channels(&self) -> Vec<usize>;
    fn get_vloc(&self) -> &[f64];
}

// Parse one attribute value, e.g. z_valence="4.0" in PP_HEADER.
fn parse_attr<T: FromStr>(name: &str, value: &str) -> Result<T, String> {
    value
        .trim()
        .parse()
        .map_err(|_| format!("invalid value '{}' for attribute '{}'", value.trim(), name))
}

// Parse the numbers inside a data block, e.g. <PP_R> ... </PP_R>, multiplied by `scale`.
fn parse_block(tag: &str, text: &str, scale: f64) -> Result<Vec<f64>, String> {
    text.split_whitespace()
        .map(|x| {
            x.parse::<f64>()
                .map(|v| v * scale)
                .map_err(|_| format!("invalid number '{}' in {}", x, tag))
        })
        .collect()
}

fn check_len(what: &str, found: usize, expected: usize) -> Result<(), String> {
    if found != expected {
        return Err(format!(
            "{} has {} values, expected {}",
            what, found, expected
        ));
    }
    Ok(())
}
