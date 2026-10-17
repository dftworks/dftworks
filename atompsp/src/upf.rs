use crate::upfread::{read_upf, UpfData};
use crate::AtomPSP;

use std::fmt;

/// Scalar-relativistic UPF v2 pseudopotential (`pot_scheme = upf`): the file's
/// projectors are used as they are.
pub struct AtomPSPUPF {
    upf: UpfData,
}

impl AtomPSPUPF {
    /// Read a UPF v2 (XML) file.
    pub fn from_file(pspfile: &str) -> Result<AtomPSPUPF, String> {
        Ok(AtomPSPUPF {
            upf: read_upf(pspfile)?,
        })
    }
}

impl AtomPSP for AtomPSPUPF {
    fn get_nlcc(&self) -> bool {
        self.upf.core_correction
    }

    fn get_nbeta(&self) -> usize {
        self.upf.nbeta
    }

    fn get_lbeta(&self, ibeta: usize) -> usize {
        self.upf.lbeta[ibeta]
    }

    fn get_beta(&self, ibeta: usize) -> &[f64] {
        &self.upf.beta[ibeta]
    }

    // dij is not always positive, so absorbing it into beta will make beta complex numbers
    fn get_dfact(&self, ibeta: usize) -> f64 {
        self.upf.dij[(ibeta, ibeta)]
    }

    fn get_zion(&self) -> f64 {
        self.upf.zion
    }

    fn get_lloc(&self) -> i32 {
        self.upf.lloc
    }

    fn get_lmax(&self) -> usize {
        self.upf.lmax
    }

    fn get_mmax(&self) -> usize {
        self.upf.mmax
    }

    fn get_rad(&self) -> &[f64] {
        &self.upf.rad
    }

    fn get_rab(&self) -> &[f64] {
        &self.upf.rab
    }

    fn get_rho(&self) -> &[f64] {
        &self.upf.rhoatom
    }

    fn get_rhocore(&self) -> &[f64] {
        &self.upf.rhocore
    }

    fn get_wfc(&self, l: usize) -> &[f64] {
        self.upf.chi.get(&l).unwrap()
    }

    fn has_wfc(&self, l: usize) -> bool {
        self.upf.chi.contains_key(&l)
    }

    fn get_wfc_channels(&self) -> Vec<usize> {
        let mut ls: Vec<usize> = self.upf.chi.keys().copied().collect();
        ls.sort_unstable();
        ls
    }

    fn get_vloc(&self) -> &[f64] {
        &self.upf.vloc
    }
}

impl fmt::Display for AtomPSPUPF {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            " element = {:?} zion = {} lmax = {} lloc = {}",
            self.upf.element, self.upf.zion, self.upf.lmax, self.upf.lloc
        )?;
        writeln!(f, " mmax = {} nbeta = {}", self.upf.mmax, self.upf.nbeta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_si_sr() {
        let path = format!("{}/src/Si-sr.upf", env!("CARGO_MANIFEST_DIR"));
        let psp = AtomPSPUPF::from_file(&path).unwrap();

        assert_eq!(psp.get_zion(), 4.0);
        assert_eq!(psp.get_nbeta(), 4);
        assert_eq!(psp.get_rad().len(), psp.get_mmax());
        for ibeta in 0..psp.get_nbeta() {
            assert_eq!(psp.get_beta(ibeta).len(), psp.get_mmax());
        }
    }
}
