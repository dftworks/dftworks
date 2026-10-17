#![allow(warnings)]
use crate::{check_len, parse_attr, parse_block, AtomPSP};

use dwconsts::*;
use types::*;
use utility;

use std::{
    fmt,
    fs::File,
    io::{BufRead, BufReader},
};

use std::collections::HashMap;

use xml;
use xml::reader::{EventReader, XmlEvent};

enum UPF {
    HEADER,
    R,
    RAB,
    LOCAL,
    BETA,
    CHI,
    NLCC,
    DIJ,
    RHOATOM,
    NULL,
}

#[derive(Debug)]
pub struct AtomPSPUPF {
    element: String,
    pseudo_type: String,
    relativistic: bool,
    core_correction: bool,
    zion: f64,
    lmax: usize,
    lloc: i32,
    lbeta: Vec<usize>,
    nbeta: usize,
    mmax: usize,
    rad: Vec<f64>,
    rab: Vec<f64>,
    vloc: Vec<f64>,
    vbeta: Vec<Vec<f64>>,
    dij: Matrix<f64>,
    chi: HashMap<usize, Vec<f64>>,
    rhocore: Vec<f64>,
    rhoatom: Vec<f64>,
}

impl Default for AtomPSPUPF {
    fn default() -> Self {
        Self {
            element: String::new(),
            pseudo_type: String::new(),
            relativistic: false,
            core_correction: false,
            zion: 0.0,
            lmax: 0,
            lloc: 0,
            lbeta: Vec::new(),
            nbeta: 0,
            mmax: 0,
            rad: Vec::new(),
            rab: Vec::new(),
            vloc: Vec::new(),
            vbeta: Vec::new(),
            dij: Matrix::<f64>::new(0, 0),
            chi: HashMap::new(),
            rhocore: Vec::new(),
            rhoatom: Vec::new(),
        }
    }
}

impl AtomPSP for AtomPSPUPF {
    fn get_nlcc(&self) -> bool {
        self.core_correction
    }

    fn get_nbeta(&self) -> usize {
        self.nbeta
    }

    fn get_lbeta(&self, ibeta: usize) -> usize {
        self.lbeta[ibeta]
    }

    fn get_beta(&self, ibeta: usize) -> &[f64] {
        &self.vbeta[ibeta]
    }

    // dij is not always positive, so absorbing it into beta will make beta complex numbers
    fn get_dfact(&self, ibeta: usize) -> f64 {
        self.dij[(ibeta, ibeta)]
    }

    fn get_zion(&self) -> f64 {
        self.zion
    }

    fn get_lloc(&self) -> i32 {
        self.lloc
    }

    fn get_lmax(&self) -> usize {
        self.lmax
    }

    fn get_mmax(&self) -> usize {
        self.mmax
    }

    fn get_rad(&self) -> &[f64] {
        &self.rad
    }

    fn get_rab(&self) -> &[f64] {
        &self.rab
    }

    fn get_rho(&self) -> &[f64] {
        &self.rhoatom
    }

    fn get_rhocore(&self) -> &[f64] {
        &self.rhocore
    }

    fn get_wfc(&self, l: usize) -> &[f64] {
        self.chi.get(&l).unwrap()
    }

    fn has_wfc(&self, l: usize) -> bool {
        self.chi.contains_key(&l)
    }

    fn get_wfc_channels(&self) -> Vec<usize> {
        let mut ls: Vec<usize> = self.chi.keys().copied().collect();
        ls.sort_unstable();
        ls
    }

    fn get_vloc(&self) -> &[f64] {
        &self.vloc
    }
}

impl AtomPSPUPF {
    /// Read a scalar-relativistic UPF (v2, XML) file.
    pub fn from_file(pspfile: &str) -> Result<AtomPSPUPF, String> {
        let mut psp = AtomPSPUPF::default();
        psp.parse_upf(pspfile)?;
        psp.check()?;
        Ok(psp)
    }

    fn parse_upf(&mut self, pspfile: &str) -> Result<(), String> {
        let file = File::open(pspfile).map_err(|err| format!("cannot open file: {}", err))?;
        let file = BufReader::new(file);

        let mut parser = xml::reader::EventReader::new(file);

        let mut data_type = UPF::NULL;
        let mut current_chi_l: Option<usize> = None;

        loop {
            match parser.next() {
                Ok(XmlEvent::StartElement {
                    name,
                    attributes,
                    namespace,
                }) => {
                    if name.local_name == "PP_R" {
                        data_type = UPF::R;
                    }

                    if name.local_name == "PP_RAB" {
                        data_type = UPF::RAB;
                    }

                    if name.local_name == "PP_LOCAL" {
                        data_type = UPF::LOCAL;
                    }

                    if name.local_name.contains("PP_BETA") {
                        data_type = UPF::BETA;
                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "angular_momentum")
                        {
                            self.lbeta
                                .push(parse_attr("angular_momentum", &attr_id.value)?);
                        }
                    }

                    if name.local_name.contains("PP_CHI") {
                        data_type = UPF::CHI;
                        current_chi_l =
                            match attributes.iter().find(|attr| attr.name.local_name == "l") {
                                Some(attr) => Some(parse_attr("l", &attr.value)?),
                                None => None,
                            };
                    }

                    if name.local_name == "PP_NLCC" {
                        data_type = UPF::NLCC;
                    }

                    if name.local_name == "PP_DIJ" {
                        data_type = UPF::DIJ;
                    }

                    if name.local_name == "PP_RHOATOM" {
                        data_type = UPF::RHOATOM;
                    }

                    if name.local_name == "PP_HEADER" {
                        data_type = UPF::HEADER;

                        for attr in attributes.iter() {
                            let value = &attr.value;
                            match attr.name.local_name.as_str() {
                                "element" => self.element = value.clone(),
                                "z_valence" => self.zion = parse_attr("z_valence", value)?,
                                "l_max" => self.lmax = parse_attr("l_max", value)?,
                                "l_local" => self.lloc = parse_attr("l_local", value)?,
                                "core_correction" => {
                                    self.core_correction = value.trim().to_lowercase() == "t"
                                }
                                "number_of_proj" => {
                                    self.nbeta = parse_attr("number_of_proj", value)?
                                }
                                "mesh_size" => {
                                    self.mmax = parse_attr("mesh_size", value)?;
                                    self.rhocore = vec![0.0; self.mmax];
                                }
                                _ => {}
                            }
                        }

                        data_type = UPF::NULL;
                    }
                }

                Ok(XmlEvent::Characters(text)) => {
                    match data_type {
                        UPF::R => self.rad = parse_block("PP_R", &text, 1.0)?,
                        UPF::RAB => self.rab = parse_block("PP_RAB", &text, 1.0)?,
                        UPF::LOCAL => self.vloc = parse_block("PP_LOCAL", &text, RY_TO_HA)?,
                        UPF::BETA => self.vbeta.push(parse_block("PP_BETA", &text, RY_TO_HA)?),
                        UPF::CHI => {
                            if let Some(l) = current_chi_l {
                                let chi = parse_block("PP_CHI", &text, 1.0)?;
                                if !chi.is_empty() {
                                    self.chi.entry(l).or_insert(chi);
                                }
                            }
                        }
                        UPF::DIJ => self.read_dij(&text)?,
                        UPF::RHOATOM => self.rhoatom = parse_block("PP_RHOATOM", &text, 1.0)?,
                        UPF::NLCC => self.rhocore = parse_block("PP_NLCC", &text, 1.0)?,
                        UPF::HEADER | UPF::NULL => {}
                    }
                    data_type = UPF::NULL;
                }

                Ok(XmlEvent::EndElement { name, .. }) => {
                    if name.local_name == "UPF" {
                        return Ok(());
                    }
                }

                Ok(XmlEvent::EndDocument) => {
                    return Err("file ends before </UPF>; expected a UPF v2 (XML) file".to_string());
                }

                Err(err) => {
                    return Err(format!("not a valid UPF v2 (XML) file: {}", err));
                }

                _ => {}
            }
        }
    }

    fn read_dij(&mut self, str_dij: &str) -> Result<(), String> {
        let v: Vec<f64> = parse_block("PP_DIJ", str_dij, 1.0)?
            .iter()
            .map(|x| x / RY_TO_HA)
            .collect();
        check_len("PP_DIJ", v.len(), self.nbeta * self.nbeta)?;

        self.dij = Matrix::<f64>::from_row_slice(self.nbeta, self.nbeta, &v);

        Ok(())
    }

    // Catch inconsistent files here rather than as an index panic later in the run.
    fn check(&self) -> Result<(), String> {
        check_len("PP_R", self.rad.len(), self.mmax)?;
        check_len("PP_RAB", self.rab.len(), self.mmax)?;
        check_len("PP_LOCAL", self.vloc.len(), self.mmax)?;
        check_len("PP_BETA blocks", self.vbeta.len(), self.nbeta)?;
        check_len("PP_BETA angular_momentum", self.lbeta.len(), self.nbeta)?;
        check_len("PP_DIJ rows", self.dij.nrow(), self.nbeta)?;
        Ok(())
    }
}

impl fmt::Display for AtomPSPUPF {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            " element = {:?} zion = {} lmax = {} lloc = {}",
            self.element, self.zion, self.lmax, self.lloc
        )?;
        writeln!(f, " mmax = {} nbeta = {}", self.mmax, self.nbeta)?;
        write!(f, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si_sr_path() -> String {
        format!("{}/src/Si-sr.upf", env!("CARGO_MANIFEST_DIR"))
    }

    // Write `content` to a temporary file and try to load it.
    fn load_text(name: &str, content: &str) -> Result<AtomPSPUPF, String> {
        let path = std::env::temp_dir().join(format!(
            "dftworks-atompsp-{}-{}.upf",
            name,
            std::process::id()
        ));
        std::fs::write(&path, content).unwrap();
        let result = AtomPSPUPF::from_file(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        result
    }

    #[test]
    fn test_load_si_sr() {
        let psp = AtomPSPUPF::from_file(&si_sr_path()).unwrap();

        assert_eq!(psp.get_zion(), 4.0);
        assert_eq!(psp.get_nbeta(), 4);
        assert_eq!(psp.get_rad().len(), psp.get_mmax());
        for ibeta in 0..psp.get_nbeta() {
            assert_eq!(psp.get_beta(ibeta).len(), psp.get_mmax());
        }
    }

    #[test]
    fn test_missing_file_is_an_error() {
        let err = AtomPSPUPF::from_file("no-such-file.upf").unwrap_err();
        assert!(err.contains("cannot open file"), "{}", err);
    }

    #[test]
    fn test_truncated_file_is_an_error() {
        let text = std::fs::read_to_string(si_sr_path()).unwrap();
        let err = load_text("truncated", &text[..text.len() / 2]).unwrap_err();
        assert!(err.contains("UPF v2"), "{}", err);
    }

    #[test]
    fn test_bad_number_is_an_error() {
        let text = std::fs::read_to_string(si_sr_path()).unwrap();
        let text = text.replacen("z_valence=\"", "z_valence=\"x", 1);
        let err = load_text("bad-number", &text).unwrap_err();
        assert!(err.contains("z_valence"), "{}", err);
    }
}
