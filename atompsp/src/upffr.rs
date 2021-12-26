#![allow(warnings)]
use crate::AtomPSP;

use dwconsts::*;
use matrix::*;
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
    NLCC,
    DIJ,
    RHOATOM,
    NULL,
}

#[derive(Debug, Default)]
pub struct AtomPSPUPFFR {
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
    dij_j: Matrix<f64>,
    vbeta_j: Vec<Vec<f64>>,
    lbeta_j: Vec<usize>,
    vjjj: Vec<f64>,
    vlll: Vec<usize>,
    dij_soc: Matrix<f64>,
    vbeta_soc: Vec<Vec<f64>>,
    lbeta_soc: Vec<usize>,
}

impl AtomPSP for AtomPSPUPFFR {
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

    fn get_dfact(&self, ibeta: usize) -> f64 {
        self.dij[[ibeta, ibeta]]
    }

    fn get_nbeta_soc(&self) -> usize {
        self.lbeta_soc.len()
    }

    fn get_beta_soc(&self, ibeta: usize) -> &[f64] {
        &self.vbeta_soc[ibeta]
    }

    fn get_lbeta_soc(&self, ibeta: usize) -> usize {
        self.lbeta_soc[ibeta]
    }

    fn get_dfact_soc(&self, ibeta: usize) -> f64 {
        self.dij_soc[[ibeta, ibeta]]
    }

    fn get_zatom(&self) -> f64 {
        14.0
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

    fn get_vloc(&self) -> &[f64] {
        &self.vloc
    }

    fn read_file(&mut self, pspfile: &str) {
        self.parse_upf(pspfile);
        self.build_scalar_beta();
        self.build_soc_beta();
    }
}

impl AtomPSPUPFFR {
    pub fn new() -> AtomPSPUPFFR {
        AtomPSPUPFFR::default()
    }

    fn build_soc_beta(&mut self) {
        // build soc potential in beta from beta_j

        let nbeta = self.nbeta;

        let mut dij_soc = Vec::new();

        let mut jbeta = 0;

        for ibeta in 0..nbeta {
            let l = self.vlll[jbeta];

            if l == 0 {
                jbeta += 1;
            } else {
                let l_f64 = l as f64;
                let dminus = self.dij_j[[jbeta, jbeta]];
                let dplus = self.dij_j[[jbeta + 1, jbeta + 1]];

                let dpm = ((l_f64 + 1.0) * dplus + l_f64 * dminus) / (2.0 * l_f64 + 1.0);

                self.vbeta_soc.push(vec![0.0; self.mmax]);

                let vsoc: &mut Vec<f64> = self.vbeta_soc.last_mut().unwrap();

                for i in 0..self.mmax {
                    vsoc[i] = -self.vbeta_j[jbeta][i] * (dminus / dpm).sqrt()
                        + self.vbeta_j[jbeta + 1][i] * (dplus / dpm).sqrt();

                    vsoc[i] *= 2.0 / (2.0 * l_f64 + 1.0);
                }

                dij_soc.push(dpm);

                self.lbeta_soc.push(l);

                // l, l - 1/2
                jbeta += 1;

                // l, l + 1/2
                jbeta += 1;
            }
        }

        let nbeta = self.lbeta_soc.len();
        self.dij_soc = Matrix::<f64>::new(nbeta, nbeta);
        for ibeta in 0..nbeta {
            self.dij_soc[[ibeta, ibeta]] = dij_soc[ibeta];
        }
    }

    fn build_scalar_beta(&mut self) {
        // find out how many beta we need for scalar-relativistic potential

        let mut nbeta = 0;
        for (&l, &j) in self.vlll.iter().zip(self.vjjj.iter()) {
            nbeta += 1;

            if l > 0 && (j as f64 - (l as f64 + 0.5)).abs() < EPS5 {
                //println!("l = {}, j = {}", l, j);
                nbeta -= 1;
            }
        }

        self.nbeta = nbeta;

        self.lbeta = vec![0; self.nbeta];
        self.dij = Matrix::<f64>::new(nbeta, nbeta);

        let mut jbeta = 0;
        for ibeta in 0..nbeta {
            let l = self.vlll[jbeta];

            // println!(" l = {}, jbeta = {}", l, jbeta);

            if l == 0 {
                self.vbeta.push(self.vbeta_j[jbeta].clone());
                self.dij[[ibeta, ibeta]] = self.dij_j[[jbeta, jbeta]];
                self.lbeta[ibeta] = self.vlll[jbeta];

                jbeta += 1;
            } else {
                let l_f64 = l as f64;
                let dminus = self.dij_j[[jbeta, jbeta]];
                let dplus = self.dij_j[[jbeta + 1, jbeta + 1]];

                let dpm = ((l_f64 + 1.0) * dplus + l_f64 * dminus) / (2.0 * l_f64 + 1.0);
                //let dpm = 1.0;

                let fm = (dminus / dpm).sqrt();
                let fp = (dplus / dpm).sqrt();

                //println!("fp = {}, fm = {}, dpm = {}", fp, fm, dpm);

                self.vbeta.push(vec![0.0; self.mmax]);

                for i in 0..self.mmax {
                    self.vbeta[ibeta][i] = l_f64 * self.vbeta_j[jbeta][i] * fm
                        + (l_f64 + 1.0) * self.vbeta_j[jbeta + 1][i] * fp;

                    self.vbeta[ibeta][i] /= 2.0 * l_f64 + 1.0;
                }

                self.dij[[ibeta, ibeta]] = dpm;

                self.lbeta[ibeta] = self.vlll[jbeta];

                // l, l - 1/2
                jbeta += 1;

                // l, l + 1/2
                jbeta += 1;
            }
        }

        // println!("dij = {}", self.dij);
    }

    fn parse_upf(&mut self, pspfile: &str) {
        let file = File::open(pspfile).unwrap();
        let file = BufReader::new(file);

        let mut parser = xml::reader::EventReader::new(file);

        let mut data_type = UPF::NULL;

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
                        //println!("name = {:?}", attributes);
                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "angular_momentum")
                        {
                            self.lbeta_j.push(attr_id.value.trim().parse().unwrap());
                        }
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

                    if name.local_name.contains("PP_RELBETA") {
                        if let Some(attr_id) =
                            attributes.iter().find(|attr| attr.name.local_name == "lll")
                        {
                            self.vlll.push(attr_id.value.trim().parse().unwrap());
                        }

                        if let Some(attr_id) =
                            attributes.iter().find(|attr| attr.name.local_name == "jjj")
                        {
                            self.vjjj.push(attr_id.value.trim().parse().unwrap());
                        }
                    }

                    if name.local_name == "PP_HEADER" {
                        data_type = UPF::HEADER;

                        //println!("name = {:?}", attributes);

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "element")
                        {
                            self.element = attr_id.value.clone();
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "z_valence")
                        {
                            self.zion = attr_id.value.trim().parse().unwrap();
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "l_max")
                        {
                            self.lmax = attr_id.value.trim().parse().unwrap();
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "l_local")
                        {
                            self.lloc = attr_id.value.trim().parse().unwrap();
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "core_correction")
                        {
                            self.core_correction = attr_id.value.trim().to_lowercase() == "t";
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "number_of_proj")
                        {
                            self.nbeta = attr_id.value.trim().parse().unwrap();
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "mesh_size")
                        {
                            self.mmax = attr_id.value.trim().parse().unwrap();

                            self.rhocore = vec![0.0; self.mmax];
                        }

                        if let Some(attr_id) = attributes
                            .iter()
                            .find(|attr| attr.name.local_name == "relativistic")
                        {
                            self.relativistic = attr_id.value.trim().to_lowercase() == "full";
                        }

                        data_type = UPF::NULL;
                    }
                }

                Ok(XmlEvent::Characters(text)) => {
                    if matches!(data_type, UPF::R) {
                        self.read_rad(&text);
                        data_type = UPF::NULL;
                    }

                    if matches!(data_type, UPF::RAB) {
                        self.read_rab(&text);
                        data_type = UPF::NULL;
                    }

                    if matches!(data_type, UPF::LOCAL) {
                        self.read_vloc(&text);
                        data_type = UPF::NULL;
                    }

                    if matches!(data_type, UPF::BETA) {
                        self.read_beta_j(&text);
                        data_type = UPF::NULL;
                    }

                    if matches!(data_type, UPF::DIJ) {
                        self.read_dij_j(&text);
                        data_type = UPF::NULL;
                    }

                    if matches!(data_type, UPF::RHOATOM) {
                        self.read_rhoatom(&text);
                        data_type = UPF::NULL;
                    }

                    if matches!(data_type, UPF::NLCC) {
                        self.read_nlcc(&text);
                        data_type = UPF::NULL;
                    }
                }

                Ok(XmlEvent::EndElement { name, .. }) => {
                    if name.local_name == "UPF" {
                        break;
                    }
                }

                _ => {}
            }
        }
    }

    fn read_nlcc(&mut self, str_nlcc: &str) {
        self.rhocore = str_nlcc
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
    }

    fn read_rhoatom(&mut self, str_rho: &str) {
        self.rhoatom = str_rho
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
    }

    fn read_dij_j(&mut self, str_dij: &str) {
        let mut v = vec![0.0; self.nbeta * self.nbeta];
        v = str_dij
            .split_whitespace()
            .map(|x| x.parse::<f64>().unwrap() / RY_TO_HA)
            .collect();

        self.dij_j = Matrix::<f64>::from_row_slice(self.nbeta, self.nbeta, &v);
    }

    fn read_beta_j(&mut self, str_beta: &str) {
        let beta = str_beta
            .split_whitespace()
            .map(|x| x.parse::<f64>().unwrap() * RY_TO_HA)
            .collect();
        self.vbeta_j.push(beta);
    }

    fn read_rad(&mut self, str_r: &str) {
        self.rad = str_r
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
    }

    fn read_rab(&mut self, str_r: &str) {
        self.rab = str_r
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
    }

    fn read_vloc(&mut self, str_vloc: &str) {
        self.vloc = str_vloc
            .split_whitespace()
            .map(|x| x.parse::<f64>().unwrap() * RY_TO_HA)
            .collect();
    }
}

impl fmt::Display for AtomPSPUPFFR {
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

/// cargo test test_upffr_xml --lib -- --nocapture                                 

#[test]
fn test_upffr_xml() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/Si-fr.upf");

    let pspfile = d.to_str().unwrap();

    let mut atompsp = AtomPSPUPFFR::new();
    atompsp.read_file(&pspfile);

    println!("{}", atompsp);

    for i in 0..atompsp.get_nbeta() {
        let l = atompsp.get_lbeta(i);
        let beta = atompsp.get_beta(i);
        println!("i = {}, l = {}", i, l);
        println!("beta = {:?}", beta);
    }

    println!("rho = {:?}", atompsp.get_vloc());
}
