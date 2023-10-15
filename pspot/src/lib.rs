use atompsp;
use atompsp::AtomPSP;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Default)]
pub struct PSPot {
    pots: HashMap<String, Box<dyn AtomPSP>>,
    atpsp_file: HashMap<String, String>,
}

impl PSPot {
    pub fn new(scheme: &str) -> PSPot {
        let pspfiles = get_psp_files("in.pot");

        let mut pots = HashMap::new();
        let mut atpsp_file = HashMap::new();

        for (sp, spfile) in pspfiles.iter() {
            let mut psp_one = atompsp::new(scheme);

            psp_one.read_file(&spfile);

            pots.insert(sp.clone(), psp_one);

            atpsp_file.insert(sp.clone(), spfile.clone());
        }

        PSPot { pots, atpsp_file }
    }

    pub fn get_psp(&self, sp: &str) -> &dyn AtomPSP {
        self.pots.get(sp).unwrap().as_ref()
    }

    pub fn get_max_lmax(&self) -> usize {
        let mut max_lmax = 0;

        for (_, atpsp) in self.pots.iter() {
            let lmax = atpsp.get_lmax();

            if lmax > max_lmax {
                max_lmax = lmax;
            }
        }

        max_lmax
    }

    pub fn display(&self) {
        if !dwmpi::is_root() {
            return;
        }

        for (sp, file) in self.atpsp_file.iter() {
            println!("   {} : {}", sp, file);
        }
    }
}

pub fn get_psp_files(inpfile: &str) -> HashMap<String, String> {
    let file = File::open(inpfile).unwrap();
    let lines = BufReader::new(file).lines();

    let mut pspmap = HashMap::new();

    for line in lines {
        let s: Vec<&str> = line.as_ref().unwrap().split_whitespace().collect();

        let specie = s[0].to_string();

        let psp = "pot/".to_owned() + &s[1].to_string();

        pspmap.insert(specie, psp);
    }

    pspmap
}
