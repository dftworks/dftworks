use atompsp;
use atompsp::AtomPSP;
use control::PotScheme;

use std::{
    any::Any,
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    panic::{catch_unwind, AssertUnwindSafe},
};

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "unknown panic".to_string()
    }
}

#[derive(Default)]
pub struct PSPot {
    // Species -> parsed pseudopotential object.
    pots: HashMap<String, Box<dyn AtomPSP>>,
    // Species -> source filename (for diagnostics/display).
    atpsp_file: HashMap<String, String>,
}

impl PSPot {
    pub fn new(scheme: PotScheme) -> PSPot {
        match Self::try_new(scheme) {
            Ok(pots) => pots,
            Err(err) => panic!("{}", err),
        }
    }

    pub fn try_new(scheme: PotScheme) -> Result<PSPot, String> {
        // Read species-to-file map from in.pot and instantiate parser by scheme.
        let pspfiles = try_get_psp_files("in.pot")?;

        let mut pots = HashMap::new();
        let mut atpsp_file = HashMap::new();

        for (sp, spfile) in pspfiles.iter() {
            let psp_one = catch_unwind(AssertUnwindSafe(|| {
                let mut psp_one = atompsp::new(scheme.as_str());
                psp_one.read_file(spfile);
                psp_one
            }))
            .map_err(|payload| {
                format!(
                    "failed to load pseudopotential '{}' for species '{}': {}",
                    spfile,
                    sp,
                    panic_payload_to_string(payload)
                )
            })?;

            pots.insert(sp.clone(), psp_one);

            atpsp_file.insert(sp.clone(), spfile.clone());
        }

        Ok(PSPot { pots, atpsp_file })
    }

    pub fn get_psp(&self, sp: &str) -> &dyn AtomPSP {
        // Fast lookup by species symbol.
        self.pots.get(sp).unwrap().as_ref()
    }

    pub fn contains_species(&self, sp: &str) -> bool {
        self.pots.contains_key(sp)
    }

    pub fn get_max_lmax(&self) -> usize {
        // Global angular-momentum bound used for Y_lm table sizing.
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
        // Human-readable mapping for runtime logs.
        println!("   {:-^88}", " atom pseudopotentials ");
        println!();

        let mut entries: Vec<(&str, &str)> = self
            .atpsp_file
            .iter()
            .map(|(species, file)| (species.as_str(), file.as_str()))
            .collect();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (species, file) in entries {
            println!("   {:<8} : {}", species, file);
        }
        println!();
    }
}

pub fn get_psp_files(inpfile: &str) -> HashMap<String, String> {
    match try_get_psp_files(inpfile) {
        Ok(files) => files,
        Err(err) => panic!("{}", err),
    }
}

pub fn try_get_psp_files(inpfile: &str) -> Result<HashMap<String, String>, String> {
    // in.pot format: "<species> <filename>" per line.
    // Filenames are resolved under "pot/" relative directory.
    let file =
        File::open(inpfile).map_err(|err| format!("failed to read '{}': {}", inpfile, err))?;
    let lines = BufReader::new(file).lines();

    let mut pspmap = HashMap::new();

    for (line_idx, line_res) in lines.enumerate() {
        let line_no = line_idx + 1;
        let line = line_res
            .map_err(|err| format!("{}:{}: failed to read line: {}", inpfile, line_no, err))?;
        let content = line.split('#').next().unwrap_or(&line);
        let s: Vec<&str> = content.split_whitespace().collect();
        if s.is_empty() {
            continue;
        }
        if s.len() < 2 {
            return Err(format!(
                "{}:{}: expected '<species> <filename>'",
                inpfile, line_no
            ));
        }

        let specie = s[0].to_string();

        let psp = "pot/".to_owned() + &s[1].to_string();

        pspmap.insert(specie, psp);
    }

    if pspmap.is_empty() {
        return Err(format!(
            "{}: no pseudopotential mappings found; expected '<species> <filename>' lines",
            inpfile
        ));
    }

    Ok(pspmap)
}

#[cfg(test)]
mod tests {
    use super::try_get_psp_files;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        std::env::temp_dir().join(format!("dftworks-pspot-{}-{}", name, nanos))
    }

    #[test]
    fn test_try_get_psp_files_reports_invalid_line() {
        let path = temp_file("invalid.in.pot");
        fs::write(&path, "Si\n").expect("write temp in.pot");

        let err = try_get_psp_files(path.to_str().expect("temp path should be utf8"))
            .expect_err("invalid in.pot should fail");
        assert!(err.contains("expected '<species> <filename>'"));

        fs::remove_file(path).expect("remove temp in.pot");
    }
}
