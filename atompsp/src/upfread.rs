//! Reader for UPF v2 (XML) pseudopotential files, shared by the scalar-relativistic
//! (`upf`) and fully relativistic (`upf-fr`) loaders. It converts energies to Hartree
//! and checks that the blocks are consistent with the header; it does not combine
//! projectors. That is left to the loaders.

use dwconsts::*;
use types::*;

use std::{collections::HashMap, fs::File, io::BufReader, str::FromStr};

use xml::reader::{EventReader, XmlEvent};

/// Contents of one UPF v2 file.
pub(crate) struct UpfData {
    pub element: String,
    /// z_valence
    pub zion: f64,
    pub lmax: usize,
    pub lloc: i32,
    /// nonlinear core correction (PP_NLCC is present)
    pub core_correction: bool,
    /// number_of_proj
    pub nbeta: usize,
    /// mesh_size
    pub mmax: usize,
    pub rad: Vec<f64>,
    pub rab: Vec<f64>,
    /// local potential (Ha)
    pub vloc: Vec<f64>,
    /// projectors (PP_BETA, Ha) and their angular momentum
    pub beta: Vec<Vec<f64>>,
    pub lbeta: Vec<usize>,
    /// projector coefficients D_ij (Ha), nbeta x nbeta
    pub dij: Matrix<f64>,
    /// atomic wavefunctions (PP_CHI) by angular momentum
    pub chi: HashMap<usize, Vec<f64>>,
    pub rhocore: Vec<f64>,
    pub rhoatom: Vec<f64>,
    /// (l, j) of each projector from PP_RELBETA; empty in a scalar-relativistic file
    pub lll: Vec<usize>,
    pub jjj: Vec<f64>,
}

// Which data block the next text belongs to.
enum Block {
    R,
    Rab,
    Local,
    Beta,
    Chi,
    Nlcc,
    Dij,
    RhoAtom,
    None,
}

pub(crate) fn read_upf(pspfile: &str) -> Result<UpfData, String> {
    let mut upf = UpfData {
        element: String::new(),
        zion: 0.0,
        lmax: 0,
        lloc: 0,
        core_correction: false,
        nbeta: 0,
        mmax: 0,
        rad: Vec::new(),
        rab: Vec::new(),
        vloc: Vec::new(),
        beta: Vec::new(),
        lbeta: Vec::new(),
        dij: Matrix::<f64>::new(0, 0),
        chi: HashMap::new(),
        rhocore: Vec::new(),
        rhoatom: Vec::new(),
        lll: Vec::new(),
        jjj: Vec::new(),
    };

    parse(pspfile, &mut upf)?;

    // A file without PP_NLCC has no core charge.
    if upf.rhocore.is_empty() && !upf.core_correction {
        upf.rhocore = vec![0.0; upf.mmax];
    }

    check(&upf)?;

    Ok(upf)
}

fn parse(pspfile: &str, upf: &mut UpfData) -> Result<(), String> {
    let file = File::open(pspfile).map_err(|err| format!("cannot open file: {}", err))?;
    let mut parser = EventReader::new(BufReader::new(file));

    let mut block = Block::None;
    let mut chi_l: Option<usize> = None;

    loop {
        match parser.next() {
            Ok(XmlEvent::StartElement {
                name, attributes, ..
            }) => {
                let tag = name.local_name.as_str();
                let attr = |key: &str| {
                    attributes
                        .iter()
                        .find(|a| a.name.local_name == key)
                        .map(|a| a.value.as_str())
                };

                if tag == "PP_HEADER" {
                    if let Some(v) = attr("element") {
                        upf.element = v.trim().to_string();
                    }
                    if let Some(v) = attr("z_valence") {
                        upf.zion = parse_attr("z_valence", v)?;
                    }
                    if let Some(v) = attr("l_max") {
                        upf.lmax = parse_attr("l_max", v)?;
                    }
                    if let Some(v) = attr("l_local") {
                        upf.lloc = parse_attr("l_local", v)?;
                    }
                    if let Some(v) = attr("core_correction") {
                        upf.core_correction = parse_logical("core_correction", v)?;
                    }
                    if let Some(v) = attr("number_of_proj") {
                        upf.nbeta = parse_attr("number_of_proj", v)?;
                    }
                    if let Some(v) = attr("mesh_size") {
                        upf.mmax = parse_attr("mesh_size", v)?;
                    }
                    block = Block::None;
                } else if tag == "PP_R" {
                    block = Block::R;
                } else if tag == "PP_RAB" {
                    block = Block::Rab;
                } else if tag == "PP_LOCAL" {
                    block = Block::Local;
                } else if tag.starts_with("PP_BETA") {
                    if let Some(v) = attr("angular_momentum") {
                        upf.lbeta.push(parse_attr("angular_momentum", v)?);
                    }
                    block = Block::Beta;
                } else if tag.starts_with("PP_CHI") {
                    chi_l = match attr("l") {
                        Some(v) => Some(parse_attr("l", v)?),
                        None => None,
                    };
                    block = Block::Chi;
                } else if tag == "PP_NLCC" {
                    block = Block::Nlcc;
                } else if tag == "PP_DIJ" {
                    block = Block::Dij;
                } else if tag == "PP_RHOATOM" {
                    block = Block::RhoAtom;
                } else if tag.starts_with("PP_RELBETA") {
                    if let Some(v) = attr("lll") {
                        upf.lll.push(parse_attr("lll", v)?);
                    }
                    if let Some(v) = attr("jjj") {
                        upf.jjj.push(parse_attr("jjj", v)?);
                    }
                }
            }

            Ok(XmlEvent::Characters(text)) => {
                match block {
                    Block::R => upf.rad = parse_block("PP_R", &text, 1.0)?,
                    Block::Rab => upf.rab = parse_block("PP_RAB", &text, 1.0)?,
                    // UPF energies are in Rydberg.
                    Block::Local => upf.vloc = parse_block("PP_LOCAL", &text, RY_TO_HA)?,
                    Block::Beta => upf.beta.push(parse_block("PP_BETA", &text, RY_TO_HA)?),
                    Block::Dij => {
                        let v = parse_block("PP_DIJ", &text, HA_TO_RY)?;
                        check_len("PP_DIJ", v.len(), upf.nbeta * upf.nbeta)?;
                        upf.dij = Matrix::<f64>::from_row_slice(upf.nbeta, upf.nbeta, &v);
                    }
                    Block::Chi => {
                        if let Some(l) = chi_l {
                            let chi = parse_block("PP_CHI", &text, 1.0)?;
                            if !chi.is_empty() {
                                upf.chi.entry(l).or_insert(chi);
                            }
                        }
                    }
                    Block::RhoAtom => upf.rhoatom = parse_block("PP_RHOATOM", &text, 1.0)?,
                    Block::Nlcc => upf.rhocore = parse_block("PP_NLCC", &text, 1.0)?,
                    Block::None => {}
                }
                block = Block::None;
            }

            Ok(XmlEvent::EndElement { name }) => {
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

// Catch inconsistent files here rather than as an index panic later in the run.
fn check(upf: &UpfData) -> Result<(), String> {
    check_len("PP_R", upf.rad.len(), upf.mmax)?;
    check_len("PP_RAB", upf.rab.len(), upf.mmax)?;
    check_len("PP_LOCAL", upf.vloc.len(), upf.mmax)?;
    check_len("PP_RHOATOM", upf.rhoatom.len(), upf.mmax)?;
    check_len("PP_NLCC", upf.rhocore.len(), upf.mmax)?;

    check_len("PP_BETA blocks", upf.beta.len(), upf.nbeta)?;
    check_len("PP_BETA angular_momentum", upf.lbeta.len(), upf.nbeta)?;
    for (ibeta, beta) in upf.beta.iter().enumerate() {
        check_len(&format!("PP_BETA.{}", ibeta + 1), beta.len(), upf.mmax)?;
    }
    check_len("PP_DIJ rows", upf.dij.nrow(), upf.nbeta)?;

    Ok(())
}

// Parse one attribute value, e.g. z_valence="4.0" in PP_HEADER.
fn parse_attr<T: FromStr>(name: &str, value: &str) -> Result<T, String> {
    value
        .trim()
        .parse()
        .map_err(|_| format!("invalid value '{}' for attribute '{}'", value.trim(), name))
}

// Fortran logical, as written by different generators: T, F, .true., .false., true, false.
fn parse_logical(name: &str, value: &str) -> Result<bool, String> {
    match value.trim().trim_matches('.').to_lowercase().as_str() {
        "t" | "true" => Ok(true),
        "f" | "false" => Ok(false),
        _ => Err(format!(
            "invalid value '{}' for attribute '{}'; expected T or F",
            value.trim(),
            name
        )),
    }
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

pub(crate) fn check_len(what: &str, found: usize, expected: usize) -> Result<(), String> {
    if found != expected {
        return Err(format!(
            "{} has {} values, expected {}",
            what, found, expected
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si_sr_text() -> String {
        std::fs::read_to_string(format!("{}/src/Si-sr.upf", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    // Write `content` to a temporary file and read it.
    fn read_text(name: &str, content: &str) -> Result<UpfData, String> {
        let path = std::env::temp_dir().join(format!(
            "dftworks-upfread-{}-{}.upf",
            name,
            std::process::id()
        ));
        std::fs::write(&path, content).unwrap();
        let result = read_upf(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();
        result
    }

    #[test]
    fn test_missing_file_is_an_error() {
        let err = read_upf("no-such-file.upf").err().unwrap();
        assert!(err.contains("cannot open file"), "{}", err);
    }

    #[test]
    fn test_truncated_file_is_an_error() {
        let text = si_sr_text();
        let err = read_text("truncated", &text[..text.len() / 2])
            .err()
            .unwrap();
        assert!(err.contains("UPF v2"), "{}", err);
    }

    #[test]
    fn test_bad_number_is_an_error() {
        let text = si_sr_text().replacen("z_valence=\"", "z_valence=\"x", 1);
        let err = read_text("bad-number", &text).err().unwrap();
        assert!(err.contains("z_valence"), "{}", err);
    }

    #[test]
    fn test_short_beta_is_an_error() {
        // Drop the first number of PP_BETA.1.
        let text = si_sr_text();
        let tag = text.find("<PP_BETA.1").unwrap();
        let start = tag + text[tag..].find('>').unwrap() + 1;
        let first = start + text[start..].find(|c: char| !c.is_whitespace()).unwrap();
        let end = first + text[first..].find(char::is_whitespace).unwrap();
        let text = format!("{}{}", &text[..first], &text[end..]);

        let err = read_text("short-beta", &text).err().unwrap();
        assert!(err.contains("PP_BETA.1 has"), "{}", err);
    }

    #[test]
    fn test_core_correction_spellings() {
        for (value, expected) in [
            ("T", true),
            (".true.", true),
            ("false", false),
            ("F", false),
        ] {
            assert_eq!(
                parse_logical("core_correction", value),
                Ok(expected),
                "{}",
                value
            );
        }
        assert!(parse_logical("core_correction", "yes").is_err());
    }

    #[test]
    fn test_core_correction_without_nlcc_is_an_error() {
        // Si-sr.upf has no core correction; claiming one without PP_NLCC must fail.
        let text = si_sr_text();
        assert!(text.contains("core_correction=\"F\""));
        let text = text.replacen("core_correction=\"F\"", "core_correction=\".true.\"", 1);
        let err = read_text("nlcc", &text).err().unwrap();
        assert!(err.contains("PP_NLCC has 0 values"), "{}", err);
    }
}
