use std::{
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Debug, Default)]
pub struct MagMoment {
    starting_moment: Vec<f64>,
}

impl MagMoment {
    pub fn new() -> MagMoment {
        MagMoment::default()
    }

    pub fn get_starting_moment(&self) -> &[f64] {
        &self.starting_moment
    }

    pub fn read_file(&mut self, inpfile: &str) {
        let file = File::open(inpfile).unwrap();
        let lines = BufReader::new(file).lines();

        for (i, line) in lines.enumerate() {
            let s: Vec<&str> = line.as_ref().unwrap().split_whitespace().collect();
            match i {
                0 => {
                    let nvalue: usize = s[0].parse().unwrap();
                    self.starting_moment = vec![0.0; nvalue];
                }

                _ => {
                    if s.len() == 0 {
                        continue;
                    }

                    if s[0].chars().nth(0).unwrap() == '#' {
                        continue;
                    }

                    let iatom: usize = s[0].parse().unwrap();
                    let ipos = iatom - 1;
                    self.starting_moment[ipos] = s[1].parse().unwrap();
                }
            }
        }
    }

    pub fn display(&self) {
        println!("{:?}", self.starting_moment);
    }
}

// cargo test --lib test_magmom -- --nocapture
#[test]
fn test_magmom() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.magmom");
    println!("{:?}", d);
    let mut magmom = MagMoment::new();
    magmom.read_file(d.to_str().unwrap());
    magmom.display();
}
