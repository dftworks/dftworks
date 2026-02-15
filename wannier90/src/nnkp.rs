use crate::types::{MeshTopology, Neighbor};
use crystal::Crystal;
use dwconsts::BOHR_TO_ANG;
use kpts::KPTS;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::Command;

pub(crate) fn build_topology_from_wannier90_pp(
    seedname: &str,
    nkpt: usize,
) -> io::Result<Option<MeshTopology>> {
    let output = match Command::new("wannier90.x")
        .arg("-pp")
        .arg(seedname)
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(io::Error::other(format!(
                "failed to run 'wannier90.x -pp {}': {}",
                seedname, err
            )));
        }
    };

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::other(format!(
            "'wannier90.x -pp {}' failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            seedname, output.status, stdout, stderr
        )));
    }

    let nnkp_filename = format!("{}.nnkp", seedname);
    let topology = parse_nnkp_topology(&nnkp_filename, nkpt)?;
    Ok(Some(topology))
}

pub(crate) fn write_nnkp_file(
    filename: &str,
    crystal: &Crystal,
    kpts: &dyn KPTS,
    topology: &MeshTopology,
) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    let a = crystal.get_latt().get_vector_a();
    let b = crystal.get_latt().get_vector_b();
    let c = crystal.get_latt().get_vector_c();
    let blatt = crystal.get_latt().reciprocal();
    let ba = blatt.get_vector_a();
    let bb = blatt.get_vector_b();
    let bc = blatt.get_vector_c();

    writeln!(writer, "begin real_lattice")?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        a.x * BOHR_TO_ANG,
        a.y * BOHR_TO_ANG,
        a.z * BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        b.x * BOHR_TO_ANG,
        b.y * BOHR_TO_ANG,
        b.z * BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        c.x * BOHR_TO_ANG,
        c.y * BOHR_TO_ANG,
        c.z * BOHR_TO_ANG
    )?;
    writeln!(writer, "end real_lattice")?;
    writeln!(writer)?;

    writeln!(writer, "begin recip_lattice")?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        ba.x / BOHR_TO_ANG,
        ba.y / BOHR_TO_ANG,
        ba.z / BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        bb.x / BOHR_TO_ANG,
        bb.y / BOHR_TO_ANG,
        bb.z / BOHR_TO_ANG
    )?;
    writeln!(
        writer,
        " {:22.14E} {:22.14E} {:22.14E}",
        bc.x / BOHR_TO_ANG,
        bc.y / BOHR_TO_ANG,
        bc.z / BOHR_TO_ANG
    )?;
    writeln!(writer, "end recip_lattice")?;
    writeln!(writer)?;

    writeln!(writer, "begin kpoints")?;
    writeln!(writer, "{}", kpts.get_n_kpts())?;
    for ik in 0..kpts.get_n_kpts() {
        let k = kpts.get_k_frac(ik);
        writeln!(writer, " {:18.12E} {:18.12E} {:18.12E}", k.x, k.y, k.z)?;
    }
    writeln!(writer, "end kpoints")?;
    writeln!(writer)?;

    writeln!(writer, "begin nnkpts")?;
    writeln!(writer, "{}", topology.nntot)?;
    for (ik, knn) in topology.neighbors.iter().enumerate() {
        for nn in knn.iter() {
            writeln!(
                writer,
                "{:8} {:8} {:4} {:4} {:4}",
                ik + 1,
                nn.ikb + 1,
                nn.gshift[0],
                nn.gshift[1],
                nn.gshift[2]
            )?;
        }
    }
    writeln!(writer, "end nnkpts")?;

    writer.flush()?;
    Ok(())
}

fn parse_nnkp_topology(filename: &str, nkpt: usize) -> io::Result<MeshTopology> {
    let file = File::open(filename)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;

    let mut cursor = 0usize;
    while cursor < lines.len() {
        if lines[cursor].trim().eq_ignore_ascii_case("begin nnkpts") {
            cursor += 1;
            break;
        }
        cursor += 1;
    }

    if cursor >= lines.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("'{}' does not contain a 'begin nnkpts' block", filename),
        ));
    }

    let nntot_line = next_data_line(&lines, &mut cursor).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("'{}' is missing nntot in nnkpts block", filename),
        )
    })?;
    let nntot = nntot_line
        .split_whitespace()
        .next()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid nntot line in '{}': '{}'", filename, nntot_line),
            )
        })?
        .parse::<usize>()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid nntot value in '{}': '{}'", filename, nntot_line),
            )
        })?;

    if nntot == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("nntot must be > 0 in '{}'", filename),
        ));
    }

    let mut neighbors = vec![Vec::with_capacity(nntot); nkpt];
    for _ in 0..(nkpt * nntot) {
        let line = next_data_line(&lines, &mut cursor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unexpected end of nnkpts entries in '{}'; expected {} entries",
                    filename,
                    nkpt * nntot
                ),
            )
        })?;

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid nnkpts entry in '{}': '{}'", filename, line),
            ));
        }

        let ik = fields[0].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid ik index in '{}': '{}'", filename, line),
            )
        })?;
        let ikb = fields[1].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid ikb index in '{}': '{}'", filename, line),
            )
        })?;
        if ik == 0 || ik > nkpt || ikb == 0 || ikb > nkpt {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("nnkpts index out of range in '{}': '{}'", filename, line),
            ));
        }

        let g1 = fields[2].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid g1 in '{}': '{}'", filename, line),
            )
        })?;
        let g2 = fields[3].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid g2 in '{}': '{}'", filename, line),
            )
        })?;
        let g3 = fields[4].parse::<i32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid g3 in '{}': '{}'", filename, line),
            )
        })?;

        neighbors[ik - 1].push(Neighbor {
            ikb: ikb - 1,
            gshift: [g1, g2, g3],
        });
    }

    Ok(MeshTopology { neighbors, nntot })
}

fn next_data_line<'a>(lines: &'a [String], cursor: &mut usize) -> Option<&'a str> {
    while *cursor < lines.len() {
        let line = lines[*cursor].trim();
        *cursor += 1;
        if line.is_empty() || line.starts_with('!') {
            continue;
        }
        return Some(line);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_parse_nnkp_topology_reads_nnkpts_block() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = std::env::temp_dir()
            .join(format!("wannier90_test_{}.nnkp", now))
            .display()
            .to_string();

        let content = r#"
begin kpoints
2
0.0 0.0 0.0
0.5 0.0 0.0
end kpoints

begin nnkpts
2
1 2 0 0 0
1 2 -1 0 0
2 1 1 0 0
2 1 0 0 0
end nnkpts
"#;
        fs::write(&filename, content).unwrap();

        let topology = parse_nnkp_topology(&filename, 2).unwrap();
        fs::remove_file(&filename).unwrap();

        assert_eq!(topology.nntot, 2);
        assert_eq!(topology.neighbors.len(), 2);
        assert_eq!(topology.neighbors[0][0].ikb, 1);
        assert_eq!(topology.neighbors[0][1].gshift, [-1, 0, 0]);
        assert_eq!(topology.neighbors[1][0].ikb, 0);
        assert_eq!(topology.neighbors[1][0].gshift, [1, 0, 0]);
    }

    #[test]
    fn test_parse_nnkp_topology_errors_without_nnkpts_block() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = std::env::temp_dir()
            .join(format!("wannier90_test_{}.bad.nnkp", now))
            .display()
            .to_string();

        fs::write(&filename, "begin kpoints\n1\n0.0 0.0 0.0\nend kpoints\n").unwrap();
        let err = parse_nnkp_topology(&filename, 1).unwrap_err();
        fs::remove_file(&filename).unwrap();

        assert!(err
            .to_string()
            .contains("does not contain a 'begin nnkpts' block"));
    }
}
