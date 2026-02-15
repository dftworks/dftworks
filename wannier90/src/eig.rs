use dfttypes::VKEigenValue;
use dwconsts::HA_TO_EV;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub(crate) fn write_local_eig_part_files(
    seedname: &str,
    vkevals: &VKEigenValue,
    ik_first: usize,
) -> io::Result<()> {
    let rank = dwmpi::get_comm_world_rank();

    match vkevals {
        VKEigenValue::NonSpin(vkevals) => {
            let filename = format!("{}.eig.part.rank{}", seedname, rank);
            write_eig_part_file(&filename, vkevals, ik_first)?;
        }
        VKEigenValue::Spin(vkevals_up, vkevals_dn) => {
            let up_filename = format!("{}.up.eig.part.rank{}", seedname, rank);
            let dn_filename = format!("{}.dn.eig.part.rank{}", seedname, rank);
            write_eig_part_file(&up_filename, vkevals_up, ik_first)?;
            write_eig_part_file(&dn_filename, vkevals_dn, ik_first)?;
        }
    }

    Ok(())
}

pub(crate) fn merge_rank_parts<F>(
    out_file: &str,
    part_file_for_rank: &F,
    nrank: i32,
) -> io::Result<()>
where
    F: Fn(i32) -> String,
{
    let out = File::create(out_file)?;
    let mut writer = BufWriter::new(out);

    for rank in 0..nrank {
        let part_file = part_file_for_rank(rank);
        if Path::new(&part_file).exists() {
            let data = fs::read(&part_file)?;
            writer.write_all(&data)?;
        }
    }

    writer.flush()?;

    for rank in 0..nrank {
        let part_file = part_file_for_rank(rank);
        if Path::new(&part_file).exists() {
            fs::remove_file(part_file)?;
        }
    }

    Ok(())
}

fn write_eig_part_file(filename: &str, eigvals: &[Vec<f64>], ik_first: usize) -> io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for (ik_local, evals_at_k) in eigvals.iter().enumerate() {
        let ik_global = ik_first + ik_local + 1;
        for (iband, eval) in evals_at_k.iter().enumerate() {
            writeln!(
                writer,
                "{:6} {:6} {:22.14E}",
                iband + 1,
                ik_global,
                eval * HA_TO_EV
            )?;
        }
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_write_eig_part_file_converts_to_ev() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let filename = std::env::temp_dir()
            .join(format!("wannier90_test_{}.eig.part", now))
            .display()
            .to_string();

        let eigvals = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        write_eig_part_file(&filename, &eigvals, 2).unwrap();

        let content = fs::read_to_string(&filename).unwrap();
        fs::remove_file(&filename).unwrap();

        assert!(content.contains("     1      3"));
        assert!(content.contains("     2      4"));

        let first_line = content.lines().next().unwrap();
        let cols: Vec<&str> = first_line.split_whitespace().collect();
        let energy_ev = cols[2].parse::<f64>().unwrap();
        assert!((energy_ev - HA_TO_EV).abs() < 1.0e-10);
    }
}
