#![allow(warnings)]

use crystal::Crystal;
use dwconsts::*;
use fftgrid::FFTGrid;
use pwdensity::PWDensity;
use rayon;
use std::time::{SystemTime, UNIX_EPOCH};
use symmetry::SymmetryDriver;

pub(crate) fn display_program_header() {
    println!();
    println!("   {:=^88}", "");
    println!("   {:^88}", "DFTWorks");
    println!(
        "   {:^88}",
        "Self-Consistent Plane-Wave Density Functional Theory"
    );
    println!("   {:=^88}", "");
    println!();
}

pub(crate) fn format_unix_seconds_as_utc_iso(timestamp_unix_s: u64) -> String {
    // Convert Unix epoch seconds to an ISO-like UTC timestamp without external crates.
    // Algorithm adapted from civil date conversion by Howard Hinnant.
    let days = (timestamp_unix_s / 86_400) as i64;
    let seconds_of_day = timestamp_unix_s % 86_400;

    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = mp + if mp < 10 { 3 } else { -9 }; // [1, 12]
    if month <= 2 {
        year += 1;
    }

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

pub(crate) fn display_system_information() {
    const OUT_WIDTH1: usize = 28;
    const OUT_WIDTH2: usize = 18;

    let backend = dwfft3d::backend_name();
    let fft_runtime = dwfft3d::backend_options();
    let mpi_rank = dwmpi::get_comm_world_rank();
    let mpi_ranks = dwmpi::get_comm_world_size();
    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let host_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let timestamp_unix_s = now.as_secs();
    let timestamp_utc = format_unix_seconds_as_utc_iso(timestamp_unix_s);

    println!("   {:-^88}", " system information ");
    println!();
    println!(
        "   {:<width1$} = {:>width2$}",
        "backend",
        backend,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "fft_threads",
        fft_runtime.threads,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "fft_planner",
        fft_runtime.planning_mode.as_str(),
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {}",
        "fft_wisdom_file",
        fft_runtime.wisdom_file.as_deref().unwrap_or("(none)"),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "hostname",
        hostname,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "os",
        std::env::consts::OS,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "arch",
        std::env::consts::ARCH,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "mpi_rank",
        mpi_rank,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {:>width2$}",
        "mpi_ranks",
        mpi_ranks,
        width1 = OUT_WIDTH1,
        width2 = OUT_WIDTH2
    );
    println!(
        "   {:<width1$} = {}",
        "timestamp_utc",
        timestamp_utc,
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:>4} (RAYON_NUM_THREADS={}, host_threads={})",
        "rayon_threads",
        rayon_threads,
        rayon_env,
        host_threads,
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {}",
        "working_directory",
        cwd,
        width1 = OUT_WIDTH1
    );
    println!();
}

pub(crate) fn display_grid_information(fftgrid: &FFTGrid, pwden: &PWDensity) {
    const OUT_WIDTH1: usize = 28;

    println!("   {:-^88}", " grid information ");
    println!();
    println!("   FFTGrid : {}", fftgrid);
    println!("   npw_rho = {}", pwden.get_n_plane_waves());
    println!(
        "   {:<width1$} = {}",
        "nfft",
        fftgrid.get_ntot(),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {}",
        "rho_gshells",
        pwden.get_n_gshell(),
        width1 = OUT_WIDTH1
    );
    println!(
        "   {:<width1$} = {:16.8} 1/bohr ({:16.8} 1/A)",
        "gmax_rho",
        pwden.get_gmax(),
        pwden.get_gmax() * ANG_TO_BOHR,
        width1 = OUT_WIDTH1
    );
}

pub(crate) fn display_symmetry_equivalent_atoms(crystal: &Crystal, symdrv: &dyn SymmetryDriver) {
    let sym_atom = symdrv.get_sym_atom();
    let natoms = sym_atom.len();
    let n_sym = symdrv.get_n_sym_ops();

    println!();
    println!("   {:-^88}", " symmetry-equivalent atoms ");
    println!("   mapping convention: atom(i) --sym_op--> atom(j), 1-based atom index");
    println!("   n_atoms = {}, n_sym_ops = {}", natoms, n_sym);

    let atom_species = crystal.get_atom_species();
    for (iat, mapping_row) in sym_atom.iter().enumerate() {
        let mapped_one_based: Vec<usize> = mapping_row
            .iter()
            .map(|&jat| jat.saturating_add(1))
            .collect();
        let species = atom_species.get(iat).map(|s| s.as_str()).unwrap_or("?");
        println!(
            "   atom {:>4} {:<6} -> {:?}",
            iat + 1,
            format!("({})", species),
            mapped_one_based
        );
    }

    let mut visited = vec![false; natoms];
    let mut classes: Vec<Vec<usize>> = Vec::new();

    for iat in 0..natoms {
        if visited[iat] {
            continue;
        }

        let mut class = vec![iat];
        if let Some(row) = sym_atom.get(iat) {
            for &jat in row.iter() {
                if jat < natoms {
                    class.push(jat);
                }
            }
        }
        class.sort_unstable();
        class.dedup();

        for &jat in class.iter() {
            visited[jat] = true;
        }
        classes.push(class);
    }

    println!("   equivalence classes ({} total)", classes.len());
    for (iclass, class) in classes.iter().enumerate() {
        let one_based: Vec<usize> = class.iter().map(|&iat| iat + 1).collect();
        println!("   class {:>4} -> {:?}", iclass + 1, one_based);
    }
}
