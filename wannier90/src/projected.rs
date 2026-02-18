use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

const CACHE_MAGIC: &str = "DFTWORKS_PROJ_CACHE_V1";
const CACHE_SENTINEL: &str = "--binary--";
const MIN_DOS_DENOM: f64 = 1.0e-12;

#[derive(Debug, Clone)]
pub struct ProjectedConfig {
    pub seedname: String,
    pub sigma_ev: f64,
    pub ne: usize,
    pub emin_ev: Option<f64>,
    pub emax_ev: Option<f64>,
    pub input_dir: PathBuf,
    pub out_dir: PathBuf,
    pub cache_path: Option<PathBuf>,
    pub validation_tol: f64,
}

impl Default for ProjectedConfig {
    fn default() -> Self {
        Self {
            seedname: String::new(),
            sigma_ev: 0.15,
            ne: 2000,
            emin_ev: None,
            emax_ev: None,
            input_dir: PathBuf::from("."),
            out_dir: PathBuf::from("."),
            cache_path: None,
            validation_tol: 0.15,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ProjectedSummary {
    pub used_cache: bool,
    pub validation_passed: bool,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    pub generated_files: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AmnDims {
    num_bands: usize,
    num_kpts: usize,
    num_proj: usize,
}

#[derive(Debug, Clone)]
struct ProjectionWeights {
    dims: AmnDims,
    weights: Vec<f64>, // flattened as [ik][ib][ip]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    modified_sec: u64,
    modified_nsec: u32,
}

#[derive(Debug, Clone, Copy)]
struct CacheHeader {
    stamp: FileStamp,
    dims: AmnDims,
}

#[derive(Debug, Clone)]
struct ProjectorMeta {
    index: usize,
    atom_id: usize,
    family: String,
}

pub fn run_projected_analysis(config: &ProjectedConfig) -> io::Result<ProjectedSummary> {
    if config.seedname.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "seedname must not be empty",
        ));
    }
    if config.sigma_ev <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "sigma must be > 0",
        ));
    }
    if config.ne < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "ne must be >= 2",
        ));
    }
    if config.validation_tol < 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "validation tolerance must be >= 0",
        ));
    }

    fs::create_dir_all(&config.out_dir)?;

    let amn_path = config
        .input_dir
        .join(format!("{}.amn", config.seedname.trim()));
    let eig_path = config
        .input_dir
        .join(format!("{}.eig", config.seedname.trim()));
    let nnkp_path = config
        .input_dir
        .join(format!("{}.nnkp", config.seedname.trim()));
    let cache_path = config.cache_path.clone().unwrap_or_else(|| {
        config
            .out_dir
            .join(format!("{}.proj.cache", config.seedname.trim()))
    });

    let (mut proj_weights, used_cache) =
        load_projection_weights_with_cache(&amn_path, &cache_path)?;
    normalize_projection_weights(&mut proj_weights);
    let dims = proj_weights.dims;
    let energies = parse_eig_energies(&eig_path, dims)?;
    let projectors = parse_nnkp_projectors(&nnkp_path, dims.num_proj)?;

    let partition_groups = build_partition_groups(&projectors);
    let all_groups = build_all_groups(&projectors, &partition_groups);
    let group_state_weights = compute_group_state_weights(&proj_weights, &all_groups);
    let total_state_weights = compute_total_state_weights(&proj_weights);

    let (emin_ev, emax_ev, grid_step, energy_grid) = build_energy_grid(
        &energies,
        config.sigma_ev,
        config.ne,
        config.emin_ev,
        config.emax_ev,
    )?;

    let unit_state_weights = vec![1.0; dims.num_bands * dims.num_kpts];
    let dos_total = compute_dos_for_state_weights(
        &unit_state_weights,
        &energies,
        dims.num_bands,
        dims.num_kpts,
        config.sigma_ev,
        &energy_grid,
    );
    let dos_projected_total = compute_dos_for_state_weights(
        &total_state_weights,
        &energies,
        dims.num_bands,
        dims.num_kpts,
        config.sigma_ev,
        &energy_grid,
    );
    let mut dos_by_group: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (name, state_weights) in group_state_weights.iter() {
        dos_by_group.insert(
            name.clone(),
            compute_dos_for_state_weights(
                state_weights,
                &energies,
                dims.num_bands,
                dims.num_kpts,
                config.sigma_ev,
                &energy_grid,
            ),
        );
    }

    let mut dos_partition_sum = vec![0.0; config.ne];
    for name in partition_groups.keys() {
        if let Some(values) = dos_by_group.get(name) {
            for i in 0..values.len() {
                dos_partition_sum[i] += values[i];
            }
        }
    }

    let max_total_dos = dos_total.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    let rel_floor = (1.0e-6 * max_total_dos).max(MIN_DOS_DENOM);

    let mut max_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;
    for i in 0..config.ne {
        let abs = (dos_partition_sum[i] - dos_total[i]).abs();
        if abs > max_abs_diff {
            max_abs_diff = abs;
        }
        let denom = dos_total[i].abs().max(rel_floor);
        let rel = abs / denom;
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
    }
    let validation_passed = max_rel_diff <= config.validation_tol;

    let mut generated_files = Vec::new();

    let pdos_total_path = config.out_dir.join("pdos_total.dat");
    write_pdos_file(&pdos_total_path, &energy_grid, &dos_total)?;
    generated_files.push(path_to_string(&pdos_total_path));

    let pdos_projected_total_path = config.out_dir.join("pdos_projected_sum.dat");
    write_pdos_file(
        &pdos_projected_total_path,
        &energy_grid,
        &dos_projected_total,
    )?;
    generated_files.push(path_to_string(&pdos_projected_total_path));

    for (name, values) in dos_by_group.iter() {
        let filename = format!("pdos_{}.dat", sanitize_label(name));
        let path = config.out_dir.join(filename);
        write_pdos_file(&path, &energy_grid, values)?;
        generated_files.push(path_to_string(&path));
    }

    let fatband_total_path = config.out_dir.join("fatband_total.dat");
    write_fatband_file(
        &fatband_total_path,
        &energies,
        &total_state_weights,
        dims.num_bands,
        dims.num_kpts,
    )?;
    generated_files.push(path_to_string(&fatband_total_path));

    for (name, state_weights) in group_state_weights.iter() {
        let filename = format!("fatband_{}.dat", sanitize_label(name));
        let path = config.out_dir.join(filename);
        write_fatband_file(
            &path,
            &energies,
            state_weights,
            dims.num_bands,
            dims.num_kpts,
        )?;
        generated_files.push(path_to_string(&path));
    }

    let validation_path = config.out_dir.join("pdos_validation_report.txt");
    write_validation_report(
        &validation_path,
        config,
        dims,
        &total_state_weights,
        emin_ev,
        emax_ev,
        grid_step,
        max_abs_diff,
        max_rel_diff,
        validation_passed,
    )?;
    generated_files.push(path_to_string(&validation_path));

    Ok(ProjectedSummary {
        used_cache,
        validation_passed,
        max_abs_diff,
        max_rel_diff,
        generated_files,
    })
}

fn build_energy_grid(
    energies: &[f64],
    sigma_ev: f64,
    ne: usize,
    emin_ev_override: Option<f64>,
    emax_ev_override: Option<f64>,
) -> io::Result<(f64, f64, f64, Vec<f64>)> {
    let mut min_energy = f64::INFINITY;
    let mut max_energy = f64::NEG_INFINITY;
    for energy in energies.iter() {
        if *energy < min_energy {
            min_energy = *energy;
        }
        if *energy > max_energy {
            max_energy = *energy;
        }
    }
    if !min_energy.is_finite() || !max_energy.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no finite eigenvalues found in eig file",
        ));
    }

    let padding = 5.0 * sigma_ev;
    let emin_ev = emin_ev_override.unwrap_or(min_energy - padding);
    let emax_ev = emax_ev_override.unwrap_or(max_energy + padding);
    if emax_ev <= emin_ev {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "emax must be greater than emin",
        ));
    }

    let step = (emax_ev - emin_ev) / ((ne - 1) as f64);
    let mut grid = vec![0.0; ne];
    for i in 0..ne {
        grid[i] = emin_ev + (i as f64) * step;
    }

    Ok((emin_ev, emax_ev, step, grid))
}

fn compute_dos_for_state_weights(
    state_weights: &[f64], // flattened [ik][ib]
    energies: &[f64],      // flattened [ik][ib]
    num_bands: usize,
    num_kpts: usize,
    sigma_ev: f64,
    energy_grid: &[f64],
) -> Vec<f64> {
    let mut dos = vec![0.0; energy_grid.len()];
    let inv_nk = 1.0 / (num_kpts as f64);
    let gauss_norm = 1.0 / (sigma_ev * std::f64::consts::PI.sqrt());
    let emin = energy_grid[0];
    let emax = energy_grid[energy_grid.len() - 1];
    let de = (emax - emin) / ((energy_grid.len() - 1) as f64);
    let cutoff = 6.0 * sigma_ev;

    for ik in 0..num_kpts {
        for ib in 0..num_bands {
            let state_idx = ik * num_bands + ib;
            let weight = state_weights[state_idx];
            if weight.abs() < MIN_DOS_DENOM {
                continue;
            }
            let energy = energies[state_idx];

            let e_lo = (energy - cutoff).max(emin);
            let e_hi = (energy + cutoff).min(emax);
            let i_lo = (((e_lo - emin) / de).floor() as isize).max(0) as usize;
            let i_hi = (((e_hi - emin) / de).ceil() as isize).min((energy_grid.len() - 1) as isize)
                as usize;

            for ie in i_lo..=i_hi {
                let x = (energy_grid[ie] - energy) / sigma_ev;
                let value = (-x * x).exp() * gauss_norm * inv_nk * weight;
                dos[ie] += value;
            }
        }
    }

    dos
}

fn compute_total_state_weights(proj_weights: &ProjectionWeights) -> Vec<f64> {
    let dims = proj_weights.dims;
    let mut state_weights = vec![0.0; dims.num_bands * dims.num_kpts];

    for ik in 0..dims.num_kpts {
        for ib in 0..dims.num_bands {
            let state_idx = ik * dims.num_bands + ib;
            let mut sum = 0.0;
            for ip in 0..dims.num_proj {
                sum += proj_weights.weights[idx_w(dims, ik, ib, ip)];
            }
            state_weights[state_idx] = sum;
        }
    }

    state_weights
}

fn normalize_projection_weights(proj_weights: &mut ProjectionWeights) {
    let dims = proj_weights.dims;
    for ik in 0..dims.num_kpts {
        for ib in 0..dims.num_bands {
            let mut sum = 0.0;
            for ip in 0..dims.num_proj {
                sum += proj_weights.weights[idx_w(dims, ik, ib, ip)];
            }
            if sum <= MIN_DOS_DENOM {
                continue;
            }
            for ip in 0..dims.num_proj {
                let idx = idx_w(dims, ik, ib, ip);
                proj_weights.weights[idx] /= sum;
            }
        }
    }
}

fn compute_group_state_weights(
    proj_weights: &ProjectionWeights,
    groups: &BTreeMap<String, Vec<usize>>,
) -> BTreeMap<String, Vec<f64>> {
    let dims = proj_weights.dims;
    let mut out = BTreeMap::new();

    for (name, proj_indices) in groups.iter() {
        let mut state_weights = vec![0.0; dims.num_bands * dims.num_kpts];
        for ik in 0..dims.num_kpts {
            for ib in 0..dims.num_bands {
                let state_idx = ik * dims.num_bands + ib;
                let mut sum = 0.0;
                for ip in proj_indices.iter() {
                    sum += proj_weights.weights[idx_w(dims, ik, ib, *ip)];
                }
                state_weights[state_idx] = sum;
            }
        }
        out.insert(name.clone(), state_weights);
    }

    out
}

fn build_partition_groups(projectors: &[ProjectorMeta]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for proj in projectors.iter() {
        let key = format!("atom{}_{}", proj.atom_id, proj.family);
        groups.entry(key).or_default().push(proj.index);
    }
    groups
}

fn build_all_groups(
    projectors: &[ProjectorMeta],
    partition_groups: &BTreeMap<String, Vec<usize>>,
) -> BTreeMap<String, Vec<usize>> {
    let mut groups = partition_groups.clone();
    for proj in projectors.iter() {
        let key = format!("family_{}", proj.family);
        groups.entry(key).or_default().push(proj.index);
    }
    groups
}

fn write_pdos_file(path: &Path, energy_grid: &[f64], dos: &[f64]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# energy_ev dos")?;
    for i in 0..energy_grid.len() {
        writeln!(writer, "{:18.10E} {:18.10E}", energy_grid[i], dos[i])?;
    }
    writer.flush()?;
    Ok(())
}

fn write_fatband_file(
    path: &Path,
    energies: &[f64],      // [ik][ib]
    state_weights: &[f64], // [ik][ib]
    num_bands: usize,
    num_kpts: usize,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "# ik ib energy_ev weight")?;
    for ik in 0..num_kpts {
        for ib in 0..num_bands {
            let idx = ik * num_bands + ib;
            writeln!(
                writer,
                "{:6} {:6} {:18.10E} {:18.10E}",
                ik + 1,
                ib + 1,
                energies[idx],
                state_weights[idx]
            )?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_validation_report(
    path: &Path,
    config: &ProjectedConfig,
    dims: AmnDims,
    total_state_weights: &[f64],
    emin_ev: f64,
    emax_ev: f64,
    grid_step: f64,
    max_abs_diff: f64,
    max_rel_diff: f64,
    validation_passed: bool,
) -> io::Result<()> {
    let mut mean_state_weight = 0.0f64;
    let mut max_state_weight_error = 0.0f64;
    for w in total_state_weights.iter() {
        mean_state_weight += *w;
        let err = (w - 1.0).abs();
        if err > max_state_weight_error {
            max_state_weight_error = err;
        }
    }
    mean_state_weight /= total_state_weights.len() as f64;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "seed={}", config.seedname.trim())?;
    writeln!(writer, "num_bands={}", dims.num_bands)?;
    writeln!(writer, "num_kpoints={}", dims.num_kpts)?;
    writeln!(writer, "num_projectors={}", dims.num_proj)?;
    writeln!(writer, "sigma_ev={:.10E}", config.sigma_ev)?;
    writeln!(writer, "num_energy_points={}", config.ne)?;
    writeln!(writer, "energy_min_ev={:.10E}", emin_ev)?;
    writeln!(writer, "energy_max_ev={:.10E}", emax_ev)?;
    writeln!(writer, "energy_step_ev={:.10E}", grid_step)?;
    writeln!(
        writer,
        "validation_tolerance={:.10E}",
        config.validation_tol
    )?;
    writeln!(writer, "max_abs_diff={:.10E}", max_abs_diff)?;
    writeln!(writer, "max_rel_diff={:.10E}", max_rel_diff)?;
    writeln!(
        writer,
        "mean_state_projector_weight={:.10E}",
        mean_state_weight
    )?;
    writeln!(
        writer,
        "max_state_projector_weight_error={:.10E}",
        max_state_weight_error
    )?;
    writeln!(
        writer,
        "status={}",
        if validation_passed { "PASS" } else { "FAIL" }
    )?;
    writer.flush()?;
    Ok(())
}

fn parse_eig_energies(path: &Path, dims: AmnDims) -> io::Result<Vec<f64>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut energies = vec![f64::NAN; dims.num_bands * dims.num_kpts];

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.len() < 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid eig entry at {}:{} (expected at least 3 columns)",
                    path_to_string(path),
                    lineno + 1
                ),
            ));
        }

        let ib = parse_usize(cols[0], path, lineno + 1, "band index")?
            .checked_sub(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "band index must be >= 1"))?;
        let ik = parse_usize(cols[1], path, lineno + 1, "k-point index")?
            .checked_sub(1)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "k-point index must be >= 1")
            })?;
        let energy = parse_f64(cols[2], path, lineno + 1, "eigenvalue")?;

        if ib >= dims.num_bands || ik >= dims.num_kpts {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "out-of-range eig index at {}:{} -> band={}, k={} (expected bands=1..{}, k=1..{})",
                    path_to_string(path),
                    lineno + 1,
                    ib + 1,
                    ik + 1,
                    dims.num_bands,
                    dims.num_kpts
                ),
            ));
        }

        let idx = ik * dims.num_bands + ib;
        if energies[idx].is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "duplicate eig entry at {}:{} for band={}, k={}",
                    path_to_string(path),
                    lineno + 1,
                    ib + 1,
                    ik + 1
                ),
            ));
        }
        energies[idx] = energy;
    }

    if let Some((idx, _)) = energies.iter().enumerate().find(|(_, e)| !e.is_finite()) {
        let ik = idx / dims.num_bands;
        let ib = idx % dims.num_bands;
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "missing eig entry for band={}, k={} in '{}'",
                ib + 1,
                ik + 1,
                path_to_string(path)
            ),
        ));
    }

    Ok(energies)
}

fn parse_nnkp_projectors(path: &Path, expected_num_proj: usize) -> io::Result<Vec<ProjectorMeta>> {
    let file = File::open(path)?;
    let lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;

    let mut cursor = 0usize;
    while cursor < lines.len()
        && !lines[cursor]
            .trim()
            .eq_ignore_ascii_case("begin projections")
    {
        cursor += 1;
    }
    if cursor >= lines.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "'{}' does not contain a 'begin projections' block",
                path_to_string(path)
            ),
        ));
    }
    cursor += 1;

    let declared_num = parse_usize_line(&lines, &mut cursor, path, "number of projections")?;
    if declared_num != expected_num_proj {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "projection count mismatch in '{}': nnkp has {}, amn has {}",
                path_to_string(path),
                declared_num,
                expected_num_proj
            ),
        ));
    }

    let mut centers: Vec<[f64; 3]> = Vec::new();
    let mut projectors = Vec::with_capacity(expected_num_proj);

    for ip in 0..expected_num_proj {
        let header = next_non_empty_line(&lines, &mut cursor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "projection #{} in '{}' is missing its header line",
                    ip + 1,
                    path_to_string(path)
                ),
            )
        })?;

        let fields: Vec<&str> = header.split_whitespace().collect();
        if fields.len() < 6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid projection header in '{}': '{}'",
                    path_to_string(path),
                    header
                ),
            ));
        }

        let x = parse_f64(fields[0], path, cursor, "projection x")?;
        let y = parse_f64(fields[1], path, cursor, "projection y")?;
        let z = parse_f64(fields[2], path, cursor, "projection z")?;
        let l = parse_i32(fields[3], path, cursor, "projection l")?;

        let _mr = parse_i32(fields[4], path, cursor, "projection m_r")?;
        let _orientation = next_non_empty_line(&lines, &mut cursor).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "projection #{} in '{}' is missing its orientation line",
                    ip + 1,
                    path_to_string(path)
                ),
            )
        })?;

        let atom_id = find_or_insert_center(&mut centers, [x, y, z]);
        let family = projector_family_label(l);

        projectors.push(ProjectorMeta {
            index: ip,
            atom_id,
            family,
        });
    }

    Ok(projectors)
}

fn next_non_empty_line(lines: &[String], cursor: &mut usize) -> Option<String> {
    while *cursor < lines.len() {
        let line = lines[*cursor].trim().to_string();
        *cursor += 1;
        if !line.is_empty() {
            return Some(line);
        }
    }
    None
}

fn parse_usize_line(
    lines: &[String],
    cursor: &mut usize,
    path: &Path,
    context: &str,
) -> io::Result<usize> {
    let line = next_non_empty_line(lines, cursor).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing '{}' line in '{}'", context, path_to_string(path)),
        )
    })?;
    parse_usize(&line, path, *cursor, context)
}

fn projector_family_label(l: i32) -> String {
    match l {
        -1 => "sp".to_string(),
        -2 => "sp2".to_string(),
        -3 => "sp3".to_string(),
        x if x < 0 => format!("hybrid{}", -x),
        0 => "s".to_string(),
        1 => "p".to_string(),
        2 => "d".to_string(),
        3 => "f".to_string(),
        x => format!("l{}", x),
    }
}

fn find_or_insert_center(centers: &mut Vec<[f64; 3]>, center: [f64; 3]) -> usize {
    const TOL2: f64 = 1.0e-14;
    for (i, existing) in centers.iter().enumerate() {
        let dx = center[0] - existing[0];
        let dy = center[1] - existing[1];
        let dz = center[2] - existing[2];
        if dx * dx + dy * dy + dz * dz <= TOL2 {
            return i + 1;
        }
    }
    centers.push(center);
    centers.len()
}

fn load_projection_weights_with_cache(
    amn_path: &Path,
    cache_path: &Path,
) -> io::Result<(ProjectionWeights, bool)> {
    let amn_stamp = file_stamp(amn_path)?;

    if cache_path.exists() {
        if let Ok((header, weights)) = read_cache_file(cache_path) {
            let expected_len = header.dims.num_bands * header.dims.num_kpts * header.dims.num_proj;
            if header.stamp == amn_stamp && weights.len() == expected_len {
                return Ok((
                    ProjectionWeights {
                        dims: header.dims,
                        weights,
                    },
                    true,
                ));
            }
        }
    }

    let parsed = parse_amn_weights(amn_path)?;
    write_cache_file(cache_path, amn_stamp, parsed.dims, &parsed.weights)?;
    Ok((parsed, false))
}

fn parse_amn_weights(path: &Path) -> io::Result<ProjectionWeights> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let _title = lines.next().transpose()?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("'{}' is empty", path_to_string(path)),
        )
    })?;

    let dims_line = lines.next().transpose()?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("'{}' is missing AMN dimensions line", path_to_string(path)),
        )
    })?;
    let dims_tokens: Vec<&str> = dims_line.split_whitespace().collect();
    if dims_tokens.len() < 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid AMN dimensions line in '{}': '{}'",
                path_to_string(path),
                dims_line
            ),
        ));
    }

    let dims = AmnDims {
        num_bands: dims_tokens[0].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid AMN num_bands in '{}'", path_to_string(path)),
            )
        })?,
        num_kpts: dims_tokens[1].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid AMN num_kpts in '{}'", path_to_string(path)),
            )
        })?,
        num_proj: dims_tokens[2].parse::<usize>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid AMN num_proj in '{}'", path_to_string(path)),
            )
        })?,
    };

    if dims.num_bands == 0 || dims.num_kpts == 0 || dims.num_proj == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "AMN dimensions must be positive in '{}'",
                path_to_string(path)
            ),
        ));
    }

    let total_entries = dims.num_bands * dims.num_kpts * dims.num_proj;
    let mut weights = vec![0.0f64; total_entries];
    let mut seen = vec![false; total_entries];

    for (line_idx, line) in lines.enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cols: Vec<&str> = trimmed.split_whitespace().collect();
        if cols.len() < 5 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid AMN entry at {}:{} (expected 5 columns)",
                    path_to_string(path),
                    line_idx + 3
                ),
            ));
        }

        let ib = parse_usize(cols[0], path, line_idx + 3, "band index")?
            .checked_sub(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "band index must be >= 1"))?;
        let ip = parse_usize(cols[1], path, line_idx + 3, "projector index")?
            .checked_sub(1)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "projector index must be >= 1")
            })?;
        let ik = parse_usize(cols[2], path, line_idx + 3, "k-point index")?
            .checked_sub(1)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "k-point index must be >= 1")
            })?;
        let re = parse_f64(cols[3], path, line_idx + 3, "overlap real part")?;
        let im = parse_f64(cols[4], path, line_idx + 3, "overlap imaginary part")?;

        if ib >= dims.num_bands || ip >= dims.num_proj || ik >= dims.num_kpts {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "out-of-range AMN index at {}:{} (band={}, proj={}, k={})",
                    path_to_string(path),
                    line_idx + 3,
                    ib + 1,
                    ip + 1,
                    ik + 1
                ),
            ));
        }

        let idx = idx_w(dims, ik, ib, ip);
        if seen[idx] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "duplicate AMN entry at {}:{} for (band={}, proj={}, k={})",
                    path_to_string(path),
                    line_idx + 3,
                    ib + 1,
                    ip + 1,
                    ik + 1
                ),
            ));
        }
        seen[idx] = true;
        weights[idx] = re * re + im * im;
    }

    if let Some((idx, _)) = seen.iter().enumerate().find(|(_, value)| !**value) {
        let per_k = dims.num_bands * dims.num_proj;
        let ik = idx / per_k;
        let rem = idx % per_k;
        let ib = rem / dims.num_proj;
        let ip = rem % dims.num_proj;
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "missing AMN entry in '{}': band={}, proj={}, k={}",
                path_to_string(path),
                ib + 1,
                ip + 1,
                ik + 1
            ),
        ));
    }

    Ok(ProjectionWeights { dims, weights })
}

fn idx_w(dims: AmnDims, ik: usize, ib: usize, ip: usize) -> usize {
    (ik * dims.num_bands + ib) * dims.num_proj + ip
}

fn write_cache_file(
    path: &Path,
    stamp: FileStamp,
    dims: AmnDims,
    weights: &[f64],
) -> io::Result<()> {
    let payload_bytes = weights.len() * std::mem::size_of::<f64>();

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", CACHE_MAGIC)?;
    writeln!(writer, "amn_size={}", stamp.size)?;
    writeln!(writer, "amn_mtime_sec={}", stamp.modified_sec)?;
    writeln!(writer, "amn_mtime_nsec={}", stamp.modified_nsec)?;
    writeln!(writer, "num_bands={}", dims.num_bands)?;
    writeln!(writer, "num_kpoints={}", dims.num_kpts)?;
    writeln!(writer, "num_projectors={}", dims.num_proj)?;
    writeln!(writer, "payload_bytes={}", payload_bytes)?;
    writeln!(writer, "{}", CACHE_SENTINEL)?;
    for value in weights.iter() {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn read_cache_file(path: &Path) -> io::Result<(CacheHeader, Vec<f64>)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;
    if first_line.trim() != CACHE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid cache magic in '{}'", path_to_string(path)),
        ));
    }

    let mut map: HashMap<String, String> = HashMap::new();
    loop {
        let mut line = String::new();
        let nread = reader.read_line(&mut line)?;
        if nread == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cache sentinel missing in '{}'", path_to_string(path)),
            ));
        }
        let trimmed = line.trim();
        if trimmed == CACHE_SENTINEL {
            break;
        }
        if trimmed.is_empty() {
            continue;
        }
        let (key, value) = trimmed.split_once('=').ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid cache header line '{}' in '{}'",
                    trimmed,
                    path_to_string(path)
                ),
            )
        })?;
        map.insert(key.to_string(), value.to_string());
    }

    let stamp = FileStamp {
        size: parse_map_u64(&map, "amn_size", path)?,
        modified_sec: parse_map_u64(&map, "amn_mtime_sec", path)?,
        modified_nsec: parse_map_u32(&map, "amn_mtime_nsec", path)?,
    };
    let dims = AmnDims {
        num_bands: parse_map_usize(&map, "num_bands", path)?,
        num_kpts: parse_map_usize(&map, "num_kpoints", path)?,
        num_proj: parse_map_usize(&map, "num_projectors", path)?,
    };
    let payload_bytes = parse_map_usize(&map, "payload_bytes", path)?;
    if payload_bytes % std::mem::size_of::<f64>() != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid payload size {} in cache '{}'",
                payload_bytes,
                path_to_string(path)
            ),
        ));
    }

    let mut payload = vec![0u8; payload_bytes];
    reader.read_exact(&mut payload)?;
    let mut weights = Vec::with_capacity(payload_bytes / std::mem::size_of::<f64>());
    for chunk in payload.chunks_exact(std::mem::size_of::<f64>()) {
        let mut bytes = [0u8; std::mem::size_of::<f64>()];
        bytes.copy_from_slice(chunk);
        weights.push(f64::from_le_bytes(bytes));
    }

    Ok((CacheHeader { stamp, dims }, weights))
}

fn parse_map_usize(map: &HashMap<String, String>, key: &str, path: &Path) -> io::Result<usize> {
    let raw = map.get(key).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing '{}' in cache '{}'", key, path_to_string(path)),
        )
    })?;
    raw.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid usize value for '{}' in cache '{}': '{}'",
                key,
                path_to_string(path),
                raw
            ),
        )
    })
}

fn parse_map_u64(map: &HashMap<String, String>, key: &str, path: &Path) -> io::Result<u64> {
    let raw = map.get(key).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing '{}' in cache '{}'", key, path_to_string(path)),
        )
    })?;
    raw.parse::<u64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid u64 value for '{}' in cache '{}': '{}'",
                key,
                path_to_string(path),
                raw
            ),
        )
    })
}

fn parse_map_u32(map: &HashMap<String, String>, key: &str, path: &Path) -> io::Result<u32> {
    let raw = map.get(key).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing '{}' in cache '{}'", key, path_to_string(path)),
        )
    })?;
    raw.parse::<u32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid u32 value for '{}' in cache '{}': '{}'",
                key,
                path_to_string(path),
                raw
            ),
        )
    })
}

fn file_stamp(path: &Path) -> io::Result<FileStamp> {
    let metadata = fs::metadata(path)?;
    let modified = metadata.modified().unwrap_or(UNIX_EPOCH);
    let duration = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
    Ok(FileStamp {
        size: metadata.len(),
        modified_sec: duration.as_secs(),
        modified_nsec: duration.subsec_nanos(),
    })
}

fn parse_usize(token: &str, path: &Path, lineno: usize, field: &str) -> io::Result<usize> {
    token.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid {} at {}:{} -> '{}'",
                field,
                path_to_string(path),
                lineno,
                token
            ),
        )
    })
}

fn parse_i32(token: &str, path: &Path, lineno: usize, field: &str) -> io::Result<i32> {
    token.parse::<i32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid {} at {}:{} -> '{}'",
                field,
                path_to_string(path),
                lineno,
                token
            ),
        )
    })
}

fn parse_f64(token: &str, path: &Path, lineno: usize, field: &str) -> io::Result<f64> {
    token.parse::<f64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid {} at {}:{} -> '{}'",
                field,
                path_to_string(path),
                lineno,
                token
            ),
        )
    })
}

fn sanitize_label(name: &str) -> String {
    let mut out = String::new();
    let mut prev_was_underscore = false;
    for ch in name.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            out.push(lowered);
            prev_was_underscore = false;
        } else if !prev_was_underscore {
            out.push('_');
            prev_was_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "group".to_string()
    } else {
        trimmed
    }
}

fn path_to_string(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn make_temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("dftworks_{}_{}", tag, nanos));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_fixture_files(dir: &Path, seed: &str) {
        let amn = dir.join(format!("{}.amn", seed));
        let eig = dir.join(format!("{}.eig", seed));
        let nnkp = dir.join(format!("{}.nnkp", seed));

        let mut amn_file = BufWriter::new(File::create(amn).unwrap());
        writeln!(amn_file, "Generated by test").unwrap();
        writeln!(amn_file, " 2 2 2").unwrap();
        // k=1
        writeln!(amn_file, " 1 1 1  8.366600265341E-01 0.0").unwrap(); // sqrt(0.7)
        writeln!(amn_file, " 2 1 1  4.472135954999E-01 0.0").unwrap(); // sqrt(0.2)
        writeln!(amn_file, " 1 2 1  5.477225575052E-01 0.0").unwrap(); // sqrt(0.3)
        writeln!(amn_file, " 2 2 1  8.944271909999E-01 0.0").unwrap(); // sqrt(0.8)
                                                                       // k=2
        writeln!(amn_file, " 1 1 2  7.745966692415E-01 0.0").unwrap(); // sqrt(0.6)
        writeln!(amn_file, " 2 1 2  7.071067811865E-01 0.0").unwrap(); // sqrt(0.5)
        writeln!(amn_file, " 1 2 2  6.324555320337E-01 0.0").unwrap(); // sqrt(0.4)
        writeln!(amn_file, " 2 2 2  7.071067811865E-01 0.0").unwrap(); // sqrt(0.5)
        amn_file.flush().unwrap();

        let mut eig_file = BufWriter::new(File::create(eig).unwrap());
        writeln!(eig_file, " 1 1 -1.0").unwrap();
        writeln!(eig_file, " 2 1  1.0").unwrap();
        writeln!(eig_file, " 1 2 -0.5").unwrap();
        writeln!(eig_file, " 2 2  1.5").unwrap();
        eig_file.flush().unwrap();

        let mut nnkp_file = BufWriter::new(File::create(nnkp).unwrap());
        writeln!(nnkp_file, "begin projections").unwrap();
        writeln!(nnkp_file, " 2").unwrap();
        writeln!(nnkp_file, " 0.0 0.0 0.0 0 1 1").unwrap();
        writeln!(nnkp_file, " 0 0 1 1 0 0 1").unwrap();
        writeln!(nnkp_file, " 0.25 0.25 0.25 1 2 1").unwrap();
        writeln!(nnkp_file, " 0 0 1 1 0 0 1").unwrap();
        writeln!(nnkp_file, "end projections").unwrap();
        nnkp_file.flush().unwrap();
    }

    #[test]
    fn projected_analysis_writes_outputs_and_uses_cache() {
        let temp_dir = make_temp_dir("proj_level3");
        let seed = "toy";
        write_fixture_files(&temp_dir, seed);

        let cache = temp_dir.join("toy.proj.cache");
        let config = ProjectedConfig {
            seedname: seed.to_string(),
            sigma_ev: 0.2,
            ne: 120,
            emin_ev: Some(-2.0),
            emax_ev: Some(2.0),
            input_dir: temp_dir.clone(),
            out_dir: temp_dir.clone(),
            cache_path: Some(cache.clone()),
            validation_tol: 1.0e-6,
        };

        let first = run_projected_analysis(&config).unwrap();
        assert!(!first.used_cache);
        assert!(first.validation_passed);
        assert!(cache.exists());
        assert!(first
            .generated_files
            .iter()
            .any(|f| f.ends_with("pdos_total.dat")));
        assert!(first
            .generated_files
            .iter()
            .any(|f| f.ends_with("pdos_projected_sum.dat")));
        assert!(first
            .generated_files
            .iter()
            .any(|f| f.ends_with("fatband_total.dat")));

        let second = run_projected_analysis(&config).unwrap();
        assert!(second.used_cache);
        assert!(second.validation_passed);

        let report = fs::read_to_string(temp_dir.join("pdos_validation_report.txt")).unwrap();
        assert!(report.contains("status=PASS"));

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn sanitize_label_collapses_non_alnum() {
        assert_eq!(sanitize_label("atom1_sp3"), "atom1_sp3");
        assert_eq!(sanitize_label("family sp2"), "family_sp2");
        assert_eq!(sanitize_label("++"), "group");
    }
}
