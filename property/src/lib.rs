use std::collections::{BTreeMap, HashSet};
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

pub const SCHEMA_VERSION: &str = "property-v1";
pub const RY_TO_EV: f64 = 13.605_693_009;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DosOutputFormat {
    Dat,
    Csv,
    Json,
}

impl Default for DosOutputFormat {
    fn default() -> Self {
        DosOutputFormat::Dat
    }
}

#[derive(Clone, Debug)]
pub struct PropertyExportOptions {
    pub dos_sigma_ev: Option<f64>,
    pub dos_ne: Option<usize>,
    pub dos_emin_ev: Option<f64>,
    pub dos_emax_ev: Option<f64>,
    pub dos_format: DosOutputFormat,
    pub fermi_tol_ev: f64,
}

impl Default for PropertyExportOptions {
    fn default() -> Self {
        Self {
            dos_sigma_ev: None,
            dos_ne: None,
            dos_emin_ev: None,
            dos_emax_ev: None,
            dos_format: DosOutputFormat::Dat,
            fermi_tol_ev: 0.2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EnergySummary {
    pub harris_ry: f64,
    pub scf_ry: f64,
    pub delta_ev: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ForceEntry {
    pub atom_index: usize,
    pub species: String,
    pub fx_ev_per_a: f64,
    pub fy_ev_per_a: f64,
    pub fz_ev_per_a: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ParsedPwLog {
    pub energy: Option<EnergySummary>,
    pub fermi_ev: Option<f64>,
    pub forces: Vec<ForceEntry>,
    pub stress_total_kbar: Option<[[f64; 3]; 3]>,
    pub pw_total_seconds: Option<f64>,
    pub user_cpu_seconds: Option<f64>,
    pub system_cpu_seconds: Option<f64>,
    pub max_rss_kb: Option<u64>,
}

pub fn ry_to_ev(energy_ry: f64) -> f64 {
    energy_ry * RY_TO_EV
}

pub fn parse_pw_log(text: &str) -> ParsedPwLog {
    let lines: Vec<&str> = text.lines().collect();

    let mut parsed = ParsedPwLog::default();

    for line in lines.iter() {
        if let Some(energy) = parse_energy_row(line) {
            parsed.energy = Some(energy);
        }
    }
    parsed.fermi_ev = parse_last_fermi_ev(&lines);

    parsed.forces = parse_total_force_block(&lines);
    parsed.stress_total_kbar = parse_total_stress_block(&lines);
    parsed.pw_total_seconds = parse_pw_total_seconds(&lines);
    parsed.max_rss_kb = parse_max_rss_kb(&lines);

    let (user_cpu, system_cpu) = parse_cpu_times(&lines);
    parsed.user_cpu_seconds = user_cpu;
    parsed.system_cpu_seconds = system_cpu;

    parsed
}

pub fn export_stage_properties(
    run_dir: &Path,
    stage: &str,
    log_path: &Path,
    workflow_wall_seconds: f64,
) -> Result<(), String> {
    export_stage_properties_with_options(
        run_dir,
        stage,
        log_path,
        workflow_wall_seconds,
        &PropertyExportOptions::default(),
    )
}

pub fn export_stage_properties_with_options(
    run_dir: &Path,
    stage: &str,
    log_path: &Path,
    workflow_wall_seconds: f64,
    options: &PropertyExportOptions,
) -> Result<(), String> {
    let log = fs::read_to_string(log_path)
        .map_err(|err| format!("failed to read stage log '{}': {}", log_path.display(), err))?;

    let parsed = parse_pw_log(&log);

    let properties_dir = run_dir.join("properties");
    fs::create_dir_all(&properties_dir).map_err(|err| {
        format!(
            "failed to create properties directory '{}': {}",
            properties_dir.display(),
            err
        )
    })?;

    write_summary_json(
        &properties_dir.join("summary.json"),
        stage,
        log_path,
        &parsed,
        workflow_wall_seconds,
    )?;
    write_timings_csv(
        &properties_dir.join("timings.csv"),
        &parsed,
        workflow_wall_seconds,
    )?;

    if let Some(energy) = parsed.energy.as_ref() {
        write_energy_csv(&properties_dir.join("energy.csv"), energy)?;
    }

    if !parsed.forces.is_empty() {
        write_force_csv(&properties_dir.join("force.csv"), &parsed.forces)?;
    }

    if let Some(stress) = parsed.stress_total_kbar.as_ref() {
        write_stress_csv(&properties_dir.join("stress.csv"), stress)?;
    }

    if stage == "nscf" {
        export_level2_properties(run_dir, &properties_dir, &parsed, options)?;
    }

    Ok(())
}

fn parse_energy_row(line: &str) -> Option<EnergySummary> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() < 7 {
        return None;
    }

    let iter = tokens[0];
    if !iter.ends_with(':') {
        return None;
    }

    let iter_no_colon = &iter[..iter.len().saturating_sub(1)];
    if iter_no_colon.parse::<usize>().is_err() {
        return None;
    }

    let harris_ry = tokens[tokens.len() - 3].parse::<f64>().ok()?;
    let scf_ry = tokens[tokens.len() - 2].parse::<f64>().ok()?;
    let delta_ev = tokens[tokens.len() - 1].parse::<f64>().ok()?;

    Some(EnergySummary {
        harris_ry,
        scf_ry,
        delta_ev,
    })
}

fn parse_total_force_block(lines: &[&str]) -> Vec<ForceEntry> {
    let start = lines
        .iter()
        .position(|line| line.contains("total-force (cartesian) (eV/A)"));
    let mut out = Vec::new();

    let Some(mut idx) = start.map(|i| i + 1) else {
        return out;
    };

    while idx < lines.len() {
        let line = lines[idx].trim();
        idx += 1;

        if line.is_empty() {
            continue;
        }

        if line.starts_with('-') && !out.is_empty() {
            break;
        }

        if let Some(entry) = parse_force_line(line) {
            out.push(entry);
            continue;
        }

        if !out.is_empty() {
            break;
        }
    }

    out
}

fn parse_force_line(line: &str) -> Option<ForceEntry> {
    let (left, right) = line.split_once(':')?;
    let left_tokens: Vec<&str> = left.split_whitespace().collect();
    if left_tokens.len() < 2 {
        return None;
    }

    let atom_index = left_tokens[0].parse::<usize>().ok()?;
    let species = left_tokens[1].to_string();

    let nums = parse_f64_list(right);
    if nums.len() < 3 {
        return None;
    }

    Some(ForceEntry {
        atom_index,
        species,
        fx_ev_per_a: nums[0],
        fy_ev_per_a: nums[1],
        fz_ev_per_a: nums[2],
    })
}

fn parse_total_stress_block(lines: &[&str]) -> Option<[[f64; 3]; 3]> {
    let start = lines
        .iter()
        .position(|line| line.contains("stress (kbar)"))
        .map(|i| i + 1)?;

    let mut idx = start;
    while idx < lines.len() {
        if lines[idx].trim() == "total" {
            if idx + 3 >= lines.len() {
                return None;
            }

            let row0 = parse_stress_row(lines[idx + 1])?;
            let row1 = parse_stress_row(lines[idx + 2])?;
            let row2 = parse_stress_row(lines[idx + 3])?;
            return Some([row0, row1, row2]);
        }
        idx += 1;
    }

    None
}

fn parse_stress_row(line: &str) -> Option<[f64; 3]> {
    let start = line.find('|')?;
    let end = line.rfind('|')?;
    if end <= start {
        return None;
    }

    let body = &line[start + 1..end];
    let nums = parse_f64_list(body);
    if nums.len() < 3 {
        return None;
    }

    Some([nums[0], nums[1], nums[2]])
}

fn parse_pw_total_seconds(lines: &[&str]) -> Option<f64> {
    for line in lines.iter() {
        let trimmed = line.trim();
        if !trimmed.starts_with("Total") {
            continue;
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        for (idx, token) in tokens.iter().enumerate() {
            if token.eq_ignore_ascii_case("seconds") && idx > 0 {
                if let Ok(v) = tokens[idx - 1].parse::<f64>() {
                    return Some(v);
                }
            }
        }
    }

    None
}

fn parse_max_rss_kb(lines: &[&str]) -> Option<u64> {
    for line in lines.iter() {
        let lower = line.to_ascii_lowercase();
        if !lower.contains("maximum resident set size") {
            continue;
        }

        if let Some((_, rhs)) = line.split_once(':') {
            if let Some(v) = parse_first_u64(rhs) {
                return Some(v);
            }
        }

        if let Some(v) = parse_first_u64(line) {
            return Some(v);
        }
    }
    None
}

fn parse_cpu_times(lines: &[&str]) -> (Option<f64>, Option<f64>) {
    let mut user_cpu = None;
    let mut system_cpu = None;

    for line in lines.iter() {
        let lower = line.to_ascii_lowercase();

        if lower.contains("user time (seconds)") {
            if let Some((_, rhs)) = line.split_once(':') {
                user_cpu = rhs.trim().parse::<f64>().ok();
            }
            continue;
        }

        if lower.contains("system time (seconds)") {
            if let Some((_, rhs)) = line.split_once(':') {
                system_cpu = rhs.trim().parse::<f64>().ok();
            }
            continue;
        }

        if lower.contains(" real ") && lower.contains(" user ") && lower.contains(" sys ") {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            for window in tokens.windows(2) {
                let value = window[0].parse::<f64>().ok();
                let label = window[1].to_ascii_lowercase();
                if label == "user" {
                    user_cpu = value;
                } else if label == "sys" {
                    system_cpu = value;
                }
            }
        }
    }

    (user_cpu, system_cpu)
}

fn parse_f64_list(text: &str) -> Vec<f64> {
    text.split_whitespace()
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn parse_first_u64(text: &str) -> Option<u64> {
    text.split_whitespace()
        .find_map(|token| token.parse::<u64>().ok())
}

#[derive(Clone, Debug)]
struct EigChannel {
    label: String,
    energies_by_k: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct EigDataset {
    seed: String,
    spin_polarized: bool,
    nkpt: usize,
    channels: Vec<EigChannel>,
}

#[derive(Clone, Debug)]
struct BandEdge {
    energy_ev: f64,
    k_index: usize,
    channel: String,
}

#[derive(Clone, Debug)]
struct BandGapSummary {
    fermi_ev: f64,
    is_metal: bool,
    vbm: Option<BandEdge>,
    cbm: Option<BandEdge>,
    indirect_gap_ev: Option<f64>,
    direct_gap_ev: Option<f64>,
    direct_gap_k_index: Option<usize>,
}

#[derive(Clone, Debug)]
struct FermiConsistencySummary {
    scf_fermi_ev: Option<f64>,
    nscf_fermi_ev: Option<f64>,
    postprocess_fermi_ev: f64,
    tolerance_ev: f64,
    scf_vs_nscf_delta_ev: Option<f64>,
    scf_vs_nscf_pass: Option<bool>,
    postprocess_in_gap: Option<bool>,
}

fn export_level2_properties(
    run_dir: &Path,
    properties_dir: &Path,
    parsed: &ParsedPwLog,
    options: &PropertyExportOptions,
) -> Result<(), String> {
    let eig = discover_eig_dataset(run_dir)?;
    let (ctrl_sigma_ev, ctrl_ne) = read_dos_defaults_from_ctrl(&run_dir.join("in.ctrl"));

    let sigma_ev = options.dos_sigma_ev.or(ctrl_sigma_ev).unwrap_or(0.1);
    if !(sigma_ev.is_finite() && sigma_ev > 0.0) {
        return Err(format!(
            "invalid DOS sigma: expected a positive finite eV value, got {}",
            sigma_ev
        ));
    }

    let ne = options.dos_ne.or(ctrl_ne).unwrap_or(500);
    if ne < 2 {
        return Err(format!(
            "invalid DOS grid size: expected --dos-ne >= 2, got {}",
            ne
        ));
    }

    let dos_points = compute_total_dos(
        &eig,
        sigma_ev,
        ne,
        options.dos_emin_ev,
        options.dos_emax_ev,
    )?;

    let dos_dat_path = properties_dir.join("dos.dat");
    write_dos_dat(&dos_dat_path, &dos_points)?;

    match options.dos_format {
        DosOutputFormat::Dat => {}
        DosOutputFormat::Csv => {
            write_dos_csv(&properties_dir.join("dos.csv"), &dos_points)?;
        }
        DosOutputFormat::Json => {
            write_dos_json(&properties_dir.join("dos.json"), &dos_points)?;
        }
    }

    let nscf_fermi_ev = parsed.fermi_ev.ok_or_else(|| {
        "failed to parse Fermi level from NSCF log; required for band-gap analysis".to_string()
    })?;
    let gap = analyze_band_gap(&eig, nscf_fermi_ev);
    write_band_gap_json(&properties_dir.join("band_gap.json"), &eig, &gap)?;

    let scf_fermi_ev = resolve_source_scf_fermi_ev(run_dir);
    let fermi_consistency =
        build_fermi_consistency_summary(scf_fermi_ev, Some(nscf_fermi_ev), &gap, options.fermi_tol_ev);
    write_fermi_consistency_json(
        &properties_dir.join("fermi_consistency.json"),
        &fermi_consistency,
    )?;

    Ok(())
}

fn read_dos_defaults_from_ctrl(ctrl_path: &Path) -> (Option<f64>, Option<usize>) {
    let Ok(text) = fs::read_to_string(ctrl_path) else {
        return (None, None);
    };

    let mut sigma = None;
    let mut ne = None;
    for raw_line in text.lines() {
        let line = strip_comments(raw_line);
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        match key.as_str() {
            "dos_sigma" => sigma = parse_float_token(value),
            "dos_ne" => ne = value.parse::<usize>().ok(),
            _ => {}
        }
    }

    (sigma, ne)
}

fn discover_eig_dataset(run_dir: &Path) -> Result<EigDataset, String> {
    let entries = fs::read_dir(run_dir)
        .map_err(|err| format!("failed to read '{}': {}", run_dir.display(), err))?;

    let mut plain_seeds = Vec::<String>::new();
    let mut up_seeds = HashSet::<String>::new();
    let mut dn_seeds = HashSet::<String>::new();

    for entry in entries {
        let path = entry
            .map_err(|err| format!("failed to inspect run artifact entry: {}", err))?
            .path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
            continue;
        };
        if let Some(seed) = name.strip_suffix(".up.eig") {
            up_seeds.insert(seed.to_string());
            continue;
        }
        if let Some(seed) = name.strip_suffix(".dn.eig") {
            dn_seeds.insert(seed.to_string());
            continue;
        }
        if let Some(seed) = name.strip_suffix(".eig") {
            plain_seeds.push(seed.to_string());
        }
    }

    if !up_seeds.is_empty() || !dn_seeds.is_empty() {
        let mut spin_seeds: Vec<String> = up_seeds
            .intersection(&dn_seeds)
            .map(|s| s.to_string())
            .collect();
        spin_seeds.sort();

        if spin_seeds.is_empty() {
            return Err(format!(
                "found spin-resolved eig files in '{}' but no matching '<seed>.up.eig' and '<seed>.dn.eig' pair",
                run_dir.display()
            ));
        }
        if spin_seeds.len() > 1 {
            return Err(format!(
                "multiple spin eig seeds found in '{}': {}; keep one seed per run",
                run_dir.display(),
                spin_seeds.join(",")
            ));
        }

        let seed = spin_seeds[0].clone();
        let up = parse_eig_file(&run_dir.join(format!("{}.up.eig", seed)))?;
        let dn = parse_eig_file(&run_dir.join(format!("{}.dn.eig", seed)))?;
        if up.len() != dn.len() {
            return Err(format!(
                "spin eig channel mismatch for seed '{}': up nkpt={} vs dn nkpt={}",
                seed,
                up.len(),
                dn.len()
            ));
        }

        return Ok(EigDataset {
            seed,
            spin_polarized: true,
            nkpt: up.len(),
            channels: vec![
                EigChannel {
                    label: "up".to_string(),
                    energies_by_k: up,
                },
                EigChannel {
                    label: "dn".to_string(),
                    energies_by_k: dn,
                },
            ],
        });
    }

    plain_seeds.sort();
    plain_seeds.dedup();
    if plain_seeds.is_empty() {
        return Err(format!(
            "no eig file found in '{}' (expected '<seed>.eig' from NSCF with wannier90_export=true)",
            run_dir.display()
        ));
    }
    if plain_seeds.len() > 1 {
        return Err(format!(
            "multiple eig seeds found in '{}': {}; keep one seed per run",
            run_dir.display(),
            plain_seeds.join(",")
        ));
    }

    let seed = plain_seeds[0].clone();
    let nonspin = parse_eig_file(&run_dir.join(format!("{}.eig", seed)))?;
    let nkpt = nonspin.len();
    Ok(EigDataset {
        seed,
        spin_polarized: false,
        nkpt,
        channels: vec![EigChannel {
            label: "nonspin".to_string(),
            energies_by_k: nonspin,
        }],
    })
}

fn parse_eig_file(path: &Path) -> Result<Vec<Vec<f64>>, String> {
    let text =
        fs::read_to_string(path).map_err(|err| format!("failed to read '{}': {}", path.display(), err))?;

    let mut by_k: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            return Err(format!(
                "invalid eig line {} in '{}': expected '<band> <k> <energy_eV>'",
                line_no + 1,
                path.display()
            ));
        }

        let iband = fields[0].parse::<usize>().map_err(|err| {
            format!(
                "invalid band index at line {} in '{}': {}",
                line_no + 1,
                path.display(),
                err
            )
        })?;
        let ik = fields[1].parse::<usize>().map_err(|err| {
            format!(
                "invalid k-point index at line {} in '{}': {}",
                line_no + 1,
                path.display(),
                err
            )
        })?;
        let energy_ev = parse_float_token(fields[2]).ok_or_else(|| {
            format!(
                "invalid eigenvalue at line {} in '{}': '{}'",
                line_no + 1,
                path.display(),
                fields[2]
            )
        })?;

        by_k.entry(ik).or_default().push((iband, energy_ev));
    }

    if by_k.is_empty() {
        return Err(format!("eig file '{}' is empty", path.display()));
    }

    let nkpt = by_k.keys().copied().max().unwrap_or(0);
    let mut out = Vec::with_capacity(nkpt);
    let mut expected_bands = None::<usize>;

    for ik in 1..=nkpt {
        let mut kbands = by_k.remove(&ik).ok_or_else(|| {
            format!(
                "missing k-point {} in eig file '{}'; expected contiguous 1..{}",
                ik,
                path.display(),
                nkpt
            )
        })?;
        kbands.sort_by_key(|(ib, _)| *ib);

        for window in kbands.windows(2) {
            if window[0].0 == window[1].0 {
                return Err(format!(
                    "duplicate band index {} at k-point {} in '{}'",
                    window[0].0,
                    ik,
                    path.display()
                ));
            }
        }

        let energies: Vec<f64> = kbands.into_iter().map(|(_, e)| e).collect();
        if let Some(nbands) = expected_bands {
            if energies.len() != nbands {
                return Err(format!(
                    "inconsistent band count in '{}': k-point {} has {} bands, expected {}",
                    path.display(),
                    ik,
                    energies.len(),
                    nbands
                ));
            }
        } else {
            expected_bands = Some(energies.len());
        }
        out.push(energies);
    }

    Ok(out)
}

fn compute_total_dos(
    eig: &EigDataset,
    sigma_ev: f64,
    ne: usize,
    emin_override: Option<f64>,
    emax_override: Option<f64>,
) -> Result<Vec<(f64, f64)>, String> {
    if eig.nkpt == 0 {
        return Err("empty eig dataset: nkpt=0".to_string());
    }

    let k_weight = 1.0 / eig.nkpt as f64;
    let channel_weight = if eig.spin_polarized { 1.0 } else { 2.0 };

    let mut states = Vec::<(f64, f64)>::new();
    for channel in eig.channels.iter() {
        for bands in channel.energies_by_k.iter() {
            for &energy_ev in bands.iter() {
                states.push((energy_ev, channel_weight * k_weight));
            }
        }
    }

    if states.is_empty() {
        return Err("empty eig dataset: no states available".to_string());
    }

    let mut min_e = f64::INFINITY;
    let mut max_e = f64::NEG_INFINITY;
    for (energy_ev, _) in states.iter() {
        min_e = min_e.min(*energy_ev);
        max_e = max_e.max(*energy_ev);
    }

    let default_padding = (5.0 * sigma_ev).max(0.5);
    let emin = emin_override.unwrap_or(min_e - default_padding);
    let emax = emax_override.unwrap_or(max_e + default_padding);
    if !(emax.is_finite() && emin.is_finite() && emax > emin) {
        return Err(format!(
            "invalid DOS energy window: emin={}, emax={}",
            emin, emax
        ));
    }

    let norm = 1.0 / (sigma_ev * (2.0 * PI).sqrt());
    let mut out = Vec::with_capacity(ne);
    for i in 0..ne {
        let t = i as f64 / (ne - 1) as f64;
        let energy = emin + (emax - emin) * t;
        let mut dos = 0.0;
        for (state_energy, state_weight) in states.iter() {
            let x = (energy - state_energy) / sigma_ev;
            dos += state_weight * norm * (-0.5 * x * x).exp();
        }
        out.push((energy, dos));
    }

    Ok(out)
}

fn analyze_band_gap(eig: &EigDataset, fermi_ev: f64) -> BandGapSummary {
    const EPS: f64 = 1.0e-9;

    let mut global_vbm: Option<BandEdge> = None;
    let mut global_cbm: Option<BandEdge> = None;
    let mut direct_gap_ev: Option<f64> = None;
    let mut direct_gap_k_index: Option<usize> = None;

    for ik in 0..eig.nkpt {
        let mut vbm_k: Option<BandEdge> = None;
        let mut cbm_k: Option<BandEdge> = None;

        for channel in eig.channels.iter() {
            if ik >= channel.energies_by_k.len() {
                continue;
            }
            for &energy_ev in channel.energies_by_k[ik].iter() {
                if energy_ev <= fermi_ev + EPS {
                    let edge = BandEdge {
                        energy_ev,
                        k_index: ik + 1,
                        channel: channel.label.clone(),
                    };
                    if global_vbm
                        .as_ref()
                        .map(|current| edge.energy_ev > current.energy_ev)
                        .unwrap_or(true)
                    {
                        global_vbm = Some(edge.clone());
                    }
                    if vbm_k
                        .as_ref()
                        .map(|current| edge.energy_ev > current.energy_ev)
                        .unwrap_or(true)
                    {
                        vbm_k = Some(edge);
                    }
                } else {
                    let edge = BandEdge {
                        energy_ev,
                        k_index: ik + 1,
                        channel: channel.label.clone(),
                    };
                    if global_cbm
                        .as_ref()
                        .map(|current| edge.energy_ev < current.energy_ev)
                        .unwrap_or(true)
                    {
                        global_cbm = Some(edge.clone());
                    }
                    if cbm_k
                        .as_ref()
                        .map(|current| edge.energy_ev < current.energy_ev)
                        .unwrap_or(true)
                    {
                        cbm_k = Some(edge);
                    }
                }
            }
        }

        if let (Some(vbm), Some(cbm)) = (vbm_k.as_ref(), cbm_k.as_ref()) {
            let gap = cbm.energy_ev - vbm.energy_ev;
            if direct_gap_ev.map(|current| gap < current).unwrap_or(true) {
                direct_gap_ev = Some(gap);
                direct_gap_k_index = Some(ik + 1);
            }
        }
    }

    let indirect_gap_ev = match (global_vbm.as_ref(), global_cbm.as_ref()) {
        (Some(vbm), Some(cbm)) => Some(cbm.energy_ev - vbm.energy_ev),
        _ => None,
    };
    let is_metal = indirect_gap_ev.map(|gap| gap <= 1.0e-6).unwrap_or(true);

    BandGapSummary {
        fermi_ev,
        is_metal,
        vbm: global_vbm,
        cbm: global_cbm,
        indirect_gap_ev,
        direct_gap_ev,
        direct_gap_k_index,
    }
}

fn resolve_source_scf_fermi_ev(run_dir: &Path) -> Option<f64> {
    for marker in ["from_scf_run.txt", "from_source_run.txt"] {
        let marker_path = run_dir.join(marker);
        if !marker_path.is_file() {
            continue;
        }

        let source_dir = fs::read_to_string(&marker_path).ok()?;
        let source_dir = source_dir.trim();
        if source_dir.is_empty() {
            continue;
        }

        let source_log = Path::new(source_dir).join("out.pw.log");
        let source_text = fs::read_to_string(&source_log).ok()?;
        let source_lines: Vec<&str> = source_text.lines().collect();
        if let Some(fermi) = parse_last_fermi_ev(&source_lines) {
            return Some(fermi);
        }
    }

    None
}

fn build_fermi_consistency_summary(
    scf_fermi_ev: Option<f64>,
    nscf_fermi_ev: Option<f64>,
    gap: &BandGapSummary,
    tolerance_ev: f64,
) -> FermiConsistencySummary {
    let scf_vs_nscf_delta_ev = match (scf_fermi_ev, nscf_fermi_ev) {
        (Some(scf), Some(nscf)) => Some((nscf - scf).abs()),
        _ => None,
    };
    let scf_vs_nscf_pass = scf_vs_nscf_delta_ev.map(|delta| delta <= tolerance_ev);
    let postprocess_in_gap = match (gap.vbm.as_ref(), gap.cbm.as_ref()) {
        (Some(vbm), Some(cbm)) => Some(gap.fermi_ev >= vbm.energy_ev && gap.fermi_ev <= cbm.energy_ev),
        _ => None,
    };

    FermiConsistencySummary {
        scf_fermi_ev,
        nscf_fermi_ev,
        postprocess_fermi_ev: gap.fermi_ev,
        tolerance_ev,
        scf_vs_nscf_delta_ev,
        scf_vs_nscf_pass,
        postprocess_in_gap,
    }
}

fn write_dos_dat(path: &Path, points: &[(f64, f64)]) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("# energy_eV dos_states_per_eV\n");
    for (energy_ev, dos) in points.iter() {
        content.push_str(&format!("{:.10} {:.10}\n", energy_ev, dos));
    }
    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_dos_csv(path: &Path, points: &[(f64, f64)]) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("energy_ev,dos_states_per_ev\n");
    for (energy_ev, dos) in points.iter() {
        content.push_str(&format!("{:.10},{:.10}\n", energy_ev, dos));
    }
    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_dos_json(path: &Path, points: &[(f64, f64)]) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("{\n");
    content.push_str(&format!("  \"schema_version\": \"{}\",\n", SCHEMA_VERSION));
    content.push_str("  \"dos\": [\n");
    for (idx, (energy_ev, dos)) in points.iter().enumerate() {
        let comma = if idx + 1 == points.len() { "" } else { "," };
        content.push_str(&format!(
            "    {{\"energy_ev\": {:.10}, \"dos_states_per_ev\": {:.10}}}{}\n",
            energy_ev, dos, comma
        ));
    }
    content.push_str("  ]\n");
    content.push_str("}\n");
    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_band_gap_json(path: &Path, eig: &EigDataset, gap: &BandGapSummary) -> Result<(), String> {
    let content = format!(
        "{{\n  \"schema_version\": \"{}\",\n  \"seed\": \"{}\",\n  \"spin_polarized\": {},\n  \"nkpt\": {},\n  \"fermi_ev\": {:.10},\n  \"is_metal\": {},\n  \"indirect_gap_ev\": {},\n  \"direct_gap_ev\": {},\n  \"direct_gap_k_index\": {},\n  \"vbm\": {},\n  \"cbm\": {}\n}}\n",
        SCHEMA_VERSION,
        escape_json_string(&eig.seed),
        if eig.spin_polarized { "true" } else { "false" },
        eig.nkpt,
        gap.fermi_ev,
        if gap.is_metal { "true" } else { "false" },
        json_opt_f64(gap.indirect_gap_ev),
        json_opt_f64(gap.direct_gap_ev),
        json_opt_usize(gap.direct_gap_k_index),
        json_band_edge(gap.vbm.as_ref()),
        json_band_edge(gap.cbm.as_ref()),
    );

    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_fermi_consistency_json(path: &Path, summary: &FermiConsistencySummary) -> Result<(), String> {
    let content = format!(
        "{{\n  \"schema_version\": \"{}\",\n  \"scf_fermi_ev\": {},\n  \"nscf_fermi_ev\": {},\n  \"postprocess_fermi_ev\": {:.10},\n  \"tolerance_ev\": {:.10},\n  \"scf_vs_nscf_delta_ev\": {},\n  \"scf_vs_nscf_pass\": {},\n  \"postprocess_in_gap\": {}\n}}\n",
        SCHEMA_VERSION,
        json_opt_f64(summary.scf_fermi_ev),
        json_opt_f64(summary.nscf_fermi_ev),
        summary.postprocess_fermi_ev,
        summary.tolerance_ev,
        json_opt_f64(summary.scf_vs_nscf_delta_ev),
        json_opt_bool(summary.scf_vs_nscf_pass),
        json_opt_bool(summary.postprocess_in_gap),
    );
    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn json_opt_bool(v: Option<bool>) -> String {
    match v {
        Some(true) => "true".to_string(),
        Some(false) => "false".to_string(),
        None => "null".to_string(),
    }
}

fn json_opt_usize(v: Option<usize>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".to_string(),
    }
}

fn json_band_edge(edge: Option<&BandEdge>) -> String {
    match edge {
        Some(edge) => format!(
            "{{\"energy_ev\": {:.10}, \"k_index\": {}, \"channel\": \"{}\"}}",
            edge.energy_ev,
            edge.k_index,
            escape_json_string(&edge.channel)
        ),
        None => "null".to_string(),
    }
}

fn parse_last_fermi_ev(lines: &[&str]) -> Option<f64> {
    let mut last = None;
    for line in lines.iter() {
        if let Some(v) = parse_fermi_from_line(line) {
            last = Some(v);
        }
    }
    last
}

fn parse_fermi_from_line(line: &str) -> Option<f64> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.contains("Fermi_level") {
        return trimmed
            .split(|c: char| c.is_whitespace() || c == '=')
            .filter(|token| !token.is_empty())
            .rev()
            .find_map(parse_float_token);
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    if tokens.len() >= 3 && tokens[0].ends_with(':') {
        let iter = &tokens[0][..tokens[0].len().saturating_sub(1)];
        if iter.parse::<usize>().is_ok() {
            return parse_float_token(tokens[2]);
        }
    }

    None
}

fn parse_float_token(token: &str) -> Option<f64> {
    let cleaned = token
        .trim()
        .trim_matches(|c: char| c == ',' || c == ';' || c == '(' || c == ')');
    if cleaned.is_empty() {
        return None;
    }

    cleaned
        .replace('D', "E")
        .replace('d', "e")
        .parse::<f64>()
        .ok()
}

fn strip_comments(line: &str) -> &str {
    let mut out = line;
    if let Some((head, _)) = out.split_once('!') {
        out = head;
    }
    if let Some((head, _)) = out.split_once('#') {
        out = head;
    }
    out.trim()
}

fn write_summary_json(
    path: &Path,
    stage: &str,
    log_path: &Path,
    parsed: &ParsedPwLog,
    workflow_wall_seconds: f64,
) -> Result<(), String> {
    let source_log = log_path
        .file_name()
        .and_then(|x| x.to_str())
        .unwrap_or("out.pw.log");

    let mut max_force_component: Option<f64> = None;
    for force in parsed.forces.iter() {
        for v in [force.fx_ev_per_a, force.fy_ev_per_a, force.fz_ev_per_a] {
            let abs_v = v.abs();
            max_force_component = Some(match max_force_component {
                Some(current) => current.max(abs_v),
                None => abs_v,
            });
        }
    }

    let mut max_stress_component: Option<f64> = None;
    if let Some(stress) = parsed.stress_total_kbar.as_ref() {
        for row in stress.iter() {
            for value in row.iter() {
                let abs_v = value.abs();
                max_stress_component = Some(match max_stress_component {
                    Some(current) => current.max(abs_v),
                    None => abs_v,
                });
            }
        }
    }

    let energy_harris_ry = parsed.energy.as_ref().map(|e| e.harris_ry);
    let energy_scf_ry = parsed.energy.as_ref().map(|e| e.scf_ry);
    let energy_scf_ev = parsed.energy.as_ref().map(|e| ry_to_ev(e.scf_ry));
    let energy_delta_ev = parsed.energy.as_ref().map(|e| e.delta_ev);

    let content = format!(
        "{{\n  \"schema_version\": \"{}\",\n  \"stage\": \"{}\",\n  \"source_log\": \"{}\",\n  \"energy\": {{\n    \"harris_ry\": {},\n    \"scf_ry\": {},\n    \"scf_ev\": {},\n    \"delta_ev\": {}\n  }},\n  \"force\": {{\n    \"n_atoms\": {},\n    \"max_abs_component_ev_per_a\": {}\n  }},\n  \"stress\": {{\n    \"max_abs_component_kbar\": {}\n  }},\n  \"timing\": {{\n    \"workflow_wall_seconds\": {:.6},\n    \"pw_total_seconds\": {},\n    \"user_cpu_seconds\": {},\n    \"system_cpu_seconds\": {},\n    \"max_rss_kb\": {}\n  }}\n}}\n",
        SCHEMA_VERSION,
        escape_json_string(stage),
        escape_json_string(source_log),
        json_opt_f64(energy_harris_ry),
        json_opt_f64(energy_scf_ry),
        json_opt_f64(energy_scf_ev),
        json_opt_f64(energy_delta_ev),
        parsed.forces.len(),
        json_opt_f64(max_force_component),
        json_opt_f64(max_stress_component),
        workflow_wall_seconds,
        json_opt_f64(parsed.pw_total_seconds),
        json_opt_f64(parsed.user_cpu_seconds),
        json_opt_f64(parsed.system_cpu_seconds),
        json_opt_u64(parsed.max_rss_kb),
    );

    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_timings_csv(
    path: &Path,
    parsed: &ParsedPwLog,
    workflow_wall_seconds: f64,
) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("metric,value,unit\n");
    content.push_str(&format!(
        "workflow_wall_seconds,{:.6},seconds\n",
        workflow_wall_seconds
    ));
    content.push_str(&format!(
        "pw_total_seconds,{},seconds\n",
        csv_opt_f64(parsed.pw_total_seconds)
    ));
    content.push_str(&format!(
        "user_cpu_seconds,{},seconds\n",
        csv_opt_f64(parsed.user_cpu_seconds)
    ));
    content.push_str(&format!(
        "system_cpu_seconds,{},seconds\n",
        csv_opt_f64(parsed.system_cpu_seconds)
    ));
    content.push_str(&format!(
        "max_rss_kb,{},kB\n",
        csv_opt_u64(parsed.max_rss_kb)
    ));

    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_energy_csv(path: &Path, energy: &EnergySummary) -> Result<(), String> {
    let content = format!(
        "metric,value,unit\nharris_energy,{:.12},Ry\nscf_energy,{:.12},Ry\nscf_energy,{:.12},eV\ndelta_energy,{:.12},eV\n",
        energy.harris_ry,
        energy.scf_ry,
        ry_to_ev(energy.scf_ry),
        energy.delta_ev
    );
    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_force_csv(path: &Path, forces: &[ForceEntry]) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("atom_index,species,fx_ev_per_a,fy_ev_per_a,fz_ev_per_a\n");
    for force in forces.iter() {
        content.push_str(&format!(
            "{},{},{:.12},{:.12},{:.12}\n",
            force.atom_index,
            force.species,
            force.fx_ev_per_a,
            force.fy_ev_per_a,
            force.fz_ev_per_a
        ));
    }

    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn write_stress_csv(path: &Path, stress: &[[f64; 3]; 3]) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("row,col,value_kbar\n");
    for (row_i, row) in stress.iter().enumerate() {
        for (col_i, value) in row.iter().enumerate() {
            content.push_str(&format!("{},{},{:.12}\n", row_i + 1, col_i + 1, value));
        }
    }

    fs::write(path, content).map_err(|err| format!("failed to write '{}': {}", path.display(), err))
}

fn json_opt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.12}", x),
        None => "null".to_string(),
    }
}

fn json_opt_u64(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "null".to_string(),
    }
}

fn csv_opt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) => format!("{:.6}", x),
        None => String::new(),
    }
}

fn csv_opt_u64(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::new(),
    }
}

fn escape_json_string(text: &str) -> String {
    text.replace('\\', "\\\\").replace('\"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn base_log_text() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../test_example/si-oncv/pbe-force-check/base/out.log");
        fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read fixture '{}': {}", path.display(), err))
    }

    #[test]
    fn parses_energy_force_stress_with_tolerance() {
        let parsed = parse_pw_log(&base_log_text());
        let energy = parsed
            .energy
            .as_ref()
            .expect("expected SCF energy row in fixture");

        assert!((energy.harris_ry - (-1.581446438659e1)).abs() < 1.0e-9);
        assert!((energy.scf_ry - (-1.581446431487e1)).abs() < 1.0e-9);
        assert!((energy.delta_ev - 9.758e-7).abs() < 1.0e-12);

        assert_eq!(parsed.forces.len(), 2);
        assert!((parsed.forces[0].fx_ev_per_a - 4.6e-5).abs() < 1.0e-9);
        assert!((parsed.forces[1].fz_ev_per_a - 1.7e-5).abs() < 1.0e-9);

        let stress = parsed
            .stress_total_kbar
            .as_ref()
            .expect("expected total stress matrix in fixture");
        assert!((stress[0][0] - 122.968403).abs() < 1.0e-6);
        assert!((stress[1][1] - 122.967846).abs() < 1.0e-6);
        assert!((stress[2][2] - 122.967963).abs() < 1.0e-6);
    }

    #[test]
    fn exports_standardized_property_files() {
        let base = std::env::temp_dir().join(format!(
            "property-export-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ));
        fs::create_dir_all(&base).expect("create temp run dir");
        let log_path = base.join("out.pw.log");
        fs::write(&log_path, base_log_text()).expect("write fixture log");

        export_stage_properties(&base, "scf", &log_path, 12.34).expect("export should succeed");

        let properties_dir = base.join("properties");
        assert!(properties_dir.join("summary.json").is_file());
        assert!(properties_dir.join("timings.csv").is_file());
        assert!(properties_dir.join("energy.csv").is_file());
        assert!(properties_dir.join("force.csv").is_file());
        assert!(properties_dir.join("stress.csv").is_file());

        let summary = fs::read_to_string(properties_dir.join("summary.json"))
            .expect("read generated summary");
        assert!(summary.contains("\"schema_version\": \"property-v1\""));
        assert!(summary.contains("\"stage\": \"scf\""));

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn exports_level2_nscf_dos_gap_and_fermi_checks() {
        let base = std::env::temp_dir().join(format!(
            "property-level2-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ));
        fs::create_dir_all(&base).expect("create temp nscf run dir");

        let scf_source = base.join("scf-source");
        fs::create_dir_all(&scf_source).expect("create source scf dir");
        let scf_log = "\
   eps(eV)  Fermi(eV)           charge               Eharris(Ry)                  Escf(Ry)       dE(eV)\n\
  1:   1.000E-6    1.900E-1       8.000000E0         -1.000000000000E1         -1.000000000001E1      1.000E-6\n";
        fs::write(scf_source.join("out.pw.log"), scf_log).expect("write source scf log");

        let nscf_log = "\
   eps(eV)  Fermi(eV)           charge               Eharris(Ry)                  Escf(Ry)       dE(eV)\n\
  1:   1.000E-6    2.000E-1       8.000000E0         -1.000000000000E1         -1.000000000001E1      1.000E-6\n\
   Total           :               1.00 seconds             0.00 hours\n";
        let log_path = base.join("out.pw.log");
        fs::write(&log_path, nscf_log).expect("write nscf log");

        fs::write(
            base.join("in.ctrl"),
            "dos_sigma = 0.20\ndos_ne = 11\nwannier90_seedname = si\n",
        )
        .expect("write in.ctrl");
        fs::write(
            base.join("si.eig"),
            "\
1 1 -1.200000\n\
2 1 -0.400000\n\
3 1 1.200000\n\
1 2 -1.100000\n\
2 2 -0.100000\n\
3 2 0.800000\n",
        )
        .expect("write si.eig");
        fs::write(base.join("from_scf_run.txt"), format!("{}\n", scf_source.display()))
            .expect("write source metadata");

        let mut options = PropertyExportOptions::default();
        options.dos_format = DosOutputFormat::Csv;
        options.fermi_tol_ev = 0.05;

        export_stage_properties_with_options(&base, "nscf", &log_path, 5.0, &options)
            .expect("nscf level2 export should succeed");

        let properties_dir = base.join("properties");
        assert!(properties_dir.join("dos.dat").is_file());
        assert!(properties_dir.join("dos.csv").is_file());
        assert!(properties_dir.join("band_gap.json").is_file());
        assert!(properties_dir.join("fermi_consistency.json").is_file());

        let dos_dat = fs::read_to_string(properties_dir.join("dos.dat")).expect("read dos.dat");
        let dos_lines: Vec<&str> = dos_dat.lines().collect();
        assert_eq!(dos_lines.len(), 12);
        for line in dos_lines.iter().skip(1) {
            let cols: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(cols.len(), 2);
            let dos = cols[1]
                .parse::<f64>()
                .expect("dos value should be parseable");
            assert!(dos.is_finite());
            assert!(dos >= 0.0);
        }

        let band_gap = fs::read_to_string(properties_dir.join("band_gap.json"))
            .expect("read band_gap.json");
        assert!(band_gap.contains("\"indirect_gap_ev\": 0.900000000000"));
        assert!(band_gap.contains("\"direct_gap_ev\": 0.900000000000"));
        assert!(band_gap.contains("\"vbm\": {\"energy_ev\": -0.1000000000"));
        assert!(band_gap.contains("\"cbm\": {\"energy_ev\": 0.8000000000"));

        let fermi = fs::read_to_string(properties_dir.join("fermi_consistency.json"))
            .expect("read fermi_consistency.json");
        assert!(fermi.contains("\"scf_vs_nscf_pass\": true"));

        let _ = fs::remove_dir_all(&base);
    }
}
