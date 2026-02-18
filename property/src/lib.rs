use std::fs;
use std::path::Path;

pub const SCHEMA_VERSION: &str = "property-v1";
pub const RY_TO_EV: f64 = 13.605_693_009;

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
}
