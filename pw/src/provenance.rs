#![allow(warnings)]

use control::Control;
use kpts::KPTS;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const PROVENANCE_SCHEMA_VERSION: &str = "provenance-v1";

#[derive(Clone, Debug)]
struct InputHashEntry {
    path: String,
    fnv1a64: String,
}

#[derive(Clone, Debug)]
struct RunProvenanceManifest {
    generated_utc: String,
    git_commit: String,
    build_features: Vec<String>,
    fft_backend: String,
    mpi_ranks: i32,
    rayon_threads: usize,
    rayon_env: String,
    spin_scheme: String,
    kpts_scheme: String,
    k_mesh: [i32; 3],
    random_seed: Option<u64>,
    input_hashes: Vec<InputHashEntry>,
    replay_fingerprint: String,
}

fn escape_json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes.iter() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

fn hash_file_fnv1a64(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read '{}': {}", path.display(), err))?;
    Ok(fnv1a64_hex(&bytes))
}

fn parse_in_pot_paths(path: &Path) -> Result<Vec<PathBuf>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }

    let file =
        File::open(path).map_err(|err| format!("failed to open '{}': {}", path.display(), err))?;
    let mut out = Vec::new();

    for (line_idx, line_res) in BufReader::new(file).lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line_res.map_err(|err| {
            format!(
                "failed to read '{}', line {}: {}",
                path.display(),
                line_no,
                err
            )
        })?;

        let line = line.split('#').next().unwrap_or(line.as_str());
        let line = line.split('!').next().unwrap_or(line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 2 {
            return Err(format!(
                "'{}' line {}: expected '<species> <filename>'",
                path.display(),
                line_no
            ));
        }

        out.push(PathBuf::from("pot").join(tokens[1]));
    }

    out.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    out.dedup();

    Ok(out)
}

fn collect_input_hashes(control: &Control) -> Result<Vec<InputHashEntry>, String> {
    let mut files = vec![
        PathBuf::from("in.ctrl"),
        PathBuf::from("in.crystal"),
        PathBuf::from("in.pot"),
    ];

    match control.get_kpts_scheme_enum() {
        control::KptsScheme::Kmesh => files.push(PathBuf::from("in.kmesh")),
        control::KptsScheme::Kline => files.push(PathBuf::from("in.kline")),
    }

    files.extend(parse_in_pot_paths(Path::new("in.pot"))?);

    files.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    files.dedup();

    let mut out = Vec::with_capacity(files.len());
    for path in files.iter() {
        let path_text = path.to_string_lossy().to_string();
        let hash = if path.is_file() {
            hash_file_fnv1a64(path)?
        } else {
            "missing".to_string()
        };
        out.push(InputHashEntry {
            path: path_text,
            fnv1a64: hash,
        });
    }

    Ok(out)
}

fn split_csv_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}

fn compute_replay_fingerprint(manifest: &RunProvenanceManifest) -> String {
    let mut canonical = String::new();
    canonical.push_str(PROVENANCE_SCHEMA_VERSION);
    canonical.push('\n');
    canonical.push_str(manifest.spin_scheme.as_str());
    canonical.push('\n');
    canonical.push_str(manifest.kpts_scheme.as_str());
    canonical.push('\n');
    canonical.push_str(&format!(
        "{},{},{}\n",
        manifest.k_mesh[0], manifest.k_mesh[1], manifest.k_mesh[2]
    ));
    canonical.push_str(
        manifest
            .random_seed
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string())
            .as_str(),
    );
    canonical.push('\n');
    canonical.push_str(&format!("{}\n", manifest.mpi_ranks));
    canonical.push_str(&format!("{}\n", manifest.rayon_threads));

    for entry in manifest.input_hashes.iter() {
        canonical.push_str(entry.path.as_str());
        canonical.push('=');
        canonical.push_str(entry.fnv1a64.as_str());
        canonical.push('\n');
    }

    fnv1a64_hex(canonical.as_bytes())
}

fn extract_json_string_field(doc: &str, field: &str) -> Option<String> {
    let needle = format!("\"{}\": \"", field);
    let start = doc.find(needle.as_str())?;
    let value_start = start + needle.len();
    let tail = &doc[value_start..];
    let value_end = tail.find('"')?;
    Some(tail[..value_end].to_string())
}

fn verify_replay_manifest(path: &str, expected_fingerprint: &str) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read replay manifest '{}': {}", path, err))?;

    let schema = extract_json_string_field(text.as_str(), "schema_version")
        .ok_or_else(|| format!("replay manifest '{}' is missing schema_version", path))?;
    if schema != PROVENANCE_SCHEMA_VERSION {
        return Err(format!(
            "replay manifest schema mismatch: expected '{}', got '{}'",
            PROVENANCE_SCHEMA_VERSION, schema
        ));
    }

    let fingerprint = extract_json_string_field(text.as_str(), "replay_fingerprint")
        .ok_or_else(|| format!("replay manifest '{}' is missing replay_fingerprint", path))?;

    if fingerprint != expected_fingerprint {
        return Err(format!(
            "replay manifest mismatch for '{}': expected {}, got {}",
            path, expected_fingerprint, fingerprint
        ));
    }

    Ok(())
}

fn render_provenance_json(manifest: &RunProvenanceManifest) -> String {
    let mut features_json = String::from("[");
    for (idx, feature) in manifest.build_features.iter().enumerate() {
        if idx > 0 {
            features_json.push_str(", ");
        }
        features_json.push('"');
        features_json.push_str(escape_json_string(feature).as_str());
        features_json.push('"');
    }
    features_json.push(']');

    let mut input_hashes_json = String::new();
    for (idx, entry) in manifest.input_hashes.iter().enumerate() {
        if idx > 0 {
            input_hashes_json.push_str(",\n");
        }
        input_hashes_json.push_str(&format!(
            "    {{ \"path\": \"{}\", \"fnv1a64\": \"{}\" }}",
            escape_json_string(entry.path.as_str()),
            entry.fnv1a64
        ));
    }

    format!(
        "{{\n  \"schema_version\": \"{}\",\n  \"generated_utc\": \"{}\",\n  \"binary\": {{\n    \"name\": \"pw\",\n    \"version\": \"{}\",\n    \"git_commit\": \"{}\",\n    \"build_features\": {},\n    \"fft_backend\": \"{}\"\n  }},\n  \"runtime\": {{\n    \"mpi_ranks\": {},\n    \"rayon_threads\": {},\n    \"rayon_env\": \"{}\"\n  }},\n  \"initialization\": {{\n    \"spin_scheme\": \"{}\",\n    \"kpts_scheme\": \"{}\",\n    \"k_mesh\": [{}, {}, {}],\n    \"random_seed\": {}\n  }},\n  \"input_hashes\": [\n{}\n  ],\n  \"replay_fingerprint\": \"{}\"\n}}\n",
        PROVENANCE_SCHEMA_VERSION,
        escape_json_string(manifest.generated_utc.as_str()),
        env!("CARGO_PKG_VERSION"),
        escape_json_string(manifest.git_commit.as_str()),
        features_json,
        escape_json_string(manifest.fft_backend.as_str()),
        manifest.mpi_ranks,
        manifest.rayon_threads,
        escape_json_string(manifest.rayon_env.as_str()),
        escape_json_string(manifest.spin_scheme.as_str()),
        escape_json_string(manifest.kpts_scheme.as_str()),
        manifest.k_mesh[0],
        manifest.k_mesh[1],
        manifest.k_mesh[2],
        manifest
            .random_seed
            .map(|seed| seed.to_string())
            .unwrap_or_else(|| "null".to_string()),
        input_hashes_json,
        manifest.replay_fingerprint
    )
}

pub(crate) fn emit_run_provenance_manifest(control: &Control, kpts: &dyn KPTS) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let generated_utc = crate::runtime_display::format_unix_seconds_as_utc_iso(now);

    let git_commit = option_env!("DWWORKS_GIT_COMMIT")
        .unwrap_or("unknown")
        .trim()
        .to_string();
    let build_features = split_csv_list(option_env!("DWWORKS_BUILD_FEATURES").unwrap_or(""));
    let fft_backend = dwfft3d::backend_name().to_string();
    let mpi_ranks = dwmpi::get_comm_world_size();
    let rayon_threads = rayon::current_num_threads();
    let rayon_env = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let spin_scheme = control.get_spin_scheme().to_string();
    let kpts_scheme = control.get_kpts_scheme().to_string();
    let k_mesh = kpts.get_k_mesh();
    let random_seed = control.get_random_seed();
    let input_hashes = collect_input_hashes(control)?;

    let mut manifest = RunProvenanceManifest {
        generated_utc,
        git_commit,
        build_features,
        fft_backend,
        mpi_ranks,
        rayon_threads,
        rayon_env,
        spin_scheme,
        kpts_scheme,
        k_mesh,
        random_seed,
        input_hashes,
        replay_fingerprint: String::new(),
    };
    manifest.replay_fingerprint = compute_replay_fingerprint(&manifest);

    if control.get_provenance_check() {
        verify_replay_manifest(
            control.get_provenance_manifest(),
            manifest.replay_fingerprint.as_str(),
        )?;
    }

    let content = render_provenance_json(&manifest);
    fs::write(control.get_provenance_manifest(), content).map_err(|err| {
        format!(
            "failed to write provenance manifest '{}': {}",
            control.get_provenance_manifest(),
            err
        )
    })?;

    Ok(())
}
