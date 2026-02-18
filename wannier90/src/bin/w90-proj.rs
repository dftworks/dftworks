use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use wannier90::{run_projected_analysis, ProjectedConfig};

#[derive(Debug, Parser)]
#[command(
    name = "w90-proj",
    about = "Projected property analysis from Wannier90 files (<seed>.amn/.eig/.nnkp)"
)]
struct Cli {
    #[arg(long, value_name = "name")]
    seed: Option<String>,
    #[arg(long, value_name = "eV")]
    sigma: Option<f64>,
    #[arg(long, value_name = "N")]
    ne: Option<usize>,
    #[arg(long, value_name = "eV")]
    emin: Option<f64>,
    #[arg(long, value_name = "eV")]
    emax: Option<f64>,
    #[arg(long = "input-dir", value_name = "dir")]
    input_dir: Option<PathBuf>,
    #[arg(long = "out-dir", value_name = "dir")]
    out_dir: Option<PathBuf>,
    #[arg(long, value_name = "file")]
    cache: Option<PathBuf>,
    #[arg(long = "validation-tol", value_name = "x")]
    validation_tol: Option<f64>,
    #[arg(long = "strict-validation")]
    strict_validation: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("w90-proj: {}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    let mut config = ProjectedConfig::default();
    if let Some(defaults) = read_in_ctrl_defaults(Path::new("in.ctrl")) {
        if let Some(seed) = defaults.seedname {
            config.seedname = seed;
        }
        if let Some(sigma) = defaults.dos_sigma {
            config.sigma_ev = sigma;
        }
        if let Some(ne) = defaults.dos_ne {
            config.ne = ne;
        }
    }

    if let Some(seed) = cli.seed {
        config.seedname = seed;
    }
    if let Some(sigma) = cli.sigma {
        config.sigma_ev = sigma;
    }
    if let Some(ne) = cli.ne {
        config.ne = ne;
    }
    if let Some(emin) = cli.emin {
        config.emin_ev = Some(emin);
    }
    if let Some(emax) = cli.emax {
        config.emax_ev = Some(emax);
    }
    if let Some(input_dir) = cli.input_dir {
        config.input_dir = input_dir;
    }
    if let Some(out_dir) = cli.out_dir {
        config.out_dir = out_dir;
    }
    if let Some(cache_path) = cli.cache {
        config.cache_path = Some(cache_path);
    }
    if let Some(validation_tol) = cli.validation_tol {
        config.validation_tol = validation_tol;
    }

    if config.seedname.trim().is_empty() {
        if let Some(seed) = detect_seed_from_amn(&config.input_dir)? {
            config.seedname = seed;
        } else {
            return Err(
                "seedname is not set. Use --seed <name> or place exactly one *.amn in input-dir"
                    .to_string(),
            );
        }
    }

    let summary = run_projected_analysis(&config).map_err(|e| e.to_string())?;

    println!("seed: {}", config.seedname);
    println!(
        "projection cache: {}",
        if summary.used_cache { "hit" } else { "miss" }
    );
    println!(
        "validation: {} (max_rel_diff={:.6E}, tol={:.6E})",
        if summary.validation_passed {
            "PASS"
        } else {
            "FAIL"
        },
        summary.max_rel_diff,
        config.validation_tol
    );
    for filename in summary.generated_files.iter() {
        println!("wrote {}", filename);
    }

    if cli.strict_validation && !summary.validation_passed {
        return Err("validation failed (strict mode enabled)".to_string());
    }

    Ok(())
}

fn detect_seed_from_amn(input_dir: &Path) -> Result<Option<String>, String> {
    let mut candidates = Vec::new();
    let entries = std::fs::read_dir(input_dir)
        .map_err(|e| format!("failed to read '{}': {}", input_dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("amn") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|x| x.to_str())
            .ok_or_else(|| format!("invalid AMN filename '{}'", path.display()))?
            .to_string();
        candidates.push(stem);
    }

    if candidates.len() == 1 {
        Ok(Some(candidates.remove(0)))
    } else if candidates.is_empty() {
        Ok(None)
    } else {
        Err(format!(
            "multiple AMN files found in '{}'; use --seed explicitly",
            input_dir.display()
        ))
    }
}

struct InCtrlDefaults {
    seedname: Option<String>,
    dos_sigma: Option<f64>,
    dos_ne: Option<usize>,
}

fn read_in_ctrl_defaults(path: &Path) -> Option<InCtrlDefaults> {
    if !path.exists() {
        return None;
    }
    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut defaults = InCtrlDefaults {
        seedname: None,
        dos_sigma: None,
        dos_ne: None,
    };

    for line in reader.lines().map_while(Result::ok) {
        let cleaned = line
            .split('#')
            .next()
            .unwrap_or("")
            .split('!')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if cleaned.is_empty() {
            continue;
        }
        let (key, value) = match cleaned.split_once('=') {
            Some((k, v)) => (k.trim().to_ascii_lowercase(), v.trim().to_string()),
            None => continue,
        };
        match key.as_str() {
            "wannier90_seedname" => defaults.seedname = Some(value),
            "dos_sigma" => {
                if let Ok(v) = value.parse::<f64>() {
                    defaults.dos_sigma = Some(v);
                }
            }
            "dos_ne" => {
                if let Ok(v) = value.parse::<usize>() {
                    defaults.dos_ne = Some(v);
                }
            }
            _ => {}
        }
    }

    Some(defaults)
}
