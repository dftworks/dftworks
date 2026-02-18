use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use wannier90::{run_projected_analysis, ProjectedConfig};

fn main() {
    if let Err(err) = run() {
        eprintln!("w90-proj: {}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
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

    let mut strict_validation = false;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--seed" => {
                i += 1;
                config.seedname = get_arg_value(&args, i, "--seed")?;
            }
            "--sigma" => {
                i += 1;
                let value = get_arg_value(&args, i, "--sigma")?;
                config.sigma_ev = value
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --sigma '{}'", value))?;
            }
            "--ne" => {
                i += 1;
                let value = get_arg_value(&args, i, "--ne")?;
                config.ne = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --ne '{}'", value))?;
            }
            "--emin" => {
                i += 1;
                let value = get_arg_value(&args, i, "--emin")?;
                config.emin_ev = Some(
                    value
                        .parse::<f64>()
                        .map_err(|_| format!("invalid --emin '{}'", value))?,
                );
            }
            "--emax" => {
                i += 1;
                let value = get_arg_value(&args, i, "--emax")?;
                config.emax_ev = Some(
                    value
                        .parse::<f64>()
                        .map_err(|_| format!("invalid --emax '{}'", value))?,
                );
            }
            "--input-dir" => {
                i += 1;
                let value = get_arg_value(&args, i, "--input-dir")?;
                config.input_dir = PathBuf::from(value);
            }
            "--out-dir" => {
                i += 1;
                let value = get_arg_value(&args, i, "--out-dir")?;
                config.out_dir = PathBuf::from(value);
            }
            "--cache" => {
                i += 1;
                let value = get_arg_value(&args, i, "--cache")?;
                config.cache_path = Some(PathBuf::from(value));
            }
            "--validation-tol" => {
                i += 1;
                let value = get_arg_value(&args, i, "--validation-tol")?;
                config.validation_tol = value
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --validation-tol '{}'", value))?;
            }
            "--strict-validation" => {
                strict_validation = true;
            }
            unknown => {
                return Err(format!("unknown argument '{}'", unknown));
            }
        }
        i += 1;
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

    if strict_validation && !summary.validation_passed {
        return Err("validation failed (strict mode enabled)".to_string());
    }

    Ok(())
}

fn get_arg_value(args: &[String], idx: usize, flag: &str) -> Result<String, String> {
    args.get(idx)
        .cloned()
        .ok_or_else(|| format!("missing value for {}", flag))
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

fn print_help() {
    println!("Usage: w90-proj [options]");
    println!();
    println!("Projected property analysis from Wannier90 files (<seed>.amn/.eig/.nnkp)");
    println!();
    println!("Options:");
    println!("  --seed <name>            Input seed (default: in.ctrl, then auto-detect)");
    println!("  --sigma <eV>             Gaussian broadening sigma in eV");
    println!("  --ne <N>                 Number of DOS energy points");
    println!("  --emin <eV>              Lower DOS energy bound");
    println!("  --emax <eV>              Upper DOS energy bound");
    println!("  --input-dir <dir>        Directory containing .amn/.eig/.nnkp inputs");
    println!("  --out-dir <dir>          Directory for pdos/fatband outputs");
    println!("  --cache <file>           Projection-weight cache file path");
    println!("  --validation-tol <x>     Max relative tolerance for sum(PDOS) validation");
    println!("  --strict-validation      Exit with failure when validation fails");
    println!("  -h, --help               Show this message");
}
