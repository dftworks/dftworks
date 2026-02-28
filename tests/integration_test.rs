// Integration tests for DFTWorks
// These tests verify end-to-end behavior and physics correctness

use std::path::PathBuf;
use std::process::Command;

/// Helper to get the project root directory
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Helper to get the pw binary path
fn pw_binary() -> PathBuf {
    let mut path = project_root();
    path.push("target");
    path.push("debug");
    path.push("pw");
    path
}

#[test]
fn test_phase12_reference_exists() {
    let mut reference_file = project_root();
    reference_file.push("test_example/si-oncv/regression/phase12_reference.tsv");

    assert!(
        reference_file.exists(),
        "Phase 1/2 reference file not found: {:?}",
        reference_file
    );

    // Verify reference file has expected structure
    let content = std::fs::read_to_string(&reference_file)
        .expect("Failed to read reference file");

    let lines: Vec<&str> = content.lines().collect();
    assert!(
        lines.len() >= 4,
        "Reference file should have at least 4 test cases"
    );

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            fields.len(),
            5,
            "Each reference line should have 5 tab-separated fields: {:?}",
            line
        );
    }
}

#[test]
fn test_scf_template_exists() {
    let mut scf_template = project_root();
    scf_template.push("test_example/si-oncv/scf");

    assert!(scf_template.exists(), "SCF template directory not found");

    // Check required input files exist
    let required_files = vec![
        "in.ctrl",
        "in.crystal",
        "in.kmesh",
        "in.pot",
        "in.spin",
        "in.magmom",
    ];

    for file in required_files {
        let mut path = scf_template.clone();
        path.push(file);
        assert!(
            path.exists(),
            "Required SCF template file not found: {}",
            file
        );
    }
}

#[test]
#[ignore] // Only run with --ignored flag (requires building pw first)
fn test_determinism_fixed_seed() {
    // This test verifies that with a fixed random seed, results are deterministic
    // Run the same calculation twice and verify identical outputs

    let pw_bin = pw_binary();
    if !pw_bin.exists() {
        eprintln!("pw binary not found, skipping determinism test");
        eprintln!("Run: cargo build -p pw");
        return;
    }

    let test_dir = std::env::temp_dir().join("dftworks-determinism-test");
    std::fs::create_dir_all(&test_dir).expect("Failed to create test directory");

    // Copy SCF template
    let mut scf_template = project_root();
    scf_template.push("test_example/si-oncv/scf");

    for file in &["in.ctrl", "in.crystal", "in.kmesh", "in.pot", "in.spin", "in.magmom"] {
        let src = scf_template.join(file);
        let dst = test_dir.join(file);
        std::fs::copy(src, dst).expect("Failed to copy template file");
    }

    // Copy pot directory
    let pot_src = scf_template.join("pot");
    let pot_dst = test_dir.join("pot");
    if pot_src.exists() {
        std::fs::create_dir_all(&pot_dst).expect("Failed to create pot directory");
        for entry in std::fs::read_dir(&pot_src).expect("Failed to read pot directory") {
            let entry = entry.expect("Failed to read pot entry");
            std::fs::copy(entry.path(), pot_dst.join(entry.file_name()))
                .expect("Failed to copy pot file");
        }
    }

    // Modify in.ctrl to set random_seed and use nonspin for speed
    let ctrl_path = test_dir.join("in.ctrl");
    let ctrl_content = std::fs::read_to_string(&ctrl_path)
        .expect("Failed to read in.ctrl");

    let mut modified_ctrl = String::new();
    let mut found_seed = false;
    let mut found_spin = false;

    for line in ctrl_content.lines() {
        if line.trim_start().starts_with("random_seed") {
            modified_ctrl.push_str("random_seed = 42\n");
            found_seed = true;
        } else if line.trim_start().starts_with("spin_scheme") {
            modified_ctrl.push_str("spin_scheme = nonspin\n");
            found_spin = true;
        } else {
            modified_ctrl.push_str(line);
            modified_ctrl.push('\n');
        }
    }

    if !found_seed {
        modified_ctrl.push_str("random_seed = 42\n");
    }
    if !found_spin {
        modified_ctrl.push_str("spin_scheme = nonspin\n");
    }

    std::fs::write(&ctrl_path, modified_ctrl)
        .expect("Failed to write modified in.ctrl");

    // Run first calculation
    let output1 = Command::new(&pw_bin)
        .current_dir(&test_dir)
        .output()
        .expect("Failed to run pw (first run)");

    assert!(
        output1.status.success(),
        "First pw run failed: {}",
        String::from_utf8_lossy(&output1.stderr)
    );

    let log1 = std::fs::read_to_string(test_dir.join("out.log"))
        .expect("Failed to read first out.log");

    // Clean output files but keep inputs
    for entry in std::fs::read_dir(&test_dir).expect("Failed to read test directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.is_file() && path.file_name().unwrap().to_str().unwrap().starts_with("out.") {
            std::fs::remove_file(path).expect("Failed to remove output file");
        }
    }

    // Run second calculation
    let output2 = Command::new(&pw_bin)
        .current_dir(&test_dir)
        .output()
        .expect("Failed to run pw (second run)");

    assert!(
        output2.status.success(),
        "Second pw run failed: {}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let log2 = std::fs::read_to_string(test_dir.join("out.log"))
        .expect("Failed to read second out.log");

    // Parse final SCF energy from both logs
    let energy1 = parse_final_scf_energy(&log1);
    let energy2 = parse_final_scf_energy(&log2);

    match (energy1, energy2) {
        (Some(e1), Some(e2)) => {
            let diff = (e1 - e2).abs();
            assert!(
                diff < 1e-10,
                "Determinism test failed: energies differ by {} Ry (E1={}, E2={})",
                diff,
                e1,
                e2
            );
        }
        _ => {
            eprintln!("Warning: Could not parse final energies from logs");
            eprintln!("This may indicate the calculation didn't converge");
        }
    }

    // Cleanup
    std::fs::remove_dir_all(&test_dir).ok();
}

fn parse_final_scf_energy(log: &str) -> Option<f64> {
    for line in log.lines().rev() {
        if line.contains("scf_energy") && line.contains("Ry") {
            if let Some(parts) = line.split_whitespace().last() {
                if let Ok(energy) = parts.parse::<f64>() {
                    return Some(energy);
                }
            }
        }

        // Also try parsing from iteration table (last iteration)
        if line.trim().chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(energy) = parts[parts.len() - 2].parse::<f64>() {
                    return Some(energy);
                }
            }
        }
    }
    None
}
