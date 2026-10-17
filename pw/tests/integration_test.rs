// Integration tests for DFTWorks
// These tests verify end-to-end behavior and physics correctness

use std::path::PathBuf;
use std::process::Command;

/// Repository root (this file lives in pw/tests/)
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("pw crate is inside the repository")
        .to_path_buf()
}

/// The pw binary that Cargo builds for this test run
fn pw_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pw"))
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
    let content = std::fs::read_to_string(&reference_file).expect("Failed to read reference file");

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
#[ignore] // Runs two SCF calculations; use `cargo test -p pw --test integration_test -- --ignored`
fn test_determinism_fixed_seed() {
    // This test verifies that with a fixed random seed, results are deterministic
    // Run the same calculation twice and verify identical outputs

    let pw_bin = pw_binary();

    // Start from an empty directory: leftovers from an earlier, failed run would
    // change what the first run prints.
    let test_dir = std::env::temp_dir().join("dftworks-determinism-test");
    std::fs::remove_dir_all(&test_dir).ok();
    std::fs::create_dir_all(&test_dir).expect("Failed to create test directory");

    // Copy SCF template
    let mut scf_template = project_root();
    scf_template.push("test_example/si-oncv/scf");

    for file in &[
        "in.ctrl",
        "in.crystal",
        "in.kmesh",
        "in.pot",
        "in.spin",
        "in.magmom",
    ] {
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
    let ctrl_content = std::fs::read_to_string(&ctrl_path).expect("Failed to read in.ctrl");

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

    std::fs::write(&ctrl_path, modified_ctrl).expect("Failed to write modified in.ctrl");

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

    let log1 = String::from_utf8_lossy(&output1.stdout).to_string();

    // Clean output files but keep inputs
    for entry in std::fs::read_dir(&test_dir).expect("Failed to read test directory") {
        let entry = entry.expect("Failed to read entry");
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("out.")
        {
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

    let log2 = String::from_utf8_lossy(&output2.stdout).to_string();

    // Every line except timings and the timestamp must be identical.
    let results1 = result_lines(&log1);
    let results2 = result_lines(&log2);
    assert!(
        results1
            .iter()
            .any(|line| line.contains("scf_convergence_success")),
        "first pw run did not converge"
    );
    assert_eq!(
        results1, results2,
        "Determinism test failed: pw output differs between two runs with random_seed = 42"
    );

    // Cleanup
    std::fs::remove_dir_all(&test_dir).ok();
}

/// pw output without the lines that change from run to run (timings, timestamp).
fn result_lines(log: &str) -> Vec<&str> {
    log.lines()
        .filter(|line| !line.contains("time") && !line.contains("seconds"))
        .collect()
}
