// Integration tests for DFTWorks
// These tests verify end-to-end behavior and physics correctness

use std::path::{Path, PathBuf};
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
    // With a fixed random seed, two runs of the same input must take the same
    // number of SCF iterations and reach the same energy.
    let template = project_root().join("test_example/si-oncv/scf");

    // One directory per test process, so concurrent runs cannot touch each other.
    let base = std::env::temp_dir().join(format!("dftworks-determinism-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();

    let scf1 = run_scf_with_seed(&template, &base.join("run1"));
    let scf2 = run_scf_with_seed(&template, &base.join("run2"));

    std::fs::remove_dir_all(&base).ok();

    let (niter1, energy1) = scf1;
    let (niter2, energy2) = scf2;
    assert_eq!(
        niter1, niter2,
        "SCF iteration counts differ between the two runs"
    );
    assert!(
        (energy1 - energy2).abs() < 1e-10,
        "final SCF energies differ: {} Ry vs {} Ry",
        energy1,
        energy2
    );
}

/// Copy the template into a fresh `dir`, set random_seed = 42 and spin_scheme = nonspin,
/// run pw there, and return (number of SCF iterations, final Escf in Ry).
fn run_scf_with_seed(template: &Path, dir: &Path) -> (usize, f64) {
    copy_dir(template, dir);

    let ctrl_path = dir.join("in.ctrl");
    let ctrl = std::fs::read_to_string(&ctrl_path).expect("Failed to read in.ctrl");
    let mut new_ctrl: String = ctrl
        .lines()
        .filter(|line| {
            let key = line.trim_start();
            !key.starts_with("random_seed") && !key.starts_with("spin_scheme")
        })
        .map(|line| format!("{}\n", line))
        .collect();
    new_ctrl.push_str("random_seed = 42\nspin_scheme = nonspin\n");
    std::fs::write(&ctrl_path, new_ctrl).expect("Failed to write in.ctrl");

    let output = Command::new(pw_binary())
        .current_dir(dir)
        .output()
        .expect("Failed to run pw");
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && log.contains("scf_convergence_success"),
        "pw run in {:?} failed or did not converge: {}",
        dir,
        String::from_utf8_lossy(&output.stderr)
    );

    final_scf_energy(&log).expect("no SCF iteration lines in pw output")
}

/// Number of SCF iterations and the last Escf (Ry) from pw's iteration table, whose
/// rows read "<iter>: eps Fermi charge Eharris Escf dE".
fn final_scf_energy(log: &str) -> Option<(usize, f64)> {
    let mut last = None;
    for line in log.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() != 7 {
            continue;
        }
        let iter = cols[0]
            .strip_suffix(':')
            .and_then(|n| n.parse::<usize>().ok());
        if let (Some(iter), Ok(escf)) = (iter, cols[5].parse::<f64>()) {
            last = Some((iter, escf));
        }
    }
    last
}

/// Recursive copy of a directory tree.
fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("Failed to create directory");
    for entry in std::fs::read_dir(src).expect("Failed to read directory") {
        let entry = entry.expect("Failed to read directory entry");
        let target = dst.join(entry.file_name());
        // follow symlinks, as cp -L does
        if std::fs::metadata(entry.path())
            .expect("Failed to stat")
            .is_dir()
        {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("Failed to copy file");
        }
    }
}
