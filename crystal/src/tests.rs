//! Test module for crystal structure functionality.

use super::*;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

// Shared mutex to prevent race conditions when multiple tests create out.crystal
static OUTPUT_LOCK: Mutex<()> = Mutex::new(());

/// Helper function to create a simple cubic crystal for testing
fn create_test_crystal() -> Crystal {
    let mut crystal = Crystal::new();
    
    // Create a simple cubic lattice (10 Angstroms = ~18.9 Bohr)
    let scale = 10.0;
    let vec_a = [scale * ANG_TO_BOHR, 0.0, 0.0];
    let vec_b = [0.0, scale * ANG_TO_BOHR, 0.0];
    let vec_c = [0.0, 0.0, scale * ANG_TO_BOHR];
    
    let latt = Lattice::new(&vec_a, &vec_b, &vec_c);
    crystal.set_lattice_vectors(&latt);
    
    // Set scales
    // Note: We can't directly set scales, but we can test other functionality
    
    crystal
}

#[test]
fn test_crystal_new() {
    let crystal = Crystal::new();
    assert_eq!(crystal.get_n_atoms(), 0);
    assert_eq!(crystal.get_n_unique_species(), 0);
}

#[test]
fn test_crystal_read_file() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    // Verify basic structure
    assert!(crystal.get_n_atoms() > 0);
    assert!(crystal.get_n_unique_species() > 0);
    
    // Verify lattice is set
    let latt = crystal.get_latt();
    let a = latt.get_vector_a();
    assert!(a.x.abs() > 0.0 || a.y.abs() > 0.0 || a.z.abs() > 0.0);
}

#[test]
fn test_get_n_atoms() {
    let mut crystal = create_test_crystal();
    
    // Initially empty
    assert_eq!(crystal.get_n_atoms(), 0);
    
    // Add some atoms manually (we'll test this via file reading)
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    crystal.read_file(d.to_str().unwrap());
    
    assert!(crystal.get_n_atoms() > 0);
}

#[test]
fn test_get_unique_species() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let unique = crystal.get_unique_species();
    assert!(!unique.is_empty());
    
    // Verify all species in unique list are actually unique
    for i in 0..unique.len() {
        for j in (i + 1)..unique.len() {
            assert_ne!(unique[i], unique[j], "Species list should contain unique values");
        }
    }
}

#[test]
fn test_get_n_unique_species() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let n_unique = crystal.get_n_unique_species();
    let unique_list = crystal.get_unique_species();
    
    assert_eq!(n_unique, unique_list.len());
    assert!(n_unique > 0);
}

#[test]
fn test_get_atom_indices_of_specie() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let n_species = crystal.get_n_unique_species();
    
    if n_species > 0 {
        let indices = crystal.get_atom_indices_of_specie(0);
        assert!(!indices.is_empty());
        
        // Verify indices are valid
        let n_atoms = crystal.get_n_atoms();
        for &idx in indices {
            assert!(idx < n_atoms, "Atom index out of bounds");
        }
    }
}

#[test]
fn test_get_atom_positions() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let positions = crystal.get_atom_positions();
    assert_eq!(positions.len(), crystal.get_n_atoms());
    
    // Verify positions are in fractional coordinates (typically 0-1 range)
    for pos in positions {
        // Fractional coordinates can be outside 0-1 for periodic systems,
        // but should be reasonable
        assert!(pos.x.is_finite());
        assert!(pos.y.is_finite());
        assert!(pos.z.is_finite());
    }
}

#[test]
fn test_get_atom_positions_cart() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let positions_frac = crystal.get_atom_positions();
    let positions_cart = crystal.get_atom_positions_cart();
    
    assert_eq!(positions_frac.len(), positions_cart.len());
    
    // Verify Cartesian positions are finite
    for pos in &positions_cart {
        assert!(pos.x.is_finite());
        assert!(pos.y.is_finite());
        assert!(pos.z.is_finite());
    }
    
    // Verify conversion: convert back and check consistency
    let positions_frac_clone = positions_frac.to_vec();
    crystal.set_atom_positions_from_cart(&positions_cart);
    let positions_frac_roundtrip = crystal.get_atom_positions();
    
    for (orig, roundtrip) in positions_frac_clone.iter().zip(positions_frac_roundtrip.iter()) {
        const TOL: f64 = 1e-10;
        assert!((orig.x - roundtrip.x).abs() < TOL, "x: {} vs {}", orig.x, roundtrip.x);
        assert!((orig.y - roundtrip.y).abs() < TOL, "y: {} vs {}", orig.y, roundtrip.y);
        assert!((orig.z - roundtrip.z).abs() < TOL, "z: {} vs {}", orig.z, roundtrip.z);
    }
}

#[test]
fn test_set_atom_positions_from_cart() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    // Get original positions
    let original_frac = crystal.get_atom_positions().to_vec();
    let positions_cart = crystal.get_atom_positions_cart();
    
    // Modify Cartesian positions slightly
    let mut modified_cart = positions_cart.clone();
    if !modified_cart.is_empty() {
        modified_cart[0].x += 0.1;
    }
    
    // Set from modified Cartesian
    crystal.set_atom_positions_from_cart(&modified_cart);
    
    // Verify positions changed
    let new_frac = crystal.get_atom_positions();
    if !original_frac.is_empty() {
        // First atom should have different fractional coordinates
        assert_ne!(original_frac[0].x, new_frac[0].x);
    }
}

#[test]
fn test_set_atom_positions_from_frac() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let original = crystal.get_atom_positions().to_vec();
    
    // Modify fractional positions
    let mut modified = original.clone();
    if !modified.is_empty() {
        modified[0].x = 0.5;
        modified[0].y = 0.5;
        modified[0].z = 0.5;
    }
    
    crystal.set_atom_positions_from_frac(&modified);
    
    let new = crystal.get_atom_positions();
    if !original.is_empty() {
        assert_eq!(new[0].x, 0.5);
        assert_eq!(new[0].y, 0.5);
        assert_eq!(new[0].z, 0.5);
    }
}

#[test]
fn test_set_lattice_vectors() {
    let mut crystal = create_test_crystal();
    
    // Create a new lattice
    let vec_a = [1.0, 0.0, 0.0];
    let vec_b = [0.0, 1.0, 0.0];
    let vec_c = [0.0, 0.0, 1.0];
    let new_latt = Lattice::new(&vec_a, &vec_b, &vec_c);
    
    crystal.set_lattice_vectors(&new_latt);
    
    let retrieved = crystal.get_latt();
    let a = retrieved.get_vector_a();
    
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 0.0);
    assert_eq!(a.z, 0.0);
}

#[test]
fn test_get_atom_types() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let types = crystal.get_atom_types();
    assert_eq!(types.len(), crystal.get_n_atoms());
    
    // Verify types are 1-based and positive
    for &t in &types {
        assert!(t > 0, "Atom type should be 1-based (positive)");
    }
    
    // Verify atoms of same species have same type
    let n_species = crystal.get_n_unique_species();
    for isp in 0..n_species {
        let indices = crystal.get_atom_indices_of_specie(isp);
        if !indices.is_empty() {
            let expected_type = (isp + 1) as i32;
            for &idx in indices {
                assert_eq!(types[idx], expected_type, 
                           "Atom {} should have type {}", idx, expected_type);
            }
        }
    }
}

#[test]
fn test_get_atom_positions_of_specie() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let n_species = crystal.get_n_unique_species();
    
    if n_species > 0 {
        let indices = crystal.get_atom_indices_of_specie(0);
        let positions = crystal.get_atom_positions_of_specie(0);
        
        assert_eq!(indices.len(), positions.len());
        
        // Verify positions match
        let all_positions = crystal.get_atom_positions();
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(positions[i].x, all_positions[idx].x);
            assert_eq!(positions[i].y, all_positions[idx].y);
            assert_eq!(positions[i].z, all_positions[idx].z);
        }
    }
}

#[test]
fn test_get_cell_mask() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let mask = crystal.get_cell_mask();
    
    // Verify mask is 3x3
    assert_eq!(mask.nrow(), 3);
    assert_eq!(mask.ncol(), 3);
    
    // Verify mask values are 0.0 or 1.0
    for i in 0..3 {
        for j in 0..3 {
            let val = mask[[i, j]];
            assert!(val == 0.0 || val == 1.0, 
                   "Cell mask value should be 0.0 or 1.0, got {}", val);
        }
    }
}

#[test]
fn test_output_file() {
    use std::env;
    
    // Use shared mutex to prevent race conditions with other tests that might create out.crystal
    let _lock = OUTPUT_LOCK.lock().unwrap();
    
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    // Get current working directory (where cargo test runs from)
    let current_dir = env::current_dir().unwrap();
    let output_path = current_dir.join("out.crystal");
    
    // Also check workspace root (parent directory)
    let workspace_root = current_dir.parent().unwrap();
    let workspace_output = workspace_root.join("out.crystal");
    
    // Clean up any existing file first (from other tests)
    let _ = fs::remove_file(&output_path);
    let _ = fs::remove_file(&workspace_output);
    let _ = fs::remove_file("out.crystal");
    
    // Output to file (creates in current working directory as "out.crystal")
    // Note: output() creates file in current working directory where cargo test runs
    crystal.output();
    
    // Wait a tiny bit to ensure file system has flushed
    std::thread::sleep(std::time::Duration::from_millis(10));
    
    // Check all possible locations - cargo test might run from workspace root or crystal dir
    let mut found_file: Option<std::path::PathBuf> = None;
    
    // Check current directory first
    if output_path.exists() {
        found_file = Some(output_path.clone());
    }
    // Check workspace root
    else if workspace_output.exists() {
        found_file = Some(workspace_output.clone());
    }
    // Check relative path
    else if Path::new("out.crystal").exists() {
        found_file = Some(Path::new("out.crystal").canonicalize().unwrap());
    }
    // Check parent directories (in case cargo runs from a subdirectory)
    else {
        let mut check_dir = current_dir.clone();
        for _ in 0..3 {  // Check up to 3 levels up
            let check_path = check_dir.join("out.crystal");
            if check_path.exists() {
                found_file = Some(check_path);
                break;
            }
            if let Some(parent) = check_dir.parent() {
                check_dir = parent.to_path_buf();
            } else {
                break;
            }
        }
    }
    
    // If still not found, try to find it anywhere in the workspace
    if found_file.is_none() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_check = manifest_dir.parent().unwrap().join("out.crystal");
        if workspace_check.exists() {
            found_file = Some(workspace_check);
        }
    }
    
    if found_file.is_none() {
        // Debug: print current directory to help diagnose
        eprintln!("Current dir: {:?}", current_dir);
        eprintln!("Output path: {:?}", output_path);
        eprintln!("Workspace output: {:?}", workspace_output);
        eprintln!("Manifest dir: {:?}", env!("CARGO_MANIFEST_DIR"));
        eprintln!("Relative path exists: {}", Path::new("out.crystal").exists());
        panic!("Output file should be created but was not found in any expected location");
    }
    
    let file_path = found_file.unwrap();
    
    // Read it back and verify basic structure
    let mut crystal2 = Crystal::new();
    crystal2.read_file(file_path.to_str().unwrap());
    
    assert_eq!(crystal.get_n_atoms(), crystal2.get_n_atoms());
    assert_eq!(crystal.get_n_unique_species(), crystal2.get_n_unique_species());
    
    // Clean up
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_coordinate_conversion_roundtrip() {
    // Create a simple cubic crystal
    let mut crystal = Crystal::new();
    let vec_a = [5.0 * ANG_TO_BOHR, 0.0, 0.0];
    let vec_b = [0.0, 5.0 * ANG_TO_BOHR, 0.0];
    let vec_c = [0.0, 0.0, 5.0 * ANG_TO_BOHR];
    let latt = Lattice::new(&vec_a, &vec_b, &vec_c);
    crystal.set_lattice_vectors(&latt);
    
    // Set some fractional positions manually
    // We need to set atom_species and atom_positions, then rebuild indices
    // For this test, let's use file reading which sets everything up correctly
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    crystal.read_file(d.to_str().unwrap());
    
    // Get original fractional positions
    let original_frac: Vec<Vector3f64> = crystal.get_atom_positions().to_vec();
    
    // Convert to Cartesian
    let cart = crystal.get_atom_positions_cart();
    
    // Convert back to fractional
    crystal.set_atom_positions_from_cart(&cart);
    let roundtrip_frac = crystal.get_atom_positions();
    
    // Verify roundtrip accuracy
    const TOL: f64 = 1e-12;
    for (orig, rt) in original_frac.iter().zip(roundtrip_frac.iter()) {
        assert!((orig.x - rt.x).abs() < TOL, 
               "x coordinate mismatch: {} vs {}", orig.x, rt.x);
        assert!((orig.y - rt.y).abs() < TOL,
               "y coordinate mismatch: {} vs {}", orig.y, rt.y);
        assert!((orig.z - rt.z).abs() < TOL,
               "z coordinate mismatch: {} vs {}", orig.z, rt.z);
    }
}

#[test]
fn test_species_grouping() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    let n_species = crystal.get_n_unique_species();
    let n_atoms = crystal.get_n_atoms();
    
    // Count atoms across all species
    let mut total_counted = 0;
    for isp in 0..n_species {
        let indices = crystal.get_atom_indices_of_specie(isp);
        total_counted += indices.len();
        
        // Verify indices are unique within species
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                assert_ne!(indices[i], indices[j], 
                           "Species {} should not have duplicate atom indices", isp);
            }
        }
    }
    
    // Verify all atoms are accounted for
    assert_eq!(total_counted, n_atoms, 
              "Total atoms counted across species should match total atoms");
}

#[test]
fn test_empty_crystal() {
    let crystal = Crystal::new();
    
    assert_eq!(crystal.get_n_atoms(), 0);
    assert_eq!(crystal.get_n_unique_species(), 0);
    assert!(crystal.get_unique_species().is_empty());
    assert!(crystal.get_atom_positions().is_empty());
    assert!(crystal.get_atom_species().is_empty());
}

#[test]
fn test_file_with_empty_lines() {
    // Create a temporary test file with empty lines
    let test_file = "test_empty_lines.crystal";
    let content = "1.0 1.0 1.0\n1.0 0.0 0.0 T T T\n0.0 1.0 0.0 T T T\n0.0 0.0 1.0 T T T\n\nSi 0.0 0.0 0.0\n\n";
    
    fs::write(test_file, content).unwrap();
    
    let mut crystal = Crystal::new();
    crystal.read_file(test_file);
    
    // Should handle empty lines gracefully
    assert!(crystal.get_n_atoms() > 0);
    
    // Clean up
    let _ = fs::remove_file(test_file);
}

#[test]
fn test_display_output() {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    
    // display() should not panic
    crystal.display();
}

#[test]
fn test_lattice_retrieval() {
    let mut crystal = create_test_crystal();
    
    // Create a specific lattice
    let vec_a = [2.0, 0.0, 0.0];
    let vec_b = [0.0, 2.0, 0.0];
    let vec_c = [0.0, 0.0, 2.0];
    let latt = Lattice::new(&vec_a, &vec_b, &vec_c);
    
    crystal.set_lattice_vectors(&latt);
    
    let retrieved = crystal.get_latt();
    let a = retrieved.get_vector_a();
    let b = retrieved.get_vector_b();
    let c = retrieved.get_vector_c();
    
    assert_eq!(a.x, 2.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(c.z, 2.0);
}

#[test]
fn test_crystal_integration() {
    use std::fs;
    use std::path::Path;
    
    // Use shared mutex to prevent race conditions with other tests that might create out.crystal
    let _lock = OUTPUT_LOCK.lock().unwrap();
    
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.push("src/in.crystal");
    println!("{:?}", d);
    let mut crystal = Crystal::new();
    crystal.read_file(d.to_str().unwrap());
    crystal.display();
    crystal.output();
    
    // Verify output file exists (check multiple locations)
    let current_dir = std::env::current_dir().unwrap();
    let output_path = current_dir.join("out.crystal");
    let workspace_output = current_dir.parent().unwrap().join("out.crystal");
    
    let file_exists = Path::new("out.crystal").exists() 
        || output_path.exists() 
        || workspace_output.exists();
    
    assert!(file_exists, "Output file should be created");
    
    // Clean up
    if output_path.exists() {
        let _ = fs::remove_file(&output_path);
    }
    if workspace_output.exists() {
        let _ = fs::remove_file(&workspace_output);
    }
    if Path::new("out.crystal").exists() {
        let _ = fs::remove_file("out.crystal");
    }
}
