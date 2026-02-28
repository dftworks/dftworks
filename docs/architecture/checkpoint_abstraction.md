# Checkpoint and Restart Architecture

This document describes the checkpoint/restart system in DFTWorks, including the repository abstraction (E22), metadata validation, and spin/nonspin unification.

## Table of Contents

1. [Overview](#overview)
2. [Repository Abstraction](#repository-abstraction)
3. [Checkpoint Types](#checkpoint-types)
4. [Metadata and Versioning](#metadata-and-versioning)
5. [Restart Workflow](#restart-workflow)
6. [Spin/Nonspin Unification](#spinnonspin-unification)
7. [Adding New Checkpoint Types](#adding-new-checkpoint-types)

---

## Overview

### Purpose

Checkpointing enables:
- **Warm restart** - Resume SCF from converged density/wavefunctions
- **Crash recovery** - Restart from last saved state
- **Post-processing** - Load converged state for band structure, DOS, etc.
- **Workflow composition** - Chain calculations (SCF → NSCF → Wannier)

### Key Features

- **Policy-driven restart** - Explicit `restart=true` flag (not implicit file existence)
- **Metadata validation** - Verify compatibility before loading
- **Schema versioning** - Support migration across format changes
- **Repository abstraction** - Backend-agnostic (currently HDF5, future: packed/parallel formats)
- **Spin/nonspin unification** - Shared code for channel iteration

---

## Repository Abstraction

### Design (E22)

Separate **domain model** (what to save) from **storage codec** (how to save):

```rust
// Domain model - what we checkpoint
pub struct VKEigenVector { /* ... */ }
pub struct RHOR { /* ... */ }

// Repository trait - where/how we save
pub trait CheckpointRepository {
    fn save_eigenvector(&self, ik: usize, channel: SpinChannel, data: &VKEigenVector) -> Result<(), String>;
    fn load_eigenvector(&self, ik: usize, channel: SpinChannel) -> Result<VKEigenVector, String>;
    // ...
}

// Codec - specific storage format
pub trait CheckpointCodec {
    fn write_hdf5(&self, path: &Path, data: &[u8]) -> Result<(), String>;
    fn read_hdf5(&self, path: &Path) -> Result<Vec<u8>, String>;
}
```

### Current Implementation

**HDF5 File-Per-K Repository:**
```rust
pub struct Hdf5FilePerKCheckpointRepository {
    base_dir: PathBuf,
    spin_scheme: SpinScheme,
}

impl CheckpointRepository for Hdf5FilePerKCheckpointRepository {
    fn save_eigenvector(&self, ik: usize, channel: SpinChannel, data: &VKEigenVector) -> Result<(), String> {
        let filename = self.eigenvector_filename(ik, channel);
        let path = self.base_dir.join(&filename);
        data.save_hdf5(&path)
    }
    // ...
}
```

**Filename convention:**
- Nonspin: `out.wfc.k.{ik}.hdf5`
- Spin: `out.wfc.k.{ik}.up.hdf5`, `out.wfc.k.{ik}.dn.hdf5`
- Density: `out.scf.rho.hdf5` (nonspin) or `out.scf.rho.up.hdf5`, `out.scf.rho.dn.hdf5` (spin)

### Future Backends

Repository abstraction enables future formats:

**Packed HDF5** (fewer files for large k-point sets):
```rust
pub struct Hdf5PackedCheckpointRepository {
    // All k-points in one file: out.wfc.hdf5
    // Datasets: /k0/eigenvectors, /k1/eigenvectors, ...
}
```

**Parallel I/O** (MPI collective writes):
```rust
pub struct MpiIOCheckpointRepository {
    // Use MPI-IO or parallel HDF5
    // Each rank writes its local k-points concurrently
}
```

---

## Checkpoint Types

### 1. Density Checkpoints (`RHOR`)

**Purpose:** Save/load converged SCF density for warm restart

**Files:**
- Nonspin: `out.scf.rho.hdf5`
- Spin: `out.scf.rho.up.hdf5`, `out.scf.rho.dn.hdf5`

**Contents:**
```rust
pub struct RHOR {
    pub rhog: Vec<c64>,      // Density in G-space
    pub rho_3d: Array3<c64>, // Density in real-space
}
```

**Metadata:**
```rust
struct DensityMetadata {
    schema_version: u32,
    spin_scheme: String,        // "nonspin", "spin", or "ncl"
    spin_channel: String,       // "total", "up", "dn", or "ncl"
    lattice_hash: u64,          // Lattice compatibility check
    npw_rho: usize,
    fft_shape: [usize; 3],
}
```

**Usage:**
```rust
// Save after SCF convergence
repo.save_density(spin_channel, &converged_density)?;

// Load for restart
if restart {
    let density = repo.load_density(spin_channel)?;
}
```

---

### 2. Wavefunction Checkpoints (`VKEigenVector`)

**Purpose:** Save/load converged wavefunctions for warm restart or post-processing

**Files:**
- Nonspin: `out.wfc.k.{ik}.hdf5` (one file per k-point)
- Spin: `out.wfc.k.{ik}.up.hdf5`, `out.wfc.k.{ik}.dn.hdf5`

**Contents:**
```rust
pub struct VKEigenVector {
    pub vkwfc: Vec<Matrix<c64>>,  // Wavefunctions for each k-point
                                   // vkwfc[ik_local] = Matrix[npw_wfc, nband]
}
```

**Metadata:**
```rust
struct WavefunctionMetadata {
    schema_version: u32,
    spin_scheme: String,
    spin_channel: String,
    ik_global: usize,           // Global k-point index
    lattice_hash: u64,
    cutoff_wfc: f64,
    nband: usize,
    npw_wfc: usize,
}
```

**Usage:**
```rust
// Save after SCF convergence (per k-point)
for ik in local_k_indices {
    repo.save_eigenvector(ik, channel, &vk_eigvec)?;
}

// Load for restart or NSCF
for ik in local_k_indices {
    let wfc = repo.load_eigenvector(ik, channel)?;
}
```

---

### 3. Eigenvalue Checkpoints (`VKEigenValue`)

**Purpose:** Save converged eigenvalues (less critical, can recompute from wavefunctions)

**Files:** Similar structure to wavefunctions

**Contents:**
```rust
pub struct VKEigenValue {
    pub vkeigval: Vec<Vec<f64>>,  // Eigenvalues for each k-point
                                   // vkeigval[ik_local] = Vec[nband]
}
```

---

## Metadata and Versioning

### Schema Versioning

Current schema version: `1`

**Version 0 (legacy):**
- No explicit schema version field
- Minimal metadata

**Version 1 (current):**
- Explicit `schema_version` field
- Full metadata (spin_scheme, lattice_hash, dimensions)
- Spin channel labels

**Future versions:**
- Version 2: Add compression metadata
- Version 3: Add chunk/parallel I/O metadata

### Migration Support

```rust
impl Hdf5FilePerKCheckpointRepository {
    fn migrate_metadata_v0_to_v1(legacy_meta: LegacyMetadata) -> Metadata {
        Metadata {
            schema_version: 1,
            spin_scheme: infer_spin_scheme(&legacy_meta),
            spin_channel: infer_spin_channel(&legacy_meta),
            // ... fill in missing fields with defaults
        }
    }
}
```

### Compatibility Checks

Before loading checkpoint:

```rust
fn validate_checkpoint_compatibility(
    checkpoint_meta: &Metadata,
    runtime_config: &Control,
    crystal: &Crystal,
) -> Result<(), String> {
    // 1. Spin scheme compatibility
    if checkpoint_meta.spin_scheme != runtime_config.spin_scheme {
        return Err(format!(
            "Spin scheme mismatch: checkpoint is '{}', runtime is '{}'",
            checkpoint_meta.spin_scheme,
            runtime_config.spin_scheme
        ));
    }

    // 2. Lattice compatibility
    let runtime_lattice_hash = compute_lattice_hash(crystal);
    if checkpoint_meta.lattice_hash != runtime_lattice_hash {
        return Err("Lattice mismatch - checkpoint from different structure".to_string());
    }

    // 3. Dimension compatibility
    if checkpoint_meta.nband != runtime_config.nband {
        return Err(format!(
            "Band count mismatch: checkpoint has {}, runtime expects {}",
            checkpoint_meta.nband,
            runtime_config.nband
        ));
    }

    // 4. Cutoff compatibility (for wavefunctions)
    if (checkpoint_meta.cutoff_wfc - runtime_config.cutoff_wfc).abs() > 1e-6 {
        return Err(format!(
            "Cutoff mismatch: checkpoint has {}, runtime expects {}",
            checkpoint_meta.cutoff_wfc,
            runtime_config.cutoff_wfc
        ));
    }

    Ok(())
}
```

**Common incompatibilities:**
- Different spin schemes (nonspin ↔ spin)
- Different lattice structures (relaxation changed cell)
- Different k-mesh (need interpolation, not supported)
- Different cutoffs (different plane-wave basis)
- Different number of bands

---

## Restart Workflow

### Explicit Restart Policy (E12)

**Before (implicit):**
```rust
// BAD: Restart based on file existence
if Path::new("out.scf.rho.hdf5").exists() {
    density = load_density();  // Implicit restart
}
```

**After (explicit):**
```rust
// GOOD: Restart controlled by user flag
if control.restart {
    density = repo.load_density(channel)?;
} else {
    density = initialize_density_from_atoms(crystal);
}
```

### Full Restart Flow

```rust
pub fn pw_main() {
    // 1. Parse input
    let control = Control::from_file("in.ctrl")?;

    // 2. Check restart policy
    if control.restart {
        // 3. Create repository
        let repo = Hdf5FilePerKCheckpointRepository::new(".", control.spin_scheme);

        // 4. Load density checkpoint
        let density_restart = match control.spin_scheme {
            SpinScheme::NonSpin => {
                repo.load_density(SpinChannel::Total)?
            }
            SpinScheme::Spin => {
                let rho_up = repo.load_density(SpinChannel::Up)?;
                let rho_dn = repo.load_density(SpinChannel::Dn)?;
                (rho_up, rho_dn)
            }
            SpinScheme::NCL => {
                return Err("NCL restart not yet supported".to_string());
            }
        };

        // 5. Validate checkpoint compatibility
        validate_checkpoint_compatibility(&density_restart.metadata, &control, &crystal)?;

        // 6. Optional: Load wavefunction checkpoints
        if control.restart_wavefunctions {
            for ik in local_k_indices {
                let wfc = repo.load_eigenvector(ik, channel)?;
                // Validate wavefunction metadata
                // Use as initial guess for SCF
            }
        }

        // 7. Run SCF with warm start
        scf.run(/* ... */, Some(density_restart))?;

    } else {
        // Cold start - initialize from atomic densities
        let density = initialize_atomic_density(crystal, atoms, pspots);
        scf.run(/* ... */, None)?;
    }
}
```

### Checkpoint Saving

After SCF convergence:

```rust
// 1. Create repository
let repo = Hdf5FilePerKCheckpointRepository::new(".", control.spin_scheme);

// 2. Save density
match control.spin_scheme {
    SpinScheme::NonSpin => {
        repo.save_density(SpinChannel::Total, &converged_density)?;
    }
    SpinScheme::Spin => {
        repo.save_density(SpinChannel::Up, &converged_density_up)?;
        repo.save_density(SpinChannel::Dn, &converged_density_dn)?;
    }
    _ => { /* ... */ }
}

// 3. Save wavefunctions (optional, controlled by flag)
if control.save_wavefunctions {
    for ik in local_k_indices {
        repo.save_eigenvector(ik, channel, &vk_eigvec)?;
        repo.save_eigenvalue(ik, channel, &vk_eigval)?;
    }
}
```

---

## Spin/Nonspin Unification

### Shared Channel Iteration (E22)

**Before (duplicated code):**
```rust
// Nonspin - one loop
repo.save_density(SpinChannel::Total, &rho)?;

// Spin - separate loops
repo.save_density(SpinChannel::Up, &rho_up)?;
repo.save_density(SpinChannel::Dn, &rho_dn)?;
```

**After (unified helper):**
```rust
// Shared helper for channel iteration
fn iterate_spin_channels(spin_scheme: SpinScheme) -> Vec<SpinChannel> {
    match spin_scheme {
        SpinScheme::NonSpin => vec![SpinChannel::Total],
        SpinScheme::Spin => vec![SpinChannel::Up, SpinChannel::Dn],
        SpinScheme::NCL => vec![SpinChannel::NCL],
    }
}

// Unified save loop
for channel in iterate_spin_channels(spin_scheme) {
    repo.save_density(channel, &density[channel])?;
}
```

### Filename Helpers

```rust
impl Hdf5FilePerKCheckpointRepository {
    fn density_filename(&self, channel: SpinChannel) -> String {
        match (self.spin_scheme, channel) {
            (SpinScheme::NonSpin, SpinChannel::Total) => "out.scf.rho.hdf5".to_string(),
            (SpinScheme::Spin, SpinChannel::Up) => "out.scf.rho.up.hdf5".to_string(),
            (SpinScheme::Spin, SpinChannel::Dn) => "out.scf.rho.dn.hdf5".to_string(),
            _ => panic!("Invalid spin_scheme/channel combination"),
        }
    }

    fn eigenvector_filename(&self, ik: usize, channel: SpinChannel) -> String {
        match (self.spin_scheme, channel) {
            (SpinScheme::NonSpin, SpinChannel::Total) => format!("out.wfc.k.{}.hdf5", ik),
            (SpinScheme::Spin, SpinChannel::Up) => format!("out.wfc.k.{}.up.hdf5", ik),
            (SpinScheme::Spin, SpinChannel::Dn) => format!("out.wfc.k.{}.dn.hdf5", ik),
            _ => panic!("Invalid spin_scheme/channel combination"),
        }
    }
}
```

---

## Adding New Checkpoint Types

### Example: Add SCF Mixing History

**Step 1:** Define checkpoint data
```rust
pub struct MixingHistory {
    pub density_history: Vec<Vec<c64>>,  // Past densities
    pub residual_history: Vec<Vec<c64>>, // Past residuals
    pub iteration_count: usize,
}
```

**Step 2:** Add to repository trait
```rust
pub trait CheckpointRepository {
    // ... existing methods ...

    fn save_mixing_history(&self, history: &MixingHistory) -> Result<(), String>;
    fn load_mixing_history(&self) -> Result<MixingHistory, String>;
}
```

**Step 3:** Implement for HDF5 repository
```rust
impl CheckpointRepository for Hdf5FilePerKCheckpointRepository {
    fn save_mixing_history(&self, history: &MixingHistory) -> Result<(), String> {
        let path = self.base_dir.join("out.scf.mixing.hdf5");
        // Serialize and write to HDF5
        write_hdf5(&path, "density_history", &history.density_history)?;
        write_hdf5(&path, "residual_history", &history.residual_history)?;
        write_hdf5(&path, "iteration_count", &history.iteration_count)?;
        Ok(())
    }

    fn load_mixing_history(&self) -> Result<MixingHistory, String> {
        let path = self.base_dir.join("out.scf.mixing.hdf5");
        // Read from HDF5 and deserialize
        let density_history = read_hdf5(&path, "density_history")?;
        let residual_history = read_hdf5(&path, "residual_history")?;
        let iteration_count = read_hdf5(&path, "iteration_count")?;
        Ok(MixingHistory { density_history, residual_history, iteration_count })
    }
}
```

**Step 4:** Integrate into restart workflow
```rust
if control.restart && control.restart_mixing_history {
    let mixing_history = repo.load_mixing_history()?;
    mixer.restore_history(mixing_history);
}
```

---

## Summary

### Key Features

- ✅ **Repository abstraction** - Backend-agnostic checkpoint API
- ✅ **Explicit restart policy** - No implicit file-based behavior
- ✅ **Metadata validation** - Compatibility checks before loading
- ✅ **Schema versioning** - Support format evolution
- ✅ **Spin/nonspin unification** - Shared channel iteration
- ✅ **Error handling** - `Result`-based APIs with context

### File Layout

```
runs/scf/
├── out.scf.rho.hdf5              # Density (nonspin)
├── out.scf.rho.up.hdf5           # Density spin-up (spin)
├── out.scf.rho.dn.hdf5           # Density spin-down (spin)
├── out.wfc.k.0.hdf5              # Wavefunction k=0 (nonspin)
├── out.wfc.k.1.hdf5              # Wavefunction k=1 (nonspin)
├── out.wfc.k.0.up.hdf5           # Wavefunction k=0 spin-up (spin)
├── out.wfc.k.0.dn.hdf5           # Wavefunction k=0 spin-down (spin)
└── ...
```

### Future Improvements (E17)

- Packed HDF5: Reduce file count for large k-point sets
- Parallel I/O: MPI collective writes for scalability
- Compression: Reduce checkpoint size (lossy/lossless)
- Chunked writes: Stream large datasets

### Further Reading

- `dfttypes/src/checkpoint_repo.rs` - Repository abstraction implementation
- `dfttypes/src/lib.rs` - Checkpoint data types (`RHOR`, `VKEigenVector`)
- `pw/src/restart.rs` - Restart workflow orchestration
- `pw/src/orchestration/outputs.rs` - Checkpoint saving
- `OPTIMIZATION_TODO.md` - E12, E17, E22 task details
