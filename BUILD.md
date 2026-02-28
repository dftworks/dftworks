# Building DFTWorks

This guide provides instructions for building DFTWorks on different platforms.

## Prerequisites

DFTWorks requires the following external dependencies:

- **Rust toolchain** (1.70 or later): https://rust-lang.org/tools/install
- **MPI library**: Open-MPI or MPICH
- **LAPACK/BLAS**: Linear algebra libraries
- **FFTW3**: Fast Fourier Transform library (with thread support)
- **HDF5**: (optional, for checkpoint I/O)
- **Fortran compiler**: gfortran (for LAPACK static linking on macOS)

## Platform-Specific Instructions

### macOS (Homebrew)

```bash
# Install dependencies
brew install rust open-mpi fftw lapack gcc hdf5

# Build DFTWorks
cargo build --release

# Test the build
cargo check
```

**Note for Apple Silicon (M1/M2/M3)**: Homebrew installs to `/opt/homebrew`. The build scripts automatically detect this location.

**Note for Intel Macs**: Homebrew installs to `/usr/local`. The build scripts automatically detect this location.

### macOS (MacPorts)

```bash
# Install dependencies
sudo port install rust mpich-default fftw-3 lapack gcc13 hdf5

# Build DFTWorks
cargo build --release
```

**Note**: MacPorts installs to `/opt/local`. The build scripts automatically detect this location.

### macOS (Conda)

```bash
# Create a conda environment
conda create -n dftworks rust openmpi fftw lapack gcc hdf5 -c conda-forge
conda activate dftworks

# Build DFTWorks
cargo build --release
```

**Note**: The build scripts automatically detect `$CONDA_PREFIX` and use it for library paths.

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential rustc cargo libopenmpi-dev libfftw3-dev liblapack-dev libhdf5-dev gfortran

# Build DFTWorks
cargo build --release
```

### Linux (RHEL/CentOS/Fedora)

```bash
# Install dependencies
sudo yum install rust cargo openmpi-devel fftw-devel lapack-devel hdf5-devel gcc-gfortran

# Load MPI module (if using environment modules)
module load mpi/openmpi-x86_64

# Build DFTWorks
cargo build --release
```

### HPC Clusters

On HPC systems, you typically need to load modules for dependencies:

```bash
# Example for a typical HPC cluster
module load rust openmpi fftw lapack hdf5

# If libraries are in non-standard locations, set environment variables
export MPI_LIB_DIR=/path/to/mpi/lib
export FFTW_DIR=/path/to/fftw
export LAPACK_DIR=/path/to/lapack

# Build DFTWorks
cargo build --release
```

## Environment Variables for Custom Installations

If dependencies are installed in non-standard locations, you can use environment variables to specify paths:

### Library Search Paths

- **`MPI_LIB_DIR`**: Path to MPI library directory
  ```bash
  export MPI_LIB_DIR=/custom/path/to/mpi/lib
  ```

- **`FFTW_DIR`**: Path to FFTW installation directory
  ```bash
  export FFTW_DIR=/custom/path/to/fftw
  # Or specify library directory directly:
  export FFTW_LIB_DIR=/custom/path/to/fftw/lib
  ```

- **`LAPACK_DIR`**: Path to LAPACK library directory
  ```bash
  export LAPACK_DIR=/custom/path/to/lapack/lib
  ```

- **`GFORTRAN_LIB_DIR`**: Path to gfortran runtime libraries (macOS only)
  ```bash
  export GFORTRAN_LIB_DIR=/custom/path/to/gcc/lib
  ```

### Runtime Library Paths (rpath)

- **`DFTWORKS_RPATH`**: Colon-separated list of runtime library search paths
  ```bash
  export DFTWORKS_RPATH=/custom/lib1:/custom/lib2:/custom/lib3
  ```

### Build Control

- **`MATRIX_LINK_GFORTRAN`**: Control gfortran linking on macOS (default: true)
  ```bash
  export MATRIX_LINK_GFORTRAN=false
  ```

## Troubleshooting

### Build Fails with Missing MPI Libraries

**Error**: `ld: library not found for -lmpi`

**Solution**: Ensure MPI is installed and set `MPI_LIB_DIR`:
```bash
# Homebrew (Apple Silicon)
export MPI_LIB_DIR=/opt/homebrew/lib

# Homebrew (Intel)
export MPI_LIB_DIR=/usr/local/lib

# Custom MPI installation
export MPI_LIB_DIR=/path/to/mpi/lib
```

### Build Fails with Missing FFTW

**Error**: `ld: library not found for -lfftw3`

**Solution**: Install FFTW or set `FFTW_DIR`:
```bash
# Homebrew
brew install fftw

# Or set custom path
export FFTW_DIR=/path/to/fftw
```

### Build Fails with Missing LAPACK

**Error**: `ld: library not found for -llapack`

**Solution**: Install LAPACK or set `LAPACK_DIR`:
```bash
# Homebrew
brew install lapack

# Or set custom path
export LAPACK_DIR=/path/to/lapack/lib
```

### Build Fails with Missing gfortran Symbols (macOS)

**Error**: `Undefined symbols: "___gfortran_..."`

**Solution**: Install GCC and set `GFORTRAN_LIB_DIR`:
```bash
# Homebrew
brew install gcc
export GFORTRAN_LIB_DIR=/opt/homebrew/lib/gcc/current

# MacPorts
sudo port install gcc13
export GFORTRAN_LIB_DIR=/opt/local/lib/libgcc
```

### Runtime Error: Library Not Found (macOS/Linux)

**Error**: `dyld: Library not loaded` or `error while loading shared libraries`

**Solution**: Set `DFTWORKS_RPATH` before building:
```bash
export DFTWORKS_RPATH=/custom/lib1:/custom/lib2
cargo clean
cargo build --release
```

Or use system-specific runtime library path variables:
```bash
# macOS
export DYLD_LIBRARY_PATH=/path/to/libs:$DYLD_LIBRARY_PATH

# Linux
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
```

## Verifying the Build

After building, verify the installation:

```bash
# Check that the main binary runs
cargo run -p pw -- --version

# Run tests
cargo test

# Build in release mode for production use
cargo build --release
```

## Cross-Platform Build Matrix

| Platform | MPI | FFTW | LAPACK | Notes |
|----------|-----|------|--------|-------|
| macOS (Homebrew, Apple Silicon) | ✅ | ✅ | ✅ | Auto-detected at `/opt/homebrew` |
| macOS (Homebrew, Intel) | ✅ | ✅ | ✅ | Auto-detected at `/usr/local` |
| macOS (MacPorts) | ✅ | ✅ | ✅ | Auto-detected at `/opt/local` |
| macOS (Conda) | ✅ | ✅ | ✅ | Auto-detected via `$CONDA_PREFIX` |
| Ubuntu/Debian | ✅ | ✅ | ✅ | System packages work out-of-box |
| RHEL/CentOS | ✅ | ✅ | ✅ | May need environment modules |
| HPC Clusters | ✅ | ✅ | ✅ | Use environment variables |

## Getting Help

If you encounter build issues not covered in this guide:

1. Check that all dependencies are installed and accessible
2. Verify library paths using `pkg-config` (if available):
   ```bash
   pkg-config --libs fftw3
   pkg-config --libs ompi
   ```
3. Set environment variables explicitly for custom installations
4. Open an issue at https://github.com/dftworks/dftworks/issues with:
   - Your operating system and version
   - Package manager used (Homebrew, MacPorts, apt, yum, conda, etc.)
   - Full build error output
   - Output of `echo $PATH` and relevant `*_DIR` environment variables
