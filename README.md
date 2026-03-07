**The goal of this project [dftworks](https://github.com/dftworks/dftworks) is to employ Rust as the programming language to implement a plane-wave pseudopotential density functional theory simulation package.**

# Code structure

* Main program: pw
* Testing: test_example
* Library: all others


# Docker build & run
<code>just docker-build</code>

<code>just docker-run</code>

# Local development environment

## Install Rust
If you are running macOS, Linux, or another Unix-like Operating Systems, to set up the Rust working environment, please run the following command in your terminal and then follow the on-screen instructions.

<code>curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh</code><br/>
<br/>

All Rust development tools will be installed to the ~/.cargo/bin directory which needs to be added to the definition of the environmental variable PATH. This can be done by adding the following line to ~/.bash_profile.

<code>export PATH=~/.cargo/bin:$PATH</code>
<br/>

Running <code>source ~/.bash_profile</code> will update PATH.

## Download the code

<code>git clone https://github.com/dftworks/dftworks.git</code>

## Numerical libraries

* lapack
* blas
* fftw

## Symmetry analysis

Symmetry detection and operations are implemented in-tree (self-contained) via the `symmetry` and `symops` crates. No external symmetry library installation is required.
Detailed workflow documentation: `symmetry/SYMMETRY_DETECTION_WORKFLOW.md`.

# Build the code

In the directory dftworks, run the following command.

<code>cargo build --release</code>

This will download the dependency modules and compile the code to generate the executable **pw** in the directory **target/release**.

# Workflow wrapper (`dwf`)

A lightweight stage wrapper is available for SCF / NSCF / bands / Wannier orchestration:

```bash
cargo run -p workflow --bin dwf -- help
```

Supported commands:

```bash
cargo run -p workflow --bin dwf -- validate <case_dir> [--config <yaml>]
cargo run -p workflow --bin dwf -- run scf <case_dir>
cargo run -p workflow --bin dwf -- run nscf <case_dir> --from scf:latest
cargo run -p workflow --bin dwf -- run bands <case_dir> --from latest
cargo run -p workflow --bin dwf -- run wannier <case_dir> --from latest
cargo run -p workflow --bin dwf -- run pipeline <case_dir> [--stages scf,nscf,bands,wannier]
cargo run -p workflow --bin dwf -- status <case_dir> [--config <yaml>]
cargo run -p workflow --bin dwf -- properties <run_dir> <scf|nscf|bands> [--log out.pw.log] [--dos-sigma 0.1] [--dos-ne 500] [--dos-emin -10] [--dos-emax 20] [--dos-format dat|csv|json] [--fermi-tol 0.2]
```

For `scf` / `nscf` / `bands`, `dwf run` now writes standardized machine-readable outputs under:

```text
<run_dir>/properties/
  summary.json
  timings.csv
  energy.csv      # when available
  force.csv       # when available
  stress.csv      # when available
  dos.dat         # nscf: total DOS from NSCF eigenvalues
  band_gap.json   # nscf: direct/indirect gap + VBM/CBM k-point indices
  fermi_consistency.json   # nscf: SCF/NSCF/postprocess Fermi checks
```

Recommended case layout:

```text
<case_dir>/
  dwf.yaml                    # optional, auto-discovered
  common/
    in.crystal
    in.pot
    in.spin            # optional, required for spin runs
  scf/
    in.ctrl
    in.kmesh
  nscf/
    in.ctrl
    in.kmesh
  bands/
    in.ctrl
    in.kline
  wannier/
    in.ctrl
    in.kmesh
    in.proj            # optional, e.g. Si1:sp3
```

`dwf` defaults pipeline order to `scf -> nscf -> bands -> wannier`.
If `wannier/in.proj` is present, `dwf` injects those projector lines into
generated `*.win` files before running `w90-amn`.

YAML config example:

```yaml
common_dir: common
stages:
  scf:
    dir: scf
  nscf:
    dir: nscf
  bands:
    dir: bands
  wannier:
    dir: wannier
pipeline:
  stages:
    - scf
    - nscf
    - bands
    - wannier
```

Full pipeline example:

* `test_example/si-oncv/workflow-pipeline-yaml` (YAML-driven full pipeline, NSCF on 4x4x4)
* `just si-dwf-pipeline-keep-all` (docker-based test that keeps outputs under `runs/`)

# Wannier90 interface (step-by-step)

The Wannier90 workflow is now split into explicit post-SCF steps:

1. Run `pw` to get converged wavefunctions (`out.wfc*`).
2. Run `w90-win` to write Wannier90 input template(s) (`*.win`).
3. Edit the `begin projections ... end projections` block in `*.win`.
4. Run `w90-amn` to write overlap files (`*.nnkp`, `*.mmn`, `*.amn`) from saved wavefunctions.
5. Run `wannier90.x <seed>`.

Required `in.ctrl` keys:

```
wannier90_seedname = dftworks
wannier90_num_wann = 8
wannier90_num_iter = 200
```

Recommended for SCF:

```
save_wfc = true
wannier90_export = true
```

`wannier90_export = true` keeps writing `*.eig` at the end of SCF, so the full Wannier90 dataset is available for runs that need eigenvalues.

Commands:

```bash
cargo run -p wannier90 --bin w90-win
cargo run -p wannier90 --bin w90-amn
cargo run -p wannier90 --bin w90-proj -- --seed <seed>
```

Notes:

* `w90-win` / `w90-amn` require `kpts_scheme = kmesh`.
* Non-spin uses `<seed>.*`; collinear spin uses `<seed>.up.*` and `<seed>.dn.*`.
* `w90-amn` reads projector semantics from `*.nnkp` generated by `wannier90.x -pp` (QE-style) and requires `wannier90.x` to be installed and available in `PATH`.
* For `sp3` entries, `w90-amn` now applies an explicit tetrahedral hybridization transform when writing `.amn`.
* `w90-proj` reads `<seed>.amn/.eig/.nnkp` and writes `pdos_*.dat`, `fatband_*.dat`, and `pdos_validation_report.txt`.
* `w90-proj` caches parsed projection weights in `<seed>.proj.cache` (or `--cache <file>`) to speed up repeated analyses.
* `w90-proj` normalizes projection weights per `(k, band)` state so `sum(PDOS)` is directly comparable with total DOS.

End-to-end examples are provided in:

* `test_example/si-oncv/wannier90` (non-spin)
* `test_example/si-oncv/wannier90-spin` (collinear spin: `up`/`dn` channels)
* `test_example/si-oncv/wannier90-projected` (non-spin with explicit `sp3` projectors, `num_wann = 8`)
* `test_example/si-oncv/workflow-pipeline-yaml` (full YAML pipeline with `scf -> nscf -> bands -> wannier`)
* `test_example/si-oncv/symmetry-enabled` (SCF + relax with `symmetry = true` and k-mesh/force/stress symmetry checks)

# DFT+U (Dudarev MVP)

DFT+U is available as a collinear-spin/non-spin MVP using pseudo-atomic
`PP_CHI` channels from the pseudopotential as local projectors.

Add these keys in `in.ctrl`:

```ini
hubbard_u_enabled = true
hubbard_species = Si1
hubbard_l = 1
hubbard_u = 4.0
hubbard_j = 0.0
```

Notes:

* `hubbard_u_eff = hubbard_u - hubbard_j` (Dudarev form).
* `hubbard_species` must match the species label used in `in.crystal`/`in.pot`.
* The pseudopotential must provide `PP_CHI` for the requested `hubbard_l`.
* Noncollinear (`spin_scheme = ncl`) is not supported for +U.

Regression check (convergence + non-zero energy shift):

```bash
bash scripts/run_hubbard_u_regression.sh
```

# HSE06 (screened hybrid, gamma-only MVP)

Set in `in.ctrl`:

```ini
xc_scheme = hse06
hse06_alpha = 0.25
hse06_omega = 0.11
```

Notes:

* Current implementation is gamma-only (`nkpt = 1`, `k = 0`).
* Local XC part uses PBE; screened exact exchange is added in the SCF Hamiltonian.
* Force/stress currently do not include the hybrid exchange contribution.

Regression check (converged SCF):

```bash
bash scripts/run_hse06_regression.sh
```

# Major-change correctness gate (required)

For each major code change, run correctness validation in Docker before commit/push.

Required baseline command:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && FORCE_BUILD=1 bash ./scripts/run_phase12_regression.sh'
```

If the change touches spin/MPI behavior, also run:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && FORCE_BUILD=0 bash ./scripts/run_spin_mpi_parity.sh'
```

For workspace/performance refactors, also run allocation trace:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && cargo run -p pw --bin workspace_alloc_trace'
```

# Input-driven memory estimate (`memory_estimate`)

Use `memory_estimate` before SCF launch to project memory footprint directly from input files.

```bash
cargo run -p pw --bin memory_estimate -- --case test_example/si-oncv/scf
cargo run -p pw --bin memory_estimate -- --case test_example/si-oncv/wannier90-spin --json
```

Output includes:

- `estimated_bytes_total` (peak per-rank estimate)
- per-component breakdown:
  - `fft_real_space_arrays`
  - `gvectors_and_plane_wave_bases`
  - `wavefunction_eigen_storage`
  - `density_potential_workspaces`
  - `nonlocal_projector_caches`
  - `runtime_process_overhead`
- per-rank and global-cluster totals (`estimated_bytes_total_global`)

Smoke test:

```bash
bash scripts/run_memory_estimator_smoke.sh
```

Calibration snapshot (2026-03-05, Docker `rust-dev`, observed peak sampled from `/proc/<pid>/status` `VmRSS`):

- `test_example/si-oncv/scf` (nonspin): estimate `65.09 MiB`, observed `64.02 MiB` (`+1.7%`)
- `test_example/si-oncv/wannier90-spin` (spin): estimate `95.90 MiB`, observed `93.08 MiB` (`+3.0%`)

Expected planning accuracy for similar workloads: typically within about `+/-20%`.

Runtime allocation statistics in `pw`:

```bash
cd test_example/si-oncv/scf
PW_ALLOC_STATS=1 ../../../target/debug/pw
```

When enabled, `pw` prints a shutdown summary with allocation/deallocation/reallocation counts,
requested/freed bytes, and peak live bytes.

# Engineering simplicity rule (required)

- Rule: "don't overenginner."
- Prefer direct, phase-oriented flow over wrapper-on-wrapper abstractions.
- Add a new wrapper/context layer only when it clearly removes duplication or encapsulates reusable/tested behavior.
- If a helper only forwards arguments without adding logic, inline or remove it.

# Test the code

In the directory test_example/si-oncv/scf (LDA-PZ case), run the following command.

<code>PW_ALLOC_STATS=1 ../../../target/release/pw</code>

which will give the following output:

```text
   ========================================================================================
                                           DFTWorks                                        
                     Self-Consistent Plane-Wave Density Functional Theory                  
   ========================================================================================

   ---------------------------------- system information ----------------------------------

   backend                          = CPU
   fft_threads                      = 1
   fft_planner                      = estimate
   fft_wisdom_file                  = (none)
   hostname                         = 26da94ade217
   os                               = linux
   arch                             = aarch64
   mpi_rank                         = 0
   mpi_ranks                        = 1
   timestamp_utc                    = 2026-03-07T16:41:38Z
   rayon_threads                    = 10 (RAYON_NUM_THREADS=unset, host_threads=10)
   working_directory                = /usr/src/app/test_example/si-oncv/scf

   ---------------------------------- control parameters ----------------------------------

   restart                          = false
   random_seed                      = auto
   provenance_check                 = false
   provenance_manifest              = run.provenance.json
   spin_scheme                      = nonspin
   verbosity                        = normal
   scf_log_format                   = none
   scf_log_file                     = out.scf.iter.jsonl
   fft_threads                      = 1
   fft_planner                      = estimate
   fft_wisdom_file                  = (none)
   electric_field_2d                = 0.000000
   electric_field_axis              = c
   electric_field_origin            = 0.5000
   surface_dipole_correction        = false
   kpoint_schedule                  = cost_aware
   pot_scheme                       = upf
   eigen_solver                     = pcg 
   energy_conv_eps                  = 1.000E-6 eV
   eig_conv_eps                     = 1.000E-6 eV
   scf_harris                       = false
   scf_max_iter                     = 60
   scf_min_iter                     = 1
   smearing_scheme                  = mp2
   temperature                      = 0 K
   ecut                             = 400.000 eV
   ecutrho                          = 1600.000 eV
   nband                            = 8
   wannier90_export                 = false
   wannier90_seedname               = dftworks
   wannier90_num_wann               = 8
   wannier90_num_iter               = 200
   hubbard_u_enabled                = false
   hubbard_species                  = 
   hubbard_l                        = -1
   hubbard_u                        = 0.000000 eV
   hubbard_j                        = 0.000000 eV
   hubbard_u_eff                    = 0.000000 eV
   hse06_alpha                      = 0.25
   hse06_omega                      = 0.11 bohr^-1
   scf_rho_mix_scheme               = broyden
   scf_rho_mix_alpha                = 0.8
   scf_rho_mix_beta                 = 0.01
   geom_optim_cell                  = true
   geom_optim_scheme                = diis
   geom_optim_history_steps         = 4
   geom_optim_max_steps             = 1
   geom_optim_alpha                 = 0.7
   geom_optim_force_tolerance       = 0.01 eV/A
   geom_optim_stress_tolerance      = 0.05 kbar

   -------------------------------- atom pseudopotentials ---------------------------------

   Si1      : pot/Si-sr.upf


   -------------------------------- k-points (fractional) ---------------------------------
             nkpt = 8

             index         k1               k2               k3         degeneracy     weight   
               1      0.000000000000   0.000000000000   0.000000000000      1         0.12500000
               2      0.000000000000   0.000000000000   0.500000000000      1         0.12500000
               3      0.000000000000   0.500000000000   0.000000000000      1         0.12500000
               4      0.000000000000   0.500000000000   0.500000000000      1         0.12500000
               5      0.500000000000   0.000000000000   0.000000000000      1         0.12500000
               6      0.500000000000   0.000000000000   0.500000000000      1         0.12500000
               7      0.500000000000   0.500000000000   0.000000000000      1         0.12500000
               8      0.500000000000   0.500000000000   0.500000000000      1         0.12500000
   provenance_manifest = run.provenance.json
   ----------------------------------- grid information -----------------------------------

   FFTGrid                          = 26 x 26 x 26
   npw_rho                          = 5817
   nfft                             = 17576
   rho_gshells                      = 2875
   gmax_rho                         = 10.83503766 1/bohr (20.47525225 1/A)
   restart                          = false: ignore existing checkpoint files and build atomic initial density
   initial_charge                   = 7.9999960761493645
   kpoint_schedule                  = cost_aware (rank_cost min/avg/max = 46536/46536.00/46536, imbalance=0.00%)

   #step: geom-1

            eps(eV)  Fermi(eV)           charge               Eharris(Ry)                  Escf(Ry)       dE(eV)
      1:   1.000E-2    6.086E0       8.000000E0         -1.570837216924E1         -1.622927045125E1      7.087E0
      2:   1.000E-3    6.678E0       8.000000E0         -1.568335888404E1         -1.566613797296E1     2.343E-1
      3:   1.000E-4    6.661E0       8.000000E0         -1.568445181084E1         -1.568207193728E1     3.238E-2
      4:   4.047E-7    6.644E0       8.000000E0         -1.568497963189E1         -1.568445732839E1     7.106E-3
      5:   8.883E-8    6.643E0       8.000000E0         -1.568496269097E1         -1.568499095442E1     3.845E-4
      6:   4.807E-9    6.643E0       8.000000E0         -1.568496278596E1         -1.568493800215E1     3.372E-4
      7:   4.215E-9    6.643E0       8.000000E0         -1.568496335060E1         -1.568497219309E1     1.203E-4
      8:   1.504E-9    6.643E0       8.000000E0         -1.568496333900E1         -1.568498019664E1     2.294E-4
      9:   2.867E-9    6.643E0       8.000000E0         -1.568496332888E1         -1.568495975451E1     4.863E-5
     10:  6.079E-10    6.643E0       8.000000E0         -1.568496333188E1         -1.568495721450E1     8.323E-5
     11:   1.040E-9    6.643E0       8.000000E0         -1.568496333336E1         -1.568495827848E1     6.878E-5
     12:  8.597E-10    6.643E0       8.000000E0         -1.568496333251E1         -1.568496444996E1     1.520E-5
     13:  1.900E-10    6.643E0       8.000000E0         -1.568496333289E1         -1.568496380635E1     6.442E-6
     14:  8.052E-11    6.643E0       8.000000E0         -1.568496333290E1         -1.568496332088E1     1.635E-7

     scf_convergence_success     

   kpoint-1 npws = 725
     k_frac = [ 0.00000000, 0.00000000, 0.00000000 ]
     k_cart = [ 0.00000000, 0.00000000, 0.00000000 ] (1/a0)

       1             -5.714015     2.000000
       2              6.418608     2.000000
       3              6.418612     2.000000
       4              6.418615     2.000000
       5              8.848735     0.000000
       6              8.848753     0.000000
       7              8.848885     0.000000
       8              9.596015     0.000000

   kpoint-2 npws = 718
     k_frac = [ 0.00000000, 0.00000000, 0.50000000 ]
     k_cart = [ 0.30670660, 0.30670659, -0.30670659 ] (1/a0)

       1             -3.327514     2.000000
       2             -0.751266     2.000000
       3              5.157775     2.000000
       4              5.157776     2.000000
       5              7.785297     0.000000
       6              9.639973     0.000000
       7              9.639980     0.000000
       8             13.688775     0.000000

   kpoint-3 npws = 718
     k_frac = [ 0.00000000, 0.50000000, 0.00000000 ]
     k_cart = [ 0.30670660, -0.30670660, 0.30670660 ] (1/a0)

       1             -3.327512     2.000000
       2             -0.751269     2.000000
       3              5.157774     2.000000
       4              5.157778     2.000000
       5              7.784949     0.000000
       6              9.639932     0.000000
       7              9.639936     0.000000
       8             13.688506     0.000000

   kpoint-4 npws = 740
     k_frac = [ 0.00000000, 0.50000000, 0.50000000 ]
     k_cart = [ 0.61341320, -0.00000001, 0.00000001 ] (1/a0)

       1             -1.514287     2.000000
       2             -1.514269     2.000000
       3              3.431511     2.000000
       4              3.431518     2.000000
       5              6.866619     0.000000
       6              6.866638     0.000000
       7             16.429813     0.000000
       8             16.430713     0.000000

   kpoint-5 npws = 718
     k_frac = [ 0.50000000, 0.00000000, 0.00000000 ]
     k_cart = [ -0.30670661, 0.30670661, 0.30670660 ] (1/a0)

       1             -3.327510     2.000000
       2             -0.751272     2.000000
       3              5.157775     2.000000
       4              5.157778     2.000000
       5              7.785466     0.000000
       6              9.639718     0.000000
       7              9.640053     0.000000
       8             13.688489     0.000000

   kpoint-6 npws = 740
     k_frac = [ 0.50000000, 0.00000000, 0.50000000 ]
     k_cart = [ -0.00000002, 0.61341320, 0.00000002 ] (1/a0)

       1             -1.514286     2.000000
       2             -1.514269     2.000000
       3              3.431510     2.000000
       4              3.431519     2.000000
       5              6.866623     0.000000
       6              6.866624     0.000000
       7             16.429721     0.000000
       8             16.430667     0.000000

   kpoint-7 npws = 740
     k_frac = [ 0.50000000, 0.50000000, 0.00000000 ]
     k_cart = [ -0.00000001, 0.00000001, 0.61341320 ] (1/a0)

       1             -1.514283     2.000000
       2             -1.514272     2.000000
       3              3.431514     2.000000
       4              3.431515     2.000000
       5              6.866649     0.000000
       6              6.866838     0.000000
       7             16.429678     0.000000
       8             16.429914     0.000000

   kpoint-8 npws = 718
     k_frac = [ 0.50000000, 0.50000000, 0.50000000 ]
     k_cart = [ 0.30670659, 0.30670660, 0.30670661 ] (1/a0)

       1             -3.327511     2.000000
       2             -0.751271     2.000000
       3              5.157773     2.000000
       4              5.157779     2.000000
       5              7.785331     0.000000
       6              9.640389     0.000000
       7              9.641084     0.000000
       8             13.688439     0.000000


   ---------------- total-force (cartesian) (eV/A) ----------------    ------------- atomic-positions (cartesian) (A) -------------

    1   Si1  :        -0.000047         0.000017        -0.000112            -0.677545             -0.677546             -0.677545
    2   Si1  :         0.000010        -0.000011        -0.000143             0.677544              0.677543              0.677544

   ---------------------------- local -----------------------------

    1   Si1  :        -0.000035        -0.000062        -0.000117
    2   Si1  :        -0.000063        -0.000058        -0.000110

   -------------------------- non-local ---------------------------

    1   Si1  :         0.000011         0.000079        -0.000006
    2   Si1  :         0.000051         0.000046        -0.000021

   ---------------------------- Ewald -----------------------------

    1   Si1  :        -0.000023         0.000000         0.000011
    2   Si1  :         0.000023        -0.000000        -0.000011

   ----------------------------- nlcc -----------------------------

    1   Si1  :         0.000000         0.000000         0.000000
    2   Si1  :         0.000000         0.000000         0.000000

   ----------------------------- vdW ------------------------------

    1   Si1  :         0.000000         0.000000         0.000000
    2   Si1  :         0.000000         0.000000         0.000000

   ------------------------------------ stress (kbar) -------------------------------------
     total
                  |            70.146168            -0.000797             0.000489   |
                  |            -0.000797            70.145977             0.000303   |
                  |             0.000488             0.000302            70.146901   |
     kinetic
                  |          2399.157663             0.000120             0.000066   |
                  |             0.000120          2399.158329            -0.000114   |
                  |             0.000066            -0.000114          2399.157983   |
     Hartree
                  |           228.675306            -0.000059            -0.000090   |
                  |            -0.000059           228.675248             0.000410   |
                  |            -0.000090             0.000410           228.675296   |
     xc
                  |          -809.370507             0.000000             0.000000   |
                  |             0.000000          -809.370507             0.000000   |
                  |             0.000000             0.000000          -809.370507   |
     xc_nlcc
                  |             0.000000             0.000000             0.000000   |
                  |             0.000000             0.000000             0.000000   |
                  |             0.000000             0.000000             0.000000   |
     local
                  |         -1098.547549             0.000628             0.000519   |
                  |             0.000628         -1098.549131            -0.003283   |
                  |             0.000519            -0.003283         -1098.547559   |
     non-local
                  |          2421.715932            -0.000283             0.000002   |
                  |            -0.000283          2421.716700             0.000586   |
                  |             0.000001             0.000584          2421.716372   |
     Ewald
                  |         -3071.484677            -0.001204            -0.000008   |
                  |            -0.001204         -3071.484662             0.002704   |
                  |            -0.000008             0.002704         -3071.484684   |
     vdW
                  |             0.000000             0.000000             0.000000   |
                  |             0.000000             0.000000             0.000000   |
                  |             0.000000             0.000000             0.000000   |

   geom_exit_max_steps_reached : 1    

   -------------------------------------- statistics --------------------------------------

   Total           :                2.53 seconds             0.00 hours

   ------------------------- runtime memory allocation statistics -------------------------

   alloc_calls                 : 816384            
   dealloc_calls               : 816219            
   realloc_calls               : 363               
   alloc_bytes                 : 81656990 (77.874 MiB)
   dealloc_bytes               : 81168379 (77.408 MiB)
   realloc_old_bytes           : 424352 (0.405 MiB)
   realloc_new_bytes           : 847730 (0.808 MiB)
   gross_requested_bytes       : 82504720 (78.683 MiB)
   gross_freed_bytes           : 81592731 (77.813 MiB)
   net_requested_bytes         : 911989 (0.870 MiB)
   live_bytes                  : 911989 (0.870 MiB)
   peak_live_bytes             : 38845753 (37.046 MiB)
```

# Runtime capability matrix (`pw`)

Unsupported mode combinations are rejected during input validation/preflight with actionable
errors (instead of reaching late runtime panics).

Current supported core combinations:

| Axis | Supported values |
| --- | --- |
| `task` | `scf`, `band` |
| `spin_scheme` | `nonspin`, `spin` |
| `xc_scheme` | `lda-pz`, `lsda-pz`, `pbe`, `hse06` |
| `eigen_solver` | `pcg` |
| `restart=true` | `spin_scheme=nonspin` or `spin_scheme=spin` |

Additional runtime constraint:

- `xc_scheme = hse06` currently requires exactly one Gamma k-point (`k=(0,0,0)`).
