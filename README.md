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

## HDF5

The Rust crate [hdf5 0.8.1](https://docs.rs/hdf5/latest/hdf5/) requires HDF5 version 1.10.

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

1. Run `pw` to get converged wavefunctions (`out.wfc*.hdf5`).
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

# Test the code

In the directory test_example/si-oncv/scf, run the following command.

<code>../../../target/release/pw</code>

which will give the following output:

```
   ------------------------------ control parameters ------------------------------

   restart                      =              false
   spin_scheme                  =            nonspin
   pot_scheme                   =                upf
   eigen_solver                 =                pcg 
   energy_conv_eps              =           1.000E-6 eV
   eig_conv_eps                 =           1.000E-6 eV
   scf_harris                   =              false
   scf_max_iter                 =                 60
   scf_min_iter                 =                  1
   smearing_scheme              =                mp2
   temperature                  =                  0 K
   ecut                         =            600.000 eV
   ecutrho                      =           2400.000 eV
   nband                        =                  8
   scf_rho_mix_scheme           =            broyden
   scf_rho_mix_alpha            =                0.8
   scf_rho_mix_beta             =               0.01
   geom_optim_cell              =               true
   geom_optim_scheme            =               diis
   geom_optim_history_steps     =                  4
   geom_optim_max_steps         =                  1
   geom_optim_alpha             =                0.7
   geom_optim_force_tolerance   =               0.01 eV/A
   geom_optim_stress_tolerance  =               0.05 kbar

   Si1 : pot/Si-sr.upf

   -------------------------------- k-points (fractional) ---------------------------------

             nkpt = 8

             index         k1               k2               k3         degeneracy 
               1      0.000000000000   0.000000000000   0.000000000000      1      
               2      0.500000000000   0.000000000000   0.000000000000      1      
               3      0.000000000000   0.500000000000   0.000000000000      1      
               4      0.500000000000   0.500000000000   0.000000000000      1      
               5      0.000000000000   0.000000000000   0.500000000000      1      
               6      0.500000000000   0.000000000000   0.500000000000      1      
               7      0.000000000000   0.500000000000   0.500000000000      1      
               8      0.500000000000   0.500000000000   0.500000000000      1      
FFTGrid : 32 x 32 x 32
npw_rho = 10777

   ---------------------------------- crystal structure -----------------------------------

   lattice_vectors

   a =       0.000000000000        2.710178582861        2.710178607242
   b =       2.710178612667        0.000000000000        2.710178685276
   c =       2.710178688029        2.710178736976        0.000000000000

   natoms = 2
   atom_positions

                fractional                                                cartesian (A)

   1    Si1 :  -0.125000415419   -0.124999840723   -0.125000173733       -0.677544701766       -0.677546261687       -0.677545355754
   2    Si1 :   0.124999795051    0.125000071769    0.124999616576        0.677543817946        0.677543070384        0.677544300624

   Si1 : [1, 2]
   load charge density from out.scf.rho
   initial_charge = 7.999999999999934

   #step: geom-1

            eps(eV)  Fermi(eV)           charge               Eharris(Ry)                  Escf(Ry)       dE(eV)
      1:   1.000E-2    6.656E0       8.000000E0         -1.567798259749E1         -1.567481336904E1     4.312E-2
      2:   1.000E-3    6.647E0       8.000000E0         -1.568376954563E1         -1.568686918077E1     4.217E-2
      3:   1.000E-4    6.641E0       8.000000E0         -1.568479022338E1         -1.568664437326E1     2.523E-2
      4:   1.000E-6    6.641E0       8.000000E0         -1.568514354165E1         -1.568645460218E1     1.784E-2
      5:   1.000E-6    6.641E0       8.000000E0         -1.568514826851E1         -1.568621031783E1     1.445E-2
      6:   1.000E-6    6.641E0       8.000000E0         -1.568514654946E1         -1.568526448899E1     1.605E-3
      7:   2.006E-7    6.641E0       8.000000E0         -1.568514383192E1         -1.568517069180E1     3.654E-4
      8:   4.568E-8    6.641E0       8.000000E0         -1.568514270512E1         -1.568508615225E1     7.694E-4
      9:   9.618E-8    6.641E0       8.000000E0         -1.568514267187E1         -1.568513790637E1     6.484E-5
     10:   8.105E-9    6.641E0       8.000000E0         -1.568514277341E1         -1.568514142638E1     1.833E-5
     11:   2.291E-9    6.641E0       8.000000E0         -1.568514277682E1         -1.568514205071E1     9.879E-6
     12:   1.235E-9    6.641E0       8.000000E0         -1.568514277693E1         -1.568514324159E1     6.322E-6
     13:  7.903E-10    6.641E0       8.000000E0         -1.568514277701E1         -1.568514256470E1     2.889E-6
     14:  3.611E-10    6.641E0       8.000000E0         -1.568514277705E1         -1.568514297279E1     2.663E-6
     15:  3.329E-10    6.641E0       8.000000E0         -1.568514277708E1         -1.568514263767E1     1.897E-6
     16:  2.371E-10    6.641E0       8.000000E0         -1.568514277709E1         -1.568514282775E1     6.891E-7

     scf_convergence_success     

   kpoint-1 npws = 1363
     k_frac = [ 0.00000000, 0.00000000, 0.00000000 ]
     k_cart = [ 0.00000000, 0.00000000, 0.00000000 ] (1/a0)

       1             -5.713884     2.000000
       2              6.417600     2.000000
       3              6.417601     2.000000
       4              6.417606     2.000000
       5              8.847432     0.000000
       6              8.847437     0.000000
       7              8.847442     0.000000
       8              9.596031     0.000000

   kpoint-2 npws = 1350
     k_frac = [ 0.50000000, 0.00000000, 0.00000000 ]
     k_cart = [ -0.30670661, 0.30670661, 0.30670660 ] (1/a0)

       1             -3.327425     2.000000
       2             -0.751176     2.000000
       3              5.157486     2.000000
       4              5.157488     2.000000
       5              7.783598     0.000000
       6              9.637182     0.000000
       7              9.637190     0.000000
       8             13.687437     0.000000

   kpoint-3 npws = 1350
     k_frac = [ 0.00000000, 0.50000000, 0.00000000 ]
     k_cart = [ 0.30670660, -0.30670660, 0.30670660 ] (1/a0)

       1             -3.327427     2.000000
       2             -0.751172     2.000000
       3              5.157485     2.000000
       4              5.157488     2.000000
       5              7.783581     0.000000
       6              9.637190     0.000000
       7              9.637194     0.000000
       8             13.687441     0.000000

   kpoint-4 npws = 1338
     k_frac = [ 0.50000000, 0.50000000, 0.00000000 ]
     k_cart = [ -0.00000001, 0.00000001, 0.61341320 ] (1/a0)

       1             -1.514199     2.000000
       2             -1.514198     2.000000
       3              3.431573     2.000000
       4              3.431573     2.000000
       5              6.865178     0.000000
       6              6.865179     0.000000
       7             16.428217     0.000000
       8             17.749600     0.000000

   kpoint-5 npws = 1350
     k_frac = [ 0.00000000, 0.00000000, 0.50000000 ]
     k_cart = [ 0.30670660, 0.30670659, -0.30670659 ] (1/a0)

       1             -3.327428     2.000000
       2             -0.751170     2.000000
       3              5.157485     2.000000
       4              5.157487     2.000000
       5              7.783581     0.000000
       6              9.637191     0.000000
       7              9.637193     0.000000
       8             13.687435     0.000000

   kpoint-6 npws = 1338
     k_frac = [ 0.50000000, 0.00000000, 0.50000000 ]
     k_cart = [ -0.00000002, 0.61341320, 0.00000002 ] (1/a0)

       1             -1.514199     2.000000
       2             -1.514198     2.000000
       3              3.431572     2.000000
       4              3.431573     2.000000
       5              6.865179     0.000000
       6              6.865179     0.000000
       7             16.428297     0.000000
       8             16.428326     0.000000

   kpoint-7 npws = 1338
     k_frac = [ 0.00000000, 0.50000000, 0.50000000 ]
     k_cart = [ 0.61341320, -0.00000001, 0.00000001 ] (1/a0)

       1             -1.514199     2.000000
       2             -1.514198     2.000000
       3              3.431573     2.000000
       4              3.431573     2.000000
       5              6.865177     0.000000
       6              6.865180     0.000000
       7             16.428216     0.000000
       8             17.749827     0.000000

   kpoint-8 npws = 1350
     k_frac = [ 0.50000000, 0.50000000, 0.50000000 ]
     k_cart = [ 0.30670659, 0.30670660, 0.30670661 ] (1/a0)

       1             -3.327426     2.000000
       2             -0.751174     2.000000
       3              5.157485     2.000000
       4              5.157489     2.000000
       5              7.783597     0.000000
       6              9.637181     0.000000
       7              9.637191     0.000000
       8             13.687438     0.000000


   ---------------- total-force (cartesian) (eV/A) ----------------    ------------- atomic-positions (cartesian) (A) -------------

    1   Si1  :        -0.000050        -0.000014         0.000019            -0.677545             -0.677546             -0.677545
    2   Si1  :         0.000004         0.000017        -0.000031             0.677544              0.677543              0.677544

   ---------------------------- local -----------------------------

    1   Si1  :         0.000023        -0.000051        -0.000021
    2   Si1  :        -0.000026        -0.000049        -0.000009

   -------------------------- non-local ---------------------------

    1   Si1  :        -0.000050         0.000038         0.000028
    2   Si1  :         0.000007         0.000066        -0.000011

   ---------------------------- Ewald -----------------------------

    1   Si1  :        -0.000023         0.000000         0.000011
    2   Si1  :         0.000023        -0.000000        -0.000011

   ----------------------------- nlcc -----------------------------

    1   Si1  :         0.000000         0.000000         0.000000
    2   Si1  :         0.000000         0.000000         0.000000

   ------------------------------------ stress (kbar) -------------------------------------
     total
                  |            70.330635            -0.000231            -0.000071   |
                  |            -0.000232            70.330614             0.000349   |
                  |            -0.000072             0.000347            70.330668   |
     kinetic
                  |          2399.334464             0.000087            -0.000044   |
                  |             0.000087          2399.334495            -0.000204   |
                  |            -0.000044            -0.000204          2399.334443   |
     Hartree
                  |           228.697100            -0.000138             0.000006   |
                  |            -0.000138           228.697102             0.000331   |
                  |             0.000006             0.000331           228.697101   |
     xc
                  |          -809.374507             0.000000             0.000000   |
                  |             0.000000          -809.374507             0.000000   |
                  |             0.000000             0.000000          -809.374507   |
     xc_nlcc
                  |             0.000000             0.000000             0.000000   |
                  |             0.000000             0.000000             0.000000   |
                  |             0.000000             0.000000             0.000000   |
     local
                  |         -1098.457253             0.001189            -0.000084   |
                  |             0.001189         -1098.457289            -0.002953   |
                  |            -0.000084            -0.002953         -1098.457174   |
     non-local
                  |          2421.558370            -0.000165             0.000059   |
                  |            -0.000165          2421.558338             0.000471   |
                  |             0.000058             0.000469          2421.558351   |
     Ewald
                  |         -3071.427539            -0.001204            -0.000008   |
                  |            -0.001204         -3071.427524             0.002705   |
                  |            -0.000008             0.002705         -3071.427546   |

   geom_exit_max_steps_reached : 1    

   -------------------------------------- statistics --------------------------------------

   Total           :                9.77 seconds             0.00 hours
```
