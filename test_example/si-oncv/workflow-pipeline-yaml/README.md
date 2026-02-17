# DWF YAML Pipeline Example (Si)

This case demonstrates the full staged workflow using `dwf.yaml`:

1. `scf` on a `2x2x2` mesh (save charge density)
2. `nscf` on a `4x4x4` mesh (save wavefunctions and Wannier export)
3. `bands` along a k-line path
4. `wannier` post-processing (`w90-win -> w90-amn -> wannier90.x`)

Run from repository root:

```bash
cargo run -p workflow --bin dwf -- validate test_example/si-oncv/workflow-pipeline-yaml
cargo run -p workflow --bin dwf -- run pipeline test_example/si-oncv/workflow-pipeline-yaml
cargo run -p workflow --bin dwf -- status test_example/si-oncv/workflow-pipeline-yaml
```

Docker keep-output recipe:

```bash
just si-dwf-pipeline-keep-all
```

`dwf` auto-discovers `dwf.yaml` in the case directory.

Pseudopotentials are listed in `common/in.pot` as filenames and provided under
`common/pot/`; `dwf` copies required files into each run directory `pot/`.

Projectors for the Wannier step are provided in `wannier/in.proj`:

```text
Si1:sp3
```

`dwf` injects these lines into generated `*.win` files before running `w90-amn`.
