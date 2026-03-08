# DFTWorks Runtime Review Checklist

## Primary Runtime Path

Review these in order when a change looks runtime-facing:

1. `control/src/lib.rs`
2. `crystal/src/lib.rs`
3. `pspot/src/lib.rs`
4. `pw/src/orchestration/bootstrap.rs`
5. `pw/src/orchestration/construction.rs`
6. `scf/src/*` and `kscf/src/*` if execution semantics changed
7. `pw/src/orchestration/outputs.rs`
8. `pw/src/restart.rs`
9. `pw/src/main.rs`

## High-Signal Questions

- Does the parser accept a mode or option that the runtime never consumes?
- If a boundary function can fail, does it return `Result` or panic?
- After MPI/runtime init, does any malformed input still crash instead of cleanly reporting an error?
- If an export/checkpoint/provenance step fails, does the program still finish successfully?
- Are saved artifacts aligned with the requested mode (`save_rho`, `save_wfc`, Wannier export, restart)?
- Do optional diagnostics measure what they claim to measure, or are they mixing states/phases?

## Common Finding Patterns In This Repo

- `unwrap()` / `panic!()` in input loaders and bootstrap helpers.
- Capability matrix added in `control`, but no matching enforcement deeper in `pw`.
- Finalization code that prints a warning and keeps going.
- Restart preflight and output persistence using slightly different invariants.
- Workflow wrapper stages assuming semantics that `pw` itself does not honor.

## Validation Commands

Run the smallest relevant set:

```bash
cargo test -p control
cargo test -p crystal
cargo test -p pspot
```

For `pw`, prefer host first only if MPI is available. Otherwise use Docker:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && cargo check -p pw'
```

If the review is broader than API/control-flow inspection, also consider the repo regression scripts documented in `README.md`.
