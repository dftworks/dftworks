# DFTWorks Runtime Hardening Notes

## Preferred Fix Shapes

### Input/bootstrap hardening

- Add a fallible `try_*` API at the boundary.
- Return line/file-aware messages for malformed inputs where practical.
- Let `pw/src/orchestration/bootstrap.rs` own cross-file consistency checks.

### Finalization/export hardening

- If an export is required for a successful run, do not just print a warning.
- Return `Result` from orchestration helpers and let `pw/src/main.rs` terminate with `shutdown_and_exit(1)`.

### Capability/runtime alignment

- If `control` parses a mode, either:
  - the runtime must consume it correctly, or
  - bootstrap/preflight must reject it early with a clear message.

### Optional diagnostics

- Make sure measurement starts before the measured work begins.
- If not possible, explicitly track the measurement window instead of reporting totals that span multiple phases.

## Files To Check While Editing

- `pw/src/main.rs`
- `pw/src/orchestration/bootstrap.rs`
- `pw/src/orchestration/outputs.rs`
- `pw/src/restart.rs`
- `control/src/lib.rs`
- `crystal/src/lib.rs`
- `pspot/src/lib.rs`

## Avoid

- Silent downgrade from failure to warning for required artifacts.
- Introducing `process::exit` outside the top-level runtime path.
- Broad refactors when a narrow boundary fix is enough.
- Staging generated run artifacts while testing.
