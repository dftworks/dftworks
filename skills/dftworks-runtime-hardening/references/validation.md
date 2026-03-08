# Runtime Hardening Validation

## Default Validation Set

Run the smallest relevant set that covers the changed boundary:

```bash
cargo test -p control
cargo test -p crystal
cargo test -p pspot
```

## `pw` Validation

If host MPI is available, host `cargo check -p pw` is fine.

If host MPI is not available, use Docker:

```bash
docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app rust-dev bash -lc 'source $HOME/.cargo/env && cargo check -p pw'
```

## When To Go Further

- If runtime behavior or artifacts changed beyond parsing/control flow, consider the regression scripts in `README.md`.
- If a workflow wrapper assumption changed, inspect `workflow/src/main.rs` as well.

## Reporting

In the close-out, state:

- what you ran,
- what needed Docker,
- and any validation you could not run.
