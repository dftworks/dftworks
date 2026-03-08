---
name: dftworks-runtime-hardening
description: Use for implementing and validating DFTWorks runtime-path fixes in pw, control, crystal, pspot, restart/checkpoint, or output/finalization code. Follow the repo's patterns for replacing panics with Result-based boundaries, preserving MPI-safe shutdown, and validating changes with host tests plus Docker cargo check for pw when MPI is unavailable.
---

# DFTWorks Runtime Hardening

## Overview

Use this skill when the task is to fix runtime-path issues in DFTWorks rather than just review them. It is for changes that harden bootstrap, validation, restart, persistence, export, or shutdown behavior.

## When To Use

- A runtime review found swallowed errors, panic-based loaders, or incorrect success exits.
- You need to add or tighten early validation for runtime modes, inputs, or cross-file consistency.
- You are converting boundary APIs to `try_*`/`Result` while keeping compatibility for older callers.
- You need to validate runtime-path fixes without relying on host MPI availability.

## Hardening Workflow

1. Trace the failing path end-to-end.
   Follow the call chain from parser/input boundary to `pw` bootstrap, orchestration, finalization, and `shutdown_and_exit`.

2. Harden the boundary first.
   Prefer adding a fallible `try_*` API at file-loading or orchestration boundaries. Keep an existing panic wrapper only if other callers still rely on it.

3. Preserve process semantics.
   In `pw`, final user-visible runtime failures should propagate back to `main` and terminate through `shutdown_and_exit(1)`. Do not replace a hard failure with a warning unless the feature is explicitly optional.

4. Add cross-file checks when needed.
   If one file references species, modes, or artifacts defined elsewhere, validate that relationship during bootstrap rather than letting it fail much later.

5. Add small regression tests close to the parser or helper you changed.
   Use temp files for malformed `in.crystal` / `in.pot`-style cases instead of broad integration tests when the bug is boundary-local.

6. Validate with the standard sequence.
   Use `scripts/validate_runtime_changes.sh` for the common unit-test plus `pw` check flow, and read `references/validation.md` for the rationale and fallbacks.

## Expected Implementation Patterns

- Boundary loaders: `read_file()` may keep panic behavior for legacy callers, but add `try_read_file()` or equivalent for runtime bootstrap.
- Finalization/export: return `Result` up the call chain if an artifact is required for a successful run.
- Diagnostics: if a counter or metric claims to cover runtime behavior, enable it before the relevant allocations/work begin or explicitly track generations/phases.

## References

- Read `references/runtime-hardening.md` for repo-specific implementation rules.
- Read `references/validation.md` before closing out a runtime fix.
