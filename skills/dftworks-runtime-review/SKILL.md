---
name: dftworks-runtime-review
description: Use for code reviews of DFTWorks runtime-path changes in pw, control, crystal, pspot, restart/checkpoint, or output/finalization logic. Focus on swallowed errors, panic-based input loading, unsupported-mode leaks, checkpoint/export correctness, exit-code behavior, and validation gaps.
---

# DFTWorks Runtime Review

## Overview

Use this skill when reviewing changes that affect how DFTWorks starts, validates inputs, runs SCF/runtime orchestration, persists artifacts, or exits. Prioritize real execution risks and behavioral regressions over style.

## When To Use

- Changes under `pw/src`, especially `main.rs`, `orchestration/`, `restart.rs`, and runtime display or diagnostics.
- Changes under `control/src`, `crystal/src`, `pspot/src`, or other input/bootstrap layers.
- Changes touching checkpoint loading/saving, Wannier export, provenance, output files, or exit behavior.
- Reviews of capability-matrix work, unsupported modes, restart semantics, or optional diagnostics like allocator statistics.

## Review Workflow

1. Map the runtime path first.
   Typical path: `control`/input parse -> bootstrap -> geometry/electronic construction -> SCF run -> output persistence -> finalization -> shutdown.

2. Review boundaries before internals.
   Check file parsing, `Result` propagation, mode validation, restart preflight, export/finalization, and process exit behavior before worrying about local implementation details.

3. Treat these as high-risk bug classes.
   - Parsed configuration that is never enforced at runtime.
   - `Err` branches that only log and continue.
   - Panic-based file/input loading after MPI or runtime setup has already started.
   - Export/checkpoint failures that still produce `exit 0`.
   - Capability matrix acceptance that does not match real implementation.
   - Generated artifacts or caches that can mask missing outputs.

4. Prefer concrete findings.
   Report the user-visible failure mode, the exact file/line, and why the current control flow allows it.

5. Validate what you can.
   Use the commands in `references/review-checklist.md`. If host `pw` validation is blocked by missing MPI, use the Docker fallback and say so explicitly.

## Output Expectations

- Findings first, ordered by severity.
- Include file and line references.
- Keep summaries brief.
- If nothing is found, say that explicitly and mention residual testing gaps.

## References

- Read `references/review-checklist.md` for the DFTWorks-specific hotspots and validation sequence.
