## Project Skills

Project-local skills for this repository live under `skills/`.

### Available skills

- `dftworks-runtime-review`: Use when reviewing DFTWorks runtime-path changes in `pw`, `control`, `crystal`, `pspot`, restart/checkpoint, or output/finalization code.
- `dftworks-runtime-hardening`: Use when implementing and validating fixes for DFTWorks runtime-path issues such as panic-based loaders, swallowed errors, incorrect exit behavior, or weak bootstrap validation.
- `dftworks-regression-testing`: Use when selecting and running the right DFTWorks validation and regression matrix for `pw`, `workflow`, input/bootstrap, symmetry, XC-mode, or Wannier-related changes.

### Usage notes

- If the user names one of these skills, or the task clearly matches, open that skill's `SKILL.md` and follow it.
- Prefer these project-local skills over generic review/fix workflows for DFTWorks runtime-path work.
