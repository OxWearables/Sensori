# AGENTS.md

## Purpose

Sensori is research software for learning representations from 24-hour
tri-axial wrist accelerometry. Keep changes reproducible, scientifically
conservative and limited to the requested scope. Sensori outputs are not
validated for clinical decision-making.

Read `README.md`, `CONTRIBUTING.md`, `pyproject.toml` and the files relevant to
the task before editing. Inspect `git status` first and preserve unrelated work.

## Repository map

- `src/`: installed as the `sensori` package; contains model architecture,
  datasets, training and inference. Use `sensori.*`, not `src.*`, in imports.
- `config/`: Hydra model and pretraining configuration.
- `scripts/get_npy.py`: standalone public raw-recording parser.
- `scripts/tutorial_har/`: public HAR preparation and evaluation helpers.
- `scripts/tutorial_nhanes/`: public NHANES preparation helpers.
- `tutorials/`: executable source notebooks.
- `docs/`: Jekyll website published from `main/docs`.

## Scientific contracts

- Model-ready days are finite `float32` XYZ acceleration in units of g at
  10 Hz, with shape `(2880, 300, 3)`.
- The public parser applies gravity calibration, 5 Hz low-pass filtering,
  10 Hz resampling, non-wear detection, complete finite calendar days, at
  least 22 hours of wear, fewer than 10 interruptions and mean ENMO no greater
  than 200 mg.
- Exit code `3` from `scripts/get_npy.py` means processing completed but no day
  passed quality control.
- Do not change scientific constants, axis order, units, shapes, output naming
  or checkpoint-loading behavior unless the task explicitly requires it.
- Keep self-supervised pretraining, frozen-representation evaluation and
  supervised fine-tuning conceptually distinct.
- Do not turn predictive results into causal, diagnostic, prognostic or
  clinical-readiness claims.
- Mocked readers test code paths; they do not establish support for a physical
  device or file format.

## Data and release safety

- Never commit participant-level data, identifiers, raw recordings, embeddings,
  model checkpoints, credentials or generated experiment outputs.
- Generate synthetic test fixtures at runtime instead of adding binary data.
- Do not invent final paper, DOI, website, Hugging Face or checkpoint details.
- `scripts/get_npy.py` is intentionally standalone. Keep its public CLI and
  strict preprocessing behavior stable.
- Do not re-execute, clear or rewrite notebook outputs unless the task is
  specifically about the notebooks. Avoid unrelated notebook metadata changes.
- Do not modify licence terms without explicit approval.

## Working conventions

- Use Python 3.13, as declared in `pyproject.toml`.
- Prefer small, focused changes over broad refactors.
- Use existing configuration and package interfaces rather than duplicating
  constants in scripts or documentation.
- Add focused tests under `tests/` when changing behavior. There is currently
  no tracked test suite, so do not claim that tests passed unless tests were
  actually collected and run.
- Preserve backward compatibility for released NPY files and checkpoints.
- Keep public documentation concise and aimed at technical researchers.
- Preserve source notebooks when creating or revising website tutorials.

## Validation

Run the checks relevant to the files changed and report exactly what ran.

For Python changes:

```bash
python -m ruff check .
python -m compileall -q src scripts
```

If tests exist, run:

```bash
python -m ruff check tests
python -m pytest -q
```

For packaging changes:

```bash
python -m build
```

For website changes, run from `docs/`:

```bash
/opt/homebrew/opt/ruby/bin/bundle exec jekyll build
```

On systems where a current Ruby is already active, `bundle exec jekyll build`
is sufficient. A successful build is not browser visual QA.

For all changes:

```bash
git diff --check
git status --short
```

Before completion, inspect the final diff, check repository-relative links and
separate non-fatal warnings from failures in the report.

## Review priorities

Prioritize scientific-contract regressions, data leakage, checkpoint
incompatibility, participant leakage across evaluation folds and unsupported
claims over formatting preferences. Leave style enforcement to Ruff and other
automated checks where possible.
