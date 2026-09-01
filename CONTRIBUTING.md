# Contributing to Sensori

Thank you for contributing to Sensori. Contributions should keep the software
reproducible, preserve its scientific contracts and remain accessible to
technical researchers.

Before starting, read [`AGENTS.md`](AGENTS.md) for the repository map,
scientific invariants and validation expectations. Open an issue before making
a change that would alter the model architecture, preprocessing contract,
checkpoint compatibility, licence or public data interface.

## Development setup

Sensori requires Python 3.13. From the repository root:

```bash
conda create -n sensori python=3.13 pip
conda activate sensori
python -m pip install -e ".[dev]"
```

To work with raw CWA, GT3X or BIN recordings, also install ActiPy and ensure
Java 8 or newer is available:

```bash
python -m pip install actipy==3.4.0
```

Do not use real participant data as a development fixture. Tests should create
small synthetic inputs in temporary directories.

## Making changes

- Keep pull requests focused on one logical change.
- Preserve existing public interfaces unless a breaking change is explicitly
  proposed and documented.
- Add focused tests under `tests/` for new behavior or bug fixes.
- Update the README or tutorials when a public command or output changes.
- Do not commit datasets, participant identifiers, embeddings, checkpoints,
  credentials or generated experiment outputs.
- Do not re-execute notebooks merely to update documentation. Notebook changes
  should be intentional and should avoid unrelated output or metadata churn.

The raw-to-NPY preprocessing constants are part of the released model contract.
Changes to sampling rate, filtering, units, quality control, array shape or
output naming require explicit scientific review.

## Validation

Run the checks relevant to your change. Install the `dev` extra before running
the Python checks.

```bash
python -m ruff check src scripts
python -m compileall -q src scripts
python -m build
git diff --check
```

Once tests are present, run:

```bash
python -m ruff check tests
python -m pytest -q
```

For website changes, build the site from `docs/`:

```bash
/opt/homebrew/opt/ruby/bin/bundle exec jekyll build
```

On systems with a current Ruby already active, use
`bundle exec jekyll build`. Inspect relevant generated pages when layout or
visual content changes; a successful build alone is not visual verification.

## Pull requests

Describe:

- what changed and why;
- any effect on inputs, outputs, checkpoints or scientific interpretation;
- the exact validation commands run and their results; and
- any checks that could not be run.

Before requesting review, inspect the complete diff and confirm that no data,
credentials, local paths or generated artifacts are included.
