# Contributing

This is research code: correctness and auditability matter more than feature
count. Keep scientific run records in the external research/data workspace.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[test,dev]"
ruff format --check . && ruff check . && mypy
python -m pytest -q --cov=src --cov-report=term-missing
```

Target interpreters: Python 3.11 and 3.12; see the README for verified platforms.
Tests must stay CPU-only, synthetic and independent of live services. Install
`".[all,dev]"` instead when also working with Zarr data. Surface and plotting
tools are maintained in [winding-field-cli](https://github.com/LimeGS/winding-field-cli).

## What a change should ship with

- **A test for non-trivial logic** (a branch, a loop, a parser, anything
  numeric with a boundary). Ordinary tests are deterministic and assert on
  behavior, not on incidental formatting.
- **Updated documentation alongside behavior changes** — API/CLI contracts,
  installation instructions and the benchmark guide belong with the code.
- **External run evidence**: preserve inputs, source identities and original
  output manifests in the data workspace. Record corrections there instead of
  rewriting an old run. Keep experiment logs, reports and snapshots out of the
  public checkout.
- **No real data**: no CT, labels, coordinates, AFV files, weights, or
  screenshots of scroll data in commits beyond the documented README
  illustrations. See [`DATA.md`](DATA.md).

## Style

One policy, enforced by `ruff` (see `pyproject.toml`):

```bash
ruff format .        # apply
ruff check .         # lint (E4/E7/E9/F/W/I/UP)
mypy                 # types; annotate new public interfaces and boundaries
```

Suppressions must be narrow and explained in a comment. Keep comments for
constraints and decisions the code cannot show, not for narrating the code.

CI enforces a 90% statement-coverage floor across all `src/` modules. Use the
same command locally to find untested behavior:

```bash
python -m pytest --cov=src --cov-report=term-missing
```

Cover real outputs, invalid inputs, source identity and cleanup/failure behavior.
Do not lower the floor or omit modules to make a change pass. Numerical coverage
is not evidence of physical correctness, and this gate does not measure branches.

Mypy checks annotated functions and `tests/test_solver_interface.py`; it does
not check every research-function body. Keep array shape/range validation at
runtime, and add typed caller cases when changing the supported interface.

## Packaging and compatibility

```bash
python -m build
```

The default build creates an sdist and builds a wheel from it. CI exercises the
synthetic example with `python -I` after installing each artifact, preventing
checkout imports from masking missing files. Run that example when verifying
a local installation too.

The version in `pyproject.toml` identifies the installable library; scientific
results also require a commit and the relevant evidence hashes. Breaking API
changes bump the major version, compatible features bump the minor version,
and fixes bump the patch version. Keep migration notes with the change. No
automated publication is configured.

## Submitting

- Keep commits focused; describe what changed and what it was supposed to
  honour (especially for anything touching measured results).
- Pull requests should state: what changed, how it was verified (the exact
  command), and any limitation or residual uncertainty.
- If a fix corrects a previously recorded result, describe the effect and
  preserve the old receipt alongside the correction in the external workspace.

## What this project does not do

No release cadence, no support SLA, no distributed package on PyPI. Treat the
default branch as the source of truth and cite a commit when referring to a
result.
