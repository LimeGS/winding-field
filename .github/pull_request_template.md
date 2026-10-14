## What changed

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The problem or decision this honours. -->

## Verification

<!-- The exact command(s) you ran, and the result. -->

```text
python -m pytest -q
ruff format --check . && ruff check . && mypy
```

## Checklist

- [ ] Tests added/updated for non-trivial logic
- [ ] Documentation updated alongside behavior changes
- [ ] No real scan data beyond the documented README illustration; no labels, coordinates, AFVs or weights
- [ ] No experiment history, run outputs or prior-release snapshots added
- [ ] Evaluation inputs are hash-bound; run receipts stay in the external data workspace
