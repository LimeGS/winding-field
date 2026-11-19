---
name: verify
description: Verify a winding-field (core solver) change end to end before committing or opening a PR. Compares the public API and solver outputs with the base branch as the oracle, runs the README examples, checks that a checkpoint written by the previous version resumes to identical bytes, and lists the REVIEW.md items the diff triggers. Use it for any change under src/, tools/ or examples/, before every commit you will push.
---

# Verify

From the repository root, with the dev environment installed (`pip install -e ".[test,dev]"`):

```bash
python tools/verify.py
```

Default baseline is `origin/main` (fetch it first); `--baseline <ref>` for another oracle,
`--steps a,b` for a subset, `--out DIR` to keep work files. About 30 seconds. Receipt:
`.verify/VERIFY.json`.

## Steps

| Step | Proves | Typical failure |
|---|---|---|
| `review` | Lists REVIEW.md items triggered by the diff (from the baseline's REVIEW.md) | — informational: check each listed item |
| `api` | Every public function/class/parameter/default and `run_coherence` option of the baseline still exists unchanged | removed parameter (as `nonconvergence_policy` in 3556a4b), changed default |
| `golden` | `solve_field` (ramp; all terms at default weights), block solves with and without caches/read workers, and the synthetic B2 build are byte-identical to the baseline | numerics changed by a refactor or optimization |
| `e2e` | `examples/minimal_field.py`, `build_demo.py`, `benchmark_demo.py` run as in the README | broken public path |
| `cross-version` | A block solve checkpointed mid-sweep by the baseline resumes with this version to the same bytes as an uninterrupted run, or is refused unchanged | checkpoint format or identity change |

## When a step fails

1. Read the evidence in `.verify/VERIFY.json`.
2. Fix the code, not the check. If the difference is intended, add it to
   `tools/verify-approved.json` with a reason and say so in the PR body (REVIEW.md R9).
3. Three rounds on the same failure without progress: stop and ask the user, with the
   evidence.

## Before committing

- `ruff format` and `ruff check` on touched files, `mypy`, the related tests, then
  `python tools/verify.py`.
- Stamp the commit message: `Verified: tools/verify.py PASS (baseline <sha7>)` or
  `Verified: none (<reason>)`.
- A public API change reaches the CLI only through its pinned core revision: run the CLI's
  verifier with this core installed before proposing the pin bump (REVIEW.md R5).
