# Review checklist

Read by reviewers (human or Claude) and by `python tools/verify.py` (`review` step), which
lists the items whose trigger matches the diff. CI reads this file from the **base
branch**, so a change cannot edit the rules that judge it.

Each item has a trigger readable from the diff, the check, and the incident it comes from.
An item triggers when a changed file matches **Paths** and, when a **Diff regex** is given,
one of that file's added or removed lines matches it. Add an item only when a miss happened
in at least two changes or caused an incident, and it is detectable from a diff. Prune items
that have not matched in a year.

| ID | Paths | Diff regex | Check | Origin |
|---|---|---|---|---|
| R1 | `src/*.py`, `tools/run_coherence.py` | `^-\s*(def \|class \|\w+\s*:.*=\|p\.add_argument)` | A public function, class, parameter, default or runner option is removed or changed only with an entry in `tools/verify-approved.json` (reason + migration); `api` step passes | 3556a4b dropped `nonconvergence_policy` (restored in #14) |
| R2 | `src/field_solver.py`, `src/block_solver.py`, `src/constraint_templates.py`, `src/field_constraints.py`, `src/field_observations.py` | — | `golden` identical to the baseline or the change is approved with the expected effect stated | Solver numerics are the product |
| R3 | `src/field_store.py`, `src/block_solver.py` | `CHECKPOINT\|identity\|resume\|sha256\|implementation` | A checkpoint written by the previous release resumes to the same bytes or is refused without being modified; `cross-version` step passes | Resume pinning; long real runs (PHerc0483A) resume across upgrades |
| R4 | `src/block_solver.py`, `src/field_store.py`, `src/field_prefetch.py`, `src/constraint_templates.py` | `cache_bytes\|read_workers\|prefetch\|workers` | Caches, read workers and prefetch change speed only: `golden` keys `block.case` and `block.case.perf` stay equal | Performance work (#46 in the CLI) |
| R5 | `src/*.py` | `^[-+]\s*(def \|class )[a-z]` | The CLI consumes these APIs through the revision pinned in its `provenance/core.json`: before bumping that pin, run the CLI's `tools/verify.py` with this core installed | Core/CLI regressions surfaced only downstream |
| R6 | `pyproject.toml`, `MANIFEST.in`, `.github/workflows/*` | — | Every module in `py-modules` ships in the wheel; sdist keeps tools, examples and tests (CI `build` job) | Flat py-modules packaging |
| R7 | `README.md`, `examples/*` | — | README commands and printed outputs match what `e2e` runs | Examples double as installed-package smoke tests |
| R8 | `tests/*.py` | `^\+.*(pytest\.mark\.skip\|xfail\|time\.sleep\|importorskip\|timeout=)` | No skip, xfail, sleep, retry or larger timeout to get green: find the root cause, or delete the test and write down why | Verification policy |
| R9 | `REVIEW.md`, `tools/verify.py`, `tools/verify-approved.json`, `.claude/*`, `CLAUDE.md` | — | Changes to the verifier, its approvals or these rules carry a reason in the PR body; approvals are removed once merged | Changes must not weaken their own checks |

## Severity

- **Important**: a bug this change adds or makes worse.
- **Pre-existing**: already broken on the base branch; not the author's to fix here.
- **Nit**: style or minor; never blocks.
