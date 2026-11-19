# winding-field (core)

## Verification

- Before every commit you will push: run the `verify` skill (`python tools/verify.py`) and
  stamp the commit with `Verified: tools/verify.py PASS (baseline <sha7>)` or
  `Verified: none (<reason>)`.
- Check the REVIEW.md items the `review` step lists; they come from past incidents.
- A failing check gets a root cause. Never skip, xfail, quarantine, add sleeps or retries,
  or raise timeouts to get green. If a flaky test has no root cause, delete it and write
  down why in the commit message.
- Three rounds on the same failure without progress: stop and bring in the user.
- Fix the code, not the check; intended differences go in `tools/verify-approved.json`
  with a reason.

## Repository rules

- CI runs `ruff format --check .`, `ruff check .` and `mypy`; keep them green.
- Modules are flat `py-modules` listed in `pyproject.toml`; a new module is added there.
- The CLI (`LimeGS/winding-field-cli`) pins this repository by revision in
  `provenance/core.json`; public API changes need its verifier before the pin moves.
- Do not merge pull requests unless the user asks explicitly.
- After a history rewrite on GitHub, re-clone or reset to `origin/main`; never push from
  an old clone.
