# Reproduction evidence

This file connects every corrected result to an executable command, a raw
log, and (for large artifacts) a SHA-256 checksum. Paths are relative to the
repository root unless noted otherwise.

## Release artifacts

Publish these five NPZ files together as release assets. Their hashes are in
`docs/CHECKSUMS.txt`; `artifacts/axis.npy` is small enough to ship in git.

| Artifact | Role |
|---|---|
| `wfield_multi.npz` | Input wrap radii used by all four experiments |
| `wfield_final_signfix_m1.npz` | Corrected confidence-MST field |
| `wfield_final_signfix_m2.npz` | Corrected weighted-LS field |
| `wfield_final_signfix_m3.npz` | Corrected triangle-filtered field |
| `wfield_final_signfix_m4.npz` | Corrected tight-tolerance field |

The corrected outputs intentionally contain `w`, `r`, and `zs` only.
`audit_segment.py` supports that schema directly. If a `labels` array is
present, as in a `build_field.py` output, the audit additionally restricts
candidates to the largest component.

## Solver reruns

Place `wfield_multi.npz` at repository root, then run:

```bash
python method1_confidence_mst_corrected.py
python method2_weighted_least_squares_corrected.py
python method3_triangle_confirmed_corrected.py
python method4_tight_tolerance_corrected.py
```

The exact logs are in `docs/evidence/method{1,2,3,4}_corrected.log`.
They report corrected GATE3 disagreement of 45.8%, 44.2%, 44.7%, and
43.8%, respectively. `docs/evidence/synthetic_sign_selftest.log` records the
6/6 known-truth sign assertions; regenerate it with:

```bash
python sign_convention_selftest.py
```

## Segment audit

The primary segment is `20240716140050`; the independent replication is
`20240618142020`. Both are public PHerc 0332 PPMs. The exact transform is:

```text
https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr/transform.json
```

Reproduce the primary Method 1 audit with:

```bash
python audit_segment.py \
  --field wfield_final_signfix_m1.npz \
  --axis artifacts/axis.npy \
  --ppm-url https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240716140050/20240716140050.ppm \
  --transform-url https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr/transform.json
```

This produces `6912/27665 = 24.985%` transition jumps. Adding `--shuffle`
produces `24960/27665 = 90.222%`, confirming that the Viterbi continuity
prior is not imposing the real field's lower jump rate. Compact outputs for
Methods 1-4, segment 2, and the shuffle control are preserved as
`docs/evidence/*_public_*.log`. Reproduce Methods 2-4 by changing only the
`--field` filename. Reproduce segment 2 by changing the Method 1 PPM URL to:

```text
https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths/20240618142020/20240618142020.ppm
```

The original as-run logs and all corrected field audits are also preserved
under `docs/evidence/`. Those older logs printed `n` path points and used it
as the denominator; the repaired public script uses the mathematically
correct number of transition opportunities, `n - 1`. This changes no
one-decimal conclusion. Corrected aggregate transition rates are:

| Field | Jumps / transitions | Rate |
|---|---:|---:|
| Method 1 | 6,912 / 27,665 | 24.985% |
| Method 2 | 5,955 / 27,665 | 21.525% |
| Method 3 | 6,825 / 25,622 | 26.637% |
| Method 4 | 7,369 / 27,665 | 26.637% |
| Method 1, segment 2 | 7,061 / 29,475 | 23.956% |

## Graph and census

With `wfield_multi.npz` at repository root:

```bash
python figures/make_field_figure.py
python -c "import numpy as np; from validate_field import merged_sheet_exposure; print(merged_sheet_exposure(np.load('wfield_multi.npz')['r']))"
```

The graph recomputation reports 67,885,430 wraps, 126,878,516 candidate
edges, 99.7527% of wraps participating in at least one edge, a 96.6671%
largest component, and 96.7406% median per-z largest-component coverage.
The census reports 510,796 double-gap signatures among 4,152,092 sampled
gaps (12.3021%). This is a heuristic anomaly signature, not a bound on the
true merged-sheet rate.

An earlier exploratory note claimed `0/75,888` redundant-edge conflicts,
but its generating script and log were not preserved. That number is
therefore deliberately excluded from the release's validated claims.

## Known refinement not shipped in this rev

A stricter auditor variant that splits Viterbi runs at missing-PPM-geometry
gaps (instead of bridging them) was measured against this exact evidence:
it removes 21 of 27,665 transitions (coverage 99.931%) and shifts Method 1
from 24.985% to 24.855% real and 90.222% to 90.200% shuffle — conclusions
unchanged. It is preserved as a patch outside the repo rather than shipped
out of sync with the checksummed logs above; adopting it requires
regenerating all five audit logs, CHECKSUMS, and the doc numbers together.
Two bookkeeping caveats measured for this segment but worth carrying to any
future rev: rows skipped by --min-points fall out of both numerator and
coverage denominator, and a bridged candidate-only hole of k cells counts
as 1 scored transition against k+1 potential pairs (conservative here).
