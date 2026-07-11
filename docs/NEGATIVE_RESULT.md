# Finding: the tested radius-proximity winding assignment remains inconsistent after a major correction

**Status, 2026-07-10:** corrected negative result for one measurement and
preprocessing pipeline. Three solver architectures and one tolerance
sensitivity test produce 43.8-45.8% GATE3 same-sheet disagreement.
Aggregate segment-audit jump rates are 21.525-26.637%; an independent public
segment gives 23.956%.

This is materially better than the first report. A shared sign-convention
bug had inflated GATE3 to 68.5-70.5% and the segment audits to roughly
57-58%. The complete correction record is retained below.

## Problem and method

For each sampled `(z, theta)` ray through the public PHerc 0332 surface
prediction, radial ray casting records detected sheet crossings. Neighboring
rays are matched by radius within a nominal tolerance. A ray-pair vote
estimates the integer offset between its ordered crossing lists; a global
solver then attempts to synchronize those offsets into winding coordinate
`W`.

The experiments are:

1. Confidence-priority maximum spanning tree.
2. Global weighted least-squares relaxation.
3. Triangle-filtered maximum spanning tree.
4. Method 1 repeated with tolerance reduced from `0.45 * 17` to
   `0.25 * 17` field voxels.
5. Method 1 evaluated on a second public segment.

Items 1-3 are solver architectures. Item 4 is a sensitivity test, not a
fourth architecture. Item 5 is an external evaluation, not a solver.

## Sign correction

All original solver scripts measured
`m = mode(IP_destination - IP_origin)` and constructed `W = IP + offset`,
but propagated `offset_destination = offset_origin + m`. Requiring matched
same-sheet values to agree gives:

```text
offset_destination = offset_origin - m + seam_step
```

The error doubled every nonzero modal offset in the final field. It affected
GATE3 and all W-dependent segment audits, but not GATE1 vote purity, GATE2
loop closure, graph connectivity, or the anomaly census.

The correction was accepted only after three checks:

1. Algebraic derivation and a worked missing-wrap example.
2. A known-truth synthetic spiral: 6/6 assertions pass, corrected winding
   recovery is 100%, original recovery is 70.59%, and corrected GATE3 is
   0.000%.
3. Full reruns reproduce every sign-independent metric exactly and swap the
   chirality mirrors as predicted.

The original scripts remain in `method*.py`; corrected twins are
`method*_corrected.py`. Raw logs and hashes are under `docs/evidence/`.

## Corrected results

### Full-volume solver metrics

| Experiment | Original GATE3 | Corrected GATE3 | Corrected valid pairs |
|---|---:|---:|---:|
| Confidence tree | 70.5% | **45.8%** | 119,595,494 |
| Weighted-LS relaxation | 70.5% | **44.2%** | 119,736,145 |
| Triangle-filtered tree | 69.2% | **44.7%** | 105,378,498 |
| Tight nominal tolerance | 68.5% | **43.8%** | 105,105,464 |

GATE3 is the fraction of matched candidate pairs whose solved W values
violate the expected same-sheet relation by more than 0.5 after the seam
step. It is not `1 - GATE1 purity`.

### Segment audits

The old logs printed path points as the denominator even though jumps occur
between points. The repaired public audit uses the correct number of
transition opportunities, `points - 1`. This bookkeeping correction changes
no one-decimal conclusion.

| Field / segment | Original jumps | Corrected jumps |
|---|---:|---:|
| Method 1 / primary | 16,086 / 27,665 (58.15%) | **6,912 / 27,665 (24.985%)** |
| Method 2 / primary | 15,695 / 27,665 (56.73%) | **5,955 / 27,665 (21.525%)** |
| Method 3 / primary | 14,578 / 25,622 (56.90%) | **6,825 / 25,622 (26.6373%)** |
| Method 4 / primary | 15,976 / 27,665 (57.75%) | **7,369 / 27,665 (26.6365%)** |
| Method 1 / segment 2 | 17,031 / 29,475 (57.78%) | **7,061 / 29,475 (23.956%)** |

(Methods 3 and 4 agreeing to three decimals is an arithmetic coincidence of
two different fractions — 6,825/25,622 vs 7,369/27,665 — not a copy-paste
error; their per-row evidence logs are distinct, hence the fourth decimal in
the table. For historical scale: the originally *published* headline rates
before either correction were 68% / 65% / 67% / 69% on the primary segment
and 51.5% on segment 2 — those were single-last-row rates, not aggregates,
and are superseded by the aggregate table above.)

For every consecutive Viterbi path transition, the audit compares the W
increment with the continuous angular increment `delta_theta / 2pi` and
flags absolute deviation above 0.5. It also reports
`abs(abs(delta_W) - abs(number_of_loops))` as an endpoint residual. These
are continuity diagnostics derived from public segment geometry, not manual
winding labels.

The repaired public Method 1 command reproduces 24.985%. Its deterministic
per-ray shuffle control gives **90.222%**, demonstrating that the Viterbi
prior does not impose the real field's lower jump rate. We do not claim a
precise clean-field baseline because no independent known-correct full field
exists for this segment.

## Candidate graph result

`figures/make_field_figure.py` recomputes the graph from the raw radii using
the exact matching semantics of `build_field.py`:

| Measurement | Result |
|---|---:|
| Wraps | 67,885,430 |
| Candidate edges | 126,878,516 |
| Wraps with at least one edge | 99.7527% |
| Largest component | 96.6671% |
| Median per-z largest-component coverage | 96.7406% |
| p10 per-z coverage | 62.9291% |

This validates connectivity, not physical edge identity. A false edge can
increase connectivity, so the graph should be described as a candidate
adjacency scaffold rather than ground truth.

An earlier note claimed `0/75,888` redundant-edge conflicts. No generator or
log survives, so the release withdraws that claim rather than presenting it
as reproducible evidence.

The separate census finds 510,796 double-gap signatures among 4,152,092
sampled gaps (12.3021%). A double gap can indicate a missing detection,
merged sheets, or spacing variation. It is a QC proxy, not a bound on the
true error rate.

## Interpretation

The sign-corrected field is substantially better than first reported but is
still too inconsistent for automatic winding annotation. The following
observations are supported:

- Median GATE1 vote purity is 0.568; with tight tolerance it is 0.583.
- Unfiltered GATE2 ray-offset loop conflict is 52.3%.
- Triangle filtering reduces GATE2 conflict to 31.4%, but GATE3 and segment
  continuity do not materially improve.
- All three solver architectures land in the same broad GATE3 band.
- The independent segment reproduces the primary audit band.

These observations support a measurement-limited explanation for this exact
pipeline. They do not rule out better ordered-sequence matching, local axis
or warp estimation, finer sampling, subvoxel detections, robust integer
optimization, surface orientation, or learned correspondence signals.
Cross-method convergence alone is explicitly not treated as proof: the sign
bug was shared by every original implementation.

## Reproduction

`docs/EVIDENCE.md` lists exact commands, public URLs, release artifacts,
checksums, raw logs, and numerator/denominator arithmetic. The minimal local
verification is:

```bash
python -m unittest discover -s tests
python sign_convention_selftest.py
(cd docs/evidence && shasum -a 256 -c CHECKSUMS.txt)
```

With the release NPZs downloaded, run the four corrected scripts, the public
segment audit, and `figures/make_field_figure.py` as documented there.

## Bottom line

- **Delivered:** a reproducible candidate graph and census, corrected solver
  scripts, exact run evidence, a known-truth sign test, and an end-to-end
  public segment audit with shuffle control.
- **Not delivered:** a reliable global winding coordinate or proof that all
  radius-derived, higher-resolution, geometric, or learned approaches fail.
