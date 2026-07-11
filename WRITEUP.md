# How noisy is the winding coordinate in compressed regions? Two measurements on PHerc 0332

Show-and-tell write-up, July 2026. In the compressed regions of PHerc 0332
we looked at, winding-derived signals came out noisier than we expected.
Below is what we measured and how, offered as our observations from one
scroll rather than general claims.

Public data only (CC BY-NC 4.0, `dl.ash2txt.org` /
`s3://vesuvius-challenge-open-data`). Code, checksums, raw logs and release
artifacts (tag v0.1.0):

- https://github.com/LimeGS/winding-field
- https://github.com/LimeGS/neural-tracing-audit

## TL;DR

1. A wrap-adjacency graph built by radius matching is well connected
   (67.9M wraps, 99.75% with at least one edge, 96.67% in one component)
   and reusable. Assigning a consistent integer winding number on top of it
   did not work for us: 43.8-45.8% wrap-level disagreement across three
   solvers plus a tolerance test, and 21.5-26.6% jump rates against real
   segment meshes (a shuffled control gives 90.2%). We first posted
   ~65-70%; about 25 of those points were a sign bug in our own code,
   caught before release.

2. The official `copy_displacement_latest` tracer misses our
   pre-registered gate as released at 7.91 µm (56-63% of points past the
   half-gap threshold at native crop + TTA, 0% wrong-wrap), and we found
   the dominant cause: a unit mismatch, not a perception failure. The
   checkpoint is trained at 4.8 µm and the released CLI applies its
   displacement output raw in volume voxels, with no unit conversion and
   no documented resolution requirement (we read the wheel). Multiplying
   displacements by 4.8/7.91 = 0.607 and changing nothing else drops
   wrong-hop to 5.9/7.7%, passing the gate. Caveat that travels with it:
   at corrected magnitudes this window is matched by a no-network
   normal-step baseline, so it certifies the fix, not directional skill.
   Details and controls in the Part B correction box.

3. Both point the same way: winding signals degrade most where sheets
   compress. In our one window local pitch was ~5× tighter than canonical,
   so a winding coordinate built on a global pitch constant would drift by
   whole turns across a few sheets, and a quarter-turn threshold would mean
   different things in different places.

## Part A: global winding assignment by radius proximity

Borrowing the Relative Geologic Time idea from seismic work: cast rays out
from the scroll axis through the public surface predictions, detect each
sheet crossing ("wrap"), connect neighbors in angle and depth by radius,
and solve for one consistent integer winding number over the graph.

### The graph works

| Step | Result |
|---|---|
| Wrap detections | 67.9M |
| Graph edges | 126,878,516 |
| Participation | 99.7527% of wraps have at least one edge |
| Connectivity (union-find) | 96.6671% in one component; median per-depth 96.7406% |
| Double-gap census | 510,796 / 4,152,092 gaps (12.3021%; a QC heuristic) |

Connectivity does not prove that each edge joins the same physical sheet,
so we call it a candidate graph. It is still useful scaffolding. (An
earlier note claimed 0/75,888 redundant-edge conflicts, but we could not
reproduce its generator, so we dropped that claim.)

### The integer assignment does not

Turning the graph into a coordinate means picking, per edge, how many whole
wraps apart its two ends are, consistently around every loop. Three solver
families, a tolerance test, and a second segment:

| # | Experiment | Wrap-level disagreement (orig -> corrected) | Aggregate audit jumps (orig -> corrected) |
|---|---|---|---|
| 1 | Max-confidence spanning tree | 70.5% -> **45.8%** | 58.15% -> **6,912/27,665 (24.985%)** |
| 2 | Global weighted least-squares | 70.5% -> **44.2%** | 56.73% -> **5,955/27,665 (21.525%)** |
| 3 | Triangle-confirmed spanning tree | 69.2% -> **44.7%** | 56.90% -> **6,825/25,622 (26.637%)** |
| 4 | Method 1 at a tighter matching window | 68.5% -> **43.8%** | 57.75% -> **7,369/27,665 (26.637%)** |
| 5 | Method 1's field on a second public segment | n/a | 57.78% -> **7,061/29,475 (23.956%)** |

### Correction (2026-07-10)

We first posted ~65-70% here. Two things were wrong, both caught in a
pre-release audit. A sign bug: the scripts propagated offsets as
`off[v] = off[u] + m` but built the coordinate as `W = IP + off`, which
needs `off[v] = off[u] - m`; the `+m` made every nonzero-offset edge
self-inconsistent by `2m`. And the mesh-audit figures we had quoted were
single-row rates, not aggregates (the originals aggregate to 56.7-58.1%).
Corrected: 43.8-45.8% disagreement, 21.5-26.6% aggregate jumps, 23.96% on
the second segment.

We checked before changing anything. A synthetic spiral with known truth
recovers exactly under the corrected sign and reproduces the predicted
errors under the old one; the two metrics that cannot depend on the sign
(vote purity ~57%, loop-closure ~52%) came back identical; and both
original and corrected runs ship as checksummed logs. Having just caught
one bug in our own stack, we would not be shocked if there is another.

### How we call it "wrong" without ground truth

Two internal checks, no human labels. Wrap-level disagreement is a
consistency count on the solved field. The mesh audit reads the winding
number along a real segment mesh, streamed from its PPM: along one
continuous sheet it must advance smoothly, and each jump is a failure. The
control that matters: shuffling the field per ray raises Method 1 from
24.985% to 90.222%, so the low real-field rate is not something the readout
imposes.

### Why we read this as noise, not solver choice

Three different solver architectures land in the same band, and halving the
matching window barely moved it (vote purity 0.568 to 0.583). Triangle
filtering fixed the coarse offset problem but not the fine wrap-to-wrap
match the audit depends on. The likely cause is quantified: wrap spacing
here is about 17 quarter-voxels and the detection noise is roughly half
that, so the corrected field carries real signal (21.5-26.6% vs 90.2%
shuffled) but sits two orders of magnitude above the ~0.2% a usable
coordinate needs. One honest note: we made this same "the methods converge"
argument at ~70% before finding the shared sign bug, so what backs it now
is the known-truth test and the identical sign-invariant metrics, not the
convergence alone.

Scripts (original and corrected), the synthetic self-test, and all logs:
`docs/NEGATIVE_RESULT.md` and `docs/EVIDENCE.md` in the winding-field repo.

## Part B: the official neural tracer in a compressed window

Model: `scrollprize/copy_displacement_latest` (public, trained at 4.8 µm).
Given a partial surface and a volume, it predicts the next wrap's surface
in front of and behind it. Volume: the public 7.91 µm Scroll 3 zarr, which
is 1.65× coarser than training (no finer chunk-addressable volume exists
for PHerc 0332 yet, and that is the one confound we could not remove).
Seed: a 200×200 window of a public segment, built from PPM rows 1145-1344.

### The reference: local geometry, not ground truth

This is the direct answer to "how do you know it is wrong." We do not score
against our Part A winding field (that one failed). We score against the
seed segment's own recorded geometry: it spirals about three revolutions
through the band, so at each point the segment itself records where the
neighboring wraps sit. We validated that reference four ways before
trusting it: a self-test that reconstructs the seed's own wrap at 0.000 vox
(three earlier metric versions failed this and were dropped), wrong-side
scoring that fails 99%, a dumb-geometry null at 100% wrong, and a raw-CT
spacing check that uses no segmentation at all and independently confirms
~103 µm sheet spacing. It only grades where the segment recorded the
neighbor, so over half our test points are excluded rather than guessed.
Full derivation in `docs/NT_AUDIT.md` in the neural-tracing-audit repo.

### Results

Hop-correct means the predicted point lands on the expected adjacent wrap
and within half the local gap. n = 10,000 per direction; parenthetical
counts are scored / excluded.

| Configuration | Front wrong-hop | Back wrong-hop | Wrong-wrap |
|---|---|---|---|
| Reduced spec (7.91 µm, 128³ crops, no TTA, Apple M2 Pro) | 94.1% (4,613 / 5,387) | 97.7% (8,724 / 1,276) | 4.5% / 7.6% |
| Native crop (128×384×384) + TTA, RTX 3090, still 7.91 µm | **63.2%** (4,697 / 5,303) | **56.3%** (8,697 / 1,303) | **0%** |
| Native + displacement × 4.8/7.91 (resolution calibration, see box below) | **5.9%** (4,731 / 5,269) | **7.7%** (8,712 / 1,288) | 0% / 4.0% |
| Null baseline (constant offset, uncalibrated magnitude) | 100% | 100% | n/a |

Where 63.2% comes from: 10,000 front points, 5,303 excluded because the
neighbor fell outside the reference band, 4,697 gradable. None landed
nearer a wrong wrap (the 0% column). The local gap is about 11 voxels
(~88 µm) and we require landing within half of it; 1,730 did (36.8%), the
other 2,967 landed on the right wrap but too far. That is the 63.2%.

We pre-registered a <30% gate before spending on the GPU rerun. The tool
as released misses it (63.2/56.3%); the unit-corrected variant passes it
(5.9/7.7%). Rerun cost: $0.23 (RTX 3090, 155.6 s).

### The resolution-calibration finding (2026-07-11)

After first posting the native numbers we found their dominant cause. The
checkpoint is trained on 4.8 µm voxels. Reading the released CLI source
(the exact `vesuvius==0.2.4` wheel we ran): the model's displacement
output is applied directly in input-volume voxels, `--volume-scale`
selects the zarr pyramid level, and nothing converts units or documents a
resolution requirement. Training targets are voxel displacements of
4.8 µm data, so on a 7.91 µm volume every predicted hop reads 1.65× too
long.

Multiplying the displacements by 4.8/7.91 = 0.607 and changing nothing
else gives the calibrated row above. Checks, in order: the unscaled rerun
reproduces the published row exactly first; the factor is physics, not a
fit (any value in 0.50-0.65 scores 90-96%); wrong-side scoring of the
calibrated predictions still fails 100%. The rereading of the failure
geometry is striking: the model's raw hops (median 21.4 vox front / 14.5
back) are almost exactly 1.65× the true local gaps (11.2 / 8.8 vox), and
in the model's own training units the back hop matches the true gap to
the decimal. The model was reading the local spacing better than the raw
numbers suggested; the scale was wrong, not the perception.

The caveat that keeps this honest: at corrected magnitudes this window
saturates. A permuted-direction null scores 93-97%, and a no-network
baseline (seed + 9 voxels along the local surface normal) scores
95.8/95.2%, matching the calibrated model (94.1/92.3%). So this window
certifies the unit mismatch and its one-line fix; it cannot rank the
calibrated model against dumb geometry. That ranking needs harder
terrain: multi-hop chaining, high curvature, strongly varying gaps. The
remaining calibrated misses concentrate exactly where the local gap
swings ~6→19 vox, so the residual sub-problem is local pitch estimation,
the same suggestion Part A makes for the winding gate.

For anyone using this checkpoint on volumes whose resolution differs from
training: scale the displacements by (training voxel pitch / volume voxel
pitch), or the hops come out systematically mis-sized. Evidence script and
saved log: `resolution_calibration.py` in the neural-tracing-audit repo.

### What we did and did not test

Resolution went from open confound to identified mechanism: the dominant
failure at 7.91 µm is the unit mismatch above, and the one-line scaling
fixes the benchmark number. True native-resolution inference is still
untested (a 4.8 µm test needs a resampling adapter for the raw 3.24 µm
TIFF stack that nobody has built). It is one blind-chosen window on one
segment, and per the saturation caveat this window cannot rank the
calibrated model against simple geometric baselines. The reference is
community segmentation validated locally, not human-annotated truth. Since
we can only grade where the segment mapped the neighbor, the gradable
region likely skews toward easier geometry, which for the as-released
negative makes that number conservative rather than inflated.

Benchmark spec, code and logs: `docs/BENCHMARK.md` and `docs/NT_AUDIT.md`
in the neural-tracing-audit repo.

## What seems worth trying next

- A locally estimated pitch (for example from the intensity-peak spacing we
  used for the CT check) instead of a global constant, so a quarter-turn
  threshold behaves consistently where sheets compress.
- Non-radius correspondence signals on the same connected graph: surface
  normals, tangent direction, mesh-derived orientation, or bridge cuts,
  scored against the same mesh-continuity audit.
- A harder tracer benchmark that can rank direction quality once magnitude
  is calibrated: multi-hop chaining (errors compound), high-curvature
  windows, strongly varying gaps. This window saturates there, so it
  cannot tell a calibrated neural tracer from a normal-step baseline.
- Native-resolution neural inference, once someone adapts the 3.24 µm stack
  into the checkpoint's frame; and, upstream, making the CLI scale
  displacements by the training/volume voxel-pitch ratio (or documenting
  the requirement).

Happy to run the audit harness against meshes of the same region from any
pipeline (VC3D, ThaumatoAnakalyptor, ScrollFiesta, the neural tracer, or
anyone's segmentation). It only needs a mesh or a set of predicted points.
