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

2. The official `copy_displacement_latest` tracer, asked to step one wrap
   out and in in an ultra-compressed window (~100 µm spacing, CT-verified),
   predicts canonical-pitch hops (~330-500 µm) and lands 2-4 wraps away.
   That is 94-98% wrong-hop at reduced spec; at native crop + TTA the
   wrong-wrap rate drops to 0% but 56-63% of points still miss the half-gap
   threshold. Our pre-registered <30% gate was not met.

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
| Null baseline (constant offset, reduced config) | 100% | 100% | n/a |

Where 63.2% comes from: 10,000 front points, 5,303 excluded because the
neighbor fell outside the reference band, 4,697 gradable. None landed
nearer a wrong wrap (the 0% column). The local gap is about 11 voxels
(~88 µm) and we require landing within half of it; 1,730 did (36.8%), the
other 2,967 landed on the right wrap but too far. That is the 63.2%.

At native config the model predicts hops of ~330-500 µm (canonical pitch)
where the real spacing is ~100 µm: right side 94% of the time, wrong
distance almost always. Our reading is that it does not pick up the local
compression and falls back on its training pitch. The raw distances are in
the repo so you can re-threshold; at full-gap the numbers look better. We
pre-registered a <30% gate, and 63.2/56.3% misses it, so as tested this is
a benchmark miss rather than a deployable tracer. Rerun cost: $0.23
(RTX 3090, 155.6 s).

### What we did and did not test

Resolution is the open axis: 7.91 µm against 4.8 µm training biases against
the model, and a native-resolution test needs a resampling adapter nobody
has built, so "fails at native resolution" is not established. It is one
blind-chosen window on one segment. The reference is community
segmentation validated locally, not human-annotated truth. Since we can
only grade where the segment mapped the neighbor, the gradable region
likely skews toward easier geometry, which for a negative result makes our
number conservative rather than inflated.

Benchmark spec, code and logs: `docs/BENCHMARK.md` and `docs/NT_AUDIT.md`
in the neural-tracing-audit repo.

## What seems worth trying next

- A locally estimated pitch (for example from the intensity-peak spacing we
  used for the CT check) instead of a global constant, so a quarter-turn
  threshold behaves consistently where sheets compress.
- Non-radius correspondence signals on the same connected graph: surface
  normals, tangent direction, mesh-derived orientation, or bridge cuts,
  scored against the same mesh-continuity audit.
- Native-resolution neural inference, once someone adapts the 3.24 µm stack
  into the checkpoint's frame.

Happy to run the audit harness against meshes of the same region from any
pipeline (VC3D, ThaumatoAnakalyptor, ScrollFiesta, the neural tracer, or
anyone's segmentation). It only needs a mesh or a set of predicted points.
