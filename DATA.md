# Data and execution artifacts

Keep real CT, AFV/SQLite fibre databases, model weights, coordinate annotations,
generated field arrays, CT pictures and evaluation output directories outside
this checkout, except for the README illustrations documented below. The
repository otherwise distributes software and synthetic test fixtures.

Use [DATASET.md](DATASET.md) to configure your own volume, predictions, AFV,
coordinates and calibration. Dataset names in the README are measured examples,
not mandatory inputs or runtime defaults.

The benchmark consumes a case descriptor that identifies external inputs by
path and SHA-256. Preserve its frozen population, calibration and original
receipts. Store new output directories and their manifests with the data bundle;
do not overwrite a previous run or regenerate its hashes to hide a mismatch.

Research history, exploratory reports and prior release snapshots are maintained
outside the public source tree. They are not installation dependencies.

## Public AFV sources

The current README measurement uses **Bruniss's AFV v2** Paris4 fibre collection.
Earlier experiments also used **Qual's AFV v1**. Credit for generating and
sharing these AFVs belongs to **Bruniss (`bruniss`)** and **Qual (`qualzz`)**,
with data hosting provided by the [Vesuvius Challenge](https://scrollprize.org/data).
The labels **v1** and **v2** identify these data collections, not releases of
this software.

| Fibre collection | Contributor | Public download | Original release post |
|---|---|---|---|
| AFV v1 | Qual (`qualzz`) | [`Paris4_v1.afv`](https://dl.ash2txt.org/community-uploads/qual/AutomatedFiberVolumes/Paris4_v1.afv) | [2026-10-05 — Automated Fiber Volume](https://discord.com/channels/1079907749569237093/1552131970136547399/1556740591377846382) |
| AFV v2 | Bruniss (`bruniss`) | [`fibers.afv`](https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/fibers.afv) | [2026-10-13 — neural tracing](https://discord.com/channels/1079907749569237093/1559046849489408000/1559566709294104668) |

Qual [describes producing v1](https://discord.com/channels/1079907749569237093/1552131970136547399/1556460574840455320)
from the publicly released Paris4 fibre predictions using Qual's spline generator
and post-processing. Bruniss released v2 in the neural-tracing thread, with
coverage from z7000 to z17000 at 9.6 µm. These collections also build on the
community's CT data, fibre annotations and prediction models.

Frozen files used in the current and earlier experiments:

- **AFV v1:** 412,086,272 bytes; HTTP Last-Modified **2026-10-05 18:14:53 UTC**.
  SHA-256: `05fa4606d6403d5e3b417afd1d6c1def17cb4c42b598f6b18b17fe58ce347504`.
- **AFV v2:** 11,127,046,144 bytes; HTTP Last-Modified **2026-10-13 03:43:29 UTC**.
  SHA-256: `1898ff5098f03a36e02c76efc5f6b3774b766b154bfc8faadca488459114c941`.

The README table evaluates a saved 19.2 µm fine-band checkpoint using the frozen
v2 pair selection. Its selected and covered populations remain explicit; it is
a development measurement from a partial checkpoint. Public URLs can be updated,
so match the recorded hashes to reproduce a measurement. Both collections retain
their upstream [terms](#external-permissions-and-attribution).

<a id="readme-illustration"></a>

## README illustrations

[`assets/paris4-six-surfaces.png`](assets/paris4-six-surfaces.png) is the supplied
9600×5400 six-surface panel, copied unchanged from
`Paris4-six-surfaces-3x6-English-continuous-only-9600x5400.png`. It shows original
geometry, CT-textured surfaces and available continuous flattened CT diagnostics
for W = −120, −100, −80, −60, −40 and −20. The W = −120 continuous flattened
surface is unavailable; masked and unmapped areas remain blank. The panel is
marked **UNACCEPTED**, with physical sheet identity unverified. It credits
Bruniss for AFV v2 and the Vesuvius Challenge for the Paris4 CT data. Its visual
diagnostics are separate from the latest-checkpoint fibre table in the README.

Image SHA-256: `d7946a44288c0aee055f6d3ad413bd40c7d28de3e41d97e19713df7874636e45`.
The underlying data terms apply; the MIT code license does not relicense this
CT-derived illustration.

[`assets/ct-winding.png`](assets/ct-winding.png) is the original CT-derived
illustration, rendered by the winding benchmark from Paris4 CT supplied by the
[Vesuvius Challenge](https://scrollprize.org/data), scan `20260411134726`, L2.
It compares the original CT with the historical global B2 and local fine-pilot
winding contours using matched views and fixed contour levels. It is visual QC,
not a certificate of physical sheet identity. The image retains the underlying data
terms; the MIT code license does not grant redistribution rights to the CT.

Image SHA-256: `0b6fdbc173b228e19207ac57a4562a7a27e10a46a7614222848e36fd39ca46e4`.
The original inputs and numeric/rendering receipts remain in the external data
workspace. This illustration is not a bundled evaluation case or test input.

## External permissions and attribution

The MIT code license does not relicense datasets, annotations, model weights,
native executables or third-party implementations. Consult the applicable
[Vesuvius Challenge data terms](https://dl.ash2txt.org/LICENSE.txt) and the license
of each supplied asset. Public accessibility is not redistribution permission;
terms for one scan must not be generalized to a different scan.

Dataset/model references in the code include Vesuvius Challenge / ScrollPrize CT
and surface predictions, contributor-provided fibre collections and annotations,
and optional VC3D/Flatboi native tooling. These external assets remain under
their respective terms. Apart from the README illustrations, they are not bundled
here.
