# Data and execution artifacts

Keep real CT, AFV/SQLite fibre databases, model weights, coordinate annotations,
generated field arrays, CT pictures and evaluation output directories outside
this checkout, except for the README illustration documented below. The
repository otherwise distributes software and synthetic test fixtures.

The benchmark consumes a case descriptor that identifies external inputs by
path and SHA-256. Preserve its frozen population, calibration and original
receipts. Store new output directories and their manifests with the data bundle;
do not overwrite a previous run or regenerate its hashes to hide a mismatch.

Research history, exploratory reports and prior release snapshots are maintained
outside the public source tree. They are not installation dependencies.

## README illustration

[`assets/ct-winding.png`](assets/ct-winding.png) is the single included CT-derived
illustration, rendered by the winding benchmark from Paris4 CT supplied by the
[Vesuvius Challenge](https://scrollprize.org/data), scan `20260411134726`, L2.
It compares the original CT with the global B2 and fine-candidate winding
contours using matched views and fixed contour levels. It is visual QC, not a
certificate of physical sheet identity. The image retains the underlying data
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
their respective terms. Apart from the README illustration, they are not bundled
here.
