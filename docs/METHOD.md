# Method

We approach winding counting as the recovery of a continuous layer coordinate,
`W(x)`, over the scroll volume. Within a locally consistent region, this
coordinate should vary little when moving along a papyrus layer and increase by
approximately one when crossing to the next. Surfaces of constant `W` are then
candidates for following the paper, while differences in `W` estimate how many
layer spacings separate two locations. The absolute zero is a reference choice;
the useful information lies in relative ordering and calibrated spacing, rather
than in assigning an integer label to every voxel at the outset.

Surface-orientation predictions suggest the direction across layers, and a
layer-density estimate supplies the expected rate of change in that direction.
Traced fibres provide complementary evidence: samples believed to lie on the
same local layer should have similar coordinate values. These observations are
noisy and need not form a mutually consistent gradient field, so integrating
them independently along different paths can accumulate drift. Our approach
instead estimates a field jointly, balancing agreement with the local gradient
observations against variation along fibres. A shared coarse reference carries
a common origin and scale into finer regions, helping their local refinements
remain mutually consistent.

The open problem is making that coherence preserve the physical winding count.
A constant field has no within-layer drift, and a compressed field can improve
that metric while losing separation between neighbouring layers. Reliable
counting therefore also requires justified orientation and calibration,
validation of between-layer separation, and explicit treatment of missing
evidence. Folds, breaks and the topology of a rolled sheet can require cuts or
linked local coordinate patches rather than one globally smooth numbering.
We assess candidates through coverage, intrafibre drift, separation controls
and matched CT overlays; smooth contours or numerical convergence alone do not
establish that the inferred layers are physically correct.

## Glossary

| Term | Meaning |
|---|---|
| Winding field `W(x)` | A continuous layer coordinate over the scroll volume: nearly constant along a papyrus sheet, about +1 per sheet crossed. |
| Sheet / layer | One wrap of the rolled papyrus. |
| AFV | A collection of traced fibre polylines used as within-sheet evidence. |
| Drift | Change of `W` along one fibre; low drift means the field follows the sheet. |
| Separation | `W` difference across neighbouring sheets; it must stay near one per sheet. |
| Coarse reference / fine band | A low-resolution global field and the higher-resolution local refinements that inherit its origin and scale. |
| B1, B2, B3, B3-R, B4 | Implemented global construction variants of `tools/run_coherence.py`; B2 is the default. |
| Frozen case | A saved, hash-bound evaluation input (fibre pairs, controls, frame) that candidates are scored against. |
