# Results: fine-grid measurement (AFV v2, 19.2 µm)

Measured on **October 20, 2026**, using the latest saved Paris4 fine-band
checkpoint and a frozen selection from
[Bruniss's `fibers.afv`](https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/fibers.afv),
snapshot **2026-10-13 03:43:29 UTC**. See
[source credits and release posts](../DATA.md#public-afv-sources).
**No real AFV is bundled or selected by default.**

Drift is `abs(phi(a)-phi(b))/1.192497988541921`, in nominal sheet units;
distance is along-fibre arclength. Each fibre contributes at most one pair per
distance bin and may appear in several bins. A pair is covered only when all
eight interpolation corners at both endpoints contain finite stored fine-grid
values, with a one-node domain margin. Unavailable fine samples are not filled
from the coarse reference.

**792,595 / 2,154,184 selected pairs covered (36.8%).** Values are rounded to
three decimals.

| Along-fibre span (mm) | Covered / selected pairs | Median | p90 | p99 |
|---|---:|---:|---:|---:|
| 0.5–1.5 | 351,754 / 913,491 | 0.155 | 0.538 | 1.291 |
| 1.5–2.5 | 134,771 / 367,547 | 0.294 | 0.897 | 2.326 |
| 2.5–3.5 | 86,226 / 237,948 | 0.384 | 1.139 | 2.929 |
| 3.5–5 | 65,112 / 182,354 | 0.452 | 1.328 | 3.405 |
| 5–7.5 | 48,300 / 137,923 | 0.536 | 1.564 | 3.830 |
| 7.5–10 | 34,911 / 100,837 | 0.626 | 1.827 | 4.382 |
| 10–15 | 27,019 / 79,017 | 0.732 | 2.066 | 4.538 |
| 15–20 | 17,955 / 53,270 | 0.879 | 2.432 | 5.383 |
| 20–25 | 12,404 / 37,796 | 1.041 | 2.737 | 5.404 |
| 25–30 | 8,384 / 25,978 | 1.184 | 3.016 | 5.723 |
| 30–45 | 5,759 / 18,023 | 1.336 | 3.343 | 6.109 |

**Interpretation:** drift grows with along-fibre distance, and substantial
long-span errors remain. Statistics describe the covered pairs; unavailable
pairs remain in the selected denominator. A constant field also has zero
intrafibre drift, so this metric must be considered alongside separation and
coverage.

This is a **partial checkpoint**, retaining an earlier computed fine-band state
plus initialized support growth; the restarted campaign stopped before
committing a new solved block.
These are development consistency measurements that can include fitting fibres,
not an independent holdout, physical validation or a final convergence claim.

<details>
<summary>Fine-grid checkpoint and evaluation identities</summary>

- Bruniss AFV v2: `1898ff5098f03a36e02c76efc5f6b3774b766b154bfc8faadca488459114c941`.
- Latest fine checkpoint used in this measurement: `7dd87597ed8ce712cbd41f32b65d1039b158106c7c7da9a82fa1aca23fcc7dd9`.
- Frozen pair selection: `57f1554a97bf9519f5cacf37e53af7cb9b7b00473aa66ca8d927f889498cd8ee`.
- Fine evaluation result: `80bf1a30c94f1c4c334461b7cd9d767b60a6acf524342173782fb622c22af834`.
- Fine checkpoint evaluation adapter: `0a23fb0f33503a813bf1ec1adac729f31bcd2ed4344b6c683b245fe2de0c13c7`.

These measurements were produced by a checkpoint-specific research adapter.
The checkpoint, frozen pairs and per-pair predictions remain external execution
artifacts. The [benchmark guide](../benchmarks/winding/README.md) describes the
evaluation contract; the offline synthetic demo does not reproduce these
Paris4 measurements.

</details>
