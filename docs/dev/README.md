# Developer notes

Engineering records kept for maintainers. They describe how past performance and
reuse work was measured and are not needed to use the library.

- [Result-preserving performance (v2)](PERFORMANCE_V2.md)
- [Solver performance v3](PERFORMANCE_V3.md) and its [plan](PERFORMANCE_V3_PLAN.md)
- [Work reuse](WORK_REUSE.md)

## Notes moved from the README

For opt-in throughput measurements (distinct from the scientific quality benchmark),
see result-preserving performance (v2). Solver performance v3 covers optional CPU
float64 spectral PCG, prepared consensus indices, bounded template/gather caches and
durable I/O measurements, including unsupported geometries and the spatial-index
regime that became slower.

`solve_blocks(..., prepared_cache_bytes=0)` keeps the historical default: no prepared
workspace retention between visits. A positive byte budget admits a stable subset of
workspaces; observation receipts, static snapshots and live boundary/constraint
targets are still validated on every visit. The budget is operational and may change
on resume; it is separate from `BlockStore` cache.

Mypy checks annotated functions and a typed solver caller; unannotated
research-function bodies and array shapes/ranges still rely on runtime checks and
tests. The October 27 separated core wheel passed its then-current 1227 tests outside
the checkout; workflow/render tests now run in the CLI repository (see its
[separation verification](https://github.com/LimeGS/winding-field-cli/blob/main/docs/verification/core-cli-separation.md)).
