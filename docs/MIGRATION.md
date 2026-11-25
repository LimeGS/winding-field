# Migration

## From `winding-field-experiments`

The Python distribution is named **`winding-field`** (formerly
`winding-field-experiments`). Python module names and solver behavior are unchanged.

- Use a fresh environment: the old and new distributions own the same module files
  and must not be installed together.
- Keep the original environment for saved runs whose runtime identity is pinned.
- The `winding-field` command comes from the separate
  [`winding-field-cli`](https://github.com/LimeGS/winding-field-cli) package.

## Upgrading to 2.0.0

Solver behavior, Jacobi run identities and checkpoints are unchanged: a block run
written by 1.x resumes with 2.0.0 to the same bytes.

Five research helpers that no module, runner or the CLI called were removed. Copy
them from the 1.7.0 tag if you still need them:

| Removed | Replacement |
|---|---|
| `common.sample_pairs`, `common.calib_gap` | none |
| `physical_grid.offset_correction` | `np.nanmedian(a - b)` |
| `paris_solve.working_shape` | `physical_grid.grid_shape(lo_xyz, hi_xyz, spacing_xyz)` |
| `paris_solve.load_pairs` | `paris_pair_check.load_pair_records` |
