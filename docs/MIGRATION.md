# Migrating from `winding-field-experiments`

The Python distribution is named **`winding-field`** (formerly
`winding-field-experiments`). Python module names and solver behavior are unchanged.

- Use a fresh environment: the old and new distributions own the same module files
  and must not be installed together.
- Keep the original environment for saved runs whose runtime identity is pinned.
- The `winding-field` command comes from the separate
  [`winding-field-cli`](https://github.com/LimeGS/winding-field-cli) package.
