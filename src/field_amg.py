"""Smoothed-aggregation algebraic multigrid preconditioner for the winding-field PCG.

The Jacobi-preconditioned CG in :func:`field_solver.solve_field` needs a number of
iterations that grows with the grid side, so the cost per node grows with block size.
This module builds an algebraic multigrid (AMG) hierarchy for the *same* free-node
system and applies one symmetric V-cycle per CG iteration. Its iteration count grows
only slowly with the grid side.

Setup runs once on the CPU with PyAMG (smoothed aggregation, optional dependency
``pyamg``); the resulting operators are moved to the solve device as Torch CSR
tensors, so every V-cycle runs on the GPU when the solve does. A prepared hierarchy
depends only on the static operator (weights, pairs, constraint rows and fixed mask):
right-hand sides, constraint targets and fixed values may change between solves.

The preconditioner is a fixed symmetric positive definite linear map (Chebyshev-Jacobi or
l1-Jacobi pre- and post-smoothing, Galerkin coarse operators, exact coarsest solve), so CG
keeps its guarantees. It changes iteration counts only, never the system that is solved.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

__all__ = ["AMGPreconditioner", "assemble_free_operator"]

# Grouped constraint rows couple every node of a group; the exact elimination of the
# group mean adds a dense block per group. Past this many entries the (larger, still
# SPD) operator without the mean elimination is used for the hierarchy instead.
_GROUP_DENSE_LIMIT = 50_000_000


def _csr(rows, cols, vals, n):
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def assemble_free_operator(
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    pairs: np.ndarray | None = None,
    lam: float = 0.0,
    constraints: Any = None,
    reference_weight: np.ndarray | float = 0.0,
) -> sp.csr_matrix:
    """Return the float64 CSR matrix that ``solve_field`` applies on free nodes.

    Fixed nodes (``mask``) have identity rows and columns, matching the solver's exact
    elimination. ``weights`` are nodal; edge weights are their arithmetic means.
    """
    w = np.asarray(weights, dtype=np.float64)
    shape = w.shape
    n = w.size
    index = np.arange(n, dtype=np.int64).reshape(shape)
    rows, cols, vals = [], [], []
    diagonal = np.zeros(n, dtype=np.float64)
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a = index[tuple(lo)].ravel()
        b = index[tuple(hi)].ravel()
        edge = (0.5 * w[tuple(lo)] + 0.5 * w[tuple(hi)]).ravel()
        rows += [a, b]
        cols += [b, a]
        vals += [-edge, -edge]
        np.add.at(diagonal, a, edge)
        np.add.at(diagonal, b, edge)
    if pairs is not None and len(pairs) and lam:
        p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        p = p[p[:, 0] != p[:, 1]]
        coupling = np.full(len(p), float(lam))
        rows += [p[:, 0], p[:, 1]]
        cols += [p[:, 1], p[:, 0]]
        vals += [-coupling, -coupling]
        np.add.at(diagonal, p[:, 0], coupling)
        np.add.at(diagonal, p[:, 1], coupling)
    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diagonal + np.broadcast_to(np.asarray(reference_weight, dtype=np.float64), shape).ravel())
    matrix = _csr(np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), n)
    if constraints is not None:
        c = sp.csr_matrix(constraints.matrix, dtype=np.float64)
        cw = np.asarray(constraints.weights, dtype=np.float64)
        matrix = matrix + (c.T @ sp.diags(cw) @ c).tocsr()
        groups = np.asarray(getattr(constraints, "groups", np.full(c.shape[0], -1)))
        if (groups >= 0).any():
            means = (constraints.members @ c).tocsr()
            dense = int(np.sum(np.diff(means.indptr).astype(np.int64) ** 2))
            if dense <= _GROUP_DENSE_LIMIT:
                total = np.asarray(constraints.group_sum, dtype=np.float64)
                inverse = np.divide(1, total, out=np.zeros_like(total), where=total > 0)
                matrix = matrix - (means.T @ sp.diags(inverse) @ means).tocsr()
    fixed = np.asarray(mask, dtype=bool).ravel()
    if fixed.any():
        keep = sp.diags((~fixed).astype(np.float64))
        matrix = (keep @ matrix @ keep + sp.diags(fixed.astype(np.float64))).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _torch_csr(matrix: sp.csr_matrix, device, dtype) -> torch.Tensor:
    matrix = sp.csr_matrix(matrix)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
        return torch.sparse_csr_tensor(
            torch.as_tensor(matrix.indptr, dtype=torch.int64, device=device),
            torch.as_tensor(matrix.indices, dtype=torch.int64, device=device),
            torch.as_tensor(matrix.data, dtype=dtype, device=device),
            size=matrix.shape,
        )


class AMGPreconditioner:
    """One symmetric V-cycle of a smoothed-aggregation hierarchy, applied on a device.

    ``matrix`` is the float64 CSR free-node operator (see :func:`assemble_free_operator`).
    Before and after each coarse correction the smoother runs ``smoothing`` steps: a
    degree-``smoothing`` Chebyshev polynomial in ``D^-1 A`` over ``[lmax / 30, lmax]``
    (default), or that many l1-Jacobi sweeps.
    """

    kind = "amg"

    def __init__(
        self,
        matrix: sp.csr_matrix,
        shape: tuple[int, ...],
        mask: torch.Tensor,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        smoothing: int = 2,
        smoother: str = "chebyshev",
        max_coarse: int = 1000,
        max_levels: int = 12,
    ) -> None:
        try:
            import pyamg
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ValueError("preconditioner='amg' requires the optional 'pyamg' package") from exc
        if smoothing < 1 or max_coarse < 1:
            raise ValueError("smoothing and max_coarse must be positive")
        if smoother not in ("chebyshev", "l1-jacobi"):
            raise ValueError("smoother must be chebyshev or l1-jacobi")
        self.smoother = smoother
        self.shape = tuple(shape)
        self.mask = mask
        self.smoothing = int(smoothing)
        self.dtype = dtype
        hierarchy = pyamg.smoothed_aggregation_solver(
            matrix,
            symmetry="symmetric",
            max_coarse=max_coarse,
            max_levels=max_levels,
            keep=False,
        )
        self.levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.bounds: list[tuple[float, float]] = []
        rng = np.random.default_rng(0)
        for level in hierarchy.levels[:-1]:
            a = sp.csr_matrix(level.A)
            if smoother == "chebyshev":
                d = a.diagonal()
                if not np.isfinite(d).all() or (d <= 0).any():
                    raise ValueError("AMG level has a nonpositive diagonal")
                scale = 1.0 / d
                x = rng.random(a.shape[0]) + 0.5
                lmax = 1.0
                for _ in range(20):
                    y = scale * (a @ x)
                    lmax = float(np.linalg.norm(y) / np.linalg.norm(x))
                    x = y / np.linalg.norm(y)
                lmax *= 1.1
                self.bounds.append((lmax / 30.0, lmax))
            else:
                l1 = np.asarray(abs(a).sum(axis=1)).ravel()
                if not np.isfinite(l1).all() or (l1 <= 0).any():
                    raise ValueError("AMG level has a nonpositive l1 row norm")
                scale = 1.0 / l1
            self.levels.append(
                (
                    _torch_csr(a, device, dtype),
                    _torch_csr(level.P, device, dtype),
                    _torch_csr(sp.csr_matrix(level.P.T), device, dtype),
                    torch.as_tensor(scale, dtype=dtype, device=device),
                )
            )
        coarse = np.asarray(hierarchy.levels[-1].A.toarray(), dtype=np.float64)
        coarse = 0.5 * (coarse + coarse.T)
        inverse = np.linalg.pinv(coarse, hermitian=True)
        self.coarse = torch.as_tensor(inverse, dtype=dtype, device=device)
        self.complexity = float(hierarchy.operator_complexity())
        self.depth = len(hierarchy.levels)
        if not math.isfinite(self.complexity):
            raise ValueError("AMG hierarchy is not finite")

    def metadata(self) -> dict[str, Any]:
        return dict(
            kind="amg",
            levels=self.depth,
            operator_complexity=self.complexity,
            smoother=self.smoother,
            smoothing=self.smoothing,
        )

    def _smooth(self, k: int, x: torch.Tensor | None, b: torch.Tensor) -> torch.Tensor:
        a, _, _, scale = self.levels[k]
        if self.smoother == "l1-jacobi":
            for _ in range(self.smoothing):
                x = scale * b if x is None else x + scale * (b - a @ x)
            assert x is not None
            return x
        lmin, lmax = self.bounds[k]
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        r = b.clone() if x is None else b - a @ x
        out = torch.zeros_like(b) if x is None else x
        rho = 1.0 / sigma
        step = scale * r / theta
        for i in range(self.smoothing):
            out = out + step
            if i == self.smoothing - 1:
                break
            r = r - a @ step
            rho_next = 1.0 / (2.0 * sigma - rho)
            step = rho_next * rho * step + (2.0 * rho_next / delta) * scale * r
            rho = rho_next
        return out

    def _cycle(self, k: int, b: torch.Tensor) -> torch.Tensor:
        if k == len(self.levels):
            return self.coarse @ b
        a, p, r, _ = self.levels[k]
        x = self._smooth(k, None, b)
        x = x + p @ self._cycle(k + 1, r @ (b - a @ x))
        return self._smooth(k, x, b)

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        b = residual.reshape(-1).to(self.dtype)
        out = self._cycle(0, b).reshape(self.shape).to(residual.dtype)
        out[self.mask] = 0
        return out
