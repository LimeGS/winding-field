"""Smoothed-aggregation algebraic multigrid preconditioner for the winding-field PCG.

The Jacobi-preconditioned CG in :func:`field_solver.solve_field` needs a number of
iterations that grows with the grid side, so the cost per node grows with block size.
This module builds an algebraic multigrid (AMG) hierarchy for the *same* free-node
system and applies one symmetric V-cycle per CG iteration. Its iteration count grows
only slowly with the grid side.

The hierarchy is built once per static operator (weights, pairs, constraint rows and
fixed mask; right-hand sides, constraint targets and fixed values may change between
solves). The default ``setup="torch"`` builds it with Torch tensor ops on the solve
device: pairwise-matching aggregation on the strong graph, Jacobi-smoothed prolongation
and Galerkin coarse operators, with no extra dependency. ``setup="pyamg"`` builds it on
the CPU with PyAMG (optional dependency ``pyamg``) and moves it to the device. Every
V-cycle runs on the solve device.

The preconditioner is a fixed symmetric positive definite linear map (Chebyshev-Jacobi or
l1-Jacobi pre- and post-smoothing, Galerkin coarse operators, exact coarsest solve), so CG
keeps its guarantees. It changes iteration counts only, never the system that is solved.
"""

from __future__ import annotations

import math
import time
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


def _compact(a: torch.Tensor, dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    """Return CSR ``a`` on ``device`` with ``dtype`` values and int32 indices when they fit."""
    index = torch.int32 if max(a.shape[0], a.shape[1], a.values().numel()) < 2**31 else torch.int64
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
        return torch.sparse_csr_tensor(
            a.crow_indices().to(device, index),
            a.col_indices().to(device, index),
            a.values().to(device, dtype),
            size=a.shape,
        )


def _csr_rows(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    crow = a.crow_indices()
    rows = torch.repeat_interleave(torch.arange(a.shape[0], device=crow.device), crow.diff())
    return rows, a.col_indices(), a.values()


def _diagonal(a: torch.Tensor) -> torch.Tensor:
    rows, cols, vals = _csr_rows(a)
    on = rows == cols
    return torch.zeros(a.shape[0], dtype=vals.dtype, device=vals.device).index_add_(0, rows[on], vals[on])


def _spectral_bound(a: torch.Tensor, scale: torch.Tensor, steps: int = 20) -> float:
    """Power-iteration estimate of the largest eigenvalue of ``diag(scale) @ a``."""
    generator = torch.Generator().manual_seed(0)
    x = (torch.rand(a.shape[0], generator=generator, dtype=torch.float64) + 0.5).to(scale.device, scale.dtype)
    bound = 1.0
    for _ in range(steps):
        y = scale * (a @ x)
        norm = torch.linalg.vector_norm(y)
        bound = float(norm / torch.linalg.vector_norm(x))
        x = y / norm
    return bound


def _pairwise_aggregates(a: torch.Tensor, theta: float, seed: int, rounds: int = 6) -> torch.Tensor:
    """One pass of parallel handshake matching on the strong graph of ``a``.

    An off-diagonal entry is strong when ``-a_ij / sqrt(a_ii a_jj)`` is at least ``theta``
    times its row maximum. Mutually best strong neighbours pair up; unmatched nodes with a
    strong edge to a matched node join that node's aggregate. Returns aggregate ids.
    """
    n = a.shape[0]
    device = a.values().device
    rows, cols, vals = _csr_rows(a)
    diagonal = _diagonal(a)
    # Off-diagonal negative entries only; every other entry is never strong.
    keep = (rows != cols) & (vals < 0)
    rows, cols = rows[keep], cols[keep]
    strength = vals[keep].neg_().div_(torch.sqrt(diagonal[rows] * diagonal[cols]))
    del vals, keep, diagonal
    rowmax = torch.zeros(n, dtype=strength.dtype, device=device).scatter_reduce_(0, rows, strength, "amax")
    keep = strength >= theta * rowmax[rows]
    del rowmax
    rows, cols, strength = rows[keep], cols[keep], strength[keep]
    del keep
    jitter = torch.minimum(rows, cols).mul_(2654435761).add_(torch.maximum(rows, cols).mul_(97531)).add_(seed * 7919)
    score = jitter.remainder_(1000003).to(strength.dtype).mul_(0.1 / 1000003.0).add_(1.0).mul_(strength)
    del jitter
    nodes = torch.arange(n, device=device)
    partner = torch.full((n,), -1, dtype=torch.int64, device=device)
    for _ in range(rounds):
        free = partner < 0
        active = free[rows] & free[cols]
        if not bool(active.any()):
            break
        r, c, sc = rows[active], cols[active], score[active]
        best = torch.full((n,), -1.0, dtype=sc.dtype, device=device).scatter_reduce_(0, r, sc, "amax")
        pick = sc == best[r]
        choice = torch.full((n,), -1, dtype=torch.int64, device=device)
        choice[r[pick]] = c[pick]
        chosen = torch.nonzero(choice >= 0).squeeze(1)
        chosen = chosen[choice[choice[chosen]] == chosen]
        partner[chosen] = choice[chosen]
    root = torch.where(partner >= 0, torch.minimum(nodes, partner), nodes)
    single = partner < 0
    edge = single[rows] & ~single[cols]
    if bool(edge.any()):
        r, c, sc = rows[edge], cols[edge], strength[edge]
        best = torch.full((n,), -1.0, dtype=sc.dtype, device=device).scatter_reduce_(0, r, sc, "amax")
        pick = sc == best[r]
        root[r[pick]] = root[c[pick]]
    return torch.unique(root, return_inverse=True)[1]


def _galerkin_aggregate(a: torch.Tensor, aggregates: torch.Tensor, size: int) -> torch.Tensor:
    rows, cols, vals = _csr_rows(a)
    index = torch.stack([aggregates[rows], aggregates[cols]])
    return torch.sparse_coo_tensor(index, vals, (size, size)).coalesce().to_sparse_csr()


def _expand(
    rows: torch.Tensor,
    inner: torch.Tensor,
    vals: torch.Tensor,
    right: torch.Tensor,
    shape: tuple[int, int],
    budget: int = 1 << 24,
) -> torch.Tensor:
    """Sparse product by expand, sort and compress.

    ``(rows, inner, vals)`` are the entries of the left factor; ``right`` is a CSR matrix
    whose rows are indexed by ``inner``. Every left entry is expanded against its right
    row and duplicates are summed. The left entries are processed in chunks of at most
    ``budget`` expanded products (unlike library SpGEMM, whose workspace can reach many
    times the operands), so peak memory stays bounded even where rows are dense.
    """
    crow, cols, values = right.crow_indices(), right.col_indices(), right.values()
    counts = crow[inner + 1] - crow[inner]
    ends = torch.cumsum(counts, 0)
    total_count = int(ends[-1]) if ends.numel() else 0
    chunks = max(1, -(-total_count // budget))
    cuts = torch.searchsorted(ends, torch.linspace(0, total_count, chunks + 1, device=ends.device).to(ends.dtype))
    edges = sorted({0, rows.numel(), *(min(int(c) + 1, rows.numel()) for c in cuts[1:-1].tolist())})
    result: torch.Tensor | None = None
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        first = crow[inner[lo:hi]]
        count = counts[lo:hi]
        size = int(count.sum())
        owner = torch.repeat_interleave(torch.arange(hi - lo, device=rows.device), count)
        starts = torch.cumsum(count, 0) - count
        position = first[owner] + torch.arange(size, device=rows.device) - starts[owner]
        index = torch.stack([rows[lo:hi][owner], cols[position]])
        part = torch.sparse_coo_tensor(index, vals[lo:hi][owner] * values[position], shape).coalesce()
        del owner, starts, position, index
        result = part if result is None else (result + part).coalesce()
    if result is None:
        empty = torch.zeros((2, 0), dtype=torch.int64, device=rows.device)
        return torch.sparse_coo_tensor(empty, vals[:0], shape).coalesce()
    return result


def _galerkin(a: torch.Tensor, p: torch.Tensor, block_nnz: int = 1 << 20) -> torch.Tensor:
    """Return ``p.T @ a @ p`` (COO), contracted over row blocks of ``a`` and ``p``.

    Each block of about ``block_nnz`` entries of ``a`` forms ``(A P)_block`` and then
    ``P_block^T (A P)_block`` by expansion, and the partial coarse operators are summed
    as they are produced, so peak memory is bounded by the block size.
    """
    n = a.shape[0]
    size = p.shape[1]
    a_crow, a_cols, a_vals = a.crow_indices(), a.col_indices(), a.values()
    p_crow, p_cols, p_vals = p.crow_indices(), p.col_indices(), p.values()
    nnz = int(a_crow[-1])
    blocks = max(1, -(-nnz // block_nnz))
    cuts = torch.searchsorted(a_crow, torch.linspace(0, nnz, blocks + 1, dtype=a_crow.dtype, device=a_crow.device))
    edges = sorted({0, n, *(min(int(e), n) for e in cuts.tolist())})
    total: torch.Tensor | None = None
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        local = torch.arange(stop - start, device=a_crow.device)
        lo, hi = int(a_crow[start]), int(a_crow[stop])
        rows = torch.repeat_interleave(local, a_crow[start : stop + 1].diff())
        ap = _expand(rows, a_cols[lo:hi], a_vals[lo:hi], p, (stop - start, size)).to_sparse_csr()
        lo, hi = int(p_crow[start]), int(p_crow[stop])
        rows = torch.repeat_interleave(local, p_crow[start : stop + 1].diff())
        part = _expand(p_cols[lo:hi], rows, p_vals[lo:hi], ap, (size, size))
        del ap, rows
        total = part if total is None else (total + part).coalesce()
    assert total is not None
    return total


def _truncate(p: torch.Tensor, threshold: float) -> torch.Tensor:
    """Drop prolongator entries below ``threshold`` times their row's largest magnitude.

    Kept entries are rescaled so every row keeps its sum (the smoothed prolongator
    reproduces constants). Returns CSR. Fewer entries per row keep the Galerkin coarse
    operators sparse on deep levels.
    """
    if threshold <= 0:
        return p.to_sparse_csr()
    index, vals = p.indices(), p.values()
    n = p.shape[0]
    zero = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    rowmax = zero.clone().scatter_reduce_(0, index[0], vals.abs(), "amax")
    keep = vals.abs() >= threshold * rowmax[index[0]]
    before = zero.clone().index_add_(0, index[0], vals)
    index, vals = index[:, keep], vals[keep]
    after = zero.clone().index_add_(0, index[0], vals)
    ratio = torch.where(after != 0, before / after, torch.ones_like(after))
    out = torch.sparse_coo_tensor(index, vals * ratio[index[0]], p.shape).coalesce()
    return out.to_sparse_csr()


def _torch_hierarchy(
    matrix: sp.csr_matrix,
    device: str | torch.device,
    *,
    max_coarse: int,
    max_levels: int,
    theta: float = 0.25,
    passes: int = 3,
    dtype: torch.dtype = torch.float64,
    truncate: float = 0.2,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, float]:
    """Smoothed-aggregation hierarchy built with Torch tensor ops on ``device``.

    Aggregates come from ``passes`` composed pairwise matchings (about 2**passes nodes
    each); the tentative prolongator is smoothed by one damped Jacobi step with weight
    ``4 / (3 lmax)`` and entries below ``truncate`` times their row maximum are dropped
    (row sums kept); coarse operators are Galerkin products. Returns ``[(A_k, P_k)]``
    for every level except the coarsest, the coarsest operator, and the operator
    complexity.
    """
    current = _torch_csr(matrix, device, dtype)
    fine_nnz = current.values().numel()
    total_nnz = fine_nnz
    levels: list[tuple[torch.Tensor, torch.Tensor]] = []
    while current.shape[0] > max_coarse and len(levels) + 1 < max_levels:
        n = current.shape[0]
        aggregates = torch.arange(n, device=current.values().device)
        reduced = current
        for k in range(passes):
            step = _pairwise_aggregates(reduced, theta, seed=10 * len(levels) + k)
            size = int(step.max()) + 1
            aggregates = step[aggregates]
            reduced = _galerkin_aggregate(reduced, step, size)
        size = int(aggregates.max()) + 1
        if size > 0.9 * n:
            break
        scale = 1.0 / _diagonal(current)
        omega = (4.0 / 3.0) / _spectral_bound(current, scale)
        nodes = torch.arange(n, device=aggregates.device)
        tentative = torch.sparse_coo_tensor(
            torch.stack([nodes, aggregates]), torch.ones(n, dtype=dtype, device=aggregates.device), (n, size)
        ).coalesce()
        # A @ P0 sums the columns of A within each aggregate: no sparse product needed.
        rows, cols, vals = _csr_rows(current)
        index = torch.stack([rows, aggregates[cols]])
        product = torch.sparse_coo_tensor(index, vals, (n, size)).coalesce()
        index = product.indices()
        smoothed = torch.sparse_coo_tensor(index, product.values() * (omega * scale[index[0]]), product.shape)
        prolong = _truncate((tentative - smoothed).coalesce(), truncate)
        coarse = _galerkin(current, prolong)
        coarse = (0.5 * (coarse + coarse.t())).coalesce()
        del product, smoothed, tentative, rows, cols, vals, index
        levels.append((current, prolong))
        current = coarse.to_sparse_csr()
        total_nnz += current.values().numel()
    return levels, current.to_dense(), total_nnz / fine_nnz


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
        setup: str = "torch",
        max_coarse: int = 1000,
        max_levels: int = 12,
    ) -> None:
        if setup not in ("torch", "pyamg"):
            raise ValueError("setup must be torch or pyamg")
        if smoothing < 1 or max_coarse < 1:
            raise ValueError("smoothing and max_coarse must be positive")
        if smoother not in ("chebyshev", "l1-jacobi"):
            raise ValueError("smoother must be chebyshev or l1-jacobi")
        self.smoother = smoother
        self.shape = tuple(shape)
        self.mask = mask
        self.smoothing = int(smoothing)
        self.dtype = dtype
        self.setup = setup
        started = time.perf_counter()
        if setup == "pyamg":
            try:
                import pyamg
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ValueError("setup='pyamg' requires the optional 'pyamg' package") from exc
            hierarchy = pyamg.smoothed_aggregation_solver(
                matrix, symmetry="symmetric", max_coarse=max_coarse, max_levels=max_levels, keep=False
            )
            built = [
                (_torch_csr(level.A, device, torch.float64), _torch_csr(level.P, device, torch.float64))
                for level in hierarchy.levels[:-1]
            ]
            coarsest = torch.as_tensor(hierarchy.levels[-1].A.toarray(), dtype=torch.float64, device=device)
            complexity = float(hierarchy.operator_complexity())
        else:
            try:
                built, coarsest, complexity = _torch_hierarchy(
                    matrix, device, max_coarse=max_coarse, max_levels=max_levels
                )
            except torch.OutOfMemoryError:
                # Sparse products need transient workspace well beyond the hierarchy;
                # build on the host instead and move only the compact result.
                torch.cuda.empty_cache()
                built, coarsest, complexity = _torch_hierarchy(
                    matrix, "cpu", max_coarse=max_coarse, max_levels=max_levels
                )
        self.levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.bounds: list[tuple[float, float]] = []
        depth = len(built) + 1
        built.reverse()
        while built:  # pop so each float64 level is freed once it is compacted
            a, p = built.pop()
            if smoother == "chebyshev":
                d = _diagonal(a)
                if not bool(torch.isfinite(d).all()) or bool((d <= 0).any()):
                    raise ValueError("AMG level has a nonpositive diagonal")
                scale = 1.0 / d
                lmax = 1.1 * _spectral_bound(a, scale)
                self.bounds.append((lmax / 30.0, lmax))
            else:
                rows, _, vals = _csr_rows(a)
                l1 = torch.zeros(a.shape[0], dtype=vals.dtype, device=vals.device).index_add_(0, rows, vals.abs())
                if not bool(torch.isfinite(l1).all()) or bool((l1 <= 0).any()):
                    raise ValueError("AMG level has a nonpositive l1 row norm")
                scale = 1.0 / l1
            pt = p.to_sparse_coo().t().coalesce().to_sparse_csr()
            self.levels.append(
                (
                    _compact(a, dtype, device),
                    _compact(p, dtype, device),
                    _compact(pt, dtype, device),
                    scale.to(device, dtype),
                )
            )
            del a, p, pt
        coarsest = 0.5 * (coarsest + coarsest.T)
        self.coarse = torch.linalg.pinv(coarsest, hermitian=True).to(device, dtype)
        self.complexity = complexity
        self.setup_seconds = time.perf_counter() - started
        self.depth = depth
        if not math.isfinite(self.complexity):
            raise ValueError("AMG hierarchy is not finite")

    def metadata(self) -> dict[str, Any]:
        return dict(
            kind="amg",
            levels=self.depth,
            operator_complexity=self.complexity,
            setup=self.setup,
            setup_seconds=round(self.setup_seconds, 3),
            dtype=str(self.dtype).removeprefix("torch."),
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
