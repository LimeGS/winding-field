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
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
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


def _workers(threads: int | None) -> int:
    """Worker threads for the CPU setup: ``threads``, else every CPU this process may use."""
    if threads is not None:
        if int(threads) < 1:
            raise ValueError("threads must be positive")
        return int(threads)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - macOS and Windows
        return max(1, os.cpu_count() or 1)


def _row_blocks(n: int, workers: int, rows_per_block: int = 1 << 19) -> list[tuple[int, int]]:
    """Contiguous row ranges: enough to balance ``workers`` and bound per-block scratch."""
    count = max(1, min(n, max(4 * workers if workers > 1 else 1, -(-n // rows_per_block))))
    cuts = np.linspace(0, n, count + 1).astype(np.int64)
    return [(int(lo), int(hi)) for lo, hi in zip(cuts[:-1], cuts[1:]) if hi > lo]


def _map_blocks(function, blocks, workers):
    if workers <= 1 or len(blocks) <= 1:
        return [function(*block) for block in blocks]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="amg-setup") as pool:
        return list(pool.map(lambda block: function(*block), blocks))


def _stack_rows(parts: list[sp.csr_matrix], columns: int, workers: int) -> sp.csr_matrix:
    """Stack canonical CSR row blocks into one CSR matrix, copying the blocks in parallel."""
    counts = np.array([part.nnz for part in parts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    rows = sum(part.shape[0] for part in parts)
    total = int(offsets[-1])
    itype = np.int32 if max(total, columns, rows) < 2**31 else np.int64
    indptr = np.empty(rows + 1, dtype=itype)
    indices = np.empty(total, dtype=itype)
    data = np.empty(total, dtype=np.float64)
    indptr[0] = 0
    row_offsets = np.concatenate([[0], np.cumsum([part.shape[0] for part in parts])])

    def copy(k):
        part, lo, hi = parts[k], int(offsets[k]), int(offsets[k + 1])
        indptr[row_offsets[k] + 1 : row_offsets[k + 1] + 1] = part.indptr[1:] + lo
        indices[lo:hi] = part.indices
        data[lo:hi] = part.data

    _map_blocks(copy, [(k,) for k in range(len(parts))], workers)
    matrix = sp.csr_matrix((data, indices, indptr), shape=(rows, columns), copy=False)
    matrix.has_sorted_indices = all(part.has_sorted_indices for part in parts)
    return matrix


def _gram_rows(left: sp.csr_matrix, weights: np.ndarray, right: sp.csr_matrix, lo: int, hi: int):
    """Rows ``lo:hi`` of ``right.T @ diag(weights) @ right`` as CSR.

    ``left`` is ``right.T`` in CSR with sorted indices. Each entry is the sum over the
    shared index in ascending order of ``right[k, j] * (right[k, i] * weights[k])``,
    the same terms in the same order as the serial column-major product, so the values
    are bitwise equal to ``(right.T @ sp.diags(weights) @ right)[lo:hi]``.
    """
    block = left[lo:hi]
    scaled = sp.csr_matrix((block.data * weights[block.indices], block.indices, block.indptr), shape=block.shape)
    return scaled @ right


def assemble_free_operator(
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    pairs: np.ndarray | None = None,
    lam: float = 0.0,
    constraints: Any = None,
    reference_weight: np.ndarray | float = 0.0,
    threads: int | None = None,
) -> sp.csr_matrix:
    """Return the float64 CSR matrix that ``solve_field`` applies on free nodes.

    Fixed nodes (``mask``) have identity rows and columns, matching the solver's exact
    elimination. ``weights`` are nodal; edge weights are their arithmetic means.

    Rows are assembled in contiguous blocks on ``threads`` worker threads (default: every
    CPU this process may use). Every entry is summed in the same order as a serial
    assembly, so the result is bitwise identical for any thread count.
    """
    w = np.asarray(weights, dtype=np.float64)
    shape = w.shape
    n = w.size
    workers = _workers(threads)
    itype = np.int32 if n < 2**31 else np.int64
    index = np.arange(n, dtype=itype).reshape(shape)
    # Stencil edges per axis; both endpoint lists are increasing, so the edges of a row
    # block are one contiguous range of each list.
    edges = []
    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a = index[tuple(lo)].ravel()
        b = index[tuple(hi)].ravel()
        edges.append((a, b, (0.5 * w[tuple(lo)] + 0.5 * w[tuple(hi)]).ravel()))
    del index
    # Pair couplings as (row ends, column ends, values), in the serial assembly's order.
    couplings: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if pairs is not None and len(pairs) and lam:
        p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).astype(itype)
        p = p[p[:, 0] != p[:, 1]]
        coupling = np.full(len(p), float(lam))
        couplings = [(p[:, 0], p[:, 1], coupling), (p[:, 1], p[:, 0], coupling)]
    reference = np.broadcast_to(np.asarray(reference_weight, dtype=np.float64), shape).ravel()
    fixed = np.asarray(mask, dtype=bool).ravel()
    gram = []
    if constraints is not None:
        c = sp.csr_matrix(constraints.matrix, dtype=np.float64)
        cw = np.asarray(constraints.weights, dtype=np.float64)
        gram.append((c.T.tocsr(), cw, c, 1.0))
        groups = np.asarray(getattr(constraints, "groups", np.full(c.shape[0], -1)))
        if (groups >= 0).any():
            means = (constraints.members @ c).tocsr()
            dense = int(np.sum(np.diff(means.indptr).astype(np.int64) ** 2))
            if dense <= _GROUP_DENSE_LIMIT:
                total = np.asarray(constraints.group_sum, dtype=np.float64)
                inverse = np.divide(1, total, out=np.zeros_like(total), where=total > 0)
                gram.append((means.T.tocsr(), inverse, means, -1.0))
    for left, _, _, _ in gram:
        left.sort_indices()

    def block(lo: int, hi: int) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        diagonal = np.zeros(hi - lo, dtype=np.float64)
        for a, b, edge in edges:
            for ends, other in ((a, b), (b, a)):
                s, e = np.searchsorted(ends, (lo, hi))
                rows.append(ends[s:e] - lo)
                cols.append(other[s:e])
                vals.append(-edge[s:e])
            for ends in (a, b):
                s, e = np.searchsorted(ends, (lo, hi))
                diagonal += np.bincount((ends[s:e] - lo).astype(np.intp), weights=edge[s:e], minlength=hi - lo)
        inside = [(ends >= lo) & (ends < hi) for ends, _, _ in couplings]
        for (ends, other, value), selected in zip(couplings, inside):
            rows.append(ends[selected] - lo)
            cols.append(other[selected])
            vals.append(-value[selected])
        for (ends, _, value), selected in zip(couplings, inside):
            diagonal += np.bincount((ends[selected] - lo).astype(np.intp), weights=value[selected], minlength=hi - lo)
        local = np.arange(hi - lo, dtype=itype)
        rows.append(local)
        cols.append(local + lo)
        vals.append(diagonal + reference[lo:hi])
        part = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(hi - lo, n)
        ).tocsr()
        for left, scale, right, sign in gram:
            product = _gram_rows(left, scale, right, lo, hi)
            part = part + product if sign > 0 else part - product
        if fixed.any():
            keep = (~fixed).astype(np.float64)
            part = (sp.diags(keep[lo:hi]) @ part @ sp.diags(keep)).tocsr()
            pinned = np.flatnonzero(fixed[lo:hi])
            part = part + sp.csr_matrix((np.ones(len(pinned)), (pinned, pinned + lo)), shape=(hi - lo, n))
        part = sp.csr_matrix(part)
        part.sum_duplicates()
        part.eliminate_zeros()
        return part

    return _stack_rows(_map_blocks(block, _row_blocks(n, workers), workers), n, workers)


def _free_submatrix(matrix: sp.csr_matrix, keep: np.ndarray, threads: int | None = None) -> sp.csr_matrix:
    """``matrix[keep][:, keep]`` for increasing ``keep``, built on row blocks in parallel."""
    matrix = sp.csr_matrix(matrix)
    workers = _workers(threads)
    keep = np.asarray(keep)
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
    position = np.full(matrix.shape[1], -1, dtype=np.int64)
    position[keep] = np.arange(len(keep))

    def block(lo: int, hi: int) -> sp.csr_matrix:
        rows = keep[lo:hi]
        starts, stops = matrix.indptr[rows], matrix.indptr[rows + 1]
        counts = (stops - starts).astype(np.int64)
        take = np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(int(counts.sum()))
        cols = position[matrix.indices[take]]
        inside = cols >= 0
        owner = np.repeat(np.arange(hi - lo), counts)[inside]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=hi - lo))])
        part = sp.csr_matrix((matrix.data[take][inside], cols[inside], indptr), shape=(hi - lo, len(keep)))
        part.has_sorted_indices = True  # positions are increasing in the original column order
        return part

    return _stack_rows(_map_blocks(block, _row_blocks(len(keep), workers), workers), len(keep), workers)


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


def _csr_tensor(crow: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, shape) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.*")
        return torch.sparse_csr_tensor(crow, cols, values, size=tuple(shape))


def _index_dtype(*sizes: int) -> torch.dtype:
    return torch.int32 if max(sizes) < 2**31 else torch.int64


def _compact(a: torch.Tensor, dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    """Return CSR ``a`` on ``device`` with ``dtype`` values and int32 indices when they fit."""
    index = _index_dtype(a.shape[0], a.shape[1], a.values().numel())
    return _csr_tensor(
        a.crow_indices().to(device, index), a.col_indices().to(device, index), a.values().to(device, dtype), a.shape
    )


def _compact_host(matrix: sp.csr_matrix, dtype: torch.dtype, device: str | torch.device) -> torch.Tensor:
    """``_compact(_torch_csr(matrix, device, float64), dtype, device)`` with no float64 copy on ``device``."""
    matrix = sp.csr_matrix(matrix)
    values = torch.as_tensor(matrix.data, dtype=torch.float64)
    index = _index_dtype(matrix.shape[0], matrix.shape[1], values.numel())
    return _csr_tensor(
        torch.as_tensor(matrix.indptr).to(index).to(device),
        torch.as_tensor(matrix.indices).to(index).to(device),
        values.to(dtype).to(device),
        matrix.shape,
    )


# The hierarchy build works on row blocks of about this many stored entries, so its scratch
# memory is bounded by the block instead of growing with the operator.
_SETUP_BLOCK = 1 << 22


def _row_spans(crow: torch.Tensor, budget: int) -> list[tuple[int, int]]:
    """Contiguous row ranges of a CSR matrix that hold about ``budget`` stored entries each."""
    n = crow.numel() - 1
    nnz = int(crow[-1])
    blocks = max(1, -(-nnz // budget))
    cuts = torch.searchsorted(crow, torch.linspace(0, nnz, blocks + 1, dtype=crow.dtype, device=crow.device))
    edges = sorted({0, n, *(min(int(e), n) for e in cuts.tolist())})
    return [(start, stop) for start, stop in zip(edges[:-1], edges[1:]) if stop > start]


def _csr_block(a: torch.Tensor, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row ids, columns and values of the stored entries of rows ``start:stop`` of CSR ``a``."""
    crow = a.crow_indices()
    lo, hi = int(crow[start]), int(crow[stop])
    counts = crow[start : stop + 1].diff()
    rows = torch.repeat_interleave(torch.arange(start, stop, device=crow.device), counts, output_size=hi - lo)
    return rows, a.col_indices()[lo:hi], a.values()[lo:hi]


def _diagonal(a: torch.Tensor, block: int = _SETUP_BLOCK) -> torch.Tensor:
    values = a.values()
    out = torch.zeros(a.shape[0], dtype=values.dtype, device=values.device)
    for start, stop in _row_spans(a.crow_indices(), block):
        rows, cols, vals = _csr_block(a, start, stop)
        on = rows == cols
        out.index_add_(0, rows[on], vals[on])
    return out


def _absolute_row_sums(a: torch.Tensor, block: int = _SETUP_BLOCK) -> torch.Tensor:
    values = a.values()
    out = torch.zeros(a.shape[0], dtype=values.dtype, device=values.device)
    for start, stop in _row_spans(a.crow_indices(), block):
        rows, _, vals = _csr_block(a, start, stop)
        out.index_add_(0, rows, vals.abs())
    return out


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


def _chebyshev_bound(a: torch.Tensor, scale: torch.Tensor) -> float:
    """Upper end of the Chebyshev interval for ``diag(scale) @ a``.

    Chebyshev smoothing amplifies modes above ``lmax``, so an estimate below the top
    eigenvalue can make the V-cycle indefinite. A 20-step power estimate times 1.1 came
    out up to 23 % low on coarse levels of real PHerc0175A blocks, and a block solve then
    occasionally aborted on a nonpositive CG curvature. This takes a 40-step estimate times
    1.3, capped by the scaled row-sum norm, which bounds the spectrum by construction but
    is up to ~2.8x loose on fine levels with stiff fibre pairs.
    """
    sums = _absolute_row_sums(a)
    return min(float((scale * sums).max()), 1.3 * _spectral_bound(a, scale, steps=40))


def _strong_edges(
    a: torch.Tensor, start: int, stop: int, diagonal: torch.Tensor, theta: float, seed: int | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Strong entries of rows ``start:stop`` of ``a``: row ids, columns, strengths, scores.

    An off-diagonal entry is strong when ``-a_ij / sqrt(a_ii a_jj)`` is at least ``theta``
    times its row maximum. The matching score is the strength times a jitter in [1, 1.1)
    hashed from the edge and ``seed`` (skipped when ``seed`` is None), which breaks ties
    between equal couplings.
    """
    rows, cols, vals = _csr_block(a, start, stop)
    # Off-diagonal negative entries only; every other entry is never strong.
    keep = (rows != cols) & (vals < 0)
    rows, cols = rows[keep], cols[keep]
    strength = vals[keep].neg_().div_(torch.sqrt(diagonal[rows] * diagonal[cols]))
    del vals, keep
    local = rows - start
    rowmax = torch.zeros(stop - start, dtype=strength.dtype, device=strength.device)
    rowmax.scatter_reduce_(0, local, strength, "amax")
    keep = strength >= theta * rowmax[local]
    rows, cols, strength = rows[keep], cols[keep], strength[keep]
    if seed is None:
        return rows, cols, strength, None
    jitter = torch.minimum(rows, cols).mul_(2654435761).add_(torch.maximum(rows, cols).mul_(97531)).add_(seed * 7919)
    score = jitter.remainder_(1000003).to(strength.dtype).mul_(0.1 / 1000003.0).add_(1.0).mul_(strength)
    return rows, cols, strength, score


def _last_best(local: torch.Tensor, value: torch.Tensor, count: int) -> torch.Tensor:
    """Per row ``0..count-1``, the position of the last entry with the row's largest value.

    Rows without entries get -1. ``value`` is positive. Taking the last of tied entries is
    what a serial ``out[rows[pick]] = cols[pick]`` keeps, made explicit so that it holds on
    every device and thread count.
    """
    best = torch.full((count,), -1.0, dtype=value.dtype, device=value.device).scatter_reduce_(0, local, value, "amax")
    pick = torch.nonzero(value == best[local]).squeeze(1)
    last = torch.full((count,), -1, dtype=torch.int64, device=local.device)
    return last.scatter_reduce_(0, local[pick], pick, "amax")


def _pairwise_aggregates(
    a: torch.Tensor, theta: float, seed: int, rounds: int = 6, block: int = _SETUP_BLOCK
) -> torch.Tensor:
    """One pass of parallel handshake matching on the strong graph of ``a``.

    Every free node picks its best-scoring free strong neighbour and mutual picks pair up,
    for up to ``rounds`` rounds; unmatched nodes with a strong edge to a matched node then
    join the aggregate of their strongest such neighbour (see :func:`_strong_edges`). Ties
    go to the last entry of the row. Returns aggregate ids.

    Strong edges are formed per row block of about ``block`` entries, from the matrix in the
    first two rounds and the joining step; the edges between nodes still unmatched after the
    first round are kept for the later rounds. Peak memory is a block plus those edges.
    """
    n = a.shape[0]
    device = a.values().device
    diagonal = _diagonal(a, block)
    spans = _row_spans(a.crow_indices(), block)

    def scratch():
        for start, stop in spans:
            rows, cols, _, score = _strong_edges(a, start, stop, diagonal, theta, seed)
            assert score is not None
            yield start, stop, rows, cols, score

    partner = torch.full((n,), -1, dtype=torch.int64, device=device)
    remaining: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    for step in range(rounds):
        free = partner < 0
        choice = torch.full((n,), -1, dtype=torch.int64, device=device)
        active = []
        found = False
        for start, stop, rows, cols, score in scratch() if remaining is None else remaining:
            if step:  # every node is free in the first round
                keep = free[rows] & free[cols]
                rows, cols, score = rows[keep], cols[keep], score[keep]
                active.append((start, stop, rows, cols, score))
            if rows.numel():
                found = True
                last = _last_best(rows - start, score, stop - start)
                choice[start:stop] = torch.where(last >= 0, cols[last.clamp(min=0)], last)
        if not found:
            break
        if step:
            remaining = active
        del active
        chosen = torch.nonzero(choice >= 0).squeeze(1)
        chosen = chosen[choice[choice[chosen]] == chosen]
        partner[chosen] = choice[chosen]
    del remaining
    nodes = torch.arange(n, device=device)
    root = torch.where(partner >= 0, torch.minimum(nodes, partner), nodes)
    single = partner < 0
    if bool(single.any()):
        target = torch.full((n,), -1, dtype=torch.int64, device=device)
        for start, stop in spans:
            rows, cols, strength, _ = _strong_edges(a, start, stop, diagonal, theta, None)
            edge = single[rows] & ~single[cols]
            rows, cols, strength = rows[edge], cols[edge], strength[edge]
            if rows.numel():
                last = _last_best(rows - start, strength, stop - start)
                target[start:stop] = torch.where(last >= 0, cols[last.clamp(min=0)], last)
        joined = torch.nonzero(target >= 0).squeeze(1)
        # Joined nodes are unmatched and their targets matched, so no read sees a write.
        root[joined] = root[target[joined]]
    return torch.unique(root, return_inverse=True)[1]


# Below this many entries the host coalesce of torch is used as is.
_HOST_COALESCE_MIN = 1 << 16


def _coalesce(t: torch.Tensor, stable: bool = False) -> torch.Tensor:
    """``t.coalesce()`` of a 2-D COO tensor, bitwise identical and faster on the host.

    On the CPU torch's coalesce sorts the flattened indices stably, then sums each run of
    equal indices left to right from its first value in one sequential pass. This does the
    same stable sort, gathers the values in that order and adds them with ``index_add_``
    into ``-0.0`` (``-0.0 + v`` is ``v`` for every ``v``, as copying the first value is), so
    every sum has the same terms in the same order while the sort and gathers use all
    torch threads. Other devices use torch's coalesce.

    Below about 32k entries torch sorts with an unstable sort, so its sums of three or more
    terms may take another order. ``stable`` takes the host path at every size, so every
    sum runs left to right in storage order, as in a coalesce of at least
    ``_HOST_COALESCE_MIN`` entries: blocked builds use it to sum each entry exactly as one
    coalesce of all the blocks' entries does.
    """
    if t.device.type != "cpu" or t.is_coalesced() or (t._nnz() < _HOST_COALESCE_MIN and not stable):
        return t.coalesce()
    index, values = t._indices(), t._values()
    columns = t.shape[1]
    keys, order = torch.sort(index[0] * columns + index[1], stable=True)
    first = torch.ones(keys.numel(), dtype=torch.bool)
    first[1:] = keys[1:] != keys[:-1]
    segment = torch.cumsum(first, 0).sub_(1)
    keys = keys[first]
    del first
    summed = torch.full((keys.numel(),), -0.0, dtype=values.dtype).index_add_(0, segment, values[order])
    del segment, order
    rows = keys // columns
    result = torch.sparse_coo_tensor(torch.stack([rows, keys - rows * columns]), summed, t.shape)
    return result._coalesced_(True)


def _sum_rows_in_order(
    parts, reach: torch.Tensor, size: int, positive_zero: bool = False, stable: bool = False
) -> torch.Tensor:
    """Sum the square COO partials ``parts`` (one per block, in order) into a CSR matrix.

    Each partial is coalesced together with the open entries of the earlier ones, which
    come first, so every entry is summed left to right in block order, exactly as one
    coalesce of every partial's entries in order. ``reach[i]`` (non-decreasing) bounds the
    last block that adds to row ``i``: rows below the first one a later block can reach are
    final and leave the working set at once, already sorted, so the scratch memory is the
    open rows and one partial. ``positive_zero`` adds +0.0 to every value, as sparse
    addition does; ``stable`` is passed to :func:`_coalesce`.
    """
    counts: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    open_index: torch.Tensor | None = None
    open_values: torch.Tensor | None = None
    flushed = 0
    for k, part in enumerate(parts):
        if open_index is not None and open_values is not None and open_values.numel():
            part = torch.sparse_coo_tensor(
                torch.cat([open_index, part._indices()], 1), torch.cat([open_values, part._values()]), (size, size)
            )
        merged = _coalesce(part, stable)
        del part
        index, vals = merged.indices(), merged.values()
        del merged
        bound = int(torch.searchsorted(reach, k, right=True))
        cut = int(torch.searchsorted(index[0], bound))
        counts.append(torch.bincount(index[0, :cut] - flushed, minlength=bound - flushed))
        cols.append(index[1, :cut].clone())
        values.append(vals[:cut] + 0.0 if positive_zero else vals[:cut].clone())
        open_index, open_values = index[:, cut:].clone(), vals[cut:].clone()
        flushed = bound
        del index, vals
    if flushed != size or open_values is None or open_values.numel():
        raise RuntimeError("coarse rows were left open")  # pragma: no cover - reach covers every block
    crow = torch.zeros(size + 1, dtype=torch.int64, device=reach.device)
    torch.cumsum(torch.cat(counts), 0, out=crow[1:])
    col = torch.cat(cols)
    cols.clear()
    value = torch.cat(values)
    values.clear()
    return _csr_tensor(crow, col, value, (size, size))


def _galerkin_aggregate(
    a: torch.Tensor, aggregates: torch.Tensor, size: int, block: int = _SETUP_BLOCK
) -> torch.Tensor:
    """``P0^T a P0`` (CSR) for the piecewise-constant prolongator ``P0`` of ``aggregates``.

    The entries of ``a`` are summed into their coarse entries in row blocks of about
    ``block`` entries; every coarse entry is the left-to-right sum of its terms in storage
    order, bitwise the coalesce of all of them at once.
    """
    crow = a.crow_indices()
    spans = _row_spans(crow, block)
    lengths = torch.tensor([stop - start for start, stop in spans], device=crow.device)
    owner = torch.repeat_interleave(torch.arange(len(spans), device=crow.device), lengths, output_size=a.shape[0])
    last = torch.full((size,), -1, dtype=torch.int64, device=crow.device).scatter_reduce_(0, aggregates, owner, "amax")
    del owner

    def parts():
        for start, stop in spans:
            rows, cols, vals = _csr_block(a, start, stop)
            yield torch.sparse_coo_tensor(torch.stack([aggregates[rows], aggregates[cols]]), vals, (size, size))

    return _sum_rows_in_order(parts(), torch.cummax(last, 0).values, size, stable=len(spans) > 1)


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
        part = _coalesce(torch.sparse_coo_tensor(index, vals[lo:hi][owner] * values[position], shape))
        del owner, starts, position, index
        result = part if result is None else _coalesce(result + part)
    if result is None:
        empty = torch.zeros((2, 0), dtype=torch.int64, device=rows.device)
        return torch.sparse_coo_tensor(empty, vals[:0], shape).coalesce()
    return result


def _galerkin(a: torch.Tensor, p: torch.Tensor, block_nnz: int = 1 << 20) -> torch.Tensor:
    """Return ``p.T @ a @ p`` (CSR), contracted over row blocks of ``a`` and ``p``.

    Each block of about ``block_nnz`` entries of ``a`` forms ``(A P)_block`` and then
    ``P_block^T (A P)_block`` by expansion, and the partial coarse operators are summed
    in block order as they are produced, so peak memory is bounded by the block size and
    the coarse rows still open. A coarse row that no later block reaches is final and is
    set aside at once (see :func:`_sum_rows_in_order`), so the work grows linearly with the
    number of blocks.
    """
    size = p.shape[1]
    a_crow, a_cols, a_vals = a.crow_indices(), a.col_indices(), a.values()
    p_crow, p_cols, p_vals = p.crow_indices(), p.col_indices(), p.values()
    spans = _row_spans(a_crow, block_nnz)
    # The last block whose prolongator rows reach each coarse row.
    last = torch.full((size,), -1, dtype=torch.int64, device=p_cols.device)
    for k, (start, stop) in enumerate(spans):
        last.index_fill_(0, p_cols[int(p_crow[start]) : int(p_crow[stop])].to(torch.int64), k)

    def parts():
        for start, stop in spans:
            local = torch.arange(stop - start, device=a_crow.device)
            lo, hi = int(a_crow[start]), int(a_crow[stop])
            rows = torch.repeat_interleave(local, a_crow[start : stop + 1].diff())
            ap = _expand(rows, a_cols[lo:hi], a_vals[lo:hi], p, (stop - start, size)).to_sparse_csr()
            lo, hi = int(p_crow[start]), int(p_crow[stop])
            rows = torch.repeat_interleave(local, p_crow[start : stop + 1].diff())
            yield _expand(p_cols[lo:hi], rows, p_vals[lo:hi], ap, (size, size))

    # The block partials of an entry are added two at a time, in either order, and sparse
    # addition starts every sum from +0, which turns a -0 entry into +0.
    return _sum_rows_in_order(parts(), torch.cummax(last, 0).values, size, positive_zero=len(spans) > 1)


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
    out = _coalesce(torch.sparse_coo_tensor(index, vals * ratio[index[0]], p.shape))
    return out.to_sparse_csr()


def _stack_csr(parts: list[torch.Tensor], columns: int) -> torch.Tensor:
    """Stack CSR row blocks into one CSR matrix, emptying ``parts``."""
    device = parts[0].values().device
    rows = sum(part.shape[0] for part in parts)
    crow = torch.zeros(rows + 1, dtype=torch.int64, device=device)
    torch.cumsum(torch.cat([part.crow_indices().diff() for part in parts]), 0, out=crow[1:])
    cols = torch.cat([part.col_indices() for part in parts])
    values = torch.cat([part.values() for part in parts])
    parts.clear()
    return _csr_tensor(crow, cols, values, (rows, columns))


def _smoothed_prolongator(
    a: torch.Tensor, aggregates: torch.Tensor, size: int, truncate: float, block: int = _SETUP_BLOCK
) -> torch.Tensor:
    """``(I - omega D^-1 a) P0`` (CSR) with small entries truncated (see :func:`_truncate`).

    ``P0`` is the piecewise-constant prolongator of ``aggregates`` and ``omega`` is
    ``4 / (3 lmax)``. Every row involves only its own row of ``a``, so the rows are formed
    in blocks of about ``block`` entries of ``a``.
    """
    values = a.values()
    device, dtype = values.device, values.dtype
    scale = 1.0 / _diagonal(a, block)
    omega = (4.0 / 3.0) / _spectral_bound(a, scale)
    spans = _row_spans(a.crow_indices(), block)
    parts = []
    for start, stop in spans:
        rows, cols, vals = _csr_block(a, start, stop)
        shape = (stop - start, size)
        # A @ P0 sums the columns of A within each aggregate: no sparse product needed.
        index = torch.stack([rows - start, aggregates[cols]])
        product = _coalesce(torch.sparse_coo_tensor(index, vals, shape), stable=len(spans) > 1)
        del rows, cols, vals
        index = product.indices()
        smoothed = torch.sparse_coo_tensor(index, product.values() * (omega * scale[index[0] + start]), shape)
        del product, index
        nodes = torch.arange(stop - start, device=device)
        tentative = _coalesce(
            torch.sparse_coo_tensor(
                torch.stack([nodes, aggregates[start:stop]]),
                torch.ones(stop - start, dtype=dtype, device=device),
                shape,
            )
        )
        parts.append(_truncate(_coalesce(tentative - smoothed), truncate))
        del smoothed, tentative
    return _stack_csr(parts, size)


def _aggregate(a: torch.Tensor, theta: float, passes: int, seed: int, block: int = _SETUP_BLOCK) -> torch.Tensor:
    """Aggregates of ``passes`` composed pairwise matchings, about ``2**passes`` nodes each."""
    aggregates: torch.Tensor | None = None
    reduced = a
    for k in range(passes):
        step = _pairwise_aggregates(reduced, theta, seed=seed + k, block=block)
        aggregates = step if aggregates is None else step[aggregates]
        if k + 1 < passes:
            reduced = _galerkin_aggregate(reduced, step, int(step.max()) + 1, block)
    assert aggregates is not None
    return aggregates


def _compact_transpose(
    p: torch.Tensor, dtype: torch.dtype, device: str | torch.device, block: int = _SETUP_BLOCK
) -> torch.Tensor:
    """``_compact(p.T, dtype, device)`` for CSR ``p``, built in row blocks of ``p``.

    Entries keep their values and each row of the result lists its columns in increasing
    order, as in the coalesced transpose, without a COO copy of ``p``: each block's entries
    are stably sorted by column and written after the earlier blocks' entries of that column.
    """
    rows_p, columns = p.shape
    crow, cols, values = p.crow_indices(), p.col_indices(), p.values()
    here = values.device
    index = _index_dtype(columns, rows_p, values.numel())
    out_crow = torch.zeros(columns + 1, dtype=torch.int64, device=here)
    torch.cumsum(torch.bincount(cols, minlength=columns), 0, out=out_crow[1:])
    fill = out_crow[:-1].clone()
    out_cols = torch.empty(values.numel(), dtype=index, device=here)
    out_values = torch.empty(values.numel(), dtype=dtype, device=here)
    for start, stop in _row_spans(crow, block):
        rows, block_cols, vals = _csr_block(p, start, stop)
        block_cols, order = torch.sort(block_cols, stable=True)
        position = torch.arange(block_cols.numel(), device=here)
        first = torch.ones(block_cols.numel(), dtype=torch.bool, device=here)
        first[1:] = block_cols[1:] != block_cols[:-1]
        group = torch.cummax(torch.where(first, position, 0), 0).values
        slot = fill[block_cols] + (position - group)
        out_cols[slot] = rows[order].to(index)
        out_values[slot] = vals[order].to(dtype)
        fill.index_add_(0, block_cols, torch.ones_like(block_cols))
    return _csr_tensor(out_crow.to(device, index), out_cols.to(device), out_values.to(device), (columns, rows_p))


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
    dense: bool = True,
    finish=None,
    block: int = _SETUP_BLOCK,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, float]:
    """Smoothed-aggregation hierarchy built with Torch tensor ops on ``device``.

    Aggregates come from ``passes`` composed pairwise matchings (about 2**passes nodes
    each); the tentative prolongator is smoothed by one damped Jacobi step with weight
    ``4 / (3 lmax)`` and entries below ``truncate`` times their row maximum are dropped
    (row sums kept); coarse operators are Galerkin products. Returns ``[(A_k, P_k)]``
    for every level except the coarsest, the coarsest operator (dense unless ``dense``
    is False, then sparse CSR), and the operator complexity.

    With ``finish``, each ``(A_k, P_k)`` is passed to ``finish(A_k, P_k)`` as soon as
    ``A_{k+1}`` is built and is not kept (the returned list is empty), so at most two
    float64 levels are held at once. Every step works on row blocks of about ``block``
    entries; the result does not depend on ``block``.
    """
    current = _torch_csr(matrix, device, dtype)
    fine_nnz = current.values().numel()
    total_nnz = fine_nnz
    levels: list[tuple[torch.Tensor, torch.Tensor]] = []
    depth = 0
    while current.shape[0] > max_coarse and depth + 1 < max_levels:
        n = current.shape[0]
        aggregates = _aggregate(current, theta, passes, seed=10 * depth, block=block)
        size = int(aggregates.max()) + 1
        if size > 0.9 * n:
            break
        prolong = _smoothed_prolongator(current, aggregates, size, truncate, block)
        del aggregates
        coarse = _galerkin(current, prolong)
        if finish is None:
            levels.append((current, prolong))
        else:
            finish(current, prolong)
        del prolong
        current = coarse
        del coarse
        depth += 1
        total_nnz += current.values().numel()
    return levels, current.to_dense() if dense else current, total_nnz / fine_nnz


class AMGPreconditioner:
    """One symmetric V-cycle of a smoothed-aggregation hierarchy, applied on a device.

    ``matrix`` is the float64 CSR free-node operator (see :func:`assemble_free_operator`).
    Before and after each coarse correction the smoother runs ``smoothing`` steps: a
    degree-``smoothing`` Chebyshev polynomial in ``D^-1 A`` over ``[lmax / 30, lmax]``
    (default), or that many l1-Jacobi sweeps. ``threads`` sets the host threads that
    restrict ``matrix`` to free nodes (default: every CPU this process may use); a
    hierarchy built on the host uses torch's thread count.

    The torch setup hands every level to the solve device as soon as the next one is
    built, so the setup holds the float64 operators of two levels, the compact levels and
    scratch of one row block; the finest operator is compacted from ``matrix`` on the host.
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
        threads: int | None = None,
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
        # Fixed nodes are identity rows with no neighbours: aggregation can never merge them, so
        # a block with fixed faces would keep them all to the dense coarsest level. Build and
        # apply the hierarchy on free nodes only; fixed nodes get a zero correction.
        fixed = mask.detach().reshape(-1).cpu().numpy().astype(bool)
        self.free: torch.Tensor | None = None
        if fixed.any():
            keep = np.flatnonzero(~fixed)
            # An all-fixed block keeps a 1x1 placeholder hierarchy that apply never reaches.
            matrix = _free_submatrix(matrix, keep, threads) if len(keep) else sp.identity(1, format="csr")
            self.free = torch.as_tensor(keep, device=device)
        # (A_k compact or None for the finest operator, P_k, P_k^T, smoother scale) per level.
        built: list[tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.bounds: list[tuple[float, float]] = []

        def finish(a: torch.Tensor, p: torch.Tensor, compacted: torch.Tensor | None) -> None:
            scale = self._smoother_scale(a)
            built.append(
                (compacted, _compact(p, dtype, device), _compact_transpose(p, dtype, device), scale.to(device, dtype))
            )

        if setup == "pyamg":
            try:
                import pyamg
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ValueError("setup='pyamg' requires the optional 'pyamg' package") from exc
            hierarchy = pyamg.smoothed_aggregation_solver(
                matrix, symmetry="symmetric", max_coarse=max_coarse, max_levels=max_levels, keep=False
            )
            for level in hierarchy.levels[:-1]:
                a = _torch_csr(level.A, device, torch.float64)
                finish(a, _torch_csr(level.P, device, torch.float64), _compact(a, dtype, device))
                del a
            coarsest = torch.as_tensor(hierarchy.levels[-1].A.toarray(), dtype=torch.float64, device=device)
            complexity = float(hierarchy.operator_complexity())
        else:

            def finish_torch(a: torch.Tensor, p: torch.Tensor) -> None:
                # The finest operator is compacted from ``matrix`` once its float64 copy is gone.
                finish(a, p, _compact(a, dtype, device) if built else None)

            out_of_memory = False
            try:
                _, coarsest, complexity = _torch_hierarchy(
                    matrix, device, max_coarse=max_coarse, max_levels=max_levels, finish=finish_torch
                )
            except torch.OutOfMemoryError:
                out_of_memory = True
            if out_of_memory:
                # Sparse products need transient workspace beyond the hierarchy; build on
                # the host instead and move only the compact levels. This runs after the
                # except block: until then the traceback keeps the failed attempt's device
                # tensors alive and the cache cannot release them.
                built.clear()
                self.bounds.clear()
                torch.cuda.empty_cache()
                _, coarsest, complexity = _torch_hierarchy(
                    matrix, "cpu", max_coarse=max_coarse, max_levels=max_levels, finish=finish_torch
                )
        self.levels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = [
            (_compact_host(matrix, dtype, device) if a is None else a, p, pt, scale) for a, p, pt, scale in built
        ]
        coarsest = 0.5 * (coarsest + coarsest.T)
        self.coarse = torch.linalg.pinv(coarsest, hermitian=True).to(device, dtype)
        self.complexity = complexity
        self.setup_seconds = time.perf_counter() - started
        self.depth = len(self.levels) + 1
        if not math.isfinite(self.complexity):
            raise ValueError("AMG hierarchy is not finite")

    def _smoother_scale(self, a: torch.Tensor) -> torch.Tensor:
        """The smoother's diagonal scaling for level ``a``; records the Chebyshev interval."""
        if self.smoother == "chebyshev":
            d = _diagonal(a)
            if not bool(torch.isfinite(d).all()) or bool((d <= 0).any()):
                raise ValueError("AMG level has a nonpositive diagonal")
            scale = 1.0 / d
            lmax = _chebyshev_bound(a, scale)
            self.bounds.append((lmax / 30.0, lmax))
            return scale
        l1 = _absolute_row_sums(a)
        if not bool(torch.isfinite(l1).all()) or bool((l1 <= 0).any()):
            raise ValueError("AMG level has a nonpositive l1 row norm")
        return 1.0 / l1

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
        b = residual.reshape(-1)
        if self.free is None:
            out = self._cycle(0, b.to(self.dtype)).to(residual.dtype)
        else:
            out = torch.zeros_like(b)
            if len(self.free):
                out[self.free] = self._cycle(0, b[self.free].to(self.dtype)).to(residual.dtype)
        out = out.reshape(self.shape)
        out[self.mask] = 0
        return out
