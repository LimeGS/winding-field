"""Two-level overlapping Schwarz with a flexible-CG outer iteration for ``solve_field``.

The global problem is the one :func:`field_solver.solve_field` solves (gradient edges, discrete
pairs, exact fixed nodes, zero gauge at node 0 without anchors). It is split into owned cores of
``chunk`` nodes per axis, each solved on a crop widened by ``halo`` with its internal faces held
at the current field, exactly like :func:`block_solver.solve_blocks`. One preconditioner
application is a coarse correction followed by one 8-colour sweep of those local solves; the
outer flexible CG (Polak-Ribiere beta, six global vectors) accelerates the sweeps.

Pairs whose second endpoint lies outside a crop enter the local solve as the exact nodal
penalty ``lam * (phi[p] - field[q])**2`` through ``solve_field``'s reference term, so every
local solve is the exact Dirichlet restriction of the global operator.

This module keeps all global vectors in memory; it is meant for validation up to a few hundred
million nodes and as the numerical core of a later out-of-core driver.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from numpy.typing import ArrayLike, NDArray

from field_amg import assemble_free_operator
from field_solver import _FieldOperator, solve_field

__all__ = ["solve_schwarz", "trilinear_prolongation"]


def _prolongation_1d(n: int, factor: int) -> sp.csr_matrix:
    """Linear interpolation from every ``factor``-th node (and the last) onto ``n`` nodes."""
    coarse = np.unique(np.append(np.arange(0, n, factor), n - 1))
    i = np.arange(n)
    k = np.clip(np.searchsorted(coarse, i, side="right") - 1, 0, max(len(coarse) - 2, 0))
    if len(coarse) == 1:
        return sp.csr_matrix(np.ones((n, 1)))
    a, c = coarse[k], coarse[k + 1]
    t = (i - a) / (c - a)
    rows = np.concatenate([i, i])
    cols = np.concatenate([k, k + 1])
    vals = np.concatenate([1 - t, t])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(coarse)))


def trilinear_prolongation(shape: tuple[int, int, int], factor: int) -> sp.csr_matrix:
    """Return the ZYX C-order trilinear prolongation from a ``factor``-coarsened grid."""
    pz, py, px = (_prolongation_1d(int(n), int(factor)) for n in shape)
    return sp.kron(pz, sp.kron(py, px), format="csr")


def _scipy(a: torch.Tensor) -> sp.csr_matrix:
    a = a.to_sparse_csr().cpu()
    return sp.csr_matrix(
        (a.values().double().numpy(), a.col_indices().numpy(), a.crow_indices().numpy()), shape=tuple(a.shape)
    )


class _Coarse:
    """Galerkin coarse space ``P^T A P`` on free nodes, solved directly or by AMG-PCG.

    ``kind='trilinear'`` interpolates from every ``factor``-th node. ``kind='aggregation'``
    composes smoothed-aggregation prolongators (as in :mod:`field_amg`) until the space has
    about ``n / factor**3`` columns; it follows strong pair couplings that a geometric
    space cannot represent.
    """

    def __init__(
        self, matrix: sp.csr_matrix, shape, mask: NDArray[np.bool_], factor: int, direct_limit: int, kind: str
    ):
        if kind == "trilinear":
            p = trilinear_prolongation(shape, factor)
            p = sp.diags((~mask.ravel()).astype(np.float64)) @ p
            p = p.tocsc()
            keep = np.flatnonzero(np.diff(p.indptr))
            self.p = p[:, keep].tocsr()
            coarse = (self.p.T @ matrix @ self.p).tocsr()
        elif kind == "aggregation":
            from field_amg import _torch_hierarchy

            target = max(1, matrix.shape[0] // factor**3)
            levels, last, _ = _torch_hierarchy(matrix, "cpu", max_coarse=target, max_levels=32, dense=False)
            p = sp.identity(matrix.shape[0], format="csr")
            for _, prolong in levels:
                p = (p @ _scipy(prolong)).tocsr()
            p = sp.diags((~mask.ravel()).astype(np.float64)) @ p
            self.p = p.tocsr()
            coarse = _scipy(last) if levels else matrix
        else:
            raise ValueError("coarse kind must be trilinear or aggregation")
        self.pt = self.p.T.tocsr()
        coarse = 0.5 * (coarse + coarse.T)
        self.size = coarse.shape[0]
        self.nnz = int(coarse.nnz)
        if self.size <= direct_limit:
            self._solve: Callable[[np.ndarray], np.ndarray] = spla.factorized(coarse.tocsc())
            self.method = "direct"
        else:
            from field_amg import AMGPreconditioner

            amg = AMGPreconditioner(coarse, (self.size,), torch.zeros(self.size, dtype=torch.bool))
            self.method = "amg-pcg"

            def solve(rhs: np.ndarray) -> np.ndarray:
                def apply(v):
                    return amg.apply(torch.as_tensor(v)).numpy()

                op = spla.LinearOperator(coarse.shape, matvec=apply, dtype=np.float64)
                out, _ = spla.cg(coarse, rhs, M=op, rtol=1e-8, maxiter=500)
                return out

            self._solve = solve

    def correct(self, residual: np.ndarray) -> np.ndarray:
        return self.p @ self._solve(self.pt @ residual)


def _tiles(shape, chunk: int):
    grid = [math.ceil(n / chunk) for n in shape]
    for key in itertools.product(*(range(n) for n in grid)):
        low = np.array(key) * chunk
        yield key, low, np.minimum(low + chunk, shape)


def solve_schwarz(
    target_xyz: ArrayLike,
    weights: ArrayLike,
    spacing_xyz: ArrayLike = (1, 1, 1),
    pairs: ArrayLike | None = None,
    lam: float = 3.0,
    fixed: ArrayLike | None = None,
    *,
    chunk: int,
    halo: int,
    coarse_factor: int | None = 8,
    coarse_kind: Literal["trilinear", "aggregation"] = "trilinear",
    rtol: float = 1e-6,
    maxiter: int = 50,
    local_rtol: float | None = None,
    local_maxiter: int = 4000,
    local_preconditioner: Literal["jacobi", "amg"] = "amg",
    device: str = "cpu",
    initial: ArrayLike | None = None,
    direct_coarse_limit: int = 200_000,
    restricted: bool = False,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Solve the ``solve_field`` system by two-level Schwarz-preconditioned flexible CG.

    Arguments mirror :func:`field_solver.solve_field` (float64 only; no continuous constraint
    rows or reference penalty yet). ``chunk`` is the owned core per axis and ``halo`` the crop
    overlap on each side; ``coarse_factor=None`` disables the coarse level (one-level Schwarz).
    ``rtol`` bounds the free-node residual relative to the reduced right-hand side; local
    solves default to ``rtol / 100``. Each outer iteration costs one sweep of local solves
    and one global operator application. Returns the field and a JSON-compatible receipt;
    ``converged`` is False when ``maxiter`` outer iterations do not reach ``rtol``.
    """
    started = time.perf_counter()
    if type(chunk) is not int or chunk < 1 or type(halo) is not int or halo < 0:
        raise ValueError("chunk must be a positive integer and halo a nonnegative integer")
    if coarse_factor is not None and (type(coarse_factor) is not int or coarse_factor < 2):
        raise ValueError("coarse_factor must be an integer of at least 2 or None")
    if type(maxiter) is not int or maxiter < 0:
        raise ValueError("maxiter must be a nonnegative integer")
    if not (isinstance(rtol, (int, float)) and math.isfinite(rtol) and rtol > 0):
        raise ValueError("rtol must be positive and finite")
    local_rtol = rtol / 100 if local_rtol is None else local_rtol
    target = np.asarray(target_xyz, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if target.ndim != 4 or target.shape[-1] != 3 or target.shape[:3] != weight.shape:
        raise ValueError("target_xyz must be (Z, Y, X, 3) matching weights")
    shape = weight.shape
    if halo == 0 and any(n > chunk for n in shape):
        raise ValueError("a positive halo is required between multiple owned cores")
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    n = weight.size
    pair_array = np.zeros((0, 2), np.int64) if pairs is None else np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if ((pair_array < 0) | (pair_array >= n)).any():
        raise ValueError("pair indices are outside the flat ZYX field")
    pair_array = pair_array[pair_array[:, 0] != pair_array[:, 1]]
    if not lam:
        pair_array = pair_array[:0]

    fixed_values = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    if fixed is not None:
        declared = np.asarray(fixed, dtype=np.float64)
        if declared.shape != shape or np.isinf(declared).any():
            raise ValueError("fixed must be a finite-or-NaN ZYX array matching weights")
        mask = np.isfinite(declared)
        fixed_values[mask] = declared[mask]
    gauge: dict[str, Any] = dict(kind="fixed", count=int(mask.sum()))
    if not mask.any():
        mask.flat[0] = True  # The global zero gauge stays fixed in every local solve.
        gauge = dict(kind="zero", flat_index=0, value=0.0)

    # Global free-node operator and reduced right-hand side, as solve_field assembles them.
    w_t = torch.as_tensor(weight)
    pairs_t = torch.as_tensor(pair_array) if len(pair_array) else None
    operator = _FieldOperator(w_t, pairs_t, float(lam))
    mask_t = torch.as_tensor(mask)
    b_t = operator.rhs(torch.as_tensor(target), spacing.tolist())
    b_t -= operator.matvec(torch.as_tensor(fixed_values))
    b_t[mask_t] = 0
    b = b_t.numpy().ravel()

    def matvec(v: np.ndarray) -> np.ndarray:
        out = operator.matvec(torch.as_tensor(v.reshape(shape))).numpy().ravel()
        out[mask.ravel()] = 0
        return out

    scale = float(np.linalg.norm(b)) or 1.0
    setup_started = time.perf_counter()
    coarse = None
    if coarse_factor is not None:
        matrix = assemble_free_operator(weight, mask, pairs=pair_array if len(pair_array) else None, lam=float(lam))
        coarse = _Coarse(matrix, shape, mask, coarse_factor, direct_coarse_limit, coarse_kind)
        del matrix
    coarse_seconds = time.perf_counter() - setup_started

    tiles = list(_tiles(shape, chunk))
    shape_a = np.array(shape)
    flat_index = np.arange(n).reshape(shape)
    stats = dict(local_solves=0, local_iterations=0, local_seconds=0.0, nonconverged=0)
    endpoints = [np.stack(np.unravel_index(pair_array[:, j], shape), axis=1) for j in (0, 1)]
    position = np.full(n, -1, np.int64)

    def sweep(field: np.ndarray) -> np.ndarray:
        """One 8-colour sweep of exact-Dirichlet local solves, in place on the full field."""
        for colour in range(8):
            for key, low, high in tiles:
                if (key[0] % 2) * 4 + (key[1] % 2) * 2 + key[2] % 2 != colour:
                    continue
                lo, hi = np.maximum(low - halo, 0), np.minimum(high + halo, shape_a)
                crop = tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))
                crop_shape = tuple(int(c - a) for a, c in zip(lo, hi))
                current = field[crop]
                local_fixed = np.where(mask[crop], fixed_values[crop], np.nan)
                for axis in range(3):
                    for side, internal in ((0, lo[axis] > 0), (-1, hi[axis] < shape[axis])):
                        if internal:
                            plane: list[slice | int] = [slice(None)] * 3
                            plane[axis] = side
                            local_fixed[tuple(plane)] = current[tuple(plane)]
                local_pairs = None
                reference = None
                reference_weight: Any = 0.0
                if len(pair_array):
                    inside = [np.all((c >= lo) & (c < hi), axis=1) for c in endpoints]
                    local_index = flat_index[crop].ravel()
                    position[local_index] = np.arange(local_index.size)
                    both = inside[0] & inside[1]
                    local_pairs = position[pair_array[both]]
                    # A pair leaving the crop is the exact nodal penalty lam*(phi[p]-field[q])**2.
                    rw = np.zeros(crop_shape)
                    numerator = np.zeros(crop_shape)
                    flat_field = field.ravel()
                    for here, there in ((0, 1), (1, 0)):
                        cut = inside[here] & ~inside[there]
                        if cut.any():
                            at = position[pair_array[cut, here]]
                            np.add.at(rw.reshape(-1), at, lam)
                            np.add.at(numerator.reshape(-1), at, lam * flat_field[pair_array[cut, there]])
                    if rw.any():
                        reference = np.divide(numerator, rw, out=np.zeros(crop_shape), where=rw > 0)
                        reference_weight = rw
                tic = time.perf_counter()
                result, info = solve_field(
                    target[crop],
                    weight[crop],
                    spacing,
                    pairs=local_pairs,
                    lam=lam,
                    fixed=local_fixed,
                    reference=reference,
                    reference_weight=reference_weight,
                    initial=current,
                    device=device,
                    dtype="float64",
                    rtol=local_rtol,
                    maxiter=local_maxiter,
                    preconditioner=local_preconditioner,
                )
                stats["local_seconds"] += time.perf_counter() - tic
                stats["local_solves"] += 1
                stats["local_iterations"] += int(info["iterations"])
                if not info["converged"]:
                    stats["nonconverged"] += 1
                if restricted:
                    owned = tuple(slice(int(a - c), int(d - c)) for a, d, c in zip(low, high, lo))
                    field[tuple(slice(int(a), int(d)) for a, d in zip(low, high))] = np.asarray(result)[owned]
                else:
                    # Same-colour crops are disjoint, so the whole crop minimiser is kept: every
                    # local solve then lowers the global energy, which RAS does not guarantee.
                    field[crop] = np.asarray(result)
        return field

    def precondition(x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """z = S(x) - x, with S = coarse correction of r followed by one sweep."""
        trial = x + (coarse.correct(r) if coarse is not None else 0.0)
        trial[mask.ravel()] = 0
        field = (trial.reshape(shape) + fixed_values).copy()
        field = sweep(field)
        z = field.ravel() - fixed_values.ravel() - x
        z[mask.ravel()] = 0
        return z

    x = np.zeros(n) if initial is None else np.asarray(initial, dtype=np.float64).reshape(-1) - fixed_values.ravel()
    x[mask.ravel()] = 0
    r = b - matvec(x)
    history = [float(np.linalg.norm(r)) / scale]
    reason = "maxiter"
    iterations = 0
    sweeps = 0
    if history[-1] <= rtol:
        reason = "converged"
    else:
        z = precondition(x, r)
        sweeps += 1
        p = z.copy()
        rz = float(r @ z)
        for _ in range(maxiter):
            q = matvec(p)
            curvature = float(p @ q)
            if not math.isfinite(curvature) or curvature <= 0 or not math.isfinite(rz):
                reason = "breakdown"
                break
            alpha = rz / curvature
            x += alpha * p
            r_old = r
            r = r - alpha * q
            iterations += 1
            history.append(float(np.linalg.norm(r)) / scale)
            if callback is not None:
                callback(dict(iteration=iterations, relative_residual=history[-1]))
            if history[-1] <= rtol:
                reason = "converged"
                break
            z = precondition(x, r)
            sweeps += 1
            next_rz = float(r @ z)
            beta = float(z @ (r - r_old)) / rz
            p = z + beta * p
            rz = next_rz
    residual = float(np.linalg.norm(b - matvec(x))) / scale
    converged = reason == "converged" and residual <= 10 * rtol
    field = (x.reshape(shape) + fixed_values).copy()
    field[mask] = fixed_values[mask]
    receipt = dict(
        converged=converged,
        reason="converged" if converged else reason,
        iterations=iterations,
        sweeps=sweeps,
        relative_residual=residual,
        residual_history=history,
        gauge=gauge,
        tiles=len(tiles),
        chunk=chunk,
        halo=halo,
        coarse=None
        if coarse is None
        else dict(
            kind=coarse_kind,
            factor=coarse_factor,
            size=coarse.size,
            nnz=coarse.nnz,
            method=coarse.method,
            setup_s=coarse_seconds,
        ),
        local=dict(stats, preconditioner=local_preconditioner, rtol=local_rtol),
        elapsed_s=time.perf_counter() - started,
    )
    return field, receipt
