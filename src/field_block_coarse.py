"""Out-of-core trilinear coarse space for :func:`block_solver.solve_blocks`.

Two-level multiplicative Schwarz: before each sweep of local block solves, the global residual
``r = b - A x`` is gathered block by block, restricted with the trilinear prolongation ``P``,
the coarse system ``P^T A P e = P^T r`` is solved in memory and ``P e`` is added to every owned
core. Only the coarse grid (``factor**3`` times smaller) is ever held in memory.

Every block contributes the rows of ``A`` that belong to its owned core, so the sum over blocks
is the Galerkin product ``P^T A P`` of the global free-node operator. Constraint rows that leave
a crop arrive from the caller's provider with their exterior terms moved into targets; their
coarse coupling across the crop face is then dropped, which only weakens the correction. The
residual itself is exact: core rows see every neighbour (``halo >= 1``) and the exterior values
of cut rows are the current field.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from field_solver import _constraint_operator, _FieldOperator

__all__ = ["CoarseSpace", "coarse_axes"]


def coarse_axes(shape, factor: int) -> list[np.ndarray]:
    """Fine-grid positions of the coarse nodes along each ZYX axis (every ``factor``-th + last)."""
    return [np.unique(np.append(np.arange(0, int(n), factor), int(n) - 1)) for n in shape]


def _axis_weights(coarse: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation of fine positions ``i`` from ``coarse``: (left index, right index, t)."""
    if len(coarse) == 1:
        zero = np.zeros(len(i), np.int64)
        return zero, zero, np.zeros(len(i))
    k = np.clip(np.searchsorted(coarse, i, side="right") - 1, 0, len(coarse) - 2)
    a, c = coarse[k], coarse[k + 1]
    return k, k + 1, (i - a) / (c - a)


class CoarseSpace:
    """Trilinear prolongation rows, the accumulated Galerkin operator and its solve."""

    def __init__(self, shape, factor: int):
        self.shape = tuple(int(n) for n in shape)
        self.factor = int(factor)
        self.axes = coarse_axes(self.shape, self.factor)
        self.coarse_shape = tuple(len(a) for a in self.axes)
        self.size = int(np.prod(self.coarse_shape))
        self._matrix = sp.csr_matrix((self.size, self.size))
        # Grid part of the operator as a 27-point stencil (row node, column offset + 1 per axis)
        # and constraint parts waiting to be summed; both are folded into ``matrix`` on read.
        self._stencil: np.ndarray | None = None
        self._pending: list[sp.coo_matrix] = []
        self._pending_nnz = 0
        self.assembled = False
        self.solve: Callable[[np.ndarray], np.ndarray] | None = None
        self.last: dict | None = None  # Iterations and convergence of the last iterative solve.
        self.info: dict | None = None

    @property
    def matrix(self) -> sp.csr_matrix:
        """The accumulated coarse operator."""
        self._fold_pairs()
        if self._stencil is not None:
            flat = self._stencil.reshape(self.size, 27)
            self._stencil = None
            _, cy, cx = self.coarse_shape
            by_offset: dict[int, np.ndarray] = {}
            for index, (oz, oy, ox) in enumerate(np.ndindex(3, 3, 3)):
                offset = ((oz - 1) * cy + (oy - 1)) * cx + (ox - 1)
                values = flat[:, index]
                if abs(offset) >= self.size or not values.any():
                    continue
                diagonal = values[: self.size - offset] if offset >= 0 else values[-offset:]
                # Offsets coincide when an axis has few coarse nodes.
                by_offset[offset] = by_offset[offset] + diagonal if offset in by_offset else diagonal
            offsets = sorted(by_offset)
            diagonals = [by_offset[k] for k in offsets]
            if diagonals:
                grid = sp.diags(diagonals, offsets, shape=(self.size, self.size), format="csr")
                grid.eliminate_zeros()
                self._matrix = (self._matrix + grid).tocsr()
        return self._matrix

    @matrix.setter
    def matrix(self, value) -> None:
        self._matrix, self._stencil, self._pending, self._pending_nnz = value, None, [], 0

    def _fold_pairs(self) -> None:
        if self._pending:
            total = self._matrix
            for part in self._pending:
                total = total + part
            self._matrix, self._pending, self._pending_nnz = total.tocsr(), [], 0

    def _pair_table(self, d: int, lo: int, hi: int, shift: int) -> tuple[np.ndarray, int]:
        """Per-axis products of interpolation weights for the nodes ``[lo, hi)`` of axis ``d``.

        Entry ``[i - lo, 3 * (a - first) + (b - a + 1)]`` is ``phi_a(i) * phi_b(i + shift)``: the
        weight of coarse node ``a`` at fine node ``i`` times that of coarse node ``b`` at its
        neighbour. A node and its neighbour lie in one closed coarse cell, so ``b - a`` is -1, 0
        or 1. Rows whose neighbour is outside the axis are zero.
        """
        i = np.arange(int(lo), int(hi))
        j = i + shift
        valid = (j >= 0) & (j < self.shape[d])
        ka0, ka1, ta = _axis_weights(self.axes[d], i)
        kb0, kb1, tb = _axis_weights(self.axes[d], np.clip(j, 0, self.shape[d] - 1))
        first = int(ka0.min())
        table = np.zeros((len(i), 3 * (int(ka1.max()) + 1 - first)))
        r = np.arange(len(i))
        for ka, wa in ((ka0, 1 - ta), (ka1, ta)):
            for kb, wb in ((kb0, 1 - tb), (kb1, tb)):
                value = wa * wb * valid
                use = value != 0
                if (abs(kb[use] - ka[use]) > 1).any():
                    raise AssertionError("coarse neighbours are more than one node apart")
                np.add.at(table, (r[use], 3 * (ka[use] - first) + (kb[use] - ka[use] + 1)), value[use])
        return table, first

    def _add_stencil(self, window, values: np.ndarray, tables, sign: float) -> None:
        """Add ``sign * sum_i values[i] p_i^T p_j`` over a box to the stencil of its window."""
        (tz, _), (ty, _), (tx, _) = tables
        v = np.tensordot(tz.T, values, axes=(1, 0))
        v = np.tensordot(v, ty, axes=(1, 0))  # (z pairs, x, y pairs)
        v = np.tensordot(v, tx, axes=(1, 0))  # (z pairs, y pairs, x pairs)
        nz, ny, nx = (t[0].shape[1] // 3 for t in tables)
        v = v.reshape(nz, 3, ny, 3, nx, 3).transpose(0, 2, 4, 1, 3, 5)
        assert self._stencil is not None
        if sign > 0:
            self._stencil[window] += v
        else:
            self._stencil[window] -= v

    def rows(self, zyx: np.ndarray, free: np.ndarray | None = None) -> sp.csr_matrix:
        """Prolongation rows for global ZYX nodes ``zyx`` (N x 3); rows of fixed nodes are zero."""
        n = len(zyx)
        parts = [_axis_weights(self.axes[d], zyx[:, d]) for d in range(3)]
        _, cy, cx = self.coarse_shape
        cols, vals = [], []
        for bz in (0, 1):
            for by in (0, 1):
                for bx in (0, 1):
                    kz, ky, kx = (parts[d][bit] for d, bit in zip(range(3), (bz, by, bx)))
                    wz = parts[0][2] if bz else 1 - parts[0][2]
                    wy = parts[1][2] if by else 1 - parts[1][2]
                    wx = parts[2][2] if bx else 1 - parts[2][2]
                    cols.append((kz * cy + ky) * cx + kx)
                    vals.append(wz * wy * wx)
        col = np.concatenate(cols)
        val = np.concatenate(vals)
        if free is not None:
            val = val * np.tile(free.astype(np.float64), 8)
        row = np.tile(np.arange(n), 8)
        out = sp.csr_matrix((val, (row, col)), shape=(n, self.size))
        out.eliminate_zeros()
        return out

    def _axis_matrices(self, lo, hi) -> tuple[list[np.ndarray], tuple[slice, ...]]:
        """Dense per-axis interpolation matrices of the box ``[lo, hi)`` and its coarse window.

        The trilinear prolongation is a tensor product, so a box needs three small matrices
        (fine nodes x coarse nodes of the window) instead of one sparse row per node.
        """
        mats, window = [], []
        for d in range(3):
            i = np.arange(int(lo[d]), int(hi[d]))
            k0, k1, t = _axis_weights(self.axes[d], i)
            first, stop = int(k0.min()), int(k1.max()) + 1
            m = np.zeros((len(i), stop - first))
            r = np.arange(len(i))
            np.add.at(m, (r, k0 - first), 1 - t)
            np.add.at(m, (r, k1 - first), t)
            mats.append(m)
            window.append(slice(first, stop))
        return mats, tuple(window)

    def restrict_box(self, lo, hi, values: np.ndarray, free: np.ndarray, out: np.ndarray) -> None:
        """Add ``P^T values`` of the box ``[lo, hi)`` to the flat coarse vector ``out``.

        Equal to ``rows(nodes, free).T @ values`` for the nodes of the box, up to rounding.
        """
        window, v = self.restrict_box_part(lo, hi, values, free)
        out.reshape(self.coarse_shape)[window] += v

    def restrict_box_part(self, lo, hi, values: np.ndarray, free: np.ndarray) -> tuple[tuple[slice, ...], np.ndarray]:
        """Return ``(window, P^T values)`` of the box ``[lo, hi)`` on its coarse window.

        Pure (reads no mutable state), so threads may call it; adding the parts to the coarse
        vector in a fixed order gives the same bytes as ``restrict_box`` in that order.
        """
        (mz, my, mx), window = self._axis_matrices(lo, hi)
        v = np.where(free, values, 0.0)
        v = np.tensordot(mz.T, v, axes=(1, 0))
        v = np.tensordot(v, my, axes=(1, 0))  # axes are now (cz, x, cy)
        v = np.tensordot(v, mx, axes=(1, 0))  # (cz, cy, cx)
        return window, v

    def prolong_box(self, lo, hi, coarse: np.ndarray) -> np.ndarray:
        """Return ``P coarse`` on the box ``[lo, hi)``; the caller zeroes fixed nodes.

        Equal to ``rows(nodes) @ coarse`` for the nodes of the box, up to rounding.
        """
        (mz, my, mx), window = self._axis_matrices(lo, hi)
        v = coarse.reshape(self.coarse_shape)[window]
        v = np.tensordot(mz, v, axes=(1, 0))  # (z, cy, cx)
        v = np.tensordot(v, my.T, axes=(1, 0))  # (z, cx, y)
        v = np.tensordot(v, mx.T, axes=(1, 0))  # (z, y, x)
        return v

    def add_block(
        self,
        lo: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        weights: np.ndarray,
        free: np.ndarray,
        rows,
        slab: int = 8,
        exterior=None,
    ) -> None:
        """Accumulate ``P_core^T A[core, crop] P_crop`` for one block crop starting at ``lo``.

        ``weights`` and ``free`` cover the crop; ``rows`` are the provider's local constraints and
        ``exterior`` the ``(row, global flat node, free, coefficient)`` terms the provider moved
        into their targets. With them the constraint rows are exact; without, cut rows lose their
        coupling across the crop face.
        The grid part is accumulated as a 27-point stencil with per-axis weight tables (the
        prolongation is a tensor product); ``slab`` is unused and kept for callers.
        """
        crop_shape = weights.shape
        core = [slice(int(a - b), int(c - b)) for a, b, c in zip(low, lo, high)]
        w = weights.astype(np.float64)
        owned_free = free[tuple(core)]
        if self._stencil is None:
            self._stencil = np.zeros((*self.coarse_shape, 3, 3, 3))
        same = [self._pair_table(d, low[d], high[d], 0) for d in range(3)]
        window = tuple(slice(t[1], t[1] + t[0].shape[1] // 3) for t in same)
        diagonal = np.zeros(owned_free.shape)
        for axis in range(3):
            for step in (-1, 1):
                # Neighbour j = i + step along axis, for every core node i whose j is in the crop.
                near = [slice(c.start, c.stop) for c in core]
                first, stop = near[axis].start + step, near[axis].stop + step
                clip = (max(-first, 0), max(stop - crop_shape[axis], 0))
                near[axis] = slice(first + clip[0], stop - clip[1])
                span = [slice(None)] * 3
                span[axis] = slice(clip[0], owned_free.shape[axis] - clip[1])
                part = tuple(span)
                edge = np.zeros(owned_free.shape)
                edge[part] = (0.5 * w[tuple(core)][part] + 0.5 * w[tuple(near)]) * owned_free[part]
                diagonal += edge
                edge[part] *= free[tuple(near)]
                tables = list(same)
                tables[axis] = self._pair_table(axis, low[axis], high[axis], step)
                self._add_stencil(window, edge, tables, -1.0)
        self._add_stencil(window, diagonal, same, 1.0)
        acc = None
        if rows is not None:
            matrix = rows.matrix.tocsr()
            members = rows.members
            if matrix.shape[1] != int(np.prod(crop_shape)):
                raise ValueError("constraint rows do not match the block crop")
            touched = np.unique(matrix.indices)
            if len(touched):
                zyx = np.stack(np.unravel_index(touched, crop_shape), 1)
                p_touched = self.rows(zyx + lo, free.ravel()[touched])
                m = matrix[:, touched]
                mp = m @ p_touched
                if exterior is not None and len(exterior[0]):
                    row, node, node_free, coeff = exterior
                    ext_zyx = np.stack(np.unravel_index(np.asarray(node, np.int64), self.shape), 1)
                    m_ext = sp.csr_matrix(
                        (np.asarray(coeff, np.float64), (np.asarray(row), np.arange(len(node)))),
                        shape=(matrix.shape[0], len(node)),
                    )
                    mp = mp + m_ext @ self.rows(ext_zyx, np.asarray(node_free, bool))
                centered = sp.diags(rows.weights) @ mp
                if members.shape[0]:
                    inverse = np.divide(1, rows.group_sum, out=np.zeros(len(rows.group_sum)), where=rows.group_sum > 0)
                    centered = centered - members.T @ (sp.diags(inverse) @ (members @ mp))
                # Rows of M^T C M P that belong to free core nodes: (M D_owned P)^T C M P.
                owned = np.all((zyx >= [c.start for c in core]) & (zyx < [c.stop for c in core]), axis=1)
                mpo = m @ (sp.diags(owned.astype(np.float64)) @ p_touched)
                acc = (mpo.T @ centered).tocoo()
        if acc is not None and acc.nnz:
            self._pending.append(acc)
            self._pending_nnz += acc.nnz
            if self._pending_nnz > 50_000_000:
                self._fold_pairs()

    def solver(self, direct_limit: int) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
        """Factor (or AMG-precondition) the symmetrized operator on its nonempty columns."""
        started = time.perf_counter()
        a = (0.5 * (self.matrix + self.matrix.T)).tocsr()
        active = np.flatnonzero(a.diagonal() > 0)
        reduced = a[active][:, active].tocsc()
        if len(active) <= direct_limit:
            factor = spla.factorized(reduced)
            method = "direct"

            def inner(rhs):
                return factor(rhs)

        else:
            from field_amg import AMGPreconditioner

            amg = AMGPreconditioner(reduced.tocsr(), (len(active),), torch.zeros(len(active), dtype=torch.bool))
            method = "amg-pcg"
            op = spla.LinearOperator(
                reduced.shape, matvec=lambda v: amg.apply(torch.as_tensor(v)).numpy(), dtype=np.float64
            )

            def inner(rhs):
                count = [0]

                def step(_):
                    count[0] += 1

                out, code = spla.cg(reduced, rhs, M=op, rtol=1e-8, maxiter=500, callback=step)
                if code < 0 or not np.isfinite(out).all():
                    raise RuntimeError("coarse solve broke down")
                # A correction short of the tolerance still reduces the error; report it.
                self.last = dict(iterations=count[0], converged=code == 0)
                return out

        def solve(rhs: np.ndarray) -> np.ndarray:
            out = np.zeros(self.size)
            if len(active):
                out[active] = inner(rhs[active])
            return out

        info = dict(size=int(len(active)), nnz=int(reduced.nnz), method=method, setup_s=time.perf_counter() - started)
        return solve, info


def block_residual(target, weights, spacing, rows, values, fixed_mask, device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(b - A x, b - A x_fixed)`` on the crop in float64; fixed rows are zero.

    ``values`` hold the current field with exact fixed values already substituted. Only rows of
    the owned core are meaningful; crop faces miss neighbours outside the crop.
    """
    # Copies: cached observations may be read-only, which torch cannot wrap.
    w = torch.as_tensor(np.array(weights, dtype=np.float64), device=device)
    g = torch.as_tensor(np.array(target, dtype=np.float64), device=device)
    x = torch.as_tensor(values, dtype=torch.float64, device=device)
    mask = torch.as_tensor(fixed_mask, device=device)
    operator = _FieldOperator(w, None, 0.0)
    extra = _constraint_operator(rows, device, np.float64)
    b = operator.rhs(g, [float(s) for s in spacing])
    if extra is not None:
        b += extra.rhs().reshape(b.shape)

    def apply(v):
        out = operator.matvec(v)
        if extra is not None:
            out += extra.matvec(v)
        return out

    r = b - apply(x)
    r[mask] = 0
    anchored = torch.where(mask, x, torch.zeros_like(x))
    r0 = b - apply(anchored)
    r0[mask] = 0
    return r.cpu().numpy(), r0.cpu().numpy()
