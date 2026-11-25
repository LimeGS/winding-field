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
        self.matrix = sp.csr_matrix((self.size, self.size))
        self.assembled = False
        self.solve: Callable[[np.ndarray], np.ndarray] | None = None
        self.info: dict | None = None

    def rows(self, zyx: np.ndarray, free: np.ndarray | None = None) -> sp.csr_matrix:
        """Prolongation rows for global ZYX nodes ``zyx`` (N x 3); rows of fixed nodes are zero."""
        n = len(zyx)
        parts = [_axis_weights(self.axes[d], zyx[:, d]) for d in range(3)]
        cz, cy, cx = self.coarse_shape
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
        Edge rows are processed in z-slabs of the core so memory stays bounded by the slab.
        """
        crop_shape = weights.shape
        core = [slice(int(a - b), int(c - b)) for a, b, c in zip(low, lo, high)]
        w = weights.astype(np.float64)
        acc = sp.csr_matrix((self.size, self.size))
        z_start, z_stop = core[0].start, core[0].stop
        for z0 in range(z_start, z_stop, slab):
            z1 = min(z0 + slab, z_stop)
            zz, yy, xx = np.meshgrid(
                np.arange(z0, z1),
                np.arange(core[1].start, core[1].stop),
                np.arange(core[2].start, core[2].stop),
                indexing="ij",
            )
            local = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1)
            keep = free[tuple(local.T)]
            local = local[keep]
            if not len(local):
                continue
            p_i = self.rows(local + lo)
            for axis in range(3):
                for step in (-1, 1):
                    nb = local.copy()
                    nb[:, axis] += step
                    inside = (nb[:, axis] >= 0) & (nb[:, axis] < crop_shape[axis])
                    if not inside.any():
                        continue
                    src, dst = local[inside], nb[inside]
                    edge = 0.5 * w[tuple(src.T)] + 0.5 * w[tuple(dst.T)]
                    pi = p_i[np.flatnonzero(inside)]
                    pj = self.rows(dst + lo, free[tuple(dst.T)])
                    acc += pi.T @ sp.diags(edge) @ (pi - pj)
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
                acc += mpo.T @ centered
        self.matrix = (self.matrix + acc).tocsr()

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
                out, _ = spla.cg(reduced, rhs, M=op, rtol=1e-8, maxiter=500)
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
