"""Fused Triton kernels for the PCG loop of ``field_solver.solve_field`` (``kernel="fused"``).

The eager loop launches ~40 PyTorch kernels per iteration and moves ~490 bytes per node in
float64; the solve is memory-bandwidth bound on every GPU measured (GTX 1660, A100). Three
kernels replace them:

- K1 ``_stencil``: the 7-point weighted Laplacian with on-the-fly edge weights, plus the
  reference penalty, the continuous constraint rows and the fixed-node mask, writing ``ap``
  and one partial ``p.ap`` sum per program. CSR rows (fibres) are gathered in K1 as A^T v,
  with v = center_weight(A p) formed by the ``_gather`` kernel; other operators come in as a
  precomputed term.
- K2 ``_update``: ``x += alpha p``, ``r -= alpha ap`` and, for Jacobi, ``z = r * inv`` without
  storing it, with partial ``r.z`` and ``r.r`` sums.
- K3 ``_direction``: ``p = p * beta + z`` (``z`` recomputed as ``r * inv`` for Jacobi).

Partial sums are reduced in a fixed order (no atomics), so one GPU and one problem give the
same bytes on every run. Floating-point contraction is disabled so each operation rounds
like the eager PyTorch ops it replaces; results still differ from eager at round-off level
because the six fluxes of a node are summed in another order. The caller keeps measuring the
final residual with the eager operator, so the convergence gate does not depend on this code.

Kernels are compiled on first use. ``TRITON_INTERPRET=1`` runs them on CPU tensors (tests).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import torch

KERNEL_VERSION = "fused-v2"
KERNELS = ("eager", "fused")

# Program tiles. K1 covers BLOCK_Y x BLOCK_X nodes of one z-plane: x rows stay coalesced and
# y/z neighbours come from the cache. K2/K3 are flat.
BLOCK_Y = 8
BLOCK_X = 32
BLOCK_FLAT = 1024

_compiled: dict[str, Any] | None = None


def _interpreted() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


def unavailable_reason(device: torch.device, preconditioner: str, has_pairs: bool) -> str | None:
    """Why the fused loop cannot run this solve, or None when it can."""
    if preconditioner not in ("jacobi", "amg"):
        return f"preconditioner={preconditioner} is not fused"
    if has_pairs:
        return "discrete pairs are not fused"
    if device.type != "cuda" and not (device.type == "cpu" and _interpreted()):
        return "fused kernels need a CUDA device"
    try:
        _kernels()
    except ImportError:
        return "triton is not installed"
    return None


def _kernels() -> dict[str, Any]:
    """Define the Triton kernels once (lazily, so importing this module never needs Triton)."""
    global _compiled
    if _compiled is not None:
        return _compiled
    import triton
    import triton.language as tl

    @triton.jit
    def _stencil(
        p_ptr,
        w_ptr,
        rw_ptr,
        extra_ptr,
        at_rowptr_ptr,
        at_cols_ptr,
        at_vals_ptr,
        v_ptr,
        mask_ptr,
        ap_ptr,
        part_ptr,
        Z,
        Y,
        X,
        RW_MODE: tl.constexpr,  # 0 none, 1 scalar (rw_ptr[0], float64), 2 nodal
        HAS_EXTRA: tl.constexpr,
        HAS_ROWS: tl.constexpr,
        BY: tl.constexpr,
        BX: tl.constexpr,
    ):
        z = tl.program_id(0)
        yb = tl.program_id(1)
        xb = tl.program_id(2)
        ys = yb * BY + tl.arange(0, BY)
        xs = xb * BX + tl.arange(0, BX)
        inside = (ys[:, None] < Y) & (xs[None, :] < X)
        plane = Y * X
        idx = z.to(tl.int64) * plane + ys[:, None].to(tl.int64) * X + xs[None, :]
        pc = tl.load(p_ptr + idx, mask=inside, other=0.0)
        wc = tl.load(w_ptr + idx, mask=inside, other=0.0)
        out = tl.zeros([BY, BX], dtype=p_ptr.dtype.element_ty)
        # Same per-node order as _FieldOperator.matvec: for each axis (z, y, x) the flux to the
        # upper neighbour is subtracted, then the flux from the lower one is added.
        # Axis z.
        zs = z + tl.zeros([BY, BX], dtype=tl.int32)
        up = inside & (zs < Z - 1)
        pn = tl.load(p_ptr + idx + plane, mask=up, other=0.0)
        wn = tl.load(w_ptr + idx + plane, mask=up, other=0.0)
        out = tl.where(up, out - (pn - pc) * (0.5 * wc + 0.5 * wn), out)
        down = inside & (zs > 0)
        pn = tl.load(p_ptr + idx - plane, mask=down, other=0.0)
        wn = tl.load(w_ptr + idx - plane, mask=down, other=0.0)
        out = tl.where(down, out + (pc - pn) * (0.5 * wn + 0.5 * wc), out)
        # Axis y.
        up = inside & (ys[:, None] < Y - 1)
        pn = tl.load(p_ptr + idx + X, mask=up, other=0.0)
        wn = tl.load(w_ptr + idx + X, mask=up, other=0.0)
        out = tl.where(up, out - (pn - pc) * (0.5 * wc + 0.5 * wn), out)
        down = inside & (ys[:, None] > 0)
        pn = tl.load(p_ptr + idx - X, mask=down, other=0.0)
        wn = tl.load(w_ptr + idx - X, mask=down, other=0.0)
        out = tl.where(down, out + (pc - pn) * (0.5 * wn + 0.5 * wc), out)
        # Axis x.
        up = inside & (xs[None, :] < X - 1)
        pn = tl.load(p_ptr + idx + 1, mask=up, other=0.0)
        wn = tl.load(w_ptr + idx + 1, mask=up, other=0.0)
        out = tl.where(up, out - (pn - pc) * (0.5 * wc + 0.5 * wn), out)
        down = inside & (xs[None, :] > 0)
        pn = tl.load(p_ptr + idx - 1, mask=down, other=0.0)
        wn = tl.load(w_ptr + idx - 1, mask=down, other=0.0)
        out = tl.where(down, out + (pc - pn) * (0.5 * wn + 0.5 * wc), out)
        if HAS_EXTRA:
            out = out + tl.load(extra_ptr + idx, mask=inside, other=0.0)
        if HAS_ROWS:
            # Constraint rows: (A^T v)[node], gathered from the int32 CSR of A^T in a fixed order.
            start = tl.load(at_rowptr_ptr + idx, mask=inside, other=0)
            stop = tl.load(at_rowptr_ptr + idx + 1, mask=inside, other=0)
            rows = tl.zeros([BY, BX], dtype=p_ptr.dtype.element_ty)
            for k in range(0, tl.max(tl.max(stop - start, axis=1), axis=0)):
                j = start + k
                live = inside & (j < stop)
                col = tl.load(at_cols_ptr + j, mask=live, other=0)
                rows = rows + tl.load(at_vals_ptr + j, mask=live, other=0.0) * tl.load(
                    v_ptr + col, mask=live, other=0.0
                )
            out = out + rows
        if RW_MODE == 1:
            out = out + pc * tl.load(rw_ptr).to(p_ptr.dtype.element_ty)
        if RW_MODE == 2:
            out = out + pc * tl.load(rw_ptr + idx, mask=inside, other=0.0)
        fixed = tl.load(mask_ptr + idx, mask=inside, other=1) != 0
        out = tl.where(fixed, 0.0, out)
        tl.store(ap_ptr + idx, out, mask=inside)
        part = tl.sum(tl.sum(pc.to(tl.float64) * out.to(tl.float64), axis=1), axis=0)
        pid = (z * tl.num_programs(1) + yb) * tl.num_programs(2) + xb
        tl.store(part_ptr + pid, part)

    @triton.jit
    def _update(
        x_ptr,
        r_ptr,
        p_ptr,
        ap_ptr,
        inv_ptr,
        alpha_ptr,
        part_ptr,
        N,
        nprog,
        JACOBI: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < N
        dtype = x_ptr.dtype.element_ty
        alpha = tl.load(alpha_ptr).to(dtype)
        p = tl.load(p_ptr + offs, mask=inside, other=0.0)
        x = tl.load(x_ptr + offs, mask=inside, other=0.0)
        tl.store(x_ptr + offs, x + alpha * p, mask=inside)
        ap = tl.load(ap_ptr + offs, mask=inside, other=0.0)
        r = tl.load(r_ptr + offs, mask=inside, other=0.0)
        r = r + (-alpha) * ap
        tl.store(r_ptr + offs, r, mask=inside)
        r64 = r.to(tl.float64)
        tl.store(part_ptr + nprog + pid, tl.sum(r64 * r64, axis=0))
        if JACOBI:
            zv = r * tl.load(inv_ptr + offs, mask=inside, other=0.0)
            tl.store(part_ptr + pid, tl.sum(r64 * zv.to(tl.float64), axis=0))

    @triton.jit
    def _direction(
        p_ptr,
        r_ptr,
        inv_ptr,
        z_ptr,
        beta_ptr,
        N,
        JACOBI: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < N
        beta = tl.load(beta_ptr).to(p_ptr.dtype.element_ty)
        if JACOBI:
            zv = tl.load(r_ptr + offs, mask=inside, other=0.0) * tl.load(inv_ptr + offs, mask=inside, other=0.0)
        else:
            zv = tl.load(z_ptr + offs, mask=inside, other=0.0)
        p = tl.load(p_ptr + offs, mask=inside, other=0.0)
        tl.store(p_ptr + offs, p * beta + zv, mask=inside)

    @triton.jit
    def _gather(
        rowptr_ptr, cols_ptr, vals_ptr, x_ptr, scale_ptr, out_ptr, R, HAS_SCALE: tl.constexpr, BLOCK: tl.constexpr
    ):
        # out[i] = scale[i] * sum_j vals[j] * x[cols[j]] over row i of an int32 CSR, in order.
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = rows < R
        start = tl.load(rowptr_ptr + rows, mask=inside, other=0)
        stop = tl.load(rowptr_ptr + rows + 1, mask=inside, other=0)
        acc = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
        for k in range(0, tl.max(stop - start, axis=0)):
            j = start + k
            live = inside & (j < stop)
            col = tl.load(cols_ptr + j, mask=live, other=0)
            acc = acc + tl.load(vals_ptr + j, mask=live, other=0.0) * tl.load(x_ptr + col, mask=live, other=0.0)
        if HAS_SCALE:
            acc = tl.load(scale_ptr + rows, mask=inside, other=0.0) * acc
        tl.store(out_ptr + rows, acc, mask=inside)

    _compiled = dict(stencil=_stencil, update=_update, direction=_direction, gather=_gather, triton=triton)
    return _compiled


def _csr32(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        matrix.crow_indices().to(torch.int32),
        matrix.col_indices().to(torch.int32),
        matrix.values().contiguous(),
    )


class FusedRows:
    """Continuous constraint rows (field_constraints._QuadraticOperator, CSR layout) for K1.

    v = center_weight(A p) is formed by int32 CSR gathers (no atomics; group means through
    a group-by-row CSR), and K1 gathers A^T v per node. Built once per operator and cached
    on it, so a PreparedField reuses it across sweeps.
    """

    def __init__(self, extra: Any) -> None:
        self._k = _kernels()
        self.a = _csr32(extra.a)
        self.at = _csr32(extra.at)
        self.w = extra.w.contiguous()
        self.rows = int(self.w.numel())
        self.v = torch.empty_like(self.w)
        self.grouped = bool(len(extra.sum))
        if self.grouped:
            use = extra.groups >= 0
            active = torch.nonzero(use, as_tuple=True)[0]
            groups = extra.groups.index_select(0, active)
            order = torch.sort(groups, stable=True).indices
            counts = torch.bincount(groups, minlength=len(extra.sum))
            self.g = (
                torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).to(torch.int32),
                active.index_select(0, order).to(torch.int32),
                torch.ones(len(active), dtype=self.w.dtype, device=self.w.device),
            )
            self.use = use
            self.group_of = extra.groups.clamp_min(0)
            self.denominator = extra.sum.clamp_min(torch.finfo(self.w.dtype).tiny)
            self.total = torch.empty_like(extra.sum)

    @staticmethod
    def supported(extra: Any) -> bool:
        a = getattr(extra, "a", None)
        return (
            isinstance(a, torch.Tensor)
            and a.layout == torch.sparse_csr
            and getattr(extra, "at", None) is not None
            and extra.at.layout == torch.sparse_csr
            and a.shape[1] < 2**31
            and a.values().numel() < 2**31
        )

    @classmethod
    def of(cls, extra: Any) -> FusedRows:
        cached = getattr(extra, "_fused_rows", None)
        if cached is None:
            cached = cls(extra)
            extra._fused_rows = cached
        return cached

    def _gather(self, csr, x, out, scale=None):
        rowptr, cols, vals = csr
        n = len(rowptr) - 1
        if n:
            self._k["gather"][((n + BLOCK_FLAT - 1) // BLOCK_FLAT,)](
                rowptr,
                cols,
                vals,
                x,
                x if scale is None else scale,
                out,
                n,
                HAS_SCALE=scale is not None,
                BLOCK=BLOCK_FLAT,
                enable_fp_fusion=False,
            )
        return out

    def center_weighted(self, p: torch.Tensor) -> torch.Tensor:
        """v = center_weight(A p), as _QuadraticOperator.matvec forms it before A^T."""
        v = self._gather(self.a, p.view(-1), self.v, self.w)
        if self.grouped:
            mean = self._gather(self.g, v, self.total) / self.denominator
            torch.where(self.use, v - self.w * mean.index_select(0, self.group_of), v, out=v)
        return v


class FusedOperator:
    """K1 bound to one solve: weights, reference penalty, mask and partial-sum buffers."""

    def __init__(self, weights: torch.Tensor, mask: torch.Tensor, rw: torch.Tensor | float) -> None:
        k = _kernels()
        self._k = k
        self.shape = tuple(weights.shape)
        self.weights = weights.contiguous()
        self.mask = mask.contiguous().view(torch.uint8)
        if isinstance(rw, torch.Tensor):
            self.rw_mode, self.rw = 2, rw.contiguous()
        elif rw:
            self.rw_mode = 1
            self.rw = torch.tensor([float(rw)], dtype=torch.float64, device=weights.device)
        else:
            self.rw_mode, self.rw = 0, self.weights
        z, y, x = self.shape
        self.grid = (z, k["triton"].cdiv(y, BLOCK_Y), k["triton"].cdiv(x, BLOCK_X))
        self.partials = torch.empty(
            self.grid[0] * self.grid[1] * self.grid[2], dtype=torch.float64, device=weights.device
        )

    def apply(
        self, p: torch.Tensor, out: torch.Tensor, extra: torch.Tensor | None = None, rows: FusedRows | None = None
    ) -> torch.Tensor:
        """Write A p to ``out`` (fixed rows zero) and return p.(A p) as a 0-d float64 tensor.

        ``extra`` is a precomputed term added after the stencil; ``rows`` adds A^T v of
        fused constraint rows instead (v from ``rows.center_weighted(p)``).
        """
        z, y, x = self.shape
        if rows is not None:
            v = rows.center_weighted(p)
            at_rowptr, at_cols, at_vals = rows.at
        else:
            v = at_rowptr = at_cols = at_vals = p
        self._k["stencil"][self.grid](
            p,
            self.weights,
            self.rw,
            p if extra is None else extra.contiguous(),
            at_rowptr,
            at_cols,
            at_vals,
            v,
            self.mask,
            out,
            self.partials,
            z,
            y,
            x,
            RW_MODE=self.rw_mode,
            HAS_EXTRA=extra is not None,
            HAS_ROWS=rows is not None,
            BY=BLOCK_Y,
            BX=BLOCK_X,
            enable_fp_fusion=False,
        )
        return self.partials.sum()


def _flat_grid(n: int) -> tuple[int]:
    return ((n + BLOCK_FLAT - 1) // BLOCK_FLAT,)


def pcg(
    *,
    x: torch.Tensor,
    b: torch.Tensor,
    r: torch.Tensor,
    z: torch.Tensor,
    p: torch.Tensor,
    ap: torch.Tensor,
    rz: float,
    inverse: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    rw: torch.Tensor | float,
    extra: Any,
    preconditioner: Any,
    eager_matvec: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], float],
    scale: float,
    rtol: float,
    maxiter: int,
) -> tuple[int, str]:
    """Run the PCG iterations of solve_field with fused kernels; same stopping rules.

    Expects r, z, p and rz initialised as the eager loop does. ``extra`` is the continuous
    constraint operator (or None): CSR operators are fused (FusedRows), others are applied
    eagerly and added by K1. Returns (iterations, reason); the caller recomputes the final
    residual with the eager operator.
    """
    k = _kernels()
    jacobi = preconditioner is None
    op = FusedOperator(weights, mask, rw)
    rows = FusedRows.of(extra) if extra is not None and FusedRows.supported(extra) else None
    n = x.numel()
    grid = _flat_grid(n)
    partials = torch.empty(2 * grid[0], dtype=torch.float64, device=x.device)
    rz_t = torch.tensor(rz, dtype=torch.float64, device=x.device)
    iterations = 0
    reason = "maxiter"
    current: float | None = None
    for _ in range(maxiter):
        if current is None:
            current = norm(r)
        if not math.isfinite(current):
            reason = "nonfinite_residual"
            break
        if current <= rtol * scale:
            break
        curvature_t = op.apply(p, ap, None if extra is None or rows is not None else extra.matvec(p), rows)
        curvature = float(curvature_t)
        if not math.isfinite(curvature) or not math.isfinite(rz):
            reason = "nonfinite_curvature"
            break
        if curvature <= 0 or rz <= 0:
            reason = "nonpositive_curvature"
            break
        alpha_t = rz_t / curvature_t  # Same IEEE division as the eager host-side alpha.
        k["update"][grid](
            x, r, p, ap, inverse, alpha_t, partials, n, grid[0], JACOBI=jacobi, BLOCK=BLOCK_FLAT, enable_fp_fusion=False
        )
        iterations += 1
        sums = partials.view(2, -1).sum(dim=1)
        if jacobi:
            next_rz, rr = sums.tolist()
            next_rz_t = sums[0]
        else:
            rr = float(sums[1])
        current = math.sqrt(rr) if rr >= 0 else math.nan
        if current <= rtol * scale:
            # Recurrence residuals drift: verify with the eager operator and restart.
            torch.sub(b, eager_matvec(x), out=r)
            current = None
            if jacobi:
                torch.mul(r, inverse, out=z)
            else:
                z.copy_(preconditioner.apply(r))
            rz = float(torch.dot(r.view(-1), z.view(-1)))
            p.copy_(z)
            rz_t = torch.tensor(rz, dtype=torch.float64, device=x.device)
            continue
        if not jacobi:
            z.copy_(preconditioner.apply(r))
            next_rz_t = torch.dot(r.view(-1), z.view(-1)).double()
            next_rz = float(next_rz_t)
        beta_t = next_rz_t / rz_t
        k["direction"][grid](p, r, inverse, z, beta_t, n, JACOBI=jacobi, BLOCK=BLOCK_FLAT, enable_fp_fusion=False)
        rz, rz_t = next_rz, next_rz_t
    return iterations, reason
