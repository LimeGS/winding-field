"""Per-iteration cost of the field PCG loop, eager vs fused (solve_field kernel="fused").

python benchmarks/performance/pcg_bench.py --nodes 224 --iterations 300 --output pcg.json

Builds a synthetic tile shaped like the 34.56 um PHerc0175A tiles: random positive nodal
weights, 0.045 fibre rows per node (two trilinear stencils of 8 nodes each, weight 1000)
and a fixed bottom face. Times, on one GPU:

- parts of one eager iteration: gradient stencil, fibre rows (extra.matvec), vector ops;
- the fused kernels K1 (stencil + p.Ap), K2 (update + dots), K3 (direction);
- whole solves of exactly --iterations iterations with each kernel (rtol below reach).

Bytes per node-iteration are counted from the code (float64: 8 B per array access) so the
effective bandwidth can be compared with a device copy. Phase 0 of the fused-kernel plan
uses the fibre share of the eager iteration to bound the speedup of fusing the rest.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import field_kernels
from field_constraints import QuadraticConstraints
from field_solver import _FieldOperator, solve_field


def fibre_rows(shape, rows_per_node, weight, rng):
    """Rows A_i phi = phi(a) - phi(b), each endpoint a trilinear stencil (16 nonzeros)."""
    n = int(np.prod(shape))
    rows = max(1, int(rows_per_node * n))
    hi = np.array(shape) - 1
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    strides = np.array([shape[1] * shape[2], shape[2], 1])

    def stencil(points):
        base = np.floor(points).astype(np.int64)
        frac = points - base
        nodes = ((base[:, None, :] + corners[None]) * strides).sum(-1)
        w = np.prod(np.where(corners[None], frac[:, None, :], 1 - frac[:, None, :]), axis=-1)
        return nodes, w

    a = rng.uniform(0, hi - 1e-6, size=(rows, 3))
    b = np.clip(a + rng.normal(scale=3.0, size=(rows, 3)), 0, hi - 1e-6)
    na, wa = stencil(a)
    nb, wb = stencil(b)
    cols = np.concatenate([na, nb], axis=1).ravel()
    data = np.concatenate([wa, -wb], axis=1).ravel()
    matrix = csr_matrix((data, (np.repeat(np.arange(rows), 16), cols)), shape=(rows, n))
    return QuadraticConstraints(matrix, weight, 0.0)


def timed(fn, repeat, device):
    fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / repeat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nodes", type=int, default=224, help="tile edge in nodes")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rows-per-node", type=float, default=0.045)
    parser.add_argument("--fibre-weight", type=float, default=1000.0)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    size = torch.finfo(dtype).bits // 8
    rng = np.random.default_rng(0)
    shape = (args.nodes,) * 3
    n = int(np.prod(shape))
    weights = rng.uniform(0.5, 2.0, shape)
    target = rng.normal(size=(*shape, 3))
    fixed = np.full(shape, np.nan)
    fixed[0] = 0.0
    constraints = fibre_rows(shape, args.rows_per_node, args.fibre_weight, rng)
    report = dict(
        device=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        torch=torch.__version__,
        shape=list(shape),
        dtype=args.dtype,
        fibre_rows=int(constraints.matrix.shape[0]),
        fibre_nonzeros=int(constraints.matrix.nnz),
    )

    w = torch.tensor(weights, dtype=dtype, device=device)
    p = torch.tensor(rng.normal(size=shape), dtype=dtype, device=device)
    out = torch.empty_like(p)
    scratch = torch.empty_like(p)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[0] = True
    operator = _FieldOperator(w, None, 0)
    extra = constraints.operator(device, np.float64 if dtype == torch.float64 else np.float32, layout="csr")
    inverse = torch.rand_like(p)
    r, x, z = (torch.rand_like(p) for _ in range(3))
    copy_src = torch.empty(n * 4, dtype=dtype, device=device)
    copy_dst = torch.empty_like(copy_src)
    copy_s = timed(lambda: copy_dst.copy_(copy_src), args.repeat, device)
    report["copy_gbps"] = 2 * copy_src.numel() * size / copy_s / 1e9

    def eager_vectors():
        torch.dot(p.view(-1), out.view(-1))
        x.add_(p, alpha=1e-3)
        r.add_(out, alpha=-1e-3)
        torch.linalg.vector_norm(r, dtype=torch.float64)
        torch.mul(r, inverse, out=z)
        torch.dot(r.view(-1), z.view(-1))
        p.mul_(0.5).add_(z)

    def eager_stencil():
        operator.matvec(p, out=out, scratch=scratch)
        out[mask] = 0

    fused = field_kernels.FusedOperator(w, mask, 0.0)
    alpha = torch.tensor(1e-3, dtype=torch.float64, device=device)
    k = field_kernels._kernels()
    grid = field_kernels._flat_grid(n)
    partials = torch.empty(2 * grid[0], dtype=torch.float64, device=device)

    def k2():
        k["update"][grid](
            x,
            r,
            p,
            out,
            inverse,
            alpha,
            partials,
            n,
            grid[0],
            JACOBI=True,
            BLOCK=field_kernels.BLOCK_FLAT,
            enable_fp_fusion=False,
        )
        partials.view(2, -1).sum(dim=1).tolist()

    def k3():
        k["direction"][grid](
            p, r, inverse, z, alpha, n, JACOBI=True, BLOCK=field_kernels.BLOCK_FLAT, enable_fp_fusion=False
        )

    parts = dict(
        eager_stencil=timed(eager_stencil, args.repeat, device),
        eager_fibres=timed(lambda: extra.matvec(p), args.repeat, device),
        eager_vectors=timed(eager_vectors, args.repeat, device),
        fused_k1=timed(lambda: float(fused.apply(p, out)), args.repeat, device),
        fused_k1_with_fibres=timed(lambda: float(fused.apply(p, out, extra.matvec(p))), args.repeat, device),
        fused_k2=timed(k2, args.repeat, device),
        fused_k3=timed(k3, args.repeat, device),
    )
    report["parts_ms"] = {key: value * 1e3 for key, value in parts.items()}
    eager_total = parts["eager_stencil"] + parts["eager_fibres"] + parts["eager_vectors"]
    report["fibre_share_of_eager_iteration"] = parts["eager_fibres"] / eager_total
    report["fused_ceiling_speedup"] = eager_total / (
        parts["fused_k1_with_fibres"] + parts["fused_k2"] + parts["fused_k3"]
    )
    # Accesses per node counted from the code (see the plan): eager ~62, fused ~15.
    report["bytes_per_node_iteration"] = dict(eager_nonfibre=62 * size, fused_nonfibre=15 * size)
    report["effective_gbps"] = dict(
        eager_nonfibre=62 * size * n / (parts["eager_stencil"] + parts["eager_vectors"]) / 1e9,
        fused_nonfibre=15 * size * n / (parts["fused_k1"] + parts["fused_k2"] + parts["fused_k3"]) / 1e9,
    )

    def solve(kernel, maxiter):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        _, info = solve_field(
            target,
            weights,
            fixed=fixed,
            constraints=constraints,
            dtype=args.dtype,
            device=device,
            rtol=1e-30,
            maxiter=maxiter,
            return_tensor=True,
            kernel=kernel,
        )
        torch.cuda.synchronize(device)
        return time.perf_counter() - start, info

    solves = {}
    for kernel in ("eager", "fused"):
        setup, _ = solve(kernel, 0)  # Assembly, transfers and the eager residuals only.
        elapsed, info = solve(kernel, args.iterations)
        solves[kernel] = dict(
            seconds=elapsed,
            loop_seconds=elapsed - setup,
            iterations=info["iterations"],
            receipt_elapsed_s=info["elapsed_s"],
            peak_vram_bytes=torch.cuda.max_memory_allocated(device),
            relative_residual=info["relative_residual"],
            kernel=info.get("kernel", "eager"),
        )
    for value in solves.values():
        value["ns_per_node_iteration"] = value["loop_seconds"] / max(value["iterations"], 1) / n * 1e9
    report["solves"] = solves
    report["loop_speedup"] = solves["eager"]["loop_seconds"] / solves["fused"]["loop_seconds"]
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
