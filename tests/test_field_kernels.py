"""Fused PCG kernels (solve_field kernel="fused") against the eager PyTorch path as oracle.

Without CUDA the kernels run in Triton's interpreter on CPU tensors (conftest sets
TRITON_INTERPRET=1); with CUDA they compile and run on the GPU.
"""

import json

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

import field_kernels
from field_constraints import QuadraticConstraints
from field_solver import PreparedField, _FieldOperator, solve_field

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPES = {"float64": torch.float64, "float32": torch.float32}


def _problem(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(*shape, 3)), rng.uniform(0.5, 2.0, shape), rng


def _constraints(n, rows, rng, weight=1000.0):
    ends = np.stack([rng.choice(n, 2, replace=False) for _ in range(rows)])
    matrix = csr_matrix((np.tile([1.0, -1.0], rows), (np.repeat(np.arange(rows), 2), ends.ravel())), shape=(rows, n))
    groups = np.where(np.arange(rows) % 3 == 0, -1, np.arange(rows) // 4)
    return QuadraticConstraints(matrix, rng.uniform(1, weight, rows), rng.normal(size=rows), groups)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 1, 7), (3, 5, 33), (6, 9, 70), (2, 17, 1)])
@pytest.mark.parametrize("rw", ["none", "scalar", "nodal"])
def test_stencil_matches_the_eager_operator(dtype, shape, rw):
    torch_dtype = DTYPES[dtype]
    rng = np.random.default_rng(3)
    weights = torch.tensor(rng.uniform(0.1, 5.0, shape), dtype=torch_dtype, device=DEVICE)
    p = torch.tensor(rng.normal(size=shape), dtype=torch_dtype, device=DEVICE)
    extra = torch.tensor(rng.normal(size=shape), dtype=torch_dtype, device=DEVICE)
    mask = torch.tensor(rng.random(shape) < 0.2, device=DEVICE)
    reference = {"none": 0.0, "scalar": 0.7, "nodal": torch.tensor(rng.random(shape), dtype=torch_dtype, device=DEVICE)}
    rws = reference[rw]
    expected = _FieldOperator(weights, None, 0).matvec(p) + extra
    if isinstance(rws, torch.Tensor):
        expected.addcmul_(p, rws)
    elif rws:
        expected.add_(p, alpha=rws)
    expected[mask] = 0
    out = torch.full_like(p, float("nan"))
    dot = field_kernels.FusedOperator(weights, mask, rws).apply(p, out, extra)
    tolerance = 1e-14 if dtype == "float64" else 1e-6
    scale = float(expected.abs().max()) or 1.0
    assert float((out - expected).abs().max()) <= tolerance * scale
    exact = float(torch.dot(p.double().view(-1), out.double().view(-1)))
    assert abs(float(dot) - exact) <= 1e-13 * max(abs(exact), float((p.double().abs() * out.double().abs()).sum()))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("preconditioner", ["jacobi", "amg"])
@pytest.mark.parametrize("prepared", [False, True])
def test_fused_solve_matches_eager(dtype, preconditioner, prepared):
    target, weights, rng = _problem((5, 7, 9))
    fixed = np.full(weights.shape, np.nan)
    fixed[0] = rng.normal(size=weights.shape[1:])
    constraints = _constraints(weights.size, 25, rng, weight=10.0)
    rtol = 1e-10 if dtype == "float64" else 1e-5
    kw = dict(fixed=fixed, constraints=constraints, preconditioner=preconditioner, dtype=dtype, rtol=rtol, maxiter=5000)
    kw["device"] = DEVICE
    eager, a = solve_field(target, weights, prepared=PreparedField() if prepared else None, **kw)
    fused, b = solve_field(target, weights, kernel="fused", prepared=PreparedField() if prepared else None, **kw)
    assert b["kernel"] == field_kernels.KERNEL_VERSION and "kernel_fallback" not in b
    assert (a["converged"], a["reason"]) == (b["converged"], b["reason"]) == (True, "converged")
    assert abs(a["iterations"] - b["iterations"]) <= max(1, 0.05 * a["iterations"])
    assert b["relative_residual"] <= rtol
    assert np.abs(eager - fused).max() <= (1e-8 if dtype == "float64" else 1e-3) * max(1.0, np.abs(eager).max())
    np.testing.assert_array_equal(fused[0], fixed[0].astype(fused.dtype))
    json.dumps(b, allow_nan=False)


@pytest.mark.parametrize("rw", [0.0, 0.3, "nodal"])
def test_fused_solve_without_constraints_matches_eager_iterations(rw):
    target, weights, rng = _problem((4, 6, 11), seed=5)
    reference = rng.normal(size=weights.shape)
    rws = rng.random(weights.shape) if rw == "nodal" else rw
    kw = dict(reference=reference, reference_weight=rws, dtype="float64", rtol=1e-9, device=DEVICE)
    eager, a = solve_field(target, weights, **kw)
    fused, b = solve_field(target, weights, kernel="fused", **kw)
    assert a["iterations"] == b["iterations"] and b["converged"]
    assert np.abs(eager - fused).max() <= 1e-12 * max(1.0, np.abs(eager).max())


def test_fused_solve_is_deterministic():
    target, weights, rng = _problem((6, 8, 40), seed=7)
    constraints = _constraints(weights.size, 60, rng)
    kw = dict(constraints=constraints, dtype="float64", rtol=1e-9, kernel="fused", return_tensor=True, device=DEVICE)
    first, a = solve_field(target, weights, **kw)
    second, b = solve_field(target, weights, **kw)
    assert a["iterations"] == b["iterations"]
    assert torch.equal(first, second)


def test_fused_failure_receipts_match_eager():
    cases = [
        (
            dict(target_xyz=np.random.default_rng(1).normal(size=(3, 4, 7, 3)), weights=np.ones((3, 4, 7)), maxiter=3),
            "maxiter",
        ),
        (
            dict(target_xyz=np.broadcast_to([1e-25, 0, 0], (2, 3, 4, 3)), weights=np.ones((2, 3, 4))),
            "nonpositive_curvature",
        ),
        (
            dict(target_xyz=np.zeros((2, 3, 4, 3)), weights=np.ones((2, 3, 4)), initial=np.full((2, 3, 4), 1e30)),
            "nonfinite_curvature",
        ),
        (
            dict(
                target_xyz=np.zeros((2, 3, 4, 3)),
                weights=np.ones((2, 3, 4)),
                initial=np.where(np.indices((2, 3, 4))[2] % 2, 1e38, -1e38),
            ),
            "nonfinite_residual",
        ),
    ]
    for kwargs, reason in cases:
        _, eager = solve_field(device=DEVICE, **kwargs)
        field, fused = solve_field(device=DEVICE, kernel="fused", **kwargs)
        assert eager["reason"] == fused["reason"] == reason
        assert not fused["converged"] and np.isfinite(field).all()
        assert eager["iterations"] == fused["iterations"]
        json.dumps(fused, allow_nan=False)


def test_unsupported_solves_fall_back_to_eager_with_a_receipt():
    target, weights, _ = _problem((3, 4, 5))
    _, info = solve_field(target, weights, pairs=[[0, 7]], kernel="fused", device=DEVICE)
    assert (info["kernel"], info["kernel_fallback"]) == ("eager", "discrete pairs are not fused")
    _, info = solve_field(target, weights, dtype="float64", preconditioner="spectral", kernel="fused")
    assert (info["kernel"], info["kernel_fallback"]) == ("eager", "preconditioner=spectral is not fused")


def test_missing_triton_or_gpu_is_a_fallback(monkeypatch):
    def missing():
        raise ImportError("triton")

    monkeypatch.setattr(field_kernels, "_kernels", missing)
    assert field_kernels.unavailable_reason(torch.device(DEVICE), "jacobi", False) == "triton is not installed"
    monkeypatch.setenv("TRITON_INTERPRET", "0")
    assert field_kernels.unavailable_reason(torch.device("cpu"), "jacobi", False) == "fused kernels need a CUDA device"


def test_default_receipt_is_unchanged_and_kernel_is_validated():
    target, weights, _ = _problem((2, 3, 4))
    _, info = solve_field(target, weights)
    assert "kernel" not in info and "kernel_fallback" not in info
    with pytest.raises(ValueError, match="kernel must be eager or fused"):
        solve_field(target, weights, kernel="triton")
