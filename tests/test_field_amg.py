"""AMG preconditioning changes iteration counts only, never the solved system."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import field_amg
from field_constraints import QuadraticConstraints
from field_solver import PreparedField, _FieldOperator, solve_field


def _problem(n=16, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, n + 3, n + 1)
    z, y, x = np.indices(shape)
    r = np.hypot(y - shape[1] / 2, x - shape[2] / 2)
    sheets = (np.abs(r / 3 - np.round(r / 3)) < 0.3) & (rng.random(shape) > 0.2)
    weights = np.where(sheets, 1.0, 1e-3)
    target = np.zeros((*shape, 3))
    target[..., 0] = (x - shape[2] / 2) / np.maximum(r, 1e-6) / 3
    target[..., 1] = (y - shape[1] / 2) / np.maximum(r, 1e-6) / 3
    target += 0.05 * rng.standard_normal(target.shape)
    flat = np.flatnonzero(sheets)
    pairs = np.stack([flat[:-1:7], flat[1::7]], 1)
    return target, weights, pairs


def _constraints(shape, seed=1, groups=False):
    rng = np.random.default_rng(seed)
    n, rows = int(np.prod(shape)), 60
    cols = rng.integers(0, n, (rows, 2))
    matrix = sp.csr_matrix((np.tile([1.0, -1.0], rows), (np.repeat(np.arange(rows), 2), cols.ravel())), (rows, n))
    group = np.repeat(np.arange(rows // 6), 6) if groups else None
    return QuadraticConstraints(matrix, weights=50.0, targets=rng.normal(0, 0.1, rows), groups=group)


@pytest.mark.parametrize("groups", [False, True])
def test_assembled_matrix_matches_matrix_free_operator(groups):
    target, weights, pairs = _problem(8)
    shape = weights.shape
    constraints = _constraints(shape, groups=groups)
    mask = np.zeros(shape, bool)
    mask[0] = True
    rw = np.random.default_rng(2).random(shape)
    matrix = field_amg.assemble_free_operator(
        weights, mask, pairs=pairs, lam=3.0, constraints=constraints, reference_weight=rw
    )
    w = torch.tensor(weights)
    op = _FieldOperator(w, torch.tensor(pairs), 3.0)
    extra = constraints.operator("cpu", np.float64, layout="csr")
    v = torch.tensor(np.random.default_rng(3).standard_normal(shape))
    v[torch.tensor(mask)] = 0
    expected = op.matvec(v) + extra.matvec(v).reshape(shape) + torch.tensor(rw) * v
    expected[torch.tensor(mask)] = 0
    np.testing.assert_allclose(matrix @ v.numpy().ravel(), expected.numpy().ravel(), rtol=1e-10, atol=1e-10)
    assert abs(matrix - matrix.T).max() < 1e-12


@pytest.mark.parametrize("smoother", ["chebyshev", "l1-jacobi"])
def test_preconditioner_is_symmetric_and_positive(smoother):
    _, weights, pairs = _problem(10)
    mask = np.zeros(weights.shape, bool)
    mask.flat[0] = True
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    amg = field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), max_coarse=50, smoother=smoother)
    rng = np.random.default_rng(4)
    u, v = (torch.tensor(rng.standard_normal(weights.shape)) for _ in range(2))
    u[torch.tensor(mask)] = v[torch.tensor(mask)] = 0
    mu, mv = amg.apply(u), amg.apply(v)
    assert float((u * mv).sum()) == pytest.approx(float((v * mu).sum()), rel=1e-9)
    assert float((u * mu).sum()) > 0
    assert amg.metadata()["levels"] >= 2
    assert amg.metadata()["smoother"] == smoother
    with pytest.raises(ValueError, match="smoother must be"):
        field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), smoother="sor")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_amg_solution_matches_jacobi_with_far_fewer_iterations(dtype):
    target, weights, pairs = _problem(18)
    constraints = _constraints(weights.shape)
    # float32 CG floors near 3e-5 on this lam=1000 system with either preconditioner.
    rtol = 1e-8 if dtype == "float64" else 1e-4
    kwargs = dict(pairs=pairs, lam=1000.0, constraints=constraints, rtol=rtol, maxiter=50000, dtype=dtype)
    jacobi, ji = solve_field(target, weights, preconditioner="jacobi", **kwargs)
    amg, ai = solve_field(target, weights, preconditioner="amg", **kwargs)
    assert ji["converged"] and ai["converged"]
    assert ai["relative_residual"] <= rtol
    assert ai["preconditioner"] == "amg" and ai["preconditioner_metadata"]["kind"] == "amg"
    assert ai["iterations"] * 4 < ji["iterations"]
    np.testing.assert_allclose(amg, jacobi, atol=2e-3 if dtype == "float32" else 1e-5)


def test_fixed_values_and_prepared_hierarchy_reuse():
    target, weights, _ = _problem(12)
    constraints = _constraints(weights.shape)
    fixed = np.full(weights.shape, np.nan)
    fixed[0], fixed[-1] = 1.0, 2.0
    kwargs = dict(constraints=constraints, fixed=fixed, rtol=1e-9, maxiter=50000, dtype="float64")
    reference, _ = solve_field(target, weights, **kwargs)
    prepared = PreparedField()
    first, info = solve_field(target, weights, preconditioner="amg", prepared=prepared, **kwargs)
    hierarchy = prepared.amg
    second, _ = solve_field(target, weights, preconditioner="amg", prepared=prepared, initial=first, **kwargs)
    assert prepared.amg is hierarchy and info["converged"]
    np.testing.assert_allclose(first, reference, atol=1e-6)
    np.testing.assert_allclose(second, reference, atol=1e-6)
    assert np.all(first[0] == 1.0) and np.all(first[-1] == 2.0)


def test_spectral_and_unknown_policies_still_rejected_for_amg_mismatch():
    target, weights, _ = _problem(6)
    with pytest.raises(ValueError, match="jacobi, spectral or amg"):
        solve_field(target, weights, preconditioner="ilu")
    prepared = PreparedField()
    solve_field(target, weights, preconditioner="amg", prepared=prepared, dtype="float64")
    with pytest.raises(ValueError, match="policy changed"):
        solve_field(target, weights, preconditioner="jacobi", prepared=prepared, dtype="float64")
