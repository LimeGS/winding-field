"""Small CPU checks against analytic fields and independent dense least squares."""

import importlib
import json

import numpy as np
import pytest
import torch


@pytest.fixture
def solver():
    try:
        return importlib.import_module("field_solver")
    except ModuleNotFoundError as exc:
        if exc.name != "field_solver":
            raise
        pytest.fail("shared field_solver has not been implemented")


@pytest.mark.parametrize("component", range(3))
def test_positive_xyz_ramps_on_unequal_dimensions(solver, component):
    shape = (3, 4, 7)
    target = np.zeros((*shape, 3))
    target[..., component] = 0.3
    field, info = solver.solve_field(target, np.ones(shape), rtol=1e-6)
    expected = np.indices(shape)[2 - component] * 0.3
    np.testing.assert_allclose(field, expected, atol=1e-3)
    assert field.dtype == np.float32
    assert info["converged"] and info["relative_residual"] <= 1e-6
    assert info["gauge"] == {"kind": "zero", "flat_index": 0, "value": 0.0}
    assert field.flat[0] == 0
    assert info["elapsed_s"] >= 0


def test_anisotropic_spacing_and_nonuniform_weights(solver):
    shape = (3, 5, 7)
    z, y, x = np.indices(shape)
    target = np.broadcast_to([0.2, -0.3, 0.4], (*shape, 3))
    field, info = solver.solve_field(target, 1 + x + 2 * y + z, spacing_xyz=(2, 3, 5), rtol=1e-6)
    np.testing.assert_allclose(field, 0.4 * x - 0.9 * y + 2 * z, atol=1e-3)
    assert info["converged"]


def dense_system(target, weights, spacing, pairs=(), lam=0, reference=None, rw=0):
    """Build incidence rows, not the production stencil."""
    shape = weights.shape
    size = weights.size
    rows, values = [], []
    for point in np.ndindex(shape):
        for axis in range(3):
            neighbour = list(point)
            neighbour[axis] += 1
            if neighbour[axis] == shape[axis]:
                continue
            neighbour = tuple(neighbour)
            row = np.zeros(size)
            weight = np.sqrt((weights[point] + weights[neighbour]) / 2)
            row[np.ravel_multi_index(point, shape)] = -weight
            row[np.ravel_multi_index(neighbour, shape)] = weight
            rows.append(row)
            values.append(weight * spacing[2 - axis] * (target[point][2 - axis] + target[neighbour][2 - axis]) / 2)
    for a, b in pairs:
        row = np.zeros(size)
        row[a] += np.sqrt(lam)
        row[b] -= np.sqrt(lam)
        rows.append(row)
        values.append(0.0)
    if rw:
        rows.extend(np.sqrt(rw) * np.eye(size))
        values.extend(np.sqrt(rw) * reference.ravel())
    design = np.asarray(rows)
    return design.T @ design, design.T @ np.asarray(values)


def test_operator_matches_dense_matrix_is_symmetric_and_preserves_constants(solver):
    shape = (2, 3, 4)
    weights = np.arange(24, dtype=float).reshape(shape) / 10 + 0.2
    pairs = np.array([[0, 23], [7, 3], [7, 3], [2, 2]])
    operator = solver._FieldOperator(torch.tensor(weights), torch.tensor(pairs), 2.5)
    matrix, _ = dense_system(np.zeros((*shape, 3)), weights, (1, 1, 1), pairs, 2.5)
    rng = np.random.default_rng(10)
    u, v = [torch.tensor(rng.normal(size=shape)) for _ in range(2)]
    np.testing.assert_allclose(operator.matvec(u).numpy().ravel(), matrix @ u.numpy().ravel(), atol=1e-12)
    np.testing.assert_allclose(operator.diagonal.numpy().ravel(), matrix.diagonal(), atol=1e-12)
    torch.testing.assert_close((u * operator.matvec(v)).sum(), (v * operator.matvec(u)).sum())
    torch.testing.assert_close(operator.matvec(torch.ones_like(u)), torch.zeros_like(u))
    # No periodic edge joining the two X faces.
    plain = solver._FieldOperator(torch.ones(shape, dtype=torch.float64), None, 0)
    impulse = torch.zeros(shape, dtype=torch.float64)
    impulse[0, 0, 0] = 1
    assert plain.matvec(impulse)[0, 0, -1] == 0


def test_fixed_nonzero_values_are_exact_even_against_conflicting_reference(solver):
    shape = (2, 3, 5)
    fixed = np.full(shape, np.nan, dtype=np.float32)
    fixed[..., 0] = 3.25
    fixed[..., -1] = -1.75
    before = fixed.copy()
    field, info = solver.solve_field(
        np.zeros((*shape, 3)),
        np.ones(shape),
        fixed=fixed,
        reference=np.full(shape, 100.0),
        reference_weight=2,
        rtol=1e-6,
    )
    mask = np.isfinite(fixed)
    np.testing.assert_array_equal(field[mask], fixed[mask])
    np.testing.assert_array_equal(fixed, before)
    assert info["converged"] and info["gauge"]["kind"] == "fixed"


def test_solution_matches_independent_dense_elimination(solver):
    rng = np.random.default_rng(3)
    shape = (2, 3, 4)
    target = rng.normal(size=(*shape, 3))
    weights = rng.uniform(0.1, 2, shape)
    pairs = np.array([[0, 23], [1, 17], [2, 2], [0, 23]])
    reference = rng.normal(size=shape)
    fixed = np.full(shape, np.nan)
    fixed.flat[[3, 22]] = [1.25, -0.5]
    initial = np.full(shape, 71.0)
    matrix, rhs = dense_system(target, weights, (2, 3, 4), pairs, 1.7, reference, 0.2)
    constrained = np.isfinite(fixed.ravel())
    free = ~constrained
    expected = fixed.ravel().copy()
    expected[free] = np.linalg.solve(
        matrix[np.ix_(free, free)], rhs[free] - matrix[np.ix_(free, constrained)] @ expected[constrained]
    )
    field, info = solver.solve_field(target, weights, (2, 3, 4), pairs, 1.7, fixed, reference, 0.2, initial, rtol=1e-6)
    np.testing.assert_allclose(field.ravel(), expected, atol=1e-4)
    residual = (rhs - matrix @ field.ravel())[free]
    scale = np.linalg.norm(rhs[free] - matrix[np.ix_(free, constrained)] @ expected[constrained])
    assert np.linalg.norm(residual) / scale < 2e-6
    assert info["converged"]
    np.testing.assert_array_equal(initial, np.full(shape, 71.0))


def test_reference_alone_fixes_gauge_without_recentering(solver):
    shape = (2, 3, 4)
    field, info = solver.solve_field(
        np.zeros((*shape, 3)), np.ones(shape), reference=np.full(shape, 7.5), reference_weight=0.3
    )
    np.testing.assert_allclose(field, 7.5, atol=1e-4)
    assert info["converged"] and info["gauge"]["kind"] == "reference"


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 1, 1)])
def test_zero_rhs_converges_without_a_cg_division(solver, shape):
    field, info = solver.solve_field(np.zeros((*shape, 3)), np.ones(shape))
    np.testing.assert_array_equal(field, np.zeros(shape))
    assert info["converged"] and info["iterations"] == 0
    assert info["relative_residual"] == 0


def test_all_fixed_is_exact_with_no_free_equations(solver):
    fixed = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    field, info = solver.solve_field(np.zeros((*fixed.shape, 3)), np.ones_like(fixed), fixed=fixed)
    np.testing.assert_array_equal(field, fixed)
    assert info["converged"] and info["iterations"] == 0


def test_zero_rhs_with_nonzero_initial_still_solves(solver):
    shape = (2, 3, 4)
    field, info = solver.solve_field(np.zeros((*shape, 3)), np.ones(shape), initial=np.ones(shape))
    np.testing.assert_allclose(field, 0, atol=1e-4)
    assert info["converged"]


def test_tensor_inputs_are_detached_without_mutation(solver):
    target = torch.zeros((2, 3, 4, 3), requires_grad=True)
    weights = torch.ones((2, 3, 4), requires_grad=True)
    reference = torch.full((2, 3, 4), 4.0, requires_grad=True)
    field, info = solver.solve_field(target, weights, reference=reference, reference_weight=0.5)
    np.testing.assert_allclose(field, 4, atol=1e-4)
    assert info["converged"]
    assert target.grad is weights.grad is reference.grad is None
    assert target.requires_grad and weights.requires_grad and reference.requires_grad


def test_noncontiguous_arrays_preserve_flat_zyx_pair_indexing(solver):
    weights = np.ones((5, 3, 2), np.float32).transpose(2, 1, 0)
    target = np.broadcast_to([0.25, 0, 0], (*weights.shape, 3)).astype(np.float32)
    initial = np.zeros((5, 3, 2), np.float32).transpose(2, 1, 0)
    field, info = solver.solve_field(target, weights, pairs=[[0, 5], [2, 17]], initial=initial)
    np.testing.assert_allclose(field, np.indices(weights.shape)[2] * 0.25, atol=1e-3)
    assert info["converged"]


@pytest.mark.parametrize("weight", [1e-25, 1e25])
def test_uniform_weight_scaling_cannot_create_false_zero_residual(solver, weight):
    shape = (2, 3, 4)
    target = np.broadcast_to([1.0, 0.0, 0.0], (*shape, 3))
    field, info = solver.solve_field(target, np.full(shape, weight))
    np.testing.assert_allclose(field, np.indices(shape)[2], atol=1e-3)
    assert info["converged"]


@pytest.mark.parametrize("pairs", [[], np.empty((0, 2)), np.empty((0, 2), dtype=np.int64)])
def test_empty_pairs_are_a_noop(solver, pairs):
    field, info = solver.solve_field(np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), pairs=pairs)
    np.testing.assert_array_equal(field, np.zeros((2, 3, 4)))
    assert info["converged"]


@pytest.mark.parametrize("maxiter", [0, 1])
def test_iteration_exhaustion_is_reported_as_failure(solver, maxiter):
    rng = np.random.default_rng(1)
    field, info = solver.solve_field(rng.normal(size=(3, 4, 7, 3)), np.ones((3, 4, 7)), rtol=1e-7, maxiter=maxiter)
    assert not info["converged"]
    assert info["iterations"] == maxiter
    assert info["reason"] == "maxiter"
    assert info["relative_residual"] > 1e-7
    assert np.isfinite(field).all()


def test_underflowed_cg_curvature_is_failure_not_false_convergence(solver):
    target = np.broadcast_to([1e-25, 0, 0], (2, 3, 4, 3))
    field, info = solver.solve_field(target, np.ones((2, 3, 4)))
    assert not info["converged"]
    assert info["reason"] == "nonpositive_curvature"
    assert info["relative_residual"] > 0.9
    assert np.isfinite(field).all()


def test_overflowed_cg_curvature_has_an_explicit_failure_receipt(solver):
    field, info = solver.solve_field(np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), initial=np.full((2, 3, 4), 1e30))
    assert not info["converged"]
    assert info["reason"] == "nonfinite_curvature"
    assert np.isfinite(field).all()
    json.dumps(info, allow_nan=False)


def test_nonfinite_residual_does_not_break_failure_receipt_serialization(solver):
    initial = np.where(np.indices((2, 3, 4))[2] % 2, 1e38, -1e38)
    _, info = solver.solve_field(np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), initial=initial)
    assert not info["converged"]
    assert info["reason"] == "nonfinite_residual"
    assert info["relative_residual"] is None
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"target_xyz": np.zeros((2, 3, 4, 2))},
        {"target_xyz": np.full((2, 3, 4, 3), np.nan)},
        {"weights": np.ones((4, 3, 2))},
        {"weights": np.zeros((2, 3, 4))},
        {"weights": np.full((2, 3, 4), -1)},
        {"weights": np.full((2, 3, 4), np.inf)},
        {"spacing_xyz": (1, 0, 1)},
        {"spacing_xyz": (1, 2)},
        {"spacing_xyz": (1, np.nan, 1)},
        {"lam": -1},
        {"lam": np.inf},
        {"lam": "not-a-number"},
        {"pairs": [[0, 24]]},
        {"pairs": [[-1, 3]]},
        {"pairs": [[0.5, 3]]},
        {"pairs": [[1, 2, 3]]},
        {"fixed": np.full((2, 3, 4), np.inf)},
        {"fixed": np.zeros((3, 4, 2))},
        {"reference_weight": -1},
        {"reference_weight": 1},
        {"reference": np.full((2, 3, 4), np.nan)},
        {"initial": np.ones((3, 4, 2))},
        {"initial": np.full((2, 3, 4), np.inf)},
        {"rtol": 0},
        {"rtol": np.nan},
        {"maxiter": -1},
        {"maxiter": 1.5},
    ],
)
def test_invalid_inputs_fail_before_solving(solver, changes):
    args = dict(target_xyz=np.zeros((2, 3, 4, 3)), weights=np.ones((2, 3, 4)))
    args.update(changes)
    with pytest.raises(ValueError):
        solver.solve_field(**args)


def test_nodal_reference_weights_match_dense_matrix_with_exact_fixed_overlap(solver):
    rng = np.random.default_rng(42)
    shape = (2, 3, 4)
    target, weights = rng.normal(size=(*shape, 3)), rng.uniform(0.2, 2, shape)
    pairs = np.array([[0, 23], [4, 13], [4, 13]])
    rw, reference = np.zeros(shape), rng.normal(size=shape) + 7
    rw.flat[[0, 4, 9, 20]] = [3, 6, 12, 3]
    fixed = np.full(shape, np.nan)
    fixed.flat[[0, 22]] = [2.25, -0.75]  # node 0 also has a reference penalty
    matrix, rhs = dense_system(target, weights, (1, 2, 3), pairs, 2.0)
    matrix += np.diag(rw.ravel())
    rhs += (rw * reference).ravel()
    constrained = np.isfinite(fixed.ravel())
    free = ~constrained
    expected = fixed.ravel().copy()
    expected[free] = np.linalg.solve(
        matrix[np.ix_(free, free)], rhs[free] - matrix[np.ix_(free, constrained)] @ expected[constrained]
    )
    field, info = solver.solve_field(target, weights, (1, 2, 3), pairs, 2.0, fixed, reference, rw, rtol=1e-6)
    assert info["converged"]
    np.testing.assert_allclose(field.ravel(), expected, atol=1e-4)
    np.testing.assert_array_equal(field.ravel()[constrained], fixed.ravel()[constrained])
    np.testing.assert_array_equal(rw.ravel()[[0, 4, 9, 20]], [3, 6, 12, 3])


def test_sparse_nodal_reference_preserves_constant_without_zero_gauge(solver):
    shape = (2, 3, 4)
    rw = np.zeros(shape)
    rw[1, 1, 2] = 3
    field, info = solver.solve_field(
        np.zeros((*shape, 3)), np.ones(shape), reference=np.full(shape, 123.0), reference_weight=rw, rtol=1e-6
    )
    assert info["converged"] and info["gauge"]["kind"] == "reference"
    np.testing.assert_allclose(field, 123.0, atol=1e-3)


def test_zero_nodal_reference_weight_keeps_explicit_zero_gauge(solver):
    field, info = solver.solve_field(np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), reference_weight=np.zeros((2, 3, 4)))
    assert info["converged"] and info["gauge"]["kind"] == "zero"
    np.testing.assert_array_equal(field, np.zeros((2, 3, 4)))


@pytest.mark.parametrize(
    "rw",
    [
        np.ones((3, 2, 4)),
        np.ones(24),
        np.full((2, 3, 4), np.nan),
        np.full((2, 3, 4), np.inf),
        np.full((2, 3, 4), -1),
        np.full((2, 3, 4), 1j),
    ],
)
def test_bad_nodal_reference_weights_are_rejected(solver, rw):
    with pytest.raises(ValueError):
        solver.solve_field(
            np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), reference=np.ones((2, 3, 4)), reference_weight=rw
        )


def test_positive_nodal_weights_require_reference(solver):
    with pytest.raises(ValueError, match="requires reference"):
        solver.solve_field(np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), reference_weight=np.ones((2, 3, 4)))
