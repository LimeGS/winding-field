"""Precision and reuse must preserve the independently assembled equations."""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

import field_solver as solver
from field_constraints import QuadraticConstraints, solve_with_relations


def observations(dtype=torch.float32):
    target = torch.zeros((3, 4, 5, 3), dtype=dtype)
    target[:] = torch.tensor([0.2, -0.1, 0.05], dtype=dtype)
    return target, torch.ones((3, 4, 5), dtype=dtype)


def test_float64_ramp_keeps_physical_precision_and_return_dtype():
    target, weights = observations(torch.float64)
    field, receipt = solver.solve_field(target, weights, [2, 3, 5], dtype="float64", rtol=1e-12)
    z, y, x = np.indices(weights.shape)
    assert field.dtype == np.float64 and receipt["converged"]
    assert receipt["dtype"] == "float64"
    np.testing.assert_allclose(field, 0.4 * x - 0.3 * y + 0.25 * z, atol=1e-10)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_float64_consensus_operator_matches_dense_eliminated_system(layout):
    a = np.array([[1, 0, 0], [0.3, 0.7, 0], [0, 0, 1.0]])
    w = np.array([1.0, 2.0, 4.0])
    targets = np.array([0.123456789, 2.345678901, 0.25])
    constraint = QuadraticConstraints(csr_matrix(a), w, targets, [0, 0, 0])
    operator = constraint.operator("cpu", np.float64, layout=layout)
    centered = np.diag(w) - np.outer(w, w) / w.sum()
    expected = a.T @ centered @ a
    assert operator.a.dtype == torch.float64
    np.testing.assert_allclose(operator.diagonal.numpy(), np.diag(expected), atol=1e-12)
    np.testing.assert_allclose(operator.rhs().numpy(), a.T @ centered @ targets, atol=1e-12)
    np.testing.assert_allclose(
        operator.matvec(torch.tensor([2.0, 3.0, 7.0], dtype=torch.float64)).numpy(), expected @ [2, 3, 7], atol=1e-12
    )


def test_target_rebinding_shares_diagonal_but_owns_targets():
    c = QuadraticConstraints(csr_matrix([[1.0, -1.0, 0], [0, 1.0, -1.0]]), [2, 3])
    target = np.array([2.0, 4.0])
    rebound = c.with_targets(target)
    target[:] = -99
    assert rebound.diagonal is c.diagonal and rebound.matrix is c.matrix
    assert not c.matrix.data.flags.writeable
    np.testing.assert_array_equal(rebound.targets, [2, 4])
    np.testing.assert_array_equal(c.targets, [0, 0])
    reference = QuadraticConstraints(c.matrix, c.weights, [2, 4], c.groups)
    assert rebound.digest == reference.digest
    with pytest.raises(ValueError):
        c.with_targets([np.nan, 0])
    with pytest.raises(ValueError):
        c.with_targets([1, 2, 3])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prepared_reuse_matches_plain_and_refreshes_targets(dtype):
    target, weights = observations(getattr(torch, dtype))
    matrix = csr_matrix(([1.0, -1.0], ([0, 0], [2, 20])), shape=(1, weights.numel()))
    constraint = QuadraticConstraints(matrix, 0.5).with_targets(0.0)
    workspace = solver.PreparedField()
    cached_operator = None
    for value in (0.0, 0.7):
        rows = constraint.with_targets(value)
        plain, expected = solver.solve_field(target, weights, constraints=rows, dtype=dtype)
        field, receipt = solver.solve_field(target, weights, constraints=rows, dtype=dtype, prepared=workspace)
        assert receipt["converged"] == expected["converged"]
        np.testing.assert_array_equal(field, plain)
        if cached_operator is not None:
            assert workspace.operator is cached_operator
        cached_operator = workspace.operator


@pytest.mark.parametrize("changed", ["target", "weight", "numpy_alias", "spacing", "rows", "dtype"])
def test_prepared_workspace_rejects_stale_static_inputs(changed):
    source = np.zeros((3, 4, 5, 3), np.float32)
    source[..., 0] = 0.2
    target, weights = torch.from_numpy(source), torch.ones((3, 4, 5))
    rows = QuadraticConstraints(csr_matrix(([1.0, -1.0], ([0, 0], [2, 20])), shape=(1, 60))).with_targets(0.0)
    workspace = solver.PreparedField()
    solver.solve_field(target, weights, constraints=rows, prepared=workspace)
    kwargs = dict(constraints=rows, prepared=workspace)
    if changed == "target":
        target[0, 0, 0, 0] += 1
    elif changed == "weight":
        weights[0, 0, 0] += 1
    elif changed == "numpy_alias":
        source[0, 0, 0, 0] += 1  # Does not increment a Torch tensor version.
    elif changed == "spacing":
        kwargs["spacing_xyz"] = [2, 1, 1]
    elif changed == "rows":
        rows.matrix.data.setflags(write=True)
        rows.matrix.data[0] = 2
    else:
        kwargs["dtype"] = "float64"
    with pytest.raises(ValueError, match="prepared"):
        solver.solve_field(target, weights, **kwargs)


def test_tensor_result_stays_on_requested_device_with_exact_fixed_values():
    target, weights = observations(torch.float64)
    fixed = np.full(weights.shape, np.nan)
    fixed[0, 0, 0] = 1.234567890123
    field, receipt = solver.solve_field(target, weights, fixed=fixed, dtype="float64", return_tensor=True)
    assert isinstance(field, torch.Tensor) and field.dtype == torch.float64
    assert field.device.type == "cpu" and not field.requires_grad and receipt["converged"]
    assert float(field[0, 0, 0]) == fixed[0, 0, 0]


def test_relations_do_not_downcast_explicit_float64():
    target, weights = observations(torch.float64)
    result, info = solve_with_relations(target.numpy(), weights.numpy(), dtype="float64", rtol=1e-12)
    assert result.dtype == np.float64 and info["converged"]


@pytest.mark.parametrize("dtype", ["float16", "complex128", None, True])
def test_unsupported_precision_is_rejected(dtype):
    with pytest.raises(ValueError, match="dtype"):
        solver.solve_field(*observations(), dtype=dtype)


def test_invalid_layout_and_operator_precision_are_rejected():
    c = QuadraticConstraints(csr_matrix([[1.0, -1.0]]))
    with pytest.raises(ValueError):
        c.operator("cpu", np.float16)
    with pytest.raises(ValueError):
        c.operator("cpu", layout="dense")


def test_prepared_rejects_unsupported_pairs_and_reference_penalty():
    target, weights = observations()
    for options in ({"pairs": [[0, 1]]}, {"reference": np.zeros(weights.shape), "reference_weight": 1.0}):
        with pytest.raises(ValueError, match="prepared"):
            solver.solve_field(target, weights, prepared=solver.PreparedField(), **options)


def test_float64_rejects_nonfinite_cast_but_supports_values_above_float32_range():
    target = np.zeros((1, 1, 2, 3))
    fixed = np.array([[[1e40, 1e40]]])
    result, info = solver.solve_field(target, np.ones((1, 1, 2)), fixed=fixed, dtype="float64")
    assert info["converged"]
    np.testing.assert_array_equal(result, fixed)
    with pytest.raises(ValueError, match="nonfinite"):
        solver.solve_field(target, np.ones((1, 1, 2)), fixed=np.full((1, 1, 2), np.inf), dtype="float64")


def test_target_rebinding_rejects_complex_values_without_discarding_imaginary_part():
    c = QuadraticConstraints(csr_matrix([[1.0, -1.0]]))
    with pytest.raises(ValueError, match="real"):
        c.with_targets(np.array([1.0 + 2.0j]))


def test_prepared_fixed_values_are_reassembled_each_time():
    target, weights = observations(torch.float64)
    prepared = solver.PreparedField()
    fixed = np.full(weights.shape, np.nan)
    for anchor in (0.0, 2.34567890123):
        fixed[0, 0, 0] = anchor
        actual, _ = solver.solve_field(target, weights, fixed=fixed, dtype="float64", prepared=prepared)
        expected, _ = solver.solve_field(target, weights, fixed=fixed, dtype="float64")
        np.testing.assert_array_equal(actual, expected)


def test_default_solver_preserves_legacy_operator_device_only_protocol():
    class LegacyConstraints(QuadraticConstraints):
        def operator(self, device):
            return super().operator(device)

    target = np.zeros((1, 1, 3, 3), np.float32)
    target[..., 0] = 1.0
    rows = LegacyConstraints(csr_matrix([[1.0, 0.0, -1.0]]), weights=3.0)
    field, receipt = solver.solve_field(target, np.ones((1, 1, 3)), constraints=rows, rtol=1e-6)
    assert receipt["converged"]
    np.testing.assert_allclose(field.ravel(), [0.0, 1 / 7, 2 / 7], atol=1e-5)
