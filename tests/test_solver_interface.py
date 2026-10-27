"""Caller contract checked by both pytest and mypy."""

from typing import assert_type

import numpy as np
import torch
from numpy.typing import NDArray

from field_solver import PreparedField, solve_field


def test_tensor_and_sequence_inputs_keep_the_typed_solver_contract() -> None:
    target = torch.zeros((1, 2, 3, 3))
    target[..., 0] = 1
    field, receipt = solve_field(
        target,
        torch.ones((1, 2, 3)),
        spacing_xyz=[1.0, 1.0, 1.0],
        pairs=[[0, 3]],
        fixed=torch.full((1, 2, 3), float("nan")),
        reference=torch.zeros((1, 2, 3)),
        reference_weight=torch.zeros((1, 2, 3)),
        initial=torch.zeros((1, 2, 3)),
        device=torch.device("cpu"),
        maxiter=np.int64(4000),
    )
    assert_type(field, NDArray[np.float32])
    assert receipt["converged"]
    np.testing.assert_allclose(field, [[[0, 1, 2], [0, 1, 2]]], atol=1e-4)


def test_nested_lists_and_numpy_spacing_are_supported() -> None:
    field, receipt = solve_field(
        [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]],
        [[[1.0, 1.0]]],
        spacing_xyz=np.array([2.0, 1.0, 1.0]),
        pairs=np.empty((0, 2), dtype=np.int64),
    )
    assert receipt["converged"]
    np.testing.assert_allclose(field, [[[0, 2]]], atol=1e-4)


def test_explicit_precision_and_tensor_results_are_typed() -> None:
    target = torch.zeros((1, 2, 3, 3), dtype=torch.float64)
    target[..., 0] = 0.125
    weights = torch.ones((1, 2, 3), dtype=torch.float64)
    prepared = PreparedField()
    field, receipt = solve_field(target, weights, dtype="float64", prepared=prepared)
    assert_type(field, NDArray[np.float64])
    assert field.dtype == np.float64 and receipt["converged"]
    tensor, _ = solve_field(target, weights, dtype="float64", prepared=prepared, return_tensor=True)
    assert_type(tensor, torch.Tensor)
    np.testing.assert_array_equal(tensor.numpy(), field)
