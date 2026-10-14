"""Synthetic installed-package smoke test: python -I examples/minimal_field.py."""

import numpy as np

from field_solver import solve_field

shape = (1, 4, 6)  # ZYX
target = np.zeros((*shape, 3), np.float32)
target[..., 0] = 1.0  # X gradient in field units per physical coordinate unit.
field, receipt = solve_field(target, np.ones(shape), spacing_xyz=(1, 1, 1))
assert receipt["converged"], receipt
np.testing.assert_allclose(field, np.broadcast_to(np.arange(6), shape), atol=1e-4)
print("converged:", receipt["converged"])
print("field[0,0,:]:", np.round(field[0, 0, :], 3).tolist())
