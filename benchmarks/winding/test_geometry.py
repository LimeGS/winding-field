"""Physical plane sampling regressions for the current CT renderer."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def implementation():
    path = Path(__file__).with_name("ct_visuals.py")
    spec = importlib.util.spec_from_file_location("winding_ct_visuals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_cubic_xyz_planes_match_zyx_coordinate_ramp():
    m = implementation()
    z, y, x = np.indices((4, 5, 6))
    field = 10 + x * 2 + 10 * (20 + y * 3) + 100 * (30 + z * 4)
    meta = dict(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4], shape_zyx=[4, 5, 6], step=2)
    for axis in range(3):
        xyz, remaining = m.plane_coordinates([10, 20, 30], [11, 13, 13], axis, 4)
        actual = m.sample_winding(field, meta, xyz)
        expected = (xyz[..., 0] + 10 * xyz[..., 1] + 100 * xyz[..., 2]) / 2
        np.testing.assert_allclose(actual, expected)
        assert np.all(xyz[..., axis] == [14, 24, 34][axis])
        assert remaining == [k for k in range(3) if k != axis]


def test_outside_or_unsupported_stencils_remain_missing():
    m = implementation()
    field = np.ones((3, 3, 3)) * 4
    meta = dict(origin_xyz=[0] * 3, spacing_xyz=[1] * 3, shape_zyx=[3] * 3, step=2)
    xyz = np.array([[[1, 1, 1], [3, 1, 1], [0.5, 0.5, 0.5]]], float)
    support = np.ones(field.shape, bool)
    support[0, 0, 0] = False
    result = m.sample_winding(field, meta, xyz, support)
    assert result[0, 0] == 2
    assert np.isnan(result[0, 1:]).all()
    meta["step"] = 0
    with pytest.raises(ValueError):
        m.sample_winding(field, meta, xyz)


def test_plane_rejects_fractional_coordinates_and_invalid_extent():
    m = implementation()
    with pytest.raises(ValueError):
        m.plane_coordinates([0.5, 0, 0], [3] * 3, 2, 1)
    with pytest.raises(ValueError):
        m.plane_coordinates([0] * 3, [3] * 3, 2, 3)
