import numpy as np
import pytest

from physical_grid import point_stencils


@pytest.mark.parametrize(
    "origin,point,axis,last",
    [
        (
            [22567.68, 33315.840000000004, 14791.68],
            [23137.920000000002, 33795.06780029992, 15080.501872644045],
            0,
            33,
        ),
        (
            [23086.08, 33315.840000000004, 14791.68],
            [23144.457887893528, 33769.60688689749, 15327.36],
            2,
            31,
        ),
    ],
)
def test_real_175a_boundary_keeps_exact_value_and_support(origin, point, axis, last):
    shape = (32, 34, 34)
    spacing = np.full(3, 17.28)
    assert (np.array(point) - origin)[axis] / spacing[axis] > last
    support = np.ones(shape, dtype=bool)
    # The preceding plane has zero interpolation weight at the final node.
    plane = [slice(None)] * 3
    plane[2 - axis] = -2
    support[tuple(plane)] = False
    nodes, weights, valid = point_stencils([point], origin, spacing, shape, support)
    assert valid.tolist() == [True]
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(1), [1], atol=1e-15, rtol=0)
    ramp = np.indices(shape)[2 - axis].ravel()
    np.testing.assert_allclose((ramp[nodes] * weights).sum(1), [last], atol=1e-13, rtol=0)
    plane[2 - axis] = -1
    support[tuple(plane)] = False
    assert not point_stencils([point], origin, spacing, shape, support)[2][0]


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("upper", [False, True])
def test_one_representable_step_outside_physical_boundary_is_rejected(axis, upper):
    origin = np.array([22567.68, 33315.840000000004, 14791.68])
    spacing = np.full(3, 17.28)
    shape = (32, 34, 34)
    high = origin + (np.array(shape)[::-1] - 1) * spacing
    point = origin + spacing
    point[axis] = np.nextafter(high[axis] if upper else origin[axis], np.inf if upper else -np.inf)
    _, weights, valid = point_stencils([point], origin, spacing, shape)
    assert valid.tolist() == [False]
    assert not weights.any()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_boundary_points_remain_invalid(value):
    _, weights, valid = point_stencils([[value, 1, 1]], [0] * 3, [1] * 3, (3, 3, 3))
    assert valid.tolist() == [False]
    assert not weights.any()


def test_imported_stride_172_8um_pair_retains_exact_final_node():
    from field_constraints import pair_constraints

    origin = np.array([137.8, 254., 364.8])
    spacing = np.full(3, 172.8)
    end = np.array([[656.2, 599.6, 1056.]])
    assert ((end - origin) / spacing)[0, 0] > 3
    rows = pair_constraints([[397., 599.6, 1056.]], end, origin, spacing, (6, 6, 4))
    # X ramp: 1.5 at the first endpoint, exactly 3 at the final node.
    ramp = np.indices((6, 6, 4))[2].ravel()
    np.testing.assert_allclose(rows.matrix @ ramp, [-1.5], atol=1e-14, rtol=0)
