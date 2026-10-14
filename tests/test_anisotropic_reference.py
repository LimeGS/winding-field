import importlib.util

import numpy as np
import pytest


def impl():
    assert importlib.util.find_spec("anisotropic_reference"), "unsigned arrival backend missing"
    import anisotropic_reference as m

    return m


def plane_case():
    n = np.zeros((3, 5, 7, 3))
    n[..., 0] = 1
    seeds = np.zeros((3, 5, 7), bool)
    seeds[:, :, 0] = True
    return n, seeds


def test_unsigned_plane_has_known_solution_and_no_wraparound():
    m = impl()
    n, seeds = plane_case()
    a, info = m.solve_arrival(n, [1] * 3, seeds, c_parallel=0.01, c_perp=1, maxiter=100)
    b, other = m.solve_arrival(-n, [1] * 3, seeds, c_parallel=0.01, c_perp=1, maxiter=100)
    assert info["converged"] and other["converged"]
    np.testing.assert_allclose(a[1, 2], np.arange(7), atol=1e-5)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_unknown_wall_cannot_be_used_as_shortcut():
    m = impl()
    n, seeds = plane_case()
    support = np.ones(seeds.shape, bool)
    support[:, :, 3] = False
    field, info = m.solve_arrival(n, [1] * 3, seeds, support=support, maxiter=100)
    assert info["converged"]
    assert np.isnan(field[:, :, 3:]).all() and info["unreachable_nodes"] > 0


def test_chart_cut_requires_explicit_transition_and_cycles_fail():
    m = impl()
    n, seeds = plane_case()
    charts = np.zeros(seeds.shape, int)
    charts[:, :, 3:] = 1
    field, _ = m.solve_arrival(n, [1] * 3, seeds, charts=charts, maxiter=100)
    assert np.isnan(field[:, :, 3:]).all()
    connected, info = m.solve_arrival(n, [1] * 3, seeds, charts=charts, transitions=[(0, 1, 0)], maxiter=100)
    assert info["converged"] and np.isfinite(connected).all()
    with pytest.raises(ValueError):
        m.solve_arrival(n, [1] * 3, seeds, charts=charts, transitions=[(0, 1, 1), (1, 0, 1)])


def test_nonconverged_arrival_is_explicit_and_unknown_seeds_rejected():
    m = impl()
    n, seeds = plane_case()
    _, info = m.solve_arrival(n, [1] * 3, seeds, maxiter=1)
    assert not info["converged"]
    with pytest.raises(ValueError):
        m.solve_arrival(n, [1] * 3, seeds, support=np.zeros(seeds.shape, bool))


def test_exterior_seeds_do_not_include_open_end_face_material():
    m = impl()
    material = np.ones((5, 5, 5), bool)
    air = np.zeros_like(material)
    air[:, :, 0] = True
    material[:, :, 0] = False
    seeds = m.exterior_seeds(material, air)
    assert seeds[:, :, 0].all() and not seeds[0, :, 1:].any()


def test_oblique_director_reaches_all_connected_supported_nodes():
    m = impl()
    n, seeds = plane_case()
    n[:] = [0.3, 0.4, np.sqrt(0.75)]
    field, info = m.solve_arrival(n, [1, 1.3, 0.8], seeds, maxiter=300, c_parallel=0.1, c_perp=1)
    assert info["converged"] and np.isfinite(field).all() and info["unreachable_nodes"] == 0


def test_partial_crop_air_requires_verified_exterior_membership():
    m = impl()
    material = np.ones((5, 5, 5), bool)
    air = np.zeros_like(material)
    air[:, :, 2] = True
    material[:, :, 2] = False
    verified = np.zeros_like(air)
    import inspect

    assert "verified_exterior" in inspect.signature(m.exterior_seeds).parameters
    assert not m.exterior_seeds(material, air, verified_exterior=verified).any()


def test_rotating_director_does_not_change_isotropic_plane_solution():
    m = impl()
    n, seeds = plane_case()
    n[:] = [0.3, 0.4, np.sqrt(0.75)]
    field, info = m.solve_arrival(n, [1] * 3, seeds, c_parallel=1, c_perp=1, maxiter=100)
    assert info["converged"]
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(7), field.shape), atol=2e-5)


def test_cut_offsets_are_not_silently_merged_into_one_turn():
    m = impl()
    n, seeds = plane_case()
    charts = np.zeros(seeds.shape, int)
    charts[:, :, 3:] = 1
    field, info = m.solve_arrival(
        n, [1] * 3, seeds, charts=charts, transitions=[(0, 1, 1)], maxiter=100, c_parallel=1, c_perp=1
    )
    assert info["converged"] and info["chart_offsets"] == {"0": 0.0, "1": 1.0}
    np.testing.assert_allclose(field[1, 2], np.array([0, 1, 2, 2, 3, 4, 5]), atol=2e-5)
    # Canonical lifted potential retains the crossing increment, chart-local scalar alone is not globally one turn.
    np.testing.assert_allclose(field[1, 2] + charts[1, 2], np.arange(7), atol=2e-5)


def test_stencil_cache_can_be_disabled_without_changing_solution():
    m = impl()
    n, seeds = plane_case()
    import inspect

    assert "max_cache_bytes" in inspect.signature(m.solve_arrival).parameters
    a, info = m.solve_arrival(n, [1] * 3, seeds, max_cache_bytes=0, maxiter=100)
    b, other = m.solve_arrival(n, [1] * 3, seeds, max_cache_bytes=1024 * 1024, maxiter=100)
    assert info["converged"] and other["converged"] and info["stencil_cache_bytes"] == 0
    np.testing.assert_array_equal(a, b)


def test_concentric_curved_layers_preserve_radial_separation():
    m = impl()
    z, y, x = np.indices((3, 17, 17))
    dx = x - 8.0
    dy = y - 8.0
    radius = np.hypot(dx, dy)
    n = np.stack([dx / np.maximum(radius, 1), dy / np.maximum(radius, 1), np.zeros_like(dx)], axis=-1)
    support = radius >= 1
    seeds = radius >= 7
    field, info = m.solve_arrival(n, [1] * 3, seeds, support=support, c_parallel=0.01, c_perp=1, maxiter=300)
    assert info["converged"]
    # Known radial metric: arrival increases by ~two units from radius5 to radius3.
    separation = field[1, 8, 11] - field[1, 8, 13]
    assert 1.6 < separation < 2.4
    assert abs(field[1, 11, 12] - field[1, 8, 13]) < 0.8  # Same radius5, different angle.
