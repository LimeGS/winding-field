import importlib.util

import numpy as np
import pytest


def impl():
    assert importlib.util.find_spec("field_topology"), "geometry topology missing"
    import field_topology as m

    return m


def test_segmenting_never_joins_unknown_breaks_or_cuts():
    m = impl()
    points = np.column_stack([np.arange(7), np.zeros(7), np.zeros(7)])
    segments = m.split_segments(
        points, valid=[1, 1, 0, 1, 1, 1, 1], charts=[0, 0, 0, 0, 0, 1, 1], break_before=[0, 0, 0, 0, 1, 0, 0]
    )
    assert [s.tolist() for s in segments] == [[0, 1], [5, 6]]


def test_uniform_arclength_quadrature_is_density_invariant():
    m = impl()
    for x in (np.array([0, 1.0]), np.linspace(0, 1, 100)):
        p = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        samples, w = m.sample_segment(p, voxel_mm=1, pitch_mm=0.15)
        assert np.isclose(w.sum(), 1)
        assert np.isclose(np.sum(samples[:, 0] * w), 0.5)


def test_order_cycle_fails_and_consistent_chart_offsets_survive():
    m = impl()
    assert m.chart_offsets([(0, 1, 1), (1, 2, 1), (0, 2, 2)]) == {0: 0.0, 1: 1.0, 2: 2.0}
    with pytest.raises(ValueError):
        m.chart_offsets([(0, 1, 1), (1, 2, 1), (2, 0, 1)])


def test_two_detection_faces_are_not_automatically_two_sheets():
    m = impl()
    offsets = np.arange(101) * 0.01
    values = np.zeros(101)
    values[30] = 1
    values[33] = 1  # 30 micrometres, possible faces/delamination.
    relations = m.profile_relations(offsets, values, min_gap_mm=0.12, max_gap_mm=0.3)
    assert not relations["accepted"] and relations["ambiguous"]
    values[:] = 0
    values[30] = 1
    values[48] = 1
    relations = m.profile_relations(offsets, values, min_gap_mm=0.12, max_gap_mm=0.3)
    assert relations["accepted"] == [(30, 48)]
    assert relations["authority"] == "prediction_geometry_not_physical_truth"


def test_unknown_between_profiles_prevents_relation():
    m = impl()
    offsets = np.arange(101) * 0.01
    v = np.zeros(101)
    v[30] = v[48] = 1
    v[40] = np.nan
    assert not m.profile_relations(offsets, v, 0.12, 0.3)["accepted"]


def test_profile_accepts_only_machine_roundoff_at_probability_boundary():
    m = impl()
    offsets = np.arange(101) * 0.01
    p = np.zeros(101)
    p[30] = np.nextafter(1.0, 2.0)
    p[48] = 1.0
    assert m.profile_relations(offsets, p)["accepted"] == [(30, 48)]
    p[30] = 1.01
    with pytest.raises(ValueError):
        m.profile_relations(offsets, p)


def test_geometry_charts_do_not_connect_separate_parallel_layers():
    m = impl()
    assert hasattr(m, "geometry_charts"), "automatic local chart inventory missing"
    p = np.zeros((3, 5, 7))
    p[:, :, 1] = p[:, :, 5] = 1
    n = np.zeros((*p.shape, 3))
    n[..., 0] = 1
    charts, info = m.geometry_charts(p, n, [0.02] * 3)
    assert info["chart_count"] == 2
    assert charts[1, 2, 1] != charts[1, 2, 5] and charts[1, 2, 3] == -1
    other, _ = m.geometry_charts(p, -n, [0.02] * 3)
    np.testing.assert_array_equal(charts, other)


def test_geometry_charts_exclude_missing_directors_without_deleting_material():
    m = impl()
    assert hasattr(m, "geometry_charts")
    p = np.ones((3, 3, 3))
    n = np.zeros((*p.shape, 3))
    charts, info = m.geometry_charts(p, n, [0.02] * 3)
    assert (charts == -1).all() and info["unresolved_material_nodes"] == 27 and info["material_nodes"] == 27


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_chart_domain_covers_interpolation_halo_but_not_competing_regions(dtype):
    m = impl()
    assert hasattr(m, "chart_domains"), "material chart domains cannot yet support interpolation"
    labels = np.full((3, 3, 5), -1, dtype)
    first, second = np.iinfo(dtype).max - 1, np.iinfo(dtype).max
    labels[:, :, 1] = first
    labels[:, :, 3] = second
    domains = m.chart_domains(labels, np.ones(labels.shape, bool))
    assert (domains[:, :, 0] == first).all() and (domains[:, :, 4] == second).all()
    assert (domains[:, :, 2] == -1).all()  # Equally near incompatible chart observations.
    support = np.ones(labels.shape, bool)
    support[:, :, 0] = False
    assert (m.chart_domains(labels, support)[:, :, 0] == -1).all()


@pytest.fixture
def parallel_layers():
    probability = np.zeros((3, 3, 101))
    probability[:, :, 30:33] = probability[:, :, 50:53] = 1
    directors = np.zeros((*probability.shape, 3))
    directors[..., 0] = 1
    reference = np.broadcast_to(np.arange(101, dtype=float), probability.shape)
    return probability, directors, reference


@pytest.mark.parametrize("sign", [1, -1])
def test_transversal_relations_use_declared_reference_order_and_physical_gap(parallel_layers, sign):
    from field_topology import build_transversal_inventory

    probability, directors, reference = parallel_layers
    args = (probability, directors, [10, 20, 30], [1, 1, 1], 0.0096)
    result = build_transversal_inventory(*args, reference=sign * reference, max_profiles=5, seed=7)
    assert result == build_transversal_inventory(*args, reference=sign * reference, max_profiles=5, seed=7)
    assert result["sampled_profiles"] == 5 and result["relations"]
    for relation in result["relations"]:
        assert relation["physical_gap_mm"] == pytest.approx(0.192)
        assert sign * (relation["b_xyz"][0] - relation["a_xyz"][0]) > 0
        assert relation["authority"] == "prediction_geometry_not_physical_truth"
        assert relation["order_authority"] == "declared_reference_prior_not_truth"


@pytest.mark.parametrize("reference_kind", ["absent", "constant", "nonfinite"])
def test_transversals_without_resolved_order_cannot_emit_constraints(parallel_layers, reference_kind):
    from field_topology import build_transversal_inventory

    probability, directors, _ = parallel_layers
    reference = (
        None
        if reference_kind == "absent"
        else np.full(probability.shape, 0 if reference_kind == "constant" else np.nan)
    )
    result = build_transversal_inventory(
        probability, directors, [0, 0, 0], [1, 1, 1], 0.0096, reference, max_profiles=5
    )
    assert result["relations"] == [] and result["unresolved_order"] > 0


def test_transversal_unknown_gap_is_not_treated_as_air(parallel_layers):
    from field_topology import build_transversal_inventory

    probability, directors, reference = parallel_layers
    support = np.ones(probability.shape, bool)
    support[:, :, 40:43] = False
    result = build_transversal_inventory(
        probability, directors, [0, 0, 0], [1, 1, 1], 0.0096, reference, max_profiles=5, support=support
    )
    assert result["relations"] == [] and result["ambiguous_profiles"] > 0
    directors.fill(0)
    result = build_transversal_inventory(probability, directors, [0, 0, 0], [1, 1, 1], 0.0096, reference)
    assert result["relations"] == []


def test_transversal_invalid_probabilities_are_rejected(parallel_layers):
    from field_topology import build_transversal_inventory

    probability, directors, reference = parallel_layers
    probability[1, 1, 1] = 1.1
    with pytest.raises(ValueError, match="invalid observation"):
        build_transversal_inventory(probability, directors, [0, 0, 0], [1, 1, 1], 0.0096, reference)


def test_transversal_sampling_uses_supplied_physical_pitch(parallel_layers):
    from field_topology import build_transversal_inventory

    probability, directors, reference = parallel_layers
    result = build_transversal_inventory(
        probability, directors, [0, 0, 0], [1, 1, 1], 0.01, reference, max_profiles=5, profile_step_mm=0.01
    )
    assert result["relations"]
    for relation in result["relations"]:
        assert relation["physical_gap_mm"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        build_transversal_inventory(probability, directors, [0, 0, 0], [1, 1, 1], 0.01, reference, profile_step_mm=0)
