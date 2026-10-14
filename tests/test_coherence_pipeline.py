import importlib.util

import numpy as np
import pytest


def impl():
    assert importlib.util.find_spec("coherence_pipeline"), "five-stage executable pipeline missing"
    import coherence_pipeline as m

    return m


def test_pair_and_consensus_compile_real_points_with_chart_break():
    m = impl()
    fibres = [dict(fibre_id=1, points_xyz=np.array([[0.1, 1, 1], [1.1, 1, 1], [2.1, 1, 1]]), break_before=[0, 0, 1])]
    pair, consensus, audit = m.compile_fibres(fibres, [0, 0, 0], [1] * 3, [4] * 3, voxel_mm=1, pitch_mm=0.15)
    assert pair.trace > 0 and consensus.trace > 0
    assert audit["segments"] == 1 and audit["fibre_ids"] == [1]
    # No sample is transported into the third, cut point.
    assert audit["max_sample_xyz"][0] <= 1.1


def test_full_segment_consensus_is_sampling_invariant_for_affine_field():
    m = impl()
    p = np.array([[0.1, 1, 1], [2.1, 1, 1]])
    a = m.compile_fibres([dict(fibre_id=1, points_xyz=p)], [0] * 3, [1] * 3, [4] * 3, voxel_mm=1, pitch_mm=0.15)[1]
    b = m.compile_fibres([dict(fibre_id=1, points_xyz=p)], [0] * 3, [1] * 3, [4] * 3, voxel_mm=1, pitch_mm=0.075)[1]
    z, y, x = np.indices((4, 4, 4))
    field = x.astype(float)
    # Trapezoid quadrature differs slightly for squared affine residuals, converges with pitch.
    e1 = np.sum(a.weights * a.residual(field) ** 2)
    e2 = np.sum(b.weights * b.residual(field) ** 2)
    assert abs(e1 - e2) / e2 < 0.01


def test_relation_compiler_keeps_order_and_units_and_requires_authority():
    m = impl()
    relations = [
        dict(
            a_xyz=[0.1, 1, 1],
            b_xyz=[2.1, 1, 1],
            min_difference_sheets=0.5,
            authority="prediction_geometry_not_physical_truth",
            order_authority="declared_reference_prior_not_truth",
        )
    ]
    c = m.compile_separation(relations, [0] * 3, [1] * 3, [4] * 3, sheet_step=2)
    assert c.targets.tolist() == [1]
    z, y, x = np.indices((4, 4, 4))
    assert np.isclose((c.matrix @ x.ravel())[0], 2)
    with pytest.raises(ValueError):
        m.compile_separation([dict(relations[0], authority="unknown")], [0] * 3, [1] * 3, [4] * 3, 2)


def test_trace_normalization_keeps_regularization_force_explicit():
    m = impl()
    from scipy.sparse import csr_matrix

    from field_constraints import QuadraticConstraints

    a = QuadraticConstraints(csr_matrix([[1, -1, 0]]), weights=[2])
    b = QuadraticConstraints(csr_matrix([[1, -1, 0], [0, 1, -1]]), weights=[1, 1])
    scaled, receipt = m.match_trace(b, a.trace)
    assert np.isclose(scaled.trace, a.trace) and receipt["scale"] == 1


@pytest.mark.parametrize("kind", ["unknown", "chart"])
def test_path_checks_thin_excursion_not_just_sampled_endpoints(kind):
    m = impl()
    shape = (3, 3, 111)
    support = np.ones(shape, bool)
    charts = np.zeros(shape, int)
    if kind == "unknown":
        support[:, :, 51] = False
    else:
        charts[:, :, 51] = 1
    fibre = dict(fibre_id=1, points_xyz=np.array([[0.1, 1, 1], [99.9, 1, 1]]))
    with pytest.raises(ValueError, match="no usable"):
        m.compile_fibres(
            [fibre], [0] * 3, [1] * 3, shape, voxel_mm=0.001, pitch_mm=0.15, support=support, chart_volume=charts
        )


def test_stencil_cannot_blend_two_charts_inside_a_segment():
    m = impl()
    shape = (3, 3, 4)
    charts = np.zeros(shape, int)
    charts[:, :, 2] = 1
    fibre = dict(fibre_id=1, points_xyz=np.array([[1.1, 1, 1], [1.4, 1, 1]]))
    with pytest.raises(ValueError, match="no usable"):
        m.compile_fibres([fibre], [0] * 3, [1] * 3, shape, voxel_mm=1, chart_volume=charts)


def test_separation_cycle_is_reported_instead_of_compressed():
    m = impl()
    nodes = [[0.1, 1, 1], [1.1, 1, 1], [2.1, 1, 1]]
    relations = [
        dict(
            a_xyz=nodes[i],
            b_xyz=nodes[(i + 1) % 3],
            min_difference_sheets=0.5,
            authority="controlled_synthetic_truth",
            order_authority="controlled_synthetic_truth",
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="cycle"):
        m.compile_separation(relations, [0] * 3, [1] * 3, [4] * 3, 1)
