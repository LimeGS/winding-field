import numpy as np
import pytest
import torch


def impl():
    import importlib.util

    assert importlib.util.find_spec("field_constraints"), "continuous constraint operator missing"
    import field_constraints

    return field_constraints


def test_stencil_affine_anisotropic_and_boundary_support():
    import physical_grid as p

    assert hasattr(p, "point_stencils"), "physical point stencils missing"
    nodes, weights, valid = p.point_stencils(
        [[13, 27.5, 40], [22, 35, 46], [9, 20, 30]], [10, 20, 30], [2, 3, 4], [5, 6, 7]
    )
    z, y, x = np.indices((5, 6, 7))
    f = 2 * x + 3 * y + 5 * z
    assert valid.tolist() == [True, True, False]
    np.testing.assert_allclose((f.ravel()[nodes[:2]] * weights[:2]).sum(1), [23, 47])
    support = np.ones(f.shape, bool)
    support[2, 2, 1] = False
    assert not p.point_stencils([[13, 27.5, 40]], [10, 20, 30], [2, 3, 4], f.shape, support)[2][0]


def test_pair_diagonal_coalesces_shared_stencil_and_annihilates_constants():
    m = impl()
    from scipy.sparse import csr_matrix

    c = m.QuadraticConstraints(csr_matrix([[0.75, -0.75, 0], [0.5, 0.5, -1]]), weights=[2, 3])
    op = c.operator("cpu", np.float32)
    matrix = c.matrix.toarray()
    want = matrix.T @ np.diag([2, 3]) @ matrix
    np.testing.assert_allclose(op.diagonal.numpy(), np.diag(want), atol=1e-6)
    for i in range(3):
        np.testing.assert_allclose(op.matvec(torch.eye(3)[i]).numpy(), want[:, i], atol=1e-6)
    np.testing.assert_allclose(op.matvec(torch.ones(3)).numpy(), 0, atol=1e-6)


def test_consensus_diagonal_and_matvec_match_eliminated_dense_system():
    m = impl()
    from scipy.sparse import csr_matrix

    a = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1.0]])
    w = np.array([1, 2, 3.0])
    c = m.QuadraticConstraints(csr_matrix(a), w, groups=[0, 0, 0])
    want = a.T @ (np.diag(w) - np.outer(w, w) / w.sum()) @ a
    op = c.operator("cpu", np.float32)
    np.testing.assert_allclose(op.diagonal.numpy(), np.diag(want), atol=1e-6)
    np.testing.assert_allclose(op.matvec(torch.tensor([1.0, 2.0, 4.0])).numpy(), want @ [1, 2, 4], atol=1e-6)


def test_tile_restriction_keeps_external_consensus_samples_and_rhs():
    m = impl()
    from scipy.sparse import csr_matrix

    c = m.QuadraticConstraints(csr_matrix(np.eye(4)), weights=[1, 2, 3, 4], groups=[0] * 4)
    previous = np.array([1, 2, 7, 11.0])
    local = c.restrict([0, 1], previous)
    full = c.operator("cpu", np.float32)
    op = local.operator("cpu", np.float32)
    x = torch.tensor([3.0, 5.0])
    assembled = previous.copy()
    assembled[:2] = x.numpy()
    residual = op.matvec(x) - op.rhs()
    np.testing.assert_allclose(
        residual.numpy(), full.matvec(torch.tensor(assembled, dtype=torch.float32)).numpy()[:2], atol=1e-5
    )


def test_constraint_solver_actually_applies_continuous_rows():
    m = impl()
    import inspect

    from scipy.sparse import csr_matrix

    from field_solver import solve_field

    assert "constraints" in inspect.signature(solve_field).parameters, "solver does not consume continuous constraints"
    g = np.zeros((1, 1, 3, 3), np.float32)
    g[..., 0] = 1
    c = m.QuadraticConstraints(csr_matrix([[1, 0, -1]]), weights=[3])
    field, info = solve_field(g, np.ones((1, 1, 3)), constraints=c, rtol=1e-6)
    assert info["converged"]
    np.testing.assert_allclose(field.ravel(), [0, 1 / 7, 2 / 7], atol=1e-5)


def test_trilinear_pair_does_not_disappear_in_same_rounded_node():
    m = impl()
    c = m.pair_constraints([[0.1, 0, 0]], [[0.4, 0, 0]], [0, 0, 0], [1, 1, 1], [2, 2, 2], weight=1)
    assert c.matrix.nnz > 0 and c.trace > 0
    same = m.pair_constraints([[0.1, 0, 0]], [[0.1, 0, 0]], [0, 0, 0], [1, 1, 1], [2, 2, 2], weight=1)
    assert same.trace == 0


def test_invalid_rows_targets_weights_groups_are_rejected():
    m = impl()
    from scipy.sparse import csr_matrix

    for kwargs in ({"weights": [-1]}, {"targets": [np.nan]}, {"groups": [-2]}):
        with pytest.raises(ValueError):
            m.QuadraticConstraints(csr_matrix([[1, -1]]), **kwargs)
    with pytest.raises(ValueError):
        m.pair_constraints([[99, 0, 0]], [[0.4, 0, 0]], [0, 0, 0], [1] * 3, [2] * 3)


def test_robust_consensus_records_downweighting_without_deleting_samples():
    m = impl()
    from scipy.sparse import csr_matrix

    c = m.QuadraticConstraints(csr_matrix(np.eye(4)), weights=np.ones(4), groups=[0] * 4)
    robust, factors = c.robust_weights(np.array([0, 0, 0, 10.0]), delta=1)
    assert robust.matrix.shape[0] == 4 and factors[-1] < factors[0]
    assert (factors > 0).all()


def test_separation_preserves_signed_minimum_and_detects_nonconvergence():
    m = impl()
    from scipy.sparse import csr_matrix

    g = np.zeros((1, 1, 3, 3), np.float32)
    relation = m.QuadraticConstraints(csr_matrix([[-1, 0, 1]]), weights=[100], targets=[1])
    field, info = m.solve_with_relations(g, np.ones((1, 1, 3)), separation=relation, rtol=1e-6)
    assert info["converged"] and field.ravel()[-1] > 0.99
    assert info["separation_violations"] == 1  # Soft hinge retains its finite violation.
    _, failed = m.solve_with_relations(g, np.ones((1, 1, 3)), separation=relation, maxiter=0)
    assert not failed["converged"]


def test_continuous_consensus_pilot_uses_same_global_equations(tmp_path):
    impl()  # asserts the module exists; the equations come from the helpers below
    from coarse_reference import run_pilot_arrays
    from coherence_pipeline import compile_fibres
    from paris_data import save_field

    shape = (5, 8, 10)
    z, y, x = np.indices(shape)
    g = np.zeros((*shape, 3), np.float32)
    g[..., 0] = 1
    ref = tmp_path / "ref.npy"
    metadata = dict(frame="fixture", origin_xyz=[0, 0, 0], spacing_xyz=[1] * 3, shape_zyx=list(shape))
    save_field(ref, x.astype(np.float32), metadata, dict(converged=True, gauge={"kind": "fixed"}), step=1)
    _, c, _ = compile_fibres(
        [dict(fibre_id=1, points_xyz=np.array([[4, 1, 1], [4, 6, 3]]))], [0] * 3, [1] * 3, shape, voxel_mm=1
    )
    _, assembled, _, receipt = run_pilot_arrays(
        g,
        np.ones(shape),
        1,
        halo=1,
        sweeps=4,
        external_reference=ref,
        external_metadata=tmp_path / "ref.npy.json",
        metadata={"frame": "fixture"},
        constraints=c,
    )
    assert receipt["status"] == "PASS"
    assert receipt["input_hashes"]["continuous_constraints"] == c.digest
    np.testing.assert_allclose(assembled, x, atol=1e-4)


def test_robust_consensus_center_solves_asymmetric_huber_objective():
    m = impl()
    from scipy.sparse import csr_matrix

    shape = (1, 1, 5)
    fixed = np.array([0, 0, np.nan, 0, 10.0]).reshape(shape)
    c = m.QuadraticConstraints(csr_matrix(np.eye(5)), weights=np.full(5, 10.0), groups=[0] * 5)
    field, info = m.solve_with_relations(
        np.zeros((*shape, 3)),
        np.ones(shape),
        constraints=c,
        robust_delta=1,
        outer_iterations=100,
        fixed=fixed,
        rtol=1e-6,
    )
    assert info["converged"]
    # Independent derivative: free x=5c/6, 4c-x=1 => x=5/19.
    assert abs(field.ravel()[2] - 5 / 19) < 0.002


def test_constraint_serialization_roundtrip_rejects_object_payload(tmp_path):
    m = impl()
    from scipy.sparse import csr_matrix

    c = m.QuadraticConstraints(csr_matrix([[1, -1, 0], [0, 1, -1]]), weights=[2, 3], groups=[0, 0])
    assert hasattr(m.QuadraticConstraints, "load"), "constraint loader missing"
    p = tmp_path / "c.npz"
    c.save(p)
    restored = m.QuadraticConstraints.load(p)
    assert restored.digest == c.digest
    np.savez(
        tmp_path / "bad.npz",
        data=np.array([{}], object),
        indices=[0],
        indptr=[0, 1],
        shape=[1, 3],
        weights=[1],
        targets=[0],
        groups=[-1],
    )
    with pytest.raises(ValueError):
        m.QuadraticConstraints.load(tmp_path / "bad.npz")
