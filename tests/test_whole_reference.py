import json
import sys

import numpy as np
import pytest


@pytest.fixture
def run_global(paris_inputs, monkeypatch):
    import whole_reference

    def run(name="global", *options):
        output = paris_inputs.root / name
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whole_reference",
                "--output",
                str(output),
                "--development",
                str(paris_inputs.development),
                "--step",
                "16",
                "--device",
                "cpu",
                "--data-root",
                str(paris_inputs.root),
                *map(str, options),
            ],
        )
        whole_reference.main()
        return output

    return run


def fibre_db(path, rows):
    import json
    import sqlite3

    c = sqlite3.connect(path)
    c.executescript("""CREATE TABLE fibers(id INTEGER PRIMARY KEY, annotation TEXT);
        CREATE TABLE blocks(fiber_id INTEGER, first_segment INTEGER, points BLOB);""")
    for fid, points, source in rows:
        c.execute("INSERT INTO fibers VALUES(?,?)", (fid, json.dumps(source)))
        c.execute("INSERT INTO blocks VALUES(?,?,?)", (fid, 0, np.asarray(points, dtype="<f8").tobytes()))
    c.commit()
    c.close()


def test_only_black_outer_samples_become_anchors():
    from whole_reference import pin_outer_face

    fixed = np.full((5, 6, 7), np.nan)
    values = np.zeros((1, 6, 7))
    values[0, 2, 3] = 10
    pin_outer_face(fixed, values, 0, 0)
    assert np.isfinite(fixed).sum() == 41
    assert np.isnan(fixed[0, 2, 3]) and np.isnan(fixed[2, 2, 3])
    with pytest.raises(ValueError):
        pin_outer_face(fixed, values, 0, 2)


def test_global_fibres_reach_solver_and_change_solution(tmp_path):
    import whole_reference as w

    assert hasattr(w, "global_fibre_constraints"), "global fibres are not implemented"
    db = tmp_path / "f.afv"
    fibre_db(db, [(1, [[0, 0, 0], [256, 0, 0]], {})])
    pairs, audit = w.global_fibre_constraints(db, [0, 0, 0], [32, 32, 32], [1, 1, 3])
    np.testing.assert_array_equal(pairs, [[0, 2]])
    from field_solver import solve_field

    g = np.zeros((1, 1, 3, 3), np.float32)
    g[..., 0] = 1 / 32
    baseline, _ = solve_field(g, np.ones((1, 1, 3)), [32] * 3, rtol=1e-6)
    new, info = w.solve_global_reference(g, np.ones((1, 1, 3)), [32] * 3, pairs, None, "cpu")
    assert info["converged"] and audit["used_fibre_ids"] == [1]
    np.testing.assert_allclose(baseline.ravel(), [0, 1, 2], atol=1e-5)
    np.testing.assert_allclose(new.ravel(), [0, 1 / 7, 2 / 7], atol=1e-5)


def test_global_selection_has_no_400_fibre_or_2000_pair_cap(tmp_path):
    import whole_reference as w

    assert hasattr(w, "global_fibre_constraints"), "global fibres are not implemented"
    db = tmp_path / "f.afv"
    points = [[i * 128, 0, 0] for i in range(8)]
    fibre_db(db, [(i, points, {}) for i in range(501)])
    pairs, audit = w.global_fibre_constraints(db, [0, 0, 0], [32] * 3, [1, 1, 8])
    assert len(audit["used_fibre_ids"]) == 501
    assert len(pairs) == 501 * 8


def test_global_exclusion_follows_ids_and_recorded_lineage(tmp_path):
    import whole_reference as w

    assert hasattr(w, "global_fibre_constraints"), "global fibres are not implemented"
    db = tmp_path / "f.afv"
    points = [[0, 0, 0], [256, 0, 0]]
    fibre_db(
        db,
        [
            (1, points, {"sourceCollectionUUID": "A", "sourceFiberIDs": [10]}),
            (2, points, {"sourceCollectionUUID": "A", "sourceFiberIDs": [10, 11]}),
            (3, points, {"sourceCollectionUUID": "B", "sourceFiberIDs": [10]}),
        ],
    )
    pairs, audit = w.global_fibre_constraints(
        db, [0, 0, 0], [32] * 3, [1, 1, 3], excluded_ids={1}, excluded_lineage={("A", "10")}
    )
    assert audit["used_fibre_ids"] == [3]
    assert audit["excluded_fibres"] == 2
    assert len(pairs) == 1


def test_global_fibres_do_not_bridge_outside_runs_or_clamp(tmp_path):
    import whole_reference as w

    assert hasattr(w, "global_fibre_constraints"), "global fibres are not implemented"
    db = tmp_path / "f.afv"
    fibre_db(db, [(1, [[0, 0, 0], [512, 0, 0], [256, 0, 0]], {}), (2, [[0, 0, 0], [1, 0, 0]], {})])
    pairs, audit = w.global_fibre_constraints(db, [0, 0, 0], [32] * 3, [1, 1, 3])
    assert pairs.shape == (0, 2)
    assert audit["used_fibre_ids"] == []


def test_global_constraint_geometry_checks_non_cubic_physical_frame(tmp_path):
    import whole_reference as w

    assert hasattr(w, "global_fibre_constraints"), "global fibres are not implemented"
    db = tmp_path / "f.afv"
    fibre_db(db, [(1, [[40, 80, 120], [296, 848, 1656]], {})])
    pairs, _ = w.global_fibre_constraints(db, [10, 20, 30], [32, 64, 96], [5, 4, 3])
    np.testing.assert_array_equal(pairs, [[0, 59]])
    with pytest.raises(ValueError):
        w.global_fibre_constraints(db, [10, 20, 30], [0, 64, 96], [5, 4, 3])


def test_reservation_rejects_another_afv_with_same_integer_ids(tmp_path):
    import whole_reference as w

    assert hasattr(w, "validate_reservation"), "AFV identity is not enforced"
    a = tmp_path / "a.afv"
    b = tmp_path / "b.afv"
    points = [[0, 0, 0], [256, 0, 0]]
    fibre_db(a, [(1, points, {"sourceCollectionUUID": "A", "sourceFiberIDs": [1]})])
    fibre_db(b, [(1, points, {"sourceCollectionUUID": "B", "sourceFiberIDs": [1]})])
    reservation = dict(fibre_ids=[1], source_lineage=[["A", "1"]], afv_sha256=w.sha256(a))
    w.validate_reservation(reservation, a)
    with pytest.raises(ValueError, match="AFV"):
        w.validate_reservation(reservation, b)


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.3, 1.0, 3.0])
def test_global_weight_changes_actual_analytic_solution(lam):
    import inspect

    from whole_reference import solve_global_reference

    assert "lam" in inspect.signature(solve_global_reference).parameters, "global weight is hardcoded"
    g = np.zeros((1, 1, 3, 3), np.float32)
    g[..., 0] = 1 / 32
    field, info = solve_global_reference(g, np.ones((1, 1, 3)), [32] * 3, np.array([[0, 2]]), None, "cpu", lam=lam)
    assert info["converged"]
    np.testing.assert_allclose(field.ravel(), [0, 1 / (1 + 2 * lam), 2 / (1 + 2 * lam)], atol=1e-5)


@pytest.mark.parametrize("lam", [-1.0, float("nan"), float("inf")])
def test_global_weight_rejects_invalid_values(lam):
    import inspect

    from whole_reference import solve_global_reference

    assert "lam" in inspect.signature(solve_global_reference).parameters, "global weight is hardcoded"
    with pytest.raises(ValueError):
        solve_global_reference(
            np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3)), [32] * 3, np.array([[0, 2]]), None, "cpu", lam=lam
        )


@pytest.mark.parametrize("mode", ["nodal", "trilinear_pairs", "fibre_consensus", "robust_consensus"])
def test_global_cli_publishes_only_hash_bound_converged_fields(run_global, mode):
    from patch_check import sha256

    output = run_global("global", "--constraint-mode", mode)
    field = np.load(output / "coarse.npy")
    mask = np.load(output / "outer_anchor_mask.npy")
    meta = json.loads((output / "coarse.npy.json").read_text())
    audit = json.loads((output / "fibre_constraints.json").read_text())
    assert field.shape == mask.shape == (3, 4, 5)
    assert mask.sum() == 54 and not mask[1, 1, 2]
    np.testing.assert_array_equal(field[mask], np.zeros(54))
    assert np.isfinite(field).all() and np.any(abs(field[~mask]) > 1e-4)
    assert meta["solver"]["converged"] and meta["field_sha256"] == sha256(output / "coarse.npy")
    assert meta["step"] == 2 and audit["used_fibre_ids"] == [1, 2]
    assert audit["pairs_file_sha256"] == sha256(output / "fibre_pairs.npy")
    assert (output / "DONE").exists()
    if mode != "nodal":
        continuous = json.loads((output / "continuous-audit.json").read_text())
        assert continuous["mode"] == mode
        assert np.isfinite(np.load(output / "robust_factors.npy")).all()
        assert (output / "continuous_constraints.npz").exists()


@pytest.mark.parametrize("change", ["none", "field", "observations", "scale", "mask"])
def test_global_baseline_guard_binds_data_scale_and_anchors(paris_inputs, run_global, change):
    from paris_solve import LB

    baseline = run_global("baseline") / "coarse.npy"
    if change == "field":
        baseline.write_bytes(baseline.read_bytes() + b"changed")
    elif change == "observations":
        paris_inputs.volumes[f"{LB}_grad_mag.ome.zarr/5"] += 10
    elif change == "scale":
        rows = json.loads(paris_inputs.development.read_text())
        rows["rows"][0]["phi_b"] = 3
        paris_inputs.development.write_text(json.dumps(rows))
    elif change == "mask":
        mask = np.load(baseline.parent / "outer_anchor_mask.npy")
        mask[1, 1, 2] = True
        np.save(baseline.parent / "outer_anchor_mask.npy", mask)
    if change == "none":
        output = run_global("candidate", "--baseline-field", baseline)
        np.testing.assert_array_equal(np.load(output / "coarse.npy"), np.load(baseline))
    else:
        with pytest.raises(ValueError, match="baseline"):
            run_global("candidate", "--baseline-field", baseline)
        assert not (paris_inputs.root / "candidate" / "DONE").exists()
        assert not (paris_inputs.root / "candidate" / "coarse.npy").exists()


@pytest.mark.parametrize("problem", ["anchors", "fibres", "numerical_overflow"])
def test_global_cli_refuses_missing_evidence_and_failed_solves(paris_inputs, run_global, problem):
    from paris_solve import CTB, LB
    from patch_check import sha256

    options = []
    if problem == "anchors":
        paris_inputs.volumes[CTB + "/5"].fill(1)
    elif problem == "fibres":
        reservation = paris_inputs.root / "exclude.json"
        reservation.write_text(
            json.dumps(dict(fibre_ids=[1, 2], source_lineage=[], afv_sha256=sha256(paris_inputs.afv)))
        )
        options = ["--exclude-fibres", reservation]
    else:
        paris_inputs.volumes[f"{LB}_grad_mag.ome.zarr/5"] *= 1e30
    with pytest.raises(ValueError, match="anchors|nonempty|cannot publish"):
        run_global("failed", *options)
    assert not (paris_inputs.root / "failed" / "DONE").exists()
    assert not (paris_inputs.root / "failed" / "coarse.npy").exists()


@pytest.mark.parametrize("fibres,lineage", [([True], []), ([-1], []), ([], [["A"]]), ([], [["A", 1]])])
def test_reservation_rejects_malformed_identity_lists(paris_inputs, fibres, lineage):
    from whole_reference import validate_reservation

    with pytest.raises(ValueError, match="invalid fibre exclusion"):
        validate_reservation(dict(fibre_ids=fibres, source_lineage=lineage), paris_inputs.afv)
