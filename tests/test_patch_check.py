"""Synthetic checks for the human-surface evaluator; no CT or GPU required."""

import numpy as np
import pytest


def test_sampling_preserves_xyz_origin_spacing_and_rejects_outside():
    from patch_check import sample_field

    z, y, x = np.indices((4, 5, 6))
    field = (100 * z + 10 * y + x).astype(float)
    points = np.array([[11, 23, 36], [10, 20, 30], [20, 32, 42], [9, 20, 30], [-1, -1, -1], [np.nan, 20, 30]])
    got = sample_field(field, points, [10, 20, 30], [2, 3, 4])
    np.testing.assert_allclose(got[:3], [160.5, 0, 345])
    assert np.isnan(got[3:]).all()  # clipping would falsely credit coverage


def test_patch_drift_uses_fixed_step_not_a_fit_to_human_values():
    from patch_check import evaluate_patch

    field = np.broadcast_to(2 * np.arange(4), (3, 3, 4)).astype(float)
    xyz = np.array([[[0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]]], float)
    result, residual = evaluate_patch(field, xyz, [0, 0, 0], [1, 1, 1], 2, 0, 2)
    assert result["covered_points"] == 4
    assert result["within_1_sheet_pct"] == 50
    np.testing.assert_allclose(residual, [[-1.5, -0.5, 0.5, 1.5]])
    shifted, _ = evaluate_patch(field + 93, xyz, [0, 0, 0], [1, 1, 1], 2, 0, 2)
    assert shifted["abs_p90_sheets"] == result["abs_p90_sheets"]


def test_edges_do_not_cross_holes_or_long_spatial_gaps():
    from patch_check import evaluate_patch

    field = np.broadcast_to(2 * np.arange(8), (3, 3, 8)).astype(float)
    xyz = np.array([[[0, 1, 1], [1, 1, 1], [-1, -1, -1], [2, 1, 1], [7, 1, 1], [20, 1, 1]]], float)
    result, _ = evaluate_patch(field, xyz, [0, 0, 0], [1, 1, 1], 1, 0, 2)
    assert result["valid_points"] == 5
    assert result["covered_points"] == 4
    assert result["covered_edges"] == 1
    assert result["edge_jump_gt1_pct"] == 100


def test_no_coverage_is_not_a_pass_and_invalid_scale_is_rejected():
    from patch_check import evaluate_patch

    xyz = np.full((2, 2, 3), 100.0)
    result, _ = evaluate_patch(np.zeros((4, 4, 4)), xyz, [0] * 3, [1] * 3, 1, 0, 2)
    assert result["covered_points"] == 0
    assert result["abs_p90_sheets"] is None
    with pytest.raises(ValueError):
        evaluate_patch(np.zeros((4, 4, 4)), xyz, [0] * 3, [1] * 3, 0, 0, 2)


def test_tifxyz_sentinel_stays_unavailable_inside_a_negative_origin_field():
    from patch_check import evaluate_patch

    xyz = np.array([[[0, 0, 0], [-1, -1, -1]]], float)
    result, residual = evaluate_patch(np.zeros((4, 4, 4)), xyz, [-2] * 3, [1] * 3, 1, margin=0)
    assert result["valid_points"] == result["covered_points"] == 1
    assert np.isnan(residual[0, 1])


def test_configuration_cannot_silently_claim_rgt(monkeypatch, paris_inputs):
    from paris_solve import run_config

    monkeypatch.setenv("ORIENT", "rgt")
    with pytest.raises(ValueError, match="orientation"):
        run_config(paris_inputs.config)
    monkeypatch.setenv("ORIENT", "umbi")
    monkeypatch.setenv("LAM", "10")
    monkeypatch.setenv("AFV", "/tmp/independent.afv")
    config = run_config(paris_inputs.config)
    assert config["lam"] == 10
    assert config["afv"] == "/tmp/independent.afv"


def test_disconnected_turns_are_not_forced_to_one_winding():
    from patch_check import evaluate_patch

    field = np.broadcast_to(np.arange(8), (3, 3, 8)).astype(float)
    xyz = np.array([[[0, 1, 1], [1, 1, 1], [-1, -1, -1], [6, 1, 1], [7, 1, 1]]], float)
    result, residual = evaluate_patch(field, xyz, [0] * 3, [1] * 3, 1, 0, 2)
    assert result["connected_components"] == 2
    assert result["local"]["within_1_sheet_pct"] == 100
    assert result["within_1_sheet_pct"] == 0  # global constancy is the wrong assumption


def test_lasagna_normals_decode_hemisphere_not_cosine_phase():
    from common import decode_lasagna_normals

    got = decode_lasagna_normals(np.array([128, 255, 128, 255]), np.array([128, 128, 255, 255]))
    np.testing.assert_allclose(got[:3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(got, axis=-1), 1, atol=1e-6)
    assert (got[:, 2] >= 0).all()


def test_paris_non_cubic_window_has_zyx_shape():
    from paris_solve import working_shape

    assert working_shape(np.array([10, 20, 30]), np.array([30, 60, 90]), spacing=2) == (30, 20, 10)


@pytest.mark.parametrize("problem", [None, "format", "dimensions", "missing_channel"])
def test_patch_loader_preserves_xyz_and_rejects_incomplete_grids(tmp_path, problem):
    import json

    import tifffile

    from patch_check import load_patch

    meta = {"format": "other" if problem == "format" else "tifxyz", "uuid": "synthetic", "scale": [10, 20]}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    shape = (2, 3, 4) if problem == "dimensions" else (2, 3)
    for i, axis in enumerate("xyz"):
        if problem == "missing_channel" and axis == "z":
            continue
        tifffile.imwrite(tmp_path / f"{axis}.tif", np.full(shape, i + 2, np.float32), photometric="minisblack")
    if problem is None:
        actual, xyz = load_patch(tmp_path)
        assert actual == meta and xyz.shape == (2, 3, 3)
        np.testing.assert_array_equal(xyz, np.broadcast_to([2, 3, 4], xyz.shape))
    else:
        with pytest.raises(FileNotFoundError if problem == "missing_channel" else ValueError):
            load_patch(tmp_path)
