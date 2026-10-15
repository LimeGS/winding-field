"""Legacy selection boundaries and spatial grouping, with no remote data."""

import json

import numpy as np
import pytest

import paris_solve


@pytest.mark.parametrize("afv", ["/tmp/independent.afv", ""])
def test_explicit_afv_does_not_require_a_default_data_directory(paris_inputs, monkeypatch, afv):
    monkeypatch.setenv("WREV_DATA", "/deliberately/absent")
    monkeypatch.setenv("AFV", afv)
    monkeypatch.setenv("ORIENT", "umbi")
    monkeypatch.delenv("RGT", raising=False)
    if afv:
        assert paris_solve.run_config(paris_inputs.config)["afv"] == afv
    else:
        with pytest.raises(ValueError, match="empty"):
            paris_solve.run_config(paris_inputs.config)


def test_pair_loader_keeps_context_but_scores_only_eligible_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("WREV_DATA", str(tmp_path))
    collections = {
        "selected": {
            "points": {
                "a": {"p": [1, 2, 100], "wind_a": 0},
                "b": {"p": [2, 2, 199], "wind_a": 1},
                "upper_boundary": {"p": [3, 2, 200], "wind_a": 2},
                "d": {"p": [4, 2, 150], "wind_a": 7},
            }
        },
        "context": {"points": {"a": {"p": [1, 2, 90], "wind_a": 0}, "b": {"p": [1, 2, 95], "wind_a": 1}}},
        "far": {"points": {"a": {"p": [1, 2, 500], "wind_a": 0}, "b": {"p": [1, 2, 501], "wind_a": 1}}},
        "singleton": {"points": {"a": {"p": [1, 2, 100], "wind_a": 0}}},
    }
    (tmp_path / "relwind.json").write_text(json.dumps({"collections": collections}))
    pairs, context = paris_solve.load_pairs(tmp_path / "relwind.json", 100, 200, max_dw=6)
    assert set(context) == {"selected", "context"}
    assert [(a.tolist(), b.tolist(), dw, cid) for a, b, dw, cid in pairs] == [
        ([1, 2, 100], [2, 2, 199], 1, "selected"),
        ([2, 2, 199], [4, 2, 150], 6, "selected"),
    ]


def test_slab_grouping_sorts_by_z_and_splits_before_exceeding_capacity(monkeypatch):
    monkeypatch.setattr(paris_solve, "MARGIN", 1)
    monkeypatch.setattr(paris_solve, "CAP", 16)
    collections = {
        "far": np.array([[0, 0, 40], [2, 2, 42]]),
        "near": np.array([[0, 0, 4], [2, 2, 6]]),
        "first": np.array([[0, 0, 0], [2, 2, 2]]),
    }
    slabs = paris_solve.group_slabs(collections, spacing=2)
    assert [[name for name, _ in members] for members, _ in slabs] == [["first", "near"], ["far"]]
    np.testing.assert_array_equal(slabs[0][1], [[-1, -1, -1], [3, 3, 7]])
    assert paris_solve.group_slabs({}, spacing=2) == []


@pytest.mark.parametrize("name,value", [("LAM", "-1"), ("LAM", "nan"), ("FIBER_LIMIT", "0"), ("PAIR_LIMIT", "0")])
def test_invalid_runtime_limits_are_rejected(paris_inputs, monkeypatch, name, value):
    monkeypatch.setenv("ORIENT", "umbi")
    monkeypatch.delenv("RGT", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="invalid fibre"):
        paris_solve.run_config(paris_inputs.config)


def test_ridge_calibration_remains_a_diagnostic_without_replacing_configured_scale():
    detection = np.zeros((5, 5, 15), bool)
    detection[:, :, 3:5] = detection[:, :, 9:11] = True
    field = np.broadcast_to(np.arange(15, dtype=float), detection.shape)
    assert hasattr(paris_solve, "ridge_diagnostics"), "legacy calibration evidence must remain available"
    result = paris_solve.ridge_diagnostics(detection, field)
    assert result["ridge_axis"] == 2 and result["ridge_count"] == 2
    assert result["ridge_line_counts"] == [2] * 25
    assert result["field_range_98pct"] == 7 and result["ridge_rule_step"] == 3.5
    missing = paris_solve.ridge_diagnostics(np.zeros_like(detection), field)
    assert missing["ridge_rule_step"] is None


@pytest.mark.parametrize("damage", ["entry_hash", "entry_frame", "file_frame", "file_units"])
def test_legacy_annotations_are_bound_before_solving(paris_inputs, monkeypatch, damage):
    from patch_check import sha256

    root = paris_inputs.root
    labels = root / "labels.json"
    labels.write_text(
        json.dumps(
            dict(
                frame="wrong" if damage == "file_frame" else paris_inputs.dataset["frame"],
                voxel_um=96 if damage == "file_units" else 9.6,
                collections={
                    "c": dict(points={"a": dict(p=[16, 16, 16], wind_a=0), "b": dict(p=[32, 16, 16], wind_a=1)})
                },
            )
        )
    )
    entry = dict(
        path=labels.name,
        sha256="0" * 64 if damage == "entry_hash" else sha256(labels),
        frame="wrong" if damage == "entry_frame" else paris_inputs.dataset["frame"],
    )
    data = paris_inputs.raw_config
    data["artifacts"] = dict(annotations=entry)
    paris_inputs.config.write_text(json.dumps(data))
    monkeypatch.setenv("SLAB", "0,80,0,64,0,48")
    monkeypatch.setenv("WREV_DEVICE", "cpu")
    monkeypatch.delenv("PREDICT_ONLY", raising=False)
    monkeypatch.delenv("AUDIT_DIR", raising=False)
    with pytest.raises(ValueError):
        paris_solve.main(["--config", str(paris_inputs.config), "--output", str(root / "field.npy")])
    assert not (root / "field.npy").exists()
