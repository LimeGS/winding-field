"""Legacy selection boundaries and spatial grouping, with no remote data."""

import json

import numpy as np
import pytest

import paris_solve


@pytest.mark.parametrize("afv", ["/tmp/independent.afv", ""])
def test_explicit_afv_does_not_require_a_default_data_directory(monkeypatch, afv):
    def unavailable_data_dir():
        raise FileNotFoundError("no data dir (set $WREV_DATA)")

    monkeypatch.setattr(paris_solve, "data_dir", unavailable_data_dir)
    monkeypatch.setenv("AFV", afv)
    monkeypatch.setenv("ORIENT", "umbi")
    monkeypatch.delenv("RGT", raising=False)
    assert paris_solve.run_config()["afv"] == afv


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
    pairs, context = paris_solve.load_pairs(100, 200)
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
    slabs = paris_solve.group_slabs(collections)
    assert [[name for name, _ in members] for members, _ in slabs] == [["first", "near"], ["far"]]
    np.testing.assert_array_equal(slabs[0][1], [[-1, -1, -1], [3, 3, 7]])
    assert paris_solve.group_slabs({}) == []


@pytest.mark.parametrize("name,value", [("LAM", "-1"), ("LAM", "nan"), ("FIBER_LIMIT", "0"), ("PAIR_LIMIT", "0")])
def test_invalid_runtime_limits_are_rejected(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv("WREV_DATA", str(tmp_path))
    monkeypatch.setenv("ORIENT", "umbi")
    monkeypatch.delenv("RGT", raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="invalid fibre"):
        paris_solve.run_config()
