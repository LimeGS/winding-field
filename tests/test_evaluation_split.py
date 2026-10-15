"""Geometry-only split controls; no scores or winding labels enter this module."""

import json

import numpy as np
import pytest


def box(x, z=20000):
    return np.array([[x, 0, z], [x + 10, 10, z + 10]], float)


def assignments(manifest):
    return {cid: group["split"] for group in manifest["groups"] for cid in group["collection_ids"]}


def test_buffered_overlap_is_transitive_and_development_propagates():
    from evaluation_split import build_split

    colls = {"dev": box(0), "middle": box(100), "end": box(200), "far": box(2000)}
    manifest = build_split(colls, ["dev"], buffer_voxels=50, frame="test/world")
    splits = assignments(manifest)
    assert splits["dev"] == splits["middle"] == splits["end"] == "development"
    assert splits["far"] in ("validation", "test")
    assert len(manifest["groups"]) == 2
    assert manifest["counts"]["collections"] == 4


def test_historical_z_band_and_inspected_roi_boxes_remain_explicit_development_context():
    from evaluation_split import build_split

    colls = {
        "historical": box(100, 10000),
        "context": box(100, 9800),
        "roi_only": np.array([[4000, 2800, 9700], [4010, 2810, 9710]]),
        "new": box(50000, 20000),
    }
    splits = assignments(
        build_split(
            colls,
            [],
            buffer_voxels=256,
            frame="test/world",
            development_z_context=[9800, 11200],
            development_boxes=[
                dict(id="previously-inspected", min_xyz=[3920, 2592, 10256], max_xyz=[4432, 3104, 10768]),
                dict(id="previous-pilot", min_xyz=[3840, 2736, 10080], max_xyz=[4352, 3248, 10592]),
            ],
        )
    )
    assert all(splits[c] == "development" for c in ("historical", "context", "roi_only"))
    assert splits["new"] in ("validation", "test")


def test_extra_development_boxes_and_shared_surface_lineage_are_preserved():
    from evaluation_split import build_split

    colls = {"near_box": box(1000), "same_surface_far": box(9000), "new": box(30000)}
    devboxes = [{"id": "inspected", "min_xyz": [999, 0, 20000], "max_xyz": [1020, 20, 20020]}]
    manifest = build_split(
        colls,
        [],
        buffer_voxels=0,
        development_boxes=devboxes,
        surface_ids={"near_box": ["surface-1"], "same_surface_far": ["surface-1"]},
        frame="test/world",
    )
    splits = assignments(manifest)
    assert splits["near_box"] == splits["same_surface_far"] == "development"
    assert splits["new"] != "development"
    assert any(g["collection_ids"] == ["near_box", "same_surface_far"] for g in manifest["groups"])


def test_split_and_manifest_hash_are_deterministic_under_input_order():
    from evaluation_split import build_split

    colls = {f"c{i}": box(i * 2000) for i in range(20)}
    first = build_split(colls, [], buffer_voxels=10, frame="test/world")
    reordered = {key: value[::-1] for key, value in reversed(list(colls.items()))}
    assert first == build_split(reordered, [], buffer_voxels=10, frame="test/world")
    assert set(assignments(first).values()) == {"validation", "test"}
    assert len(first["manifest_sha256"]) == 64
    assert first["blind_test_complete"] is False
    assert any("lineage" in limitation for limitation in first["limitations"])
    assert any("single" in limitation.lower() for limitation in first["limitations"])
    json.dumps(first, allow_nan=False)


def test_min_z_keeps_whole_components_in_development_instead_of_dropping_points():
    from evaluation_split import build_split

    colls = {"low": box(0, 17000), "high_same": box(5000, 20000), "new": box(20000, 21000)}
    manifest = build_split(colls, [], min_z=19000, surface_ids={"low": ["s"], "high_same": ["s"]}, frame="test/world")
    splits = assignments(manifest)
    assert splits["low"] == splits["high_same"] == "development"
    assert manifest["counts"]["collections"] == 3


@pytest.mark.parametrize(
    "colls,ids,kwargs",
    [
        ({"c": box(0)}, ["unknown"], {}),
        ({"c": np.array([[0, 0, np.nan]])}, [], {}),
        ({"c": np.empty((0, 3))}, [], {}),
        ({"c": box(0)}, [], {"buffer_voxels": -1}),
        ({"c": box(0)}, [], {"buffer_voxels": float("nan")}),
        ({"c": box(0)}, [], {"surface_ids": {"c": "not-a-list"}}),
        ({"c": box(0)}, [], {"development_boxes": [{"id": "bad", "min_xyz": [2, 0, 0], "max_xyz": [1, 1, 1]}]}),
    ],
)
def test_invalid_geometry_or_unresolved_development_ids_fail_closed(colls, ids, kwargs):
    from evaluation_split import build_split

    with pytest.raises(ValueError):
        build_split(colls, ids, frame="test/world", **kwargs)


def test_new_dataset_does_not_inherit_inspected_windows_or_positive_origin_assumptions():
    from evaluation_split import build_split

    colls = {"old-coordinate-number": box(100, 10000), "translated": box(-10000, -100)}
    result = build_split(colls, [], buffer_voxels=0, frame="another-volume/world")
    assert result["frame"] == "another-volume/world"
    assert "development" not in assignments(result).values()
    assert result["development_z_context"] is None and result["development_boxes"] == []
