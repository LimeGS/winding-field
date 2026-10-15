"""Offline pair receipts: known answers, unavailable data and degenerate fields."""

import json

import numpy as np
import pytest

FRAME = "PHercParis4/20260411134726:L2"


def metadata(shape=(3, 3, 8)):
    return {"frame": FRAME, "shape_zyx": list(shape), "origin_xyz": [0, 0, 0], "spacing_xyz": [1, 1, 1], "step": 1}


def record(pid="p", a=(0, 1, 1), b=(2, 1, 1), dw=2, collection="c"):
    return {
        "pair_id": pid,
        "collection_id": collection,
        "point_a_id": pid + "a",
        "point_b_id": pid + "b",
        "a_xyz": list(a),
        "b_xyz": list(b),
        "wind_a": 10,
        "wind_b": 10 + dw,
        "dw": dw,
    }


def test_loader_preserves_legacy_selection_boundaries_and_stable_identity(tmp_path):
    from paris_pair_check import load_pair_records

    points = {
        "a": {"p": [1, 1, 10000], "wind_a": 0},
        "b": {"p": [2, 1, 10999], "wind_a": 1},
        "c": {"p": [3, 1, 10001], "wind_a": 6},
        "d": {"p": [4, 1, 11000], "wind_a": 2},
        "e": {"p": [5, 1, 9999], "wind_a": 2},
        "f": {"p": [6, 1, 10002], "wind_a": 0},
        "g": {"p": [7, 1, 10003], "wind_a": 20},
    }
    path = tmp_path / "relwind.json"
    path.write_text(json.dumps({"collections": {"c": {"points": points}}}))
    rows, colls, exclusions = load_pair_records(path, 10000, 11000, max_dw=6)
    assert {(r["point_a_id"], r["point_b_id"], r["dw"]) for r in rows} == {
        ("a", "b", 1),
        ("a", "c", 6),
        ("b", "c", 5),
        ("b", "f", 1),
        ("c", "f", 6),
    }
    assert colls["c"].shape == (7, 3)
    assert len({r["pair_id"] for r in rows}) == 5
    path.write_text(json.dumps({"collections": {"c": {"points": dict(reversed(list(points.items())))}}}))
    reordered, _, _ = load_pair_records(path, 10000, 11000, max_dw=6)
    assert rows == reordered
    assert all("reason" in e for e in exclusions)


def test_loader_reports_missing_nonfinite_labels_without_inventing_zero(tmp_path):
    from paris_pair_check import load_pair_records

    path = tmp_path / "relwind.json"
    points = {
        "good": {"p": [0, 0, 10000], "wind_a": 1},
        "missing": {"p": [1, 0, 10000]},
        "nan": {"p": [2, 0, 10000], "wind_a": float("nan")},
        "inf": {"p": [3, 0, 10000], "wind_a": float("inf")},
    }
    path.write_text(json.dumps({"collections": {"c": {"points": points}}}))
    rows, colls, exclusions = load_pair_records(path)
    assert rows == []
    assert len(colls["c"]) == 4  # geometry does not depend on wind-label validity
    assert {(e["point_id"], e["reason"]) for e in exclusions} == {
        ("missing", "missing_wind_label"),
        ("nan", "nonfinite_wind_label"),
        ("inf", "nonfinite_wind_label"),
    }
    json.dumps(exclusions, allow_nan=False)


@pytest.mark.parametrize(
    "distance,predicted,error,margin",
    [(1.5, 2, -0.5, 0), (2.5, 2, 0.5, 0), (2.49, 2, 0.49, 0.01), (2.51, 3, 0.51, 0.01), (1, 1, -1, 0.5)],
)
def test_known_continuous_residuals_and_ties_to_even(distance, predicted, error, margin):
    from paris_pair_check import score_pairs

    field = np.broadcast_to(np.arange(8, dtype=float), (3, 3, 8))
    rows, summary = score_pairs(field, metadata(), [record(b=(distance, 1, 1))])
    row = rows[0]
    assert row["phi_a"] == 0
    assert row["phi_b"] == pytest.approx(distance)
    assert row["prediction_continuous"] == pytest.approx(distance)
    assert row["prediction_rounded"] == predicted
    assert row["error_integer"] == predicted - 2
    assert row["error_continuous"] == pytest.approx(error)
    assert row["rounding_margin"] == pytest.approx(margin)
    assert row["exact"] is (predicted == 2)
    assert summary["eligible"] == summary["scored"] == 1


def test_sampler_preserves_xyz_origin_and_spacing_even_for_integer_field():
    from paris_pair_check import score_pairs

    field = np.broadcast_to(np.arange(8), (3, 3, 8))
    meta = metadata()
    meta.update(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4])
    rows, _ = score_pairs(field, meta, [record(a=(10, 23, 34), b=(13, 23, 34))])
    assert rows[0]["prediction_continuous"] == 1.5


def test_unavailable_pairs_stay_in_all_eligible_and_collection_summaries():
    from paris_pair_check import score_pairs, summarize_rows

    field = np.broadcast_to(np.arange(8, dtype=float), (3, 3, 8)).copy()
    pairs = [record("ok"), record("outside", b=(8, 1, 1), collection="uncovered")]
    rows, summary = score_pairs(field, metadata(), pairs)
    assert summary == summarize_rows(rows)
    assert (summary["eligible"], summary["scored"], summary["correct"], summary["unavailable"]) == (2, 1, 1, 1)
    assert summary["micro_accuracy_all_eligible"] == 0.5
    assert summary["micro_accuracy_scored"] == 1
    assert rows[1]["coverage"]["b"] == "outside_domain"
    assert rows[1]["phi_b"] is rows[1]["exact"] is rows[1]["error_continuous"] is None
    missing = summary["per_collection"]["uncovered"]
    assert missing["unavailable"] == 1
    assert missing["micro_accuracy_all_eligible"] == 0
    assert missing["micro_accuracy_scored"] is None
    assert missing["continuous_errors"]["mae"] is None
    json.dumps({"rows": rows, "summary": summary}, allow_nan=False)


def test_nonfinite_samples_margin_missing_annotations_and_empty_input():
    from paris_pair_check import score_pairs, summarize_rows

    field = np.zeros((3, 3, 8))
    field[:, :, 5:] = np.nan
    missing = record("missing")
    missing.pop("wind_a")
    rows, summary = score_pairs(field, metadata(), [record("nan", b=(6, 1, 1)), missing])
    assert rows[0]["coverage"]["b"] == "nonfinite_sample"
    assert rows[1]["status"] == "invalid_annotation"
    assert summary["scored"] == 0
    assert summary["micro_accuracy_scored"] is None
    assert summary["continuous_errors"]["mae"] is None
    rows, _ = score_pairs(field, metadata(), [record()], margin=1)
    assert rows[0]["coverage"]["a"] == "inside_excluded_margin"
    empty = summarize_rows([])
    assert empty["eligible"] == 0
    assert empty["micro_accuracy_all_eligible"] is empty["micro_accuracy_scored"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"step": 0},
        {"step": float("nan")},
        {"step": -1},
        {"shape_zyx": [8, 3, 3]},
        {"frame": ""},
        {"origin_xyz": [0, 0, float("inf")]},
        {"spacing_xyz": [1, 0, 1]},
    ],
)
def test_invalid_metadata_is_rejected_even_without_pairs(change):
    from paris_pair_check import score_pairs

    meta = metadata()
    meta.update(change)
    with pytest.raises(ValueError):
        score_pairs(np.zeros((3, 3, 8)), meta, [])


def test_duplicate_pair_receipts_are_rejected():
    from paris_pair_check import score_pairs, summarize_rows

    field = np.zeros((3, 3, 8))
    with pytest.raises(ValueError, match="duplicate"):
        score_pairs(field, metadata(), [record(), record()])
    rows, _ = score_pairs(field, metadata(), [record()])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rows(rows + rows)


def test_pair_frame_is_explicit_and_negative_world_coordinates_are_valid():
    from paris_pair_check import score_pairs

    field = np.broadcast_to(np.arange(8, dtype=float), (3, 3, 8))
    meta = dict(metadata(), frame="different-volume/world", origin_xyz=[-10, -20, -30])
    pair = dict(record(a=(-10, -19, -29), b=(-8, -19, -29)), frame=meta["frame"])
    rows, summary = score_pairs(field, meta, [pair])
    assert summary["correct"] == 1 and rows[0]["frame"] == meta["frame"]
    pair["frame"] = "not-the-field-frame"
    with pytest.raises(ValueError, match="frame"):
        score_pairs(field, meta, [pair])


def test_annotation_loader_has_no_implicit_scan_window_or_maximum_winding(tmp_path):
    from paris_pair_check import load_pair_records

    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            dict(
                frame="different-volume/world",
                collections={"c": dict(points={"a": dict(p=[1, 1, 12], wind_a=0), "b": dict(p=[2, 1, 13], wind_a=12)})},
            )
        )
    )
    rows, _, _ = load_pair_records(path)
    assert len(rows) == 1 and rows[0]["dw"] == 12 and rows[0]["frame"] == "different-volume/world"


@pytest.mark.parametrize("values", [[0] * 8, [0, 0, 1, 1, 2, 2, 3, 3]])
def test_constant_and_merged_fields_fail_different_turn_labels(values):
    from paris_pair_check import score_pairs

    rows, summary = score_pairs(
        np.broadcast_to(np.array(values, float), (3, 3, 8)),
        metadata(),
        [record("adjacent", b=(1, 1, 1), dw=1), record("far", b=(4, 1, 1), dw=4)],
    )
    assert summary["correct"] == 0
    assert all(row["status"] == "scored_undercount" for row in rows)


def test_disconnected_piece_centering_cannot_hide_wrong_relative_offset():
    from paris_pair_check import score_pairs
    from patch_check import evaluate_patch

    field = np.broadcast_to(np.array([0, 0, 0, 0, 3, 3, 3, 3], float), (3, 3, 8))
    xyz = np.array([[[0, 1, 1], [1, 1, 1], [-1, -1, -1], [4, 1, 1], [5, 1, 1]]], float)
    patch, _ = evaluate_patch(field, xyz, [0] * 3, [1] * 3, 1, margin=0, max_edge=2)
    assert patch["local"]["abs_p95_sheets"] == 0
    rows, _ = score_pairs(field, metadata(), [record(b=(4, 1, 1), dw=2)])
    assert rows[0]["error_integer"] == 1
    assert rows[0]["exact"] is False


def test_cli_hashes_exclusive_outputs_and_json_csv_receipts(tmp_path):
    from paris_pair_check import main
    from patch_check import sha256

    field, meta, pairs = [tmp_path / name for name in ("field.npy", "meta.json", "pairs.json")]
    np.save(field, np.broadcast_to(np.arange(8, dtype=float), (3, 3, 8)))
    meta.write_text(json.dumps(dict(metadata(), field_sha256=sha256(field))))
    pairs.write_text(json.dumps([record(), record("out", b=(9, 1, 1))]))
    output, csv = tmp_path / "receipt.json", tmp_path / "pairs.csv"
    args = [
        "--field",
        str(field),
        "--metadata",
        str(meta),
        "--pairs-json",
        str(pairs),
        "--output",
        str(output),
        "--csv",
        str(csv),
    ]
    with pytest.raises(ValueError, match="metadata.*SHA"):
        main(args + ["--metadata-sha256", "0" * 64])
    assert not output.exists()
    main(args + ["--metadata-sha256", sha256(meta)])
    receipt = json.loads(output.read_text())
    assert receipt["summary"]["eligible"] == 2
    assert receipt["metadata_sha256"] == sha256(meta)
    assert receipt["field_sha256"] == sha256(field)
    assert "error_continuous" in csv.read_text().splitlines()[0]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == original


def test_cli_relative_windings_route_and_field_hash_guard(tmp_path):
    from paris_pair_check import main
    from patch_check import sha256

    field, meta, labels = [tmp_path / name for name in ("field.npy", "meta.json", "relwind.json")]
    np.save(field, np.broadcast_to(np.arange(8, dtype=float), (3, 3, 8)))
    m = metadata()
    m.update(origin_xyz=[0, 0, 10000], field_sha256="0" * 64)
    meta.write_text(json.dumps(m))
    labels.write_text(
        json.dumps(
            {
                "collections": {
                    "c": {
                        "points": {
                            "a": {"p": [0, 1, 10001], "wind_a": 1},
                            "b": {"p": [2, 1, 10001], "wind_a": 3},
                            "missing": {"p": [4, 1, 10001]},
                        }
                    }
                }
            }
        )
    )
    output = tmp_path / "receipt.json"
    args = ["--field", str(field), "--metadata", str(meta), "--relwind", str(labels), "--output", str(output)]
    with pytest.raises(ValueError, match="field SHA"):
        main(args)
    assert not output.exists()
    m["field_sha256"] = sha256(field)
    meta.write_text(json.dumps(m))
    main(args)
    result = json.loads(output.read_text())
    assert result["summary"]["correct"] == result["summary"]["eligible"] == 1
    assert result["exclusions"][0]["reason"] == "missing_wind_label"


def test_cli_manifest_frame_and_existing_csv_fail_before_json_write(tmp_path):
    from paris_pair_check import main

    field, meta, pairs = [tmp_path / name for name in ("field.npy", "meta.json", "pairs.json")]
    np.save(field, np.zeros((3, 3, 8)))
    meta.write_text(json.dumps(metadata()))
    pairs.write_text(json.dumps({"frame": "wrong", "records": [record()]}))
    output, csv = tmp_path / "receipt.json", tmp_path / "existing.csv"
    args = ["--field", str(field), "--metadata", str(meta), "--pairs-json", str(pairs), "--output", str(output)]
    with pytest.raises(ValueError, match="frame"):
        main(args)
    csv.write_text("keep me")
    with pytest.raises(FileExistsError):
        main(args + ["--csv", str(csv)])
    assert csv.read_text() == "keep me"
    assert not output.exists()
