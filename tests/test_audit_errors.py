"""Receipt reconciliation and label-blind, whole-group holdout planning."""

import copy
import json

import numpy as np
import pytest

from paris_pair_check import FRAME, score_pairs


def fixture_rows():
    meta = {
        "frame": FRAME,
        "shape_zyx": [3, 3, 8],
        "origin_xyz": [0, 0, 0],
        "spacing_xyz": [1, 1, 1],
        "step": 1,
        "field_sha256": "f" * 64,
        "collection_ids": ["c"],
        "ridge_count": 3,
        "field_range_98pct": 3,
        "ridge_line_counts": [2, 3, 3],
        "relative_residual": 0.0009,
        "iterations": 50,
    }
    pairs = [
        {
            "pair_id": str(i),
            "collection_id": "c",
            "point_a_id": "a",
            "point_b_id": str(i),
            "a_xyz": [0, 1, 1],
            "b_xyz": [distance, 1, 1],
            "wind_a": 0,
            "wind_b": 2,
            "dw": 2,
        }
        for i, distance in enumerate((1, 2, 3, 9))
    ]
    rows, summary = score_pairs(np.broadcast_to(np.arange(8, dtype=float), (3, 3, 8)), meta, pairs)
    for row in rows:
        row.update(slab_id=0, field_sha256=meta["field_sha256"])
    return rows, summary, meta


def write_audit(tmp_path):
    rows, summary, meta = fixture_rows()
    (tmp_path / "pairs.json").write_text(json.dumps({"rows": rows, "summary": summary}))
    (tmp_path / "slab0_pairs.json").write_text(json.dumps({"rows": rows, "summary": summary}))
    (tmp_path / "slab0.npy.json").write_text(json.dumps(meta))
    (tmp_path / "DONE").write_text("completed")
    return rows


def test_confusion_residuals_and_single_scale_infeasibility():
    from audit_errors import diagnose_rows

    rows, _, meta = fixture_rows()
    result = diagnose_rows(rows, {"0": meta, "1": meta})
    assert result["summary"]["correct"] == 1
    assert result["summary"]["incorrect"] == 2
    assert result["summary"]["unavailable"] == 1
    assert result["per_dw"]["2"]["undercount"] == 1
    assert result["per_dw"]["2"]["overcount"] == 1
    assert result["per_dw"]["2"]["continuous_errors"]["mae"] == pytest.approx(2 / 3)
    assert result["per_dw"]["2"]["confusion"] == {"1": 1, "2": 1, "3": 1}
    assert result["per_slab"]["1"]["scored"] == 0
    assert result["per_slab"]["1"]["micro_accuracy_scored"] is None
    cal = result["calibration"]["0"]
    assert cal["recorded_step_matches_ridge_rule"] is True
    assert cal["single_step_interval"]["feasible"] is False  # lower 3/2.5 > upper 1/1.5
    assert cal["implied_step_over_used_step"]["median"] == 1
    assert cal["cause_status"] == "unattributed"


@pytest.mark.parametrize("mutation", [{"prediction_rounded": 7}, {"error_continuous": 0}, {"step": 0}, {"exact": True}])
def test_tampered_pair_arithmetic_is_not_accepted(mutation):
    from audit_errors import diagnose_rows

    rows, _, meta = fixture_rows()
    rows[0].update(mutation)
    with pytest.raises(ValueError):
        diagnose_rows(rows, {"0": meta})


def test_audit_hashes_reconcile_aggregate_slab_rows_and_expected_counts(tmp_path):
    from audit_errors import analyze_audit

    write_audit(tmp_path)
    result = analyze_audit(tmp_path, expected_correct=1, expected_eligible=4)
    assert result["reconciliation"]["matched"] is True
    assert result["reconciliation"]["incorrect"] == 2
    assert len(result["sources"]["pairs.json"]) == 64
    assert result["field_bytes_rehashed"] is False
    with pytest.raises(ValueError, match="expected"):
        analyze_audit(tmp_path, expected_correct=2, expected_eligible=4)
    slab = json.loads((tmp_path / "slab0_pairs.json").read_text())
    slab["rows"] = slab["rows"][:-1]
    (tmp_path / "slab0_pairs.json").write_text(json.dumps(slab))
    with pytest.raises(ValueError, match="reconcil|summary"):
        analyze_audit(tmp_path, expected_correct=1, expected_eligible=4)


def test_aggregate_summary_and_field_identity_mismatch_fail_closed(tmp_path):
    from audit_errors import analyze_audit

    write_audit(tmp_path)
    path = tmp_path / "pairs.json"
    data = json.loads(path.read_text())
    data["summary"]["correct"] = 4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="summary"):
        analyze_audit(tmp_path, expected_correct=1, expected_eligible=4)
    write_audit(tmp_path)
    path = tmp_path / "slab0.npy.json"
    data = json.loads(path.read_text())
    data["field_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="field"):
        analyze_audit(tmp_path, expected_correct=1, expected_eligible=4)


def geometry_collections():
    return {
        f"c{i}": {
            "points": {
                "a": {"p": [i * 2000, 0, 20000], "wind_a": "SECRET"},
                "b": {"p": [i * 2000 + 10, 10, 20010], "wind_a": "SECRET"},
            }
        }
        for i in range(20)
    }


def test_holdout_plan_is_geometry_only_accounts_for_empties_and_keeps_whole_groups():
    from audit_errors import plan_holdout

    collections = geometry_collections()
    collections["empty"] = {"points": {}}
    plan = plan_holdout(collections, {})
    assert plan["source_collection_count"] == 21
    assert plan["geometry_exclusions"] == [{"collection_id": "empty", "reason": "empty_collection"}]
    assert len(plan["initial_test"]["groups"]) == 3
    test_groups = [g for g in plan["split"]["groups"] if g["split"] == "test"]
    assert [g["group_id"] for g in plan["initial_test"]["groups"]] == [g["group_id"] for g in test_groups[:3]]
    assert "SECRET" not in json.dumps(plan)
    changed = copy.deepcopy(collections)
    for coll in changed.values():
        for point in coll["points"].values():
            point["wind_a"] = -9876543
    assert plan == plan_holdout(changed, {})
    for group in plan["initial_test"]["groups"]:
        original = next(g for g in test_groups if g["group_id"] == group["group_id"])
        assert group["collection_ids"] == original["collection_ids"]
        assert group["grid"]["working_voxels"] <= 40_000_000


def test_budget_or_all_development_blocks_instead_of_trimming_groups():
    from audit_errors import plan_holdout

    plan = plan_holdout(geometry_collections(), {}, max_group_voxels=1)
    assert plan["initial_test"]["groups"] == []
    assert plan["initial_test"]["blocker"] == "whole_group_budget_exceeded"
    collections = geometry_collections()
    for coll in collections.values():
        for point in coll["points"].values():
            point["p"][2] = 10500
    plan = plan_holdout(collections, {})
    assert plan["initial_test"]["groups"] == []
    assert plan["initial_test"]["blocker"] == "no_independent_test_groups"
    assert plan["split"]["buffer_voxels"] == 256


def test_sealing_writes_private_labels_without_printing_or_exporting_values(tmp_path, capsys):
    from audit_errors import freeze_holdout

    source = tmp_path / "relwind.json"
    source.write_text(json.dumps({"collections": geometry_collections()}))
    audit = tmp_path / "audit"
    audit.mkdir()
    output = tmp_path / "sealed"
    public = freeze_holdout(source, audit, output)
    assert "SECRET" not in json.dumps(public)
    assert "SECRET" not in (output / "geometry-manifest.json").read_text()
    assert "SECRET" not in capsys.readouterr().out
    label = output / "initial-test-labels.json"
    assert label.stat().st_mode & 0o777 == 0o600
    assert len(public["label_files"]["initial-test-labels.json"]["sha256"]) == 64
    private = json.loads(label.read_text())
    assert sorted(private["collections"]) == public["initial_test"]["collection_ids"]
    with pytest.raises(FileExistsError):
        freeze_holdout(source, audit, output)


def test_cli_exclusive_diagnostics_and_markdown_report(tmp_path):
    from audit_errors import main

    audit = tmp_path / "audit"
    audit.mkdir()
    write_audit(audit)
    output, report = tmp_path / "diagnostics.json", tmp_path / "REPORT.md"
    args = [
        "audit",
        "--audit-dir",
        str(audit),
        "--output",
        str(output),
        "--report",
        str(report),
        "--expected-correct",
        "1",
        "--expected-eligible",
        "4",
    ]
    main(args)
    assert json.loads(output.read_text())["reconciliation"]["matched"] is True
    assert "hypotheses" in report.read_text().lower()
    with pytest.raises(FileExistsError):
        main(args)
