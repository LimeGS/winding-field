"""CPU-only baseline receipt diagnosis and geometry-only holdout freezing.

python audit_errors.py audit --audit-dir baseline-audit --output diagnostics.json --report REPORT.md
python audit_errors.py freeze --config dataset.json --relwind labels.json --audit-dir baseline-audit --output-dir sealed
The freeze command never scores or examines winding values; label subsets stay private.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dataset_config import load_dataset, xyz
from evaluation_split import build_split
from paris_pair_check import summarize_rows
from patch_check import sha256

COUNT_KEYS = ("eligible", "covered", "scored", "correct", "incorrect", "unavailable")


def _same(a, b):
    return a is b if a is None or b is None or isinstance(b, bool) else math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def _stats(values):
    return (
        {
            "count": len(values),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
        if len(values)
        else {"count": 0, "min": None, "median": None, "p10": None, "p90": None, "max": None}
    )


def _group_summary(rows):
    result = summarize_rows(rows)
    result.pop("per_collection")
    result.pop("collection_macro_accuracy_all_eligible")
    scored = [r for r in rows if r["exact"] is not None]
    result.update(
        undercount=sum(r["error_integer"] < 0 for r in scored),
        overcount=sum(r["error_integer"] > 0 for r in scored),
        confusion=dict(sorted(Counter(str(r["prediction_rounded"]) for r in scored).items())),
        integer_errors=dict(sorted(Counter(format(r["error_integer"], "g") for r in scored).items())),
    )
    return result


def diagnose_rows(rows, slab_metadata):
    """Verify stored arithmetic and return per-slab/dw errors plus scale evidence.

    Positive winding distances only. The scale interval is a posthoc constraint
    intersection on development labels, not a fitted or recommended calibration.
    """
    for row in rows:
        sid = str(row["slab_id"])
        if sid not in slab_metadata:
            raise ValueError(f"missing metadata for slab {sid}")
        meta = slab_metadata[sid]
        if row.get("field_sha256") != meta.get("field_sha256"):
            raise ValueError(f"field identity mismatch for slab {sid}")
        if not np.isfinite(row["step"]) or row["step"] <= 0 or not _same(row["step"], meta["step"]):
            raise ValueError(f"invalid/mismatched step for slab {sid}")
        if row["exact"] is None:
            if row["error_continuous"] is not None or row["error_integer"] is not None:
                raise ValueError("unavailable pair has fabricated residuals")
            continue
        d = row["dw"]
        if not np.isfinite(d) or d < 1 or not _same(d, abs(row["wind_b"] - row["wind_a"])):
            raise ValueError("invalid positive winding-distance annotation")
        q = abs(row["phi_b"] - row["phi_a"]) / row["step"]
        if not np.isfinite(q) or not row["covered"]:
            raise ValueError("scored pair lacks finite covered prediction")
        pred = round(q)
        expected = {
            "prediction_continuous": q,
            "prediction_rounded": pred,
            "error_integer": pred - d,
            "error_continuous": q - d,
            "exact": pred == d,
            "rounding_margin": abs(q - (math.floor(q) + 0.5)),
        }
        if any(not _same(row[key], value) for key, value in expected.items()):
            raise ValueError(f"pair arithmetic mismatch: {row['pair_id']}")
        status = "scored_exact" if pred == d else "scored_undercount" if pred < d else "scored_overcount"
        if row["status"] != status:
            raise ValueError("pair status disagrees with raw prediction")
    summary = summarize_rows(rows)
    slabs, differences = defaultdict(list), defaultdict(list)
    for row in rows:
        slabs[str(row["slab_id"])].append(row)
        differences[format(row["dw"], "g") if row["dw"] is not None else "unavailable"].append(row)
    calibration, per_slab_dw = {}, {}
    for sid, meta in sorted(slab_metadata.items(), key=lambda item: int(item[0])):
        scored = [r for r in slabs[sid] if r["exact"] is not None]
        raw = [abs(r["phi_b"] - r["phi_a"]) for r in scored]
        required = [v / r["dw"] for v, r in zip(raw, scored)]
        ratios = [r["prediction_continuous"] / r["dw"] for r in scored]
        lower = max((v / (r["dw"] + 0.5) for v, r in zip(raw, scored)), default=None)
        upper = min((v / (r["dw"] - 0.5) for v, r in zip(raw, scored)), default=None)
        feasible = None
        if raw:
            feasible = False if 0 in raw or lower > upper else True if lower < upper else None
        count, field_range = meta.get("ridge_count"), meta.get("field_range_98pct")
        expected_step = (
            (field_range / max(count, 1) if count >= 2 else 1)
            if count is not None and field_range is not None
            else None
        )
        calibration[sid] = {
            "step": meta["step"],
            "ridge_count": count,
            "ridge_axis": meta.get("ridge_axis"),
            "ridge_line_counts": meta.get("ridge_line_counts"),
            "ridge_line_statistics": _stats(meta.get("ridge_line_counts") or []),
            "field_range_98pct": field_range,
            "ridge_rule_step": expected_step,
            "recorded_step_matches_ridge_rule": _same(meta["step"], expected_step)
            if expected_step is not None
            else None,
            "iterations": meta.get("iterations"),
            "relative_residual": meta.get("relative_residual"),
            "implied_step": _stats(required),
            "implied_step_over_used_step": _stats(ratios),
            "single_step_interval": {
                "lower": lower,
                "upper": upper,
                "feasible": feasible,
                "equal_boundary": "unknown: ties-to-even endpoints require separate handling",
            },
            "cause_status": "unattributed",
            "interpretation": "Empty interval rules out one positive scalar step fixing every scored pair; it does not identify the physical cause.",
        }
        by_dw = defaultdict(list)
        for row in slabs[sid]:
            by_dw[format(row["dw"], "g") if row["dw"] is not None else "unavailable"].append(row)
        per_slab_dw[sid] = {d: _group_summary(items) for d, items in sorted(by_dw.items())}
    return {
        "summary": summary,
        "per_slab": {sid: _group_summary(slabs[sid]) for sid in calibration},
        "per_dw": {d: _group_summary(items) for d, items in sorted(differences.items())},
        "per_slab_dw": per_slab_dw,
        "calibration": calibration,
        "hypotheses": [
            "Calibration may contribute to systematic distance bias; implied steps use development labels only.",
            "Orientation, incorrect fibre constraints, local merges and annotation ambiguity remain unassigned causes.",
        ],
        "limitations": [
            "Correlated pairs are not independent statistical units; no significance claim or pair bootstrap.",
            "Receipt arithmetic is verified; raw field arrays are not re-interpolated or rehashed.",
        ],
    }


def _check_summary(receipt):
    computed = summarize_rows(receipt["rows"])
    if any(receipt["summary"].get(key) != computed[key] for key in COUNT_KEYS):
        raise ValueError("receipt summary counts do not reconcile with rows")


def _metadata(directory):
    return {p.name[4:-9]: json.loads(p.read_text()) for p in sorted(Path(directory).glob("slab[0-9]*.npy.json"))}


def analyze_audit(directory, expected_correct=None, expected_eligible=None):
    """Reconcile complete slab receipts against the aggregate; hash small inputs."""
    directory = Path(directory)
    if not (directory / "DONE").is_file():
        raise ValueError("audit is not marked complete")
    aggregate = directory / "pairs.json"
    if not aggregate.exists():
        aggregate = directory / "finalpairs.json"
    receipt = json.loads(aggregate.read_text())
    _check_summary(receipt)
    metas = _metadata(directory)
    combined, sources = [], {aggregate.name: sha256(aggregate)}
    for sid in metas:
        path = directory / f"slab{sid}_pairs.json"
        part = json.loads(path.read_text())
        _check_summary(part)
        if any(str(row["slab_id"]) != sid for row in part["rows"]):
            raise ValueError("slab receipt membership mismatch")
        combined.extend(part["rows"])
        for source in (path, directory / f"slab{sid}.npy.json", directory / f"slab{sid}_constraints.npz"):
            if source.exists():
                sources[source.name] = sha256(source)
    if len({r["pair_id"] for r in combined}) != len(combined) or {r["pair_id"]: r for r in combined} != {
        r["pair_id"]: r for r in receipt["rows"]
    }:
        raise ValueError("slab rows do not reconcile with aggregate rows")
    result = diagnose_rows(receipt["rows"], metas)
    summary = result["summary"]
    if (expected_correct is not None and summary["correct"] != expected_correct) or (
        expected_eligible is not None and summary["eligible"] != expected_eligible
    ):
        raise ValueError("observed counts differ from expected baseline counts")
    for name in ("selection.json", "DONE", "progress.json"):
        if (directory / name).is_file():
            sources[name] = sha256(directory / name)
    result.update(
        audit_directory=str(directory),
        sources=sources,
        field_bytes_rehashed=False,
        analyzer_sha256=sha256(__file__),
        summarizer_sha256=sha256(Path(__file__).resolve().parents[1] / "src" / "paris_pair_check.py"),
        declared_field_sha256={sid: meta["field_sha256"] for sid, meta in metas.items()},
        reconciliation={
            "matched": True,
            "expected_correct": expected_correct,
            "expected_eligible": expected_eligible,
            **{key: summary[key] for key in COUNT_KEYS},
            "slabs": len(metas),
            "slab_rows_identical": True,
        },
    )
    return result


def render_report(result):
    summary = result["summary"]
    fmt = lambda value: "n/a" if value is None else f"{value:.4f}"
    lines = [
        "# Baseline error diagnosis",
        "",
        (
            f"Reconciled **{summary['correct']}/{summary['eligible']} exact**, **{summary['incorrect']} nonmatches**, "
            f"{summary['unavailable']} unavailable across {len(result['per_slab'])} slabs."
        ),
        "",
        (
            f"Continuous errors (sheets): bias {fmt(summary['continuous_errors']['mean_signed'])}, "
            f"MAE {fmt(summary['continuous_errors']['mae'])}, p95 absolute {fmt(summary['continuous_errors']['abs_p95'])}."
        ),
        "",
        "## By true winding difference",
        "",
        "| dw | Exact/scored | Under | Over | Bias | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d, s in result["per_dw"].items():
        lines.append(
            f"| {d} | {s['correct']}/{s['scored']} | {s['undercount']} | {s['overcount']} | "
            f"{fmt(s['continuous_errors']['mean_signed'])} | {fmt(s['continuous_errors']['mae'])} |"
        )
    lines += [
        "",
        "## Per slab",
        "",
        "| Slab | Exact/scored | Under | Over | MAE | Step | Implied/used step median | One step fixes all? |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sid, s in result["per_slab"].items():
        cal = result["calibration"][sid]
        lines.append(
            f"| {sid} | {s['correct']}/{s['scored']} | {s['undercount']} | {s['overcount']} | "
            f"{fmt(s['continuous_errors']['mae'])} | {fmt(cal['step'])} | "
            f"{fmt(cal['implied_step_over_used_step']['median'])} | {cal['single_step_interval']['feasible']} |"
        )
    lines += [
        "",
        "## Calibration evidence and hypotheses",
        "",
        "Implied step = raw field difference / known development winding difference. It is diagnostic, not a fitted candidate.",
        (
            "A disjoint single-step interval proves a uniform rescaling cannot make all scored pairs in that slab exact. "
            "An equal interval boundary remains unknown here. This does not assign orientation, fibre, merge or annotation causes."
        ),
        *["- " + text for text in result["hypotheses"]],
        "",
        "## Provenance and limits",
        "",
        f"- Analyzer SHA-256: `{result['analyzer_sha256']}`.",
        "- Full per-slab/dw confusion, residuals, calibration evidence and input SHA-256 values are in diagnostics.json.",
        *["- " + text for text in result["limitations"]],
    ]
    return "\n".join(lines) + "\n"


def plan_holdout(
    collections,
    slab_metadata,
    max_groups=3,
    max_group_voxels=40_000_000,
    max_total_voxels=120_000_000,
    *,
    frame,
    spacing_xyz,
    buffer_world,
    margin_world,
    alignment_xyz,
    grid_origin_xyz=(0, 0, 0),
    lower_bound_xyz=None,
    upper_bound_xyz=None,
    development_z_context=None,
    development_boxes=(),
):
    """Use only point geometry; whole-group stable-ID prefix, never score selection."""
    if max_groups < 1 or max_group_voxels < 1 or max_total_voxels < 1:
        raise ValueError("positive holdout budgets required")
    spacing = xyz(spacing_xyz, "holdout spacing", positive=True)
    alignment = xyz(alignment_xyz, "holdout alignment", positive=True)
    origin = xyz(grid_origin_xyz, "holdout grid origin")
    if not np.isfinite(margin_world) or margin_world < 0:
        raise ValueError("holdout margin must be finite and nonnegative")
    lower_bound = xyz(lower_bound_xyz, "lower bound") if lower_bound_xyz is not None else None
    upper_bound = xyz(upper_bound_xyz, "upper bound") if upper_bound_xyz is not None else None
    if lower_bound is not None and upper_bound is not None and (upper_bound <= lower_bound).any():
        raise ValueError("invalid holdout domain bounds")
    if any(meta.get("frame") != frame for meta in slab_metadata.values()):
        raise ValueError("baseline and holdout frames differ")
    colls, exclusions = {}, []
    for cid, collection in sorted(collections.items()):
        points = collection["points"]
        if not points:
            exclusions.append({"collection_id": cid, "reason": "empty_collection"})
        else:
            colls[cid] = np.array([p["p"] for p in points.values()], dtype=float)
    dev = sorted({cid for meta in slab_metadata.values() for cid in meta["collection_ids"]})
    boxes = [
        {
            "id": "baseline-slab" + sid,
            "min_xyz": meta["origin_xyz"],
            "max_xyz": (
                np.array(meta["origin_xyz"]) + np.array(meta["shape_zyx"])[::-1] * meta["spacing_xyz"]
            ).tolist(),
        }
        for sid, meta in sorted(slab_metadata.items())
    ]
    split = build_split(
        colls,
        dev,
        buffer_voxels=buffer_world,
        frame=frame,
        development_boxes=[*boxes, *development_boxes],
        development_z_context=development_z_context,
    )
    candidates = [g for g in split["groups"] if g["split"] == "test"]
    selected, total, blocker = [], 0, None if candidates else "no_independent_test_groups"
    for group in candidates[:max_groups]:
        minimum, maximum = np.array(group["min_xyz"]), np.array(group["max_xyz"])
        if (lower_bound is not None and (minimum < lower_bound).any()) or (
            upper_bound is not None and (maximum >= upper_bound).any()
        ):
            blocker = "whole_group_outside_declared_domain"
            break
        lo = origin + np.floor((minimum - margin_world - origin) / alignment) * alignment
        hi = origin + np.ceil((maximum + margin_world - origin) / alignment) * alignment
        if lower_bound is not None:
            lo = np.maximum(lo, lower_bound)
        if upper_bound is not None:
            hi = np.minimum(hi, upper_bound)
        shape = np.ceil((hi - lo) / spacing).astype(int)[::-1]
        if (shape <= 0).any():
            blocker = "empty_solver_domain"
            break
        voxels = math.prod(int(v) for v in shape)
        if voxels > max_group_voxels or total + voxels > max_total_voxels:
            blocker = "whole_group_budget_exceeded"
            break
        # A group's rectangular solver domain must not cover another group's labelled geometry.
        if any(
            c["group_id"] != group["group_id"] and np.all(lo <= c["max_xyz"]) and np.all(np.asarray(c["min_xyz"]) <= hi)
            for c in split["collections"]
        ):
            blocker = "solver_box_overlaps_other_group"
            break
        selected.append(
            {
                "group_id": group["group_id"],
                "collection_ids": group["collection_ids"],
                "points": group["points"],
                "grid": {
                    "origin_xyz": lo.tolist(),
                    "hi_xyz": hi.tolist(),
                    "spacing_xyz": spacing.tolist(),
                    "shape_zyx": shape.tolist(),
                    "working_voxels": voxels,
                },
            }
        )
        total += voxels
    return {
        "frame": frame,
        "source_collection_count": len(collections),
        "geometry_exclusions": exclusions,
        "split": split,
        "initial_test": {
            "groups": selected,
            "collection_ids": sorted(c for g in selected for c in g["collection_ids"]),
            "working_voxels": total,
            "candidate_test_groups": len(candidates),
            "blocker": blocker,
            "max_groups": max_groups,
            "max_group_voxels": max_group_voxels,
            "max_total_voxels": max_total_voxels,
            "margin_world": margin_world,
            "alignment_xyz": alignment.tolist(),
            "grid_origin_xyz": origin.tolist(),
            "selection": "stable group-ID prefix, stop before a whole group exceeds budget or overlaps another group; no label values",
        },
    }


def freeze_holdout(relwind_path, audit_directory, output_directory, *, geometry):
    """Write a new manifest and mode-0600 opaque label subsets; return public data only."""
    data = json.loads(Path(relwind_path).read_text())
    if data.get("frame", geometry["frame"]) != geometry["frame"]:
        raise ValueError("annotation/holdout frame mismatch")
    plan = plan_holdout(data["collections"], _metadata(audit_directory), **geometry)
    output = Path(output_directory)
    output.mkdir(exist_ok=False)
    labels = {}
    subsets = {
        f"{split}-labels.json": [c["collection_id"] for c in plan["split"]["collections"] if c["split"] == split]
        for split in ("validation", "test")
    }
    subsets["initial-test-labels.json"] = plan["initial_test"]["collection_ids"]
    for name, ids in subsets.items():
        path = output / name
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
            # Labels are copied opaquely, never used for selection or printed.
            json.dump(
                {"frame": geometry["frame"], "collections": {cid: data["collections"][cid] for cid in ids}},
                handle,
                sort_keys=True,
            )
        labels[name] = {"path": str(path), "sha256": sha256(path), "collections": len(ids)}
    plan.update(
        relwind_sha256=sha256(relwind_path),
        label_files=labels,
        split_helper_sha256=sha256(Path(__file__).resolve().parents[1] / "src" / "evaluation_split.py"),
        freezer_sha256=sha256(__file__),
        label_values_inspected=False,
        seal_limitation="Logical access discipline and file permissions only; no independent custodian or verified surface lineage.",
    )
    plan["freeze_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with (output / "geometry-manifest.json").open("x") as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--audit-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--expected-correct", type=int)
    audit.add_argument("--expected-eligible", type=int)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--relwind", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--audit-dir", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        dataset = load_dataset(args.config)
        policy = dataset.get("context", {}).get("holdout")
        if policy is None:
            raise ValueError("declare context.holdout buffer_world, margin_world and alignment_xyz")
        grid = dataset["grid"]
        geometry = dict(
            policy,
            frame=dataset["frame"],
            spacing_xyz=grid["spacing_xyz"],
            grid_origin_xyz=grid["origin_xyz"],
            lower_bound_xyz=grid["origin_xyz"],
            upper_bound_xyz=(
                np.array(grid["origin_xyz"]) + np.array(grid["spacing_xyz"]) * np.array(grid["shape_zyx"][::-1])
            ).tolist(),
        )
        public = freeze_holdout(args.relwind, args.audit_dir, args.output_dir, geometry=geometry)
        print(
            json.dumps(
                {
                    "counts": public["split"]["counts"],
                    "empty_collections": len(public["geometry_exclusions"]),
                    "initial_test": public["initial_test"],
                    "label_files": public["label_files"],
                    "freeze_sha256": public["freeze_sha256"],
                }
            )
        )
        return
    if args.output.resolve() == args.report.resolve():
        raise ValueError("output and report paths must differ")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(path)
    result = analyze_audit(args.audit_dir, args.expected_correct, args.expected_eligible)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    report = render_report(result)
    with args.output.open("x") as handle:
        handle.write(text)
    with args.report.open("x") as handle:
        handle.write(report)
    print(json.dumps(result["reconciliation"]))


if __name__ == "__main__":
    main()
