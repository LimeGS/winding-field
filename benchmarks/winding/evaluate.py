#!/usr/bin/env python3
"""Evaluate saved fields on a hash-bound case; never import or invoke a producer."""

import argparse
import gzip
import hashlib
import json
import platform
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import scipy

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import interpolation as verify
import metrics as bench

import evaluation_split
import paris_pair_check
import patch_check
from dataset_config import frame as validate_frame
from dataset_config import load_dataset, native_to_world, positive, xyz


def load_inputs(args):
    case = json.loads(args.case.read_text())
    if case.get("schema") not in ("winding-benchmark-case-v1", "winding-benchmark-case-v2"):
        raise ValueError("unsupported benchmark case schema")
    validate_frame(case.get("frame"))
    population = case.get("population", "all_and_reserved" if case["schema"].endswith("v1") else None)
    if population not in ("all", "all_and_reserved"):
        raise ValueError("case must declare all or all_and_reserved population")
    if type(case.get("synthetic", False)) is not bool:
        raise ValueError("synthetic case flag must be boolean")
    step = float(case["step"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError("case scale must be positive and finite")
    inputs = {}
    paths = {}
    names = ["selection", "plan", "seal", "audit", "afv"]
    names += [name for name in ("split", "human") if name in case["artifacts"]]
    for name in names:
        entry = case["artifacts"][name]
        path = (args.case.parent / entry["path"]).resolve()
        actual = bench.sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"case artifact hash mismatch: {name}: {path}")
        paths[name] = path
        inputs[name] = dict(path=str(path), sha256=actual, bytes=path.stat().st_size)
    plan = json.loads(paths["plan"].read_text())
    seal = json.loads(paths["seal"].read_text())
    if seal["selected_sha256"] != inputs["selection"]["sha256"] or seal["plan_sha256"] != inputs["plan"]["sha256"]:
        raise ValueError("selection is not bound to the frozen plan/seal")
    audit = json.loads(paths["audit"].read_text())
    if plan["constraint_audit_sha256"] != inputs["audit"]["sha256"] or audit["afv_sha256"] != inputs["afv"]["sha256"]:
        raise ValueError("selection plan/audit is not bound to this AFV")
    configured = None
    config_path = getattr(args, "config", None)
    if config_path is not None:
        configured = load_dataset(config_path)
        if configured["frame"] != case["frame"] or bench.sha256(configured["afv"]["path"]) != inputs["afv"]["sha256"]:
            raise ValueError("dataset configuration differs from case frame/AFV")
        inputs["dataset_config"] = dict(
            path=str(config_path.resolve()), sha256=bench.sha256(config_path), bytes=config_path.stat().st_size
        )
    transform = case.get("native_to_world", configured["afv"] if configured else None)
    unit = case.get("voxel_um", configured["voxel_um"] if configured else None)
    if transform is None or unit is None:
        raise ValueError("explicit native_to_world and voxel_um are required; supply a new case descriptor or --config")
    unit = positive(unit, "case voxel_um")
    native_to_world(np.empty((0, 3)), transform)
    transform = {key: list(transform[key]) for key in ("scale_xyz", "offset_xyz")}
    if configured is not None and (
        unit != configured["voxel_um"]
        or any(not np.allclose(transform[key], configured["afv"][key], rtol=1e-12, atol=0) for key in transform)
    ):
        raise ValueError("case/config coordinate transform differs")
    native_mm = np.broadcast_to(np.asarray(plan["native_voxel_mm"], float), (3,))
    if (
        plan["frame"] != case["frame"]
        or plan["fixed_sheet_step"] != step
        or not np.isfinite(native_mm).all()
        or not np.allclose(
            native_mm, xyz(transform["scale_xyz"], "scale", positive=True) * unit / 1000, rtol=1e-12, atol=0
        )
        or plan["bins"] != [list(b) for b in bench.BINS]
    ):
        raise ValueError("case differs from frozen frame, scale, units or bins")
    if "split" in paths:
        split = json.loads(paths["split"].read_text())
        evaluation_split.validate_fibre_split(split, inputs["afv"]["sha256"])
        known_ids = {r["fibre_id"] for r in split["fibres"]}
        reserved = (
            {r["fibre_id"] for r in split["fibres"] if r["split"] == "reserved"}
            if population == "all_and_reserved"
            else None
        )
        del split
    else:
        if population != "all":
            raise ValueError("reserved diagnostics require an identity-bound split")
        with closing(sqlite3.connect(paths["afv"].as_uri() + "?mode=ro", uri=True)) as db:
            known_ids = {r[0] for r in db.execute("SELECT id FROM fibers")}
        reserved = None
    with gzip.open(paths["selection"], "rt") as stream:
        rows = json.load(stream)[case["scope"]]
    if len(rows) != case["expected_selected"] or not rows:
        raise ValueError("selected population size differs from case")
    seen = set()
    for row in rows:
        fid, bin_index = row["fiber_id"], row["bin"]
        if (
            type(fid) is not int
            or fid not in known_ids
            or type(bin_index) is not int
            or not 0 <= bin_index < len(bench.BINS)
        ):
            raise ValueError("invalid selected fibre/bin or missing split assignment")
        key = fid, bin_index
        if key in seen:
            raise ValueError("duplicate fibre/bin in frozen selection")
        seen.add(key)
        coordinates = np.asarray([row["a_native_xyz"], row["b_native_xyz"]], float)
        lo, hi = bench.BINS[bin_index]
        if coordinates.shape != (2, 3) or not np.isfinite(coordinates).all() or not lo <= row["span_mm"] < hi:
            raise ValueError("invalid selected coordinates or arclength bin")
    human = json.loads(paths["human"].read_text())["methods"][case["human_method"]]["rows"] if "human" in paths else []
    if len(human) != case["expected_human"]:
        raise ValueError("human development control population changed")
    meta = json.loads(args.metadata.read_text())
    field_hash = bench.sha256(args.field)
    if field_hash != meta["field_sha256"]:
        raise ValueError("candidate field hash mismatch")
    if meta["frame"] != case["frame"] or not np.isclose(meta["step"], step, rtol=1e-12, atol=0):
        raise ValueError("candidate frame/scale differs from case; no refitting is permitted")
    if "voxel_um" in meta and not np.isclose(meta["voxel_um"], unit, rtol=1e-12, atol=0):
        raise ValueError("candidate world units differ from case")
    if "solver" in meta and meta["solver"].get("converged") is not True:
        raise ValueError("candidate declares a nonconverged solver")
    if meta.get("pilot_gate_status", "PASS") != "PASS":
        raise ValueError("candidate declares a failed pilot gate")
    field = np.load(args.field, mmap_mode="r", allow_pickle=False)
    origin, spacing = np.asarray(meta["origin_xyz"], float), np.asarray(meta["spacing_xyz"], float)
    if (
        field.ndim != 3
        or min(field.shape) < 3
        or field.dtype.kind not in "fiu"
        or list(field.shape) != meta["shape_zyx"]
        or origin.shape != (3,)
        or spacing.shape != (3,)
        or not np.isfinite([origin, spacing]).all()
        or (spacing <= 0).any()
    ):
        raise ValueError("invalid candidate XYZ/ZYX grid")
    declared_support = any(meta.get(key) is not None for key in ("support", "support_sha256", "support_array_sha256"))
    if (case.get("requires_support") or declared_support or args.support_sha256) and args.support is None:
        raise ValueError("case/candidate requires an explicit support mask")
    support = None if args.support is None else np.load(args.support, mmap_mode="r", allow_pickle=False)
    if support is not None and (support.shape != field.shape or support.dtype.kind != "b"):
        raise ValueError("support must be a matching boolean array")
    if support is not None:
        expected_file = [v for v in (args.support_sha256, meta.get("support_sha256")) if v is not None]
        expected_array = meta.get("support_array_sha256")
        if not expected_file and expected_array is None:
            raise ValueError("support needs a declared hash in metadata or --support-sha256")
        actual = bench.sha256(args.support)
        if any(expected != actual for expected in expected_file):
            raise ValueError("candidate support file hash mismatch")
        if (
            expected_array is not None
            and hashlib.sha256(np.ascontiguousarray(support).tobytes()).hexdigest() != expected_array
        ):
            raise ValueError("candidate support array hash mismatch")
    invalid = ~np.isfinite(field)
    if (invalid if support is None else invalid & support).any():
        raise ValueError("nonfinite field values in declared support")
    if field.dtype.kind != "f":
        field = field.astype(float)
    for name, path in [
        ("case", args.case),
        ("field", args.field),
        ("metadata", args.metadata),
        ("support", args.support),
    ]:
        if path is not None:
            inputs[name] = dict(path=str(path.resolve()), sha256=bench.sha256(path), bytes=path.stat().st_size)
    sources = [
        Path(__file__),
        Path(bench.__file__),
        Path(verify.__file__),
        Path(evaluation_split.__file__),
        Path(paris_pair_check.__file__),
        Path(patch_check.__file__),
        Path(bench.common.__file__),
        Path(native_to_world.__code__.co_filename),
    ]
    receipt = dict(
        status="PASS_PREFLIGHT",
        scope=case["scope"],
        frame=case["frame"],
        step=step,
        selected=len(rows),
        reserved_selected=sum(r["fiber_id"] in reserved for r in rows) if reserved is not None else None,
        population=population,
        voxel_um=unit,
        native_to_world=transform,
        human_eligible=len(human),
        inputs=inputs,
        source_sha256={str(p): bench.sha256(p) for p in sources},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        producer_invoked=False,
        candidate_solver_receipt=meta.get("solver"),
        synthetic=case.get("synthetic", False),
    )
    result = dict(
        case=case,
        rows=rows,
        reserved=reserved,
        human=human,
        meta=meta,
        field=field,
        support=support,
        receipt=receipt,
        transform=transform,
    )
    return result


def evaluate(inputs, output):
    rows, field, meta, support = (inputs[k] for k in ("rows", "field", "meta", "support"))
    values = np.full((len(rows), 2), np.nan)
    errors = np.full(len(rows), np.nan)
    max_difference = 0.0
    max_roundoff = 0.0
    for start in range(0, len(rows), 10000):
        chunk = rows[start : start + 10000]
        sampled = bench.evaluate(field, meta, chunk, support=support, transform=inputs["transform"])
        for endpoint, column in (("a", 0), ("b", 1)):
            actual = np.array([np.nan if r["phi_" + endpoint] is None else r["phi_" + endpoint] for r in sampled])
            coordinates = native_to_world([r[endpoint + "_native_xyz"] for r in chunk], inputs["transform"])
            independent = verify.interpolate(field, meta, coordinates, support)
            if not np.array_equal(np.isfinite(actual), np.isfinite(independent)):
                raise ValueError("independent interpolation coverage mismatch")
            known = np.isfinite(actual)
            differences = np.abs(actual[known] - independent[known])
            delta = float(np.max(differences, initial=0))
            max_difference = max(max_difference, delta)
            # SciPy rounds its interpolation output to the input array dtype.
            roundoff = 0.5 * np.spacing(np.abs(actual[known]).astype(field.dtype)).astype(float)
            max_roundoff = max(max_roundoff, float(np.max(roundoff, initial=0)))
            if (differences > 2e-5 + roundoff).any():
                raise ValueError("independent interpolation value mismatch")
            values[start : start + len(chunk), column] = actual
        errors[start : start + len(chunk)] = [np.nan if r["error"] is None else r["error"] for r in sampled]
        if start % 100000 == 0:
            print("EVALUATE", start, "/", len(rows), flush=True)
    np.savez_compressed(output / "predictions.npz", phi_a=values[:, 0], phi_b=values[:, 1], error=errors)
    del values
    methods = {}
    for name, factor in (("candidate", 1.0), ("constant_control", 0.0), ("compressed_control", 0.1)):
        compact = [
            dict(fiber_id=r["fiber_id"], bin=r["bin"], error=float(e * factor) if np.isfinite(e) else None)
            for r, e in zip(rows, errors)
        ]
        human_field = field * factor if factor != 1 else field
        if support is not None:
            human_field = np.where(support, human_field, np.nan)
        _, human = paris_pair_check.score_pairs(human_field, meta, inputs["human"])
        methods[name] = dict(
            all=bench.summarize(compact),
            reserved=bench.summarize([r for r in compact if r["fiber_id"] in inputs["reserved"]])
            if inputs["reserved"] is not None
            else None,
            human=human,
        )
        del compact, human_field
    return dict(
        status="PASS_EVALUATED",
        synthetic=inputs["case"].get("synthetic", False),
        scope=inputs["case"]["scope"],
        methods=methods,
        independent_interpolation=dict(
            points=2 * len(rows),
            max_abs_difference=max_difference,
            tolerance=2e-5,
            max_output_roundoff=max_roundoff,
            comparison_rule="absolute tolerance plus half an output-dtype ULP per value",
        ),
        physical_acceptance=False,
        limitations=[
            "Evaluation completion is not physical acceptance or proof of whole-scroll sheet identity.",
            "All-fibre results may include fitting fibres; reserved lineage is not spatial/pretraining independence.",
            "Analytic synthetic separation references; no real scan data or human annotations."
            if inputs["case"].get("synthetic", False)
            else "Human separation controls are reused development data, not a blind test.",
            "Margin is one candidate-grid node; compare coverage when resolutions or support differ.",
        ],
    )


def write_tables(output, result):
    lines = ["# Artifact benchmark", "", f"Scope: `{result['scope']}`", ""]
    if result.get("synthetic", False):
        lines += ["Synthetic demonstration only — no real scroll data or human annotations.", ""]
    for name, method in result["methods"].items():
        for population in ("all", "reserved"):
            if method[population] is None:
                lines += [f"## {name} / {population}", "", "Not evaluated: no reserved-population claim.", ""]
                continue
            lines += [
                f"## {name} / {population}",
                "",
                "| mm | selected | covered | median | p90 | p99 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for r in method[population]:
                values = ["—" if r[k] is None else f"{r[k]:.6f}" for k in ("median", "p90", "p99")]
                lines.append(f"| {r['span_mm']} | {r['selected']} | {r['covered']} | " + " | ".join(values) + " |")
            lines.append("")
        h = method["human"]
        label = "Synthetic separation control" if result.get("synthetic", False) else "Human control"
        lines.append(
            f"{label}: {h['correct']}/{h['eligible']} correct; {h['covered']} covered.\n"
            if h["scored"]
            else f"{label}: not evaluated; {h['covered']}/{h['eligible']} covered.\n"
        )
    (output / "TABLES.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "field", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument(
        "--config", type=Path, help="explicit geometry for a historical case lacking coordinate metadata"
    )
    parser.add_argument("--support-sha256", help="expected support-file hash, e.g. from a frozen producer receipt")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(exist_ok=False)
    except OSError as exc:
        print(f"New output directory required: {exc}", file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        inputs = load_inputs(args)
        bench.write_json(args.output / "PREFLIGHT.json", inputs["receipt"])
        if not args.check_only:
            result = evaluate(inputs, args.output)
            for entry in inputs["receipt"]["inputs"].values():
                if bench.sha256(entry["path"]) != entry["sha256"]:
                    raise ValueError("input changed during evaluation: " + entry["path"])
            result.update(
                preflight_sha256=bench.sha256(args.output / "PREFLIGHT.json"), elapsed_s=time.monotonic() - started
            )
            write_tables(args.output, result)
            bench.write_json(args.output / "SUMMARY.json", result)
        bench.write_json(
            args.output / "MANIFEST.json",
            {p.name: bench.sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        )
        print("PASS_PREFLIGHT" if args.check_only else "PASS_EVALUATED", str(args.output), flush=True)
        return 0
    except Exception as exc:
        bench.write_json(args.output / "ERROR.json", dict(status="ERROR", type=type(exc).__name__, message=str(exc)))
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
