#!/usr/bin/env python3
"""Seal prediction artifacts before official scoring; not label-access isolation.

Historical baseline runs read the public source annotation table while selecting
development records. No test-winding dependence was established. The current
bundled baseline skips that read in prediction-only mode; this is not OS access
isolation. The private scoring subset opens after prediction hashes are recorded.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from dataset_config import load_dataset, mode_constraints
from patch_check import sha256


def development_step(rows):
    grouped = defaultdict(list)
    for r in rows:
        if r.get("covered") and r.get("dw", 0) > 0 and r.get("phi_a") is not None and r.get("phi_b") is not None:
            ratio = abs(r["phi_b"] - r["phi_a"]) / r["dw"]
            if np.isfinite(ratio):
                grouped[r["collection_id"]].append(ratio)
    value = float(np.median([np.median(v) for v in grouped.values()])) if grouped else 0
    if not np.isfinite(value) or value <= 0:
        raise ValueError("no positive development calibration")
    return value


def seal_extra_global(output, field):
    output, field = Path(output), Path(field)
    if (output / "test_results.json").exists():
        raise ValueError("cannot add a method after test outcomes exist")
    value = dict(
        role="secondary_before_unsealing",
        field=str(field),
        field_sha256=sha256(field),
        metadata_sha256=sha256(str(field) + ".json"),
        primary_seal_sha256=sha256(output / "PREDICTIONS_SEALED.json"),
        sealed_utc=time.time(),
    )
    with (output / "GLOBAL_SECONDARY_SEAL.json").open("x") as f:
        json.dump(value, f, indent=2)
    return value


def baseline_environment(args, baseline, lo, hi):
    dataset = load_dataset(args.config)
    controls = dataset.get("constraints", {})
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("AUDIT_DIR", "RGT", "SLAB", "OUT", "AFV", "WREV_CONFIG", "ORIENT")
    }
    env.update(
        ORIENT=dataset["orientation"]["mode"],
        LAM=str(controls.get("pair_weight", 3)),
        FIBER_LIMIT=str(controls.get("fiber_limit", 400)),
        PAIR_LIMIT=str(controls.get("pair_limit", 2000)),
        WREV_CONFIG=str(args.config.resolve()),
        PREDICT_ONLY="1",
        WREV_DEVICE=args.device or dataset.get("solver", {}).get("device", "cpu"),
        OUT=str(Path(baseline).resolve()),
        AFV=dataset["afv"]["path"],
        SLAB=",".join(map(str, np.stack([lo, hi], 1).ravel())),
    )
    return env


def predict(args):
    from coarse_reference import load_external_reference
    from field_solver import solve_field
    from paris_data import load_paris_box, save_field
    from physical_grid import grid_shape

    dataset = load_dataset(args.config)
    mode_constraints(dataset, {"pair_weight", "fiber_limit", "pair_limit"}, "sealed nodal prediction")
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    geometry = json.loads(args.manifest.read_text())
    geometry_frame = geometry.get("frame", geometry.get("split", {}).get("frame"))
    if (
        geometry_frame != dataset["frame"]
        or geometry.get("split", {}).get("frame", dataset["frame"]) != dataset["frame"]
    ):
        raise ValueError("geometry manifest and dataset frames differ")
    if geometry.get("voxel_um", dataset["voxel_um"]) != dataset["voxel_um"]:
        raise ValueError("geometry manifest and dataset units differ")
    step = dataset["sheet_step"]
    if not np.isclose(development_step(json.loads(args.development.read_text())["rows"]), step, rtol=1e-12, atol=0):
        raise ValueError("development evidence differs from configured calibration")
    config = dict(
        geometry_sha256=sha256(args.manifest),
        dataset_config_sha256=dataset["config_sha256"],
        development_sha256=sha256(args.development),
        global_step=step,
        lam=dataset.get("constraints", {}).get("pair_weight", 3),
        rtol=dataset.get("solver", {}).get("rtol", 0.001),
        maxiter=dataset.get("solver", {}).get("maxiter", 4000),
        fiber_limit=dataset.get("constraints", {}).get("fiber_limit", 400),
        pair_limit=dataset.get("constraints", {}).get("pair_limit", 2000),
        calibration="median of collection medians of abs(phi_b-phi_a)/dw on development only",
        candidate="physical_grid_signed_shared_operator",
        baseline="configured_legacy_index_solver_fixed_calibration",
        predictor_sha256=sha256(__file__),
        legacy_solver_sha256=sha256(args.legacy_solver),
        code_sha256={
            n: sha256(Path(__file__).with_name(n))
            for n in ("field_solver.py", "physical_grid.py", "paris_data.py", "dataset_config.py")
        },
        private_scoring_subset_opened=False,
        label_access_isolated=False,
        baseline_prediction_only_requested=True,
        label_access_limitation="Current bundled baseline skips winding-label loading in prediction-only mode; historical runs did not. No OS access isolation.",
        started_utc=time.time(),
    )
    (out / "frozen_config.json").write_text(json.dumps(config, indent=2) + "\n")
    artifacts = []
    for i, group in enumerate(geometry["initial_test"]["groups"]):
        grid = group["grid"]
        lo, hi = np.array(grid["origin_xyz"]), np.array(grid["hi_xyz"])
        spacing = np.asarray(grid["spacing_xyz"], float)
        shape = grid_shape(lo, hi, spacing, grid.get("shape_zyx"))
        baseline = out / f"group{i}_baseline.npy"
        env = baseline_environment(args, baseline, lo, hi)
        print("PREDICT baseline", i, group["group_id"], flush=True)
        command = [
            sys.executable,
            str(args.legacy_solver.resolve()),
            "--config",
            str(args.config.resolve()),
            "--output",
            str(baseline),
            "--prediction-only",
            "--spacing",
            *map(str, spacing),
        ]
        with (out / f"group{i}_baseline.log").open("x") as f:
            subprocess.run(
                command,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        initial, baseline_meta, baseline_identity = load_external_reference(
            baseline,
            Path(str(baseline) + ".json"),
            dataset["frame"],
            lo,
            spacing,
            shape,
            step,
            voxel_um=dataset["voxel_um"],
        )
        if (
            baseline_meta["origin_xyz"] != lo.tolist()
            or baseline_meta["spacing_xyz"] != spacing.tolist()
            or baseline_meta["shape_zyx"] != list(shape)
        ):
            raise ValueError("baseline grid differs from requested geometry")
        g, w, pairs, meta = load_paris_box(
            lo,
            hi,
            spacing=grid["spacing_xyz"],
            afv_path=dataset["afv"]["path"],
            dataset=dataset,
            shape_zyx=shape,
        )
        start = -np.asarray(initial, dtype=np.float32)
        start -= start.flat[0]
        print("PREDICT candidate", i, flush=True)
        field, info = solve_field(
            g,
            w,
            grid["spacing_xyz"],
            pairs,
            lam=config["lam"],
            initial=start,
            device=args.device or dataset.get("solver", {}).get("device", "cpu"),
            rtol=config["rtol"],
            maxiter=config["maxiter"],
        )
        name = out / f"group{i}_candidate.npy"
        save_field(name, field, meta, info, step=step)
        if (
            sha256(baseline) != baseline_identity["field_sha256"]
            or sha256(str(baseline) + ".json") != baseline_identity["metadata_sha256"]
        ):
            raise ValueError("baseline changed after validation")
        artifacts.append(
            dict(
                group=group,
                baseline_command=command,
                baseline=str(baseline),
                baseline_sha256=sha256(baseline),
                candidate=str(name),
                candidate_sha256=sha256(name),
                baseline_metadata_sha256=sha256(str(baseline) + ".json"),
                candidate_metadata_sha256=sha256(str(name) + ".json"),
            )
        )
        (out / "progress.json").write_text(
            json.dumps(dict(completed=i + 1, total=len(geometry["initial_test"]["groups"])))
        )
    seal = dict(
        config=config,
        artifacts=artifacts,
        geometry_manifest=str(args.manifest),
        sealed_utc=time.time(),
        labels_expected_sha256=geometry["label_files"]["initial-test-labels.json"]["sha256"],
    )
    (out / "PREDICTIONS_SEALED.json").write_text(json.dumps(seal, indent=2) + "\n")
    print("PREDICTIONS SEALED before official scoring; source-table label access is not isolated", flush=True)


def score(args):
    from paris_pair_check import load_pair_records, score_pairs, summarize_rows

    seal_path = args.output / "PREDICTIONS_SEALED.json"
    seal = json.loads(seal_path.read_text())
    extra_path = args.output / "GLOBAL_SECONDARY_SEAL.json"
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else None
    if extra and (
        extra["primary_seal_sha256"] != sha256(seal_path)
        or extra["field_sha256"] != sha256(extra["field"])
        or extra["metadata_sha256"] != sha256(extra["field"] + ".json")
    ):
        raise ValueError("global prediction identity differs from pre-unsealing seal")
    if sha256(args.labels) != seal["labels_expected_sha256"]:
        raise ValueError("sealed label identity mismatch")
    for art in seal["artifacts"]:
        for kind in ("baseline", "candidate"):
            if (
                sha256(art[kind]) != art[kind + "_sha256"]
                or sha256(art[kind] + ".json") != art[kind + "_metadata_sha256"]
            ):
                raise ValueError("prediction changed after sealing")
    records, _, exclusions = load_pair_records(args.labels)
    results = {k: [] for k in ("baseline", "candidate", "constant_control")}
    if extra:
        results["whole_coarse_secondary"] = []
    for art in seal["artifacts"]:
        group = art["group"]
        selected = [r for r in records if r["collection_id"] in group["collection_ids"]]
        for kind in results:
            source = "candidate" if kind == "constant_control" else kind
            path = extra["field"] if kind == "whole_coarse_secondary" else art[source]
            field = np.load(path, mmap_mode="r")
            meta = json.loads(Path(path + ".json").read_text())
            if kind == "constant_control":
                field = np.zeros(field.shape, np.float32)
            rows, _ = score_pairs(field, meta, selected)
            for row in rows:
                row["region_cluster_id"] = group["group_id"]
            results[kind].extend(rows)
    result = dict(
        predictions_seal_sha256=sha256(seal_path),
        labels_sha256=sha256(args.labels),
        secondary_seal_sha256=sha256(extra_path) if extra else None,
        scored_utc=time.time(),
        selection_exclusions=exclusions,
        methods={k: dict(rows=v, summary=summarize_rows(v)) for k, v in results.items()},
        limitations=[
            "Three spatial groups, one scroll, unknown surface lineage/pretraining overlap.",
            "Prediction artifacts were sealed; the legacy baseline could read the public annotation table.",
            "Descriptive evaluation; no confirmatory superiority confidence interval.",
        ],
    )
    with (args.output / "test_results.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: v["summary"] for k, v in result["methods"].items()}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["predict", "score"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--config", type=Path, help="required dataset configuration for prediction")
    parser.add_argument("--legacy-solver", type=Path, default=Path(__file__).resolve().with_name("paris_solve.py"))
    parser.add_argument("--device")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "predict":
        if not args.manifest or not args.development or not args.config:
            parser.error("predict needs --config, geometry manifest and development receipts")
        predict(args)
    else:
        if not args.labels:
            parser.error("score needs sealed labels")
        score(args)
