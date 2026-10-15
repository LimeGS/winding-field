#!/usr/bin/env python3
"""Configured coarse reference; only observed black OUTER box samples are pinned.

This produces a globally gauged hypothesis, not certified sheet identities.
Interior air is never an exterior anchor (inter-layer air can be connected).
"""

import argparse
import hashlib
import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np

from common import arclength
from dataset_config import artifact_path, load_dataset, mode_constraints, native_to_world, validate_afv
from field_solver import solve_field
from paris_data import black_value, exterior_faces, load_paris_box, sample_source, save_field
from patch_check import sha256
from sealed_predict import development_step


def pin_outer_face(fixed, values, axis, side, background=0):
    if axis not in (0, 1, 2) or side not in (0, fixed.shape[axis] - 1):
        raise ValueError("only outer faces may be background anchors")
    shape = list(fixed.shape)
    shape[axis] = 1
    if values.shape != tuple(shape) or not np.isfinite(values).all():
        raise ValueError("invalid face samples")
    sl = [slice(None)] * 3
    sl[axis] = slice(side, side + 1)
    fixed[tuple(sl)] = np.where(values == background, 0, np.nan)


def global_fibre_constraints(
    afv_path, origin_xyz, spacing_xyz, shape_zyx, excluded_ids=(), excluded_lineage=(), *, transform, voxel_mm
):
    """Stream all AFV fibres with the declared native/world/ZYX geometry.

    Eight distinct nodal pairs per fibre at most; no global/local fibre cap.
    Nearest-node quantization is explicit and can merge sheets on a coarse grid.
    """
    origin = np.asarray(origin_xyz, float)
    h = np.asarray(spacing_xyz, float)
    raw_shape = np.asarray(shape_zyx)
    if (
        origin.shape != (3,)
        or h.shape != (3,)
        or raw_shape.shape != (3,)
        or raw_shape.dtype.kind not in "iu"
        or (raw_shape < 1).any()
        or not np.isfinite([origin, h]).all()
        or (h <= 0).any()
    ):
        raise ValueError("invalid physical XYZ/ZYX grid")
    shape = raw_shape.astype(int)
    last = origin + (shape[::-1] - 1) * h
    excluded_ids = set(excluded_ids)
    excluded_lineage = set(map(tuple, excluded_lineage))
    audit = dict(
        afv_sha256=sha256(afv_path),
        fibres_seen=0,
        excluded_fibres=0,
        invalid_fibres=0,
        no_eligible_nodal_pair=0,
        used_fibre_ids=[],
        max_pairs_per_fibre=8,
        min_span_mm=0.3,
        seed_rule="default_rng(fibre_id)",
        quantization="nearest grid node, np.rint; no clamping",
        origin_xyz=origin.tolist(),
        spacing_xyz=h.tolist(),
        shape_zyx=shape.tolist(),
    )
    arrays = []
    db = sqlite3.connect(f"file:{Path(afv_path).resolve()}?mode=ro", uri=True)
    try:
        query = """SELECT f.id,f.annotation,b.points FROM fibers f JOIN blocks b
                 ON f.id=b.fiber_id ORDER BY f.id,b.first_segment"""
        for fid, group in itertools.groupby(db.execute(query), key=lambda r: r[0]):
            chunks = []
            for _, annotation, blob in group:
                chunks.append(np.frombuffer(blob, dtype="<f8").reshape(-1, 3))
            audit["fibres_seen"] += 1
            annotation = json.loads(annotation)
            collection = annotation.get("sourceCollectionUUID")
            lineage = {(str(collection), str(v)) for v in annotation.get("sourceFiberIDs", [])} if collection else set()
            if fid in excluded_ids or lineage & excluded_lineage:
                audit["excluded_fibres"] += 1
                continue
            points = np.concatenate(chunks)
            if len(points) < 2 or not np.isfinite(points).all():
                audit["invalid_fibres"] += 1
                continue
            points = native_to_world(points, transform)
            inside = ((points >= origin) & (points <= last)).all(1)
            borders = np.diff(np.r_[False, inside, False].astype(np.int8))
            candidates = []
            for a, b in zip(np.flatnonzero(borders == 1), np.flatnonzero(borders == -1)):
                run = points[a:b]
                if len(run) < 2:
                    continue
                s = arclength(run, voxel_mm)
                nodes = np.rint((run - origin) / h).astype(int)[:, ::-1]
                assert (nodes >= 0).all() and (nodes < shape).all()
                flat = np.ravel_multi_index(nodes.T, shape)
                i, j = np.triu_indices(len(run), 1)
                use = ((s[j] - s[i]) >= 0.3) & (flat[i] != flat[j])
                if use.any():
                    candidates.append(np.sort(np.column_stack([flat[i[use]], flat[j[use]]]), axis=1))
            if not candidates:
                audit["no_eligible_nodal_pair"] += 1
                continue
            pairs = np.unique(np.concatenate(candidates), axis=0)
            if len(pairs) > 8:
                select = np.random.default_rng(int(fid)).choice(len(pairs), 8, replace=False)
                pairs = pairs[np.sort(select)]
            arrays.append(pairs)
            audit["used_fibre_ids"].append(int(fid))
    finally:
        db.close()
    result = np.concatenate(arrays).astype(np.int64) if arrays else np.empty((0, 2), np.int64)
    audit.update(constraint_pairs=len(result), pairs_array_sha256=hashlib.sha256(result.tobytes()).hexdigest())
    return result, audit


def solve_global_reference(target, weights, spacing, pairs, fixed, device, lam=3.0, rtol=0.001, maxiter=4000):
    return solve_field(
        target, weights, spacing, pairs=pairs, lam=lam, fixed=fixed, device=device, rtol=rtol, maxiter=maxiter
    )


def validate_reservation(excluded, afv):
    if (
        not isinstance(excluded.get("fibre_ids"), list)
        or any(type(v) is not int or v < 0 for v in excluded["fibre_ids"])
        or not isinstance(excluded.get("source_lineage"), list)
        or any(
            not isinstance(v, list) or len(v) != 2 or not all(isinstance(x, str) for x in v)
            for v in excluded["source_lineage"]
        )
    ):
        raise ValueError("invalid fibre exclusion manifest")
    if excluded.get("afv_sha256") != sha256(afv):
        raise ValueError("reservation AFV identity differs from constraint source")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--development", type=Path, help="optional evidence that must agree with configured calibration")
    p.add_argument("--step", type=float, help="optional isotropic reconstruction spacing override")
    p.add_argument("--device")
    p.add_argument("--fibre-weight", type=float)
    p.add_argument("--afv", type=Path, help="optional AFV override using the declared native geometry")
    p.add_argument("--exclude-fibres", type=Path, help="JSON with fibre_ids and source_lineage")
    p.add_argument("--baseline-field", type=Path, help="verify unchanged grid, observations, anchors and calibration")
    p.add_argument(
        "--constraint-mode",
        choices=("nodal", "trilinear_pairs", "fibre_consensus", "robust_consensus"),
        default="nodal",
    )
    p.add_argument("--split-manifest", type=Path, help="AFV-bound lineage/length split; train-only unless --production")
    p.add_argument("--production", action="store_true", help="use all eligible fibres; never call its scores held-out")
    a = p.parse_args()
    dataset = load_dataset(a.config)
    controls = mode_constraints(
        dataset,
        {"pair_weight"} if a.constraint_mode == "nodal" else {"pair_weight", "pitch_mm", "trace_target"},
        "whole-reference " + a.constraint_mode,
    )
    solver_options = dataset.get("solver", {})
    if a.split_manifest is None:
        a.split_manifest = artifact_path(dataset, "split")
    if a.fibre_weight is None:
        a.fibre_weight = dataset.get("constraints", {}).get("pair_weight", 3.0)
    if not np.isfinite(a.fibre_weight) or a.fibre_weight < 0:
        p.error("fibre weight must be finite and nonnegative")
    a.device = a.device or dataset.get("solver", {}).get("device", "cpu")
    a.output.mkdir(exist_ok=False)
    afv = a.afv or Path(dataset["afv"]["path"])
    validate_afv(dataset, afv)
    excluded = json.loads(a.exclude_fibres.read_text()) if a.exclude_fibres else dict(fibre_ids=[], source_lineage=[])
    if a.exclude_fibres:
        validate_reservation(excluded, afv)
    if a.split_manifest:
        split = json.loads(a.split_manifest.read_text())
        from evaluation_split import validate_fibre_split

        validate_fibre_split(split, sha256(afv))
        if not a.production:
            excluded["fibre_ids"] = sorted(
                set(excluded["fibre_ids"]) | {r["fibre_id"] for r in split["fibres"] if r["split"] == "reserved"}
            )
    grid = dataset["grid"]
    lo = np.asarray(grid["origin_xyz"], float)
    h = np.full(3, a.step) if a.step is not None else np.asarray(grid["spacing_xyz"], float)
    hi = lo + np.asarray(grid["spacing_xyz"]) * np.asarray(grid["shape_zyx"][::-1])
    target, weights, pairs, meta = load_paris_box(
        lo, hi, spacing=h, dataset=dataset, shape_zyx=grid["shape_zyx"] if a.step is None else None
    )
    fixed = np.full(weights.shape, np.nan, np.float32)
    faces = []
    for axis, edge in exterior_faces(dataset):
        side = 0 if edge == 0 else weights.shape[axis] - 1
        shape = list(weights.shape)
        shape[axis] = 1
        origin = lo.copy()
        origin[2 - axis] += side * h[2 - axis]
        receipts = []
        values = sample_source(dataset, "ct", origin, h, shape, order=0, receipts=receipts)
        pin_outer_face(fixed, values, axis, side, black_value(dataset))
        faces.append(
            dict(
                axis_zyx=axis,
                index=side,
                black=int((values == black_value(dataset)).sum()),
                black_value=black_value(dataset),
                total=values.size,
                source=receipts,
            )
        )
    pins = int(np.isfinite(fixed).sum())
    if pins < 10:
        raise ValueError("insufficient verified outer background anchors")
    step = dataset["sheet_step"]
    if a.development and not np.isclose(
        development_step(json.loads(a.development.read_text())["rows"]), step, rtol=1e-12, atol=0
    ):
        raise ValueError("development calibration differs from the fixed baseline/configured scale")
    if a.baseline_field:
        baseline = json.loads(Path(str(a.baseline_field) + ".json").read_text())
        if sha256(a.baseline_field) != baseline["field_sha256"]:
            raise ValueError("baseline field hash mismatch")
        for key in (
            "frame",
            "origin_xyz",
            "spacing_xyz",
            "shape_zyx",
            "voxel_um",
            "prediction_parameters",
            "normal_encoding",
            "normal_orientation",
            "source_receipts",
            "umbi_sha256",
            "source_sha256",
        ):
            if meta[key] != baseline[key]:
                raise ValueError(f"baseline inputs changed: {key}")
        if (
            step != baseline["step"]
            or faces != baseline["whole_reference"]["faces"]
            or not np.array_equal(np.isfinite(fixed), np.load(a.baseline_field.parent / "outer_anchor_mask.npy"))
        ):
            raise ValueError("baseline scale or outer anchors changed")
    pairs, audit = global_fibre_constraints(
        afv,
        meta["origin_xyz"],
        meta["spacing_xyz"],
        weights.shape,
        excluded["fibre_ids"],
        excluded["source_lineage"],
        transform=dataset["afv"],
        voxel_mm=dataset["voxel_um"] / 1000,
    )
    if not len(pairs):
        raise ValueError("global reference requires nonempty usable fibre constraints")
    audit.update(
        exclusion_manifest_sha256=sha256(a.exclude_fibres) if a.exclude_fibres else None,
        split_manifest_sha256=sha256(a.split_manifest) if a.split_manifest else None,
        production=a.production,
        selector_source_sha256=sha256(__file__),
        lambda_value=a.fibre_weight,
    )
    np.save(a.output / "fibre_pairs.npy", pairs)
    audit["pairs_file_sha256"] = sha256(a.output / "fibre_pairs.npy")
    (a.output / "fibre_constraints.json").write_text(json.dumps(audit, indent=2) + "\n")
    meta.update(
        afv_sha256=audit["afv_sha256"],
        constraint_pairs=len(pairs),
        fibre_constraints_sha256=sha256(a.output / "fibre_constraints.json"),
    )
    config = dict(
        scope="whole_scan_coarse_reference_hypothesis",
        dataset_config_sha256=dataset["config_sha256"],
        working_shape_zyx=list(weights.shape),
        spacing_xyz=h.tolist(),
        sample_spacing_um=(h * dataset["voxel_um"]).tolist(),
        fixed_outer_samples=pins,
        exterior_authority=dataset.get("context", {}).get("exterior_authority"),
        faces=faces,
        global_calibration_step=step,
        fibre_constraints=dict(
            pairs=len(pairs),
            fibres=len(audit["used_fibre_ids"]),
            lambda_value=a.fibre_weight,
            audit_sha256=sha256(a.output / "fibre_constraints.json"),
        ),
        baseline_field_sha256=sha256(a.baseline_field) if a.baseline_field else None,
        limitations=[
            "No fine-resolution whole-scroll reconstruction.",
            "No proof of winding topology or ordering from numerical convergence.",
        ],
    )
    (a.output / "config.json").write_text(json.dumps(config, indent=2))
    print("WHOLE SOLVE", weights.shape, "pins", pins, "fibre pairs", len(pairs), flush=True)
    if a.constraint_mode == "nodal":
        field, info = solve_global_reference(
            target,
            weights,
            h,
            pairs,
            fixed,
            a.device,
            lam=a.fibre_weight,
            rtol=solver_options.get("rtol", 0.001),
            maxiter=solver_options.get("maxiter", 4000),
        )
    else:
        from coherence_pipeline import compile_fibres, match_trace
        from common import iter_fibre_records
        from field_constraints import solve_with_relations

        omitted = set(excluded["fibre_ids"])
        lineage = set(map(tuple, excluded["source_lineage"]))

        def usable():
            for record in iter_fibre_records(afv):
                annotation = record["annotation"]
                collection = annotation.get("sourceCollectionUUID")
                source = (
                    {(str(collection), str(v)) for v in annotation.get("sourceFiberIDs", [])} if collection else set()
                )
                if record["fibre_id"] in omitted or source & lineage:
                    continue
                record["points_xyz"] = native_to_world(record["points_xyz"], dataset["afv"])
                yield record

        continuous, consensus, compiled = compile_fibres(
            usable(),
            meta["origin_xyz"],
            meta["spacing_xyz"],
            weights.shape,
            voxel_mm=dataset["voxel_um"] / 1000,
            pair_weight=a.fibre_weight,
            pitch_mm=controls.get("pitch_mm", 0.15),
        )
        endpoints = dict(a_xyz=compiled.pop("a_xyz"), b_xyz=compiled.pop("b_xyz"))
        np.savez_compressed(a.output / "physical_constraint_points.npz", **endpoints)
        active = continuous
        if a.constraint_mode in ("fibre_consensus", "robust_consensus"):
            active, normalization = match_trace(consensus, continuous.trace)
            compiled["trace_normalization"] = normalization
        if "trace_target" in controls:
            active, normalization = match_trace(active, controls["trace_target"])
            compiled["common_trace_matching"] = normalization
        active.save(a.output / "continuous_constraints.npz")
        compiled.update(
            mode=a.constraint_mode,
            constraint_digest=active.digest,
            split_manifest_sha256=sha256(a.split_manifest) if a.split_manifest else None,
            production=a.production,
        )
        (a.output / "continuous-audit.json").write_text(json.dumps(compiled, indent=2) + "\n")
        config["continuous_constraints"] = compiled
        (a.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        meta.update(
            constraint_mode=a.constraint_mode, continuous_audit_sha256=sha256(a.output / "continuous-audit.json")
        )
        field, info = solve_with_relations(
            target,
            weights,
            h,
            constraints=active,
            robust_delta=0.5 * step if a.constraint_mode == "robust_consensus" else None,
            fixed=fixed,
            device=a.device,
            rtol=solver_options.get("rtol", 0.001),
            maxiter=solver_options.get("maxiter", 4000),
        )
        np.save(a.output / "robust_factors.npy", np.array(info.pop("robust_factors"), np.float32))
    (a.output / "solver.json").write_text(json.dumps(info, indent=2))
    save_field(a.output / "coarse.npy", field, {**meta, "whole_reference": config}, info, step)
    np.save(a.output / "outer_anchor_mask.npy", np.isfinite(fixed))
    (a.output / "DONE").write_text("Globally gauged coarse field saved; physical validation required.\n")
    print(info, flush=True)


if __name__ == "__main__":
    main()
