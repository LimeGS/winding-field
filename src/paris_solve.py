#!/usr/bin/env python3
"""Legacy index-grid solver on explicitly configured data and calibration.

The historical filename remains importable; use --config for any dataset.
The shared physical solver is the preferred construction interface.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dataset_config import artifact_path, load_dataset, mode_constraints, positive
from paris_data import load_paris_box, sample_source, save_field
from patch_check import sha256
from physical_grid import grid_shape

MARGIN = 100
CAP = 40_000_000


def working_shape(lo_xyz, hi_xyz, spacing):
    return grid_shape(lo_xyz, hi_xyz, np.broadcast_to(spacing, (3,)))


def run_config(path=None):
    path = path or os.environ.get("WREV_CONFIG")
    if not path:
        raise ValueError("explicit dataset configuration required: --config or WREV_CONFIG")
    dataset = load_dataset(path)
    if os.environ.get("RGT"):
        raise ValueError("RGT is not connected; provide declared observation sources")
    requested = os.environ.get("ORIENT")
    if requested and {"umbi": "radial"}.get(requested, requested) != dataset["orientation"]["mode"]:
        raise ValueError("orientation override differs from dataset configuration")
    controls = mode_constraints(dataset, {"pair_weight", "fiber_limit", "pair_limit"}, "legacy nodal solver")
    afv = os.environ.get("AFV", dataset["afv"]["path"])
    if not afv:
        raise ValueError("explicit AFV path must not be empty")
    config = dict(
        dataset=dataset,
        orient=dataset["orientation"]["mode"],
        lam=float(os.environ.get("LAM", controls.get("pair_weight", 3))),
        afv=afv,
        fiber_limit=int(os.environ.get("FIBER_LIMIT", controls.get("fiber_limit", 400))),
        pair_limit=int(os.environ.get("PAIR_LIMIT", controls.get("pair_limit", 2000))),
        device=os.environ.get("WREV_DEVICE", dataset.get("solver", {}).get("device", "cpu")),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    if not np.isfinite(config["lam"]) or config["lam"] < 0 or config["fiber_limit"] < 1 or config["pair_limit"] < 1:
        raise ValueError("invalid fibre weight or limits")
    return config


def load_pairs(path, zlo, zhi, *, max_dw=None):
    from paris_pair_check import load_pair_records

    records, collections, _ = load_pair_records(path, zlo, zhi, max_dw=max_dw)
    return [(np.array(r["a_xyz"]), np.array(r["b_xyz"]), r["dw"], r["collection_id"]) for r in records], collections


def group_slabs(colls, spacing):
    h = np.broadcast_to(np.asarray(spacing, float), (3,))
    if not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError("invalid working spacing")
    items = sorted(colls.items(), key=lambda kv: kv[1][:, 2].mean())
    slabs, cur, curbox = [], [], None
    for k, points in items:
        lo, hi = points.min(0) - MARGIN, points.max(0) + MARGIN
        if curbox is None:
            cur, curbox = [(k, points)], [lo, hi]
            continue
        nlo, nhi = np.minimum(curbox[0], lo), np.maximum(curbox[1], hi)
        if np.prod((nhi - nlo) // h) > CAP:
            slabs.append((cur, curbox))
            cur, curbox = [(k, points)], [lo, hi]
        else:
            cur.append((k, points))
            curbox = [nlo, nhi]
    if cur:
        slabs.append((cur, curbox))
    return slabs


def solve_legacy(target, weights, pairs, spacing, device, lam, rtol=0.001, maxiter=3000):
    """Retain the legacy index-space operator and median gauge; report convergence."""
    if not np.isfinite(rtol) or rtol <= 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError("invalid solver tolerance/iteration budget")
    shape = weights.shape
    size = weights.size
    w = torch.as_tensor(weights, dtype=torch.float32, device=device).ravel()
    g = torch.as_tensor(target * np.asarray(spacing), dtype=torch.float32, device=device).reshape(-1, 3)
    fp, fq = (torch.as_tensor(pairs[:, i], dtype=torch.int64, device=device) for i in (0, 1))

    def shift(value, axis, delta):
        return torch.roll(value.reshape(shape), delta, axis).ravel()

    def noborder(value, axis, last):
        value = value.reshape(shape)
        index = [slice(None)] * 3
        index[axis] = shape[axis] - 1 if last else 0
        value[tuple(index)] = 0
        return value.ravel()

    def matvec(value):
        result = torch.zeros_like(value)
        for axis in range(3):
            wf = noborder(0.5 * (w + shift(w, axis, -1)), axis, True)
            wb = noborder(0.5 * (w + shift(w, axis, 1)), axis, False)
            result += wf * (value - shift(value, axis, -1)) + wb * (value - shift(value, axis, 1))
        if len(pairs):
            result += lam * torch.bincount(
                torch.cat([fp, fq]), torch.cat([value[fp] - value[fq], value[fq] - value[fp]]), minlength=size
            )
        return result

    b, diagonal = torch.zeros(size, device=device), torch.zeros(size, device=device)
    for axis in range(3):
        ga = g[:, 2 - axis]
        gf = noborder(0.5 * (ga + shift(ga, axis, -1)), axis, True)
        gb = noborder(0.5 * (ga + shift(ga, axis, 1)), axis, False)
        wf = noborder(0.5 * (w + shift(w, axis, -1)), axis, True)
        wb = noborder(0.5 * (w + shift(w, axis, 1)), axis, False)
        b += wf * gf - wb * gb
        diagonal += wf + wb
    if len(pairs):
        diagonal += lam * torch.bincount(torch.cat([fp, fq]), torch.ones(2 * len(pairs), device=device), minlength=size)
    inverse = 1 / diagonal.clamp_min(1e-6)
    x = torch.zeros(size, device=device)
    residual = b.clone()
    norm = float(b.norm())
    z = inverse * residual
    direction = z.clone()
    rz = (residual * z).sum()
    iteration = 0
    for iteration in range(1, maxiter + 1):
        if norm == 0 or float(residual.norm()) <= rtol * norm:
            break
        product = matvec(direction)
        denominator = (direction * product).sum()
        if not torch.isfinite(denominator) or float(denominator) <= 0:
            break
        alpha = rz / denominator
        x += alpha * direction
        residual -= alpha * product
        z = inverse * residual
        rz2 = (residual * z).sum()
        direction = z + (rz2 / rz) * direction
        rz = rz2
    relative = float(residual.norm()) / norm if norm else 0.0
    field = (x.reshape(shape) - x.median()).cpu().numpy()
    info = dict(
        converged=bool(np.isfinite(field).all() and np.isfinite(relative) and relative <= rtol),
        iterations=iteration,
        relative_residual=relative,
        method="legacy_index_grid",
        gauge="median",
    )
    return field, info


def ridge_diagnostics(detection, field):
    """Retain the historical ridge estimate as evidence, not an implicit scale fit."""
    detection = np.asarray(detection, bool)
    if detection.shape != field.shape or detection.ndim != 3 or not np.isfinite(field).all():
        raise ValueError("ridge diagnostics require matching finite volumes")
    result = dict(
        ridge_axis=None,
        ridge_count=None,
        ridge_line_counts=[],
        field_range_98pct=None,
        ridge_rule_step=None,
        ridge_calibration_applied=False,
    )
    if min(detection.shape) < 2 or not detection.any():
        return result
    gradients = np.stack(np.gradient(detection.astype(np.float32)), -1).reshape(-1, 3)
    axis = int(np.argmax(np.linalg.eigh(np.cov(gradients.T))[1][:, -1] ** 2))
    other = [i for i in range(3) if i != axis]
    centers = [
        range(max(0, detection.shape[i] // 2 - 2), min(detection.shape[i], detection.shape[i] // 2 + 3)) for i in other
    ]
    counts = []
    for a in centers[0]:
        for b in centers[1]:
            index = [slice(None)] * 3
            index[other[0]], index[other[1]] = a, b
            line = detection[tuple(index)]
            counts.append(int((np.diff(line.astype(np.int8)) == 1).sum()) + int(line[0]))
    positive_counts = [n for n in counts if n > 0]
    count = float(np.median(positive_counts)) if positive_counts else None
    span = float(np.percentile(field[detection], 99) - np.percentile(field[detection], 1))
    result.update(
        ridge_axis=axis,
        ridge_count=count,
        ridge_line_counts=counts,
        field_range_98pct=span,
        ridge_rule_step=span / count if count is not None and count >= 2 else None,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--spacing", type=float, nargs=3, help="explicit XYZ working-spacing override")
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args(argv)
    config = run_config(args.config)
    dataset = config["dataset"]
    grid = dataset["grid"]
    h = np.array(args.spacing if args.spacing is not None else grid["spacing_xyz"], float)
    if h.shape != (3,) or not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError("invalid working spacing")
    config["working_spacing_xyz"] = h.tolist()
    lo = np.array(grid["origin_xyz"], float)
    hi = lo + np.array(grid["spacing_xyz"]) * np.array(grid["shape_zyx"][::-1])
    predict_only = args.prediction_only or os.environ.get("PREDICT_ONLY") == "1"
    audit_value = args.audit_dir or os.environ.get("AUDIT_DIR")
    if predict_only and audit_value:
        raise ValueError("prediction-only cannot open a label audit")
    annotations = dataset.get("artifacts", {}).get("annotations")
    records, colls = [], {}
    if not predict_only and annotations is not None:
        from paris_pair_check import load_pair_records

        annotation_path = artifact_path(dataset, "annotations")
        annotation_header = json.loads(annotation_path.read_text())
        if annotation_header.get("frame", dataset["frame"]) != dataset["frame"]:
            raise ValueError("annotation file and dataset frames differ")
        if "voxel_um" in annotation_header and not np.isclose(
            positive(annotation_header["voxel_um"], "annotation voxel_um"), dataset["voxel_um"], rtol=1e-12, atol=0
        ):
            raise ValueError("annotation file and dataset physical units differ")
        config["annotations_sha256"] = sha256(annotation_path)
        config["annotation_coordinate_authority"] = "file declarations reconciled with explicit dataset context"
        zlo, zhi = dataset.get("context", {}).get("label_z_range", [float(lo[2]), float(hi[2])])
        records, colls, _ = load_pair_records(annotation_path, zlo, zhi)
    bounds = os.environ.get("SLAB")
    if bounds:
        raw = np.array([float(v) for v in bounds.split(",")]).reshape(3, 2)
        lo, hi = raw[:, 0], raw[:, 1]
    slabs = [(list(colls.items()), [lo, hi])]
    if colls and not bounds:
        slabs = group_slabs(colls, h)
    output = args.output or os.environ.get("OUT")
    if output and len(slabs) != 1:
        raise ValueError("one output field requires one explicit slab")
    if not output and not audit_value:
        raise ValueError("--output or --audit-dir is required")
    audit = Path(audit_value) if audit_value else None
    if audit:
        audit.mkdir(exist_ok=False)
    if output and (Path(output).exists() or Path(str(output) + ".json").exists()):
        raise FileExistsError(output)
    config["prediction_only"] = predict_only
    audit_rows = []
    started = time.time()
    for i, (members, (lower, upper)) in enumerate(slabs):
        target, weights, pairs, meta = load_paris_box(
            lower,
            upper,
            spacing=h,
            afv_path=config["afv"],
            dataset=dataset,
            fiber_limit=config["fiber_limit"],
            pair_limit=config["pair_limit"],
            shape_zyx=grid["shape_zyx"] if args.spacing is None and not bounds and not colls else None,
        )
        field, info = solve_legacy(
            target,
            weights,
            pairs,
            h,
            config["device"],
            config["lam"],
            dataset.get("solver", {}).get("rtol", 0.001),
            dataset.get("solver", {}).get("maxiter", 3000),
        )
        path = audit / f"slab{i}.npy" if audit else Path(output)
        if not info["converged"]:
            with Path(str(path) + ".solver.json").open("x") as stream:
                json.dump(info, stream, allow_nan=False)
            raise ValueError("legacy solver did not converge; field was not published")
        detection = (
            sample_source(dataset, "surface", lower, h, weights.shape, order=0)
            > dataset["prediction"]["surface_threshold"]
        )
        meta.update(ridge_diagnostics(detection, field))
        meta["observation_source_sha256"] = meta["source_sha256"]
        meta.update(
            config=config,
            constraint_pairs=len(pairs),
            source_sha256=sha256(__file__),
            collection_ids=[str(key) for key, _ in members],
            human_patches_used_as_constraints=False,
        )
        metadata = save_field(path, field, meta, info, dataset["sheet_step"])
        if audit:
            from paris_pair_check import score_pairs, summarize_rows

            selected = [r for r in records if r["collection_id"] in metadata["collection_ids"]]
            rows, summary = score_pairs(field, metadata, selected)
            for row in rows:
                row.update(slab_id=i, field_sha256=metadata["field_sha256"])
            audit_rows.extend(rows)
            (audit / f"slab{i}_pairs.json").write_text(json.dumps(dict(rows=rows, summary=summary), allow_nan=False))
            np.savez(audit / f"slab{i}_constraints.npz", pairs=pairs, fiber_ids=meta["constraint_fibre_ids"])
    if audit:
        (audit / "pairs.json").write_text(
            json.dumps(dict(rows=audit_rows, summary=summarize_rows(audit_rows)), allow_nan=False)
        )
        (audit / "DONE").write_text("Configured fields and identifiable pair receipts saved.\n")
    print(
        json.dumps(
            dict(
                status="CONVERGED",
                slabs=len(slabs),
                wall_s=time.time() - started,
                human_controls="NOT_EVALUATED" if not records else "supplied development annotations",
            )
        )
    )


if __name__ == "__main__":
    main()
