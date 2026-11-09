#!/usr/bin/env python3
"""Evaluate a held-out human TIFXYZ surface against a calibrated voxel field.

The patch's XYZ values are volume coordinates; meta.scale describes its UV
sampling, NOT a scale to multiply into XYZ. No outside points are clamped.
Constancy is necessary, not sufficient: a collapsed field also looks constant.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sample_field(field, xyz, origin, spacing, margin=0):
    xyz = np.asarray(xyz, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (
        field.ndim != 3
        or xyz.shape[-1] != 3
        or origin.shape != (3,)
        or spacing.shape != (3,)
        or not np.isfinite(spacing).all()
        or not np.isfinite(origin).all()
        or (spacing <= 0).any()
        or margin < 0
    ):
        raise ValueError("expected finite XYZ origin, positive spacing and ZYX field")
    grid = (xyz - origin) / spacing
    valid = (
        np.isfinite(xyz).all(-1)
        & (grid >= margin).all(-1)
        & (grid <= np.asarray(field.shape[::-1]) - 1 - margin).all(-1)
    )
    values = np.full(xyz.shape[:-1], np.nan)
    if valid.any():
        values[valid] = map_coordinates(
            field, grid[valid, ::-1].T, order=1, prefilter=False, mode="constant", cval=np.nan
        )
    return values


def error_summary(residual):
    values = residual[np.isfinite(residual)]
    result = {f"abs_p{q}_sheets": None for q in (50, 90, 95)}
    result.update(
        within_0_5_sheet_pct=None, within_1_sheet_pct=None, within_2_sheets_pct=None, span_p05_p95_sheets=None
    )
    if values.size:
        for q in (50, 90, 95):
            result[f"abs_p{q}_sheets"] = float(np.percentile(np.abs(values), q))
        for t, key in ((0.5, "within_0_5_sheet_pct"), (1, "within_1_sheet_pct"), (2, "within_2_sheets_pct")):
            result[key] = float(100 * (np.abs(values) <= t).mean())
        result["span_p05_p95_sheets"] = float(np.diff(np.percentile(values, [5, 95]))[0])
    return result


def evaluate_patch(field, xyz, origin, spacing, step, margin=8, max_edge=100, min_component=2):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("calibration step must be positive and finite")
    valid = np.isfinite(xyz).all(-1) & (xyz >= 0).all(-1)
    values = sample_field(field, xyz, origin, spacing, margin)
    values[~valid] = np.nan  # TIFXYZ uses negative sentinels; generic world coordinates may be negative.
    covered = np.isfinite(values)
    residual = np.full(values.shape, np.nan)
    count = int(covered.sum())
    result = dict(
        valid_points=int(valid.sum()),
        covered_points=count,
        coverage_pct=100 * count / max(int(valid.sum()), 1),
        step=float(step),
        margin_grid_voxels=margin,
        abs_p50_sheets=None,
        abs_p90_sheets=None,
        abs_p95_sheets=None,
        within_0_5_sheet_pct=None,
        within_1_sheet_pct=None,
        within_2_sheets_pct=None,
        span_p05_p95_sheets=None,
        covered_edges=0,
        edge_jump_gt1_pct=None,
        connected_components=0,
        local=error_summary(residual),
        components=[],
    )
    if not count:
        return result, residual
    residual[covered] = (values[covered] - np.median(values[covered])) / step
    result.update(error_summary(residual))
    jumps, rows, cols = [], [], []
    index = np.full(values.shape, -1, dtype=int)
    index[covered] = np.arange(count)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        distance = np.linalg.norm(xyz[a] - xyz[b], axis=-1)
        use = covered[a] & covered[b] & (distance > 0) & (distance <= max_edge)
        jumps.extend(np.abs(residual[a][use] - residual[b][use]).tolist())
        rows.extend(index[a][use].tolist())
        cols.extend(index[b][use].tolist())
    result["covered_edges"] = len(jumps)
    if jumps:
        result["edge_jump_gt1_pct"] = float(100 * (np.asarray(jumps) > 1).mean())
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(count, count))
    nc, labels = connected_components(graph, directed=False)
    result["connected_components"] = int(nc)
    local = np.full(count, np.nan)
    samples = values[covered]
    # A crop can intersect multiple turns of a single long spiral surface.
    # Their independent local pieces need not have the same winding index.
    for label in range(nc):
        use = labels == label
        if use.sum() < min_component:
            continue
        local[use] = (samples[use] - np.median(samples[use])) / step
        result["components"].append(dict(points=int(use.sum()), **error_summary(local[use])))
    result["local"] = dict(points=int(np.isfinite(local).sum()), **error_summary(local))
    residual[covered] = local
    return result, residual


def load_patch(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta["format"] != "tifxyz":
        raise ValueError(f"{path}: not tifxyz")
    xyz = np.stack([tifffile.imread(path / f"{axis}.tif") for axis in "xyz"], -1)
    if xyz.ndim != 3:
        raise ValueError(f"{path}: expected a 2D coordinate grid")
    return meta, xyz
