#!/usr/bin/env python3
"""Shared numerical and read-only fibre helpers.

Conventions (follow these in every script here):
- Fibre coordinates are raw stored XYZ values; callers declare their physical
  units and conversion to the selected field's world frame.
- A "field" maps position -> winding index (float sheets from the outside).
  Level sets at integers are sheets.
- Benchmark numbers come in (span_mm, abs difference in sheets) pairs and
  are reported with report() using Jaime's bins, so tables stay comparable.

Data paths are explicit; no scroll registry or host-location search is used.
"""

import itertools
import json
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

import numpy as np

# Jaime's table bins (span in mm). Keep identical everywhere.
BINS = [
    (0.5, 1.5),
    (1.5, 2.5),
    (2.5, 3.5),
    (3.5, 5),
    (5, 7.5),
    (7.5, 10),
    (10, 15),
    (15, 20),
    (20, 25),
    (25, 30),
    (30, 45),
]


def decode_lasagna_normals(nx, ny):
    """Hemisphere encoding per villa/lasagna/docs/normal_smoothing.md."""
    x = np.clip((np.asarray(nx, dtype=np.float32) - 128) / 127, -1, 1)
    y = np.clip((np.asarray(ny, dtype=np.float32) - 128) / 127, -1, 1)
    z = np.sqrt(np.maximum(0, 1 - x * x - y * y))
    n = np.stack([x, y, z], -1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def load_fibers(db, bbox, min_pts=2, clip=True):
    """Fibre centrelines inside a native XYZ bbox: [(fiber_id, float64 points)].

    Block order is preserved (matters for
    zero-cost backbones: never sort, only dedupe consecutive voxels).
    With clip=False the full centrelines are returned (callers mask manually).
    """
    if not isinstance(db, (str, Path)) or not str(db):
        raise ValueError("explicit AFV path required")
    q = """SELECT DISTINCT b.fiber_id FROM block_bounds r JOIN blocks b ON b.id=r.id
           WHERE r.max_x>=:x0 AND r.min_x<=:x1 AND r.max_y>=:y0 AND r.min_y<=:y1
             AND r.max_z>=:z0 AND r.min_z<=:z1"""
    out = []
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)) as c:
        ids = [r[0] for r in c.execute(q, bbox)]
        for fid in ids:
            rows = c.execute("SELECT first_segment, points FROM blocks WHERE fiber_id=? ORDER BY first_segment", (fid,))
            # 3 x float64 little-endian per point.
            P = np.concatenate(
                [
                    np.array(struct.unpack(f"<{3 * (len(b) // 24)}d", b), dtype=np.float64).reshape(-1, 3)
                    for _, b in rows
                ]
            )
            keep = (
                (P[:, 0] >= bbox["x0"])
                & (P[:, 0] < bbox["x1"])
                & (P[:, 1] >= bbox["y0"])
                & (P[:, 1] < bbox["y1"])
                & (P[:, 2] >= bbox["z0"])
                & (P[:, 2] < bbox["z1"])
            )
            if not clip:
                out.append((fid, P))
            elif keep.sum() >= min_pts:
                out.append((fid, P[keep]))
    out.sort()
    return out


def arclength(P, vox_mm):
    """Cumulative arclength (mm) along a centreline."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1) * vox_mm
    return np.concatenate([[0.0], np.cumsum(seg)])


def iter_fibre_records(afv_path):
    """Read-only streaming native XYZ polylines, annotation lineage and stable IDs."""
    connection = sqlite3.connect(f"file:{Path(afv_path).resolve()}?mode=ro", uri=True)
    try:
        query = """SELECT f.id,f.annotation,b.points FROM fibers f JOIN blocks b
                 ON b.fiber_id=f.id ORDER BY f.id,b.first_segment"""
        for fid, rows in itertools.groupby(connection.execute(query), key=lambda r: r[0]):
            chunks = []
            for _, annotation, blob in rows:
                chunks.append(np.frombuffer(blob, dtype="<f8").reshape(-1, 3))
            yield dict(fibre_id=int(fid), points_xyz=np.concatenate(chunks), annotation=json.loads(annotation))
    finally:
        connection.close()


def sample_pairs(P, s, rng, per_fiber=40, span_lo=0.5, span_hi=45.0):
    """Random index pairs on one fibre -> [(span_mm, p1, p2)]."""
    if s[-1] < span_lo:
        return []
    out = []
    for _ in range(per_fiber):
        i, j = rng.integers(0, len(P), 2)
        if i == j:
            continue
        span = abs(s[j] - s[i])
        if span_lo <= span <= span_hi:
            out.append((span, P[i], P[j]))
    return out


def trilinear(F, z, y, x):
    """Trilinear sample of a ZYX field at fractional indices (clipped)."""
    nz, ny, nx = F.shape
    z = min(max(z, 0.0), nz - 1.001)
    y = min(max(y, 0.0), ny - 1.001)
    x = min(max(x, 0.0), nx - 1.001)
    z0, y0, x0 = int(z), int(y), int(x)
    fz, fy, fx = z - z0, y - y0, x - x0
    c = F[z0 : z0 + 2, y0 : y0 + 2, x0 : x0 + 2]
    return (
        c[0, 0, 0] * (1 - fz) * (1 - fy) * (1 - fx)
        + c[1, 0, 0] * fz * (1 - fy) * (1 - fx)
        + c[0, 1, 0] * (1 - fz) * fy * (1 - fx)
        + c[0, 0, 1] * (1 - fz) * (1 - fy) * fx
        + c[1, 1, 0] * fz * fy * (1 - fx)
        + c[1, 0, 1] * fz * (1 - fy) * fx
        + c[0, 1, 1] * (1 - fz) * fy * fx
        + c[1, 1, 1] * fz * fy * fx
    )


def calib_gap(det, n=1500, seed=0):
    """Median low-detection run length along random axis lines (voxels).

    Estimates the typical air-gap width from the detection volume alone
    (no fibres, no labels). Used to scale travel costs to sheet units.
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = det.shape
    widths = []
    for _ in range(n):
        ax = rng.integers(0, 3)
        line = (
            det[:, rng.integers(0, ny), rng.integers(0, nx)]
            if ax == 0
            else det[rng.integers(0, nz), :, rng.integers(0, nx)]
            if ax == 1
            else det[rng.integers(0, nz), rng.integers(0, ny), :]
        )
        low = np.asarray(line) < 128
        if not low.any() or low.all():
            continue
        d = np.diff(low.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if low[0]:
            starts = [0] + starts
        if low[-1]:
            ends = ends + [len(line)]
        widths += [b - a for a, b in zip(starts, ends) if b - a >= 2]
    return float(np.median(widths)) if widths else 10.0


def report(diffs, bins=BINS, extra_p99=True):
    """Jaime-style table over (span_mm, sheet-diff) pairs."""
    head = (
        f"{'span (mm)':>9} {'n':>6} {'median':>7} {'p90':>6} "
        + (f"{'p99':>6} " if extra_p99 else "")
        + f"{'%<0.5':>6} {'%<1':>5} {'%<2':>5}"
    )
    print(head)
    for lo, hi in bins:
        v = np.array([d for s, d in diffs if lo <= s < hi])
        if len(v) == 0:
            print(f"{lo:>4}-{hi:<4} {'-':>6}")
            continue
        line = (
            f"{lo:>4}-{hi:<4} {len(v):>6} {np.median(v):>7.3f} {np.percentile(v, 90):>6.2f} "
            + (f"{np.percentile(v, 99):>6.2f} " if extra_p99 else "")
            + f"{100 * (v < 0.5).mean():>6.1f} {100 * (v < 1).mean():>5.1f} "
            f"{100 * (v < 2).mean():>5.1f}"
        )
        print(line)
