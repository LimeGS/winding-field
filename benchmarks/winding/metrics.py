"""Deterministic fibre sampling and physical-coordinate field metrics."""

import hashlib
import heapq
import itertools
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import common as common
from common import BINS, arclength
from dataset_config import frame as validate_frame
from dataset_config import native_to_world
from patch_check import sample_field
from patch_check import sha256 as sha256

SEED = 20261012


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def lineage(annotation):
    collection = annotation.get("sourceCollectionUUID")
    return {(str(collection), str(fid)) for fid in annotation.get("sourceFiberIDs", [])} if collection else set()


def choose_pair(points_mm, inside, lo, hi, seed):
    """Uniform over eligible stored-vertex pairs; never bridge an outside run."""
    s = arclength(points_mm, 1.0)
    n = len(s)
    if n < 2 or s[-1] < lo:
        return None
    lower = np.maximum(np.searchsorted(s, s + lo, side="left"), np.arange(n) + 1)
    upper = np.searchsorted(s, s + hi, side="left")
    next_out = np.minimum.accumulate(np.where(~inside, np.arange(n), n)[::-1])[::-1]
    upper = np.minimum(upper, next_out)
    counts = np.where(inside, np.maximum(upper - lower, 0), 0)
    ends = np.cumsum(counts)
    if not ends[-1]:
        return None
    rank = int(np.random.default_rng(seed).integers(int(ends[-1])))
    i = int(np.searchsorted(ends, rank, side="right"))
    j = int(lower[i] + rank - (ends[i - 1] if i else 0))
    return i, j, float(s[j] - s[i])


def iter_fibres(db):
    connection = sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro", uri=True)
    try:
        query = """SELECT f.id, f.annotation, b.points FROM fibers f
                   JOIN blocks b ON b.fiber_id=f.id ORDER BY f.id,b.first_segment"""
        for fid, blocks in itertools.groupby(connection.execute(query), key=lambda r: r[0]):
            arrays = []
            for _, annotation, blob in blocks:
                arrays.append(np.frombuffer(blob, dtype="<f8").reshape(-1, 3))
            yield int(fid), np.concatenate(arrays), json.loads(annotation)
    finally:
        connection.close()


def select_pairs(fibres, scopes, blocked, blocked_lineage, limit=5000, *, native_mm):
    native_mm = np.broadcast_to(np.asarray(native_mm, float), (3,))
    if not np.isfinite(native_mm).all() or (native_mm <= 0).any():
        raise ValueError("native voxel sizes must be positive finite XYZ values")
    heaps = {name: [[] for _ in BINS] for name in scopes}
    counts = {name: [0 for _ in BINS] for name in scopes}
    excluded = Counter()
    seen = 0
    boxes = {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in scopes.items()}
    for fid, points, annotation in fibres:
        seen += 1
        if fid in blocked or lineage(annotation) & blocked_lineage:
            excluded["constraint_or_recorded_lineage"] += 1
            continue
        if len(points) < 2 or not np.isfinite(points).all():
            excluded["malformed_fibre"] += 1
            continue
        pmm = points * native_mm
        if arclength(pmm, 1.0)[-1] < BINS[0][0]:
            excluded["shorter_than_0_5_mm"] += 1
            continue
        for name, (a, b) in boxes.items():
            inside = ((points >= a) & (points <= b)).all(1)
            if inside.sum() < 2:
                continue
            for k, (lo, hi) in enumerate(BINS):
                digest = hashlib.sha256(f"{SEED}:{name}:{fid}:{k}".encode()).digest()
                priority = int.from_bytes(digest[:8], "big")
                pair = choose_pair(pmm, inside, lo, hi, int.from_bytes(digest[8:16], "big"))
                if pair is None:
                    continue
                counts[name][k] += 1
                heap = heaps[name][k]
                if limit is not None and len(heap) >= limit and priority >= -heap[0][0]:
                    continue
                i, j, span = pair
                row = dict(
                    fiber_id=fid,
                    bin=k,
                    i=i,
                    j=j,
                    span_mm=span,
                    chord_mm=float(np.linalg.norm(pmm[j] - pmm[i])),
                    a_native_xyz=points[i].tolist(),
                    b_native_xyz=points[j].tolist(),
                )
                entry = (-priority, fid, row)
                if limit is None:
                    heap.append(entry)
                elif len(heap) < limit:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heapreplace(heap, entry)
        if seen % 50000 == 0:
            print("SELECT", seen, counts, flush=True)
    result = {
        name: [entry[2] for heap in bins for entry in sorted(heap, key=lambda e: e[1])] for name, bins in heaps.items()
    }
    return result, dict(fibres_scanned=seen, excluded=dict(excluded), eligible_fibres_per_bin=counts)


def sample_checked(field, meta, xyz, support=None, margin=1):
    xyz = np.asarray(xyz, float)
    values = sample_field(field, xyz, meta["origin_xyz"], meta["spacing_xyz"], margin)
    if support is not None:
        if support.shape != field.shape:
            raise ValueError("support/field shape mismatch")
        indices = np.flatnonzero(np.isfinite(values))
        if len(indices):
            base = np.floor((xyz[indices] - meta["origin_xyz"]) / meta["spacing_xyz"]).astype(int)
            good = np.ones(len(indices), bool)
            for corner in itertools.product((0, 1), repeat=3):
                q = base + corner
                good &= support[q[:, 2], q[:, 1], q[:, 0]].astype(bool)
            values[indices[~good]] = np.nan
    return values


def open_field(path, meta):
    if sha256(path) != meta["field_sha256"]:
        raise ValueError("field hash mismatch")
    validate_frame(meta["frame"])
    if meta["solver"].get("converged") is not True:
        raise ValueError("unconverged field")
    if not np.isfinite(meta["step"]) or meta["step"] <= 0:
        raise ValueError("invalid sheet step")
    f = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(f.shape) != meta["shape_zyx"] or not np.isfinite(f).all():
        raise ValueError("invalid field shape/values")
    return f


def summarize(rows):
    result = []
    for k, (lo, hi) in enumerate(BINS):
        selected = [r for r in rows if r["bin"] == k]
        covered = [r for r in selected if r["error"] is not None]
        values = np.array([r["error"] for r in covered])
        n = len(values)
        r = dict(
            span_mm=[lo, hi],
            selected=len(selected),
            covered=n,
            fibers=len({r["fiber_id"] for r in covered}),
            coverage_pct=100 * n / len(selected) if selected else None,
            tail_sample_warning=n < 100,
        )
        r.update(
            {
                key: float(np.percentile(values, q)) if n else None
                for key, q in (("median", 50), ("p90", 90), ("p99", 99))
            }
        )
        r.update(
            {
                key: float(100 * np.mean(values < t)) if n else None
                for key, t in (("lt0_5_pct", 0.5), ("lt1_pct", 1), ("lt2_pct", 2))
            }
        )
        result.append(r)
    return result


def evaluate(field, meta, rows, support=None, ownership=None, *, transform):
    if not rows:
        return []
    a = native_to_world([r["a_native_xyz"] for r in rows], transform)
    b = native_to_world([r["b_native_xyz"] for r in rows], transform)
    va = sample_checked(field, meta, a, support)
    vb = sample_checked(field, meta, b, support)
    out = []
    for i, r in enumerate(rows):
        good = bool(np.isfinite(va[i]) and np.isfinite(vb[i]))
        seam = "not_applicable"
        if ownership is not None:
            ix = np.rint((np.array([a[i], b[i]]) - meta["origin_xyz"]) / meta["spacing_xyz"]).astype(int)
            seam = "unknown"
            if (ix >= 0).all() and (ix < np.array(ownership.shape[::-1])).all():
                owners = ownership[ix[:, 2], ix[:, 1], ix[:, 0]]
                if (owners >= 0).all():
                    seam = "cross" if owners[0] != owners[1] else "same"
        out.append(
            dict(
                r,
                phi_a=float(va[i]) if np.isfinite(va[i]) else None,
                phi_b=float(vb[i]) if np.isfinite(vb[i]) else None,
                error=float(abs(va[i] - vb[i]) / meta["step"]) if good else None,
                seam=seam,
                status="covered" if good else "unavailable_stencil_or_margin",
            )
        )
    return out
