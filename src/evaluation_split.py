"""Deterministic geometry-only groups in an explicitly declared frame.

This is a candidate split, not a claim of blind-test or training independence.
No winding labels or model scores are accepted. All coordinates are world XYZ.
"""

import hashlib
import json

import numpy as np

from dataset_config import frame as validate_frame


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build_fibre_split(records, calibration_ids, seed=20261012, reserved_fraction=0.2):
    """Transitive source families; length strata use family maximum, never scores."""
    if not 0 < reserved_fraction < 1 or type(seed) is not int:
        raise ValueError("invalid split fraction/seed")
    records = sorted(records, key=lambda r: r["fibre_id"])
    ids = [r["fibre_id"] for r in records]
    if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("duplicate/invalid fibre IDs")
    forced = set(calibration_ids)
    if forced - set(ids):
        raise ValueError("calibration fibre missing")
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for r in records:
        if not np.isfinite(r["length_mm"]) or r["length_mm"] < 0:
            raise ValueError("invalid fibre length")
        for key in r["source_lineage"]:
            if len(key) != 2 or any(not isinstance(v, str) or not v for v in key):
                raise ValueError("invalid source lineage")
            key = tuple(key)
            if key in owners:
                a, b = root(r["fibre_id"]), root(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = r["fibre_id"]
    members = {}
    for r in records:
        members.setdefault(root(r["fibre_id"]), []).append(r)
    groups = []
    bins = [0, 1.5, 3.5, 7.5, 15, 25, float("inf")]
    for family in members.values():
        length = max(r["length_mm"] for r in family)
        gids = [r["fibre_id"] for r in family]
        verified = all(bool(r["source_lineage"]) for r in family)
        group = dict(
            group_id=_digest(gids),
            fibre_ids=gids,
            length_mm=length,
            stratum=int(np.searchsorted(bins, length, side="right") - 1),
            forced_calibration=bool(forced & set(gids)),
            lineage_verified=verified,
            split="train",
        )
        groups.append(group)
    for k in range(len(bins) - 1):
        eligible = [g for g in groups if g["stratum"] == k and g["lineage_verified"] and not g["forced_calibration"]]
        eligible.sort(key=lambda g: _digest([seed, g["group_id"]]))
        count = int(np.floor(len(eligible) * reserved_fraction + 0.5))
        for g in eligible[:count]:
            g["split"] = "reserved"
    assignment = {i: g for g in groups for i in g["fibre_ids"]}
    fibres = [
        dict(
            fibre_id=r["fibre_id"],
            length_mm=float(r["length_mm"]),
            group_id=assignment[r["fibre_id"]]["group_id"],
            split=assignment[r["fibre_id"]]["split"],
            lineage_verified=assignment[r["fibre_id"]]["lineage_verified"],
        )
        for r in records
    ]
    strata = [
        dict(
            stratum=k,
            groups=sum(g["stratum"] == k for g in groups),
            reserved_groups=sum(g["stratum"] == k and g["split"] == "reserved" for g in groups),
            fibres=sum(assignment[r["fibre_id"]]["stratum"] == k for r in records),
            reserved_fibres=sum(
                assignment[r["fibre_id"]]["stratum"] == k and assignment[r["fibre_id"]]["split"] == "reserved"
                for r in records
            ),
        )
        for k in range(len(bins) - 1)
    ]
    result = dict(
        schema="lineage_length_split_v1",
        seed=seed,
        reserved_fraction=reserved_fraction,
        bins_mm=[0, 1.5, 3.5, 7.5, 15, 25, None],
        groups=sorted(groups, key=lambda g: g["group_id"]),
        fibres=fibres,
        strata=strata,
        limitations=[
            "Recorded lineage disjointness, not spatial or pretrained-model independence.",
            "Missing lineage and calibration antecedents are train-only; grouped quota is not exact fibre quota.",
        ],
    )
    result["manifest_sha256"] = _digest(result)
    return result


def validate_fibre_split(manifest, afv_sha256):
    if (
        manifest.get("afv_sha256") != afv_sha256
        or manifest.get("schema") != "lineage_length_split_v1"
        or manifest.get("manifest_sha256")
        != _digest({k: v for k, v in manifest.items() if k not in ("afv_sha256", "manifest_sha256")})
    ):
        raise ValueError("frozen fibre split identity or assignment changed")
    assignments = {r["fibre_id"]: (r["group_id"], r["split"]) for r in manifest["fibres"]}
    if len(assignments) != len(manifest["fibres"]):
        raise ValueError("duplicate fibre assignment")
    seen = set()
    for group in manifest["groups"]:
        if group["split"] not in ("train", "reserved") or (group["forced_calibration"] and group["split"] != "train"):
            raise ValueError("invalid or leaking calibration group")
        for fid in group["fibre_ids"]:
            if fid in seen or assignments.get(fid) != (group["group_id"], group["split"]):
                raise ValueError("group/assignment mismatch")
            seen.add(fid)
    if seen != set(assignments):
        raise ValueError("incomplete group assignment")


def build_split(
    colls,
    development_ids,
    buffer_voxels=256,
    min_z=None,
    *,
    frame,
    development_boxes=(),
    surface_ids=None,
    development_z_context=None,
):
    """Group buffered bounding-box overlaps and shared surface IDs transitively.

    colls: {collection_id: finite Nx3 XYZ array}, ONE declared world frame.
    surface_ids: optional {collection_id: [source_surface_id, ...]} lineage aliases.
    development_boxes: extra {id, min_xyz, max_xyz} inspected XYZ boxes.
    Each bbox is expanded by buffer_voxels, so a gap <=2*buffer joins groups.
    Any development member/box or min-Z violation makes the WHOLE group dev.
    Previously inspected Z context/boxes must be supplied explicitly; no dataset
    inherits another dataset's development exclusions or calibration history.
    Remaining group SHA-256 parity selects validation (even) or test (odd).
    """
    validate_frame(frame)
    if development_z_context is not None:
        context = np.asarray(development_z_context, float)
        if context.shape != (2,) or not np.isfinite(context).all() or context[0] >= context[1]:
            raise ValueError("invalid declared development Z context")
        development_z_context = context.tolist()
    if not np.isfinite(buffer_voxels) or buffer_voxels < 0:
        raise ValueError("buffer_voxels must be finite and nonnegative")
    if min_z is not None and not np.isfinite(min_z):
        raise ValueError("min_z must be finite")
    ids = sorted(colls)
    if any(not isinstance(cid, str) or not cid for cid in ids):
        raise ValueError("collection IDs must be nonempty strings")
    development_ids = set(development_ids)
    if development_ids - set(ids):
        raise ValueError("development IDs absent from collection geometry")
    surface_ids = surface_ids or {}
    if set(surface_ids) - set(ids):
        raise ValueError("surface lineage IDs absent from collection geometry")
    boxes = []
    for box in development_boxes:
        lo, hi = np.asarray(box["min_xyz"], float), np.asarray(box["max_xyz"], float)
        if (
            not isinstance(box.get("id"), str)
            or not box["id"]
            or lo.shape != (3,)
            or hi.shape != (3,)
            or not np.isfinite([lo, hi]).all()
            or (lo > hi).any()
        ):
            raise ValueError("invalid development bbox")
        boxes.append({"id": box["id"], "min_xyz": lo.tolist(), "max_xyz": hi.tolist()})
    boxes.sort(key=lambda box: (box["id"], box["min_xyz"], box["max_xyz"]))

    def overlaps(lo, hi, other_lo, other_hi):
        return bool(
            np.all(lo - buffer_voxels <= np.asarray(other_hi) + buffer_voxels)
            and np.all(np.asarray(other_lo) - buffer_voxels <= hi + buffer_voxels)
        )

    entries, bounds, reasons = {}, {}, {}
    for cid in ids:
        points = np.asarray(colls[cid], float)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"{cid}: expected nonempty finite Nx3 geometry")
        surfaces = surface_ids.get(cid, [])
        if not isinstance(surfaces, (list, tuple, set)) or any(not isinstance(s, str) or not s for s in surfaces):
            raise ValueError("surface_ids must contain lists of nonempty IDs")
        lo, hi = points.min(0), points.max(0)
        bounds[cid] = (lo, hi)
        entries[cid] = {
            "collection_id": cid,
            "min_xyz": lo.tolist(),
            "max_xyz": hi.tolist(),
            "points": len(points),
            "surface_ids": sorted(set(surfaces)),
        }
        reasons[cid] = []
        if cid in development_ids:
            reasons[cid].append("explicit_development_id")
        if development_z_context is not None and hi[2] >= development_z_context[0] and lo[2] < development_z_context[1]:
            reasons[cid].append("declared_development_z_context")
        if min_z is not None and lo[2] < min_z:
            reasons[cid].append("below_min_z")
        for box in boxes:
            if overlaps(lo, hi, box["min_xyz"], box["max_xyz"]) or box["id"] == cid or box["id"] in surfaces:
                reasons[cid].append("development_box_or_surface:" + box["id"])

    parent = {cid: cid for cid in ids}

    def root(cid):
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if overlaps(*bounds[a], *bounds[b]) or set(entries[a]["surface_ids"]) & set(entries[b]["surface_ids"]):
                ra, rb = root(a), root(b)
                parent[max(ra, rb)] = min(ra, rb)
    members = {}
    for cid in ids:
        members.setdefault(root(cid), []).append(cid)
    groups = []
    for group_ids in members.values():
        group_id = _digest(group_ids)
        dev = {cid: reasons[cid] for cid in group_ids if reasons[cid]}
        split = "development" if dev else ("validation" if int(group_id, 16) % 2 == 0 else "test")
        groups.append(
            {
                "group_id": group_id,
                "split": split,
                "collection_ids": group_ids,
                "collection_count": len(group_ids),
                "points": sum(entries[c]["points"] for c in group_ids),
                "min_xyz": np.min([bounds[c][0] for c in group_ids], axis=0).tolist(),
                "max_xyz": np.max([bounds[c][1] for c in group_ids], axis=0).tolist(),
                "development_reasons": dev,
            }
        )
        for cid in group_ids:
            entries[cid].update(group_id=group_id, split=split)
    groups.sort(key=lambda group: group["group_id"])
    manifest = {
        "schema_version": 1,
        "frame": frame,
        "status": "geometry_only_candidate_split",
        "buffer_voxels": float(buffer_voxels),
        "min_z": float(min_z) if min_z is not None else None,
        "development_z_context": development_z_context,
        "development_ids": sorted(development_ids),
        "development_boxes": boxes,
        "assignment": "SHA-256(sorted collection IDs): even validation, odd test",
        "collections": [entries[cid] for cid in ids],
        "groups": groups,
        "counts": {
            "collections": len(ids),
            "groups": len(groups),
            "groups_per_split": {
                s: sum(g["split"] == s for g in groups) for s in ("development", "validation", "test")
            },
        },
        "blind_test_complete": False,
        "limitations": [
            "Single-scroll geometry cannot establish cross-scroll generalization.",
            "Surface lineage completeness and pretrained-model overlap are not independently verified.",
            "Sparse point bboxes omit unprovided surfaces, solver halos and inference context; supply those as geometry.",
            "Hash allocation is deterministic, not label sealing or a guarantee of enough independent groups.",
        ],
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest
