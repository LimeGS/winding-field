"""Offline Paris4 pair receipts; no fitting, GPU, network, or coordinate clamping.

Usage: python paris_pair_check.py --field phi.npy --metadata phi.npy.json \
    --relwind relwind.json --output receipt.json [--csv pairs.csv]
Instead of --relwind, --pairs-json accepts an identifiable record list or
{"frame": FRAME, "records": [...]}. Outputs must not already exist.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from patch_check import sample_field, sha256

FRAME = "PHercParis4/20260411134726:L2"
IDENTITY = ("pair_id", "collection_id", "point_a_id", "point_b_id")


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and not isinstance(value, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _xyz(value):
    try:
        result = np.asarray(value, dtype=float)
        if result.shape == (3,):
            return result
    except (TypeError, ValueError):
        pass
    return np.full(3, np.nan)


def load_pair_records(path, zlo=10000, zhi=11000):
    """Return (records, collection XYZ arrays, exclusions) from relative_windings.

    The legacy selection is unchanged for valid annotations: collections with
    >=2 points overlapping [zlo-200, zhi+200), endpoint Z in [zlo,zhi), 1<=dw<=6.
    IDs use SHA-256 of the collection and sorted endpoint IDs, not coordinates.
    Invalid labels are point exclusions; they never become winding zero.
    """
    if _number(zlo) is None or _number(zhi) is None or zlo >= zhi:
        raise ValueError("expected finite zlo < zhi")
    data = json.loads(Path(path).read_text())
    records, colls, exclusions = [], {}, []
    for cid, collection in sorted(data["collections"].items()):
        points = []
        for pid, point in sorted(collection["points"].items()):
            xyz, wind = _xyz(point.get("p")), _number(point.get("wind_a"))
            if not np.isfinite(xyz).all():
                exclusions.append({"collection_id": cid, "point_id": pid, "reason": "invalid_coordinate"})
                continue
            points.append((pid, xyz, wind))
            if wind is None:
                reason = "missing_wind_label" if "wind_a" not in point else "nonfinite_wind_label"
                exclusions.append({"collection_id": cid, "point_id": pid, "reason": reason})
        if len(points) < 2:
            continue
        P = np.array([p for _, p, _ in points])
        if not ((P[:, 2] >= zlo - 200).any() and (P[:, 2] < zhi + 200).any()):
            continue
        colls[cid] = P
        labelled = [p for p in points if p[2] is not None]
        for (aid, a, wa), (bid, b, wb) in itertools.combinations(labelled, 2):
            identity = json.dumps([cid, aid, bid], ensure_ascii=True, separators=(",", ":"))
            pair_id = hashlib.sha256(identity.encode()).hexdigest()
            dw = abs(wa - wb)
            reason = (
                "outside_z_range"
                if not (zlo <= a[2] < zhi and zlo <= b[2] < zhi)
                else "winding_difference_outside_1_6"
                if not 1 <= dw <= 6
                else None
            )
            if reason:
                exclusions.append(
                    {"pair_id": pair_id, "collection_id": cid, "point_a_id": aid, "point_b_id": bid, "reason": reason}
                )
                continue
            records.append(
                {
                    "pair_id": pair_id,
                    "collection_id": cid,
                    "point_a_id": aid,
                    "point_b_id": bid,
                    "a_xyz": a.tolist(),
                    "b_xyz": b.tolist(),
                    "wind_a": wa,
                    "wind_b": wb,
                    "dw": dw,
                }
            )
    return records, colls, exclusions


def score_pairs(field, metadata, records, margin=0):
    """Return JSON-safe rows and summaries; unavailable exact/errors remain None.

    Calibration/metadata errors abort the evaluation. Per-pair missing labels,
    coordinates or finite samples are retained as unavailable, never clipped.
    """
    field = np.asarray(field)
    step = _number(metadata.get("step"))
    if step is None or step <= 0:
        raise ValueError("calibration step must be positive and finite")
    if metadata.get("frame") != FRAME:
        raise ValueError(f"field frame must be {FRAME}")
    if (
        field.ndim != 3
        or not all(field.shape)
        or field.dtype.kind not in "fiu"
        or list(field.shape) != metadata.get("shape_zyx")
    ):
        raise ValueError("field shape/dtype does not match shape_zyx")
    if _number(margin) is None or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    # Integer ndimage inputs otherwise round interpolated values to integers.
    if field.dtype.kind in "iu":
        field = field.astype(float)
    origin, spacing = _xyz(metadata.get("origin_xyz")), _xyz(metadata.get("spacing_xyz"))
    sample_field(field, np.empty((0, 3)), origin, spacing, margin)
    rows, seen = [], set()
    for record in records:
        if any(not isinstance(record.get(key), str) or not record[key] for key in IDENTITY):
            raise ValueError("every pair needs nonempty string identity fields")
        if record["pair_id"] in seen:
            raise ValueError(f"duplicate pair_id: {record['pair_id']}")
        seen.add(record["pair_id"])
        if record.get("frame", FRAME) != FRAME:
            raise ValueError("pair/field frames differ")
        xyz = np.array([_xyz(record.get("a_xyz")), _xyz(record.get("b_xyz"))])
        values = sample_field(field, xyz, origin, spacing, margin)
        coverage = {}
        for i, endpoint in enumerate(("a", "b")):
            grid = (xyz[i] - origin) / spacing
            if not np.isfinite(xyz[i]).all() or (xyz[i] < 0).any():
                status = "invalid_coordinate"
            elif (grid < 0).any() or (grid > np.array(field.shape[::-1]) - 1).any():
                status = "outside_domain"
            elif (grid < margin).any() or (grid > np.array(field.shape[::-1]) - 1 - margin).any():
                status = "inside_excluded_margin"
            else:
                status = "covered" if np.isfinite(values[i]) else "nonfinite_sample"
            coverage[endpoint] = status
        wa, wb, dw = [_number(record.get(key)) for key in ("wind_a", "wind_b", "dw")]
        row = {key: record[key] for key in IDENTITY}
        if "region_cluster_id" in record:
            row["region_cluster_id"] = record["region_cluster_id"]
        row.update(
            a_xyz=[_number(v) for v in xyz[0]],
            b_xyz=[_number(v) for v in xyz[1]],
            wind_a=wa,
            wind_b=wb,
            dw=dw,
            frame=FRAME,
            coverage=coverage,
            covered=all(v == "covered" for v in coverage.values()),
            status="unavailable",
            phi_a=_number(values[0]),
            phi_b=_number(values[1]),
            step=step,
            prediction_continuous=None,
            prediction_rounded=None,
            error_integer=None,
            error_continuous=None,
            exact=None,
            rounding_margin=None,
        )
        if (
            wa is None
            or wb is None
            or dw is None
            or dw < 0
            or not math.isclose(dw, abs(wb - wa), rel_tol=0, abs_tol=1e-9)
        ):
            row["status"] = "invalid_annotation"
        elif row["covered"]:
            continuous = abs(float(values[1]) - float(values[0])) / step
            if not math.isfinite(continuous):
                row["status"] = "nonfinite_prediction"
            else:
                predicted = round(continuous)
                row.update(
                    prediction_continuous=continuous,
                    prediction_rounded=predicted,
                    error_integer=predicted - dw,
                    error_continuous=continuous - dw,
                    exact=predicted == dw,
                    rounding_margin=abs(continuous - (math.floor(continuous) + 0.5)),
                    status="scored_exact"
                    if predicted == dw
                    else "scored_undercount"
                    if predicted < dw
                    else "scored_overcount",
                )
        rows.append(row)
    return rows, summarize_rows(rows)


def summarize_rows(rows):
    """All-eligible micro and per-collection summaries; no independent-pair CI."""
    rows = list(rows)
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pair_id in receipts")

    def reduce(items):
        scored = [r for r in items if r["exact"] is not None]
        correct = sum(r["exact"] is True for r in scored)
        errors = np.array([r["error_continuous"] for r in scored], dtype=float)
        if not np.isfinite(errors).all():
            raise ValueError("scored receipts require finite continuous errors")
        absolute = np.abs(errors)
        stats = {key: None for key in ("mean_signed", "mae", "abs_p50", "abs_p90", "abs_p95", "max_abs")}
        if len(errors):
            stats.update(
                mean_signed=float(errors.mean()),
                mae=float(absolute.mean()),
                max_abs=float(absolute.max()),
                **{f"abs_p{q}": float(np.percentile(absolute, q)) for q in (50, 90, 95)},
            )
        return {
            "eligible": len(items),
            "covered": sum(r["covered"] for r in items),
            "scored": len(scored),
            "correct": correct,
            "incorrect": len(scored) - correct,
            "unavailable": len(items) - len(scored),
            "micro_accuracy_all_eligible": correct / len(items) if items else None,
            "micro_accuracy_scored": correct / len(scored) if scored else None,
            "coverage_fraction": sum(r["covered"] for r in items) / len(items) if items else None,
            "continuous_errors": stats,
            "statuses": dict(sorted(Counter(r["status"] for r in items).items())),
        }

    result = reduce(rows)
    collections = defaultdict(list)
    for row in rows:
        collections[row["collection_id"]].append(row)
    result["per_collection"] = {key: reduce(values) for key, values in sorted(collections.items())}
    macro = [c["micro_accuracy_all_eligible"] for c in result["per_collection"].values()]
    result["collection_macro_accuracy_all_eligible"] = float(np.mean(macro)) if macro else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-sha256", help="expected SHA-256 of the metadata file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--relwind", type=Path)
    source.add_argument("--pairs-json", type=Path)
    parser.add_argument("--zlo", type=float, default=10000)
    parser.add_argument("--zhi", type=float, default=11000)
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    outputs = [args.output] + ([args.csv] if args.csv else [])
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("JSON and CSV outputs must differ")
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
    metadata_hash = sha256(args.metadata)
    if args.metadata_sha256 and args.metadata_sha256 != metadata_hash:
        raise ValueError("metadata file SHA-256 mismatch")
    meta = json.loads(args.metadata.read_text())
    field_hash = sha256(args.field)
    if meta.get("field_sha256") is not None and meta["field_sha256"] != field_hash:
        raise ValueError("field SHA-256 differs from metadata")
    if args.relwind:
        records, _, exclusions = load_pair_records(args.relwind, args.zlo, args.zhi)
    else:
        data = json.loads(args.pairs_json.read_text())
        if isinstance(data, dict):
            if data.get("frame") != FRAME:
                raise ValueError("pair manifest frame differs")
            records = data["records"]
        else:
            records = data
        exclusions = []
    rows, summary = score_pairs(np.load(args.field, mmap_mode="r", allow_pickle=False), meta, records, args.margin)
    receipt = {
        "schema_version": 1,
        "frame": FRAME,
        "field": str(args.field),
        "field_sha256": field_hash,
        "metadata": meta,
        "metadata_sha256": metadata_hash,
        "pairs_source": str(args.relwind or args.pairs_json),
        "pairs_source_sha256": sha256(args.relwind or args.pairs_json),
        "evaluator_sha256": sha256(__file__),
        "sampler_sha256": sha256(Path(__file__).with_name("patch_check.py")),
        "selection": {"zlo": args.zlo, "zhi": args.zhi, "dw": [1, 6]} if args.relwind else "explicit_records",
        "margin_grid_voxels": args.margin,
        "rounding": "Python round: ties to even",
        "exclusions": exclusions,
        "rows": rows,
        "summary": summary,
    }
    text = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(text)
    if args.csv:
        with args.csv.open("x", newline="") as handle:
            keys = list(dict.fromkeys(key for row in rows for key in row)) or list(IDENTITY)
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()
                    }
                )
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
