#!/usr/bin/env python3
"""Generate and evaluate a tiny analytic case, entirely offline and without a solver."""

import argparse
import gzip
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks/winding"))
import evaluate
import metrics

from evaluation_split import build_fibre_split

FRAME = "synthetic-example/world"


def build_case(folder):
    shape, step = (32, 32, 128), 2.0
    z = np.arange(shape[0], dtype=np.float32)[:, None, None]
    field = np.broadcast_to(z / 4, shape).copy()  # W = phi/2 = z/8: one layer every 8 voxels.
    field_path = folder / "field.npy"
    np.save(field_path, field)
    metadata = dict(
        frame=FRAME,
        voxel_um=9.6,
        origin_xyz=[0, 0, 0],
        spacing_xyz=[1, 1, 1],
        shape_zyx=list(shape),
        step=step,
        field_sha256=metrics.sha256(field_path),
        synthetic=True,
        producer="analytic parallel layers",
    )
    metrics.write_json(folder / "field.npy.json", metadata)

    afv = folder / "fibres.afv"
    split_records = []
    with closing(sqlite3.connect(afv)) as db:
        db.executescript("""
            CREATE TABLE fibers(id INTEGER PRIMARY KEY, annotation TEXT);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY, fiber_id INTEGER, first_segment INTEGER, points BLOB);
            CREATE TABLE block_bounds(id INTEGER, min_x REAL, max_x REAL, min_y REAL, max_y REAL,
                                      min_z REAL, max_z REAL);
        """)
        for fid, (layer, y) in enumerate(((layer, y) for layer in (8, 16, 24) for y in (8, 24)), start=1):
            points = np.array([[8, y, layer], [108, y, layer]], dtype="<f8") * 4
            annotation = dict(sourceCollectionUUID="synthetic", sourceFiberIDs=[fid])
            db.execute("INSERT INTO fibers VALUES (?, ?)", (fid, json.dumps(annotation)))
            db.execute("INSERT INTO blocks VALUES (?, ?, ?, ?)", (fid, fid, 0, points.tobytes()))
            bounds = np.stack([points.min(0), points.max(0)], axis=1).ravel().tolist()
            db.execute("INSERT INTO block_bounds VALUES (?, ?, ?, ?, ?, ?, ?)", (fid, *bounds))
            split_records.append(dict(fibre_id=fid, length_mm=0.96, source_lineage=[["synthetic", str(fid)]]))
        db.commit()
    split = build_fibre_split(split_records, set(), reserved_fraction=1 / 3)
    split["afv_sha256"] = metrics.sha256(afv)
    metrics.write_json(folder / "split.json", split)

    maximum = ((np.array(shape[::-1]) - 1) * 4).tolist()
    selection, _ = metrics.select_pairs(
        metrics.iter_fibres(afv), {"synthetic_layers": ([0, 0, 0], maximum)}, set(), set(), limit=None, native_mm=0.0024
    )
    selected_path = folder / "selected-pairs.json.gz"
    with gzip.open(selected_path, "xt") as stream:
        json.dump(selection, stream, allow_nan=False)
    metrics.write_json(folder / "audit.json", dict(afv_sha256=metrics.sha256(afv), synthetic=True))
    metrics.write_json(
        folder / "plan.json",
        dict(
            frame=FRAME,
            fixed_sheet_step=step,
            native_voxel_mm=0.0024,
            bins=metrics.BINS,
            population="all",
            synthetic=True,
            constraint_audit_sha256=metrics.sha256(folder / "audit.json"),
        ),
    )
    metrics.write_json(
        folder / "seal.json",
        dict(selected_sha256=metrics.sha256(selected_path), plan_sha256=metrics.sha256(folder / "plan.json")),
    )
    controls = [
        dict(
            pair_id=f"synthetic-{a}-{b}",
            collection_id="synthetic",
            point_a_id=str(a),
            point_b_id=str(b),
            a_xyz=[64, 16, a],
            b_xyz=[64, 16, b],
            wind_a=a // 8,
            wind_b=b // 8,
            dw=(b - a) // 8,
        )
        for a, b in ((8, 16), (16, 24), (8, 24))
    ]
    metrics.write_json(
        folder / "controls.json", dict(methods={"synthetic_reference": dict(rows=controls)}, synthetic=True)
    )

    files = {
        "selection": selected_path,
        "plan": folder / "plan.json",
        "seal": folder / "seal.json",
        "audit": folder / "audit.json",
        "afv": afv,
        "split": folder / "split.json",
        "human": folder / "controls.json",
    }
    case = dict(
        schema="winding-benchmark-case-v2",
        scope="synthetic_layers",
        frame=FRAME,
        voxel_um=9.6,
        native_to_world=dict(scale_xyz=[0.25] * 3, offset_xyz=[0, 0, 0]),
        population="all_and_reserved",
        step=step,
        expected_selected=6,
        expected_human=3,
        human_method="synthetic_reference",
        requires_support=False,
        synthetic=True,
        artifacts={name: dict(path=path.name, sha256=metrics.sha256(path)) for name, path in files.items()},
    )
    metrics.write_json(folder / "case.json", case)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for synthetic inputs and results")
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output already exists; choose a new directory")
    folder = args.output / "inputs"
    folder.mkdir()
    build_case(folder)
    status = evaluate.main(
        [
            "--case",
            str(folder / "case.json"),
            "--field",
            str(folder / "field.npy"),
            "--metadata",
            str(folder / "field.npy.json"),
            "--output",
            str(args.output / "evaluation"),
        ]
    )
    if status:
        return status
    summary = json.loads((args.output / "evaluation/SUMMARY.json").read_text())
    candidate = summary["methods"]["candidate"]
    if (
        candidate["all"][0]["covered"] != 6
        or candidate["all"][0]["median"] != 0
        or candidate["human"]["correct"] != 3
        or any(summary["methods"][name]["human"]["correct"] != 0 for name in ("constant_control", "compressed_control"))
    ):
        raise RuntimeError("synthetic analytic checks failed")
    metrics.write_json(
        args.output / "DEMO.json",
        dict(
            status="PASS",
            synthetic=True,
            selected=6,
            covered=6,
            separation_correct=3,
            constant_control_correct=0,
            compressed_control_correct=0,
            note="Analytic demonstration only: no real scan, annotation or model data.",
        ),
    )
    print("Synthetic demo: 6/6 fibre pairs covered; separation 3/3; collapse controls 0/3.")
    print("Scores:", args.output / "evaluation/TABLES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
