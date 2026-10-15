#!/usr/bin/env python3
"""Build a synthetic field through the public config/CLI path; no downloads."""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for synthetic data and outputs")
    args = parser.parse_args()
    root = args.output.resolve()
    if args.output.is_symlink():
        raise ValueError("output must be a new directory, not a symlink")
    root.mkdir(exist_ok=False)
    shape = (5, 7, 9)
    frame = "synthetic-example/world"
    sources = {}
    for name in ("ct", "surface", "nx", "ny", "nz", "density"):
        value = 255 if name in ("ct", "surface") else 1 if name in ("nx", "density") else 0
        array = np.full(shape, value, np.float32)
        if name in ("ct", "surface"):
            array[:, :, 0] = 0
        np.save(root / (name + ".npy"), array)
        sources[name] = dict(path=name + ".npy", frame=frame, origin_xyz=[10, 20, 30], spacing_xyz=[1, 1, 1])
    with sqlite3.connect(root / "fibres.afv") as db:
        db.executescript("""CREATE TABLE fibers(id INTEGER PRIMARY KEY, annotation TEXT);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY, fiber_id INTEGER, first_segment INTEGER, points BLOB);
            CREATE TABLE block_bounds(id INTEGER, min_x REAL, max_x REAL, min_y REAL, max_y REAL,
                                      min_z REAL, max_z REAL);""")
        for fid, x in enumerate((4, 8, 12), 1):
            points = np.array([[x, 2, 4], [x, 10, 4]], dtype="<f8")
            db.execute("INSERT INTO fibers VALUES (?,?)", (fid, "{}"))
            db.execute("INSERT INTO blocks VALUES (?,?,?,?)", (fid, fid, 0, points.tobytes()))
            db.execute("INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)", (fid, x, x, 2, 10, 4, 4))
    config = dict(
        schema="winding-dataset-v1",
        frame=frame,
        voxel_um=1000,
        grid=dict(origin_xyz=[10, 20, 30], spacing_xyz=[1, 1, 1], shape_zyx=list(shape)),
        sheet_step=1,
        calibration="SYNTHETIC: analytic unit-spaced parallel sheets; no real CT or human labels",
        afv=dict(
            path="fibres.afv",
            native_frame="synthetic-example/native",
            native_voxel_um=500,
            scale_xyz=[0.5, 0.5, 0.5],
            offset_xyz=[10, 20, 30],
        ),
        sources=sources,
        prediction=dict(
            normal_encoding="signed_xyz",
            density_scale=1,
            surface_threshold=128,
            surface_scale=1 / 255,
            material_weight=1,
            background_weight=0.1,
        ),
        orientation=dict(mode="provided"),
        # Leave float32 roundoff headroom across CPU backends; check the full analytic field below.
        solver=dict(device="cpu", rtol=1e-5, maxiter=1000),
    )
    path = root / "dataset.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(BASE / "tools/run_coherence.py"),
            "global",
            "--config",
            str(path),
            "--output",
            str(root / "field"),
            "--stages",
            "B2",
        ],
        check=True,
    )
    field = np.load(root / "field/B2.npy", allow_pickle=False)
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(9), shape), atol=0.001)
    print("SYNTHETIC construction verified; dataset.json is the editable input contract.")


if __name__ == "__main__":
    main()
