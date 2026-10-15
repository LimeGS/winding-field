"""Tiny synthetic observations; only the external Zarr transport is replaced."""

import json
import sqlite3
import sys
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

FRAME = "PHercParis4/20260411134726:L2"  # Explicit historical-parameter fixture, never a runtime default.
CTB = "memory://ct"
LB = "memory://normal"
MB = "memory://surface"


@pytest.fixture
def paris_inputs(tmp_path, monkeypatch):
    afv = tmp_path / "fibres.afv"
    points = np.column_stack([np.arange(0, 65, 16), np.full(5, 16), np.full(5, 16)]) * 4
    with closing(sqlite3.connect(afv)) as db:
        db.executescript("""
            CREATE TABLE fibers(id INTEGER PRIMARY KEY, annotation TEXT);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY, fiber_id INTEGER, first_segment INTEGER, points BLOB);
            CREATE TABLE block_bounds(id INTEGER, min_x REAL, max_x REAL, min_y REAL, max_y REAL,
                                      min_z REAL, max_z REAL);
        """)
        for fid, offset in [(1, 0), (2, 64)]:
            db.execute(
                "INSERT INTO fibers VALUES (?, ?)",
                (fid, json.dumps({"sourceCollectionUUID": "A", "sourceFiberIDs": [fid]})),
            )
            line = points + [0, offset, 0]
            # Insert later segments first to catch readers depending on insertion order.
            for first, chunk in [(3, line[3:]), (0, line[:3])]:
                row = db.execute(
                    "INSERT INTO blocks(fiber_id, first_segment, points) VALUES (?, ?, ?)",
                    (fid, first, np.asarray(chunk, dtype="<f8").tobytes()),
                )
                lo, hi = chunk.min(0), chunk.max(0)
                db.execute(
                    "INSERT INTO block_bounds VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (row.lastrowid, *np.stack([lo, hi], axis=1).ravel().tolist()),
                )
        db.commit()
    (tmp_path / "umbi.json").write_text(
        json.dumps({"control_points": [{"x": -100, "y": -100, "z": 0}, {"x": -100, "y": -100, "z": 100}]})
    )
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps({"rows": [{"collection_id": "dev", "covered": True, "phi_a": 0, "phi_b": 2, "dw": 1}]})
    )
    volumes = {
        CTB + "/0": np.broadcast_to(np.uint8(0), (192, 256, 320)),
        CTB + "/5": np.zeros((6, 8, 10), np.uint8),
    }
    for level in (4, 5):
        shape = tuple(n // 2**level for n in (192, 256, 320))
        volumes[f"{LB}_nx.ome.zarr/{level}"] = np.full(shape, 255, np.uint8)
        volumes[f"{LB}_ny.ome.zarr/{level}"] = np.full(shape, 128, np.uint8)
        volumes[f"{LB}_grad_mag.ome.zarr/{level}"] = (100 + 20 * np.indices(shape)[2]).astype(np.float32)
    for level in range(4):
        shape = tuple(n // 2**level for n in (48, 64, 80))
        volumes[f"{MB}/{level}"] = np.full(shape, 255, np.uint8)

    def open_volume(url, mode):
        if mode != "r":
            raise ValueError("fixture observations are read-only")
        return volumes[url]

    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(open=open_volume))

    def source(path, spacing):
        return dict(path=path, frame=FRAME, origin_xyz=[0, 0, 0], spacing_xyz=[spacing] * 3)

    data = dict(
        schema="winding-dataset-v1",
        frame=FRAME,
        voxel_um=9.6,
        grid=dict(origin_xyz=[0, 0, 0], spacing_xyz=[16] * 3, shape_zyx=[3, 4, 5]),
        sheet_step=2,
        calibration="synthetic development fixture",
        afv=dict(
            path=afv.name,
            native_frame="synthetic-regression/native",
            native_voxel_um=2.4,
            scale_xyz=[0.25] * 3,
            offset_xyz=[0, 0, 0],
        ),
        sources=dict(
            ct=source(CTB + "/5", 8),
            surface=source(MB + "/0", 1),
            nx=source(f"{LB}_nx.ome.zarr/4", 4),
            ny=source(f"{LB}_ny.ome.zarr/4", 4),
            density=source(f"{LB}_grad_mag.ome.zarr/4", 4),
        ),
        prediction=dict(
            normal_encoding="hemisphere_xy",
            density_scale=0.001,
            surface_threshold=128,
            surface_scale=1 / 255,
            material_weight=1,
            background_weight=0.1,
        ),
        orientation=dict(mode="radial", centerline=dict(path="umbi.json", frame=FRAME)),
        context=dict(
            exterior_faces=[f"{axis}_{side}" for axis in "xyz" for side in ("min", "max")],
            exterior_authority="controlled synthetic complete-volume exterior",
        ),
    )
    config = tmp_path / "dataset.json"
    config.write_text(json.dumps(data))
    from dataset_config import load_dataset

    return SimpleNamespace(
        root=tmp_path,
        afv=afv,
        points=points,
        volumes=volumes,
        development=development,
        config=config,
        dataset=load_dataset(config),
        raw_config=data,
    )
