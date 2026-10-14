"""Tiny synthetic observations; only the external Zarr transport is replaced."""

import json
import sqlite3
import sys
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def paris_inputs(tmp_path, monkeypatch):
    from paris_solve import CTB, LB, MB

    afv = tmp_path / "Paris4_v1.afv"
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
    return SimpleNamespace(root=tmp_path, afv=afv, points=points, volumes=volumes, development=development)
