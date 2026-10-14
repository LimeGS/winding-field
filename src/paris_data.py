"""Measured Paris4 inputs sampled on one explicit physical L2 grid."""

import json
from pathlib import Path

import numpy as np

from common import arclength, decode_lasagna_normals, load_fibers
from paris_solve import CTB as CTB  # re-exported for callers
from paris_solve import LB, MB
from patch_check import sha256
from physical_grid import grid_shape, sample_volume

FRAME = "PHercParis4/20260411134726:L2"


def orient_radial(normals, origin_xyz, spacing_xyz, umbilicus):
    cp = umbilicus[np.argsort(umbilicus[:, 2])]
    z = origin_xyz[2] + np.arange(normals.shape[0]) * spacing_xyz[2]
    cx = np.interp(z, cp[:, 2], cp[:, 0])[:, None, None]
    cy = np.interp(z, cp[:, 2], cp[:, 1])[:, None, None]
    x = origin_xyz[0] + np.arange(normals.shape[2]) * spacing_xyz[0]
    y = origin_xyz[1] + np.arange(normals.shape[1]) * spacing_xyz[1]
    dot = normals[..., 0] * (x[None, None, :] - cx) + normals[..., 1] * (y[None, :, None] - cy)
    return normals * np.where(dot < 0, -1, 1)[..., None]


def contiguous_inside(points, lo, hi):
    keep = np.isfinite(points).all(1) & (points >= lo).all(1) & (points < hi).all(1)
    borders = np.diff(np.r_[False, keep, False].astype(np.int8))
    return [points[a:b] for a, b in zip(np.flatnonzero(borders == 1), np.flatnonzero(borders == -1)) if b - a >= 2]


def load_paris_box(
    lo_xyz, hi_xyz, spacing=2, afv_path=None, data_root="/root/wrev", fiber_limit=400, pair_limit=2000, normal_level=4
):
    import zarr

    lo, hi = np.asarray(lo_xyz, float), np.asarray(hi_xyz, float)
    h = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()
    shape = grid_shape(lo, hi, h)
    receipts = []

    def grab(url, source_h, order=1):
        print("READ", url, shape, flush=True)
        volume = zarr.open(url, mode="r")
        out = sample_volume(volume, lo, h, shape, [source_h] * 3, order=order, receipts=receipts)
        receipts[-1]["url"] = url
        return out

    source_h = 2**normal_level / 4
    nx = grab(f"{LB}_nx.ome.zarr/{normal_level}", source_h)
    ny = grab(f"{LB}_ny.ome.zarr/{normal_level}", source_h)
    density = grab(f"{LB}_grad_mag.ome.zarr/{normal_level}", source_h) / 1000
    normals = decode_lasagna_normals(nx, ny)
    del nx, ny
    root = Path(data_root)
    cp = json.loads((root / "umbi.json").read_text())["control_points"]
    cp = np.array([[p["x"], p["y"], p["z"]] for p in cp])
    target = orient_radial(normals, lo, h, cp).astype(np.float32) * density[..., None]
    del normals, density
    # Sample at physical positions; use a coarse source only for coarse global weights.
    ml = max(0, int(np.floor(np.log2(h.min()))) - 1)
    det = grab(f"{MB}/{ml}", 2**ml, order=0) > 128
    weight = np.where(det, 1, 0.1).astype(np.float32)
    pairs = []
    if afv_path:
        bb = dict(zip(("x0", "x1", "y0", "y1", "z0", "z1"), np.stack([lo * 4, hi * 4], 1).ravel().tolist()))
        fibers = load_fibers(str(afv_path), bb, clip=False)
        rng = np.random.default_rng(0)
        for fid, P in fibers[:fiber_limit]:
            for run in contiguous_inside(P / 4, lo, hi):
                s = arclength(run, 0.0096)
                for _ in range(8):
                    i, j = rng.integers(0, len(run), 2)
                    indices = np.rint((run[[i, j]] - lo) / h).astype(int)[:, ::-1]
                    if (indices < 0).any() or (indices >= np.asarray(shape)).any() or abs(s[j] - s[i]) < 0.3:
                        continue
                    a, b = np.ravel_multi_index(indices.T, shape)
                    if a != b and len(pairs) < pair_limit:
                        pairs.append((int(a), int(b)))
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    meta = dict(
        frame=FRAME,
        voxel_um=9.6,
        origin_xyz=lo.tolist(),
        spacing_xyz=h.tolist(),
        shape_zyx=list(shape),
        source_receipts=receipts,
        normal_encoding="hemisphere_nx_ny",
        normal_orientation="radial_prior_not_ground_truth",
        density_units="raw_field_per_L2_voxel",
        source_sha256=sha256(__file__),
        umbi_sha256=sha256(root / "umbi.json"),
        afv_sha256=sha256(afv_path) if afv_path else None,
        constraint_pairs=len(pairs),
    )
    return target, weight, pairs, meta


def save_field(path, field, metadata, info, step=1):
    path = Path(path)
    if path.exists() or Path(str(path) + ".json").exists():
        raise FileExistsError(path)
    if not info["converged"] or not np.isfinite(field).all() or not np.isfinite(step) or step <= 0:
        raise ValueError("cannot publish unconverged/nonfinite/uncalibrated field")
    metadata = {**metadata, "step": float(step), "solver": info}
    json.dumps(metadata, allow_nan=False)  # reject bad receipts before publishing array bytes
    np.save(path, field.astype(np.float32))
    metadata["field_sha256"] = sha256(path)
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata
