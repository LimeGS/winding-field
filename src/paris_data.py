"""Configured observations sampled on an explicit physical grid.

The historical module name is retained; no dataset is selected implicitly.
"""

import json
from pathlib import Path

import numpy as np

from common import arclength, decode_lasagna_normals, load_fibers
from dataset_config import artifact_path, native_to_world, observation_recipe, open_source, validate_afv
from patch_check import sha256
from physical_grid import grid_shape, sample_volume


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


def sample_source(dataset, name, origin, spacing, shape, *, order=1, receipts=None):
    source = dataset["sources"][name]
    if source["frame"] != dataset["frame"]:
        raise ValueError("source and dataset frames differ")
    result = sample_volume(
        open_source(source),
        origin,
        spacing,
        shape,
        source["spacing_xyz"],
        source["origin_xyz"],
        order=order,
        receipts=receipts,
    )
    if receipts is not None:
        receipts[-1].update(url=source["path"], frame=source["frame"], source_origin_xyz=source["origin_xyz"])
    return result


def black_value(dataset):
    value = dataset.get("context", {}).get("black_value", 0)
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError("black_value must be a finite numeric CT value")
    return value


def exterior_faces(dataset):
    """Return only caller-declared exterior faces; a crop boundary is not evidence."""
    context = dataset.get("context", {})
    names = context.get("exterior_faces", [])
    known = {
        f"{axis}_{side}": (2 - i, 0 if side == "min" else -1) for i, axis in enumerate("xyz") for side in ("min", "max")
    }
    if (
        not isinstance(names, list)
        or any(not isinstance(n, str) or n not in known for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("invalid declared exterior faces")
    if names and (not isinstance(context.get("exterior_authority"), str) or not context["exterior_authority"].strip()):
        raise ValueError("exterior faces require explicit provenance in exterior_authority")
    return [known[name] for name in names]


def load_paris_box(
    lo_xyz, hi_xyz, spacing, afv_path=None, *, dataset, fiber_limit=None, pair_limit=None, shape_zyx=None
):
    afv_identity = validate_afv(dataset, afv_path) if afv_path is not None else None
    split_path = artifact_path(dataset, "split") if afv_path is not None else None
    train = None
    if split_path is not None:
        from evaluation_split import validate_fibre_split

        split = json.loads(split_path.read_text())
        validate_fibre_split(split, afv_identity["sha256"])
        train = {r["fibre_id"] for r in split["fibres"] if r["split"] == "train"}
    controls = dataset.get("constraints", {})
    fiber_limit = controls.get("fiber_limit", 400) if fiber_limit is None else fiber_limit
    pair_limit = controls.get("pair_limit", 2000) if pair_limit is None else pair_limit
    if type(fiber_limit) is not int or fiber_limit < 1 or type(pair_limit) is not int or pair_limit < 1:
        raise ValueError("positive integer fibre and pair limits required")

    lo, hi = np.asarray(lo_xyz, float), np.asarray(hi_xyz, float)
    h = np.broadcast_to(np.asarray(spacing, float), (3,)).copy()
    shape = grid_shape(lo, hi, h, shape_zyx)
    receipts = []

    def grab(name, order=1):
        return sample_source(dataset, name, lo, h, shape, order=order, receipts=receipts)

    prediction = dataset["prediction"]
    nx, ny = grab("nx"), grab("ny")
    density = grab("density") * prediction["density_scale"]
    if (density < 0).any():
        raise ValueError("density must be nonnegative")
    if prediction["normal_encoding"] == "hemisphere_xy":
        normals = decode_lasagna_normals(nx, ny)
    elif prediction["normal_encoding"] == "signed_xyz":
        normals = np.stack([nx, ny, grab("nz")], axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        if (norm <= 0).any():
            raise ValueError("signed normals require nonzero observations")
        normals /= norm
    else:
        raise ValueError("unsupported normal encoding")
    del nx, ny
    centerline_sha = None
    if dataset["orientation"]["mode"] == "radial":
        centerline = dataset["orientation"]["centerline"]
        centerline_sha = sha256(centerline["path"])
        if centerline.get("sha256", centerline_sha) != centerline_sha or centerline["frame"] != dataset["frame"]:
            raise ValueError("centerline identity or frame mismatch")
        cp = json.loads(Path(centerline["path"]).read_text())["control_points"]
        cp = np.array([[p["x"], p["y"], p["z"]] for p in cp], dtype=float)
        if cp.ndim != 2 or cp.shape[1] != 3 or len(cp) < 2 or not np.isfinite(cp).all():
            raise ValueError("centerline must contain at least two finite XYZ points")
        normals = orient_radial(normals, lo, h, cp)
    target = normals.astype(np.float32) * density[..., None]
    del normals, density
    det = grab("surface", order=0) > prediction["surface_threshold"]
    weight = np.where(det, prediction["material_weight"], prediction["background_weight"]).astype(np.float32)
    pairs = []
    used_fibres = set()
    if afv_path:
        transform = dataset["afv"]
        scale, offset = np.asarray(transform["scale_xyz"]), np.asarray(transform["offset_xyz"])
        bb = dict(
            zip(
                ("x0", "x1", "y0", "y1", "z0", "z1"),
                np.stack([(lo - offset) / scale, (hi - offset) / scale], 1).ravel().tolist(),
            )
        )
        fibers = load_fibers(str(afv_path), bb, clip=False)
        if train is not None:
            fibers = [(fid, p) for fid, p in fibers if fid in train]
        rng = np.random.default_rng(0)
        for fid, P in fibers[:fiber_limit]:
            for run in contiguous_inside(native_to_world(P, transform), lo, hi):
                s = arclength(run, dataset["voxel_um"] / 1000)
                for _ in range(8):
                    i, j = rng.integers(0, len(run), 2)
                    indices = np.rint((run[[i, j]] - lo) / h).astype(int)[:, ::-1]
                    if (indices < 0).any() or (indices >= np.asarray(shape)).any() or abs(s[j] - s[i]) < 0.3:
                        continue
                    a, b = np.ravel_multi_index(indices.T, shape)
                    if a != b and len(pairs) < pair_limit:
                        pairs.append((int(a), int(b)))
                        used_fibres.add(int(fid))
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    meta = dict(
        frame=dataset["frame"],
        voxel_um=dataset["voxel_um"],
        origin_xyz=lo.tolist(),
        spacing_xyz=h.tolist(),
        shape_zyx=list(shape),
        source_receipts=receipts,
        normal_encoding=prediction["normal_encoding"],
        prediction_parameters=dict(prediction),
        observation_config=observation_recipe(dataset),
        normal_orientation=dataset["orientation"]["mode"],
        density_units="raw_field_per_declared_world_unit",
        source_sha256=sha256(__file__),
        umbi_sha256=centerline_sha,
        dataset_config_sha256=dataset["config_sha256"],
        native_to_world={k: dataset["afv"][k] for k in ("scale_xyz", "offset_xyz")},
        afv_sha256=afv_identity["sha256"] if afv_identity is not None else None,
        afv_identity=afv_identity,
        constraint_pairs=len(pairs),
        constraint_fibre_ids=sorted(used_fibres),
        fibre_split_sha256=sha256(split_path) if split_path is not None else None,
        fibre_limit=fiber_limit,
        pair_limit=pair_limit,
    )
    return target, weight, pairs, meta


def save_field(path, field, metadata, info, step=1):
    """Publish without invalidating a solver's declared float32/float64 precision.

    Legacy receipts without dtype retain float32 output only for lossless casts;
    a lossy export requires a newly verified receipt, never an inherited PASS.
    """
    path = Path(path)
    if path.exists() or Path(str(path) + ".json").exists():
        raise FileExistsError(path)
    if not info["converged"] or not np.isfinite(field).all() or not np.isfinite(step) or step <= 0:
        raise ValueError("cannot publish unconverged/nonfinite/uncalibrated field")
    field = np.asarray(field)
    dtype = info.get("dtype", "float32")
    if dtype not in ("float32", "float64") or field.dtype.kind not in "biuf":
        raise ValueError("cannot publish unsupported field precision")
    if "dtype" in info and field.dtype != np.dtype(dtype):
        raise ValueError("field dtype differs from solver receipt precision")
    with np.errstate(over="ignore", invalid="ignore"):
        stored = field.astype(dtype, copy=False)
        lossless = np.isfinite(stored).all() and np.array_equal(stored.astype(field.dtype), field)
    if not lossless:
        raise ValueError("lossy precision export cannot inherit solver convergence")
    metadata = {**metadata, "step": float(step), "solver": info}
    json.dumps(metadata, allow_nan=False)  # reject bad receipts before publishing array bytes
    np.save(path, stored)
    metadata["field_sha256"] = sha256(path)
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata
