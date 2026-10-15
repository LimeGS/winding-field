"""Explicit dataset geometry and read-only sources; no scroll or host defaults."""

import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from patch_check import sha256


def xyz(value, name, positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError(f"{name} must contain three finite {'positive ' if positive else ''}XYZ values")
    return result


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def frame(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("an explicit nonempty coordinate frame is required")
    return value


def native_to_world(points, transform):
    points = np.asarray(points, dtype=float)
    scale = xyz(transform["scale_xyz"], "native scale", positive=True)
    offset = xyz(transform["offset_xyz"], "native offset")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("native coordinates must be finite Nx3 XYZ points")
    return points * scale + offset


def _path(entry, base):
    value = entry["path"]
    if not isinstance(value, str) or not value:
        raise ValueError("source path must be a nonempty string")
    if not urlsplit(value).scheme:
        entry["path"] = str((base / value).resolve())
    if "sha256" in entry:
        digest = entry["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source sha256 must be a lowercase hexadecimal digest")


def _keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"unknown or invalid {name} fields")


def load_dataset(path):
    """Validate without loading volumes; relative paths belong to the JSON file."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    try:
        allowed = {
            "schema",
            "frame",
            "voxel_um",
            "grid",
            "sheet_step",
            "calibration",
            "afv",
            "sources",
            "prediction",
            "orientation",
            "solver",
            "constraints",
            "artifacts",
            "cache",
            "context",
        }
        if not isinstance(data, dict) or data.get("schema") != "winding-dataset-v1" or set(data) - allowed:
            raise ValueError("unsupported dataset schema or unknown configuration fields")
        sections = {
            "grid": {"origin_xyz", "spacing_xyz", "shape_zyx"},
            "afv": {"path", "sha256", "native_frame", "native_voxel_um", "scale_xyz", "offset_xyz"},
            "prediction": {
                "normal_encoding",
                "density_scale",
                "surface_threshold",
                "surface_scale",
                "material_weight",
                "background_weight",
            },
            "orientation": {"mode", "centerline"},
            "sources": {"ct", "surface", "nx", "ny", "nz", "density"},
            "artifacts": {"split", "reference_field", "reference_metadata", "evaluation_case", "annotations"},
            "context": {
                "calibration_ids",
                "split_seed",
                "black_value",
                "halo",
                "sweeps",
                "air_sheets_per_mm",
                "arrival_max_nodes_per_axis",
                "arrival_tolerance",
                "profile_step_mm",
                "label_z_range",
                "exterior_faces",
                "exterior_authority",
                "surface",
                "holdout",
            },
        }
        for name, keys in sections.items():
            if name in data:
                _keys(data[name], keys, name)
        if "arrival_tolerance" in data.get("context", {}):
            positive(data["context"]["arrival_tolerance"], "arrival_tolerance")
        for source in data.get("sources", {}).values():
            _keys(source, {"path", "sha256", "frame", "origin_xyz", "spacing_xyz"}, "source")
        for entry in data.get("artifacts", {}).values():
            _keys(entry, {"path", "sha256", "frame"}, "artifact")
        if "centerline" in data.get("orientation", {}):
            _keys(data["orientation"]["centerline"], {"path", "sha256", "frame"}, "centerline")
        if "surface" in data.get("context", {}):
            _keys(
                data["context"]["surface"],
                {"seed_margin", "ct_padding_samples", "pitch_world", "depth_offsets_world", "mesh_step", "ct_window"},
                "surface settings",
            )
        if "holdout" in data.get("context", {}):
            _keys(
                data["context"]["holdout"],
                {
                    "buffer_world",
                    "margin_world",
                    "alignment_xyz",
                    "max_groups",
                    "max_group_voxels",
                    "max_total_voxels",
                    "development_boxes",
                    "development_z_context",
                },
                "holdout settings",
            )
        setting_keys = {
            "solver": {"device", "rtol", "maxiter"},
            "constraints": {"pair_weight", "pitch_mm", "trace_target", "fiber_limit", "pair_limit"},
        }
        for group, keys in setting_keys.items():
            settings = data.get(group, {})
            if not isinstance(settings, dict) or set(settings) - keys:
                raise ValueError(f"unknown {group} settings")
            for key in ("rtol", "pitch_mm", "trace_target"):
                if key in settings:
                    positive(settings[key], key)
            for key in ("maxiter", "fiber_limit", "pair_limit"):
                if key in settings and (type(settings[key]) is not int or settings[key] < 1):
                    raise ValueError(f"{key} must be a positive integer")
            if "pair_weight" in settings and (
                type(settings["pair_weight"]) not in (int, float)
                or not np.isfinite(settings["pair_weight"])
                or settings["pair_weight"] < 0
            ):
                raise ValueError("pair_weight must be finite and nonnegative")
            if "device" in settings and (not isinstance(settings["device"], str) or not settings["device"]):
                raise ValueError("device must be a nonempty string")
        declared_frame = frame(data["frame"])
        unit = positive(data["voxel_um"], "voxel_um")
        positive(data["sheet_step"], "sheet_step")
        if not isinstance(data["calibration"], str) or not data["calibration"].strip():
            raise ValueError("declare calibration provenance")
        grid = data["grid"]
        xyz(grid["origin_xyz"], "grid origin")
        xyz(grid["spacing_xyz"], "grid spacing", positive=True)
        shape = grid["shape_zyx"]
        if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape):
            raise ValueError("grid shape must contain three integer dimensions >=2 in ZYX order")
        afv = data["afv"]
        frame(afv["native_frame"])
        scale = xyz(afv["scale_xyz"], "AFV scale", positive=True)
        xyz(afv["offset_xyz"], "AFV offset")
        native_um = np.broadcast_to(np.asarray(afv["native_voxel_um"], dtype=float), (3,))
        if (
            not np.isfinite(native_um).all()
            or (native_um <= 0).any()
            or not np.allclose(native_um, scale * unit, rtol=1e-12, atol=0)
        ):
            raise ValueError("native voxel units disagree with native-to-world scale")
        _path(afv, path.parent)
        if not data.get("cache"):
            prediction = data["prediction"]
            encoding = prediction["normal_encoding"]
            if encoding not in ("hemisphere_xy", "signed_xyz"):
                raise ValueError("unsupported normal encoding")
            positive(prediction["density_scale"], "density_scale")
            positive(prediction["material_weight"], "material_weight")
            positive(prediction["background_weight"], "background_weight")
            if not np.isfinite(prediction["surface_threshold"]):
                raise ValueError("surface threshold must be finite")
            required = {"ct", "surface", "nx", "ny", "density"} | ({"nz"} if encoding == "signed_xyz" else set())
            if required - data["sources"].keys():
                raise ValueError("missing required observation source")
            orientation = data["orientation"]
            if orientation["mode"] == "radial":
                centerline = orientation["centerline"]
                if frame(centerline["frame"]) != declared_frame:
                    raise ValueError("centerline/world frames differ")
                _path(centerline, path.parent)
            elif orientation["mode"] != "provided" or encoding != "signed_xyz":
                raise ValueError("unsigned normals require an explicit radial orientation prior")
        else:
            cache = data["cache"]
            if set(cache) != {"target", "weights", "fixed", "metadata"} or any(
                "sha256" not in v for v in cache.values()
            ):
                raise ValueError("cache requires target/weights/fixed/metadata with SHA-256 identities")
            if "orientation" in data:
                orientation = data["orientation"]
                if orientation["mode"] == "radial":
                    centerline = orientation["centerline"]
                    if frame(centerline["frame"]) != declared_frame:
                        raise ValueError("centerline/world frames differ")
                    _path(centerline, path.parent)
                elif orientation["mode"] != "provided":
                    raise ValueError("unsupported orientation")
        for source in data.get("sources", {}).values():
            if frame(source["frame"]) != declared_frame:
                raise ValueError("source/world frames differ")
            xyz(source["origin_xyz"], "source origin")
            xyz(source["spacing_xyz"], "source spacing", positive=True)
            _path(source, path.parent)
        for group in ("artifacts", "cache"):
            for artifact in data.get(group, {}).values():
                _path(artifact, path.parent)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"incomplete or invalid dataset configuration: {exc}") from exc
    data["config_path"] = str(path)
    data["config_sha256"] = sha256(path)
    return data


def validate_afv(dataset, path=None):
    """Compare declared geometry with every available AFV metadata field."""
    declared = dataset["afv"]
    path = Path(path or declared["path"])
    actual = sha256(path)
    if declared.get("sha256", actual) != actual:
        raise ValueError("AFV hash mismatch")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"fibers", "blocks"} <= tables:
            raise ValueError("AFV is missing fibre/block tables")
        metadata = (
            {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM metadata")} if "metadata" in tables else {}
        )
    if "complete" in metadata and metadata["complete"] is not True:
        raise ValueError("AFV is not marked complete")
    root = metadata.get("root", {})
    expected = dict(
        vc_open_data_coordinate_space=declared["native_frame"], coordinate_order="XYZ", coordinate_units="voxel"
    )
    for key, value in expected.items():
        if key in root and root[key] != value:
            raise ValueError(f"AFV coordinate metadata mismatch: {key}")
    if "voxel_size_mm" in root:
        units = np.broadcast_to(np.asarray(root["voxel_size_mm"], float), (3,))
        wanted = np.broadcast_to(np.asarray(declared["native_voxel_um"], float), (3,)) / 1000
        if not np.isfinite(units).all() or not np.allclose(units, wanted, rtol=1e-12, atol=0):
            raise ValueError("AFV native voxel units disagree with configuration")
    return dict(
        sha256=actual,
        native_frame=declared["native_frame"],
        metadata_fields_checked=sorted(set(root) & {*expected, "voxel_size_mm"}),
        unrecorded_coordinates_authority="caller-declared configuration",
    )


def mode_constraints(dataset, allowed, mode):
    controls = dataset.get("constraints", {})
    unsupported = set(controls) - set(allowed)
    if unsupported:
        raise ValueError(f"unsupported constraint settings for {mode}: {', '.join(sorted(unsupported))}")
    return controls


def observation_recipe(dataset):
    """Copy applicable declarations so a cache cannot silently override a request."""
    recipe = {name: deepcopy(dataset[name]) for name in ("prediction", "orientation") if name in dataset}
    sources = {name: source for name, source in dataset.get("sources", {}).items() if name != "ct"}
    if sources:
        recipe["sources"] = deepcopy(sources)
    return recipe


def artifact_path(dataset, name):
    """Resolve an already-normalized artifact and enforce its declared identity."""
    entry = dataset.get("artifacts", {}).get(name)
    if entry is None:
        return None
    path = Path(entry["path"])
    if entry.get("sha256") is not None and sha256(path) != entry["sha256"]:
        raise ValueError(f"artifact hash mismatch: {name}")
    if entry.get("frame", dataset["frame"]) != dataset["frame"]:
        raise ValueError(f"artifact frame mismatch: {name}")
    return path


def open_source(source):
    """Read a local NPY or declared Zarr array; validate supplied file identity."""
    path = source["path"]
    if source.get("sha256") is not None and sha256(path) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    if Path(path).suffix == ".npy":
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        import zarr

        volume = zarr.open(path, mode="r")
    if len(volume.shape) != 3 or np.dtype(volume.dtype).kind not in "biuf":
        raise ValueError("source must be a numeric ZYX volume")
    return volume
