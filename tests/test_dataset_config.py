"""Explicit physical inputs, independent of scroll names and working directories."""

import importlib
import importlib.util
import json

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("dataset_config"), "explicit dataset configuration is not implemented"
    return importlib.import_module("dataset_config")


def configuration(frame="test-scroll/world"):
    source = dict(frame=frame, origin_xyz=[10, -8, 4], spacing_xyz=[2, 2, 2])
    return dict(
        schema="winding-dataset-v1",
        frame=frame,
        voxel_um=6,
        grid=dict(origin_xyz=[10, -8, 4], spacing_xyz=[2, 2, 2], shape_zyx=[5, 6, 7]),
        sheet_step=1.5,
        calibration="synthetic analytic calibration",
        afv=dict(
            path="fibres.afv",
            native_frame="test-scroll/native",
            native_voxel_um=[3, 6, 9],
            scale_xyz=[0.5, 1, 1.5],
            offset_xyz=[10, -8, 4],
        ),
        sources={name: dict(source, path=name + ".npy") for name in ("ct", "surface", "nx", "ny", "density")},
        prediction=dict(
            normal_encoding="hemisphere_xy",
            density_scale=0.001,
            surface_threshold=128,
            material_weight=1,
            background_weight=0.1,
        ),
        orientation=dict(mode="radial", centerline=dict(path="axis.json", frame=frame)),
    )


@pytest.mark.parametrize("frame", ["test-scroll/world", "another-volume/calibrated"])
@pytest.mark.parametrize("cached", [False, True])
def test_config_paths_and_transform_do_not_depend_on_scroll_or_cwd(tmp_path, monkeypatch, frame, cached):
    module = api()
    directory = tmp_path / "dataset"
    directory.mkdir()
    path = directory / "scan.json"
    data = configuration(frame)
    if cached:
        data["cache"] = {
            name: dict(path=name + ".npy", sha256="0" * 64) for name in ("target", "weights", "fixed", "metadata")
        }
    path.write_text(json.dumps(data))
    monkeypatch.chdir(tmp_path)
    config = module.load_dataset(path)
    assert config["frame"] == frame
    assert config["afv"]["path"] == str(directory / "fibres.afv")
    assert config["sources"]["ct"]["path"] == str(directory / "ct.npy")
    assert config["orientation"]["centerline"]["path"] == str(directory / "axis.json")
    np.testing.assert_array_equal(module.native_to_world([[2, 4, 6]], config["afv"]), [[11, -4, 13]])


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "frame",
        "units",
        "nan",
        "shape",
        "spacing",
        "scale",
        "offset",
        "source_frame",
        "encoding",
        "calibration",
        "missing",
        "unknown",
    ],
)
def test_incomplete_or_inconsistent_physical_config_fails(tmp_path, damage):
    module = api()
    data = configuration()
    if damage == "schema":
        data["schema"] = "unknown"
    elif damage == "frame":
        data["frame"] = ""
    elif damage == "units":
        data["voxel_um"] = 0
    elif damage == "nan":
        data["sheet_step"] = float("nan")
    elif damage == "shape":
        data["grid"]["shape_zyx"][0] = 5.5
    elif damage == "spacing":
        data["grid"]["spacing_xyz"][0] = -1
    elif damage == "scale":
        data["afv"]["scale_xyz"][0] = 0.25
    elif damage == "offset":
        data["afv"]["offset_xyz"] = [0, 1]
    elif damage == "source_frame":
        data["sources"]["ct"]["frame"] = "different-volume"
    elif damage == "encoding":
        data["prediction"]["normal_encoding"] = "unknown"
    elif damage == "calibration":
        data["calibration"] = ""
    elif damage == "missing":
        del data["sources"]["density"]
    else:
        data["voxl_um"] = 6
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.load_dataset(path)


def test_signed_normals_require_three_explicit_channels(tmp_path):
    module = api()
    data = configuration()
    data["prediction"]["normal_encoding"] = "signed_xyz"
    data["orientation"] = {"mode": "provided"}
    path = tmp_path / "scan.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        module.load_dataset(path)
    data["sources"]["nz"] = dict(data["sources"]["nx"], path="nz.npy")
    path.write_text(json.dumps(data))
    assert module.load_dataset(path)["orientation"]["mode"] == "provided"


def test_local_source_is_read_only_and_checks_content_identity(tmp_path):
    module = api()
    from patch_check import sha256

    path = tmp_path / "source.npy"
    np.save(path, np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    source = dict(path=str(path), sha256=sha256(path))
    volume = module.open_source(source)
    assert volume.shape == (2, 3, 4) and volume[1, 2, 3] == 23
    assert not volume.flags.writeable
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        module.open_source(source)


@pytest.mark.parametrize(
    "bad", [dict(scale_xyz=[0, 1, 1], offset_xyz=[0, 0, 0]), dict(scale_xyz=[1, 1, 1], offset_xyz=[0, float("nan"), 0])]
)
def test_transform_rejects_invalid_geometry(bad):
    with pytest.raises(ValueError):
        api().native_to_world([[1, 2, 3]], bad)


@pytest.mark.parametrize(
    "group,values",
    [
        ("solver", {"maxitre": 1}),
        ("constraints", {"trce_target": 100}),
        ("solver", {"rtol": float("nan")}),
        ("solver", {"maxiter": 0}),
        ("constraints", {"pair_limit": 0}),
    ],
)
def test_solver_settings_are_validated_instead_of_silently_ignored(tmp_path, group, values):
    data = configuration()
    data[group] = values
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        api().load_dataset(path)


@pytest.mark.parametrize("location", ["grid", "afv", "prediction", "orientation", "context"])
def test_unknown_nested_configuration_cannot_change_semantics_silently(tmp_path, location):
    data = configuration()
    data.setdefault(location, {})["misspelled_setting"] = 1
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown"):
        api().load_dataset(path)
