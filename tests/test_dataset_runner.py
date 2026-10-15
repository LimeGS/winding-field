"""Real CLI construction from portable synthetic observations and SQLite fibres."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_dataset_config import configuration

ROOT = Path(__file__).resolve().parents[1]


def dataset_files(root, frame="synthetic-one/world", voxel_um=1000):
    root.mkdir()
    config = configuration(frame)
    config["voxel_um"] = voxel_um
    config["afv"]["native_voxel_um"] = [voxel_um * v for v in config["afv"]["scale_xyz"]]
    config["prediction"]["normal_encoding"] = "signed_xyz"
    config["prediction"]["density_scale"] = 1
    config["orientation"] = {"mode": "provided"}
    config["sources"]["nz"] = dict(config["sources"]["nx"], path="nz.npy")
    config["solver"] = dict(rtol=1e-6, maxiter=1000)
    shape = tuple(config["grid"]["shape_zyx"])
    for name in config["sources"]:
        value = 1 if name in ("nx", "density") else 255 if name in ("surface", "ct") else 0
        volume = np.full(shape, value, np.float32)
        if name in ("surface", "ct"):
            volume[:, :, 0] = 0
        np.save(root / f"{name}.npy", volume)
    scale = np.array(config["afv"]["scale_xyz"])
    offset = np.array(config["afv"]["offset_xyz"])
    with sqlite3.connect(root / "fibres.afv") as db:
        db.executescript("""CREATE TABLE fibers(id INTEGER PRIMARY KEY, length REAL, annotation TEXT);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY, fiber_id INTEGER, first_segment INTEGER, points BLOB);
            CREATE TABLE block_bounds(id INTEGER, min_x REAL, max_x REAL, min_y REAL, max_y REAL,
                                      min_z REAL, max_z REAL);""")
        for fid, x in enumerate((12, 16), 1):
            points = (np.array([[x, -6, 6], [x, 0, 6]], float) - offset) / scale
            db.execute("INSERT INTO fibers VALUES (?,?,?)", (fid, 6, "{}"))
            db.execute("INSERT INTO blocks VALUES (?,?,?,?)", (fid, fid, 0, points.astype("<f8").tobytes()))
            bounds = np.stack([points.min(0), points.max(0)], axis=1).ravel().tolist()
            db.execute("INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)", [fid, *bounds])
    path = root / "dataset.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("frame,unit", [("synthetic-one/world", 1000), ("another-scroll/grid", 500)])
def test_observation_loader_uses_declared_sources_units_and_offset(tmp_path, frame, unit):
    from dataset_config import load_dataset
    from paris_data import load_paris_box

    path = dataset_files(tmp_path / "inputs", frame, unit)
    dataset = load_dataset(path)
    target, weights, pairs, meta = load_paris_box(
        [10, -8, 4], [24, 4, 14], spacing=[2, 2, 2], dataset=dataset, afv_path=dataset["afv"]["path"]
    )
    assert target.shape == (5, 6, 7, 3) and meta["frame"] == frame and meta["voxel_um"] == unit
    np.testing.assert_array_equal(target[..., 0], 1)
    np.testing.assert_array_equal(target[..., 1:], 0)
    assert len(pairs) and weights.shape == (5, 6, 7)
    assert meta["dataset_config_sha256"] == dataset["config_sha256"]


@pytest.mark.parametrize("frame,unit", [("synthetic-one/world", 1000), ("another-scroll/grid", 500)])
def test_global_cli_builds_without_private_paths_or_human_labels(tmp_path, frame, unit):
    path = dataset_files(tmp_path / "inputs", frame, unit)
    output = tmp_path / "output"
    command = [
        sys.executable,
        str(ROOT / "tools/run_coherence.py"),
        "global",
        "--config",
        str(path),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--stages",
        "B2",
    ]
    run = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    field = np.load(output / "B2.npy")
    meta = json.loads((output / "B2.npy.json").read_text())
    assert field.shape == (5, 6, 7) and meta["frame"] == frame and meta["voxel_um"] == unit
    assert meta["step"] == 1.5 and meta["solver"]["converged"]
    np.testing.assert_allclose(field[2, 2], np.arange(7) * 2, atol=0.001)
    assert (output / "MANIFEST.json").is_file()
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert repeated.returncode != 0
    np.testing.assert_array_equal(np.load(output / "B2.npy"), field)


@pytest.mark.parametrize("maxiter", [1, 1000])
def test_legacy_solver_uses_explicit_config_and_refuses_nonconvergence(tmp_path, maxiter):
    path = dataset_files(tmp_path / "inputs")
    config = json.loads(path.read_text())
    config["solver"].update(maxiter=maxiter, device="cpu")
    path.write_text(json.dumps(config))
    output = tmp_path / "legacy.npy"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/paris_solve.py"),
            "--config",
            str(path),
            "--output",
            str(output),
            "--prediction-only",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "ORIENT": "provided", "WREV_DATA": str(tmp_path)},
    )
    if maxiter == 1:
        assert run.returncode != 0 and not output.exists()
        assert "converge" in run.stderr.lower()
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        meta = json.loads(Path(str(output) + ".json").read_text())
        assert meta["frame"] == "synthetic-one/world" and meta["voxel_um"] == 1000
        assert meta["step"] == 1.5 and meta["solver"]["converged"]


def test_fibre_compilation_requires_physical_units():
    from coherence_pipeline import compile_fibres

    with pytest.raises(TypeError):
        compile_fibres([dict(fibre_id=1, points_xyz=[[1, 0, 1], [1, 3, 1]])], [0, 0, 0], [1, 1, 1], [4, 4, 4])


def test_configured_trace_budget_reaches_the_saved_operator(tmp_path):
    from field_constraints import QuadraticConstraints

    path = dataset_files(tmp_path / "inputs")
    config = json.loads(path.read_text())
    config["constraints"] = dict(trace_target=83.75)
    path.write_text(json.dumps(config))
    output = tmp_path / "output"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_coherence.py"),
            "global",
            "--config",
            str(path),
            "--output",
            str(output),
            "--stages",
            "B2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert QuadraticConstraints.load(output / "pairs.npz").trace == pytest.approx(83.75)


@pytest.mark.parametrize("damage", [None, "frame", "hash"])
def test_cache_only_build_is_bound_to_declared_geometry_and_hashes(tmp_path, damage):
    from patch_check import sha256

    path = dataset_files(tmp_path / "inputs")
    data = json.loads(path.read_text())
    for key in ("sources", "prediction", "orientation"):
        del data[key]
    shape = tuple(data["grid"]["shape_zyx"])
    target = np.zeros((*shape, 3), np.float32)
    target[..., 0] = 1
    fixed = np.full(shape, np.nan, np.float32)
    fixed[:, :, 0] = 0
    cache = {}
    for name, values in dict(target=target, weights=np.ones(shape, np.float32), fixed=fixed).items():
        file = path.parent / f"cache-{name}.npy"
        np.save(file, values)
        cache[name] = dict(path=file.name, sha256=sha256(file))
    meta = dict(
        data["grid"],
        frame="wrong" if damage == "frame" else data["frame"],
        voxel_um=data["voxel_um"],
        step=data["sheet_step"],
    )
    metadata = path.parent / "cache.json"
    metadata.write_text(json.dumps(meta))
    cache["metadata"] = dict(path=metadata.name, sha256=sha256(metadata))
    if damage == "hash":
        cache["weights"]["sha256"] = "0" * 64
    data["cache"] = cache
    path.write_text(json.dumps(data))
    output = tmp_path / "output"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_coherence.py"),
            "global",
            "--config",
            str(path),
            "--output",
            str(output),
            "--stages",
            "B2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if damage:
        assert run.returncode != 0 and not (output / "B2.npy").exists()
        assert "cache" in run.stderr.lower()
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        np.testing.assert_allclose(np.load(output / "B2.npy")[2, 2], np.arange(7) * 2, atol=0.001)


def test_public_build_demo_runs_the_configured_cli_offline(tmp_path):
    script = ROOT / "examples/build_demo.py"
    assert script.is_file(), "portable construction example is missing"
    output = tmp_path / "demo"
    command = [sys.executable, str(script), "--output", str(output)]
    run = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    field = np.load(output / "field/B2.npy")
    np.testing.assert_allclose(field[2, 3], np.arange(9), atol=0.001)
    assert "synthetic" in run.stdout.lower()
    assert (output / "dataset.json").is_file()
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert repeated.returncode != 0
    np.testing.assert_array_equal(np.load(output / "field/B2.npy"), field)


def test_global_cli_returns_failure_and_retains_failed_solver_evidence(tmp_path):
    path = dataset_files(tmp_path / "inputs")
    data = json.loads(path.read_text())
    data["solver"].update(maxiter=1, rtol=1e-10)
    path.write_text(json.dumps(data))
    output = tmp_path / "output"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_coherence.py"),
            "global",
            "--config",
            str(path),
            "--output",
            str(output),
            "--stages",
            "B2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert run.returncode != 0, run.stdout + run.stderr
    assert not (output / "B2.npy").exists()
    assert (output / "B2-FAILED.npy").exists()
    assert json.loads((output / "B2-solver.json").read_text())["converged"] is False


@pytest.mark.parametrize("damage", ["frame", "units", "order", "incomplete"])
def test_declared_afv_geometry_must_agree_with_available_metadata(tmp_path, damage):
    path = dataset_files(tmp_path / "inputs")
    root = dict(
        vc_open_data_coordinate_space="test-scroll/native",
        coordinate_order="XYZ",
        coordinate_units="voxel",
        voxel_size_mm=[0.5, 1.0, 1.5],
    )
    if damage == "frame":
        root["vc_open_data_coordinate_space"] = "a-different-scan"
    elif damage == "units":
        root["voxel_size_mm"] = 0.0024
    elif damage == "order":
        root["coordinate_order"] = "ZYX"
    with sqlite3.connect(path.parent / "fibres.afv") as db:
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [("root", json.dumps(root)), ("complete", json.dumps(damage != "incomplete"))],
        )
    output = tmp_path / "output"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_coherence.py"),
            "global",
            "--config",
            str(path),
            "--output",
            str(output),
            "--stages",
            "B2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert run.returncode != 0, run.stdout + run.stderr
    assert not (output / "B2.npy").exists()
    assert "AFV" in run.stderr


@pytest.mark.parametrize("declared_exterior", [False, True])
def test_black_ct_is_not_implicitly_treated_as_verified_exterior(tmp_path, declared_exterior):
    path = dataset_files(tmp_path / "inputs")
    data = json.loads(path.read_text())
    np.save(path.parent / "ct.npy", np.zeros(tuple(data["grid"]["shape_zyx"]), np.float32))
    if declared_exterior:
        data["context"] = dict(exterior_faces=["x_min"], exterior_authority="controlled synthetic outside face")
    path.write_text(json.dumps(data))
    output = tmp_path / "output"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_coherence.py"),
            "global",
            "--config",
            str(path),
            "--output",
            str(output),
            "--stages",
            "B2",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    field = np.load(output / "B2.npy")
    np.testing.assert_allclose(field[2, 2], np.arange(7) * 2, atol=0.001)
    info = json.loads((output / "B2-solver.json").read_text())
    assert info["gauge"]["kind"] == ("fixed" if declared_exterior else "zero")
    if declared_exterior:
        assert info["gauge"]["count"] == 30


def test_explicit_decimal_grid_shape_survives_observation_loading(tmp_path):
    from run_coherence import observations

    from dataset_config import load_dataset

    path = dataset_files(tmp_path / "inputs")
    data = json.loads(path.read_text())
    data["grid"] = dict(origin_xyz=[0.1] * 3, spacing_xyz=[0.1] * 3, shape_zyx=[3] * 3)
    for name, source in data["sources"].items():
        source.update(origin_xyz=[0.1] * 3, spacing_xyz=[0.1] * 3)
        value = 1 if name in ("nx", "density") else 255 if name in ("ct", "surface") else 0
        np.save(path.parent / source["path"], np.full((3, 3, 3), value, np.float32))
    path.write_text(json.dumps(data))
    target, weights, _, meta = observations(load_dataset(path))
    assert target.shape == (3, 3, 3, 3) and weights.shape == (3, 3, 3)
    assert meta["shape_zyx"] == [3, 3, 3]


@pytest.mark.parametrize("change", [None, "density_scale", "source", "orientation"])
def test_cache_binds_declared_observations_without_reopening_sources(tmp_path, change):
    from run_coherence import observations

    from dataset_config import load_dataset
    from patch_check import sha256

    path = dataset_files(tmp_path / "inputs")
    original = load_dataset(path)
    target, weights, fixed, metadata = observations(original)
    cache = {}
    for name, values in dict(target=target, weights=weights, fixed=fixed).items():
        p = path.parent / f"cached-{name}.npy"
        np.save(p, values)
        cache[name] = dict(path=p.name, sha256=sha256(p))
    p = path.parent / "cached-metadata.json"
    p.write_text(json.dumps(metadata))
    cache["metadata"] = dict(path=p.name, sha256=sha256(p))
    data = json.loads(path.read_text())
    data["cache"] = cache
    if change == "density_scale":
        data["prediction"]["density_scale"] = 2
    elif change == "source":
        data["sources"]["density"]["path"] = "different.npy"
    elif change == "orientation":
        data["orientation"] = dict(mode="radial", centerline=dict(path="axis.json", frame=data["frame"]))
    path.write_text(json.dumps(data))
    for source in original["sources"].values():
        Path(source["path"]).unlink()
    requested = load_dataset(path)
    if change:
        with pytest.raises(ValueError, match="observation"):
            observations(requested)
    else:
        cached, _, _, _ = observations(requested)
        np.testing.assert_array_equal(cached, target)


def arrival_dataset(root, *, black_value=0, face="x_min", crop=32):
    from paris_data import save_field
    from patch_check import sha256

    path = dataset_files(root)
    data = json.loads(path.read_text())
    shape = tuple(data["grid"]["shape_zyx"])
    ct, surface = np.full(shape, 255, np.int16), np.full(shape, 255, np.uint8)
    side = 0 if face == "x_min" else -1
    ct[:, :, side] = black_value
    surface[:, :, side] = 0
    np.save(root / "ct.npy", ct)
    np.save(root / "surface.npy", surface)
    data["context"] = dict(
        exterior_faces=[face],
        exterior_authority="synthetic exterior",
        black_value=black_value,
        arrival_max_nodes_per_axis=crop,
        profile_step_mm=0.01,
    )
    data["prediction"]["surface_scale"] = 1 / 255
    reference = root / "reference.npy"
    metadata = dict(data["grid"], frame=data["frame"], voxel_um=data["voxel_um"])
    save_field(
        reference, np.broadcast_to(np.arange(shape[2]) * 2.0, shape), metadata, {"converged": True}, data["sheet_step"]
    )
    data["artifacts"] = dict(reference_field=dict(path=reference.name, sha256=sha256(reference)))
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("background", [0, -1000])
def test_arrival_boundary_uses_declared_background(tmp_path, background):
    from run_coherence import arrival_boundary_run

    from dataset_config import load_dataset

    path = arrival_dataset(tmp_path / "inputs", black_value=background)
    data = json.loads(path.read_text())
    data["solver"].pop("rtol")
    path.write_text(json.dumps(data))
    result = arrival_boundary_run(load_dataset(path), tmp_path / "arrival")
    assert result["cases"]["B5"]["converged"]


def test_arrival_only_rejects_cg_settings_and_honors_its_own_budget(tmp_path):
    from run_coherence import arrival_boundary_run

    from dataset_config import load_dataset

    path = arrival_dataset(tmp_path / "inputs")
    with pytest.raises(ValueError, match="rtol"):
        arrival_boundary_run(load_dataset(path), tmp_path / "bad")
    data = json.loads(path.read_text())
    data["solver"].pop("rtol")
    data["solver"]["maxiter"] = 1
    data["context"]["arrival_tolerance"] = 1e-12
    path.write_text(json.dumps(data))
    result = arrival_boundary_run(load_dataset(path), tmp_path / "limited")
    assert result["cases"]["B5"]["iterations"] == 1
    assert result["cases"]["B5"]["tolerance"] == 1e-12
    assert not result["cases"]["B5"]["converged"]


@pytest.mark.parametrize("face,crop,available", [("x_min", 32, True), ("x_max", 4, False)])
def test_local_arrival_uses_only_exterior_faces_retained_in_its_crop(tmp_path, face, crop, available):
    from run_coherence import local_run

    from dataset_config import load_dataset

    path = arrival_dataset(tmp_path / "inputs", black_value=-1000, face=face, crop=crop)
    result = local_run(load_dataset(path), tmp_path / "local", "cpu")
    if available:
        assert result["cases"]["B5"]["converged"], result["cases"]["B5"]
    else:
        assert result["cases"]["B5"]["status"] == "UNAVAILABLE_NO_OBSERVED_EXTERIOR"


def test_local_mode_forwards_gradient_constraint_settings(tmp_path):
    from run_coherence import local_run

    from dataset_config import load_dataset
    from field_constraints import QuadraticConstraints

    path = arrival_dataset(tmp_path / "inputs")
    data = json.loads(path.read_text())
    data["constraints"] = dict(pair_weight=0.2, pitch_mm=0.4, trace_target=83.75)
    path.write_text(json.dumps(data))
    output = tmp_path / "local"
    local_run(load_dataset(path), output, "cpu")
    assert QuadraticConstraints.load(output / "pairs.npz").trace == pytest.approx(83.75)
    assert json.loads((output / "constraint-audit.json").read_text())["pitch_mm"] == 0.4
    with np.load(output / "constraint_points.npz") as points:
        indices = (points["a_xyz"][:, 1] + 6) / 0.4
        np.testing.assert_allclose(indices, np.rint(indices), atol=1e-10)


def test_continuous_pilot_forwards_solver_and_constraint_settings(tmp_path):
    from run_coherence import continuous_pilot_run

    from dataset_config import load_dataset
    from field_constraints import QuadraticConstraints

    path = arrival_dataset(tmp_path / "inputs")
    data = json.loads(path.read_text())
    data["solver"].update(maxiter=1, rtol=1e-10)
    data["context"].update(halo=1, sweeps=1)
    data["constraints"] = dict(pair_weight=0.2, pitch_mm=0.4, trace_target=83.75)
    shape = tuple(data["grid"]["shape_zyx"])
    np.save(path.parent / "density.npy", 1 + np.indices(shape)[1].astype(np.float32) * 0.3)
    path.write_text(json.dumps(data))
    output = tmp_path / "pilot"
    result = continuous_pilot_run(load_dataset(path), output, "cpu")
    assert not result["cases"]["continuous_pilot"]["converged"]
    receipt = json.loads((output / "pilot/receipt.json").read_text())
    assert receipt["configuration"]["maxiter"] == 1
    assert receipt["solvers"]["monolithic"]["iterations"] == 1
    assert QuadraticConstraints.load(output / "continuous.npz").trace == pytest.approx(83.75)
    assert json.loads((output / "audit.json").read_text())["pitch_mm"] == 0.4


def test_continuous_pilot_auto_evaluation_binds_support_and_full_receipts(tmp_path):
    import gzip

    from run_coherence import continuous_pilot_run

    from common import BINS
    from dataset_config import load_dataset
    from patch_check import sha256

    path = arrival_dataset(tmp_path / "inputs")
    data = json.loads(path.read_text())
    data["context"].update(halo=1, sweeps=1)
    scale = np.array(data["afv"]["scale_xyz"])
    offset = np.array(data["afv"]["offset_xyz"])
    world = np.array([[12, -6, 6], [12, -5, 6], [12, 0, 6]], float)
    native = (world - offset) / scale
    afv = path.parent / "fibres.afv"
    with sqlite3.connect(afv) as db:
        db.execute("UPDATE blocks SET points=? WHERE fiber_id=1", (native.astype("<f8").tobytes(),))
    root = path.parent / "case"
    root.mkdir()
    selection = root / "selection.json.gz"
    row = dict(
        fiber_id=1,
        bin=0,
        i=0,
        j=1,
        span_mm=1.0,
        chord_mm=1.0,
        a_native_xyz=native[0].tolist(),
        b_native_xyz=native[1].tolist(),
    )
    with gzip.open(selection, "wt") as stream:
        json.dump({"pilot": [row]}, stream)
    audit = root / "audit.json"
    audit.write_text(json.dumps(dict(afv_sha256=sha256(afv))))
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                frame=data["frame"],
                fixed_sheet_step=data["sheet_step"],
                bins=BINS,
                native_voxel_mm=(np.array(data["afv"]["native_voxel_um"]) / 1000).tolist(),
                constraint_audit_sha256=sha256(audit),
            )
        )
    )
    seal = root / "seal.json"
    seal.write_text(json.dumps(dict(selected_sha256=sha256(selection), plan_sha256=sha256(plan))))
    artifacts = {
        name: dict(path=p.name, sha256=sha256(p))
        for name, p in dict(selection=selection, audit=audit, plan=plan, seal=seal).items()
    }
    artifacts["afv"] = dict(path="../fibres.afv", sha256=sha256(afv))
    case = root / "case.json"
    case.write_text(
        json.dumps(
            dict(
                schema="winding-benchmark-case-v2",
                scope="pilot",
                frame=data["frame"],
                voxel_um=data["voxel_um"],
                step=data["sheet_step"],
                population="all",
                expected_selected=1,
                expected_human=0,
                requires_support=False,
                artifacts=artifacts,
                native_to_world={k: data["afv"][k] for k in ("scale_xyz", "offset_xyz")},
            )
        )
    )
    data["artifacts"]["evaluation_case"] = dict(path="case/case.json", sha256=sha256(case))
    path.write_text(json.dumps(data))
    output = tmp_path / "pilot"
    result = continuous_pilot_run(load_dataset(path), output, "cpu")
    assert result["cases"]["continuous_pilot"]["converged"]
    for name in ("assembled", "monolithic"):
        evaluation = output / "pilot" / (name + "-evaluation")
        assert (evaluation / "PREFLIGHT.json").is_file() and (evaluation / "MANIFEST.json").is_file()
        summary = json.loads((evaluation / "SUMMARY.json").read_text())
        assert summary["methods"]["candidate"]["all"][0]["covered"] == 1
    meta = json.loads((output / "pilot/assembled.npy.json").read_text())
    producer = json.loads((output / "pilot/receipt.json").read_text())
    assert meta["support_sha256"] == producer["support_sha256"] == sha256(output / "pilot/support.npy")
    assert (output / "RESULTS.json").is_file()
