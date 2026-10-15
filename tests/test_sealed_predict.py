import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def scoring_case(tmp_path):
    from conftest import FRAME

    from paris_data import save_field
    from patch_check import sha256

    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {"collections": {"c": {"points": {"a": {"p": [0, 1, 1], "wind_a": 0}, "b": {"p": [2, 1, 1], "wind_a": 2}}}}}
        )
    )
    output = tmp_path / "out"
    output.mkdir()
    field = np.broadcast_to(2 * np.arange(5, dtype=float), (3, 3, 5))
    meta = dict(frame=FRAME, origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1], shape_zyx=[3, 3, 5])
    artifact = {"group": {"group_id": "region", "collection_ids": ["c"]}}
    for kind in ("baseline", "candidate"):
        path = output / f"{kind}.npy"
        save_field(path, field, meta, {"converged": True}, step=2)
        artifact.update(
            {kind: str(path), kind + "_sha256": sha256(path), kind + "_metadata_sha256": sha256(str(path) + ".json")}
        )
    (output / "PREDICTIONS_SEALED.json").write_text(
        json.dumps({"artifacts": [artifact], "labels_expected_sha256": sha256(labels)})
    )
    return SimpleNamespace(output=output, labels=labels)


def test_scale_calibration_balances_collections_and_excludes_unavailable():
    from sealed_predict import development_step

    rows = [dict(collection_id="a", phi_a=0, phi_b=2, dw=1, covered=True)] * 20
    rows += [
        dict(collection_id="b", phi_a=0, phi_b=4, dw=1, covered=True),
        dict(collection_id="c", phi_a=None, phi_b=None, dw=1, covered=False),
    ]
    assert development_step(rows) == 3
    with pytest.raises(ValueError):
        development_step([])


def test_secondary_seal_preserves_primary_and_refuses_after_scoring(tmp_path):
    import json

    import numpy as np

    from sealed_predict import seal_extra_global

    root = tmp_path / "run"
    root.mkdir()
    primary = root / "PREDICTIONS_SEALED.json"
    primary.write_text('{"frozen":true}')
    field = tmp_path / "global.npy"
    np.save(field, np.zeros((2, 2, 2), np.float32))
    (tmp_path / "global.npy.json").write_text("{}")
    seal_extra_global(root, field)
    assert primary.read_text() == '{"frozen":true}'
    assert json.loads((root / "GLOBAL_SECONDARY_SEAL.json").read_text())["role"] == "secondary_before_unsealing"
    (root / "test_results.json").write_text("{}")
    with pytest.raises(ValueError):
        seal_extra_global(root, field)


def test_portable_baseline_parser_and_actual_cpu_child(tmp_path, monkeypatch, paris_inputs):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import sealed_predict
    from sealed_predict import baseline_environment, build_parser

    args = build_parser().parse_args(
        ["predict", "--output", str(tmp_path / "out"), "--device", "cpu", "--config", str(paris_inputs.config)]
    )
    assert args.legacy_solver.is_file()
    assert args.legacy_solver.parent == Path(sealed_predict.__file__).resolve().parent
    monkeypatch.setenv("WREV_DEVICE", "cuda:99")
    env = baseline_environment(args, tmp_path / "field.npy", [0, 0, 0], [4, 6, 8])
    env["PYTHONPATH"] = str(args.legacy_solver.parent) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-c", "import json,paris_solve; print(json.dumps(paris_solve.run_config()))"]
    got = json.loads(subprocess.check_output(command, cwd=tmp_path, env=env, text=True))
    assert got["device"] == "cpu"


def test_baseline_prediction_runs_without_any_winding_label_file(tmp_path, monkeypatch, paris_inputs):
    import json
    import sqlite3
    import sys
    from types import SimpleNamespace

    import numpy as np

    import paris_solve

    for name, value in {
        "WREV_CONFIG": str(paris_inputs.config),
        "WREV_DEVICE": "cpu",
        "SLAB": "0,16,0,16,0,16",
        "OUT": str(tmp_path / "field.npy"),
        "PREDICT_ONLY": "1",
        "ORIENT": "umbi",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AUDIT_DIR", raising=False)
    monkeypatch.delenv("RGT", raising=False)
    monkeypatch.delenv("AFV", raising=False)
    (tmp_path / "umbi.json").write_text(
        json.dumps({"control_points": [{"x": 100, "y": 100, "z": 0}, {"x": 100, "y": 100, "z": 100}]})
    )
    with sqlite3.connect(tmp_path / "Paris4_v1.afv") as db:
        db.execute("CREATE TABLE blocks(id INTEGER, fiber_id INTEGER, first_segment INTEGER, points BLOB)")
        db.execute(
            "CREATE TABLE block_bounds(id INTEGER,min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL)"
        )

    def volume(url, mode):
        value = 100 if "grad_mag" in url else 128 if "lasagna" in url else 255
        return np.full((16, 16, 16), value, np.uint8)

    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(open=volume))
    paris_solve.main([])
    assert not (tmp_path / "relwind.json").exists()
    assert np.isfinite(np.load(tmp_path / "field.npy")).all()


@pytest.mark.parametrize("secondary", [False, True])
def test_scoring_uses_sealed_predictions_and_a_real_constant_control(scoring_case, secondary):
    from sealed_predict import score, seal_extra_global

    args = scoring_case
    if secondary:
        seal_extra_global(args.output, args.output / "candidate.npy")
    score(args)
    result_path = args.output / "test_results.json"
    result = json.loads(result_path.read_text())
    methods = result["methods"]
    for kind in ["baseline", "candidate"] + (["whole_coarse_secondary"] if secondary else []):
        assert methods[kind]["summary"]["correct"] == 1
        assert methods[kind]["rows"][0]["prediction_continuous"] == 2
        assert methods[kind]["rows"][0]["region_cluster_id"] == "region"
    assert methods["constant_control"]["summary"]["correct"] == 0
    assert methods["constant_control"]["rows"][0]["prediction_continuous"] == 0
    assert (result["secondary_seal_sha256"] is not None) is secondary
    before = result_path.read_bytes()
    with pytest.raises(FileExistsError):
        score(args)
    assert result_path.read_bytes() == before


@pytest.mark.parametrize(
    "tampered",
    [
        "labels",
        "baseline",
        "candidate",
        "baseline_metadata",
        "candidate_metadata",
        "primary",
        "secondary",
        "secondary_metadata",
    ],
)
def test_tampering_aborts_scoring_without_a_result(scoring_case, tampered):
    from sealed_predict import score, seal_extra_global

    args = scoring_case
    extra = args.output / "extra.npy"
    np.save(extra, np.zeros((3, 3, 5), np.float32))
    extra_meta = args.output / "extra.npy.json"
    extra_meta.write_bytes((args.output / "candidate.npy.json").read_bytes())
    seal_extra_global(args.output, extra)
    paths = {
        "labels": args.labels,
        "primary": args.output / "PREDICTIONS_SEALED.json",
        "secondary": extra,
        "secondary_metadata": extra_meta,
    }
    for kind in ("baseline", "candidate"):
        paths[kind] = args.output / f"{kind}.npy"
        paths[kind + "_metadata"] = args.output / f"{kind}.npy.json"
    path = paths[tampered]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity|changed"):
        score(args)
    assert not (args.output / "test_results.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "geometry_frame",
        "baseline_frame",
        "baseline_units",
        "baseline_grid",
        "baseline_hash",
        "nonconverged",
        "different_spacing",
    ],
)
def test_prediction_seals_real_solver_artifacts_without_opening_labels(paris_inputs, damage):
    from patch_check import sha256
    from sealed_predict import build_parser, predict

    root = paris_inputs.root
    manifest = root / "geometry.json"
    manifest.write_text(
        json.dumps(
            {
                "frame": "different-frame" if damage == "geometry_frame" else paris_inputs.dataset["frame"],
                "initial_test": {
                    "groups": [
                        {
                            "group_id": "synthetic",
                            "collection_ids": ["c"],
                            "grid": {
                                "origin_xyz": [0, 0, 0],
                                "hi_xyz": [80, 64, 48],
                                "spacing_xyz": [8, 8, 8] if damage == "different_spacing" else [16, 16, 16],
                            },
                        }
                    ]
                },
                "label_files": {"initial-test-labels.json": {"sha256": "a" * 64}},
            }
        )
    )
    # A real child process supplies the historical baseline; there is no label file.
    child = root / "baseline.py"
    child.write_text(
        "import os,json,hashlib,sys\nfrom pathlib import Path\nimport numpy as np\n"
        f"damage={damage!r}\n"
        "config=json.loads(Path(os.environ['WREV_CONFIG']).read_text())\n"
        "h=np.array([float(v) for v in sys.argv[sys.argv.index('--spacing')+1:sys.argv.index('--spacing')+4]]) if '--spacing' in sys.argv else np.array([16.,16.,16.])\n"
        "bounds=np.array([float(v) for v in os.environ['SLAB'].split(',')]).reshape(3,2)\n"
        "shape=np.ceil((bounds[:,1]-bounds[:,0])/h).astype(int)[::-1]\n"
        "path = os.environ['OUT']\nnp.save(path, np.zeros(tuple(shape), np.float32))\n"
        "meta=dict(frame=config['frame'],voxel_um=config['voxel_um'],origin_xyz=bounds[:,0].tolist(),"
        "spacing_xyz=h.tolist(),shape_zyx=shape.tolist(),step=config['sheet_step'],solver={'converged':True},"
        "field_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())\n"
        "if damage=='baseline_frame': meta['frame']='different-frame'\n"
        "if damage=='baseline_units': meta['voxel_um']=12\n"
        "if damage=='baseline_grid': meta['spacing_xyz']=[32,32,32]\n"
        "if damage=='baseline_hash': meta['field_sha256']='0'*64\n"
        "if damage=='nonconverged': meta['solver']['converged']=False\n"
        "Path(path + '.json').write_text(json.dumps(meta))\n"
    )
    args = build_parser().parse_args(
        [
            "predict",
            "--output",
            str(root / "predictions"),
            "--manifest",
            str(manifest),
            "--development",
            str(paris_inputs.development),
            "--config",
            str(paris_inputs.config),
            "--legacy-solver",
            str(child),
            "--device",
            "cpu",
        ]
    )
    if damage not in (None, "different_spacing"):
        with pytest.raises(ValueError):
            predict(args)
        assert not (args.output / "PREDICTIONS_SEALED.json").exists()
        assert not (args.output / "group0_candidate.npy").exists()
        return
    predict(args)
    seal = json.loads((args.output / "PREDICTIONS_SEALED.json").read_text())
    artifact = seal["artifacts"][0]
    assert artifact["candidate_sha256"] == sha256(artifact["candidate"])
    assert artifact["baseline_metadata_sha256"] == sha256(artifact["baseline"] + ".json")
    candidate = np.load(artifact["candidate"])
    expected_shape = (6, 8, 10) if damage == "different_spacing" else (3, 4, 5)
    assert candidate.shape == expected_shape and np.isfinite(candidate).all() and np.ptp(candidate) > 0
    assert seal["config"]["global_step"] == 2
    assert seal["config"]["private_scoring_subset_opened"] is False
    assert seal["config"]["label_access_isolated"] is False
    assert json.loads((args.output / "progress.json").read_text()) == {"completed": 1, "total": 1}
    # Failure of a real child cannot leave a successful seal behind.
    child.write_text("raise SystemExit(17)\n")
    args.output = root / "failed-predictions"
    with pytest.raises(subprocess.CalledProcessError) as failure:
        predict(args)
    assert failure.value.returncode == 17
    assert not (args.output / "PREDICTIONS_SEALED.json").exists()
    assert not (args.output / "group0_candidate.npy").exists()
