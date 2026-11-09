import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation_split import build_fibre_split
from patch_check import sha256


def evaluator():
    path = Path(__file__).with_name("evaluate.py")
    assert path.is_file(), "artifact evaluator is missing"
    spec = importlib.util.spec_from_file_location("artifact_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path):
    frame = "PHercParis4/20260411134726:L2"
    afv = tmp_path / "fibres.afv"
    afv.write_bytes(b"synthetic AFV identity")
    split = build_fibre_split(
        [dict(fibre_id=i, length_mm=1, source_lineage=[["synthetic", str(i)]]) for i in (1, 2, 3)],
        {1},
        reserved_fraction=0.5,
    )
    split["afv_sha256"] = sha256(afv)
    reserved = next(r["fibre_id"] for r in split["fibres"] if r["split"] == "reserved")
    other = next(i for i in (2, 3) if i != reserved)
    rows = [
        dict(
            fiber_id=i,
            bin=0,
            i=0,
            j=1,
            span_mm=0.576,
            chord_mm=0.576,
            a_native_xyz=[x * 4, 8, 8],
            b_native_xyz=[(x + 60) * 4, 8, 8],
        )
        for i, x in [(1, 1), (reserved, 2), (other, 0)]
    ]
    selection = tmp_path / "selected.json.gz"
    with gzip.open(selection, "wt") as stream:
        json.dump({"whole_coarse": rows}, stream)
    field = tmp_path / "field.npy"
    np.save(field, np.broadcast_to(np.arange(80, dtype=float) / 30, (5, 6, 80)))
    meta = dict(
        frame=frame,
        voxel_um=9.6,
        origin_xyz=[0, 0, 0],
        spacing_xyz=[1, 1, 1],
        shape_zyx=[5, 6, 80],
        step=2.0,
        field_sha256=sha256(field),
        solver={"converged": True},
    )
    metadata = tmp_path / "field.npy.json"
    metadata.write_text(json.dumps(meta))
    from common import BINS

    plan = dict(
        frame=frame,
        fixed_sheet_step=2.0,
        native_voxel_mm=0.0024,
        bins=BINS,
        population="all",
        fields={"whole_coarse": {"metadata": meta}},
    )
    human = dict(
        methods={
            "whole_coarse_secondary": {
                "rows": [
                    dict(
                        pair_id="human",
                        collection_id="c",
                        point_a_id="a",
                        point_b_id="b",
                        a_xyz=[1, 2, 2],
                        b_xyz=[61, 2, 2],
                        wind_a=0,
                        wind_b=1,
                        dw=1,
                    )
                ]
            }
        }
    )
    artifacts = {
        "selection": {"path": selection.name, "sha256": sha256(selection)},
        "afv": {"path": afv.name, "sha256": sha256(afv)},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(dict(afv_sha256=sha256(afv))))
    artifacts["audit"] = dict(path=audit_path.name, sha256=sha256(audit_path))
    plan["constraint_audit_sha256"] = sha256(audit_path)
    for name, content in [("plan", plan), ("split", split), ("human", human)]:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(content))
        artifacts[name] = dict(path=path.name, sha256=sha256(path))
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps(dict(selected_sha256=sha256(selection), plan_sha256=artifacts["plan"]["sha256"])))
    artifacts["seal"] = dict(path=seal.name, sha256=sha256(seal))
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            dict(
                schema="winding-benchmark-case-v1",
                scope="whole_coarse",
                frame=frame,
                voxel_um=9.6,
                native_to_world=dict(scale_xyz=[0.25] * 3, offset_xyz=[0, 0, 0]),
                step=2.0,
                expected_selected=3,
                expected_human=1,
                human_method="whole_coarse_secondary",
                requires_support=False,
                artifacts=artifacts,
            )
        )
    )
    return dict(root=tmp_path, case=case_path, field=field, metadata=metadata)


def arguments(case, name="run"):
    return [
        "--case",
        str(case["case"]),
        "--field",
        str(case["field"]),
        "--metadata",
        str(case["metadata"]),
        "--output",
        str(case["root"] / name),
    ]


def reseal_fixture(case, descriptor, plan, selection=None, human=None):
    root = case["root"]
    if selection is not None:
        with gzip.open(root / "selected.json.gz", "wt") as stream:
            json.dump(selection, stream)
    if human is not None:
        (root / "human.json").write_text(json.dumps(human))
    (root / "plan.json").write_text(json.dumps(plan))
    artifacts = descriptor["artifacts"]
    for name in ("selection", "plan", "human", "afv"):
        if name in artifacts:
            artifacts[name]["sha256"] = sha256(root / artifacts[name]["path"])
    (root / "seal.json").write_text(
        json.dumps(dict(selected_sha256=artifacts["selection"]["sha256"], plan_sha256=artifacts["plan"]["sha256"]))
    )
    artifacts["seal"]["sha256"] = sha256(root / "seal.json")
    case["case"].write_text(json.dumps(descriptor))


def test_another_frame_scale_and_offset_preserve_physical_scores(case):
    descriptor = json.loads(case["case"].read_text())
    plan = json.loads((case["root"] / "plan.json").read_text())
    with gzip.open(case["root"] / "selected.json.gz", "rt") as stream:
        selection = json.load(stream)
    transform = dict(scale_xyz=[0.5, 1, 2], offset_xyz=[10, -8, 4])
    descriptor.update(
        schema="winding-benchmark-case-v2",
        frame="another-scroll/world",
        voxel_um=12,
        native_to_world=transform,
        population="all_and_reserved",
    )
    meta = json.loads(case["metadata"].read_text())
    meta.update(frame=descriptor["frame"], voxel_um=12, origin_xyz=[10, -8, 4], spacing_xyz=[0.8] * 3)
    case["metadata"].write_text(json.dumps(meta))
    for row in selection["whole_coarse"]:
        for endpoint in ("a_native_xyz", "b_native_xyz"):
            row[endpoint] = (np.array(row[endpoint]) * 0.2 / np.array(transform["scale_xyz"])).tolist()
    plan.update(frame=descriptor["frame"], native_voxel_mm=[0.006, 0.012, 0.024])
    plan["fields"]["whole_coarse"]["metadata"] = meta
    human = json.loads((case["root"] / "human.json").read_text())
    for row in human["methods"]["whole_coarse_secondary"]["rows"]:
        for endpoint in ("a_xyz", "b_xyz"):
            row[endpoint] = (np.array(row[endpoint]) * 0.8 + [10, -8, 4]).tolist()
        row["frame"] = descriptor["frame"]
    reseal_fixture(case, descriptor, plan, selection, human)
    assert evaluator().main(arguments(case)) == 0
    result = json.loads((case["root"] / "run/SUMMARY.json").read_text())
    candidate = result["methods"]["candidate"]
    assert candidate["all"][0]["covered"] == 2
    assert candidate["all"][0]["median"] == pytest.approx(1)
    assert candidate["human"]["correct"] == 1
    assert result["independent_interpolation"]["max_abs_difference"] < 2e-5


def test_absent_human_and_split_are_explicitly_not_evaluated(case):
    import sqlite3

    descriptor = json.loads(case["case"].read_text())
    descriptor.update(schema="winding-benchmark-case-v2", population="all", expected_human=0)
    for name in ("human", "split"):
        del descriptor["artifacts"][name]
    afv = case["root"] / "fibres.afv"
    afv.unlink()
    with sqlite3.connect(afv) as db:
        db.execute("CREATE TABLE fibers(id INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO fibers VALUES (?)", [(1,), (2,), (3,)])
    audit = case["root"] / "audit.json"
    audit.write_text(json.dumps(dict(afv_sha256=sha256(afv))))
    descriptor["artifacts"]["audit"]["sha256"] = sha256(audit)
    plan = json.loads((case["root"] / "plan.json").read_text())
    plan["constraint_audit_sha256"] = sha256(audit)
    reseal_fixture(case, descriptor, plan)
    assert evaluator().main(arguments(case)) == 0
    candidate = json.loads((case["root"] / "run/SUMMARY.json").read_text())["methods"]["candidate"]
    assert candidate["all"][0]["covered"] == 2
    assert candidate["reserved"] is None
    assert candidate["human"]["status"] == "NOT_EVALUATED" and candidate["human"]["eligible"] == 0
    assert "not evaluated" in (case["root"] / "run/TABLES.md").read_text().lower()


def test_known_answer_and_controls_keep_all_and_reserved_denominators(case):
    assert evaluator().main(arguments(case)) == 0
    result = json.loads((case["root"] / "run/SUMMARY.json").read_text())
    candidate = result["methods"]["candidate"]
    assert result["status"] == "PASS_EVALUATED"
    assert candidate["all"][0]["selected"] == 3 and candidate["all"][0]["covered"] == 2
    assert candidate["all"][0]["median"] == pytest.approx(1)
    assert candidate["reserved"][0]["selected"] == candidate["reserved"][0]["covered"] == 1
    assert candidate["human"]["correct"] == 1
    for name, error in [("constant_control", 0), ("compressed_control", 0.1)]:
        control = result["methods"][name]
        assert control["all"][0]["median"] == pytest.approx(error)
        assert control["all"][0]["covered"] == 2 and control["human"]["correct"] == 0
    assert result["independent_interpolation"]["max_abs_difference"] < 2e-5
    with np.load(case["root"] / "run/predictions.npz") as samples:
        np.testing.assert_allclose(samples["error"][:2], 1)
        assert np.isnan(samples["error"][2])
    manifest = json.loads((case["root"] / "run/MANIFEST.json").read_text())
    assert all(sha256(case["root"] / "run" / name) == digest for name, digest in manifest.items())


@pytest.mark.parametrize(
    "problem", ["missing", "tampered", "scale", "frame", "shape", "nonconverged", "support_required"]
)
def test_preflight_refuses_bad_inputs_without_a_success_report(case, problem):
    if problem in ("missing", "tampered"):
        path = case["root"] / "selected.json.gz"
        if problem == "missing":
            path.unlink()
        else:
            path.write_bytes(path.read_bytes() + b"changed")
    elif problem == "support_required":
        data = json.loads(case["case"].read_text())
        data["requires_support"] = True
        case["case"].write_text(json.dumps(data))
    else:
        meta = json.loads(case["metadata"].read_text())
        if problem == "scale":
            meta["step"] = 1.0
        elif problem == "frame":
            meta["frame"] = "another-scroll"
        elif problem == "shape":
            meta["shape_zyx"] = [6, 5, 80]
        else:
            meta["solver"]["converged"] = False
        case["metadata"].write_text(json.dumps(meta))
    assert evaluator().main(arguments(case)) == 2
    output = case["root"] / "run"
    assert (output / "ERROR.json").is_file() and not (output / "SUMMARY.json").exists()


def test_preflight_only_and_existing_output(case):
    m = evaluator()
    assert m.main(arguments(case) + ["--check-only"]) == 0
    output = case["root"] / "run"
    before = (output / "PREFLIGHT.json").read_bytes()
    assert not (output / "SUMMARY.json").exists()
    assert m.main(arguments(case)) == 2
    assert (output / "PREFLIGHT.json").read_bytes() == before


def test_case_cannot_rebind_a_selection_to_another_afv_audit(case):
    audit = case["root"] / "audit.json"
    audit.write_text(json.dumps(dict(afv_sha256="0" * 64)))
    descriptor = json.loads(case["case"].read_text())
    descriptor["artifacts"]["audit"]["sha256"] = sha256(audit)
    case["case"].write_text(json.dumps(descriptor))
    assert evaluator().main(arguments(case)) == 2
    assert not (case["root"] / "run/SUMMARY.json").exists()


def test_support_and_other_grid_resolutions_preserve_physical_sampling(case):
    field = np.broadcast_to(np.arange(160, dtype=float) / 60, (10, 12, 160)).copy()
    np.save(case["field"], field)
    meta = json.loads(case["metadata"].read_text())
    meta.update(shape_zyx=list(field.shape), spacing_xyz=[0.5] * 3, field_sha256=sha256(case["field"]))
    case["metadata"].write_text(json.dumps(meta))
    support = np.ones(field.shape, bool)
    support[4, 4, 2] = False
    support_path = case["root"] / "support.npy"
    np.save(support_path, support)
    meta["support_sha256"] = sha256(support_path)
    case["metadata"].write_text(json.dumps(meta))
    assert evaluator().main(arguments(case) + ["--support", str(support_path)]) == 0
    result = json.loads((case["root"] / "run/SUMMARY.json").read_text())
    assert result["methods"]["candidate"]["all"][0]["selected"] == 3
    assert result["methods"]["candidate"]["all"][0]["covered"] == 1
    assert result["methods"]["candidate"]["all"][0]["median"] == pytest.approx(1)


@pytest.mark.parametrize("problem", ["omitted", "changed"])
def test_declared_candidate_support_cannot_be_omitted_or_replaced(case, problem):
    support = np.ones((5, 6, 80), bool)
    support[2, 2, 1] = False
    path = case["root"] / "support.npy"
    np.save(path, support)
    meta = json.loads(case["metadata"].read_text())
    meta.update(support=path.name, support_sha256=sha256(path))
    case["metadata"].write_text(json.dumps(meta))
    args = arguments(case)
    if problem == "changed":
        np.save(path, np.ones(support.shape, bool))
        args += ["--support", str(path)]
    assert evaluator().main(args) == 2
    assert not (case["root"] / "run/SUMMARY.json").exists()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interpolation_verification_accounts_for_float32_output_rounding(case, dtype):
    # Same nodal values, a fractional sample, and a harmless gauge offset.
    field = np.broadcast_to((1000 + np.arange(80) / 30).astype(np.float32), (5, 6, 80)).astype(dtype)
    np.save(case["field"], field)
    meta = json.loads(case["metadata"].read_text())
    meta.update(origin_xyz=[-0.25, 0, 0], field_sha256=sha256(case["field"]))
    case["metadata"].write_text(json.dumps(meta))
    assert evaluator().main(arguments(case)) == 0


def test_independent_interpolation_still_rejects_real_disagreement(case, monkeypatch):
    m = evaluator()
    original = m.verify.interpolate
    monkeypatch.setattr(m.verify, "interpolate", lambda *args: original(*args) + 0.01)
    assert m.main(arguments(case)) == 2
    assert not (case["root"] / "run/SUMMARY.json").exists()


def test_cli_does_not_import_a_solver_or_torch(case):
    path = Path(__file__).with_name("evaluate.py")
    code = (
        "import runpy,sys; m=runpy.run_path(sys.argv[1]); "
        "status=m['main'](sys.argv[2:]); "
        "assert 'torch' not in sys.modules and 'field_solver' not in sys.modules; sys.exit(status)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(path), *arguments(case)],
        text=True,
        capture_output=True,
        cwd=case["root"],
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_numerical_benchmark_does_not_require_visual_archives(case):
    descriptor = json.loads(case["case"].read_text())
    descriptor["visualization"] = {"archive_manifest": {"path": "unavailable/MANIFEST.json"}}
    case["case"].write_text(json.dumps(descriptor))
    assert evaluator().main(arguments(case)) == 0
    output = case["root"] / "run"
    assert json.loads((output / "SUMMARY.json").read_text())["status"] == "PASS_EVALUATED"
    assert not list(output.glob("*.png"))
