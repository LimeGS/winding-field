"""Small contract checks; throughput measurements remain opt-in."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks/performance/run.py"


def harness():
    assert HARNESS.exists(), "opt-in performance harness is missing"
    spec = importlib.util.spec_from_file_location("performance_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_keeps_raw_samples_and_robust_statistics():
    result = harness().summarize([5, 1, 3, 2, 4])
    assert result == dict(samples_s=[5, 1, 3, 2, 4], median_s=3, iqr_s=2, min_s=1, max_s=5)
    with pytest.raises(ValueError):
        harness().summarize([])


def test_input_hash_binds_seed_shape_dtype_and_contents():
    api = harness()
    import numpy as np

    a = np.random.default_rng(73).normal(size=(2, 3))
    b = np.random.default_rng(73).normal(size=(2, 3))
    assert api.array_hash(a) == api.array_hash(b)
    assert api.array_hash(a) != api.array_hash(a.reshape(3, 2))
    assert api.array_hash(a) != api.array_hash(a.astype(np.float32))
    b[0, 0] += 1
    assert api.array_hash(a) != api.array_hash(b)


def test_source_hash_detects_dirty_source(tmp_path):
    api = harness()
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/example.py"
    source.write_text("value = 1\n")
    first = api.source_identity(tmp_path, "frozen-commit")
    source.write_text("value = 2\n")
    second = api.source_identity(tmp_path, "frozen-commit")
    assert first["commit"] == second["commit"] == "frozen-commit"
    assert first["files"] != second["files"]
    assert first["source_sha256"] != second["source_sha256"]


def test_requested_workloads_are_available():
    api = harness()
    assert {"mesh-offrange", "mesh-large", "arrival-0", "arrival-70000", "arrival-120000", "arrival-1000000"} <= set(
        api.CASES
    )
    assert "pcg-large" in api.CASES
    assert "blocks-constraints" in api.CASES


def test_cpu_metadata_explicitly_limits_cuda_claims():
    info = harness().environment()
    assert info["cuda_validation"] is False
    assert info["cuda_driver"] == "not queried (CPU-only harness)"
    assert "cuda_runtime" in info


def test_offrange_mesh_workload_uses_supported_public_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location("performance_cases", HARNESS.with_name("cases.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, inputs, operation, _ = module.make_case("mesh-offrange", tmp_path, 73, 0)
    assert inputs["values"].shape == (64, 64, 64)
    assert len(config["levels"]) == 32
    result = operation()
    assert result["status"] == "MESH_READY"
    assert all(level["faces"] == 0 for level in result["levels"])


def test_isolated_cpu_worker_repeats_inputs_and_reports_metadata(tmp_path):
    harness()
    results = []
    frozen = tmp_path / "snapshot"
    shutil.copytree(ROOT / "src", frozen / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for i in range(2):
        output = tmp_path / str(i)
        subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--source",
                str(ROOT if i == 0 else frozen),
                "--output",
                str(output),
                "--cases",
                "pcg",
                *(["mesh", "mesh-offrange"] if i == 0 else []),
                "--repetitions",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        results.append(json.loads((output / "pcg.json").read_text()))
    a, b = results
    assert a["input_hashes"] == b["input_hashes"]
    assert a["config_sha256"] == b["config_sha256"]
    assert a["validation"] == b["validation"]
    assert len(a["timing"]["samples_s"]) == 2
    assert a["cold_s"] >= 0 and a["warmup_s"] >= 0
    assert a["environment"]["device"] == "cpu"
    assert set(a["environment"]["versions"]) >= {"python", "numpy", "scipy", "torch", "skimage"}
    assert a["memory"]["peak_rss_bytes"] > 0
    assert a["memory"]["tracemalloc_peak_bytes"] > 0
    assert a["stages"]["non_additive"] is True
    assert a["source"]["files"]["src/field_solver.py"]
    assert a["source"]["root"] == str(ROOT)
    assert b["source"]["root"] == str(frozen)
    assert a["imported_sources"]["field_solver"].startswith(str(tmp_path / "0/source/src"))
    assert b["imported_sources"]["field_solver"].startswith(str(tmp_path / "1/source/src"))
    assert a["stages"]["calls_s"]["pcg.iterative_envelope_including_norms"]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [sys.executable, str(HARNESS), "--source", str(ROOT), "--output", str(output)],
            check=True,
            capture_output=True,
        )
