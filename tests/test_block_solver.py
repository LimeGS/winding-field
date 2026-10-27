import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from constraint_templates import StencilConstraints
from field_prefetch import ObservationPrefetch
from field_runtime import ResourceUnavailable
from field_sources import NpyObservations
from field_store import BlockStore


def runner():
    assert importlib.util.find_spec("block_solver"), "integrated block runner is missing"
    from block_solver import solve_blocks

    return solve_blocks


def case(tmp_path, *, gradient=0.125):
    shape = (3, 4, 6)
    arrays = {
        "target": np.broadcast_to(np.array([gradient, 0.0, 0.0], np.float32), (*shape, 3)).copy(),
        "weights": np.ones(shape, np.float32),
        "observed": np.ones(shape, bool),
    }
    for key, value in arrays.items():
        np.save(tmp_path / (key + ".npy"), value)
    hashes = {k: hashlib.sha256((tmp_path / (k + ".npy")).read_bytes()).hexdigest() for k in arrays}
    load = NpyObservations(*(tmp_path / (k + ".npy") for k in arrays), hashes)
    origin, spacing = np.array([11.0, -4.0, 19.0]), np.array([2.0, 3.0, 5.0])
    constraints = StencilConstraints.pairs(
        origin + spacing * [[0.25, 1.0, 1.0]],
        origin + spacing * [[4.75, 2.0, 1.0]],
        origin,
        spacing,
        shape,
        targets=[-9 * gradient],
        weight=2.0,
        cache_bytes=10000,
    )
    return dict(
        shape=shape,
        chunk=4,
        identity={"test_case": "analytic", "inputs": hashes},
        load_observations=load,
        origin_xyz=origin,
        spacing_xyz=spacing,
        order=[[0, 0, 0], [0, 0, 1]],
        constraints=constraints,
        halo=2,
        sweeps=2,
        rtol=1e-10,
        maxiter=1000,
        cache_bytes=20000,
        read_workers=2,
    )


def state(root):
    metadata = json.loads((root / "CHECKPOINT.json").read_text())
    with BlockStore(root, metadata["shape"], metadata["chunk"], metadata["identity"], resume=True) as store:
        field = store.read([0] * 3, store.shape)
        observed = store.read_observed([0] * 3, store.shape)
    return metadata, field, observed


def test_anisotropic_cross_block_run_persists_only_cores_and_pre_encoding_receipt(tmp_path):
    options = case(tmp_path)
    result = runner()(tmp_path / "run", **options)
    checkpoint, field, observed = state(tmp_path / "run")
    truth = np.broadcast_to(np.arange(6) * 0.25, options["shape"])
    np.testing.assert_allclose(field, truth, atol=2e-7)
    assert observed.all()
    assert set(checkpoint["chunks"]) == {"0-0-0", "0-0-1"}
    assert result["status"] == "SWEEP_BUDGET_REACHED"
    assert result["global_converged"] is False and result["physical_acceptance"] is False
    assert checkpoint["progress"]["solver"]["scope"] == "pre_encoding_local_solve"
    assert result["stored_local_residual"] is None
    assert checkpoint["progress"]["next_block"] == 2
    assert len(result["completed_sweeps"]) == 2


def test_checkpoint_resume_reproduces_stored_bytes_support_and_sweep_maxima(tmp_path):
    options = case(tmp_path)
    uninterrupted = runner()(tmp_path / "full", **options)
    path = tmp_path / "split"

    def stop():
        if not (path / "CHECKPOINT.json").exists():
            return False
        progress = json.loads((path / "CHECKPOINT.json").read_text())["progress"]
        return progress.get("sweep") == 1 and progress.get("next_block") == 1

    stopped = runner()(path, stop=stop, **options)
    assert stopped["status"] == "CHECKPOINTED_STOP" and stopped["next_block"] == 1
    resumed = runner()(path, resume=True, **options)
    before, field, observed = state(tmp_path / "full")
    after, actual, actual_observed = state(path)
    np.testing.assert_array_equal(actual, field)
    np.testing.assert_array_equal(actual_observed, observed)
    assert before["progress"]["completed_sweeps"] == after["progress"]["completed_sweeps"]
    assert before["progress"]["source_receipts"] == after["progress"]["source_receipts"]
    assert {k: v["sha256"] for k, v in before["chunks"].items()} == {k: v["sha256"] for k, v in after["chunks"].items()}
    assert resumed["completed_sweeps"] == uninterrupted["completed_sweeps"]


@pytest.mark.parametrize(
    "change", ["identity", "origin_xyz", "spacing_xyz", "order", "rtol", "maxiter", "dtype", "sweeps"]
)
def test_resume_rejects_changed_problem_without_rewriting_checkpoint(tmp_path, change):
    options = case(tmp_path)
    root = tmp_path / "run"
    runner()(root, stop=lambda: True, **options)
    original = (root / "CHECKPOINT.json").read_bytes()
    altered = dict(options)
    altered.update(
        {
            change: {
                "identity": {"different": True},
                "origin_xyz": [0, 0, 0],
                "spacing_xyz": [1, 1, 1],
                "order": list(reversed(options["order"])),
                "rtol": 1e-7,
                "maxiter": 999,
                "dtype": "float32",
                "sweeps": 3,
            }[change]
        }
    )
    with pytest.raises(ValueError, match="identity"):
        runner()(root, resume=True, **altered)
    assert (root / "CHECKPOINT.json").read_bytes() == original


def test_cpu_prefetch_has_primary_parity_and_matches_sequential_run(tmp_path):
    options = case(tmp_path)
    runner()(tmp_path / "plain", **options)
    with ObservationPrefetch(options["load_observations"], options["shape"], max_bytes=100000) as prefetch:
        result = runner()(tmp_path / "prefetch", prefetch=prefetch, **options)
    np.testing.assert_array_equal(state(tmp_path / "plain")[1], state(tmp_path / "prefetch")[1])
    assert result["prefetch_parity"] is True and result["prefetched_blocks"] > 0


def test_expired_prefetch_falls_back_without_changing_values(tmp_path):
    options = case(tmp_path)
    with ObservationPrefetch(
        options["load_observations"], options["shape"], max_bytes=100000, deadline=1.0
    ) as prefetch:
        result = runner()(tmp_path / "run", prefetch=prefetch, **options)
    assert result["status"] == "SWEEP_BUDGET_REACHED" and result["prefetched_blocks"] == 0


def test_prefetched_changed_observations_fail_before_any_commit(tmp_path):
    options = case(tmp_path)
    original = options["load_observations"]

    def changed(lo, hi):
        target, weights, observed, receipts = original(lo, hi)
        target += 0.5
        return target, weights, observed, receipts

    with ObservationPrefetch(original, options["shape"], max_bytes=100000) as prefetch:
        options["load_observations"] = changed
        with pytest.raises(ValueError, match="parity"):
            runner()(tmp_path / "run", prefetch=prefetch, **options)
    checkpoint, field, _ = state(tmp_path / "run")
    assert checkpoint["progress"]["next_block"] == 0 and not field.any()


def test_resource_error_has_retryable_status_and_exact_initial_cursor(tmp_path):
    def admit():
        raise ResourceUnavailable("effective memory unavailable")

    with pytest.raises(ResourceUnavailable):
        runner()(tmp_path / "run", admit=admit, **case(tmp_path))
    progress = state(tmp_path / "run")[0]["progress"]
    assert progress["next_block"] == 0
    assert json.loads((tmp_path / "run/STATUS.json").read_text())["status"] == "RESOURCE_UNAVAILABLE"


def test_local_failure_is_terminal_and_preserves_last_committed_cursor(tmp_path):
    options = case(tmp_path)
    options["maxiter"] = 0
    with pytest.raises(RuntimeError, match="converge"):
        runner()(tmp_path / "run", **options)
    checkpoint, field, _ = state(tmp_path / "run")
    assert checkpoint["progress"]["next_block"] == 0 and not field.any()
    assert json.loads((tmp_path / "run/STATUS.json").read_text())["status"] == "FAILED"


def test_changed_repeated_receipt_is_not_silently_cached(tmp_path):
    options = case(tmp_path)
    original, calls = options["load_observations"], [0]

    def changed(lo, hi):
        result = original(lo, hi)
        calls[0] += 1
        if calls[0] > 2:
            result[3]["calibration"] = "f" * 64
        else:
            result[3]["calibration"] = "0" * 64
        return result

    options["load_observations"] = changed
    with pytest.raises(ValueError, match="source"):
        runner()(tmp_path / "run", **options)
    assert state(tmp_path / "run")[0]["progress"]["sweep"] == 1


def test_zero_field_retains_observed_and_unsupported_distinction(tmp_path):
    options = case(tmp_path, gradient=0.0)
    original = options["load_observations"]

    def loader(lo, hi):
        target, weights, observed, receipts = original(lo, hi)
        observed[0] = False
        return target, weights, observed, receipts

    options["load_observations"] = loader
    runner()(tmp_path / "run", **options)
    _, field, observed = state(tmp_path / "run")
    assert not field.any() and not observed[0].any() and observed[1:].all()


def test_external_anchors_and_stop_file_are_honored(tmp_path):
    options = case(tmp_path)

    def fixed(lo, hi):
        result = np.full(tuple(np.asarray(hi) - lo), np.nan)
        if tuple(lo) == (0, 0, 0):
            result[0, 0, 0] = 2.0
        return result

    result = runner()(tmp_path / "run", fixed=fixed, **options)
    np.testing.assert_allclose(
        state(tmp_path / "run")[1], 2.0 + np.broadcast_to(np.arange(6) * 0.25, options["shape"]), atol=2e-7
    )
    assert result["status"] == "SWEEP_BUDGET_REACHED"
    (tmp_path / "run/STOP").touch()
    assert runner()(tmp_path / "run", fixed=fixed, resume=True, **options)["status"] == "CHECKPOINTED_STOP"


def test_fresh_supervised_run_directory_is_accepted_without_overwriting_data(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / ".supervisor.lock").touch()
    result = runner()(root, **case(tmp_path))
    assert result["status"] == "SWEEP_BUDGET_REACHED"


def test_supervisor_accepts_realistic_checkpoint_metadata_above_one_mib(tmp_path):
    from field_runtime import _checkpoint

    checkpoint = {
        "identity": {"input": "f" * 64},
        "progress": {"sweep": 1, "next_block": 7},
        "chunks": {str(n): {"sha256": "a" * 64, "path": str(n) + ".npz", "bytes": 1000} for n in range(12000)},
    }
    path = tmp_path / "CHECKPOINT.json"
    path.write_text(json.dumps(checkpoint))
    assert path.stat().st_size > 1024**2 and _checkpoint(path)


@pytest.mark.parametrize("extra", ["data.npy", ".writer.lock", "CHECKPOINT.json"])
def test_supervisor_marker_does_not_authorize_overwriting_existing_files(tmp_path, extra):
    root = tmp_path / "run"
    root.mkdir()
    (root / ".supervisor.lock").touch()
    (root / extra).write_bytes(b"preserve me")
    with pytest.raises(FileExistsError):
        runner()(root, **case(tmp_path))
    assert (root / extra).read_bytes() == b"preserve me"


@pytest.mark.parametrize(
    "override",
    [
        {"shape": [True, 4, 6]},
        {"chunk": False},
        {"halo": -1},
        {"sweeps": 0},
        {"rtol": np.nan},
        {"maxiter": True},
        {"order": [[0, 0, 0], [0, 0, 0]]},
        {"order": [[0, 0, 0]]},
        {"identity": {}},
        {"dtype": "float16"},
    ],
)
def test_invalid_run_configuration_fails_before_output(tmp_path, override):
    options = case(tmp_path)
    options.update(override)
    with pytest.raises(ValueError):
        runner()(tmp_path / "run", **options)
    assert not (tmp_path / "run").exists()


def test_supervisor_runs_real_block_worker_and_resumes_resource_failure(tmp_path, monkeypatch):
    from field_runtime import run_supervised

    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project / "src"), str(project / "tests")]))
    script = tmp_path / "worker.py"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from test_block_solver import case, runner\nfrom field_runtime import ResourceUnavailable\n"
        "home=Path(sys.argv[1]); options=case(home)\n"
        "def admit():\n"
        "    flag=home/'resource-available'\n"
        "    if not flag.exists():\n"
        "        flag.write_text('next attempt')\n"
        "        raise ResourceUnavailable('synthetic resource wait')\n"
        "try:\n"
        "    runner()(home/'run',resume='--resume' in sys.argv,admit=admit,**options)\n"
        "except ResourceUnavailable:\n"
        "    sys.exit(75)\n"
    )
    command = [sys.executable, str(script), str(tmp_path)]
    result = run_supervised(
        command,
        [*command, "--resume"],
        tmp_path / "run",
        tmp_path / "service",
        source_sha256={str(script): hashlib.sha256(script.read_bytes()).hexdigest()},
        retry_seconds=0.01,
        stop_grace_seconds=1.0,
    )
    assert result == 0, (tmp_path / "service/worker.log").read_text()
    events = [json.loads(line)["status"] for line in (tmp_path / "service/service.log").read_text().splitlines()]
    assert events.count("RETRYING") == 1 and events[-1] == "SWEEP_BUDGET_REACHED"
    metadata, field, _ = state(tmp_path / "run")
    assert metadata["progress"]["sweep"] == 2 and metadata["progress"]["next_block"] == 2
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


def test_zero_halo_cannot_permanently_fix_owned_internal_faces(tmp_path):
    options = case(tmp_path)
    options["halo"] = 0
    with pytest.raises(ValueError, match="halo"):
        runner()(tmp_path / "run", **options)
    assert not (tmp_path / "run").exists()


def test_single_core_needs_no_halo(tmp_path):
    options = case(tmp_path)
    options.update(chunk=8, order=[[0, 0, 0]], halo=0)
    result = runner()(tmp_path / "run", **options)
    assert result["status"] == "SWEEP_BUDGET_REACHED"
    field = state(tmp_path / "run")[1]
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


def test_callable_constraints_allow_block_selected_geometry(tmp_path):
    options = case(tmp_path)
    compiled, calls = options["constraints"], []

    def selected(lo, hi, read_nodes):
        calls.append((tuple(lo), tuple(hi)))
        return compiled.restrict(lo, hi, read_nodes)

    options["constraints"] = selected
    options["identity"]["constraint_provider"] = {"geometry": compiled.digest, "recipe": "selected complete groups"}
    result = runner()(tmp_path / "run", **options)
    assert result["status"] == "SWEEP_BUDGET_REACHED" and len(calls) == 4
    field = state(tmp_path / "run")[1]
    np.testing.assert_allclose(field, np.broadcast_to(np.arange(6) * 0.25, field.shape), atol=2e-7)


def test_callable_constraints_reject_invalid_local_rows_before_commit(tmp_path):
    options = case(tmp_path)
    options["constraints"] = lambda lo, hi, read_nodes: np.zeros((2, 2))
    with pytest.raises(ValueError, match="constraint"):
        runner()(tmp_path / "run", **options)
    assert state(tmp_path / "run")[0]["progress"]["next_block"] == 0
