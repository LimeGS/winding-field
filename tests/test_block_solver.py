import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_continue_policy_commits_budget_exhausted_blocks_and_counts_them(tmp_path):
    options = case(tmp_path)
    options["maxiter"] = 0
    with pytest.raises(ValueError, match="nonconvergence_policy"):
        runner()(tmp_path / "invalid", nonconvergence_policy="ignore", **options)
    assert not (tmp_path / "invalid").exists()
    result = runner()(tmp_path / "run", nonconvergence_policy="continue", **options)
    checkpoint, field, observed = state(tmp_path / "run")
    assert result["status"] == "SWEEP_BUDGET_REACHED" and len(result["completed_sweeps"]) == 2
    assert result["nonconverged_solves"] == checkpoint["progress"]["nonconverged_solves"] == 4
    assert result["local_solves_converged"] is False and result["nonconvergence_policy"] == "continue"
    assert checkpoint["progress"]["solver"]["converged"] is False
    assert checkpoint["progress"]["solver"]["reason"] == "maxiter"
    assert np.isfinite(field).all() and observed.all()
    assert result["global_converged"] is False and result["physical_acceptance"] is False


def test_converged_run_reports_zero_nonconverged_solves(tmp_path):
    result = runner()(tmp_path / "run", nonconvergence_policy="continue", **case(tmp_path))
    assert result["nonconverged_solves"] == 0 and result["local_solves_converged"] is True
    assert state(tmp_path / "run")[0]["progress"]["nonconverged_solves"] == 0


def test_continue_policy_still_fails_on_numerical_breakdown(tmp_path, monkeypatch):
    import block_solver

    original = block_solver.solve_field

    def breakdown(*args, **kwargs):
        field, info = original(*args, **kwargs)
        return field, dict(info, converged=False, reason="nonpositive_curvature")

    monkeypatch.setattr(block_solver, "solve_field", breakdown)
    with pytest.raises(RuntimeError, match="converge"):
        runner()(tmp_path / "run", nonconvergence_policy="continue", **case(tmp_path))
    assert state(tmp_path / "run")[0]["progress"]["next_block"] == 0


def test_nonconverged_count_survives_resume_and_policy_may_change(tmp_path):
    options = case(tmp_path)
    options["maxiter"] = 0
    paused = runner()(
        tmp_path / "run",
        nonconvergence_policy="continue",
        stop=lambda: (
            (tmp_path / "run/CHECKPOINT.json").exists()
            and json.loads((tmp_path / "run/CHECKPOINT.json").read_text())["progress"]["next_block"] == 1
        ),
        **options,
    )
    assert paused["status"] == "CHECKPOINTED_STOP" and paused["nonconverged_solves"] == 1
    resumed = runner()(tmp_path / "run", nonconvergence_policy="continue", resume=True, **options)
    assert resumed["nonconverged_solves"] == 4 and resumed["local_solves_converged"] is False
    options["maxiter"] = 1000
    with pytest.raises(ValueError):
        # The numerical problem (maxiter) is part of the identity; only the policy may change.
        runner()(tmp_path / "run", nonconvergence_policy="fail", resume=True, **options)


def test_legacy_progress_without_counter_resumes_as_zero(tmp_path):
    options = case(tmp_path)
    paused = runner()(
        tmp_path / "run",
        stop=lambda: (
            (tmp_path / "run/CHECKPOINT.json").exists()
            and json.loads((tmp_path / "run/CHECKPOINT.json").read_text())["progress"]["next_block"] == 1
        ),
        **options,
    )
    assert paused["status"] == "CHECKPOINTED_STOP"
    metadata = json.loads((tmp_path / "run/CHECKPOINT.json").read_text())
    with BlockStore(tmp_path / "run", metadata["shape"], metadata["chunk"], metadata["identity"], resume=True) as store:
        progress = dict(store.manifest["progress"])
        progress.pop("nonconverged_solves")
        store.save_progress(progress)
    resumed = runner()(tmp_path / "run", resume=True, **options)
    assert resumed["nonconverged_solves"] == 0 and resumed["status"] == "SWEEP_BUDGET_REACHED"


def test_partial_order_solves_only_owned_chunks_against_fixed_neighbours(tmp_path):
    options = dict(case(tmp_path), constraints=None, order=[[0, 0, 1]], partial_order=True)
    truth = 2.0 + np.broadcast_to(np.arange(6) * 0.25, options["shape"])

    def fixed(lo, hi):
        # Unowned chunk 0-0-0 (x < 4) is held at its known values; owned nodes are free.
        result = np.full(tuple(np.asarray(hi) - lo), np.nan)
        x = np.arange(lo[2], hi[2])
        result[..., x < 4] = truth[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]][..., x < 4]
        return result

    result = runner()(tmp_path / "run", fixed=fixed, **options)
    assert result["status"] == "SWEEP_BUDGET_REACHED"
    checkpoint, field, observed = state(tmp_path / "run")
    assert set(checkpoint["chunks"]) == {"0-0-1"}
    assert checkpoint["identity"]["order_scope"] == "partial"
    np.testing.assert_allclose(field[..., 4:], truth[..., 4:], atol=2e-7)
    assert not observed[..., :4].any() and observed[..., 4:].all()


def test_partial_order_requires_fixed_values_and_full_order_identity_is_unchanged(tmp_path):
    options = dict(case(tmp_path), order=[[0, 0, 1]], partial_order=True)
    with pytest.raises(ValueError, match="fixed provider"):
        runner()(tmp_path / "partial", **options)
    with pytest.raises(ValueError, match="each owned chunk"):
        runner()(tmp_path / "short", **dict(options, partial_order=False))
    runner()(tmp_path / "full", **case(tmp_path))
    assert "order_scope" not in json.loads((tmp_path / "full/CHECKPOINT.json").read_text())["identity"]


def test_amg_block_run_matches_jacobi_and_is_part_of_identity(tmp_path):
    runner()(tmp_path / "jacobi", **case(tmp_path))
    result = runner()(tmp_path / "amg", **dict(case(tmp_path), preconditioner="amg", prepared_cache_bytes=10**8))
    assert result["local_solves_converged"] is True
    jacobi_meta, jacobi_field, observed = state(tmp_path / "jacobi")
    amg_meta, amg_field, amg_observed = state(tmp_path / "amg")
    np.testing.assert_allclose(amg_field, jacobi_field, atol=1e-6)
    np.testing.assert_array_equal(amg_observed, observed)
    assert "preconditioner" not in jacobi_meta["identity"]
    assert amg_meta["identity"]["preconditioner"] == "amg"
    with pytest.raises(ValueError):
        runner()(tmp_path / "amg", resume=True, **case(tmp_path))


def test_spectral_block_runs_are_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="'jacobi' or 'amg'"):
        runner()(tmp_path / "spectral", **dict(case(tmp_path), preconditioner="spectral"))


def test_fused_block_run_matches_eager_and_is_part_of_identity(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eager = runner()(tmp_path / "eager", **dict(case(tmp_path), device=device))
    result = runner()(tmp_path / "fused", **dict(case(tmp_path), kernel="fused", device=device))
    assert result["local_solves_converged"] is True
    assert set(result["solver_kernels"]) == {"fused-v2"} and "solver_kernels" not in eager
    eager_meta, eager_field, observed = state(tmp_path / "eager")
    fused_meta, fused_field, fused_observed = state(tmp_path / "fused")
    np.testing.assert_allclose(fused_field, eager_field, atol=1e-6)
    np.testing.assert_array_equal(fused_observed, observed)
    assert "kernel" not in eager_meta["identity"]
    assert fused_meta["identity"]["kernel"] == "fused"
    with pytest.raises(ValueError):
        runner()(tmp_path / "fused", resume=True, **dict(case(tmp_path), device=device))
    with pytest.raises(ValueError, match="kernel must be eager or fused"):
        runner()(tmp_path / "bad", **dict(case(tmp_path), kernel="triton"))
