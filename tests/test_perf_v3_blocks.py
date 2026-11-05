import json
import os

import numpy as np
import pytest
from test_block_solver import case, state

import block_solver
from block_solver import _decoded_anchor_error, solve_blocks
from constraint_templates import StencilConstraints
from field_solver import PreparedField


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("offset", [0, 1e12])
@pytest.mark.parametrize("anchors", [False, True])
def test_anchor_error_matches_full_core_codec(dtype, strided, offset, anchors):
    rng = np.random.default_rng(43)
    core = (rng.normal(size=(7, 8, 9)) + offset).astype(dtype)
    if strided:
        core = core[::-2, ::2, ::-2]
    fixed = np.full(core.shape, np.nan)
    if anchors:
        fixed[::2, ::2, ::2] = core[::2, ::2, ::2] + 0.125
    values = core.astype(np.float64, copy=False)
    mean = values.mean()
    restored = (values - mean).astype(np.float32).astype(np.float64) + mean
    mask = np.isfinite(fixed)
    expected = float(np.max(abs(restored[mask] - fixed[mask]), initial=0))
    assert _decoded_anchor_error(core, fixed) == expected


@pytest.mark.parametrize("budget,expected_snapshots,skipped", [(4096, 4, 2), (8192, 2, 4)])
def test_denied_keys_bound_metadata_and_skip_only_snapshots(tmp_path, monkeypatch, budget, expected_snapshots, skipped):
    options = case(tmp_path)
    options["sweeps"] = 3
    snapshots = []
    original = PreparedField.bind

    def bind(self, *args):
        snapshots.append(1)
        return original(self, *args)

    monkeypatch.setattr(PreparedField, "bind", bind)
    result = solve_blocks(tmp_path / "denied", prepared_cache_bytes=budget, **options)
    stats = result["prepared_cache_stats"]
    assert len(snapshots) == expected_snapshots
    assert stats["builds"] == 6 and stats["hits"] == 0
    assert stats["skipped_snapshots"] == skipped
    assert stats["peak_denied_entries"] == budget // 4096
    assert stats["peak_denied_bytes"] == stats["peak_bytes"] == budget
    assert stats["denied_entries"] == stats["denied_bytes"] == stats["retained_bytes"] == 0
    solve_blocks(tmp_path / "cold", **options)
    a, field, support = state(tmp_path / "denied")
    b, cold, cold_support = state(tmp_path / "cold")
    np.testing.assert_array_equal(field, cold)
    np.testing.assert_array_equal(support, cold_support)
    assert a["progress"]["completed_sweeps"] == b["progress"]["completed_sweeps"]
    assert {k: v["sha256"] for k, v in a["chunks"].items()} == {k: v["sha256"] for k, v in b["chunks"].items()}


@pytest.mark.parametrize("change", ["target", "weights", "receipts", "nonfinite", "constraint"])
def test_denied_revisit_is_cold_and_still_validates(tmp_path, monkeypatch, change):
    options = case(tmp_path)
    original_load = options["load_observations"]
    visits = []
    original_solve = block_solver.solve_field
    calls = 0

    def load(lo, hi):
        nonlocal calls
        calls += 1
        values = original_load(lo, hi)
        if calls == 3:
            if change == "receipts":
                values[3][next(iter(values[3]))] = "0" * 64
            elif change != "constraint":
                array = values[1] if change == "weights" else values[0]
                array.flat[0] += np.nan if change == "nonfinite" else 0.01
        return values

    def solve(*args, **kwargs):
        visits.append(kwargs["prepared"] is None)
        assert kwargs["preconditioner"] == "jacobi"
        return original_solve(*args, **kwargs)

    options["load_observations"] = load
    monkeypatch.setattr(block_solver, "solve_field", solve)
    original_restrict = StencilConstraints.restrict

    def restrict(self, *args):
        rows = original_restrict(self, *args)
        if change == "constraint" and calls == 3:
            rows.matrix.data.setflags(write=True)
            rows.matrix.data.flat[0] = np.nan
        return rows

    monkeypatch.setattr(StencilConstraints, "restrict", restrict)
    if change in ("receipts", "nonfinite", "constraint"):
        with pytest.raises(ValueError, match="changed|finite"):
            solve_blocks(tmp_path / "run", prepared_cache_bytes=8192, **options)
        status = json.loads((tmp_path / "run/STATUS.json").read_text())
        assert status["status"] == "FAILED"
        assert status["prepared_cache_stats"]["denied_entries"] == 0
    else:
        result = solve_blocks(tmp_path / "run", prepared_cache_bytes=8192, **options)
        assert visits == [False, False, True, True]
        assert result["prepared_cache_stats"]["skipped_snapshots"] == 2
        calls = 0
        solve_blocks(tmp_path / "cold", **options)
        np.testing.assert_array_equal(state(tmp_path / "run")[1], state(tmp_path / "cold")[1])


def test_status_publication_counts_and_each_checkpoint_parity(tmp_path, monkeypatch):
    options = case(tmp_path)
    publications, commits = {}, {}
    original_json = block_solver.atomic_json
    original_commit = block_solver.BlockStore.commit

    def publish(path, value):
        publications.setdefault(path.parent.name, []).append((value["status"], value.get("phase")))
        assert value["pid"] == os.getpid()
        return original_json(path, value)

    def commit(self, *args, **kwargs):
        result = original_commit(self, *args, **kwargs)
        manifest = self.manifest
        commits.setdefault(self.root.name, []).append(
            ({k: v["sha256"] for k, v in manifest["chunks"].items()}, self.read([0, 0, 0], self.shape))
        )
        return result

    monkeypatch.setattr(block_solver, "atomic_json", publish)
    monkeypatch.setattr(block_solver.BlockStore, "commit", commit)
    solve_blocks(tmp_path / "all", **options)
    solve_blocks(tmp_path / "checkpoints", status_policy="checkpoints", **options)
    assert publications["all"] == [
        *([("RUNNING", "observations"), ("RUNNING", "solving"), ("RUNNING", "checkpointed")] * 4),
        ("SWEEP_BUDGET_REACHED", None),
    ]
    assert publications["checkpoints"] == [
        ("RUNNING", "observations"),
        *([("RUNNING", "checkpointed")] * 4),
        ("SWEEP_BUDGET_REACHED", None),
    ]
    for (a, field), (b, actual) in zip(commits["all"], commits["checkpoints"], strict=True):
        assert a == b
        np.testing.assert_array_equal(field, actual)


def test_denied_decisions_are_released_on_stop_and_resume_starts_cold(tmp_path):
    options = case(tmp_path)
    root = tmp_path / "split"

    def stop():
        checkpoint = json.loads((root / "CHECKPOINT.json").read_text())
        return checkpoint["progress"].get("next_block") == 2

    stopped = solve_blocks(root, prepared_cache_bytes=8192, stop=stop, status_policy="checkpoints", **options)
    stats = stopped["prepared_cache_stats"]
    assert stopped["status"] == "CHECKPOINTED_STOP"
    assert stats["peak_denied_entries"] == 2 and stats["denied_entries"] == stats["denied_bytes"] == 0
    resumed = solve_blocks(root, prepared_cache_bytes=8192, resume=True, status_policy="all", **options)
    assert resumed["prepared_cache_stats"]["skipped_snapshots"] == 0
    assert resumed["prepared_cache_stats"]["builds"] == 2
    solve_blocks(tmp_path / "full", **options)
    np.testing.assert_array_equal(state(root)[1], state(tmp_path / "full")[1])


@pytest.mark.parametrize("failure", ["solve", "resource"])
def test_checkpoints_policy_always_publishes_terminal_error(tmp_path, failure):
    options = case(tmp_path)
    if failure == "solve":
        options["maxiter"] = 0
    else:
        from field_runtime import ResourceUnavailable

        def admit():
            raise ResourceUnavailable("test resource failure")

        options["admit"] = admit
    with pytest.raises(RuntimeError):
        solve_blocks(tmp_path / "run", status_policy="checkpoints", **options)
    status = json.loads((tmp_path / "run/STATUS.json").read_text())
    assert status["status"] == ("FAILED" if failure == "solve" else "RESOURCE_UNAVAILABLE")
    assert status["next_block"] == 0


@pytest.mark.parametrize("policy", ["all", "checkpoints"])
def test_first_running_is_published_before_first_solve_and_resume(tmp_path, monkeypatch, policy):
    options = case(tmp_path)
    root = tmp_path / "run"
    original = options["load_observations"]

    def load(lo, hi):
        status = json.loads((root / "STATUS.json").read_text())
        assert status["status"] == "RUNNING" and status["phase"] == "observations"
        assert status["pid"] == os.getpid()
        (root / "STOP").touch()
        return original(lo, hi)

    options["load_observations"] = load
    for resume in (False, True):
        if resume:
            (root / "STOP").unlink()
        result = solve_blocks(root, status_policy=policy, resume=resume, **options)
        assert result["status"] == "CHECKPOINTED_STOP" and result["next_block"] == 0
        assert json.loads((root / "STATUS.json").read_text())["status"] == "CHECKPOINTED_STOP"


@pytest.mark.parametrize(
    "setting,value",
    [("preconditioner", "spectral"), ("preconditioner", "bad"), ("preconditioner", None), ("status_policy", "bad")],
)
def test_invalid_policies_fail_before_output(tmp_path, setting, value):
    with pytest.raises(ValueError, match="preconditioner|status_policy"):
        solve_blocks(tmp_path / "run", **{setting: value}, **case(tmp_path))
    assert not (tmp_path / "run").exists()


def test_legacy_identity_implies_jacobi_without_rewriting_checkpoint(tmp_path):
    options = case(tmp_path)
    root = tmp_path / "legacy"
    solve_blocks(root, stop=lambda: True, **options)
    path = root / "CHECKPOINT.json"
    checkpoint = json.loads(path.read_text())
    assert "preconditioner" not in checkpoint["identity"]
    before = path.read_bytes()
    solve_blocks(root, resume=True, stop=lambda: True, status_policy="checkpoints", **options)
    assert path.read_bytes() == before
    altered = dict(options, identity={"different": True})
    with pytest.raises(ValueError, match="identity"):
        solve_blocks(root, resume=True, **altered)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="preconditioner"):
        solve_blocks(root, resume=True, preconditioner="spectral", **options)
    assert path.read_bytes() == before
    solve_blocks(root, resume=True, **options)
    resumed = json.loads(path.read_text())
    assert resumed["identity"] == checkpoint["identity"]
    assert resumed["progress"]["solver"]["preconditioner"] == "jacobi"
