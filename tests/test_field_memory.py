"""Write-back field checkpoints and the host-memory tier plan."""

import json

import numpy as np
import pytest
from test_block_coarse import field, problem, run

from field_memory import field_bytes, observation_bytes, plan_memory
from field_store import BlockStore

GIB = 1 << 30


def test_write_back_publishes_only_on_flush_and_reads_its_own_commits(tmp_path):
    root = tmp_path / "store"
    with BlockStore(root, (8, 8, 8), 4, {"t": 1}, cache_bytes=10**7, write_back=True) as store:
        values = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
        receipt = store.commit([0, 0, 0], values, {"n": 1}, values > 10)
        np.testing.assert_allclose(store.read([0, 0, 0], [4, 4, 4]), values, atol=1e-5)
        assert store.read_observed([0, 0, 0], [4, 4, 4]).sum() == 53
        assert store.pending_chunks == 1 and receipt["quantization_max"] < 1e-5
        on_disk = json.loads((root / "CHECKPOINT.json").read_text())
        assert on_disk["chunks"] == {} and on_disk["progress"] == {}
        store.commit([1, 0, 0], np.zeros((4, 4, 4)), {"n": 2})  # Zero without support: no file.
        store.flush(wait=False)
        store.commit([0, 1, 0], np.ones((4, 4, 4)), {"n": 3})
        store.flush(wait=True)
        assert store.flush_stats["flushes"] == 2 and store.pending_chunks == 0
    with BlockStore(root, (8, 8, 8), 4, {"t": 1}, resume=True) as reopened:
        assert reopened.manifest["progress"] == {"n": 3}
        assert set(reopened.manifest["chunks"]) == {"0-0-0", "0-1-0"}
        np.testing.assert_allclose(reopened.read([0, 0, 0], [4, 4, 4]), values, atol=1e-5)
        assert reopened.read_observed([0, 0, 0], [4, 4, 4]).sum() == 53


def test_an_unflushed_crash_keeps_the_previous_checkpoint(tmp_path):
    root = tmp_path / "store"
    store = BlockStore(root, (8, 8, 8), 4, {"t": 1}, write_back=True)
    store.commit([0, 0, 0], np.full((4, 4, 4), 2.0), {"n": 1})
    store.flush()
    store.commit([0, 0, 0], np.full((4, 4, 4), 5.0), {"n": 2})
    store.close(flush=False)  # What a killed process leaves behind.
    with BlockStore(root, (8, 8, 8), 4, {"t": 1}, resume=True) as reopened:
        assert reopened.manifest["progress"] == {"n": 1}
        np.testing.assert_allclose(reopened.read([0, 0, 0], [4, 4, 4]), 2.0)
    assert len(list(root.glob("*.npz"))) == 1  # The replaced generation was removed.


def test_write_back_rejects_prepare_observers(tmp_path):
    with BlockStore(tmp_path / "s", (4, 4, 4), 4, {"t": 1}, write_back=True) as store:
        with pytest.raises(ValueError, match="write_back"):
            store.commit([0, 0, 0], np.ones((4, 4, 4)), {}, prepare=lambda event: None)


@pytest.mark.parametrize("coarse", [None, 4])
def test_sweep_checkpoints_give_the_same_field_and_resume(tmp_path, coarse):
    base = problem(tmp_path, pairs=coarse is not None)
    options = dict(halo=2, sweeps=3, cache_bytes=10**8)
    if coarse:
        options["coarse_factor"] = coarse
    block = run(tmp_path / "block", base, 8, **options)
    report = {"tier": "field", "available_bytes": 123}
    sweep = run(tmp_path / "sweep", base, 8, checkpoint="sweep", memory_report=report, **options)
    assert block["status"] == sweep["status"] == "SWEEP_BUDGET_REACHED"
    expected, meta = field(tmp_path / "block")
    np.testing.assert_array_equal(field(tmp_path / "sweep")[0], expected)
    assert field(tmp_path / "sweep")[1]["identity"] == meta["identity"]
    status = json.loads((tmp_path / "sweep/STATUS.json").read_text())
    assert status["memory"] == report
    assert status["checkpoint_policy"]["policy"] == "sweep" and status["checkpoint_policy"]["pending_chunks"] == 0
    assert status["checkpoint_policy"]["flushes"] >= 3
    assert json.loads((tmp_path / "block/STATUS.json").read_text())["checkpoint_policy"]["flushes"] == 0

    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] in (7, 30)

    root = tmp_path / "stopped"
    result = run(root, base, 8, checkpoint="sweep", stop=stop, **options)
    stops = 0
    while result["status"] == "CHECKPOINTED_STOP":
        stops += 1
        # Switching the checkpoint policy on resume is allowed: it is not part of the identity.
        policy = "block" if stops % 2 else "sweep"
        result = run(root, base, 8, checkpoint=policy, stop=stop, resume=True, **options)
    assert stops == 2 and result["status"] == "SWEEP_BUDGET_REACHED"
    np.testing.assert_array_equal(field(root)[0], expected)


def test_sweep_checkpoints_refuse_an_observer(tmp_path):
    class Observer:
        def restore(self, event):
            return None

        prepare = complete = restore

    with pytest.raises(ValueError, match="observer"):
        run(tmp_path / "x", problem(tmp_path, pairs=False), 8, halo=2, checkpoint="sweep", observer=Observer())


def test_memory_sizes():
    assert field_bytes((64, 64, 64), 64) == 4 * 64**3 + 64**3 // 8 + 512
    # One block: its crop is the whole grid. Two blocks along z share a 2 x halo overlap.
    assert observation_bytes((64, 64, 64), 64, 8) == 17 * 64**3 + 4096
    assert observation_bytes((128, 64, 64), 64, 8) == 17 * (72 + 72) * 64 * 64 + 2 * 4096


@pytest.mark.parametrize(
    "available, tier, warned",
    [(2 * GIB, "disk", True), (12 * GIB, "field", False), (60 * GIB, "all", False)],
)
def test_auto_tier_follows_available_memory(available, tier, warned):
    shape = (1000, 1000, 1000)  # ~4.1 GiB of field, ~19 GiB of observation crops
    plan = plan_memory(shape, 64, 8, available=available)
    assert plan["tier"] == tier and (plan["warning"] is not None) == warned
    assert plan["checkpoint"] == ("block" if tier == "disk" else "sweep")
    if tier == "disk":
        assert plan["field_cache_bytes"] == plan["observation_cache_bytes"] == 0
    if tier == "field":
        assert plan["field_cache_bytes"] == plan["field_bytes"]
        assert 0 < plan["observation_cache_bytes"] < plan["observation_bytes"]
        assert 0 < plan["observations_cached_fraction"] < 1
    if tier == "all":
        assert plan["observation_cache_bytes"] == plan["observation_bytes"]
    assert plan["usable_bytes"] == available - max(4 * GIB, int(0.15 * available)) or tier == "disk"
    json.dumps(plan, allow_nan=False)


def test_forced_and_unknown_memory():
    forced = plan_memory((1000, 1000, 1000), 64, 8, tier="all", available=8 * GIB)
    assert forced["tier"] == "all" and "forced" in forced["warning"]
    plan = plan_memory((64, 64, 64), 64, 8, available=None, tier="auto")
    # Measured here, so it fits; the explicit None means "measure".
    assert plan["source"] != "given"
    with pytest.raises(ValueError):
        plan_memory((64, 64, 64), 64, 8, tier="ram")
