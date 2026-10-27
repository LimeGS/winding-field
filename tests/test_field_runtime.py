"""CPU process/FD lifecycle evidence; GPU inventory queries alone are mocked."""

import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from prefetch_fixtures import checkpoint, wait_path

FIXTURE = Path(__file__).with_name("prefetch_fixtures.py").resolve()


def test_public_runtime_module_exists():
    assert importlib.util.find_spec("field_runtime") is not None, "Task5 runtime module is missing"


@pytest.fixture
def r():
    assert importlib.util.find_spec("field_runtime") is not None, "Task5 runtime module is missing"
    return importlib.import_module("field_runtime")


@pytest.fixture
def gpu_query(r, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    responses = ["0, GPU-test, 128", ""]

    def query(command, **kwargs):
        assert kwargs["timeout"] > 0
        return responses[1 if any("--query-compute-apps=" in v for v in command) else 0]

    monkeypatch.setattr(r.subprocess, "check_output", query)
    return responses


def test_lease_is_exclusive_rechecked_and_close_does_not_unlock_inherited_child(r, gpu_query, tmp_path):
    lock, marker = tmp_path / "gpu.lock", tmp_path / "inherited.ready"
    with r.GpuLease(0, "GPU-test", lock, min_free_bytes=100 * 1024**2) as lease:
        with pytest.raises(r.ResourceUnavailable):
            with r.GpuLease(0, "GPU-test", lock):
                pass
        lease.recheck()
        child = subprocess.Popen(
            [sys.executable, str(FIXTURE), "inherited", str(lease.fileno()), str(lock), str(marker)],
            pass_fds=(lease.fileno(),),
            env=dict(os.environ, PYTHONPATH=str(Path(r.__file__).parent)),
        )
        wait_path(marker)
    try:
        with lock.open("a") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker.with_suffix(".stop").touch()
        assert child.wait(timeout=5) == 0
        with r.GpuLease(0, "GPU-test", lock):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


@pytest.mark.parametrize(
    "row,processes,error",
    [
        ("0, GPU-test, 1", "", "resource"),
        ("1, GPU-test, 128", "", "identity"),
        ("0, GPU-other, 128", "", "identity"),
        ("0, GPU-test, 128", "GPU-test, 999999999", "resource"),
        ("0, GPU-test, N/A", "", "resource"),
    ],
)
def test_gpu_admission_failures_are_typed(r, gpu_query, tmp_path, row, processes, error):
    gpu_query[:] = [row, processes]
    with pytest.raises(r.ResourceUnavailable if error == "resource" else ValueError):
        with r.GpuLease(0, "GPU-test", tmp_path / "gpu.lock", min_free_bytes=100 * 1024**2):
            pass


def test_gpu_deadline_rechecked_after_slow_admission_and_before_work(r, gpu_query, monkeypatch, tmp_path):
    clock = [100.0]
    monkeypatch.setattr(r.time, "time", lambda: clock[0])
    query = r.subprocess.check_output

    def slow(*args, **kwargs):
        value = query(*args, **kwargs)
        clock[0] = 102.0
        return value

    monkeypatch.setattr(r.subprocess, "check_output", slow)
    with pytest.raises(r.ResourceUnavailable, match="deadline"):
        with r.GpuLease(0, "GPU-test", tmp_path / "gpu.lock", deadline=101):
            pytest.fail("expired admission")
    monkeypatch.setattr(r.subprocess, "check_output", query)
    with r.GpuLease(0, "GPU-test", tmp_path / "gpu.lock", deadline=104) as lease:
        clock[0] = 105
        with pytest.raises(r.ResourceUnavailable, match="deadline"):
            lease.recheck()


def test_inherited_wrong_inode_and_replaced_lock_are_rejected(r, gpu_query, tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    with (tmp_path / "foreign.lock").open("a") as foreign:
        with pytest.raises(ValueError, match="lease|inode|identity"):
            with r.GpuLease(0, "GPU-test", lock, inherited_fd=foreign.fileno()):
                pass
        assert foreign.fileno() >= 0
    with r.GpuLease(0, "GPU-test", lock) as lease:
        lock.rename(tmp_path / "old.lock")
        lock.touch()
        with pytest.raises(ValueError, match="lease|inode|identity"):
            lease.recheck()


def test_resources_use_explicit_effective_memory_and_disk_reserve(r, tmp_path):
    r.check_resources(tmp_path, required_memory_bytes=50, memory_available=lambda: 50)
    r.check_resources(tmp_path)
    for memory in (None, lambda: None, lambda: 49):
        with pytest.raises(r.ResourceUnavailable, match="memory"):
            r.check_resources(tmp_path, required_memory_bytes=50, memory_available=memory)
    with pytest.raises(r.ResourceUnavailable, match="disk"):
        r.check_resources(tmp_path, reserve_bytes=2**63)
    with pytest.raises(ValueError):
        r.check_resources(tmp_path, required_memory_bytes=-1)


@pytest.fixture
def service(r, tmp_path):
    run, folder = tmp_path / "run", tmp_path / "service"
    source = tmp_path / "pinned.txt"
    source.write_text("pinned")
    sources = {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
        str(FIXTURE): hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
    }

    def start(mode="ok", **kwargs):
        command = [sys.executable, str(FIXTURE), "service", str(run), str(folder), mode, "fresh"]
        return r.run_supervised(
            command,
            command[:-1] + ["resume"],
            run,
            folder,
            source_sha256=sources,
            retry_seconds=0.01,
            stop_grace_seconds=0.1,
            **kwargs,
        )

    return start, run, folder, source


def test_retry_requires_explicit_status_and_resumes_exact_checkpoint(service):
    start, run, folder, _ = service
    assert start("retry") == 0
    assert json.loads((run / "CHECKPOINT.json").read_text()) == checkpoint()
    lines = (run / "attempts").read_text().splitlines()
    assert len(lines) == 2 and lines[0].endswith("fresh") and lines[1].endswith("resume")
    assert json.loads((folder / "SERVICE.json").read_text())["status"] == "SWEEP_BUDGET_REACHED"
    assert (folder / "worker.log").read_text().count("bounded synthetic worker evidence") == 2
    assert start("retry") == 0
    assert (run / "attempts").read_text().splitlines() == lines


@pytest.mark.parametrize("owned", [False, True])
def test_explicit_resume_hands_off_a_dead_stopped_checkpoint_without_automatic_restart(service, owned):
    start, run, folder, _ = service
    if owned:
        assert start("checkpoint_stop") == 0
        assert start("checkpoint_stop") == 0
    else:
        subprocess.run(
            [sys.executable, str(FIXTURE), "service", str(run), str(folder), "checkpoint_stop", "fresh"],
            check=True,
            timeout=5,
        )
    before = (run / "CHECKPOINT.json").read_bytes()
    assert len((run / "attempts").read_text().splitlines()) == 1
    assert start("checkpoint_stop", resume=True) == 0
    attempts = (run / "attempts").read_text().splitlines()
    assert len(attempts) == 2 and attempts[-1].endswith("resume")
    assert (run / "CHECKPOINT.json").read_bytes() == before
    assert json.loads((folder / "SERVICE.json").read_text())["status"] == "SWEEP_BUDGET_REACHED"


def test_explicit_resume_never_retries_a_scientific_failure(service):
    start, run, _, _ = service
    assert start("science") == 1
    (run / "CHECKPOINT.json").write_text(json.dumps(checkpoint()))
    before = (run / "attempts").read_bytes()
    assert start("science", resume=True) == 1
    assert (run / "attempts").read_bytes() == before


@pytest.mark.parametrize("failed", [False, True])
def test_explicit_resume_recovers_owned_forced_stop_but_not_failed_receipts(service, failed):
    start, run, folder, _ = service
    assert start("checkpoint_forced") == 0
    state = json.loads((folder / "SERVICE.json").read_text())
    receipt = json.loads((run / "STATUS.json").read_text())
    assert state["status"] == "STOPPED" and state["last_worker_pid"] == receipt["pid"]
    assert receipt["status"] == "RUNNING"
    before = (run / "CHECKPOINT.json").read_bytes()
    attempts = (run / "attempts").read_text().splitlines()
    (run / "STOP").unlink()
    (folder / "STOP").unlink()
    if failed:
        receipt["status"] = "FAILED"
        (run / "STATUS.json").write_text(json.dumps(receipt))
    assert start("checkpoint_forced", resume=True) == (1 if failed else 0)
    after = (run / "attempts").read_text().splitlines()
    assert len(after) == len(attempts) + int(not failed)
    if not failed:
        assert after[-1].endswith("resume")
        assert json.loads((folder / "SERVICE.json").read_text())["status"] == "SWEEP_BUDGET_REACHED"
    assert (run / "CHECKPOINT.json").read_bytes() == before


@pytest.mark.parametrize("mode", ["science", "unknown", "false_retry", "stale", "missing", "tamper"])
def test_unknown_stale_or_science_failure_is_terminal(service, mode):
    start, run, folder, _ = service
    assert start(mode) == 1
    assert len((run / "attempts").read_text().splitlines()) == 1
    assert json.loads((folder / "SERVICE.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("where", ["run", "service"])
def test_stop_before_start_never_launches(service, where):
    start, run, folder, _ = service
    path = run if where == "run" else folder
    path.mkdir()
    (path / "STOP").touch()
    assert start() == 0
    assert not (run / "attempts").exists()


@pytest.mark.parametrize("mode", ["stop", "stubborn"])
def test_stop_during_work_is_bounded_and_preserves_foreign_process(service, mode):
    start, run, folder, _ = service
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        before = time.monotonic()
        assert start(mode) == 0
        assert time.monotonic() - before < 5
        assert foreign.poll() is None
        assert (run / "STOP").exists()
        assert json.loads((folder / "SERVICE.json").read_text())["worker_pid"] is None
    finally:
        foreign.kill()
        foreign.wait(timeout=3)


def test_stop_racing_popen_is_honored(service, r, monkeypatch):
    start, run, folder, _ = service
    real = subprocess.Popen

    def launch(*args, **kwargs):
        (folder / "STOP").touch()
        return real(*args, **kwargs)

    monkeypatch.setattr(r.subprocess, "Popen", launch)
    assert start() == 0
    assert (run / "STOP").exists()


def test_sources_rechecked_before_launch_and_after_exit(service, monkeypatch):
    start, run, folder, source = service
    monkeypatch.setenv("RUNTIME_TEST_SOURCE", str(source))
    assert start("source_change") == 1
    assert "source" in json.loads((folder / "SERVICE.json").read_text())["error"]
    assert len((run / "attempts").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    "saved",
    [
        {},
        {"identity": {}, "progress": {}},
        {"identity": {"a": "b"}, "progress": {"sweep": -1, "next_block": 0}},
        {"identity": {"a": "b"}, "progress": {"sweep": 0, "next_block": True}},
    ],
)
def test_invalid_checkpoint_is_not_resumed(service, saved):
    start, run, _, _ = service
    run.mkdir()
    (run / "CHECKPOINT.json").write_text(json.dumps(saved))
    assert start() == 1
    assert not (run / "attempts").exists()


def test_live_pid_and_foreign_service_metadata_never_adopt_or_kill(service):
    start, run, folder, _ = service
    run.mkdir()
    (run / "STATUS.json").write_text(json.dumps({"pid": os.getpid(), "status": "RUNNING"}))
    assert start() == 1
    assert not (run / "STOP").exists() and not (run / "attempts").exists()
    state = json.loads((folder / "SERVICE.json").read_text())
    state["worker_pid"] = os.getpid()
    (folder / "SERVICE.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="alive|live|owner"):
        start()


def test_configuration_binding_and_ambiguous_launch_reject_reuse(service):
    start, _, folder, _ = service
    assert start() == 0
    with pytest.raises(ValueError, match="binding|config"):
        start("other")
    state = json.loads((folder / "SERVICE.json").read_text())
    state["status"] = "STARTING"
    (folder / "SERVICE.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="ambiguous|ownership"):
        start()


@pytest.mark.parametrize(
    "options",
    [
        {"index": True},
        {"uuid": "not-gpu"},
        {"lock_path": "relative"},
        {"min_free_bytes": -1},
        {"deadline": float("nan")},
        {"inherited_fd": -1},
    ],
)
def test_bad_gpu_config_has_no_admission_side_effects(r, tmp_path, options):
    with pytest.raises(ValueError):
        r.GpuLease(**({"index": 0, "uuid": "GPU-test", "lock_path": tmp_path / "lease"} | options))
    assert not (tmp_path / "lease").exists()


def test_gpu_environment_and_closed_lease_are_not_usable(r, gpu_query, tmp_path, monkeypatch):
    lock = tmp_path / "lease"
    with r.GpuLease(0, "GPU-test", lock) as lease:
        with pytest.raises(ValueError, match="entered"):
            lease.__enter__()
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
        with pytest.raises(ValueError, match="environment"):
            lease.recheck()
    with pytest.raises(ValueError, match="closed"):
        lease.fileno()
    with pytest.raises(ValueError, match="bound"):
        with r.GpuLease(0, "GPU-test", lock):
            pass


@pytest.mark.parametrize("problem", ["missing", "timeout", "malformed", "bad_process", "own_process"])
def test_gpu_unknown_inventory_and_owned_context(r, gpu_query, tmp_path, monkeypatch, problem):
    if problem in ("missing", "timeout"):

        def query(*a, **kw):
            raise FileNotFoundError() if problem == "missing" else subprocess.TimeoutExpired("nvidia-smi", 1)

        monkeypatch.setattr(r.subprocess, "check_output", query)
    elif problem == "malformed":
        gpu_query[0] = "unknown"
    else:
        gpu_query[1] = "bad" if problem == "bad_process" else f"GPU-test, {os.getpid()}\nGPU-other, 999999999"
    if problem == "own_process":
        with r.GpuLease(0, "GPU-test", tmp_path / "lease"):
            pass
    else:
        with pytest.raises(r.ResourceUnavailable):
            with r.GpuLease(0, "GPU-test", tmp_path / "lease"):
                pass


@pytest.mark.parametrize("failure", ["source", "spawn"])
def test_source_mismatch_and_spawn_failure_are_terminal(service, r, monkeypatch, failure):
    start, run, folder, source = service

    def fail(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    if failure == "source":
        source.write_text("modified before launch")
    else:
        monkeypatch.setattr(r.subprocess, "Popen", fail)
    assert start() == 1
    assert not (run / "attempts").exists()
    assert failure in json.loads((folder / "SERVICE.json").read_text())["error"]


@pytest.mark.parametrize(
    "file,value",
    [("CHECKPOINT.json", []), ("STATUS.json", {}), ("STATUS.json", {"pid": True, "status": "RESOURCE_UNAVAILABLE"})],
)
def test_bad_json_envelopes_stop_before_launch(service, file, value):
    start, run, _, _ = service
    run.mkdir()
    (run / file).write_text(json.dumps(value))
    assert start() == 1
    assert not (run / "attempts").exists()


def test_foreign_service_and_live_locks_are_preserved(service):
    start, run, folder, _ = service
    folder.mkdir()
    evidence = folder / "foreign"
    evidence.write_text("preserve")
    with pytest.raises(ValueError, match="ownership"):
        start()
    assert evidence.read_text() == "preserve"


@pytest.mark.parametrize("where", ["service", "run"])
def test_live_advisory_locks_exclude_other_supervisors(service, where):
    start, run, folder, _ = service
    run.mkdir()
    folder.mkdir()
    path = folder / "SERVICE.lock" if where == "service" else run / ".supervisor.lock"
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if where == "service":
            with pytest.raises(ValueError, match="active"):
                start()
        else:
            assert start() == 1
    assert not (run / "attempts").exists()


def test_atomic_publication_failure_preserves_last_json(r, tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    r._atomic_json(path, {"generation": 1})
    original = os.replace

    def fail(source, destination):
        raise OSError("injected rename failure")

    monkeypatch.setattr(r.os, "replace", fail)
    with pytest.raises(OSError, match="rename"):
        r._atomic_json(path, {"generation": 2})
    assert json.loads(path.read_text()) == {"generation": 1}
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(r.os, "replace", original)
    with pytest.raises(ValueError, match="byte bound"):
        r._atomic_json(path, {"large": "x" * 1024**2})
    assert json.loads(path.read_text()) == {"generation": 1}


def test_ownership_is_rechecked_between_resource_retries(service, r, monkeypatch):
    start, run, folder, _ = service
    publish = r._atomic_json

    def changed(path, value):
        publish(path, value)
        if value.get("status") == "RETRYING":
            (run / "STATUS.json").write_text(json.dumps({"pid": os.getpid(), "status": "RESOURCE_UNAVAILABLE"}))

    monkeypatch.setattr(r, "_atomic_json", changed)
    assert start("retry") == 1
    assert len((run / "attempts").read_text().splitlines()) == 1
    assert "status" in json.loads((folder / "SERVICE.json").read_text())["error"]


def test_stop_at_starting_publication_never_launches(service, r, monkeypatch):
    start, run, folder, _ = service
    publish = r._atomic_json

    def stopped(path, value):
        publish(path, value)
        if value.get("status") == "STARTING":
            (folder / "STOP").touch()

    monkeypatch.setattr(r, "_atomic_json", stopped)
    assert start() == 0
    assert not (run / "attempts").exists()


def test_signal_interrupt_checkpoints_owned_child_and_restores_handler(service):
    import signal

    start, run, folder, _ = service
    old_handler = signal.getsignal(signal.SIGTERM)
    assert start("interrupt") == 130
    assert signal.getsignal(signal.SIGTERM) == old_handler
    assert json.loads((run / "STATUS.json").read_text())["status"] == "CHECKPOINTED_STOP"
    assert json.loads((folder / "SERVICE.json").read_text())["worker_pid"] is None


@pytest.mark.parametrize("status", ["RETRYING", "FAILED"])
def test_missing_previous_receipt_cannot_authorize_recovery(service, status):
    start, run, folder, _ = service
    assert start() == 0
    before = (run / "attempts").read_bytes()
    (run / "CHECKPOINT.json").write_text(json.dumps(checkpoint()))
    metadata = folder / "SERVICE.json"
    state = json.loads(metadata.read_text())
    state.update(status=status, exit_code=75)
    metadata.write_text(json.dumps(state))
    (run / "STATUS.json").unlink()
    assert start() == 1
    assert (run / "attempts").read_bytes() == before


@pytest.mark.parametrize("mode", ["stale_stop", "failed_stop"])
def test_forced_stop_does_not_hide_published_integrity_or_science_failure(service, mode):
    start, run, folder, _ = service
    assert start(mode) == 1
    assert json.loads((run / "STATUS.json").read_text())["status"] == "FAILED"
    assert json.loads((folder / "SERVICE.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize(
    "problem",
    [
        "argv_string",
        "relative_exe",
        "empty_resume",
        "relative_run",
        "overlap",
        "no_sources",
        "bad_hash",
        "relative_source",
        "negative_retry",
    ],
)
def test_invalid_supervisor_configuration_does_not_create_service(r, tmp_path, problem):
    options = dict(
        command=[sys.executable, "-c", "pass"],
        resume_command=[sys.executable, "-c", "pass"],
        run_dir=tmp_path / "run",
        service_dir=tmp_path / "service",
        source_sha256={str(FIXTURE): hashlib.sha256(FIXTURE.read_bytes()).hexdigest()},
    )
    if problem == "argv_string":
        options["command"] = "python worker.py"
    elif problem == "relative_exe":
        options["command"] = ["python", "worker.py"]
    elif problem == "empty_resume":
        options["resume_command"] = []
    elif problem == "relative_run":
        options["run_dir"] = "relative"
    elif problem == "overlap":
        options["service_dir"] = tmp_path / "run" / "service"
    elif problem == "no_sources":
        options["source_sha256"] = {}
    elif problem == "bad_hash":
        options["source_sha256"] = {str(FIXTURE): "not-a-digest"}
    elif problem == "relative_source":
        options["source_sha256"] = {"relative": "a" * 64}
    else:
        options["retry_seconds"] = -1
    with pytest.raises(ValueError):
        r.run_supervised(**options)
    assert list(tmp_path.iterdir()) == []
