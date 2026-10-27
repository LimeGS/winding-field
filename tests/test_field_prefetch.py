"""Real subprocess/socket tests; all observations are tiny CPU-only fixtures."""

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from prefetch_fixtures import Loader, payload, wait_path

LO, HI, SHAPE = (1, 2, 3), (3, 5, 7), (64, 64, 64)


def test_public_prefetch_module_exists():
    assert importlib.util.find_spec("field_prefetch") is not None, "Task5 prefetch module is missing"


@pytest.fixture
def p():
    assert importlib.util.find_spec("field_prefetch") is not None, "Task5 prefetch module is missing"
    return importlib.import_module("field_prefetch")


def test_one_pending_bit_exact_and_environment(p, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited-foreign-device")
    with p.ObservationPrefetch(
        Loader("environment"), SHAPE, max_bytes=100_000, env={"PREFETCH_TEST": "bound"}
    ) as worker:
        assert worker.available and worker.pending is None
        assert worker.receive(LO, HI) is None
        assert worker.submit(np.array(LO), np.array(HI))
        assert worker.pending == (LO, HI)
        assert not worker.submit(LO, HI)
        with pytest.raises(ValueError, match="bounds"):
            worker.receive((0, 0, 0), HI)
        actual = worker.receive(LO, HI)
        for want, got in zip(payload(LO, HI)[:3], actual[:3]):
            assert want.tobytes() == got.tobytes()
        assert actual[3] == {"source": "a" * 64}
        assert worker.pending is None
    assert not worker.available
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "inherited-foreign-device"


@pytest.mark.parametrize(
    "mode",
    [
        "target_dtype",
        "weights_dtype",
        "observed_dtype",
        "target_shape",
        "weights_shape",
        "observed_shape",
        "nan",
        "inf",
        "negative",
        "receipts",
        "error",
    ],
)
def test_invalid_payload_or_worker_failure_is_visible(p, mode):
    with p.ObservationPrefetch(Loader(mode), SHAPE, max_bytes=100_000) as worker:
        assert worker.submit(LO, HI)
        with pytest.raises((ValueError, RuntimeError)):
            worker.receive(LO, HI)
        assert not worker.available


def test_receipt_changes_are_integrity_failures(p):
    with p.ObservationPrefetch(Loader("changed"), SHAPE, max_bytes=100_000) as worker:
        assert worker.submit(LO, HI)
        assert worker.receive(LO, HI) is not None
        assert worker.submit(LO, HI)
        with pytest.raises(ValueError, match="receipt"):
            worker.receive(LO, HI)


@pytest.mark.parametrize("mode", ["resource", "sleep"])
def test_only_explicit_unavailability_or_timeout_falls_back(p, mode):
    with p.ObservationPrefetch(Loader(mode), SHAPE, max_bytes=100_000, timeout=2) as worker:
        assert worker.available and worker.submit(LO, HI)
        started = time.monotonic()
        assert worker.receive(LO, HI) is None
        assert time.monotonic() - started < 5
        assert not worker.available and not worker.submit(LO, HI)


@pytest.mark.parametrize(
    "lo,hi", [((-1, 0, 0), HI), (LO, (65, 5, 7)), (LO, LO), ((0, 0), HI), ((True, 2, 3), HI), ((1.5, 2, 3), HI)]
)
def test_invalid_bounds_raise_before_sending(p, lo, hi):
    with p.ObservationPrefetch(payload, SHAPE, max_bytes=100_000) as worker:
        with pytest.raises(ValueError, match="bounds"):
            worker.submit(lo, hi)
        assert worker.pending is None


def test_serialized_init_and_result_have_explicit_bounds(p):
    loader = Loader()
    loader.padding = b"x" * 5000
    with pytest.raises(ValueError, match="byte|size|bound"):
        p.ObservationPrefetch(loader, SHAPE, max_bytes=1000)
    with p.ObservationPrefetch(payload, SHAPE, max_bytes=1000) as worker:
        with pytest.raises(ValueError, match="byte|size|bound"):
            worker.submit((0, 0, 0), SHAPE)
        assert worker.pending is None


@pytest.fixture
def wire(p, monkeypatch, tmp_path):
    real = subprocess.Popen
    clients = []

    def start(mode, **options):
        marker = tmp_path / mode

        def launch(command, **kwargs):
            assert command[1:3] == ["-m", "field_prefetch"]
            (fd,) = kwargs["pass_fds"]
            return real(
                [
                    sys.executable,
                    str(Path(__file__).with_name("prefetch_fixtures.py")),
                    "wire",
                    str(fd),
                    mode,
                    str(marker),
                ],
                **kwargs,
            )

        monkeypatch.setattr(p.subprocess, "Popen", launch)
        loader = Loader()
        if mode == "init_blocked":
            loader.padding = b"x" * 900_000
        worker = p.ObservationPrefetch(loader, SHAPE, max_bytes=6_000_000, timeout=2, **options)
        clients.append(worker)
        return worker, marker

    yield start
    for worker in clients:
        worker.close()


@pytest.mark.parametrize("mode", ["partial_header", "partial_body"])
def test_timeout_bounds_entire_partial_frame(wire, mode):
    worker, _ = wire(mode)
    assert worker.submit(LO, HI)
    started = time.monotonic()
    assert worker.receive(LO, HI) is None
    assert time.monotonic() - started < 5
    assert worker._process.poll() is not None


@pytest.mark.parametrize(
    "mode",
    [
        "oversize",
        "bad_digest",
        "bad_pickle",
        "bad_reply",
        "bad_payload",
        "bad_dtype",
        "bad_values",
        "bad_receipts",
        "wrong_bounds",
        "eof",
    ],
)
def test_protocol_corruption_cannot_fall_back(wire, mode):
    worker, _ = wire(mode)
    assert worker.submit(LO, HI)
    with pytest.raises((ValueError, RuntimeError)):
        worker.receive(LO, HI)
    assert not worker.available


@pytest.mark.parametrize("mode", ["startup_timeout", "init_blocked"])
def test_startup_and_blocked_init_are_bounded(wire, mode):
    started = time.monotonic()
    worker, _ = wire(mode)
    assert not worker.available
    assert time.monotonic() - started < 5


def test_large_payload_and_blocked_sender_close_only_owned_child(p, wire):
    with p.ObservationPrefetch(payload, SHAPE, max_bytes=6_000_000) as worker:
        assert worker.submit((0, 0, 0), SHAPE)
        actual = worker.receive((0, 0, 0), SHAPE)
        assert actual[0].tobytes() == payload((0, 0, 0), SHAPE)[0].tobytes()
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        worker, marker = wire("blocked_sender")
        assert worker.submit((0, 0, 0), SHAPE)
        wait_path(marker)
        started = time.monotonic()
        worker.close()
        worker.close()
        assert time.monotonic() - started < 4
        assert worker._process.poll() is not None and foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait(timeout=3)


def test_expired_deadline_does_not_launch_and_idle_worker_expires(p, monkeypatch):
    real = subprocess.Popen
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **kw: pytest.fail("expired launch"))
    with p.ObservationPrefetch(payload, SHAPE, max_bytes=100_000, deadline=time.time() - 1) as worker:
        assert not worker.available
    monkeypatch.setattr(p.subprocess, "Popen", real)
    with p.ObservationPrefetch(payload, SHAPE, max_bytes=100_000, deadline=time.time() + 2) as worker:
        assert worker.available
        worker._process.wait(timeout=4)
        assert not worker.available


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"deadline": float("inf")},
        {"max_bytes": True},
        {"gpu": {}},
        {"gpu": {"index": 0, "uuid": "GPU-test", "lock_path": "/tmp/test", "extra": 1}},
    ],
)
def test_invalid_configuration_raises(p, options):
    with pytest.raises(ValueError):
        p.ObservationPrefetch(payload, SHAPE, **({"max_bytes": 100_000} | options))


def test_launch_oserror_is_explicit_backend_unavailability(p, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("synthetic missing executable")

    monkeypatch.setattr(p.subprocess, "Popen", missing)
    with p.ObservationPrefetch(payload, SHAPE, max_bytes=100_000) as worker:
        assert not worker.available and worker.receive(LO, HI) is None


def test_malformed_startup_is_not_unavailability(wire):
    with pytest.raises(ValueError, match="startup"):
        wire("bad_ready")


@pytest.mark.parametrize("mode", ["gpu_ok", "gpu_foreign", "gpu_busy"])
def test_gpu_worker_lifetime_and_before_work_recheck_with_mocked_queries(p, monkeypatch, tmp_path, mode):
    import fcntl

    real = subprocess.Popen
    lock = tmp_path / "gpu.lock"

    def launch(argv, **kwargs):
        assert argv[1:3] == ["-m", "field_prefetch"]
        return real(
            [sys.executable, str(Path(__file__).with_name("prefetch_fixtures.py")), "gpu", *argv[3:], mode], **kwargs
        )

    monkeypatch.setattr(p.subprocess, "Popen", launch)
    with lock.open("a") as contender:
        if mode == "gpu_busy":
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with p.ObservationPrefetch(
            payload,
            SHAPE,
            max_bytes=100_000,
            gpu={
                "index": 0,
                "uuid": "GPU-test",
                "lock_path": lock,
                "min_free_bytes": 100 * 1024**2,
                "deadline": time.time() + 10,
            },
        ) as worker:
            if mode == "gpu_busy":
                assert not worker.available
                return
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert worker.submit(LO, HI)
            result = worker.receive(LO, HI)
            assert (result is None) == (mode == "gpu_foreign")
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_deadline_wins_over_long_receive_timeout(p, tmp_path):
    marker = tmp_path / "load.ready"
    with p.ObservationPrefetch(
        Loader("sleep", str(marker)), SHAPE, max_bytes=100_000, timeout=30, deadline=time.time() + 2
    ) as worker:
        assert worker.submit(LO, HI)
        wait_path(marker)
        started = time.monotonic()
        assert worker.receive(LO, HI) is None
        assert time.monotonic() - started < 4


@pytest.mark.parametrize(
    "loader,shape,options", [(None, SHAPE, {}), (payload, None, {}), (payload, SHAPE, {"env": {"X": 1}})]
)
def test_bad_loader_shape_and_environment_fail_before_launch(p, loader, shape, options):
    with pytest.raises(ValueError):
        p.ObservationPrefetch(loader, shape, max_bytes=100_000, **options)


def test_gpu_lease_precedes_loader_unpickling_and_initialization(p, monkeypatch, tmp_path):
    from prefetch_fixtures import LeaseBoundLoader

    real = subprocess.Popen
    lock, marker = tmp_path / "gpu.lock", tmp_path / "initialized"

    def launch(argv, **kwargs):
        return real(
            [sys.executable, str(Path(__file__).with_name("prefetch_fixtures.py")), "gpu", *argv[3:], "gpu_ok"],
            **kwargs,
        )

    monkeypatch.setattr(p.subprocess, "Popen", launch)
    with p.ObservationPrefetch(
        LeaseBoundLoader(lock, marker),
        SHAPE,
        max_bytes=100_000,
        gpu={"index": 0, "uuid": "GPU-test", "lock_path": lock},
    ) as worker:
        assert worker.available
        assert marker.read_text() == "lease-held"
        assert worker.submit(LO, HI) and worker.receive(LO, HI) is not None
