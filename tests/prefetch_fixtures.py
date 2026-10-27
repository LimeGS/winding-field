"""Importable, CPU-only loaders and real subprocess fault fixtures."""

import hashlib
import json
import os
import pickle
import signal
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np


def payload(lo, hi):
    shape = tuple(b - a for a, b in zip(lo, hi))
    target = np.arange(np.prod(shape) * 3, dtype=np.float32).reshape(*shape, 3) / 8
    target.flat[:3] = [-0.0, np.nextafter(np.float32(0), np.float32(1)), -np.finfo(np.float32).max]
    return target, np.full(shape, 0.25, np.float32), np.ones(shape, bool), {"source": "a" * 64}


class Loader:
    def __init__(self, mode="ok", marker=None):
        self.mode, self.marker, self.calls = mode, marker, 0

    def __call__(self, lo, hi):
        from field_runtime import ResourceUnavailable

        self.calls += 1
        if self.mode == "sleep_ignoring_term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if self.marker:
            Path(self.marker).write_text(str(os.getpid()))
        if self.mode in ("sleep", "sleep_ignoring_term"):
            time.sleep(60)
        if self.mode == "resource":
            raise ResourceUnavailable("synthetic backend busy")
        if self.mode == "error":
            raise RuntimeError("synthetic science failure")
        if self.mode == "environment":
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
            assert os.environ["PREFETCH_TEST"] == "bound"
            assert "torch" not in sys.modules
        result = list(payload(lo, hi))
        if self.mode.endswith("_dtype"):
            i = {"target_dtype": 0, "weights_dtype": 1, "observed_dtype": 2}[self.mode]
            result[i] = result[i].astype(np.float64)
        if self.mode.endswith("_shape"):
            i = {"target_shape": 0, "weights_shape": 1, "observed_shape": 2}[self.mode]
            result[i] = result[i][:-1]
        if self.mode == "nan":
            result[0].flat[0] = np.nan
        if self.mode == "inf":
            result[1].flat[0] = np.inf
        if self.mode == "negative":
            result[1].flat[0] = -1
        if self.mode == "receipts":
            result[3] = {"source": "not a digest"}
        if self.mode == "changed" and self.calls > 1:
            result[3] = {"source": "b" * 64}
        return tuple(result)


class LeaseBoundLoader:
    def __init__(self, lock, marker):
        self.lock, self.marker = str(lock), str(marker)

    def __setstate__(self, state):
        import fcntl

        self.__dict__.update(state)
        with open(self.lock, "a") as contender:
            try:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                status = "lease-held"
            else:
                status = "unleased-initialization"
        Path(self.marker).write_text(status)

    def __call__(self, lo, hi):
        return payload(lo, hi)


def wait_path(path, seconds=5):
    until = time.monotonic() + seconds
    while not Path(path).exists():
        if time.monotonic() >= until:
            raise AssertionError(f"fixture did not publish {path}")
        time.sleep(0.01)


def wire_worker(fd, mode, marker):
    """Independent framing implementation for partial/oversize/blocked I/O."""

    def exact(count):
        data = b""
        while len(data) < count:
            part = channel.recv(count - len(data))
            if not part:
                raise EOFError
            data += part
        return data

    def receive():
        size = struct.unpack("!Q", exact(8))[0]
        digest, raw = exact(32), exact(size)
        assert hashlib.sha256(raw).digest() == digest
        return pickle.loads(raw)

    def send(value):
        raw = pickle.dumps(value, protocol=5)
        channel.sendall(struct.pack("!Q", len(raw)) + hashlib.sha256(raw).digest() + raw)

    with socket.socket(fileno=fd) as channel:
        if mode == "init_blocked":
            Path(marker).touch()
            time.sleep(60)
        receive()
        if mode == "startup_timeout":
            time.sleep(60)
        if mode == "bad_ready":
            send(("surprise",))
            time.sleep(60)
        send(("ready",))
        bounds = receive()
        if mode == "eof":
            return
        if mode == "partial_header":
            channel.sendall(b"\x00\x00")
        elif mode == "partial_body":
            channel.sendall(struct.pack("!Q", 4096) + b"x" * 32 + b"partial")
        elif mode == "oversize":
            channel.sendall(struct.pack("!Q", 2**40))
        elif mode == "bad_digest":
            channel.sendall(struct.pack("!Q", 3) + b"x" * 32 + b"bad")
        elif mode == "bad_pickle":
            channel.sendall(struct.pack("!Q", 3) + hashlib.sha256(b"bad").digest() + b"bad")
        elif mode == "bad_reply":
            send(None)
        elif mode in ("bad_payload", "bad_dtype", "bad_values", "bad_receipts"):
            values = list(payload(*bounds))
            if mode == "bad_dtype":
                values[0] = values[0].astype(np.float64)
            elif mode == "bad_values":
                values[1].flat[0] = np.nan
            elif mode == "bad_receipts":
                values[3] = {"source": "invalid"}
            send(("result", bounds, None if mode == "bad_payload" else values))
        elif mode == "wrong_bounds":
            send(("result", ((0, 0, 0), (1, 1, 1)), payload(*bounds)))
        elif mode == "blocked_sender":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            Path(marker).touch()
            try:
                send(("result", bounds, payload(*bounds)))
            except OSError:
                pass
        time.sleep(60)


def gpu_worker(fd, limit, mode):
    """Real lease/worker implementation, replacing only NVIDIA query output."""
    import field_prefetch
    import field_runtime

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert "torch" not in sys.modules
    calls = 0

    def query(command, **kwargs):
        nonlocal calls
        if any("--query-gpu=" in v for v in command):
            calls += 1
            return "0, GPU-test, 128"
        return "GPU-test, 999999999" if mode == "gpu_foreign" and calls > 1 else ""

    field_runtime.subprocess.check_output = query
    field_prefetch._worker(fd, limit)


def checkpoint():
    return {"identity": {"synthetic": "a" * 64}, "progress": {"sweep": 2, "next_block": 3}}


def service_worker(run, service, mode, resumed):
    run, service = Path(run), Path(service)
    run.mkdir(exist_ok=True)
    with (run / "attempts").open("a") as stream:
        stream.write(f"{os.getpid()} {resumed}\n")
    saved = run / "CHECKPOINT.json"
    if resumed == "resume":
        assert json.loads(saved.read_text()) == checkpoint()
    if mode == "retry" and resumed != "resume":
        saved.write_text(json.dumps(checkpoint()))
        status, code = "RESOURCE_UNAVAILABLE", 75
    elif mode == "checkpoint_stop" and resumed != "resume":
        saved.write_text(json.dumps(checkpoint()))
        status, code = "CHECKPOINTED_STOP", 0
    elif mode == "checkpoint_forced" and resumed != "resume":
        saved.write_text(json.dumps(checkpoint()))
        (run / "STATUS.json").write_text(json.dumps(dict(pid=os.getpid(), status="RUNNING")))
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (service / "STOP").touch()
        time.sleep(60)
        return 1
    elif mode in ("stale_stop", "failed_stop"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (run / "STATUS.json").write_text(
            json.dumps(
                {
                    "pid": os.getppid() if mode == "stale_stop" else os.getpid(),
                    "status": "FAILED",
                }
            )
        )
        (service / "STOP").touch()
        time.sleep(60)
        return 1
    elif mode in ("stop", "stubborn", "interrupt"):
        if mode == "stubborn":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if mode == "interrupt":
            os.kill(os.getppid(), signal.SIGTERM)
        else:
            (service / "STOP").touch()
        if mode == "stubborn":
            time.sleep(60)
        wait_path(run / "STOP")
        status, code = "CHECKPOINTED_STOP", 0
    elif mode == "science":
        status, code = "FAILED", 1
    elif mode == "unknown":
        status, code = "RESOURCE_UNAVAILABLE", 3
    elif mode == "false_retry":
        status, code = "FAILED", 75
    elif mode == "stale":
        status, code = "RESOURCE_UNAVAILABLE", 75
    elif mode == "missing":
        return 0
    elif mode == "tamper":
        metadata = service / "SERVICE.json"
        state = json.loads(metadata.read_text())
        state["binding"]["command"] = ["/unrelated"]
        metadata.write_text(json.dumps(state))
        status, code = "SWEEP_BUDGET_REACHED", 0
    elif mode == "source_change":
        Path(os.environ["RUNTIME_TEST_SOURCE"]).write_text("changed")
        status, code = "SWEEP_BUDGET_REACHED", 0
    else:
        status, code = "SWEEP_BUDGET_REACHED", 0
    (run / "STATUS.json").write_text(
        json.dumps(
            {
                "pid": os.getpid() if mode != "stale" else os.getppid(),
                "status": status,
            }
        )
    )
    print("bounded synthetic worker evidence", flush=True)
    return code


if __name__ == "__main__":
    if sys.argv[1] == "wire":
        wire_worker(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "service":
        raise SystemExit(service_worker(*sys.argv[2:]))
    elif sys.argv[1] == "inherited":
        fd, lock, marker = int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
        import field_runtime

        field_runtime.subprocess.check_output = lambda argv, **kw: (
            "0, GPU-test, 128" if any("--query-gpu=" in arg for arg in argv) else ""
        )
        with field_runtime.GpuLease(0, "GPU-test", lock, inherited_fd=fd):
            marker.touch()
            wait_path(marker.with_suffix(".stop"))
    elif sys.argv[1] == "gpu":
        gpu_worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
