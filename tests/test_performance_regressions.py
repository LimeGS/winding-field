"""Cross-component precision and process-ownership regressions."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_block_solver import case, runner

from field_prefetch import ObservationPrefetch
from field_runtime import run_supervised
from field_solver import solve_field
from field_store import BlockStore
from paris_data import save_field


def test_float64_publication_preserves_the_certified_equations(tmp_path):
    target = np.zeros((1, 1, 2, 3))
    target[..., 0] = 1.00000001
    field, info = solve_field(target, np.ones((1, 1, 2)), dtype="float64", rtol=1e-12)
    assert info["converged"]
    path = tmp_path / "field.npy"
    metadata = save_field(path, field, {"source": "synthetic"}, info)
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float64
    residual = abs((stored[0, 0, 1] - stored[0, 0, 0]) - 1.00000001) / 1.00000001
    assert residual <= 1e-12 and metadata["solver"]["dtype"] == "float64"


def test_publisher_rejects_array_precision_different_from_its_solver_receipt(tmp_path):
    target = np.zeros((1, 1, 2, 3))
    target[..., 0] = 1.00000001
    field, info = solve_field(target, np.ones((1, 1, 2)), dtype="float64", rtol=1e-12)
    with pytest.raises(ValueError, match="precision|dtype"):
        save_field(tmp_path / "field.npy", field.astype(np.float32), {}, info)
    assert not list(tmp_path.iterdir())


def test_legacy_receipt_cannot_certify_a_lossy_float32_export(tmp_path):
    with pytest.raises(ValueError, match="precision|lossy"):
        save_field(tmp_path / "field.npy", np.array([[[0.0, 1.00000001]]]), {}, {"converged": True})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("changed", ["dtype", "order", "shape"])
def test_early_run_rejection_closes_supplied_prefetch(tmp_path, changed):
    options = case(tmp_path)
    client = ObservationPrefetch(options["load_observations"], options["shape"], max_bytes=100000)
    process = client._process
    assert client.available
    options.update({changed: {"dtype": "invalid", "order": [[0, 0, 0]], "shape": (4, 4, 6)}[changed]})
    try:
        with pytest.raises(ValueError):
            runner()(tmp_path / "run", prefetch=client, **options)
        assert not client.available and process.poll() is not None
        assert not (tmp_path / "run").exists()
    finally:
        client.close()


@pytest.mark.parametrize("capped", [False, True])
def test_uncapped_progress_does_not_stat_every_payload(tmp_path, monkeypatch, capped):
    root = tmp_path / "field"
    with BlockStore(root, (2, 2, 12), 2, {"test": "bounded metadata"}, max_bytes=100000 if capped else None) as store:
        for x in range(6):
            store.commit((0, 0, x), np.full((2, 2, 2), x + 0.125), {"next_block": x + 1})
        payloads = {root / item["path"] for item in store.manifest["chunks"].values()}
        visited = []
        original = Path.lstat

        def recorded(path, *args, **kwargs):
            if path in payloads:
                visited.append(path)
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", recorded)
        store.save_progress({"next_block": 6, "sweep": 2})
        assert bool(visited) is capped
        np.testing.assert_array_equal(store.read([0, 0, 0], [2, 2, 12])[0, 0], np.repeat(np.arange(6) + 0.125, 2))


def live(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False)
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_worker_exit_between_grace_timeout_and_signal_is_not_a_stop_failure(tmp_path, monkeypatch):
    from field_runtime import _stop_child

    marker = tmp_path / "may-exit"
    program = "from pathlib import Path; import sys,time\np=Path(sys.argv[1])\nwhile not p.exists(): time.sleep(.01)"
    child = subprocess.Popen([sys.executable, "-c", program, str(marker)], start_new_session=True)
    real_group = os.getpgid

    def exit_before_signal(pid):
        if pid == child.pid:
            marker.touch()
            until = time.monotonic() + 5
            while live(pid) and time.monotonic() < until:
                time.sleep(0.01)
            assert not live(pid)
        return real_group(pid)

    monkeypatch.setattr(os, "getpgid", exit_before_signal)
    try:
        _stop_child(child, tmp_path, 0.0)
        assert child.returncode == 0
    finally:
        marker.touch()
        child.wait(timeout=5)


@pytest.mark.parametrize("mode", ["sleep", "sleep_ignoring_term"])
def test_forced_supervisor_stop_releases_prefetch_but_preserves_unrelated_process(tmp_path, monkeypatch, mode):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project / "src"), str(project / "tests")]))
    signals, real_signal = [], os.killpg

    def recorded_signal(group, sig):
        try:
            result = real_signal(group, sig)
        except OSError as error:
            signals.append((int(sig), error.errno))
            raise
        signals.append((int(sig), None))
        return result

    monkeypatch.setattr(os, "killpg", recorded_signal)
    script, marker = tmp_path / "worker.py", tmp_path / "prefetch.pid"
    script.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from test_block_solver import case, runner\n"
        "from field_prefetch import ObservationPrefetch\nfrom prefetch_fixtures import Loader\n"
        "home=Path(sys.argv[1]); options=case(home)\n"
        "client=ObservationPrefetch(Loader(sys.argv[2],str(home/'prefetch.pid')),options['shape'],max_bytes=100000,timeout=60)\n"
        "runner()(home/'run',prefetch=client,**options)\n"
    )
    command = [sys.executable, str(script), str(tmp_path), mode]
    service = tmp_path / "service"
    cancelled = threading.Event()

    def stop_when_loading():
        until = time.monotonic() + 15
        while not cancelled.wait(0.01):
            if marker.exists() or time.monotonic() >= until:
                (service / "STOP").touch()
                return

    stopper = threading.Thread(target=stop_when_loading)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    stopper.start()
    try:
        status = run_supervised(
            command,
            [*command, "--resume"],
            tmp_path / "run",
            service,
            source_sha256={str(script): hashlib.sha256(script.read_bytes()).hexdigest()},
            stop_grace_seconds=0.1,
        )
        assert status == 0, ((service / "worker.log").read_text(), signals, (service / "SERVICE.json").read_text())
        assert marker.exists(), "worker never reached the prefetch loader"
        pid = int(marker.read_text())
        until = time.monotonic() + 3
        while live(pid) and time.monotonic() < until:
            time.sleep(0.01)
        assert not live(pid), "owned prefetch survived forced runner shutdown"
        assert unrelated.poll() is None
        assert json.loads((service / "SERVICE.json").read_text())["status"] == "STOPPED"
    finally:
        cancelled.set()
        stopper.join(timeout=2)
        if marker.exists():
            pid = int(marker.read_text())
            saved = json.loads((service / "SERVICE.json").read_text())
            if live(pid) and os.getsid(pid) == saved["last_worker_pid"]:
                os.kill(pid, signal.SIGKILL)  # Only the fixture-owned, session-matched leaked child.
        unrelated.terminate()
        unrelated.wait(timeout=3)
