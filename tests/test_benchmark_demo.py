"""The public first-run example must work without private data or a solver."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_benchmark_demo_runs_and_preserves_existing_output(tmp_path):
    output = tmp_path / "demo"
    code = (
        "import runpy,sys; m=runpy.run_path(sys.argv[1]); status=m['main'](sys.argv[2:]); "
        "assert not {'torch', 'matplotlib', 'PIL', 'skimage', 'winding_field'} & sys.modules.keys(); sys.exit(status)"
    )
    command = [sys.executable, "-I", "-c", code, str(ROOT / "examples/benchmark_demo.py"), "--output", str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary_path = output / "evaluation/SUMMARY.json"
    summary = json.loads(summary_path.read_text())
    assert summary["synthetic"] is True and summary["physical_acceptance"] is False
    candidate = summary["methods"]["candidate"]
    assert candidate["all"][0]["selected"] == candidate["all"][0]["covered"] == 6
    assert candidate["all"][0]["median"] == 0
    assert candidate["human"]["correct"] == candidate["human"]["eligible"] == 3
    for control in ("constant_control", "compressed_control"):
        assert summary["methods"][control]["human"]["correct"] == 0
    assert not list((output / "evaluation").glob("*.png"))
    assert "Synthetic separation control: 3/3 correct" in (output / "evaluation/TABLES.md").read_text()
    assert all("reused development data" not in note for note in summary["limitations"])
    before = summary_path.read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and summary_path.read_bytes() == before
