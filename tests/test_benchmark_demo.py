"""The public first-run example must work without private data or a solver."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_benchmark_demo_runs_and_preserves_existing_output(tmp_path):
    output = tmp_path / "demo"
    code = (
        "import runpy,sys; m=runpy.run_path(sys.argv[1]); status=m['main'](sys.argv[2:]); "
        "assert 'torch' not in sys.modules; sys.exit(status)"
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
    visuals = json.loads((output / "evaluation/VISUALS.json").read_text())
    assert visuals["synthetic"] is True
    with Image.open(output / "evaluation/ct-winding.png") as image:
        image.verify()
    with np.load(output / "evaluation/ct-winding-samples.npz") as arrays:
        np.testing.assert_allclose(arrays["XY_candidate"], 2)
    assert "Synthetic separation control: 3/3 correct" in (output / "evaluation/TABLES.md").read_text()
    assert all("reused development data" not in note for note in summary["limitations"])
    before = summary_path.read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and summary_path.read_bytes() == before


def test_demo_figures_are_visibly_labeled_synthetic(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    savefig = Figure.savefig

    def save_and_inspect(figure, *args, **kwargs):
        titles.append(([text.get_text() for text in figure.texts], figure.axes[0].get_title()))
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save_and_inspect)
    spec = importlib.util.spec_from_file_location("benchmark_demo", ROOT / "examples/benchmark_demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    assert demo.main(["--output", str(tmp_path / "demo")]) == 0
    assert len(titles) == 4
    assert all(any("synthetic CT-like data" in title for title in figure_titles) for figure_titles, _ in titles)
    assert all("Synthetic CT-like data" in panel_title for _, panel_title in titles)
