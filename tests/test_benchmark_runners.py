"""Provenance must bind the installed source layout, not the former root layout."""

import importlib.util
from pathlib import Path

import pytest

from patch_check import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative",
    [
        "tools/run_coherence.py",
    ],
)
def test_runner_source_manifest_hashes_existing_producer_files(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("runner_provenance", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sources = runner.source_hashes()
    assert sources
    by_name = {Path(name).name: digest for name, digest in sources.items()}
    assert by_name["field_solver.py"] == sha256(ROOT / "src/field_solver.py")
    for name, digest in sources.items():
        source = Path(name) if Path(name).is_absolute() else ROOT / "src" / name
        assert source.is_file() and digest == sha256(source)
