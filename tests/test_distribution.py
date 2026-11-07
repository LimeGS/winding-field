import tomllib
from pathlib import Path


def test_core_distribution_uses_repository_name():
    config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert config["project"]["name"] == "winding-field"
