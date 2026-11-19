"""The verifier and its rules stay usable: tools/verify.py, REVIEW.md, approvals."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import verify

ROOT = Path(__file__).resolve().parents[1]

# Public API whose silent loss already happened once (3556a4b) or would change results.
CRITICAL = [
    "block_solver.solve_blocks:nonconvergence_policy='fail'",
    "block_solver.solve_blocks:resume",
    "field_solver.solve_field:lam=3.0",
    "field_store.BlockStore",
    "run_coherence --stages=B2",
    "run_coherence <mode>=global",
]


def test_api_inventory_sees_modules_runner_and_critical_entries(tmp_path):
    paths = os.pathsep.join(str(ROOT / d) for d in ("src", "tools", "tests"))
    env = dict(os.environ, PYTHONPATH=paths, VERIFY_TREE=str(ROOT))
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools/verify.py"), "--probe", "api", "--work", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    inventory = set(json.loads(done.stdout.splitlines()[-1]))
    assert len(inventory) > 500
    assert [entry for entry in CRITICAL if entry not in inventory] == []


def test_review_items_parse_with_valid_triggers():
    items = verify.review_items(ROOT / "REVIEW.md")
    ids = [item["id"] for item in items]
    assert ids == [f"R{i}" for i in range(1, len(ids) + 1)] and len(ids) >= 9
    files = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if ".git" not in p.parts]
    for item in items:
        assert item["check"] and item["origin"]
        pattern = item["pattern"].strip(" `")
        if pattern not in ("—", ""):
            re.compile(pattern)
        for glob in (p.strip(" `") for p in item["paths"].split(",")):
            assert any(Path(f).match(glob) or f == glob for f in files), (item["id"], glob)


def test_review_step_matches_paths_and_changed_lines_only(tmp_path):
    (tmp_path / "REVIEW.md").write_text(
        "| R1 | `src/a.py` | — | path check | x |\n"
        "| R2 | `src/*.py` | `^-.*def ` | api check | y |\n"
        "| R3 | `src/*.py` | `identity` | identity check | z |\n"
        "| R4 | `tests/*.py` | `def ` | other file check | w |\n"
    )
    diff = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n"
        "-def gone():\n identity in context only\n"
        "diff --git a/docs/x.md b/docs/x.md\n+def mentioned in prose\n"
    )
    step = verify.Step("review")
    verify.step_review(step, dict(base_tree=tmp_path, lines=verify.changed_lines(diff)))
    assert step.evidence["checklist"] == ["R1: path check", "R2: api check"]


def test_approvals_are_explicit_and_reasoned():
    approved = json.loads((ROOT / "tools/verify-approved.json").read_text())
    assert set(approved) >= {"api", "golden"}
    for section in ("api", "golden"):
        assert all(isinstance(r, str) and len(r) > 10 for r in approved[section].values())


@pytest.mark.parametrize("name", sorted(verify.RUNNERS))
def test_every_step_is_documented_in_the_skill(name):
    skill = (ROOT / ".claude/skills/verify/SKILL.md").read_text()
    assert f"`{name}`" in skill
