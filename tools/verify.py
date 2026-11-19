#!/usr/bin/env python3
"""Verify a change end to end, against the previous version as the oracle.

    python tools/verify.py                      # all steps, baseline origin/main
    python tools/verify.py --baseline <ref>     # another oracle
    python tools/verify.py --steps golden,e2e

Steps (see .claude/skills/verify/SKILL.md):

- ``review``: REVIEW.md items whose triggers match the diff (read from the baseline).
- ``api``: every public function, class, parameter and default of the installed modules,
  and every ``tools/run_coherence.py`` option of the baseline still exists unchanged.
- ``golden``: solver outputs are byte-identical to the baseline (solve_field, block solves
  with and without caches/read workers, the synthetic B2 build).
- ``e2e``: the README examples run as a user runs them.
- ``cross-version``: a block solve checkpointed by the baseline resumes with this version
  to the same bytes as an uninterrupted run, or is refused without being modified.

Intended differences go in ``tools/verify-approved.json`` with a reason. The receipt is
written to ``.verify/VERIFY.json``; exit status is 1 when any step FAILs.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[1]
REPO = DEFAULT_REPO  # CI runs the base branch's verifier on the checkout via --repo
STEPS = ("review", "api", "golden", "e2e", "cross-version")


# --------------------------------------------------------------------------------------
# Probes run in a subprocess with <tree>/src, <tree>/tools and <tree>/tests on sys.path,
# so the same probe code measures the baseline and the working tree.
# --------------------------------------------------------------------------------------


def _tree():
    return Path(os.environ["VERIFY_TREE"])


def _files_digest(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix in (".npy", ".npz"):
            h.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
    return h.hexdigest()


def probe_api(work):
    """Public callables with parameters and defaults, plus runner options."""
    import argparse as ap
    import inspect
    import tomllib
    from importlib import import_module

    project = tomllib.loads((_tree() / "pyproject.toml").read_text())
    found = set()

    def signature(prefix, function):
        try:
            parameters = inspect.signature(function).parameters.values()
        except (TypeError, ValueError):
            return
        for p in parameters:
            if p.name in ("self", "cls"):
                continue
            found.add(f"{prefix}:{p.name}")
            if p.default is not p.empty:
                found.add(f"{prefix}:{p.name}={p.default!r}")

    for name in project["tool"]["setuptools"]["py-modules"]:
        module = import_module(name)
        for attr, value in vars(module).items():
            if attr.startswith("_") or getattr(value, "__module__", None) != name:
                continue
            if inspect.isclass(value):
                found.add(f"{name}.{attr}")
                signature(f"{name}.{attr}", value)
                for method, member in vars(value).items():
                    if not method.startswith("_") and callable(member):
                        found.add(f"{name}.{attr}.{method}")
                        signature(f"{name}.{attr}.{method}", member)
            elif inspect.isfunction(value):
                found.add(f"{name}.{attr}")
                signature(f"{name}.{attr}", value)

    class Captured(Exception):
        pass

    def capture(self, *args, **kwargs):
        raise Captured(self)

    ap.ArgumentParser.parse_args = capture  # type: ignore[method-assign]
    runner = import_module("run_coherence")
    try:
        runner.main()
    except Captured as captured:
        for action in captured.args[0]._actions:
            if isinstance(action, ap._HelpAction):
                continue
            names = action.option_strings or [f"<{action.dest}>"]
            for option in names:
                found.add(f"run_coherence {option}")
            for choice in action.choices or ():
                found.add(f"run_coherence {names[-1]}={choice}")
    return sorted(found)


def probe_golden(work):
    """Hashes of solver outputs on fixed synthetic problems."""
    import numpy as np
    from test_block_solver import case, state

    from block_solver import solve_blocks
    from field_solver import solve_field

    work = Path(work)
    results = {}
    shape = (1, 4, 6)
    target = np.zeros((*shape, 3), np.float32)
    target[..., 0] = 1.0
    field, receipt = solve_field(target, np.ones(shape), spacing_xyz=(1, 1, 1))
    results["solve_field.ramp"] = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
    results["solve_field.ramp.iterations"] = str(receipt.get("iterations"))
    # Every term at its default weights (lam, rtol, preconditioner), so a default change shows.
    rng = np.random.default_rng(7)
    shape = (2, 3, 4)
    fixed = np.full(shape, np.nan)
    fixed.flat[[3, 22]] = [1.25, -0.5]
    field, receipt = solve_field(
        rng.normal(size=(*shape, 3)),
        rng.uniform(0.5, 2.0, size=shape),
        (2, 3, 4),
        pairs=np.array([[0, 23], [1, 17], [5, 9]]),
        fixed=fixed,
        reference=rng.normal(size=shape),
        reference_weight=0.2,
    )
    results["solve_field.full"] = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
    for name, extra in (("block.case", dict(read_workers=1, cache_bytes=0)), ("block.case.perf", {})):
        root = work / name
        root.mkdir()
        options = case(root)
        options.update(extra)
        solve_blocks(root / "run", **options)
        checkpoint, _, _ = state(root / "run")
        results[name] = hashlib.sha256(
            "".join(v["sha256"] for _, v in sorted(checkpoint["chunks"].items())).encode()
        ).hexdigest()
    demo = work / "build-demo"
    subprocess.run(
        [sys.executable, str(_tree() / "examples/build_demo.py"), "--output", str(demo)],
        check=True,
        capture_output=True,
        env=dict(os.environ),
    )
    results["build_demo.field"] = _files_digest(demo / "field")
    return results


def _case_options(root):
    from test_block_solver import case

    root.mkdir(parents=True, exist_ok=True)
    return case(root)


def probe_partial(work):
    """Checkpoint a block solve mid-sweep with this tree."""
    from block_solver import solve_blocks

    path = Path(work) / "partial"

    def stop():
        checkpoint = path / "CHECKPOINT.json"
        if not checkpoint.exists():
            return False
        progress = json.loads(checkpoint.read_text())["progress"]
        return progress.get("sweep") == 1 and progress.get("next_block") == 1

    result = solve_blocks(path, stop=stop, **_case_options(Path(work) / "partial-case"))
    return dict(status=result["status"], path=str(path))


def probe_resume(work):
    """Resume the baseline checkpoint with this tree; compare with an uninterrupted run."""
    from test_block_solver import state

    from block_solver import solve_blocks

    work = Path(work)
    solve_blocks(work / "full", **_case_options(work / "full-case"))
    full, _, _ = state(work / "full")
    partial = work / "partial"
    before = (partial / "CHECKPOINT.json").read_bytes()
    try:
        result = solve_blocks(partial, resume=True, **_case_options(work / "resume-case"))
    except Exception as error:  # refusal is acceptable only if nothing was modified
        return dict(outcome="refused", error=f"{type(error).__name__}: {error}"[:300],
                    unchanged=(partial / "CHECKPOINT.json").read_bytes() == before)  # fmt: skip
    resumed, _, _ = state(partial)

    def chunks(checkpoint):
        return {k: v["sha256"] for k, v in checkpoint["chunks"].items()}

    return dict(
        outcome="resumed",
        status=result["status"],
        identical=chunks(resumed) == chunks(full),
        sweeps_identical=resumed["progress"]["completed_sweeps"] == full["progress"]["completed_sweeps"],
    )


PROBES = dict(api=probe_api, golden=probe_golden, partial=probe_partial, resume=probe_resume)


def run_probe(name, tree, work):
    paths = [str(tree / "src"), str(tree / "tools"), str(tree / "tests")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), VERIFY_TREE=str(tree))
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", name, "--work", str(work)]
    done = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True)
    if done.returncode:
        return None, (done.stderr or done.stdout)[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1]), None


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------


class Step:
    def __init__(self, name):
        self.name, self.status, self.evidence, self.start = name, "PASS", {}, time.monotonic()

    def fail(self, reason, **evidence):
        self.status = "FAIL"
        self.evidence.setdefault("failures", []).append(reason)
        self.evidence.update(evidence)

    def warn(self, reason):
        if self.status == "PASS":
            self.status = "WARN"
        self.evidence.setdefault("warnings", []).append(reason)

    def record(self):
        return dict(status=self.status, seconds=round(time.monotonic() - self.start, 1), **self.evidence)


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def changed_lines(diff, untracked=()):
    """``{path: changed lines}`` from a unified diff; untracked files count as all-added."""
    files, current = {}, None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[-1]
            files[current] = []
        elif current and line[:1] in "+-" and not line.startswith(("+++", "---")):
            files[current].append(line)
    for path in untracked:
        try:
            files[path] = ["+" + line for line in (REPO / path).read_text().splitlines()]
        except (OSError, UnicodeDecodeError):
            files[path] = []
    return files


def review_items(path):
    if not path.is_file():
        return []
    items = []
    for line in path.read_text().splitlines():
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) == 5 and re.fullmatch(r"R\d+", cells[0]):
            items.append(dict(zip(("id", "paths", "pattern", "check", "origin"), cells)))
    return items


def step_review(step, ctx):
    """An item triggers when a changed file matches its Paths and, if it has a Diff regex,
    a changed (+/-) line of such a file matches it. Context lines never trigger."""
    # The checklist that judges a change is read from the baseline, never from the change.
    source = ctx["base_tree"] / "REVIEW.md"
    if not source.is_file():
        step.warn("REVIEW.md is not on the baseline yet; using the working tree copy")
        source = REPO / "REVIEW.md"
    files = ctx["lines"]
    matched = []
    for row in review_items(source):
        paths = [p.strip(" `") for p in row["paths"].split(",") if p.strip(" `")]
        selected = [f for f in files if any(fnmatch.fnmatch(f, p) for p in paths)]
        pattern = row["pattern"].strip(" `")
        if pattern and pattern != "—":
            text = "\n".join(line for f in selected for line in files[f])
            selected = selected if re.search(pattern, text, re.MULTILINE) else []
        if selected:
            matched.append(f"{row['id']}: {row['check']}")
    step.evidence.update(changed_files=len(files), checklist=matched)


def _differential(step, ctx, probe):
    base, error = run_probe(probe, ctx["base_tree"], Path(tempfile.mkdtemp(dir=ctx["work"])))
    head, head_error = run_probe(probe, REPO, Path(tempfile.mkdtemp(dir=ctx["work"])))
    if head_error:
        step.fail(f"working tree {probe} probe failed", error=head_error)
        return None, None
    if error:
        step.warn(f"baseline {probe} probe failed: {error[-500:]}")
        return None, None
    return base, head


def step_api(step, ctx):
    base, head = _differential(step, ctx, "api")
    if base is None:
        return
    approved = ctx["approved"].get("api", {})
    removed = sorted(set(base) - set(head) - set(approved))
    step.evidence.update(entries=len(head), added=sorted(set(head) - set(base)), approved_removals=sorted(approved))
    if removed:
        step.fail(f"{len(removed)} public API entries removed or changed", removed=removed)


def step_golden(step, ctx):
    base, head = _differential(step, ctx, "golden")
    if base is None:
        return
    approved = ctx["approved"].get("golden", {})
    changed = sorted(k for k in base.keys() & head.keys() if base[k] != head[k])
    step.evidence.update(identical=sorted(k for k in base if head.get(k) == base[k]))
    for key in sorted(base.keys() ^ head.keys()):
        step.warn(f"{key} measured on one side only")
    if head.get("block.case") != head.get("block.case.perf"):
        step.fail("cache/read-worker settings changed block solve bytes")
    unapproved = [k for k in changed if k not in approved]
    if changed:
        step.evidence["approved_changes"] = sorted(set(changed) - set(unapproved))
    if unapproved:
        step.fail("outputs differ from the baseline", changed=unapproved)


def step_e2e(step, ctx):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO / "tools")]))
    commands = {
        "minimal_field": ["examples/minimal_field.py"],
        "build_demo": ["examples/build_demo.py", "--output", str(ctx["work"] / "e2e-build")],
        "benchmark_demo": ["examples/benchmark_demo.py", "--output", str(ctx["work"] / "e2e-bench")],
    }
    for name, args in commands.items():
        done = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
        step.evidence[name] = done.stdout.strip().splitlines()[-1:] if done.stdout else []
        if done.returncode:
            step.fail(f"{name} exited {done.returncode}", **{f"{name}_stderr": done.stderr[-2000:]})


def step_cross_version(step, ctx):
    work = Path(tempfile.mkdtemp(dir=ctx["work"]))
    partial, error = run_probe("partial", ctx["base_tree"], work)
    if error:
        return step.warn(f"baseline checkpoint failed: {error[-500:]}")
    if partial["status"] != "CHECKPOINTED_STOP":
        return step.warn(f"baseline did not stop mid-solve: {partial['status']}")
    result, error = run_probe("resume", REPO, work)
    if error:
        return step.fail("resuming the baseline checkpoint crashed", error=error)
    step.evidence["resume_baseline_checkpoint"] = result
    if result["outcome"] == "refused" and not result["unchanged"]:
        step.fail("refused resume modified the baseline checkpoint")
    if result["outcome"] == "resumed" and not (result["identical"] and result["sweeps_identical"]):
        step.fail("resumed baseline checkpoint differs from an uninterrupted run")


RUNNERS = {
    "review": step_review,
    "api": step_api,
    "golden": step_golden,
    "e2e": step_e2e,
    "cross-version": step_cross_version,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", default="origin/main", help="oracle git ref")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="checkout to verify")
    parser.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset")
    parser.add_argument("--out", type=Path, help="keep work files here (default: temporary)")
    parser.add_argument("--receipt", type=Path, help="receipt path (default: <repo>/.verify/VERIFY.json)")
    parser.add_argument("--probe", choices=sorted(PROBES), help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        sys.path[:0] = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        print(json.dumps(PROBES[args.probe](args.work), default=str))
        return 0
    global REPO
    REPO = args.repo.resolve()
    approved = REPO / "tools/verify-approved.json"
    steps = [s for s in args.steps.split(",") if s]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")
    work = (args.out or Path(tempfile.mkdtemp(prefix="verify-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base_sha = git("rev-parse", args.baseline)
    base_tree = work / "baseline"
    if not base_tree.exists():
        git("worktree", "add", "--detach", str(base_tree), base_sha)
    ctx = dict(
        work=work,
        base_tree=base_tree,
        approved=json.loads(approved.read_text()) if approved.is_file() else {},
        lines=changed_lines(git("diff", base_sha), git("ls-files", "--others", "--exclude-standard").splitlines()),
    )
    receipt = dict(
        head=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        baseline=dict(ref=args.baseline, sha=base_sha),
        steps={},
    )
    try:
        for name in steps:
            step = Step(name)
            try:
                RUNNERS[name](step, ctx)
            except Exception as error:  # a crashing step is a failure, never a skip
                step.fail(f"{type(error).__name__}: {error}")
            receipt["steps"][name] = step.record()
            print(f"{step.status:4}  {name}  ({receipt['steps'][name]['seconds']} s)", flush=True)
            for line in step.evidence.get("failures", []) + step.evidence.get("warnings", []):
                print(f"      {line}")
            for line in step.evidence.get("checklist", []):
                print(f"      review {line}")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(base_tree)], cwd=REPO, check=False)
    receipt["status"] = "FAIL" if any(s["status"] == "FAIL" for s in receipt["steps"].values()) else "PASS"
    receipt_path = args.receipt or REPO / ".verify/VERIFY.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"{receipt['status']}  receipt: {receipt_path}")
    if not args.out:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if receipt["status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
