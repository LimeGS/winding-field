"""Balanced CPU measurements in isolated, idle-between-command worker processes.

Example: python benchmarks/performance/v3.py --baseline /frozen/base
  --candidate . --output /new/external/evidence --repetitions 8
"""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from run import THREAD_ENV, Stages, digest, environment, source_identity, summarize

SOLVERS = {"pairs", "consensus", "jump", "faces", "tiny"}
CASES = [
    "pairs",
    "consensus",
    "jump",
    "faces",
    "tiny",
    "templates",
    "cache-abc",
    "store",
    "gather",
    "blocks",
    "blocks-constraints",
]


def balanced_order(labels, repetition):
    shift = repetition % len(labels)
    return labels[shift:] + labels[:shift]


def check_structural_metadata(case, actual, expected):
    if case in {"templates", "cache-abc"}:
        assert actual["digests"] == expected["digests"]
    elif case in {"blocks", "blocks-constraints"}:
        for key in ("chunk_sha256", "completed_sweeps", "anchor_error"):
            assert actual[key] == expected[key], key
        check_structural_metadata("pairs", actual["solver"], expected["solver"])
    elif case in SOLVERS:
        for key in ("converged", "iterations", "relative_residual", "gauge", "reason", "dtype"):
            assert actual[key] == expected[key], key


def instrument(stages):
    from cases import solver_stages

    import block_solver
    import field_solver
    from constraint_templates import StencilConstraints
    from field_store import BlockStore

    solver_stages(stages)
    if hasattr(field_solver, "_SpectralPreconditioner"):
        stages.wrap(field_solver._SpectralPreconditioner, "__init__", "spectral_setup")
        stages.wrap(field_solver._SpectralPreconditioner, "apply", "spectral_apply")
    for owner, method, label in [
        (BlockStore, "commit", "commit_inclusive"),
        (BlockStore, "read", "read_inclusive"),
        (block_solver, "atomic_json", "status_publication"),
        (StencilConstraints, "_build", "template_build"),
    ]:
        stages.wrap(owner, method, label)


def worker(args):
    import numpy as np
    import torch
    from v3_cases import make_case

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    operation, validate, config, retained = make_case(args.worker, Path(args.output), args.policy, args.size)
    import field_solver

    if not Path(field_solver.__file__).resolve().is_relative_to(Path(args.source).resolve()):
        raise RuntimeError("wrong source import")
    initial_source = source_identity(Path(args.source))
    provenance = json.loads((Path(args.output).parents[1] / "provenance.json").read_text())
    if initial_source["files"] != provenance["variants"][Path(args.source).name]["files"]:
        raise RuntimeError("worker source differs from recorded provenance")
    print(
        json.dumps(dict(ready=True, config=config, environment=environment(), imported_core=field_solver.__file__)),
        flush=True,
    )
    for line in sys.stdin:
        command = json.loads(line)
        if command["mode"] == "stop":
            break
        if command["mode"] == "diagnostic":
            tracemalloc.start()
            operation()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            with Stages() as stages:
                instrument(stages)
                start = time.perf_counter_ns()
                result = operation()
                elapsed = (time.perf_counter_ns() - start) / 1e9
            field, info = validate(result)
            response = dict(
                instrumented_wall_s=elapsed,
                non_additive_calls_s=stages.samples,
                memory=dict(
                    tracemalloc_peak_bytes=peak,
                    tracemalloc_current_bytes=current,
                    lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                    **retained(result),
                    scope="Retained cache storage from diagnostic result; output arrays excluded. Separate prepared workspace is an extra untimed solve. Tracemalloc excludes untracked native allocations; RSS is lifetime high-water including imports and all passes.",
                ),
                validation=info,
            )
        else:
            start = time.perf_counter_ns()
            result = operation()
            elapsed = (time.perf_counter_ns() - start) / 1e9
            # Validation, hashes and file writes are deliberately outside timing.
            field, info = validate(result)
            path = Path(args.output) / (command["label"] + ".npy")
            np.save(path, field)
            response = dict(seconds=elapsed, field=str(path), sha256=digest(field.tobytes()), validation=info)
        if initial_source["files"] != source_identity(Path(args.source))["files"]:
            raise RuntimeError("frozen source changed")
        imported = {}
        project_names = {Path(p).stem for p in initial_source["files"]}
        for name, module in tuple(sys.modules.items()):
            if name in project_names and getattr(module, "__file__", None):
                path = Path(module.__file__).resolve()
                if not path.is_relative_to(Path(args.source) / "src"):
                    raise RuntimeError(f"wrong imported source: {name}: {path}")
                imported[name] = str(path)
        response["imported_sources"] = imported
        print(json.dumps(response, allow_nan=False), flush=True)


def freeze(source, destination):
    destination.mkdir()
    shutil.copytree(source / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__"))


def variants(output, baseline, candidate):
    roots = output / "sources"
    roots.mkdir()
    for name, source in [
        ("baseline", baseline),
        ("combined", candidate),
        ("indices", baseline),
        ("spectral", candidate),
    ]:
        freeze(source, roots / name)
    shutil.copyfile(candidate / "src/field_constraints.py", roots / "indices/src/field_constraints.py")
    shutil.copyfile(baseline / "src/field_constraints.py", roots / "spectral/src/field_constraints.py")
    return {name: roots / name for name in ("baseline", "indices", "spectral", "combined")}


class Worker:
    def __init__(self, source, output, case, policy, size, harness):
        output.mkdir(parents=True)
        self.log = (output / "stderr.log").open("w")
        env = dict(os.environ, PYTHONPATH=str(source / "src"), PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="20261104")
        env.update({name: "1" for name in THREAD_ENV})
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(harness / "v3.py"),
                "--worker",
                case,
                "--source",
                str(source),
                "--output",
                str(output),
                "--policy",
                policy,
                "--size",
                str(size),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            env=env,
            cwd=output,
        )
        self.metadata = self.receive()

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"benchmark worker exited; see {self.log.name}")
        return json.loads(line)

    def call(self, mode, label="diagnostic"):
        self.process.stdin.write(json.dumps(dict(mode=mode, label=label)) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"mode":"stop"}\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=30)
        self.log.close()
        if self.process.returncode:
            raise RuntimeError("worker failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-commit", help="revision of an archived baseline source")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--worker", choices=CASES)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--policy", default="default")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repetitions < 6 or args.repetitions % 4:
        parser.error("repetitions must be >=6 and divisible by four variants")
    if args.baseline is None or args.candidate is None or args.size < 2:
        parser.error("baseline/candidate and size>=2 required")
    output = args.output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("use a new external output directory")
    output.mkdir(parents=True, exist_ok=False)
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    provenance = dict(
        baseline=source_identity(baseline, args.baseline_commit), candidate=source_identity(candidate), command=sys.argv
    )
    roots = variants(output, baseline, candidate)
    provenance["variants"] = {name: source_identity(root) for name, root in roots.items()}
    provenance["harness"] = {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
    shutil.copytree(Path(__file__).parent, output / "harness", ignore=shutil.ignore_patterns("__pycache__"))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    summary = {}
    for case in args.cases:
        choices = (
            [
                ("baseline", "baseline", "default"),
                ("indices", "indices", "default"),
                ("spectral", "spectral", "spectral"),
                ("combined", "combined", "spectral"),
            ]
            if case in SOLVERS
            else [("baseline", "baseline", "default"), ("candidate", "combined", "tuned")]
        )
        workers, record = {}, dict(order=[], variants={})
        try:
            for label, source, policy in choices:
                workers[label] = Worker(
                    roots[source], output / case / label, case, policy, args.size, output / "harness"
                )
                record["variants"][label] = dict(metadata=workers[label].metadata, samples=[])
            for label, worker_process in workers.items():
                item = record["variants"][label]
                item["first_call"] = worker_process.call("timed", "first")
                item["warmup"] = worker_process.call("timed", "warmup")
            labels = list(workers)
            for repetition in range(args.repetitions):
                order = balanced_order(labels, repetition)
                record["order"].append(order)
                for label in order:
                    record["variants"][label]["samples"].append(workers[label].call("timed", f"sample-{repetition}"))
            import numpy as np

            reference = np.load(record["variants"]["baseline"]["samples"][0]["field"])
            for label, worker_process in workers.items():
                item = record["variants"][label]
                item["timing"] = summarize([sample["seconds"] for sample in item["samples"]])
                differences = []
                for sample in item["samples"]:
                    actual = np.load(sample["field"])
                    delta = actual - reference
                    differences.append(
                        dict(max_abs=float(np.max(abs(delta), initial=0)), rms=float(np.sqrt(np.mean(delta**2))))
                    )
                    if case not in SOLVERS or label in {"baseline", "indices"}:
                        np.testing.assert_array_equal(actual, reference)
                        check_structural_metadata(
                            case, sample["validation"], record["variants"]["baseline"]["samples"][0]["validation"]
                        )
                item["differences"] = differences
                item["diagnostic"] = worker_process.call("diagnostic")
            (output / f"{case}.json").write_text(json.dumps(record, indent=2, allow_nan=False))
            summary[case] = {label: item["timing"] for label, item in record["variants"].items()}
            print(case, {label: round(item["median_s"] * 1000, 3) for label, item in summary[case].items()}, flush=True)
        finally:
            for worker_process in workers.values():
                worker_process.close()
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
