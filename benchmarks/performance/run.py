"""Opt-in CPU throughput evidence. Each case imports exactly one source tree in a child.

Run with the SAME interpreter for baseline and candidate; see --help. Outputs must
be fresh external directories. No installation, datasets or production edits.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_hash(value) -> str:
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def array_hash(value) -> str:
    import numpy as np

    a = np.ascontiguousarray(value)
    return digest(json.dumps([a.shape, a.dtype.str]).encode() + a.tobytes())


def summarize(samples) -> dict:
    import numpy as np

    if not samples or not np.isfinite(samples).all() or min(samples) < 0:
        raise ValueError("finite nonnegative timing samples required")
    q1, q3 = np.percentile(samples, [25, 75])
    return dict(
        samples_s=list(samples),
        median_s=float(np.median(samples)),
        iqr_s=float(q3 - q1),
        min_s=float(min(samples)),
        max_s=float(max(samples)),
    )


def source_identity(root: Path, commit: str | None = None) -> dict:
    files = {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted((root / "src").rglob("*.py"))}
    if not files:
        raise ValueError("source must contain src/*.py")
    diff = None
    status = None
    if (root / ".git").exists():
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        if commit is not None and commit != head:
            raise ValueError("commit override does not match checkout HEAD")
        commit = head
        diff = digest(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root))
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root).decode()
    return dict(
        root=str(root),
        commit=commit,
        files=files,
        source_sha256=json_hash(files),
        dirty_diff_sha256=diff,
        git_status=status,
    )


def environment() -> dict:
    import torch

    def version(dist):
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    return dict(
        device="cpu",
        cuda_runtime=torch.version.cuda,
        cuda_driver="not queried (CPU-only harness)",
        cuda_validation=False,
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu=cpu,
        logical_cpus=os.cpu_count(),
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        thread_environment={k: os.environ.get(k) for k in THREAD_ENV},
        versions={
            "python": platform.python_version(),
            **{
                name: version(dist)
                for name, dist in [
                    ("numpy", "numpy"),
                    ("scipy", "scipy"),
                    ("torch", "torch"),
                    ("skimage", "scikit-image"),
                ]
            },
        },
        timer=dict(name="perf_counter_ns", monotonic=time.get_clock_info("perf_counter").monotonic),
    )


class Stages:
    """Inclusive wrapper durations: nested or concurrent calls must NOT be summed."""

    def __init__(self):
        self.samples = {}
        self.stack = ExitStack()

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.samples.setdefault(label, []).append((time.perf_counter_ns() - start) / 1e9)

        self.stack.enter_context(patch.object(owner, name, measured))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stack.close()


THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CASES = (
    "fibres-800",
    "fibres-2000",
    "fibres-4000",
    "pcg",
    "pcg-prepared",
    "pcg-large",
    "blocks",
    "blocks-constraints",
    "mesh",
    "arrival",
    "mesh-offrange",
    "mesh-large",
    "arrival-0",
    "arrival-70000",
    "arrival-120000",
    "arrival-1000000",
)


def worker(args) -> None:
    import numpy as np
    import torch
    from cases import make_case

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(args.source)
    before = source_identity(root, args.commit)
    output = Path(args.output)
    expected = json.loads((output / "run.json").read_text())["source"]
    if before["files"] != expected["files"]:
        raise RuntimeError("source changed between cases; use a frozen candidate or rerun")
    case_output = output / args.worker
    case_output.mkdir()
    config, inputs, operation, instrument = make_case(args.worker, case_output, args.seed, args.prepared_cache_bytes)

    def timed():
        start = time.perf_counter_ns()
        result = operation()
        return (time.perf_counter_ns() - start) / 1e9, result

    cold, cold_result = timed()
    warmup, _ = timed()
    samples = []
    for _ in range(args.repetitions):
        seconds, validation = timed()
        samples.append(seconds)
    # Memory and instrumentation are separate passes so they cannot inflate samples.
    tracemalloc.start()
    operation()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    with Stages() as stages:
        instrument(stages)
        diagnostic_seconds, diagnostic = timed()
    after = source_identity(root, args.commit)
    if before["files"] != after["files"]:
        raise RuntimeError("source changed during measurement; discard this run")
    imported = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and name in {Path(p).stem for p in before["files"]}:
            resolved = Path(file).resolve()
            if not resolved.is_relative_to(root / "src"):
                raise RuntimeError(f"source isolation failed: {name}: {file}")
            imported[name] = str(resolved)
    result = dict(
        schema="winding-cpu-throughput-v1",
        case=args.worker,
        seed=args.seed,
        source=expected,
        harness_hashes={p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")},
        config=config,
        config_sha256=json_hash(config),
        input_hashes={k: array_hash(v) for k, v in inputs.items()},
        environment=environment(),
        imported_sources=imported,
        cold_s=cold,
        warmup_s=warmup,
        timing=summarize(samples),
        cold_validation=cold_result,
        validation=validation,
        memory=dict(
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            tracemalloc_peak_bytes=peak,
            tracemalloc_current_bytes=current,
            scope="RSS is process lifetime high-water including imports/all passes; tracemalloc is one separate warm pass, excludes untracked native allocations. Neither is retained-cache size.",
        ),
        stages=dict(
            non_additive=True,
            scope="Separate instrumented warm pass; inclusive calls may nest. Never sum with wall time or concurrent prefetch.",
            wall_s=diagnostic_seconds,
            calls_s=stages.samples,
            validation=diagnostic,
        ),
        limitations=[
            "Cold means first invocation after imports/input creation, not a flushed OS page cache.",
            "Synthetic CPU throughput only; no physical acceptance or CUDA evidence.",
            "PCG envelope spans first RHS norm through final residual norm; output/receipt tail includes CPU conversion, not a CUDA transfer. Nested matvec/setup timings are non-additive.",
            "Wall samples include small output validation/digest costs. Input generation/imports are excluded.",
            "Baseline inline mesh support/range diagnostic is line-traced and excludes later mask construction; candidate helper includes mask construction. Compare call counts, not their durations as identical regions.",
        ],
    )
    (output / f"{args.worker}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"{args.worker}: median={np.median(samples):.6f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="checkout or frozen root containing src/")
    parser.add_argument("--output", required=True, help="new external evidence directory; never overwritten")
    parser.add_argument("--commit", help="commit identity for frozen src snapshot")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=[c for c in CASES if not c.startswith("mesh")],
        help="core cases by default; mesh cases require historical surface-enabled --source",
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument(
        "--prepared-cache-bytes", type=int, default=0, help="blocks only; nonzero requires candidate public API"
    )
    parser.add_argument("--worker", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source, args.output = str(Path(args.source).resolve()), str(Path(args.output).resolve())
    if args.repetitions < 1 or args.prepared_cache_bytes < 0:
        parser.error("positive repetitions and nonnegative cache budget required")
    if args.worker:
        worker(args)
        return
    source = source_identity(Path(args.source), args.commit)
    output = Path(args.output)
    checkout = Path(__file__).resolve().parents[2]
    if output.is_relative_to(checkout) or output.is_relative_to(Path(args.source) / "src"):
        parser.error("evidence must be outside the checkout/source")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(
        json.dumps(dict(source=source, arguments=vars(args), command=sys.argv, interpreter=sys.executable), indent=2)
    )
    # Preserve exact inputs to measurement, including uncommitted candidate files.
    for relative, expected_hash in source["files"].items():
        data = (Path(args.source) / relative).read_bytes()
        if digest(data) != expected_hash:
            raise RuntimeError("source changed while snapshotting")
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    scripts = output / "harness"
    scripts.mkdir()
    for file in Path(__file__).parent.glob("*.py"):
        (scripts / file.name).write_bytes(file.read_bytes())
    env = dict(
        os.environ,
        PYTHONPATH=str(output / "source/src"),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONHASHSEED=str(args.seed),
    )
    env.update({k: "1" for k in THREAD_ENV})
    for case in args.cases:
        command = [
            sys.executable,
            str(scripts / "run.py"),
            "--source",
            str(output / "source"),
            "--output",
            args.output,
            "--worker",
            case,
            "--seed",
            str(args.seed),
            "--repetitions",
            str(args.repetitions),
            "--prepared-cache-bytes",
            str(args.prepared_cache_bytes),
        ]
        if args.commit:
            command += ["--commit", args.commit]
        with (output / f"{case}.log").open("w") as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print((output / f"{case}.log").read_text(), end="")
    summary = {}
    for case in args.cases:
        record = json.loads((output / f"{case}.json").read_text())
        summary[case] = {key: record[key] for key in ("timing", "cold_s", "warmup_s", "config_sha256", "input_hashes")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
