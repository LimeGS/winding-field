"""Opt-in, single-writer block solves with durable progress and immutable prefetch."""

import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import threading
import time
import warnings
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import ArrayLike

from constraint_templates import StencilConstraints
from field_block_coarse import CoarseSpace, block_residual
from field_constraints import QuadraticConstraints
from field_prefetch import ObservationPrefetch
from field_runtime import ResourceUnavailable, check_resources
from field_solver import PreparedField, solve_field
from field_store import BlockStore, atomic_json


class AdvisoryIntegrityError(RuntimeError):
    """An observer's verified scientific reader found real corruption."""


def _new_advice(sweep, block, scientific_sha):
    return dict(
        schema="field-observer-v1",
        scientific_sha256=scientific_sha,
        measurement_scope="pre_encoding_owned_observed",
        sweep=sweep,
        next_block=block,
        observed_max_update=None,
        observed_sum_sq=0.0,
        observed_count=0,
        complete=block == 0,
        candidates=[],
        work_s=0.0,
        control_s=0.0,
        timing_complete=block == 0,
    )


def _restore_advice(saved, sweep, block, scientific_sha):
    if saved is None:
        return _new_advice(sweep, block, scientific_sha)
    saved = deepcopy(saved)
    if (
        saved.get("schema") != "field-observer-v1"
        or saved.get("scientific_sha256") != scientific_sha
        or saved.get("sweep") != sweep
        or saved.get("next_block") != block
        or type(saved.get("observed_count")) is not int
        or saved["observed_count"] < 0
        or type(saved.get("complete")) is not bool
        or not isinstance(saved.get("candidates"), list)
        or len(saved["candidates"]) > 32
    ):
        raise ValueError("incompatible advisory accumulator")
    for key in ("observed_sum_sq", "work_s", "control_s"):
        if type(saved.get(key)) not in (int, float) or not math.isfinite(saved[key]) or saved[key] < 0:
            raise ValueError("invalid advisory accumulator")
    maximum = saved.get("observed_max_update")
    if maximum is not None and (type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum < 0):
        raise ValueError("invalid advisory maximum")
    saved["timing_complete"] = False  # The unrecorded last-commit tail cannot be invented after a restart.
    return saved


def default_coarse_workers() -> int:
    """Threads for the coarse residual pass: three quarters of the CPUs, at most 8.

    A worker holds one block's crop arrays; past a few the calling thread's in-order reads
    and constraint rows set the pace.
    """
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(8, math.ceil(count * 0.75)))


def _integer(value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("integer run setting outside bounds")
    return int(value)


def _vector(value, name, positive=False):
    if np.iscomplexobj(value):
        raise ValueError(name + " must be real XYZ")
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError("invalid " + name)
    return result


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _decoded_anchor_error(core, fixed):
    anchor = np.isfinite(fixed)
    if not anchor.any():
        return 0.0
    values = core.astype(np.float64, copy=False)
    offset = values.mean()  # The codec centers the whole core, including nonanchors.
    restored = (values[anchor] - offset).astype(np.float32).astype(np.float64) + offset
    return float(np.max(abs(restored - fixed[anchor]), initial=0))


def _observations(value, shape):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("observation loader must return target, weights, observed, receipts")
    for array, expected, scalar in zip(value[:3], ((*shape, 3), shape, shape), ("float32", "float32", "bool")):
        if isinstance(array, torch.Tensor):
            valid = tuple(array.shape) == tuple(expected) and array.dtype == getattr(torch, scalar)
            finite = bool(torch.isfinite(array).all())
        elif isinstance(array, np.ndarray):
            valid = array.shape == tuple(expected) and array.dtype == np.dtype(scalar)
            finite = bool(np.isfinite(array).all())
        else:
            valid = finite = False
        if not valid or not finite:
            raise ValueError("observation shape/dtype/finite values differ")
    if not bool((value[1] > 0).all()):
        raise ValueError("observation weights must be strictly positive")
    receipts = value[3]
    if not isinstance(receipts, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
        for k, v in receipts.items()
    ):
        raise ValueError("observation receipts must contain named SHA256 digests")
    return value


def _record_sources(seen, receipts):
    for name, digest in receipts.items():
        if name in seen and seen[name] != digest:
            raise ValueError("observation source changed: " + name)
        seen[name] = digest


def _block_order(order, shape, chunk, partial=False):
    raw = np.asarray(order, dtype=object)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("order must be Nx3 integer chunk keys")
    try:
        ordered = np.array([_integer(n) for n in raw.flat], np.int64).reshape(raw.shape)
    except OverflowError as error:
        raise ValueError("order keys exceed integer indexing capacity") from error
    grid_shape = (np.asarray(shape) + chunk - 1) // chunk
    if (
        (len(ordered) != math.prod(map(int, grid_shape)) if not partial else not len(ordered))
        or (ordered >= grid_shape).any()
        or len(np.unique(ordered, axis=0)) != len(ordered)
    ):
        raise ValueError(
            "partial order must list distinct in-grid chunks"
            if partial
            else "order must contain each owned chunk exactly once"
        )
    return ordered


def solve_blocks(
    root: str | Path,
    shape: Sequence[int],
    chunk: int,
    identity: dict[str, Any],
    load_observations: Callable[..., Any],
    *,
    origin_xyz: ArrayLike,
    spacing_xyz: ArrayLike,
    order: Sequence[Sequence[int]],
    partial_order: bool = False,
    constraints: StencilConstraints | Callable[..., QuadraticConstraints | None] | None = None,
    fixed: Callable[..., ArrayLike] | None = None,
    halo: int = 1,
    sweeps: int = 1,
    rtol: float = 1e-5,
    maxiter: int = 4000,
    dtype: Literal["float32", "float64"] = "float64",
    preconditioner: Literal["jacobi", "amg"] = "jacobi",
    nonconvergence_policy: Literal["fail", "continue"] = "fail",
    device: str = "cpu",
    kernel: Literal["eager", "fused"] = "eager",
    cache_bytes: int = 0,
    prepared_cache_bytes: int = 0,
    status_policy: Literal["all", "checkpoints"] = "all",
    read_workers: int = 1,
    max_bytes: int | None = None,
    reserve_bytes: int = 0,
    resume: bool = False,
    stop: Callable[[], bool] | None = None,
    admit: Callable[[], Any] | None = None,
    prefetch: ObservationPrefetch | None = None,
    observer: Any = None,
    coarse_factor: int | None = None,
    coarse_direct_limit: int = 20_000,
    coarse_workers: int | None = None,
    checkpoint: Literal["block", "sweep"] = "block",
    memory_report: dict[str, Any] | None = None,
    colours: int = 0,
    local_workers: int = 1,
    local_devices: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Solve each explicit owned core, persisting its exact next-block cursor.

    The caller's nonempty JSON identity must pin source/provider/code identity,
    coordinate frame, calibration and boundary policy. Geometry, traversal,
    stencil digest and numerical settings are additionally bound automatically.
    Readers are immutable; constraint targets/fixed/exterior values are never prefetched.
    Only immutable gradient observations may be prepared on a secondary device.

    Internal solve faces read the current shared field; global faces are natural
    except caller-supplied exact fixed values. With no supplied fixed provider,
    the global first node is the zero gauge. ``order`` lists every chunk once;
    with ``partial_order`` it lists only the owned chunks, the others are never
    solved or stored, and the required ``fixed`` provider must supply the values
    their nodes take inside an owned block's halo (unset nodes there read as 0).
    Stop/admission are checked at block boundaries and immediately before solve
    and commit. A long solver call is not a hard real-time interruption point.

    Local convergence applies before core quantization, not to the decoded core
    or assembled field. Budget exhaustion is explicitly not global convergence.
    Cache/workers/storage budgets may change on resume; the numerical problem may
    not. The caller budgets observations, solver workspace and optional prefetch
    in ``admit``. This function closes a supplied prefetch client on exit.
    Large fibre archives can supply ``constraints(lo, hi, read_nodes)`` selecting
    complete groups per block. Pin that provider/geometry in ``identity``; it must
    return local QuadraticConstraints and read current exterior values each time.

    Optional prepared_cache_bytes bounds retained CPU/device workspaces, not peak
    memory or allocator reservation. Stable first-fit admission avoids thrashing.
    Callables opt in with prepared_key(lo, hi) returning an immutable string token
    for the static observations/row structure; return None to bypass. Change the
    token when those inputs change. A retained token still checks structural and
    observation equality; targets and fixed values are refreshed every visit.
    Cache state is released on exit, never checkpointed, and may change on resume.
    Nonadmission decisions cost 4096 bytes each from the same budget when room
    remains. Remembered keys solve cold without new snapshots; forgetting a
    decision affects performance only, since no numerical operator is retained.
    ``status_policy='checkpoints'`` publishes the first RUNNING, every committed
    checkpoint and every terminal status, suppressing later observation/solve
    notices. This advisory policy may change on resume. Jacobi (default) and the
    algebraic multigrid ``'amg'`` support the partial fixed-face
    masks used by block solves. Jacobi is an implicit invariant of the v1 run identity;
    ``'amg'`` is recorded in the identity, so a run cannot switch policy on resume.
    The AMG hierarchy is built per block and retained with the prepared workspace.
    Spectral preconditioning is rejected before opening the store.

    ``coarse_factor`` (an integer >= 2) turns the sweeps into two-level multiplicative Schwarz:
    each sweep starts with a pass that gathers the global residual, restricts it with the
    trilinear prolongation from every ``coarse_factor``-th node, solves the Galerkin coarse
    system in memory and adds the prolonged correction to every owned core (see
    :mod:`field_block_coarse`). The coarse operator is assembled once, during the first pass,
    and saved next to the checkpoint. The factor is part of the run identity; a stop inside the
    coarse phase resumes there. Not available with ``partial_order``. Up to
    ``coarse_direct_limit`` unknowns the coarse system is factorized (a sparse LU, paid by every
    process and growing steeply with the size); above it, it is solved by AMG-preconditioned CG
    to 1e-8, whose iterations and convergence each sweep records. The residual pass reads
    observations, fixed values and constraint rows in block order on the calling thread and
    computes each block's residual and restriction on ``coarse_workers`` threads (default
    three quarters of the CPUs, at most 8); parts are added in block order, so the coarse
    right-hand side has the same bytes for any worker count. An execution choice, not part of
    the identity.

    ``nonconvergence_policy='fail'`` (default) stops at the first local solve that
    does not reach ``rtol``. ``'continue'`` commits a local solve that only exhausted
    ``maxiter`` and keeps the sweep budget; numerical breakdowns always fail. The
    cumulative ``nonconverged_solves`` count is checkpointed with progress and every
    status reports it with ``local_solves_converged``. The policy is an execution
    choice, not part of the checkpoint identity, and may change on resume.

    ``checkpoint='sweep'`` keeps committed cores in memory and publishes them once per
    sweep, in the background while the next sweep runs, and before every
    ``CHECKPOINTED_STOP`` or ``SWEEP_BUDGET_REACHED`` status. A crash then resumes from the
    last published sweep instead of the last block. Memory must hold the whole field
    (see :mod:`field_memory`); ``cache_bytes`` should cover it too, or published chunks
    are read back from disk. Not available with an observer. ``memory_report`` is shown
    verbatim as ``STATUS.json["memory"]``. Neither is part of the run identity.

    ``colours=8`` (opt-in; 0 keeps the caller's sequential order) solves the blocks in eight
    colour batches by the parity of their chunk key along each axis (2 x 2 x 2), keeping the
    caller's order inside a batch. Blocks of one colour never read each other's cores through a
    crop (``halo`` must not exceed ``chunk``), so every block of a batch reads the field as it
    was before the batch, pairs and coarse correction included, and the batch commits in block
    order afterwards. Between colours the sweep stays multiplicative. The colouring changes the
    iteration and is part of the run identity; it may need more sweeps than the sequential
    order. It requires ``checkpoint='sweep'`` (a crash resumes from the last published sweep,
    a stop takes effect between batches) and holds one owned core per block of the open batch.
    ``local_workers`` runs that many local solves of a batch at once on threads, each with its
    own CUDA stream, while the main thread prepares the next blocks; ``local_devices`` (for
    example ``["cuda:0", "cuda:1"]``) assigns workers to devices round robin. Neither is part
    of the run identity: the field does not depend on them beyond the device's own rounding.
    """
    if observer is not None and not all(
        callable(getattr(observer, name, None)) for name in ("restore", "prepare", "complete")
    ):
        raise ValueError("observer requires restore/prepare/complete")
    observer_requested = observer is not None
    observer_errors: list[str] = []
    advice_state = None
    prepared_cache: dict[tuple, PreparedField] = {}
    denied_keys: set[tuple] = set()
    prepared = None
    kernel_counts: dict[str, int] = {}  # Solves per kernel actually used (counted this process).
    prepared_stats = dict(
        budget=0,
        retained_bytes=0,
        entries=0,
        peak_bytes=0,
        peak_entries=0,
        hits=0,
        builds=0,
        skipped_snapshots=0,
        denied_entries=0,
        denied_bytes=0,
        peak_denied_entries=0,
        peak_denied_bytes=0,
    )

    def release_prepared():
        nonlocal prepared
        prepared_cache.clear()
        denied_keys.clear()
        prepared = None
        prepared_stats.update(retained_bytes=0, entries=0, denied_entries=0, denied_bytes=0)

    try:
        if preconditioner not in ("jacobi", "amg"):
            raise ValueError(
                "block solves require preconditioner='jacobi' or 'amg'; spectral partial block masks are unsupported"
            )
        if nonconvergence_policy not in ("fail", "continue"):
            raise ValueError("nonconvergence_policy must be fail or continue")
        if kernel not in ("eager", "fused"):
            raise ValueError("kernel must be eager or fused")
        if status_policy not in ("all", "checkpoints"):
            raise ValueError("status_policy must be all or checkpoints")
        prepared_cache_bytes = _integer(prepared_cache_bytes)
        prepared_stats["budget"] = prepared_cache_bytes
        shape = tuple(_integer(n, 2) for n in shape)
        if len(shape) != 3 or math.prod(shape) > np.iinfo(np.intp).max:
            raise ValueError("invalid ZYX run shape")
        chunk, halo, sweeps, maxiter = _integer(chunk, 1), _integer(halo), _integer(sweeps, 1), _integer(maxiter)
        origin, spacing = _vector(origin_xyz, "origin"), _vector(spacing_xyz, "spacing", True)
        if (
            dtype not in ("float32", "float64")
            or not isinstance(rtol, (int, float))
            or not math.isfinite(rtol)
            or rtol <= 0
        ):
            raise ValueError("invalid dtype or positive solver tolerance")
        if not isinstance(identity, dict) or not identity or not callable(load_observations):
            raise ValueError("nonempty source identity and observation loader required")
        if fixed is not None and not callable(fixed):
            raise ValueError("fixed must be a bounds callable")
        for callback in (stop, admit):
            if callback is not None and not callable(callback):
                raise ValueError("stop/admit must be callable")
        ordered = _block_order(order, shape, chunk, partial_order)
        if coarse_factor is not None:
            coarse_factor = _integer(coarse_factor, 2)
            coarse_direct_limit = _integer(coarse_direct_limit)
            if coarse_workers is None:
                coarse_workers = default_coarse_workers()
            coarse_workers = _integer(coarse_workers, 1)
            if partial_order:
                raise ValueError("coarse correction needs the full block order")
        if checkpoint not in ("block", "sweep"):
            raise ValueError("checkpoint must be block or sweep")
        if type(colours) is not int or colours not in (0, 8):
            raise ValueError("colours must be 0 (sequential order) or 8")
        local_workers = _integer(local_workers, 1)
        if local_devices is not None:
            local_devices = [str(torch.device(d)) for d in local_devices]
            if not local_devices:
                raise ValueError("local_devices must name at least one device")
        if colours and checkpoint != "sweep":
            raise ValueError("colour batches require checkpoint='sweep'")
        if colours and halo > chunk:
            raise ValueError("colour batches require halo <= chunk")
        if not colours and (local_workers > 1 or local_devices):
            raise ValueError("parallel local solves require colour batches (colours=8)")
        if checkpoint == "sweep" and observer_requested:
            raise ValueError("sweep checkpoints do not support a field observer")
        if memory_report is not None:
            memory_report = json.loads(json.dumps(memory_report, allow_nan=False))
        if partial_order and fixed is None:
            raise ValueError("partial_order requires a fixed provider for unowned halo nodes")
        if halo == 0 and len(ordered) > 1:
            raise ValueError("a positive halo is required between multiple owned cores")
        constraint_identity = None
        if isinstance(constraints, StencilConstraints):
            if constraints.shape != shape:
                raise ValueError("constraint/grid identity differs")
            constraint_identity = constraints.digest
        elif constraints is not None:
            if not callable(constraints):
                raise ValueError("constraints must be static stencils or a block provider")
            constraint_identity = "caller-bound-provider"
        if prefetch is not None and tuple(prefetch.shape) != shape:
            raise ValueError("prefetch/grid identity differs")
        problem = dict(
            schema="winding-block-solve-v1",
            source_identity=deepcopy(identity),
            shape=list(shape),
            chunk=chunk,
            origin_xyz=origin.tolist(),
            spacing_xyz=spacing.tolist(),
            order_sha256=hashlib.sha256(ordered.astype("<i8").tobytes()).hexdigest(),
            **({"order_scope": "partial"} if partial_order else {}),
            constraints=constraint_identity,
            boundary="caller-fixed" if fixed is not None else "natural-zero-gauge",
            halo=halo,
            sweeps=sweeps,
            rtol=float(rtol),
            maxiter=maxiter,
            dtype=dtype,
            device=str(device),
            # Jacobi stays implicit so v1 identities (and their checkpoints) are unchanged.
            **({} if preconditioner == "jacobi" else {"preconditioner": preconditioner}),
            # Fused kernels round differently, so a run never resumes across kernels.
            **({} if kernel == "eager" else {"kernel": kernel}),
            **({} if coarse_factor is None else {"coarse": dict(kind="trilinear", factor=coarse_factor)}),
            **({"colours": colours} if colours else {}),
        )
        json.dumps(problem, allow_nan=False)
        if colours:
            colour = (ordered[:, 0] % 2) * 4 + (ordered[:, 1] % 2) * 2 + ordered[:, 2] % 2
            ordered = ordered[np.argsort(colour, kind="stable")]
            batch = np.unique(np.sort(colour), return_inverse=True)[1].astype(np.int64)
            batch_starts = [*np.flatnonzero(np.diff(batch, prepend=-1)).tolist(), len(ordered)]
        else:
            batch = np.arange(len(ordered), dtype=np.int64)
            batch_starts = list(range(len(ordered) + 1))
        root, started = Path(root), time.perf_counter()
        prefetched, parity = 0, False
        with BlockStore(
            root,
            shape,
            chunk,
            problem,
            resume=resume,
            cache_bytes=cache_bytes,
            workers=read_workers,
            max_bytes=max_bytes,
            reserve_bytes=reserve_bytes,
            write_back=checkpoint == "sweep",
        ) as store:
            manifest = store.manifest
            progress = manifest["progress"]
            stored_bytes = sum(v["bytes"] for v in manifest["chunks"].values())
            if not progress:
                progress = dict(sweep=1, next_block=0, sweep_max_update=0.0, completed_sweeps=[], source_receipts={})
                store.save_progress(progress)
            nonconverged = progress.get("nonconverged_solves", 0)
            if type(nonconverged) is not int or nonconverged < 0:
                raise ValueError("invalid persisted nonconverged solve count")
            first_sweep = _integer(progress.get("sweep"), 1)
            first_block = _integer(progress.get("next_block"))
            if first_sweep > sweeps or first_block > len(ordered):
                raise ValueError("checkpoint cursor exceeds run budget")
            if first_block not in batch_starts:
                raise ValueError("checkpoint cursor lies inside a colour batch")
            maximum = progress.get("sweep_max_update")
            completed, seen = progress.get("completed_sweeps"), progress.get("source_receipts")
            if (
                type(maximum) not in (int, float)
                or not math.isfinite(maximum)
                or maximum < 0
                or not isinstance(completed, list)
                or not isinstance(seen, dict)
                or len(completed) != first_sweep - int(first_block < len(ordered))
            ):
                raise ValueError("invalid persisted sweep statistics/source receipts")
            for i, value in enumerate(completed, 1):
                if (
                    not isinstance(value, dict)
                    or value.get("sweep") != i
                    or value.get("cores") != len(ordered)
                    or type(value.get("max_core_update")) not in (int, float)
                    or not math.isfinite(value["max_core_update"])
                    or value["max_core_update"] < 0
                ):
                    raise ValueError("invalid completed sweep statistics")
            if any(
                not isinstance(k, str) or not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v)
                for k, v in seen.items()
            ):
                raise ValueError("invalid persisted source receipts")

            scientific_problem = {k: v for k, v in problem.items() if k != "sweeps"}
            scientific_sha = hashlib.sha256(
                json.dumps(scientific_problem, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()

            def snapshot():
                path = root / "CHECKPOINT.json"
                try:
                    raw = path.read_bytes()
                except OSError as error:
                    raise AdvisoryIntegrityError("cannot read scientific checkpoint") from error
                return dict(
                    checkpoint_path=str(path.resolve()),
                    checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                    scientific_sha256=scientific_sha,
                    manifest=store.manifest,
                )

            def observer_error(error):
                message = f"{type(error).__name__}: {error}"
                if len(observer_errors) < 16:
                    observer_errors.append(message)
                    warnings.warn("field advice unavailable: " + message, RuntimeWarning, stacklevel=2)

            if observer is not None:
                try:
                    initial_snapshot = snapshot()
                    advice_state = _restore_advice(
                        observer.restore(initial_snapshot), first_sweep, first_block, scientific_sha
                    )
                except AdvisoryIntegrityError:
                    raise
                except Exception as error:
                    observer_error(error)
                    observer = None

            running_published = False

            def emit(status, **extra):
                nonlocal running_published
                if (
                    status == "RUNNING"
                    and running_published
                    and status_policy == "checkpoints"
                    and extra.get("phase") in ("observations", "solving")
                ):
                    return None
                if status != "RUNNING":
                    release_prepared()
                    if checkpoint == "sweep" and status in ("CHECKPOINTED_STOP", "SWEEP_BUDGET_REACHED"):
                        store.flush(wait=True)  # The terminal status names a published checkpoint.
                result = dict(
                    status=status,
                    pid=os.getpid(),
                    updated_epoch=time.time(),
                    checkpoint=str(root / "CHECKPOINT.json"),
                    sweep=progress["sweep"],
                    next_block=progress["next_block"],
                    total_blocks=len(ordered),
                    completed_sweeps=completed,
                    global_converged=False,
                    physical_acceptance=False,
                    stored_local_residual=None,
                    stored_bytes=stored_bytes,
                    cache_stats=dict(store.stats),
                    prepared_cache_stats=dict(prepared_stats),
                    **({} if kernel == "eager" else {"solver_kernels": dict(kernel_counts)}),
                    prefetched_blocks=prefetched,
                    prefetch_parity=parity,
                    nonconvergence_policy=nonconvergence_policy,
                    nonconverged_solves=nonconverged,
                    local_solves_converged=nonconverged == 0,
                    elapsed_s=time.perf_counter() - started,
                    **({"coarse": dict(coarse_status)} if coarse_factor is not None else {}),
                    checkpoint_policy=dict(policy=checkpoint, pending_chunks=store.pending_chunks, **store.flush_stats),
                    **({} if memory_report is None else {"memory": memory_report}),
                    **extra,
                )
                atomic_json(root / "STATUS.json", result)
                if observer_requested and status != "RUNNING":
                    result.update(advisory=advice_state, advisory_errors=observer_errors)
                if status == "RUNNING":
                    running_published = True
                return result

            boundary_s = 0.0

            def stopped():
                nonlocal boundary_s
                if not observer_requested:
                    return (root / "STOP").exists() or (stop is not None and bool(stop()))
                before_boundary = time.perf_counter()
                try:
                    return (root / "STOP").exists() or (stop is not None and bool(stop()))
                finally:
                    boundary_s += time.perf_counter() - before_boundary

            def admitted():
                check_resources(root, reserve_bytes=reserve_bytes)
                if admit is not None:
                    admit()

            def bounds(key):
                low = key * chunk
                high = np.minimum(low + chunk, shape)
                return low, high, np.maximum(low - halo, 0), np.minimum(high + halo, shape)

            def global_fixed(lo, hi, crop_shape):
                """Exact global fixed values of a crop (NaN where free), as the block solve uses them."""
                values = np.full(crop_shape, np.nan)
                if fixed is None:
                    if not lo.any():
                        values[0, 0, 0] = 0.0
                    return values
                declared = np.asarray(fixed(lo, hi))
                if declared.shape != crop_shape or declared.dtype.kind not in "iuf" or np.isinf(declared).any():
                    raise ValueError("invalid fixed-value observation")
                declared = declared.astype(np.float64)
                label = "fixed:" + ",".join(map(str, (*lo, *hi)))
                _record_sources(seen, {label: hashlib.sha256(declared.tobytes()).hexdigest()})
                valid = np.isfinite(declared)
                values[valid] = declared[valid]
                return values

            def local_rows(lo, hi):
                if constraints is None:
                    rows = None
                elif isinstance(constraints, StencilConstraints):
                    rows = constraints.restrict(lo, hi, store.nodes)
                else:
                    rows = constraints(lo, hi, store.nodes)
                if rows is not None and not isinstance(rows, QuadraticConstraints):
                    raise ValueError("constraint provider must return local QuadraticConstraints")
                return rows

            def exterior_terms(lo, hi, rows):
                """Terms the constraint provider moved out of the crop, when it can name them."""
                if rows is None:
                    return None
                if isinstance(constraints, StencilConstraints):
                    template = constraints.prepare(lo, hi)
                    terms = None if template is None else template.exterior()
                elif callable(getattr(constraints, "exterior", None)):
                    terms = constraints.exterior(lo, hi)
                else:
                    return None
                if terms is None:
                    return None
                row, node, coeff = (np.asarray(a) for a in terms)
                # Exterior fixed values are known only for the zero gauge; others count as free.
                node_free = node != 0 if fixed is None else np.ones(len(node), bool)
                return row, node, node_free, coeff

            space = None
            coarse_status: dict[str, Any] = {}
            operator_path = root / "COARSE_OPERATOR.npz"
            correction_path = root / "COARSE_CORRECTION.npy"

            def file_sha(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            correction = None
            grid_shape = tuple(int(n) for n in (np.asarray(shape) + chunk - 1) // chunk)
            visit = np.full(grid_shape, -1, np.int64)
            visit[tuple(ordered.T)] = batch  # The block, or its colour batch.

            def pending_correction(lo, hi, block):
                """``P e`` on the crop nodes whose chunk this sweep has not committed yet (else 0).

                ``block`` is the visit index of the reading block: its position, or its colour batch.
                """
                out = np.zeros(tuple(map(int, hi - lo)))
                declared = global_fixed(lo, hi, out.shape)
                pending = False
                for key in itertools.product(*(range(a // chunk, (b - 1) // chunk + 1) for a, b in zip(lo, hi))):
                    if visit[key] < block:
                        continue
                    a = np.maximum(np.array(key) * chunk, lo)
                    b = np.minimum((np.array(key) + 1) * chunk, hi)
                    out[tuple(slice(int(x - o), int(y - o)) for x, y, o in zip(a, b, lo))] = 1.0
                    pending = True
                if pending:
                    out *= space.prolong_box(lo, hi, correction)
                    out[np.isfinite(declared)] = 0.0
                return out

            def corrected_nodes(block):
                """A node reader that adds the pending correction outside committed chunks."""

                def read(flat):
                    flat = np.asarray(flat, np.int64)
                    values = np.array(store.nodes(flat), dtype=np.float64)
                    zyx = np.stack(np.unravel_index(flat.ravel(), shape), 1)
                    late = visit[tuple((zyx // chunk).T)] >= block
                    if late.any():
                        free = zyx[late].any(1) if fixed is None else np.ones(int(late.sum()), bool)
                        values.reshape(-1)[late] += space.rows(zyx[late], free) @ correction
                    return values

                return read

            def coarse_phase(sweep):
                """Prepare the coarse correction of ``sweep``; a returned status ends the run.

                The correction is not written in a separate pass: every block read of the sweep adds
                ``P e`` to the chunks the sweep has not committed yet, so the sweep's own commits
                apply it. A resume inside the sweep reloads the saved correction.
                """
                nonlocal space, progress, correction
                state = progress.get("coarse")
                if space is None:
                    space = CoarseSpace(shape, coarse_factor)
                    recorded = progress.get("coarse_operator")
                    if recorded is not None:
                        if not operator_path.is_file() or file_sha(operator_path) != recorded.get("sha256"):
                            raise ValueError("coarse operator file differs from the checkpoint")
                        space.matrix = sp.load_npz(operator_path).tocsr()
                    space.assembled = recorded is not None
                if isinstance(state, dict) and state.get("sweep") == sweep:
                    if not correction_path.is_file() or file_sha(correction_path) != state.get("correction_sha256"):
                        raise ValueError("coarse correction file differs from the checkpoint")
                    correction = np.load(correction_path)
                    coarse_status.update(state, factor=coarse_factor)
                    return None
                phase_started = time.perf_counter()
                assemble = not space.assembled
                rhs = np.zeros(space.size)
                norm_sq = norm0_sq = 0.0

                def residual_part(lo, hi, low, high, target, weights, declared, mask, rows):
                    """Owned-core residual norms and ``P^T r`` of one block; pure, thread-safe."""
                    values = np.where(mask, declared, store.read(lo, hi))
                    r, r0 = block_residual(target, weights, spacing, rows, values, mask, device)
                    owned = tuple(slice(int(a), int(b)) for a, b in zip(low - lo, high - lo))
                    r_core, r0_core = r[owned], r0[owned]
                    window, part = space.restrict_box_part(low, high, r_core, ~mask[owned])
                    return float(np.sum(r_core * r_core)), float(np.sum(r0_core * r0_core)), window, part

                def reduce(item):
                    """Add one block's part in block order; assemble its operator rows if needed."""
                    nonlocal norm_sq, norm0_sq
                    lo, low, high, weights, mask, rows, exterior, job = item
                    block_norm, block_norm0, window, part = job.result() if workers else job
                    norm_sq += block_norm
                    norm0_sq += block_norm0
                    rhs.reshape(space.coarse_shape)[window] += part
                    if assemble:
                        space.add_block(lo, low, high, weights, ~mask, rows, exterior=exterior)

                # Providers (observations, fixed values, constraint rows) are called here, in block
                # order: they may cache per caller. Workers only compute on the arrays they get.
                workers = coarse_workers if coarse_workers > 1 and len(ordered) > 1 else 0
                pool = ThreadPoolExecutor(workers, thread_name_prefix="coarse-residual") if workers else None
                inflight: deque = deque()
                try:
                    for block, key in enumerate(ordered):
                        if stopped():
                            return emit("CHECKPOINTED_STOP")
                        admitted()
                        coarse_status.update(stage="residual", sweep=sweep, next_block=block)
                        emit("RUNNING", phase="coarse-residual", active_sweep=sweep, active_block=block)
                        low, high, lo, hi = bounds(key)
                        crop_shape = tuple(map(int, hi - lo))
                        target, weights, _, receipts = _observations(load_observations(lo, hi), crop_shape)
                        _record_sources(seen, receipts)
                        target, weights = _numpy(target), _numpy(weights)
                        declared = global_fixed(lo, hi, crop_shape)
                        mask = np.isfinite(declared)
                        rows = local_rows(lo, hi)
                        exterior = exterior_terms(lo, hi, rows) if assemble else None
                        arguments = (lo, hi, low, high, target, weights, declared, mask, rows)
                        job = pool.submit(residual_part, *arguments) if pool else residual_part(*arguments)
                        inflight.append((lo, low, high, weights, mask, rows, exterior, job))
                        del target, weights, declared, mask, rows, exterior, arguments, job
                        while inflight and (len(inflight) > 2 * workers or inflight[0][-1].done()):
                            reduce(inflight.popleft())
                    while inflight:
                        reduce(inflight.popleft())
                finally:
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                    inflight.clear()
                if assemble:
                    sp.save_npz(operator_path, space.matrix, compressed=False)
                    space.assembled = True
                    progress = dict(progress, coarse_operator=dict(sha256=file_sha(operator_path)))
                    store.save_progress(progress)
                if space.solve is None:
                    coarse_status.update(stage="solve")
                    emit("RUNNING", phase="coarse-solve", active_sweep=sweep)
                    space.solve, space.info = space.solver(coarse_direct_limit)
                if stopped():
                    return emit("CHECKPOINTED_STOP")
                solve_started = time.perf_counter()
                correction = space.solve(rhs)
                solve_s = time.perf_counter() - solve_started
                np.save(correction_path, correction)
                norm, norm0 = math.sqrt(norm_sq), math.sqrt(norm0_sq)
                state = dict(
                    sweep=sweep,
                    stage="apply",
                    correction_sha256=file_sha(correction_path),
                    residual_norm=norm,
                    relative_residual=norm / norm0 if norm0 else None,
                    solve_s=solve_s,
                    seconds=time.perf_counter() - phase_started,
                    **(space.last or {}),
                )
                progress = dict(progress, coarse=state)
                store.save_progress(progress)
                coarse_status.update(state, **(space.info or {}), factor=coarse_factor, next_block=0)
                return None

            segment_started = time.perf_counter()
            segment_boundary_start = boundary_s
            segment_prefix = advice_state["work_s"] if advice_state is not None else 0.0

            def completed_advice():
                nonlocal observer, advice_state
                if observer is None or advice_state is None:
                    return
                started_observer = time.perf_counter()
                try:
                    event = snapshot()
                    metrics = deepcopy(advice_state)
                    count = metrics["observed_count"]
                    metrics["observed_rms_update"] = math.sqrt(metrics["observed_sum_sq"] / count) if count else None
                    metrics["active_wall_s"] = (
                        segment_prefix + time.perf_counter() - segment_started - (boundary_s - segment_boundary_start)
                    )
                    metrics["timing_scope"] = "active sweep including commits/observer; stop callbacks excluded"
                    event["metrics"] = metrics
                    observer.complete(event)  # Void: never a stop/sweep/publication decision.
                except AdvisoryIntegrityError:
                    raise
                except Exception as error:
                    observer_error(error)
                    observer = None
                    advice_state = None
                    return
                advice_state["control_s"] += time.perf_counter() - started_observer
                advice_state["active_wall_s"] = (
                    segment_prefix + time.perf_counter() - segment_started - (boundary_s - segment_boundary_start)
                )

            def local_device(slot):
                """The device of local solve worker ``slot`` (round robin over ``local_devices``)."""
                return device if not local_devices else local_devices[slot % len(local_devices)]

            def prepare_block(sweep, block, stamp, gate):
                """Read everything one local solve needs (main thread); a returned str is a stop.

                ``stamp`` is the visit index the coarse correction compares against: the block itself,
                or its colour batch, whose blocks all read the field as it was before the batch.
                ``gate`` checks stop requests here; colour batches check them between batches only.
                """
                nonlocal prefetched, parity, prepared
                if gate and stopped():
                    return "stop"
                admitted()
                key = ordered[block]
                low, high, lo, hi = bounds(key)
                crop_shape = tuple(map(int, hi - lo))
                emit("RUNNING", phase="observations", active_sweep=sweep, active_block=block)
                fetched = None
                if prefetch is not None and prefetch.available:
                    if prefetch.pending is None:
                        prefetch.submit(lo, hi)
                    fetched = prefetch.receive(lo, hi)
                if fetched is None:
                    values = _observations(load_observations(lo, hi), crop_shape)
                else:
                    values = _observations(fetched, crop_shape)
                    if not parity:
                        primary = _observations(load_observations(lo, hi), crop_shape)
                        if (
                            any(not np.array_equal(_numpy(a), _numpy(b)) for a, b in zip(values[:3], primary[:3]))
                            or values[3] != primary[3]
                        ):
                            raise ValueError("primary/secondary observation parity failed")
                        parity = True
                        del primary
                    prefetched += 1
                target, weights, observed, receipts = values
                _record_sources(seen, receipts)
                if prefetch is not None and prefetch.available and block + 1 < len(ordered):
                    _, _, next_lo, next_hi = bounds(ordered[block + 1])
                    prefetch.submit(next_lo, next_hi)
                current = store.read(lo, hi)
                reader = store.nodes
                if correction is not None:
                    current += pending_correction(lo, hi, stamp)
                    reader = corrected_nodes(stamp)
                    coarse_status.update(stage="apply", next_block=block)
                if constraints is None:
                    rows = None
                elif isinstance(constraints, StencilConstraints):
                    rows = constraints.restrict(lo, hi, reader)
                else:
                    rows = constraints(lo, hi, reader)
                if rows is not None and not isinstance(rows, QuadraticConstraints):
                    raise ValueError("constraint provider must return local QuadraticConstraints")
                fixed_values = np.full(crop_shape, np.nan)
                for axis in range(3):
                    for side, internal in ((0, lo[axis] > 0), (-1, hi[axis] < shape[axis])):
                        if internal:
                            plane: list[slice | int] = [slice(None)] * 3
                            plane[axis] = side
                            fixed_values[tuple(plane)] = current[tuple(plane)]
                if fixed is None:
                    if not lo.any():
                        fixed_values[0, 0, 0] = 0.0
                else:
                    declared = np.asarray(fixed(lo, hi))
                    if declared.shape != crop_shape or declared.dtype.kind not in "iuf" or np.isinf(declared).any():
                        raise ValueError("invalid fixed-value observation")
                    declared = declared.astype(np.float64)
                    label = "fixed:" + ",".join(map(str, (*lo, *hi)))
                    _record_sources(seen, {label: hashlib.sha256(declared.tobytes()).hexdigest()})
                    valid = np.isfinite(declared)
                    fixed_values[valid] = declared[valid]
                if gate and stopped():
                    return "stop"
                admitted()
                emit("RUNNING", phase="solving", active_sweep=sweep, active_block=block)
                token = "static" if constraints is None or isinstance(constraints, StencilConstraints) else None
                key_provider = getattr(constraints, "prepared_key", None)
                if prepared_cache_bytes and callable(key_provider):
                    token = key_provider(lo, hi)
                    if token is not None and (not isinstance(token, str) or len(token) > 256):
                        raise ValueError("prepared_key must return a string of at most 256 characters or None")
                prepared_key = (*map(int, (*lo, *hi)), token)
                prepared = prepared_cache.get(prepared_key)
                if prepared is not None:
                    prepared_stats["hits"] += 1
                else:
                    prepared_stats["builds"] += 1
                    if prepared_cache_bytes and token is not None:
                        if prepared_key in denied_keys:
                            prepared_stats["skipped_snapshots"] += 1
                        else:
                            prepared = PreparedField()
                job = dict(
                    block=block,
                    key=key,
                    low=low,
                    lo=lo,
                    owned=tuple(slice(int(a), int(b)) for a, b in zip(low - lo, high - lo)),
                    target=target,
                    weights=weights,
                    observed=observed,
                    current=current,
                    rows=rows,
                    fixed_values=fixed_values,
                    prepared=prepared,
                    prepared_key=prepared_key,
                )
                prepared = None
                return job

            def solve_block(job, slot=0):
                """The local solve of a prepared block; safe on a worker thread (own CUDA stream).

                Returns the owned core and everything the commit needs, so the crop arrays can go.
                """
                where = torch.device(local_device(slot))
                context = contextlib.ExitStack()
                if where.type == "cuda" and (local_workers > 1 or local_devices):
                    streams = getattr(worker_state, "streams", None)
                    if streams is None:
                        streams = worker_state.streams = {}
                    if where not in streams:
                        streams[where] = torch.cuda.Stream(where)
                    context.enter_context(torch.cuda.device(where))
                    context.enter_context(torch.cuda.stream(streams[where]))
                with context:
                    result, solver = solve_field(
                        job["target"],
                        job["weights"],
                        spacing,
                        constraints=job["rows"],
                        fixed=job["fixed_values"],
                        initial=job["current"],
                        device=str(where) if local_devices else device,
                        dtype=dtype,
                        preconditioner=preconditioner,
                        kernel=kernel,
                        rtol=rtol,
                        maxiter=maxiter,
                        return_tensor=True,
                        prepared=job["prepared"],
                    )
                    owned = job["owned"]
                    core = _numpy(result[owned])
                del result
                delta = core - job["current"][owned]
                return dict(
                    core=core,
                    delta=delta if observer is not None else None,
                    change=float(np.max(abs(delta))),
                    solver=solver,
                    owned_observed=_numpy(job["observed"][owned]),
                    fixed_error=_decoded_anchor_error(core, job["fixed_values"][owned]),
                )

            def commit_block(sweep, job, solved, block_started, block_boundary_start):
                """Account for and commit one solved block (main thread, in block order)."""
                nonlocal maximum, progress, completed, nonconverged, stored_bytes, observer, advice_state
                block, key, prepared_key = job["block"], job["key"], job["prepared_key"]
                prepared_job, solver = job["prepared"], solved["solver"]
                if prepared_job is not None and prepared_key not in prepared_cache:
                    retained = prepared_job.retained_bytes + 4096  # Bounds key and cache-container overhead.
                    if retained <= prepared_cache_bytes - prepared_stats["retained_bytes"]:
                        prepared_cache[prepared_key] = prepared_job
                        prepared_stats["retained_bytes"] += retained
                        prepared_stats["entries"] = len(prepared_cache)
                        prepared_stats["peak_bytes"] = prepared_stats["retained_bytes"]
                        prepared_stats["peak_entries"] = len(prepared_cache)
                    elif 4096 <= prepared_cache_bytes - prepared_stats["retained_bytes"]:
                        denied_keys.add(prepared_key)
                        prepared_stats["denied_entries"] = len(denied_keys)
                        prepared_stats["denied_bytes"] += 4096
                        prepared_stats["retained_bytes"] += 4096
                        prepared_stats["peak_denied_entries"] = len(denied_keys)
                        prepared_stats["peak_denied_bytes"] = prepared_stats["denied_bytes"]
                        prepared_stats["peak_bytes"] = prepared_stats["retained_bytes"]
                job["prepared"] = None
                if kernel != "eager":
                    used = solver.get("kernel", "eager")
                    kernel_counts[used] = kernel_counts.get(used, 0) + 1
                block_converged = bool(solver["converged"])
                if not block_converged and (nonconvergence_policy != "continue" or solver.get("reason") != "maxiter"):
                    raise RuntimeError("conditional block did not converge")
                core, change, owned_observed = solved["core"], solved["change"], solved["owned_observed"]
                maximum = max(maximum, change)
                if observer is not None:
                    telemetry_started = time.perf_counter()
                    delta, selected = solved.pop("delta"), None
                    try:
                        assert advice_state is not None and block_started is not None
                        selected = delta[owned_observed]
                        square_sum = float(np.sum(selected * selected, dtype=np.float64))
                        cumulative = advice_state["observed_sum_sq"] + square_sum
                        if not math.isfinite(cumulative):
                            raise ArithmeticError("nonfinite advisory sum of squares")
                        advice_state["observed_sum_sq"] = cumulative
                        advice_state["observed_count"] += int(selected.size)
                        if selected.size:
                            local_max = float(np.max(abs(selected)))
                            advice_state["observed_max_update"] = max(
                                advice_state["observed_max_update"] or 0.0, local_max
                            )
                            # One representative per block, bounded deterministic rank.
                            ij = np.asarray(np.unravel_index(int(owned_observed.argmax()), core.shape)) + job["low"]
                            rank = hashlib.sha256(key.astype("<i8").tobytes()).hexdigest()
                            candidate = dict(node=int(np.ravel_multi_index(tuple(ij), shape)), rank=rank)
                            advice_state["candidates"] = sorted(
                                advice_state["candidates"] + [candidate], key=lambda value: value["rank"]
                            )[:32]
                        advice_state["next_block"] = block + 1
                        advice_state["control_s"] += time.perf_counter() - telemetry_started
                        advice_state["work_s"] += (
                            time.perf_counter() - block_started - (boundary_s - block_boundary_start)
                        )
                    except Exception as error:
                        observer_error(error)
                        advice_state = None
                        observer = None
                    finally:
                        del selected, delta
                next_completed = list(completed)
                if block + 1 == len(ordered):
                    stats = dict(sweep=sweep, cores=len(ordered), max_core_update=maximum)
                    if coarse_factor is not None:
                        opened = progress.get("coarse") or {}
                        stats.update(relative_residual=opened.get("relative_residual"), coarse_s=opened.get("seconds"))
                    next_completed.append(stats)
                # Report precision loss separately from the pre-encoding local gate.
                pending = dict(
                    sweep=sweep,
                    next_block=block + 1,
                    sweep_max_update=maximum,
                    completed_sweeps=next_completed,
                    source_receipts=seen,
                    nonconverged_solves=nonconverged + (not block_converged),
                    solver=dict(solver, scope="pre_encoding_local_solve"),
                    core_max_update=change,
                    decoded_fixed_value_error_max=solved["fixed_error"],
                    global_converged=False,
                    physical_acceptance=False,
                    **{k: progress[k] for k in ("coarse", "coarse_operator") if k in progress},
                )
                admitted()
                if observer is None:
                    commit = store.commit(key, core, pending, owned_observed)
                else:

                    def prepare(event):
                        observed_at = time.perf_counter()
                        try:
                            observer.prepare(event, deepcopy(advice_state))
                        finally:
                            advice_state["control_s"] += time.perf_counter() - observed_at

                    commit = store.commit(key, core, pending, owned_observed, prepare=prepare)
                    if commit["prepare_error"]:
                        observer_error(RuntimeError(commit["prepare_error"]))
                stored_bytes = commit["stored_bytes"]
                progress, completed = pending, next_completed
                nonconverged = pending["nonconverged_solves"]
                if block + 1 == len(ordered):
                    completed_advice()
                    if coarse_factor is not None:
                        coarse_status.update(stage="done", next_block=len(ordered))
                    # Publish the finished sweep while the next one starts from memory.
                    store.flush(wait=False)
                emit("RUNNING", phase="checkpointed")

            def colour_batch(sweep, start, end):
                """Solve blocks ``start:end`` (one colour) against the field as it was before them.

                Every read of the batch happens before its first commit, and commits follow block
                order, so the field does not depend on how many workers ran or in which order the
                solves finished. Up to ``2 * local_workers`` prepared blocks wait for a worker.
                """
                nonlocal warm
                solved: dict[int, dict[str, Any]] = {}
                jobs: dict[int, dict[str, Any]] = {}
                if local_workers == 1:
                    for block in range(start, end):
                        job = prepare_block(sweep, block, batch[block], gate=False)
                        solved[block] = solve_block(job)
                        jobs[block] = dict(
                            job, target=None, weights=None, observed=None, current=None, rows=None, fixed_values=None
                        )
                        del job
                else:
                    waiting: deque = deque()

                    def collect():
                        block, future = waiting.popleft()
                        solved[block] = future.result()
                        jobs[block].update(
                            target=None, weights=None, observed=None, current=None, rows=None, fixed_values=None
                        )

                    for block in range(start, end):
                        job = prepare_block(sweep, block, batch[block], gate=False)
                        jobs[block] = job
                        waiting.append((block, pool.submit(worker, job)))
                        del job
                        # The first solve of a process runs alone: kernels and probes initialise once.
                        while waiting and (not warm or len(waiting) >= 2 * local_workers):
                            collect()
                            warm = True
                    while waiting:
                        collect()
                for block in range(start, end):
                    commit_block(sweep, jobs.pop(block), solved.pop(block), None, boundary_s)
                return None

            worker_state = threading.local()
            worker_slots = itertools.count()
            warm = False
            pool = ThreadPoolExecutor(local_workers, thread_name_prefix="block-solve") if local_workers > 1 else None

            def worker(job):
                slot = getattr(worker_state, "slot", None)
                if slot is None:
                    slot = worker_state.slot = next(worker_slots)
                return solve_block(job, slot)

            try:
                if first_block == len(ordered):
                    completed_advice()
                if stopped():
                    return emit("CHECKPOINTED_STOP")
                if first_block == len(ordered):
                    first_sweep, first_block, maximum = first_sweep + 1, 0, 0.0
                for sweep in range(first_sweep, sweeps + 1):
                    if sweep != first_sweep:
                        maximum = 0.0
                    if observer is not None and (advice_state is None or advice_state["sweep"] != sweep):
                        advice_state = _new_advice(sweep, 0, scientific_sha)
                        segment_started, segment_prefix = time.perf_counter(), 0.0
                        segment_boundary_start = boundary_s
                    if coarse_factor is not None:
                        finished = coarse_phase(sweep)
                        if finished is not None:
                            return finished
                    begin = first_block if sweep == first_sweep else 0
                    if colours:
                        for start, end in zip(batch_starts[:-1], batch_starts[1:]):
                            if end <= begin:
                                continue
                            if stopped():
                                return emit("CHECKPOINTED_STOP")
                            colour_batch(sweep, start, end)
                        continue
                    for block in range(begin, len(ordered)):
                        block_started = time.perf_counter() if observer is not None else None
                        block_boundary_start = boundary_s
                        job = prepare_block(sweep, block, block, gate=True)
                        if isinstance(job, str):
                            return emit("CHECKPOINTED_STOP")
                        solved = solve_block(job)
                        commit_block(sweep, job, solved, block_started, block_boundary_start)
                        del job, solved
                return emit("SWEEP_BUDGET_REACHED")
            except Exception as error:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)  # No solve outlives the failure.
                emit("RESOURCE_UNAVAILABLE" if isinstance(error, ResourceUnavailable) else "FAILED", error=str(error))
                raise
            finally:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
    finally:
        release_prepared()
        if prefetch is not None:
            prefetch.close()
