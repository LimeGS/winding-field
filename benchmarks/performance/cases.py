"""Small deterministic workloads through existing public entrypoints."""

import hashlib
import inspect
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def solver_stages(stages):
    import field_solver as solver

    original_norm, original_solve = solver._norm, solver.solve_field
    boundaries = []

    def norm(*args, **kwargs):
        start = time.perf_counter_ns()
        result = original_norm(*args, **kwargs)
        end = time.perf_counter_ns()
        boundaries.append((start, end))
        stages.samples.setdefault("pcg.norm", []).append((end - start) / 1e9)
        return result

    def solve(*args, **kwargs):
        first = len(boundaries)
        start = time.perf_counter_ns()
        result = original_solve(*args, **kwargs)
        end = time.perf_counter_ns()
        if len(boundaries) > first:
            begin_iteration, end_iteration = boundaries[first][0], boundaries[-1][1]
            for label, duration in (
                ("pcg.preparation_before_first_norm", begin_iteration - start),
                ("pcg.iterative_envelope_including_norms", end_iteration - begin_iteration),
                ("pcg.output_and_receipt_after_last_norm", end - end_iteration),
            ):
                stages.samples.setdefault(label, []).append(duration / 1e9)
        return result

    stages.stack.enter_context(patch.object(solver, "_norm", norm))
    stages.stack.enter_context(patch.object(solver, "solve_field", solve))
    # block_solver holds a directly imported alias of the same public entrypoint.
    import block_solver

    stages.stack.enter_context(patch.object(block_solver, "solve_field", solve))
    stages.wrap(solver, "_array", "pcg.input_conversion_validation")
    stages.wrap(solver._FieldOperator, "__init__", "pcg.operator_setup")
    stages.wrap(solver._FieldOperator, "rhs", "pcg.gradient_rhs")
    stages.wrap(solver._FieldOperator, "matvec", "pcg.matvec_including_boundary_residual")
    stages.wrap(solver, "_constraint_operator", "pcg.constraint_setup")
    stages.wrap(solver.PreparedField, "bind", "pcg.prepared_bind")


def mesh_preparation_stages(stages, module):
    if hasattr(module, "_prepare_tile"):
        stages.wrap(module, "_prepare_tile", "mesh.tile_preparation")
        return
    # The frozen baseline inlines support/range preparation; trace only that region.
    original = module._tile_mesh_once
    lines, first = inspect.getsourcelines(original)
    begin = next(first + i for i, line in enumerate(lines) if "cells = np.ones" in line)
    end = next(first + i for i, line in enumerate(lines) if "work = np.where" in line)
    empty = next(first + i for i, line in enumerate(lines) if "return [], np.empty" in line)

    def measured(*args, **kwargs):
        started = None

        def trace(frame, event, arg):
            nonlocal started
            if frame.f_code is not original.__code__:
                return None
            if event == "line":
                if frame.f_lineno == begin:
                    started = time.perf_counter_ns()
                elif frame.f_lineno in (end, empty) and started is not None:
                    stages.samples.setdefault("mesh.inline_support_range_traced", []).append(
                        (time.perf_counter_ns() - started) / 1e9
                    )
                    started = None
                    frame.f_trace_lines = False
            return trace

        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            return original(*args, **kwargs)
        finally:
            sys.settrace(previous)

    stages.stack.enter_context(patch.object(module, "_tile_mesh_once", measured))


def make_case(name: str, output: Path, seed: int, prepared_cache_bytes: int):
    rng = np.random.default_rng(seed)
    config = dict(device="cpu", seed=seed, dtype="float64", cache_bytes=0)
    if name.startswith("fibres-"):
        from coherence_pipeline import compile_fibres

        count = int(name.split("-")[1])
        points = np.array([[0.0, 0.5, 0.5], [float(count - 1), 0.5, 0.5]])
        config.update(
            shape=[2, 2, count],
            resampled_points=count,
            origin=[0, 0, 0],
            spacing=[1, 1, 1],
            voxel_mm=0.15,
            pitch_mm=0.15,
            pair_weight=0.1,
            fibre_id=seed,
        )

        def operation():
            pair, consensus, audit = compile_fibres(
                [dict(fibre_id=seed, points_xyz=points)],
                config["origin"],
                config["spacing"],
                config["shape"],
                voxel_mm=config["voxel_mm"],
                pitch_mm=config["pitch_mm"],
                pair_weight=config["pair_weight"],
            )
            return dict(
                samples=audit["samples"],
                pair_rows=audit["pair_rows"],
                pair_digest=pair.digest,
                consensus_digest=consensus.digest,
                selected_pairs_sha256=hashlib.sha256(audit["a_xyz"].tobytes() + audit["b_xyz"].tobytes()).hexdigest(),
            )

        return config, {"points_xyz": points}, operation, lambda stages: None

    if name in ("pcg", "pcg-prepared", "pcg-large"):
        import field_solver as solver
        from field_constraints import pair_constraints

        shape = (32, 32, 32) if name == "pcg-large" else (12, 12, 12)
        target = rng.normal(0, 0.1, (*shape, 3))
        weights = rng.uniform(0.5, 1.5, shape)
        a, b = rng.uniform(0, shape[0] - 1, (2, 32, 3))
        rows = pair_constraints(a, b, [0, 0, 0], [1, 1, 1], shape, weight=0.1)
        prepared = solver.PreparedField() if name == "pcg-prepared" else None
        config.update(
            shape=shape,
            spacing=[1, 1, 1],
            rtol=1e-8,
            maxiter=500,
            pairs=32,
            pair_weight=0.1,
            prepared=prepared is not None,
            return_tensor=False,
            gauge="zero",
        )

        def operation():
            result, info = solver.solve_field(
                target,
                weights,
                constraints=rows,
                dtype="float64",
                rtol=1e-8,
                maxiter=500,
                prepared=prepared,
                device="cpu",
            )
            if not info["converged"]:
                raise RuntimeError(info)
            return dict(
                converged=info["converged"],
                iterations=info["iterations"],
                residual=info["relative_residual"],
                field_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            )

        return config, dict(target=target, weights=weights, pairs_a=a, pairs_b=b), operation, solver_stages

    if name in ("blocks", "blocks-constraints"):
        import block_solver
        from constraint_templates import StencilConstraints
        from field_store import BlockStore

        shape = (8, 8, 16) if name == "blocks" else (3, 4, 6)
        target = np.zeros((*shape, 3), np.float32)
        target[:] = [0.2, -0.1, 0.05]
        weights = rng.uniform(0.8, 1.2, shape).astype(np.float32) if name == "blocks" else np.ones(shape, np.float32)
        observed = np.ones(shape, bool)
        receipt = hashlib.sha256(target.tobytes() + weights.tobytes()).hexdigest()
        config.update(
            shape=shape,
            input_dtype="float32",
            chunk=8,
            halo=1,
            sweeps=2,
            rtol=1e-8,
            maxiter=500,
            spacing=[1, 1, 1],
            origin=[0, 0, 0],
            read_workers=1,
            prepared_cache_bytes=prepared_cache_bytes,
            max_bytes=2**20,
            reserve_bytes=0,
            prefetch=False,
            order=[[0, 0, 0], [0, 0, 1]],
        )
        rows = None
        inputs = {}
        if name == "blocks-constraints":
            target[:] = [0.125, 0, 0]
            weights[:] = 1
            config.update(
                chunk=4,
                halo=2,
                sweeps=8,
                spacing=[2, 3, 5],
                origin=[11, -4, 19],
                maxiter=1000,
                cache_bytes=20000,
                read_workers=2,
                constraint_rows=1000,
                constraint_weight=0.01,
                stencil_cache_bytes=10**7,
            )
            a = rng.random((1000, 3)) * (np.array(shape)[::-1] - 1)
            b = rng.random((1000, 3)) * (np.array(shape)[::-1] - 1)
            targets = (a[:, 0] - b[:, 0]) * 0.25
            inputs = dict(a=a, b=b, targets=targets)
            rows = StencilConstraints.pairs(
                config["origin"] + a * config["spacing"],
                config["origin"] + b * config["spacing"],
                config["origin"],
                config["spacing"],
                shape,
                targets=targets,
                weight=0.01,
                cache_bytes=10**7,
            )
            receipt = hashlib.sha256(target.tobytes() + weights.tobytes() + rows.digest.encode()).hexdigest()
        options = {}
        if prepared_cache_bytes:
            if "prepared_cache_bytes" not in inspect.signature(block_solver.solve_blocks).parameters:
                raise ValueError("this source has no prepared_cache_bytes API")
            options["prepared_cache_bytes"] = prepared_cache_bytes

        class Loader:
            def read(self, lo, hi):
                crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                return target[crop].copy(), weights[crop].copy(), observed[crop].copy(), {"synthetic": receipt}

        loader = Loader()
        count = 0

        def operation():
            nonlocal count
            root = output / f"blocks-{count}"
            count += 1
            result = block_solver.solve_blocks(
                root,
                shape,
                config["chunk"],
                {"synthetic_input": receipt},
                loader.read,
                origin_xyz=config["origin"],
                spacing_xyz=config["spacing"],
                constraints=rows,
                order=config["order"],
                halo=config["halo"],
                sweeps=config["sweeps"],
                rtol=1e-8,
                maxiter=config["maxiter"],
                dtype="float64",
                device="cpu",
                cache_bytes=config["cache_bytes"],
                read_workers=config["read_workers"],
                max_bytes=2**20,
                **options,
            )
            if result["status"] != "SWEEP_BUDGET_REACHED":
                raise RuntimeError(result)
            checkpoint = json.loads((root / "CHECKPOINT.json").read_text())
            return {k: result[k] for k in ("status", "completed_sweeps", "cache_stats", "stored_bytes")} | {
                "prepared_cache_stats": result.get("prepared_cache_stats"),
                "checkpoint": str(root / "CHECKPOINT.json"),
                "chunk_sha256": {k: v["sha256"] for k, v in checkpoint["chunks"].items()},
                "stencil_cache_stats": None if rows is None else dict(rows.stats),
            }

        def instrument(stages):
            solver_stages(stages)
            stages.wrap(Loader, "read", "blocks.observations")
            stages.wrap(BlockStore, "read", "blocks.field_read")
            stages.wrap(block_solver, "solve_field", "blocks.solve_including_setup")
            stages.wrap(BlockStore, "commit", "blocks.commit_checkpoint")
            stages.wrap(block_solver, "atomic_json", "blocks.status_json")

        return config, dict(target=target, weights=weights, observed=observed, **inputs), operation, instrument

    if name in ("mesh", "mesh-offrange", "mesh-large"):
        import skimage.measure

        import surface_mesh
        from surface_sources import FieldSource

        shape = (9, 10, 12) if name == "mesh" else (64, 64, 64)
        levels = (
            [2.5, 5.5, 8.5]
            if name == "mesh"
            else (np.arange(-32.5, -0.5, 1.0).tolist() if name == "mesh-offrange" else [20.5, 40.5])
        )
        tile = 4 if name == "mesh" else 64
        z, y, x = np.indices(shape)
        values = x.astype(float) + y * 0.1 + z * 0.03
        path = output / "mesh-input.npy"
        np.save(path, values)
        descriptor = dict(kind="npy", path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        geometry = dict(
            frame="synthetic/world",
            voxel_um=1.0,
            sheet_step=1.0,
            origin_xyz=[0, 0, 0],
            spacing_xyz=[1, 1, 1],
            shape_zyx=list(values.shape),
        )
        config.update(
            shape=list(values.shape), geometry=geometry, levels=levels, tile=tile, level_batch=3, reserve_bytes=0
        )
        count = 0

        def operation():
            nonlocal count
            root = output / f"mesh-{count}"
            count += 1
            with FieldSource(descriptor, geometry) as source:
                result = surface_mesh.extract_mesh(source, root, config["levels"], tile=tile, level_batch=3)
            if result["status"] != "MESH_READY":
                raise RuntimeError(result)
            return dict(
                status=result["status"],
                levels=result["levels"],
                obj_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.obj")},
            )

        def instrument(stages):
            mesh_preparation_stages(stages, surface_mesh)
            stages.wrap(FieldSource, "read", "mesh.tile_read")
            stages.wrap(surface_mesh, "_tile_mesh", "mesh.tile_geometry_including_marching_cubes")
            stages.wrap(skimage.measure, "marching_cubes", "mesh.marching_cubes")
            stages.wrap(surface_mesh.MeshDB, "add", "mesh.sqlite_add")
            stages.wrap(surface_mesh.MeshDB, "checkpoint_tile", "mesh.sqlite_checkpoint")
            stages.wrap(surface_mesh.MeshDB, "finish", "mesh.sqlite_finish_export")

        return config, dict(values=values), operation, instrument

    if name == "arrival" or name.startswith("arrival-"):
        import anisotropic_reference

        shape = (6, 7, 8) if name == "arrival" else (3, 5, 7)
        budget = 8 * 1024**2 if name == "arrival" else int(name.split("-")[1])
        chunk = 128 if name == "arrival" else 40
        maxiter = 50 if name == "arrival" else 8
        directors = np.zeros((*shape, 3))
        directors[:] = [1, 0.2, 0.1]
        seeds = np.zeros(shape, bool)
        seeds[:, :, 0] = True
        config.update(
            shape=shape,
            spacing=[1, 1, 1],
            c_parallel=0.1,
            c_perp=1.0,
            maxiter=maxiter,
            tolerance=1e-5,
            chunk=chunk,
            max_cache_bytes=budget,
        )

        def operation():
            field, info = anisotropic_reference.solve_arrival(
                directors,
                [1, 1, 1],
                seeds,
                c_parallel=0.1,
                c_perp=1.0,
                maxiter=maxiter,
                tolerance=1e-5,
                chunk=chunk,
                max_cache_bytes=budget,
            )
            return dict(
                field_sha256=hashlib.sha256(field.tobytes()).hexdigest(),
                receipt={k: v for k, v in info.items() if k != "elapsed_s"},
            )

        def instrument(stages):
            stages.wrap(anisotropic_reference, "point_stencils", "arrival.stencil_build")

        return config, dict(directors=directors, seeds=seeds), operation, instrument
    raise ValueError(name)
