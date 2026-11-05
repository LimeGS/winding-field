"""Uninstrumented operations; validation and hashing belong outside their timers."""

import hashlib
import itertools
import json
import time

import numpy as np


def solver_fixture(case, size=32):
    from field_constraints import consensus_constraints, pair_constraints

    if case not in {"pairs", "consensus", "jump", "faces", "tiny"}:
        raise ValueError("unknown solver fixture")
    if case == "tiny":
        size = 3
    consensus = case == "consensus"
    rows = 20000 if consensus else 32
    rng = np.random.default_rng(20261105 if consensus else 20261104)
    shape = (size,) * 3
    target = rng.normal(0, 0.1, (*shape, 3))
    weights = rng.uniform(0.5, 1.5, shape)
    if consensus:
        points = rng.uniform(0, size - 1, (rows, 3))
        groups = np.arange(rows) // 20
        length_weights = rng.uniform(0.5, 1.5, rows)
        start = time.perf_counter()
        constraints = consensus_constraints(points, groups, length_weights, [0] * 3, [1] * 3, shape, weight=0.1)
    else:
        a, b = rng.uniform(0, size - 1, (2, rows, 3))
        start = time.perf_counter()
        constraints = pair_constraints(a, b, [0] * 3, [1] * 3, shape, weight=0.1)
    construction_s = time.perf_counter() - start
    if case == "jump":
        weights = np.ones(shape)
        weights[:, :, size // 2 :] = 100
    options = dict(
        target_xyz=target,
        weights=weights,
        spacing_xyz=[1] * 3,
        constraints=constraints,
        dtype="float64",
        rtol=1e-8,
        maxiter=1500,
        device="cpu",
    )
    if case == "faces":
        fixed = np.full(shape, np.nan)
        fixed[[0, -1], :, :] = 0
        fixed[:, [0, -1], :] = 0
        fixed[:, :, [0, -1]] = 0
        options["fixed"] = fixed
    return options, construction_s


def make_case(name, output, policy, size=32):
    import field_solver
    from constraint_templates import StencilConstraints
    from field_store import BlockStore

    if name in {"pairs", "consensus", "jump", "faces", "tiny"}:
        options, construction_s = solver_fixture(name, size)
        if policy == "spectral":
            options["preconditioner"] = "spectral"

        def operation():
            return field_solver.solve_field(**options)

        def validate(result):
            field, info = result
            fixed = options.get("fixed")
            error = abs(field.flat[0]) if fixed is None else np.max(abs(field[np.isfinite(fixed)]))
            assert info["converged"] and info["relative_residual"] <= options["rtol"]
            assert error == 0
            return field, dict(info, anchor_error=float(error))

        def retained(result):
            prepared = field_solver.PreparedField()
            field_solver.solve_field(**options, prepared=prepared)
            return dict(retained_bytes=0, separate_prepared_workspace_bytes=prepared.retained_bytes)

        return (
            operation,
            validate,
            dict(
                case=name,
                shape=list(options["weights"].shape),
                construction_s=construction_s,
                seed=20261105 if name == "consensus" else 20261104,
                target_sha256=hashlib.sha256(options["target_xyz"].tobytes()).hexdigest(),
                weights_sha256=hashlib.sha256(options["weights"].tobytes()).hexdigest(),
                constraints_digest=options["constraints"].digest,
                rtol=1e-8,
                maxiter=1500,
                dtype="float64",
                policy=policy,
            ),
            retained,
        )

    counter = itertools.count()
    if name in {"blocks", "blocks-constraints"}:
        from block_solver import solve_blocks

        constrained = name == "blocks-constraints"
        shape = (3, 4, 6) if constrained else (8, 8, 16)
        target = np.broadcast_to([0.125, 0, 0] if constrained else [0.2, -0.1, 0.05], (*shape, 3)).astype(np.float32)
        rng = np.random.default_rng(20261104)
        weights = np.ones(shape, np.float32) if constrained else rng.uniform(0.8, 1.2, shape).astype(np.float32)
        observed = np.ones(shape, bool)
        spacing, origin = ([2, 3, 5], [11, -4, 19]) if constrained else ([1] * 3, [0] * 3)
        rows = None
        construction_s = 0.0
        if constrained:
            a = rng.random((1000, 3)) * (np.array(shape)[::-1] - 1)
            b = rng.random((1000, 3)) * (np.array(shape)[::-1] - 1)
            options = {} if policy == "default" else dict(spatial_index_bytes=10**7, template_cache_policy="stable")
            start = time.perf_counter()
            rows = StencilConstraints.pairs(
                origin + a * spacing,
                origin + b * spacing,
                origin,
                spacing,
                shape,
                targets=(a[:, 0] - b[:, 0]) * 0.25,
                weight=0.01,
                cache_bytes=10**7,
                **options,
            )
            construction_s = time.perf_counter() - start
        receipt = hashlib.sha256(target.tobytes() + weights.tobytes()).hexdigest()

        def load(lo, hi):
            crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            return target[crop].copy(), weights[crop].copy(), observed[crop].copy(), {"fixture": receipt}

        def operation():
            root = output / f"blocks-{next(counter)}"
            options = {} if policy == "default" else dict(prepared_cache_bytes=10**7, status_policy="checkpoints")
            result = solve_blocks(
                root,
                shape,
                4 if constrained else 8,
                {"fixture": receipt},
                load,
                origin_xyz=origin,
                spacing_xyz=spacing,
                order=[[0, 0, 0], [0, 0, 1]],
                constraints=rows,
                halo=2 if constrained else 1,
                sweeps=8 if constrained else 2,
                rtol=1e-8,
                maxiter=1000 if constrained else 500,
                dtype="float64",
                cache_bytes=20000 if constrained else 0,
                max_bytes=2**20,
                read_workers=2 if constrained else 1,
                **options,
            )
            return root, result

        def validate(result):
            root, info = result
            assert info["status"] == "SWEEP_BUDGET_REACHED"
            manifest = json.loads((root / "CHECKPOINT.json").read_text())
            with BlockStore(root, shape, manifest["chunk"], manifest["identity"], resume=True) as store:
                field = store.read([0] * 3, shape)
            return field, dict(
                info,
                solver=manifest["progress"]["solver"],
                anchor_error=manifest["progress"]["decoded_fixed_value_error_max"],
                chunk_sha256={k: v["sha256"] for k, v in manifest["chunks"].items()},
                template_stats=None if rows is None else dict(rows.stats),
            )

        return (
            operation,
            validate,
            dict(case=name, shape=shape, policy=policy, construction_s=construction_s),
            lambda result: dict(
                retained_bytes=0 if rows is None else rows.stats["bytes"] + getattr(rows, "spatial_index_nbytes", 0)
            ),
        )

    if name == "gather":
        shape = (16, 16, 48)
        rng = np.random.default_rng(20261104)
        nodes = rng.integers(0, np.prod(shape), 10000)
        store = BlockStore(output / "gather-store", shape, 16, {"fixture": "gather-v3"}, cache_bytes=2**20)
        for x in range(3):
            store.commit([0, 0, x], np.full((16, 16, 16), x + 1.0), {"x": x})
        start = time.perf_counter()
        plan = None if policy == "default" else store.prepare_nodes(nodes)
        construction_s = time.perf_counter() - start

        def operation():
            return store.nodes(nodes) if plan is None else store.read_nodes_plan(plan)

        def validate(result):
            np.testing.assert_array_equal(result, nodes % 48 // 16 + 1)
            return result, dict(cache_stats=dict(store.stats))

        return (
            operation,
            validate,
            dict(case=name, construction_s=construction_s),
            lambda result: dict(retained_bytes=store.stats["peak_cache_bytes"] + (0 if plan is None else plan.nbytes)),
        )

    if name == "store":
        shape = (32, 32, 32)
        values = np.random.default_rng(20261104).normal(size=shape) + 10000
        generations = [values + generation for generation in range(3)]
        expected = [
            (value - value.mean()).astype(np.float32).astype(np.float64) + value.mean() for value in generations
        ]

        def operation():
            root = output / f"store-{next(counter)}"
            reads = []
            with BlockStore(root, shape, 32, {"fixture": "store-v3"}, cache_bytes=2**20) as store:
                for generation, value in enumerate(generations):
                    store.commit([0, 0, 0], value, {"generation": generation})
                    reads.append(store.read([0] * 3, shape))
                return reads, dict(store.stats)

        def validate(result):
            reads, stats = result
            for actual, wanted in zip(reads, expected, strict=True):
                np.testing.assert_array_equal(actual, wanted)
            return np.stack(reads), stats

        return operation, validate, dict(case=name, shape=shape), lambda result: dict(retained_bytes=0)

    if name in {"templates", "cache-abc"}:
        shape = (32, 32, 96)
        rng = np.random.default_rng(20261104)
        if name == "cache-abc":
            nodes = np.array([[0], [32], [64]])
            coefficients = np.ones_like(nodes, dtype=float)
            groups = None
        else:
            nodes = rng.integers(0, np.prod(shape), (20000, 4))
            coefficients = rng.normal(size=nodes.shape)
            groups = np.arange(len(nodes)) // 20
        extra = {} if policy == "default" else dict(spatial_index_bytes=10**8, template_cache_policy="stable")
        probe = StencilConstraints(nodes, coefficients, shape, groups=groups, cache_bytes=10**8, **extra)
        bounds = [([0, 0, x], [32, 32, x + 32]) for x in (0, 32, 64)]
        for lo, hi in bounds[:2]:
            probe.restrict(lo, hi, lambda n: np.zeros(n.shape))
        budget = probe.stats["bytes"] if name == "cache-abc" else 0

        def operation():
            rows = StencilConstraints(nodes, coefficients, shape, groups=groups, cache_bytes=budget, **extra)
            results = []
            for sweep in range(3):
                for lo, hi in bounds:
                    results.append(rows.restrict(lo, hi, lambda n: np.sin(n + sweep)))
            return rows, results

        def validate(result):
            rows, results = result
            if name == "cache-abc":
                assert rows.stats["hits"] == (0 if policy == "default" else 4)
                assert rows.stats["misses"] == (9 if policy == "default" else 5)
                assert rows.stats["evictions"] == (7 if policy == "default" else 0)
            arrays = []
            digests = []
            for constraints in results:
                if constraints is not None:
                    arrays.extend([constraints.targets, constraints.diagonal])
                    digests.append(constraints.digest)
            return np.concatenate(arrays), dict(
                cache_stats=dict(rows.stats),
                digests=digests,
                spatial_index_bytes=getattr(rows, "spatial_index_nbytes", 0),
            )

        return (
            operation,
            validate,
            dict(case=name, shape=shape, policy=policy, cache_bytes=budget),
            lambda result: dict(
                retained_bytes=result[0].stats["bytes"] + getattr(result[0], "spatial_index_nbytes", 0)
            ),
        )
    raise ValueError("unknown benchmark case")
