"""Two-level (coarse + block sweep) solves through ``solve_blocks`` and its checkpoint."""

import hashlib
import itertools
import json

import numpy as np
import pytest
import scipy.sparse as sp

from block_solver import solve_blocks
from constraint_templates import StencilConstraints
from field_amg import assemble_free_operator
from field_block_coarse import CoarseSpace
from field_schwarz import trilinear_prolongation
from field_sources import NpyObservations
from field_store import BlockStore


def problem(tmp_path, shape=(20, 16, 16), seed=0, pairs=True, reach=3):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape)
    r = np.hypot(y - shape[1] / 2 + 0.5, x - shape[2] / 2 + 0.5)
    sheets = (np.abs(r / 3 - np.round(r / 3)) < 0.3) & (rng.random(shape) > 0.2)
    arrays = dict(
        target=np.stack(
            [(x - shape[2] / 2) / np.maximum(r, 1e-6) / 3, (y - shape[1] / 2) / np.maximum(r, 1e-6) / 3, 0 * r], -1
        ).astype(np.float32)
        + 0.05 * rng.standard_normal((*shape, 3)).astype(np.float32),
        weights=np.where(sheets, 1.0, 1e-3).astype(np.float32),
        observed=sheets,
    )
    for key, value in arrays.items():
        np.save(tmp_path / f"{key}.npy", value)
    hashes = {k: hashlib.sha256((tmp_path / f"{k}.npy").read_bytes()).hexdigest() for k in arrays}
    origin, spacing = np.zeros(3), np.ones(3)
    constraints = None
    if pairs:
        start = rng.choice(np.flatnonzero(sheets), size=40, replace=False)
        a = np.stack(np.unravel_index(start, shape), 1)
        b = np.clip(a + rng.integers(-reach, reach + 1, size=a.shape), 0, np.array(shape) - 1)
        constraints = StencilConstraints.pairs(
            a[:, ::-1].astype(float), b[:, ::-1].astype(float), origin, spacing, shape, weight=5.0, cache_bytes=10**6
        )
    return dict(
        shape=shape,
        identity={"test_case": "coarse", "inputs": hashes},
        load_observations=NpyObservations(*(tmp_path / f"{k}.npy" for k in arrays), hashes),
        origin_xyz=origin,
        spacing_xyz=spacing,
        constraints=constraints,
        rtol=1e-9,
        maxiter=4000,
    )


def blocks(shape, chunk):
    grid = [-(-n // chunk) for n in shape]
    return [list(k) for k in itertools.product(*map(range, grid))]


def field(root):
    meta = json.loads((root / "CHECKPOINT.json").read_text())
    with BlockStore(root, meta["shape"], meta["chunk"], meta["identity"], resume=True) as store:
        return store.read([0] * 3, store.shape), meta


def run(root, base, chunk, **options):
    return solve_blocks(root, chunk=chunk, order=blocks(base["shape"], chunk), **base, **options)


@pytest.mark.parametrize("pairs", [False, True])
def test_galerkin_assembly_matches_the_global_product(pairs):
    rng = np.random.default_rng(1)
    shape, factor, chunk, halo = (9, 7, 10), 3, 4, 1
    weights = rng.uniform(0.1, 2.0, shape)
    mask = np.zeros(shape, bool)
    mask[0, 0, 0] = True
    rows = None
    if pairs:  # Long pairs: most of them leave the crop of the block that owns one endpoint.
        a = rng.uniform(0, 1, (30, 3)) * (np.array(shape[::-1]) - 1)
        b = rng.uniform(0, 1, (30, 3)) * (np.array(shape[::-1]) - 1)
        rows = StencilConstraints.pairs(a, b, np.zeros(3), np.ones(3), shape, weight=3.0, cache_bytes=10**6)
    space = CoarseSpace(shape, factor)
    for key in itertools.product(*(range(-(-n // chunk)) for n in shape)):
        low = np.array(key) * chunk
        high = np.minimum(low + chunk, shape)
        lo, hi = np.maximum(low - halo, 0), np.minimum(high + halo, shape)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        local = exterior = None
        if rows is not None:
            local = rows.restrict(lo, hi, lambda flat: np.zeros(np.shape(flat)))
            template = rows.prepare(lo, hi)
            if template is not None:
                row, node, coeff = template.exterior()
                exterior = (row, node, node != 0, coeff)
        space.add_block(lo, low, high, weights[crop], ~mask[crop], local, slab=2, exterior=exterior)
    p = sp.diags((~mask.ravel()).astype(float)) @ trilinear_prolongation(shape, factor)
    full = None if rows is None else rows.restrict([0, 0, 0], shape, lambda flat: np.zeros(np.shape(flat)))
    expected = (p.T @ assemble_free_operator(weights, mask, constraints=full) @ p).toarray()
    np.testing.assert_allclose(space.matrix.toarray(), expected, atol=1e-9)


def test_coarse_sweeps_reach_the_single_block_solution_faster(tmp_path):
    base = problem(tmp_path)
    run(tmp_path / "exact", base, 32, halo=0, sweeps=1)
    exact, _ = field(tmp_path / "exact")
    errors = {}
    for name, options in (("one", {}), ("two", dict(coarse_factor=4))):
        result = run(tmp_path / name, base, 8, halo=2, sweeps=6, **options)
        assert result["status"] == "SWEEP_BUDGET_REACHED"
        errors[name] = float(np.abs(field(tmp_path / name)[0] - exact).max())
    assert errors["two"] < 0.01 * errors["one"], errors
    meta = field(tmp_path / "two")[1]
    assert meta["identity"]["coarse"] == dict(kind="trilinear", factor=4)
    assert "coarse" not in field(tmp_path / "one")[1]["identity"]
    sweeps = meta["progress"]["completed_sweeps"]
    residuals = [s["relative_residual"] for s in sweeps]
    assert residuals[0] == pytest.approx(1.0) and residuals[-1] < 1e-2 and all(s["coarse_s"] >= 0 for s in sweeps)
    status = json.loads((tmp_path / "two/STATUS.json").read_text())
    assert status["coarse"]["factor"] == 4 and status["coarse"]["stage"] == "done"
    assert status["coarse"]["method"] == "direct"


def test_iterative_coarse_solve_matches_the_direct_one_and_reports_its_iterations(tmp_path):
    base = problem(tmp_path)
    fields = {}
    for name, limit in (("direct", 10**6), ("amg", 0)):
        result = run(tmp_path / name, base, 8, halo=2, sweeps=4, coarse_factor=4, coarse_direct_limit=limit)
        assert result["status"] == "SWEEP_BUDGET_REACHED"
        fields[name], meta = field(tmp_path / name)
        coarse = meta["progress"]["coarse"]
        status = json.loads((tmp_path / name / "STATUS.json").read_text())["coarse"]
        if name == "amg":
            assert status["method"] == "amg-pcg" and coarse["converged"] and 0 < coarse["iterations"] < 100
        else:
            assert status["method"] == "direct" and "iterations" not in coarse
    np.testing.assert_allclose(fields["amg"], fields["direct"], atol=1e-6)


def test_stop_inside_the_coarse_phase_resumes_to_the_same_field(tmp_path):
    base = problem(tmp_path, pairs=False)
    straight = run(tmp_path / "straight", base, 8, halo=2, sweeps=3, coarse_factor=4)
    assert straight["status"] == "SWEEP_BUDGET_REACHED"
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] in (5, 40, 75)

    root = tmp_path / "stopped"
    stops = 0
    result = run(root, base, 8, halo=2, sweeps=3, coarse_factor=4, stop=stop)
    while result["status"] == "CHECKPOINTED_STOP":
        stops += 1
        result = run(root, base, 8, halo=2, sweeps=3, coarse_factor=4, stop=stop, resume=True)
    assert stops == 3 and result["status"] == "SWEEP_BUDGET_REACHED"
    np.testing.assert_allclose(field(root)[0], field(tmp_path / "straight")[0], atol=1e-6)
    with pytest.raises(ValueError):
        run(root, base, 8, halo=2, sweeps=4, coarse_factor=8, resume=True)


def test_tampered_coarse_correction_is_rejected_on_resume(tmp_path):
    base = problem(tmp_path, pairs=False)
    root = tmp_path / "run"
    calls = {"n": 0}

    def stop():  # Stop during the first apply pass, after the correction is saved.
        calls["n"] += 1
        return calls["n"] == 16

    assert run(root, base, 8, halo=2, sweeps=2, coarse_factor=4, stop=stop)["status"] == "CHECKPOINTED_STOP"
    assert field(root)[1]["progress"]["coarse"]["stage"] == "apply"
    np.save(root / "COARSE_CORRECTION.npy", np.zeros(5))
    with pytest.raises(ValueError, match="coarse correction"):
        run(root, base, 8, halo=2, sweeps=2, coarse_factor=4, resume=True)


@pytest.mark.parametrize("factor", [1, 0, -2])
def test_invalid_coarse_factor_is_rejected(tmp_path, factor):
    with pytest.raises(ValueError):
        run(tmp_path / "bad", problem(tmp_path, pairs=False), 8, halo=2, coarse_factor=factor)


@pytest.mark.parametrize("box", [((0, 0, 0), (9, 7, 10)), ((2, 1, 3), (7, 6, 10)), ((4, 3, 9), (5, 4, 10))])
def test_box_transfers_match_the_prolongation_rows(box):
    rng = np.random.default_rng(2)
    space = CoarseSpace((9, 7, 10), 3)
    lo, hi = (np.array(v) for v in box)
    size = tuple(hi - lo)
    zyx = np.stack([g.ravel() for g in np.meshgrid(*map(np.arange, lo, hi), indexing="ij")], 1)
    values, free = rng.standard_normal(size), rng.random(size) > 0.3
    coarse = rng.standard_normal(space.size)
    out = coarse.copy()
    space.restrict_box(lo, hi, values, free, out)
    np.testing.assert_allclose(out, coarse + space.rows(zyx, free.ravel()).T @ values.ravel(), atol=1e-13)
    np.testing.assert_allclose(space.prolong_box(lo, hi, coarse).ravel(), space.rows(zyx) @ coarse, atol=1e-13)


@pytest.mark.parametrize("pairs", [False, True])
def test_threaded_residual_pass_gives_the_same_bytes(tmp_path, pairs):
    """Workers only compute; parts are added in block order, so nothing depends on threads."""
    base = problem(tmp_path, pairs=pairs)
    out = {}
    for workers in (1, 3, 8):
        root = tmp_path / f"w{workers}"
        result = run(root, base, 8, halo=2, sweeps=3, coarse_factor=4, coarse_workers=workers)
        assert result["status"] == "SWEEP_BUDGET_REACHED"
        values, meta = field(root)
        sweeps = [{k: v for k, v in s.items() if not k.endswith("_s")} for s in meta["progress"]["completed_sweeps"]]
        operator = hashlib.sha256((root / "COARSE_OPERATOR.npz").read_bytes()).hexdigest()
        correction = hashlib.sha256((root / "COARSE_CORRECTION.npy").read_bytes()).hexdigest()
        out[workers] = (values.tobytes(), json.dumps(sweeps, sort_keys=True), operator, correction)
    assert out[1] == out[3] == out[8]


def test_stop_inside_a_threaded_residual_pass_resumes_to_the_same_field(tmp_path):
    base = problem(tmp_path, pairs=False)
    straight = run(tmp_path / "straight", base, 8, halo=2, sweeps=3, coarse_factor=4, coarse_workers=1)
    assert straight["status"] == "SWEEP_BUDGET_REACHED"
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] in (5, 40, 75)

    root = tmp_path / "stopped"
    result = run(root, base, 8, halo=2, sweeps=3, coarse_factor=4, coarse_workers=4, stop=stop)
    while result["status"] == "CHECKPOINTED_STOP":
        result = run(root, base, 8, halo=2, sweeps=3, coarse_factor=4, coarse_workers=4, stop=stop, resume=True)
    assert calls["n"] > 75 and result["status"] == "SWEEP_BUDGET_REACHED"
    assert field(root)[0].tobytes() == field(tmp_path / "straight")[0].tobytes()


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_coarse_workers_are_rejected(tmp_path, workers):
    with pytest.raises(ValueError):
        run(tmp_path / "run", problem(tmp_path, pairs=False), 8, halo=2, coarse_factor=4, coarse_workers=workers)


def test_threaded_residual_pass_leaves_sparse_invariant_checks_off(tmp_path):
    """A process-wide torch flag toggled by concurrent operator builds once stayed on after the pass."""
    import torch

    assert not torch.sparse.check_sparse_tensor_invariants.is_enabled()
    for seed in range(3):
        (tmp_path / str(seed)).mkdir()
        base = problem(tmp_path / str(seed), seed=seed)
        run(tmp_path / f"run{seed}", base, 8, halo=2, sweeps=1, coarse_factor=4, coarse_workers=8)
        assert not torch.sparse.check_sparse_tensor_invariants.is_enabled()


COLOURED = dict(halo=2, coarse_factor=4, checkpoint="sweep", cache_bytes=10**8, colours=8)


def chunks(root):
    return {k: v["sha256"] for k, v in field(root)[1]["chunks"].items()}


def test_colour_batches_do_not_depend_on_the_worker_count(tmp_path):
    # Long pairs couple blocks of one colour: they must read the field as it was before the batch.
    base = problem(tmp_path, reach=12)
    for name, workers in (("one", 1), ("three", 3)):
        result = run(tmp_path / name, base, 8, sweeps=3, local_workers=workers, **COLOURED)
        assert result["status"] == "SWEEP_BUDGET_REACHED"
    assert chunks(tmp_path / "one") == chunks(tmp_path / "three")
    np.testing.assert_array_equal(field(tmp_path / "one")[0], field(tmp_path / "three")[0])
    meta = field(tmp_path / "one")[1]
    assert meta["identity"]["colours"] == 8
    run(tmp_path / "plain", base, 8, sweeps=1, halo=2, coarse_factor=4)
    plain = field(tmp_path / "plain")[1]["identity"]
    assert "colours" not in plain and plain["order_sha256"] == meta["identity"]["order_sha256"]


def test_colour_batches_reach_the_single_block_solution(tmp_path):
    base = problem(tmp_path, reach=12)
    run(tmp_path / "exact", base, 32, halo=0, sweeps=1)
    exact, _ = field(tmp_path / "exact")
    errors = {}
    for name, options in (("sequential", dict(halo=2, coarse_factor=4)), ("colours", dict(COLOURED, local_workers=2))):
        run(tmp_path / name, base, 8, sweeps=10, **options)
        errors[name] = float(np.abs(field(tmp_path / name)[0] - exact).max())
    # Same order of accuracy as the sequential sweep after the same number of sweeps.
    assert errors["sequential"] < 0.01 and errors["colours"] < 1.5 * errors["sequential"], errors


def test_stop_between_colour_batches_resumes_to_the_same_field(tmp_path):
    base = problem(tmp_path, reach=12)
    run(tmp_path / "straight", base, 8, sweeps=2, local_workers=2, **COLOURED)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] in (3, 30, 41)

    root, stops = tmp_path / "stopped", 0
    result = run(root, base, 8, sweeps=2, local_workers=2, stop=stop, **COLOURED)
    while result["status"] == "CHECKPOINTED_STOP":
        stops += 1
        result = run(root, base, 8, sweeps=2, local_workers=2, stop=stop, resume=True, **COLOURED)
    assert stops == 3 and result["status"] == "SWEEP_BUDGET_REACHED"
    assert chunks(root) == chunks(tmp_path / "straight")


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(COLOURED, checkpoint="block"), "checkpoint='sweep'"),
        (dict(COLOURED, halo=9), "halo <= chunk"),
        (dict(COLOURED, colours=4), "colours must be"),
        (dict(halo=2, local_workers=2), "colour batches"),
        (dict(halo=2, local_devices=["cpu"]), "colour batches"),
    ],
)
def test_invalid_colour_options_are_rejected(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        run(tmp_path / "bad", problem(tmp_path, pairs=False), 8, **options)
