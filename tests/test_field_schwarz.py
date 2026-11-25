"""Two-level Schwarz with flexible CG solves the solve_field system, only faster in sweeps."""

import numpy as np
import pytest

from field_schwarz import solve_schwarz, trilinear_prolongation
from field_solver import solve_field


def _problem(shape=(24, 16, 16), seed=0, pairs=True):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape)
    r = np.hypot(y - shape[1] / 2 + 0.5, x - shape[2] / 2 + 0.5)
    sheets = (np.abs(r / 3 - np.round(r / 3)) < 0.3) & (rng.random(shape) > 0.2)
    weights = np.where(sheets, 1.0, 1e-3)
    target = np.zeros((*shape, 3))
    target[..., 0] = (x - shape[2] / 2) / np.maximum(r, 1e-6) / 3
    target[..., 1] = (y - shape[1] / 2) / np.maximum(r, 1e-6) / 3
    target += 0.05 * rng.standard_normal(target.shape)
    if not pairs:
        return target, weights, None
    # Short pairs on the sheets, some crossing core and crop boundaries.
    start = rng.choice(np.flatnonzero(sheets), size=sheets.size // 40)
    zz, yy, xx = np.unravel_index(start, shape)
    offset = rng.integers(-2, 3, size=(3, start.size))
    end = np.ravel_multi_index(tuple(np.clip(c + o, 0, n - 1) for c, o, n in zip((zz, yy, xx), offset, shape)), shape)
    return target, weights, np.stack([start, end], 1)


def _reference(target, weights, pairs, lam, fixed=None):
    field, info = solve_field(
        target, weights, pairs=pairs, lam=lam, fixed=fixed, dtype="float64", rtol=1e-12, maxiter=100000
    )
    assert info["converged"]
    return field


def test_trilinear_prolongation_is_a_partition_of_unity_and_exact_for_linear_fields():
    shape = (9, 7, 5)
    p = trilinear_prolongation(shape, 3)
    np.testing.assert_allclose(np.asarray(p.sum(axis=1)).ravel(), 1.0)
    coarse_axes = [np.unique(np.append(np.arange(0, n, 3), n - 1)) for n in shape]
    cz, cy, cx = np.meshgrid(*coarse_axes, indexing="ij")
    z, y, x = np.indices(shape)
    np.testing.assert_allclose(p @ (2 * cz - cy + 0.5 * cx).ravel(), (2 * z - y + 0.5 * x).ravel(), atol=1e-12)


def test_single_tile_is_the_global_solve():
    target, weights, pairs = _problem((10, 8, 9))
    expected = _reference(target, weights, pairs, 10.0)
    field, info = solve_schwarz(
        target,
        weights,
        pairs=pairs,
        lam=10.0,
        chunk=16,
        halo=0,
        coarse_factor=None,
        rtol=1e-9,
        local_preconditioner="jacobi",
    )
    assert info["converged"] and info["tiles"] == 1 and info["sweeps"] == 1
    np.testing.assert_allclose(field, expected, atol=1e-7)


@pytest.mark.parametrize("kind", ["trilinear", "aggregation"])
def test_two_level_schwarz_reaches_the_global_solution(kind):
    target, weights, pairs = _problem()
    expected = _reference(target, weights, pairs, 10.0)
    field, info = solve_schwarz(
        target,
        weights,
        pairs=pairs,
        lam=10.0,
        chunk=8,
        halo=2,
        coarse_factor=4,
        coarse_kind=kind,
        rtol=1e-9,
        maxiter=40,
        local_preconditioner="jacobi",
    )
    assert info["converged"], info["residual_history"]
    assert info["tiles"] == 12 and info["coarse"]["kind"] == kind
    assert info["local"]["nonconverged"] == 0
    np.testing.assert_allclose(field, expected, atol=1e-5)


def test_coarse_level_cuts_sweeps():
    target, weights, _ = _problem(pairs=False)
    options = dict(chunk=8, halo=2, rtol=1e-8, maxiter=60, local_preconditioner="jacobi")
    _, one = solve_schwarz(target, weights, coarse_factor=None, **options)
    _, two = solve_schwarz(target, weights, coarse_factor=4, **options)
    assert two["converged"]
    assert two["sweeps"] < one["sweeps"] or not one["converged"]


def test_fixed_nodes_are_exact_and_gauge_is_reported():
    target, weights, pairs = _problem()
    fixed = np.full(weights.shape, np.nan)
    fixed[0] = 1.25
    fixed[-1, :, :4] = -3.0
    expected = _reference(target, weights, pairs, 10.0, fixed)
    field, info = solve_schwarz(
        target,
        weights,
        pairs=pairs,
        lam=10.0,
        fixed=fixed,
        chunk=8,
        halo=2,
        coarse_factor=4,
        rtol=1e-9,
        maxiter=40,
        local_preconditioner="jacobi",
    )
    assert info["converged"] and info["gauge"] == dict(kind="fixed", count=int(np.isfinite(fixed).sum()))
    assert np.array_equal(field[np.isfinite(fixed)], fixed[np.isfinite(fixed)])
    np.testing.assert_allclose(field, expected, atol=1e-5)


def test_restricted_writes_stay_available():
    target, weights, _ = _problem(pairs=False)
    expected = _reference(target, weights, None, 0.0)
    field, info = solve_schwarz(
        target,
        weights,
        chunk=8,
        halo=2,
        coarse_factor=4,
        rtol=1e-9,
        maxiter=60,
        restricted=True,
        local_preconditioner="jacobi",
    )
    assert info["converged"]
    np.testing.assert_allclose(field, expected, atol=1e-5)


@pytest.mark.parametrize(
    "options",
    [
        dict(chunk=0, halo=1),
        dict(chunk=4, halo=-1),
        dict(chunk=4, halo=0),
        dict(chunk=4, halo=1, coarse_factor=1),
        dict(chunk=4, halo=1, maxiter=-1),
        dict(chunk=4, halo=1, rtol=0.0),
        dict(chunk=4, halo=1, coarse_kind="spectral"),
    ],
)
def test_invalid_arguments_are_rejected(options):
    target, weights, _ = _problem((8, 8, 8), pairs=False)
    with pytest.raises(ValueError):
        solve_schwarz(target, weights, local_preconditioner="jacobi", **options)


def test_amg_coarse_solve_and_progress_callback():
    target, weights, _ = _problem(pairs=False)
    expected = _reference(target, weights, None, 0.0)
    seen = []
    field, info = solve_schwarz(
        target,
        weights,
        chunk=8,
        halo=2,
        coarse_factor=2,
        direct_coarse_limit=0,
        rtol=1e-8,
        maxiter=40,
        local_preconditioner="jacobi",
        callback=seen.append,
    )
    assert info["converged"] and info["coarse"]["method"] == "amg-pcg"
    assert [event["iteration"] for event in seen] == list(range(1, info["iterations"] + 1))
    np.testing.assert_allclose(field, expected, atol=1e-4)
