import itertools

import numpy as np
import pytest
import torch

import field_solver as solver
from block_solver import solve_blocks
from constraint_templates import StencilConstraints


def test_norm_reuse_keeps_independent_residual(monkeypatch):
    rng = np.random.default_rng(73)
    target = rng.normal(size=(4, 5, 6, 3))
    weights = rng.uniform(0.5, 2, (4, 5, 6))
    calls = []
    original = solver._norm

    def counted(value):
        calls.append(value.clone())
        return original(value)

    monkeypatch.setattr(solver, "_norm", counted)
    field, info = solver.solve_field(target, weights, dtype="float64", maxiter=5, rtol=1e-12)
    assert info["iterations"] == 5
    assert len(calls) == 8
    op = solver._FieldOperator(torch.tensor(weights), None, 0)
    rhs = op.rhs(torch.tensor(target), [1, 1, 1])
    rhs[0, 0, 0] = 0
    residual = rhs - op.matvec(torch.tensor(field))
    residual[0, 0, 0] = 0
    torch.testing.assert_close(calls[-1], residual, rtol=0, atol=0)


def test_restart_refreshes_reused_norm(monkeypatch):
    rng = np.random.default_rng(73)
    target = rng.normal(size=(4, 5, 6, 3))
    original = solver._norm
    calls = 0

    def premature(value):
        nonlocal calls
        calls += 1
        return 0.0 if calls == 3 else original(value)

    monkeypatch.setattr(solver, "_norm", premature)
    _, info = solver.solve_field(target, np.ones((4, 5, 6)), dtype="float64", maxiter=5, rtol=1e-12)
    assert info["iterations"] == 5 and not info["converged"]
    assert calls == 9


def test_callable_prepared_cache_refreshes_targets_and_is_bounded(tmp_path):
    shape = (3, 3, 6)
    rows = StencilConstraints.pairs([[0, 1, 1]], [[5, 1, 1]], [0] * 3, [1] * 3, shape)

    class Provider:
        def prepared_key(self, lo, hi):
            return rows.digest

        def __call__(self, lo, hi, read):
            return rows.restrict(lo, hi, read)

    def load(lo, hi):
        s = tuple(hi - lo)
        return (
            np.broadcast_to(np.array([0.2, 0.1, 0], "f4"), (*s, 3)),
            np.ones(s, "f4"),
            np.ones(s, bool),
            {"test": "a" * 64},
        )

    results = []
    for budget in [0, 1, 1 << 20]:
        result = solve_blocks(
            tmp_path / str(budget),
            shape,
            3,
            {"test": 1},
            load,
            origin_xyz=[0] * 3,
            spacing_xyz=[1] * 3,
            order=list(itertools.product(range(1), range(1), range(2))),
            sweeps=3,
            constraints=Provider(),
            prepared_cache_bytes=budget,
            rtol=1e-10,
        )
        stats = result["prepared_cache_stats"]
        assert stats["peak_bytes"] <= budget
        assert stats["retained_bytes"] == 0
        assert stats["hits"] == (4 if budget > 1 else 0)
        results.append(result["completed_sweeps"])
    assert results[0] == results[1] == results[2]


def test_prepared_restriction_keeps_exterior_live():
    rows = StencilConstraints.pairs([[0, 0, 0]], [[3, 0, 0]], [0] * 3, [1] * 3, (2, 2, 4))
    template = rows.prepare([0] * 3, [2, 2, 2])
    assert template.nbytes > 0
    first = template.restrict(lambda nodes: np.full(nodes.shape, 2.0))
    second = template.restrict(lambda nodes: np.full(nodes.shape, 7.0))
    np.testing.assert_array_equal(first.targets, [2.0])
    np.testing.assert_array_equal(second.targets, [7.0])
    np.testing.assert_array_equal(first.matrix.toarray(), second.matrix.toarray())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_operator_reuses_buffers_without_changing_inputs(dtype):
    rng = np.random.default_rng(19)
    field = torch.tensor(rng.normal(size=(3, 4, 5)), dtype=dtype)
    before = field.clone()
    op = solver._FieldOperator(torch.ones_like(field), torch.tensor([[0, 59], [0, 59]]), 2.0)
    expected = op.matvec(field)
    out, scratch = torch.empty_like(field), torch.empty_like(field)
    for _ in range(2):
        actual = op.matvec(field, out=out, scratch=scratch)
        assert actual.data_ptr() == out.data_ptr()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(field, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_prepared_buffers_do_not_alias_returned_fields(dtype):
    rng = np.random.default_rng(51)
    target = rng.normal(size=(3, 4, 5, 3))
    weights = np.ones((3, 4, 5))
    prepared = solver.PreparedField()
    first, info = solver.solve_field(target, weights, dtype=dtype, prepared=prepared, rtol=1e-5)
    assert info["converged"]
    saved = first.copy()
    pointers = [v.data_ptr() for v in prepared.work]
    retained = prepared.retained_bytes
    fixed = np.full(weights.shape, np.nan)
    fixed[0, 0, 0] = 3.0
    second, info = solver.solve_field(target, weights, dtype=dtype, prepared=prepared, fixed=fixed, rtol=1e-5)
    plain, other = solver.solve_field(target, weights, dtype=dtype, fixed=fixed, rtol=1e-5)
    assert info["converged"] and other["converged"]
    np.testing.assert_array_equal(first, saved)
    np.testing.assert_array_equal(second, plain)
    assert pointers == [v.data_ptr() for v in prepared.work]
    # CPython shared-key dictionaries can shrink when the plain operator is created.
    assert prepared.retained_bytes <= retained


@pytest.mark.parametrize("versioned", [False, True])
def test_callable_key_invalidates_changed_observations(tmp_path, versioned):
    visits = 0

    def load(lo, hi):
        nonlocal visits
        visits += 1
        shape = tuple(hi - lo)
        target = np.zeros((*shape, 3), "f4")
        target[..., 0] = visits
        return target, np.ones(shape, "f4"), np.ones(shape, bool), {"test": "a" * 64}

    class Provider:
        def __call__(self, lo, hi, read):
            return None

        def prepared_key(self, lo, hi):
            return str(visits) if versioned else "unchanged"

    def run():
        return solve_blocks(
            tmp_path / "field",
            (3, 3, 3),
            3,
            {"test": 1},
            load,
            origin_xyz=[0] * 3,
            spacing_xyz=[1] * 3,
            order=[[0] * 3],
            sweeps=2,
            constraints=Provider(),
            prepared_cache_bytes=1 << 20,
        )

    if versioned:
        result = run()
        assert result["prepared_cache_stats"]["hits"] == 0
        assert result["prepared_cache_stats"]["peak_entries"] == 2
    else:
        with pytest.raises(ValueError, match="prepared field inputs changed"):
            run()
