import gc
import json
import weakref

import numpy as np
import pytest
import torch
from test_block_solver import case, state

import field_solver
from block_solver import solve_blocks
from constraint_templates import StencilConstraints
from field_solver import PreparedField, solve_field


def test_cache_reuses_stable_subset_and_preserves_exact_output(tmp_path):
    options = case(tmp_path)
    options["sweeps"] = 3
    off = solve_blocks(tmp_path / "off", **options)
    on = solve_blocks(tmp_path / "on", prepared_cache_bytes=10**7, **options)
    stats = on["prepared_cache_stats"]
    assert stats["builds"] == 2 and stats["hits"] == 4
    assert stats["peak_entries"] == 2 and stats["retained_bytes"] == 0
    partial = solve_blocks(tmp_path / "partial", prepared_cache_bytes=stats["peak_bytes"] * 3 // 4, **options)
    assert partial["prepared_cache_stats"]["hits"] == 2
    assert partial["prepared_cache_stats"]["peak_entries"] == 1
    assert off["prepared_cache_stats"]["peak_bytes"] == 0
    before, field, support = state(tmp_path / "off")
    for name, result in [("on", on), ("partial", partial)]:
        after, actual, observed = state(tmp_path / name)
        np.testing.assert_array_equal(field, actual)
        np.testing.assert_array_equal(support, observed)
        assert off["completed_sweeps"] == result["completed_sweeps"]
        assert {k: v["sha256"] for k, v in before["chunks"].items()} == {
            k: v["sha256"] for k, v in after["chunks"].items()
        }


@pytest.mark.parametrize("budget", [-1, True, 1.5, np.nan, "1"])
def test_bad_prepared_budget(tmp_path, budget):
    with pytest.raises(ValueError):
        solve_blocks(tmp_path / "run", prepared_cache_bytes=budget, **case(tmp_path))
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("constraint_kind", ["none", "consensus"])
def test_live_exterior_targets_fixed_and_support_are_exact(tmp_path, monkeypatch, dtype, constraint_kind):
    options = case(tmp_path)
    options.update(dtype=dtype, rtol=1e-5, sweeps=3)
    shape = options["shape"]
    # One group has an exterior-only row in the second block.
    options["constraints"] = (
        None
        if constraint_kind == "none"
        else StencilConstraints([[0], [5], [71]], [[1.0], [1.0], [1.0]], shape, groups=[0, 0, 0])
    )
    original = options["load_observations"]

    def load(lo, hi):
        target, weights, observed, receipts = original(lo, hi)
        observed[0] = False
        return target, weights, observed, receipts

    def fixed(lo, hi):
        values = np.full(tuple(hi - lo), np.nan)
        if not lo.any():
            values[0, 0, 0] = 2.0
        return values

    options.update(load_observations=load, fixed=fixed)
    visits = []
    original_solve = solve_field

    def traced(*args, **kwargs):
        rows = kwargs["constraints"]
        visits.append((kwargs["fixed"].copy(), None if rows is None else rows.targets.copy()))
        return original_solve(*args, **kwargs)

    monkeypatch.setattr("block_solver.solve_field", traced)
    off = solve_blocks(tmp_path / "off", **options)
    before = visits.copy()
    visits.clear()
    on = solve_blocks(tmp_path / "on", prepared_cache_bytes=10**7, **options)
    assert on["prepared_cache_stats"]["hits"] == 4
    assert off["completed_sweeps"] == on["completed_sweeps"]
    for (fa, ta), (fb, tb) in zip(before, visits):
        np.testing.assert_array_equal(fa, fb)
        np.testing.assert_array_equal(ta, tb)
    assert not np.array_equal(visits[1][0], visits[3][0], equal_nan=True)
    if constraint_kind == "consensus":
        assert not np.array_equal(visits[1][1], visits[3][1])
        assert len(visits[1][1]) == 3
    for index in (1, 2):
        np.testing.assert_array_equal(state(tmp_path / "off")[index], state(tmp_path / "on")[index])
    assert abs(state(tmp_path / "on")[1][0, 0, 0] - 2.0) < 1e-6


@pytest.mark.parametrize("change", ["target", "weights", "receipts", "nonfinite"])
def test_revisit_validates_mutated_numpy_alias_and_receipts(tmp_path, change):
    options = case(tmp_path)
    original = options["load_observations"]
    arrays = {}
    calls = 0

    def load(lo, hi):
        nonlocal calls
        key = tuple(lo)
        if key not in arrays:
            arrays[key] = original(lo, hi)
        target, weights, observed, receipts = arrays[key]
        calls += 1
        if calls == 3:
            if change == "receipts":
                receipts[next(iter(receipts))] = "0" * 64
            else:
                array = weights if change == "weights" else target
                array.flat[0] += 1 if change != "nonfinite" else np.nan
        # NumPy writes to these tensors do not increment Torch version counters.
        return torch.from_numpy(target), torch.from_numpy(weights), torch.from_numpy(observed), receipts

    options["load_observations"] = load
    with pytest.raises(ValueError, match="changed|finite"):
        solve_blocks(tmp_path / "run", prepared_cache_bytes=10**7, **options)
    assert state(tmp_path / "run")[0]["progress"]["sweep"] == 1
    stats = json.loads((tmp_path / "run/STATUS.json").read_text())["prepared_cache_stats"]
    assert stats["retained_bytes"] == stats["entries"] == 0


@pytest.mark.parametrize("budget,callable_rows", [(0, False), (1, False), (10**7, True)])
def test_disabled_oversized_and_callable_have_no_retention(tmp_path, monkeypatch, budget, callable_rows):
    options = case(tmp_path)
    if callable_rows:
        compiled = options["constraints"]
        options["constraints"] = lambda lo, hi, read: compiled.restrict(lo, hi, read)
    builds = []
    original = field_solver._FieldOperator.__init__

    def counted(self, *args):
        builds.append(1)
        original(self, *args)

    monkeypatch.setattr(field_solver._FieldOperator, "__init__", counted)
    result = solve_blocks(tmp_path / "run", prepared_cache_bytes=budget, **options)
    stats = result["prepared_cache_stats"]
    assert len(builds) == stats["builds"] == 4
    assert stats["peak_bytes"] == stats["hits"] == stats["entries"] == 0


def test_stop_resume_changes_budget_and_starts_cold(tmp_path):
    options = case(tmp_path)
    full = solve_blocks(tmp_path / "full", prepared_cache_bytes=10**7, **options)
    root = tmp_path / "split"

    def stop():
        if not (root / "CHECKPOINT.json").exists():
            return False
        p = json.loads((root / "CHECKPOINT.json").read_text())["progress"]
        return p.get("sweep") == 1 and p.get("next_block") == 2

    stopped = solve_blocks(root, prepared_cache_bytes=1, stop=stop, **options)
    assert stopped["status"] == "CHECKPOINTED_STOP"
    resumed = solve_blocks(root, prepared_cache_bytes=10**7, resume=True, **options)
    assert resumed["prepared_cache_stats"]["builds"] == 2
    assert resumed["prepared_cache_stats"]["hits"] == 0
    assert resumed["completed_sweeps"] == full["completed_sweeps"]
    a, field, support = state(tmp_path / "full")
    b, actual, observed = state(root)
    np.testing.assert_array_equal(field, actual)
    np.testing.assert_array_equal(support, observed)
    assert a["identity"] == b["identity"]
    assert "prepared_cache_bytes" not in a["identity"]
    assert {k: v["sha256"] for k, v in a["chunks"].items()} == {k: v["sha256"] for k, v in b["chunks"].items()}


@pytest.mark.parametrize("fail", [False, True])
def test_workspace_release_and_real_setup_avoidance(tmp_path, monkeypatch, fail):
    options = case(tmp_path)
    refs, operators = [], []
    original_bind = PreparedField.bind

    def bind(self, *args):
        if self.signature is None:
            refs.append(weakref.ref(self))
        result = original_bind(self, *args)
        operators.append(id(result[0]))
        return result

    monkeypatch.setattr(PreparedField, "bind", bind)
    if fail:
        original = options["load_observations"]
        calls = 0

        def load(lo, hi):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("injected failure after cache hit")
            return original(lo, hi)

        options["load_observations"] = load
        with pytest.raises(RuntimeError, match="injected"):
            solve_blocks(tmp_path / "run", prepared_cache_bytes=10**7, **options)
    else:
        solve_blocks(tmp_path / "run", prepared_cache_bytes=10**7, **options)
    gc.collect()
    assert len(refs) == 2 and all(ref() is None for ref in refs)
    assert operators[0] == operators[2]
    status = json.loads((tmp_path / "run/STATUS.json").read_text())
    assert status["prepared_cache_stats"]["retained_bytes"] == 0


def test_failed_solve_releases_new_workspace(tmp_path, monkeypatch):
    options = case(tmp_path)
    options["maxiter"] = 0
    refs = []
    original = PreparedField.bind

    def bind(self, *args):
        refs.append(weakref.ref(self))
        return original(self, *args)

    monkeypatch.setattr(PreparedField, "bind", bind)
    with pytest.raises(RuntimeError, match="converge"):
        solve_blocks(tmp_path / "run", prepared_cache_bytes=10**7, **options)
    gc.collect()
    assert len(refs) == 1 and refs[0]() is None
    status = json.loads((tmp_path / "run/STATUS.json").read_text())
    assert status["prepared_cache_stats"]["entries"] == status["prepared_cache_stats"]["retained_bytes"] == 0


def test_hits_avoid_operator_and_gradient_rhs_setup_without_template_cache(tmp_path, monkeypatch):
    options = case(tmp_path)
    options["sweeps"] = 3
    options["constraints"].max_cache_bytes = 0
    counts = dict(gradient_operator=0, constraint_operator=0, gradient_rhs=0, restriction_template=0)

    def counter(owner, attribute, name):
        original = getattr(owner, attribute)

        def counted(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, counted)

    counter(field_solver._FieldOperator, "__init__", "gradient_operator")
    counter(field_solver._FieldOperator, "rhs", "gradient_rhs")
    counter(field_solver, "_constraint_operator", "constraint_operator")
    counter(StencilConstraints, "_build", "restriction_template")
    solve_blocks(tmp_path / "off", **options)
    assert counts == dict(gradient_operator=6, constraint_operator=6, gradient_rhs=6, restriction_template=6)
    counts.update({key: 0 for key in counts})
    result = solve_blocks(tmp_path / "on", prepared_cache_bytes=10**7, **options)
    assert result["prepared_cache_stats"]["hits"] == 4
    assert counts == dict(gradient_operator=2, constraint_operator=2, gradient_rhs=2, restriction_template=6)
    assert not options["constraints"]._entries
    np.testing.assert_array_equal(state(tmp_path / "off")[1], state(tmp_path / "on")[1])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_accounting_includes_sparse_arrays_snapshots_and_full_tensor_backing(dtype):
    target = np.zeros((2, 2, 2, 3), np.float32)
    weights = np.ones((2, 2, 2), np.float32)
    rows = StencilConstraints([[0], [7]], [[1.0], [1.0]], weights.shape, groups=[0, 0]).restrict(
        [0, 0, 0], [2, 2, 2], lambda nodes: np.zeros(nodes.shape)
    )
    prepared = PreparedField()
    solve_field(target, weights, constraints=rows, prepared=prepared, dtype=dtype)
    tensors = [prepared.target, prepared.weights, prepared.gradient_rhs, prepared.operator.diagonal]
    tensors += [edge[2] for edge in prepared.operator.edges]
    extra = prepared.extra
    tensors += [extra.w, extra.target, extra.groups, extra.use, extra.sum, extra.diagonal]
    for matrix in [extra.a, extra.at]:
        tensors += (
            [matrix.crow_indices(), matrix.col_indices(), matrix.values()]
            if dtype == "float64"
            else [matrix._indices(), matrix._values()]
        )
    lower_bound = sum(t.numel() * t.element_size() for t in tensors)
    lower_bound += sum(a.nbytes for a in prepared.structure)
    assert prepared.retained_bytes >= lower_bound
    before = prepared.retained_bytes
    backing = torch.zeros(100000, dtype=torch.float64)
    prepared.gradient_rhs = backing[:8].reshape(2, 2, 2)
    assert prepared.retained_bytes >= before + backing.untyped_storage().nbytes() - 8 * 8


def test_cache_charges_entry_keys_and_container_overhead(tmp_path, monkeypatch):
    sizes = []
    original = PreparedField.bind

    def counted(self, *args):
        first = self.signature is None
        result = original(self, *args)
        if first:
            sizes.append(self.retained_bytes)
        return result

    monkeypatch.setattr(PreparedField, "bind", counted)
    result = solve_blocks(tmp_path / "run", prepared_cache_bytes=10**7, **case(tmp_path))
    assert result["prepared_cache_stats"]["peak_bytes"] >= sum(sizes) + 4096 * len(sizes)
