"""Small reproducibility checks; no timing thresholds."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def cases():
    path = Path(__file__).resolve().parents[1] / "benchmarks/performance/v3_cases.py"
    spec = importlib.util.spec_from_file_location("v3_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_rng_order_and_jump_after_constraints():
    api = cases()
    pairs, _ = api.solver_fixture("pairs", 4)
    jump, _ = api.solver_fixture("jump", 4)
    rng = np.random.default_rng(20261104)
    np.testing.assert_array_equal(pairs["target_xyz"], rng.normal(0, 0.1, (4, 4, 4, 3)))
    np.testing.assert_array_equal(pairs["weights"], rng.uniform(0.5, 1.5, (4, 4, 4)))
    assert pairs["constraints"].digest == jump["constraints"].digest
    assert np.all(jump["weights"][:, :, :2] == 1)
    assert np.all(jump["weights"][:, :, 2:] == 100)


@pytest.mark.parametrize("policy,hits,misses", [("default", 0, 9), ("tuned", 4, 5)])
def test_abc_fixture_exact_cache_pattern(tmp_path, policy, hits, misses):
    operation, validate, _, _ = cases().make_case("cache-abc", tmp_path, policy)
    field, info = validate(operation())
    assert np.isfinite(field).all()
    assert info["cache_stats"]["hits"] == hits
    assert info["cache_stats"]["misses"] == misses


def test_solver_fixture_true_residual_and_exact_anchor(tmp_path):
    operation, validate, config, retained = cases().make_case("tiny", tmp_path, "spectral")
    field, info = validate(operation())
    assert field.shape == (3, 3, 3)
    assert config["rtol"] == 1e-8 and config["maxiter"] == 1500
    assert info["anchor_error"] == 0
    memory = retained(None)
    assert memory["retained_bytes"] == 0
    assert memory["separate_prepared_workspace_bytes"] > 0


def test_consensus_fixture_draws_match_independent_construction():
    from field_constraints import consensus_constraints, pair_constraints

    rng = np.random.default_rng(20261105)
    target = rng.normal(0, 0.1, (4, 4, 4, 3))
    weights = rng.uniform(0.5, 1.5, (4, 4, 4))
    points = rng.uniform(0, 3, (20000, 3))
    length = rng.uniform(0.5, 1.5, 20000)
    expected = consensus_constraints(points, np.arange(20000) // 20, length, [0] * 3, [1] * 3, (4,) * 3, weight=0.1)
    actual, _ = cases().solver_fixture("consensus", 4)
    np.testing.assert_array_equal(actual["target_xyz"], target)
    np.testing.assert_array_equal(actual["weights"], weights)
    assert actual["constraints"].digest == expected.digest
    rng = np.random.default_rng(20261104)
    rng.normal(0, 0.1, (4, 4, 4, 3))
    rng.uniform(0.5, 1.5, (4, 4, 4))
    a, b = rng.uniform(0, 3, (2, 32, 3))
    expected = pair_constraints(a, b, [0] * 3, [1] * 3, (4,) * 3, weight=0.1)
    assert cases().solver_fixture("pairs", 4)[0]["constraints"].digest == expected.digest


def test_frozen_variants_and_balanced_order(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "benchmarks/performance"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("v3_benchmark", directory / "v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for label in ("old", "new"):
        (tmp_path / label / "src").mkdir(parents=True)
        (tmp_path / label / "src/field_solver.py").write_text(label + " solver")
        (tmp_path / label / "src/field_constraints.py").write_text(label + " constraints")
    roots = module.variants(tmp_path, tmp_path / "old", tmp_path / "new")
    assert (roots["indices"] / "src/field_solver.py").read_text() == "old solver"
    assert (roots["indices"] / "src/field_constraints.py").read_text() == "new constraints"
    assert (roots["spectral"] / "src/field_solver.py").read_text() == "new solver"
    assert (roots["spectral"] / "src/field_constraints.py").read_text() == "old constraints"
    for labels in [["a", "b"], ["a", "b", "c", "d"]]:
        orders = [module.balanced_order(labels, i) for i in range(8)]
        for position in range(len(labels)):
            assert all(sum(order[position] == label for order in orders) == 8 // len(labels) for label in labels)
