"""Independent equations and legacy arithmetic oracles for solver performance v3."""

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix, csr_matrix

import field_solver as solver
from field_constraints import QuadraticConstraints, _QuadraticOperator
from tests.test_field_solver import dense_system


def legacy_center(self, r):
    v = self.w * r
    if len(self.sum):
        total = torch.zeros_like(self.sum)
        total.index_add_(0, self.groups[self.use], v[self.use])
        mean = total / self.sum.clamp_min(torch.finfo(self.w.dtype).tiny)
        v = v.clone()
        v[self.use] -= self.w[self.use] * mean[self.groups[self.use]]
    return v


def rows_for(shape, mode="mixed"):
    n = int(np.prod(shape))
    if mode == "empty":
        return QuadraticConstraints(csr_matrix((0, n)))
    # Duplicate coordinates, a wholly exterior/empty row, and a zero-mass group.
    matrix = coo_matrix(
        (
            [0.3, 0.7, -1, 1, -1, 1, -1, 1, -1, 0.5],
            ([0, 0, 0, 1, 1, 2, 2, 3, 3, 5], [0, 0, n - 1, 2, 3, 1, 4, 0, 4, 3]),
        ),
        shape=(6, n),
    )
    groups = [-1] * 6 if mode == "ungrouped" else [-1, 0, 0, 1, 0, 1]
    return QuadraticConstraints(matrix, [2, 0.7, 1.3, 0, 3, 0], [0.1, -0.7, 2, 0, 3.1, 4], groups)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["mixed", "ungrouped", "empty"])
@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_cpu_rows_exact_legacy_matvec_rhs_and_solve(monkeypatch, dtype, mode, layout):
    rng = np.random.default_rng(81)
    shape = (2, 3, 4)
    c = rows_for(shape, mode)
    op = c.operator("cpu", np.dtype(dtype), layout=layout)
    x = torch.tensor(rng.normal(size=shape), dtype=getattr(torch, dtype))
    before = x.clone()
    actual_mv, actual_rhs = op.matvec(x), op.rhs()
    with monkeypatch.context() as m:
        m.setattr(_QuadraticOperator, "center_weight", legacy_center)
        assert torch.equal(actual_mv, op.matvec(x))
        assert torch.equal(actual_rhs, op.rhs())
    assert torch.equal(x, before)
    assert torch.equal(op.center_weight(op.target), legacy_center(op, op.target))
    # The retained backend route is the same legacy oracle, exercised on CPU.
    op.active_rows = None
    assert torch.equal(actual_rhs, op.rhs())
    target = rng.normal(size=(*shape, 3))
    weights = rng.uniform(0.5, 2, size=shape)
    workspace = solver.PreparedField()
    for offset in [0, 0.4]:
        c.targets += offset
        actual, info = solver.solve_field(target, weights, constraints=c, dtype=dtype, prepared=workspace)
        with monkeypatch.context() as m:
            m.setattr(_QuadraticOperator, "center_weight", legacy_center)
            expected, old_info = solver.solve_field(target, weights, constraints=c, dtype=dtype)
        np.testing.assert_array_equal(actual, expected)
        assert {k: v for k, v in info.items() if k != "elapsed_s"} == {
            k: v for k, v in old_info.items() if k != "elapsed_s"
        }


def fixed_for(shape, kind):
    fixed = np.full(shape, np.nan)
    if kind == "auto":
        return None
    if kind == "point":
        fixed[tuple(n // 2 for n in shape)] = 1.23456789
    elif kind == "faces":
        mask = np.ones(shape, dtype=bool)
        mask[1:-1, 1:-1, 1:-1] = False
        fixed[mask] = np.arange(np.prod(shape)).reshape(shape)[mask] / 10
    else:
        fixed[:] = np.arange(np.prod(shape)).reshape(shape) / 10
    return fixed


def effective_mask(shape, fixed):
    if fixed is not None:
        return np.isfinite(fixed)
    mask = np.zeros(shape, dtype=bool)
    mask.flat[0] = True
    return mask


@pytest.mark.parametrize(
    "shape,kind",
    [
        ((3, 4, 5), "point"),
        ((1, 3, 5), "point"),
        ((1, 1, 6), "auto"),
        ((4, 5, 6), "faces"),
        ((3, 3, 3), "faces"),
        ((1, 4, 5), "faces"),
        ((2, 3, 4), "all"),
        ((1, 1, 1), "auto"),
    ],
)
def test_spectral_inverse_action_symmetry_and_positive_energy(shape, kind):
    rng = np.random.default_rng(67)
    weights = np.full(shape, 2.3)
    mask = effective_mask(shape, fixed_for(shape, kind))
    pre = solver._SpectralPreconditioner(torch.tensor(weights), torch.tensor(mask))
    matrix, _ = (
        dense_system(np.zeros((*shape, 3)), weights, (1, 1, 1)) if np.prod(shape) > 1 else (np.zeros((1, 1)), None)
    )
    a, b = [rng.normal(size=shape) for _ in range(2)]
    a[mask], b[mask] = 0, 0
    pa, pb = [pre.apply(torch.tensor(x)).numpy() for x in [a, b]]
    np.testing.assert_array_equal(pa[mask], 0)
    np.testing.assert_allclose((matrix @ pa.ravel()).reshape(shape)[~mask], a[~mask], atol=2e-13)
    np.testing.assert_allclose(np.sum(a * pb), np.sum(b * pa), atol=2e-13)
    assert np.sum(a * pa) > 0 if (~mask).any() else np.sum(a * pa) == 0
    if pre.kind == "point":
        assert pre.inverse[0, 0, 0] == 0


@pytest.mark.parametrize("kind", ["auto", "point", "faces", "all"])
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_spectral_solve_matches_independent_reduced_system(kind, heterogeneous):
    rng = np.random.default_rng(53)
    shape = (3, 4, 5)
    target = rng.normal(size=(*shape, 3))
    weights = rng.uniform(0.5, 2, shape) if heterogeneous else np.full(shape, 1.7)
    fixed = fixed_for(shape, kind)
    c = rows_for(shape) if heterogeneous else None
    pairs = [[0, 17], [0, 17], [3, 3]] if heterogeneous else []
    reference = rng.normal(size=shape)
    rw = 0.2 if heterogeneous else 0
    matrix, rhs = dense_system(target, weights, (2, 3, 4), pairs, 0.3, reference, rw)
    if c is not None:
        center = np.diag(c.weights)
        for group in np.unique(c.groups[c.groups >= 0]):
            w = c.weights * (c.groups == group)
            if w.sum():
                center -= np.outer(w, w) / w.sum()
        a = c.matrix.toarray()
        matrix += a.T @ center @ a
        rhs += a.T @ center @ c.targets
    # A reference without fixed nodes deliberately does not get the auto gauge.
    if kind == "auto" and heterogeneous:
        with pytest.raises(ValueError, match="fixed"):
            solver.solve_field(
                target, weights, reference=reference, reference_weight=rw, dtype="float64", preconditioner="spectral"
            )
        return
    mask = effective_mask(shape, fixed).ravel()
    expected = np.zeros(weights.size) if fixed is None else np.nan_to_num(fixed).ravel()
    reduced_rhs = rhs[~mask] - matrix[np.ix_(~mask, mask)] @ expected[mask]
    expected[~mask] = np.linalg.solve(matrix[np.ix_(~mask, ~mask)], reduced_rhs)
    for initial in [None, rng.normal(size=shape), expected.reshape(shape)]:
        actual, info = solver.solve_field(
            target,
            weights,
            (2, 3, 4),
            pairs=pairs,
            lam=0.3,
            fixed=fixed,
            constraints=c,
            reference=reference,
            reference_weight=rw,
            initial=initial,
            dtype="float64",
            preconditioner="spectral",
            rtol=1e-11,
            return_tensor=True,
        )
        assert info["converged"] and info["preconditioner"] == "spectral"
        np.testing.assert_allclose(actual.numpy().ravel(), expected, atol=2e-10)
        np.testing.assert_array_equal(actual.numpy().ravel()[mask], expected[mask])
        assert info["preconditioner_metadata"]["kind"] == {"auto": "point", "all": "all_fixed"}.get(kind, kind)
        if not heterogeneous:
            assert info["iterations"] <= 1


@pytest.mark.parametrize("maxiter", [0, 1, 10])
@pytest.mark.parametrize("kind", ["auto", "faces", "all"])
def test_zero_rhs_and_maxiter_zero(kind, maxiter):
    shape = (3, 4, 5)
    fixed = fixed_for(shape, kind)
    if fixed is not None:
        fixed[np.isfinite(fixed)] = 0
    out, info = solver.solve_field(
        np.zeros((*shape, 3)), np.ones(shape), fixed=fixed, dtype="float64", preconditioner="spectral", maxiter=maxiter
    )
    assert info["converged"] and info["iterations"] == 0
    np.testing.assert_array_equal(out, 0)
    if kind != "all":
        initial = np.ones(shape)
        out, info = solver.solve_field(
            np.zeros((*shape, 3)),
            np.ones(shape),
            fixed=fixed,
            initial=initial,
            dtype="float64",
            preconditioner="spectral",
            maxiter=maxiter,
        )
        assert info["converged"] == (maxiter > 0)
        if maxiter == 0:
            mask = effective_mask(shape, fixed)
            np.testing.assert_array_equal(out[~mask], initial[~mask])


@pytest.mark.parametrize("geometry", ["two", "one_face", "faces_plus_interior", "missing_face_node", "reference"])
def test_spectral_rejects_unsupported_effective_masks(geometry):
    shape = (4, 5, 6)
    fixed = np.full(shape, np.nan)
    kwargs = {}
    if geometry == "two":
        fixed.flat[[0, 1]] = 0
    elif geometry == "one_face":
        fixed[0] = 0
    elif geometry == "reference":
        kwargs = dict(reference=np.zeros(shape), reference_weight=1)
    else:
        fixed = fixed_for(shape, "faces")
        if geometry == "faces_plus_interior":
            fixed[1, 1, 1] = 0
        else:
            fixed[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="spectral"):
        solver.solve_field(
            np.zeros((*shape, 3)), np.ones(shape), fixed=fixed, dtype="float64", preconditioner="spectral", **kwargs
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preconditioner": "unknown"},
        {"preconditioner": None},
        {"preconditioner": "spectral", "dtype": "float32"},
        {"preconditioner": "spectral", "device": "cuda"},
        {"preconditioner": "spectral", "device": "mps"},
    ],
)
def test_invalid_policy_backend_dtype(kwargs):
    options = dict(dtype="float64") | kwargs
    with pytest.raises(ValueError, match="precondition"):
        solver.solve_field(np.zeros((2, 3, 4, 3)), np.ones((2, 3, 4)), **options)


@pytest.mark.parametrize("value", [np.finfo(float).max, np.nextafter(0.0, 1.0)])
def test_spectral_rejects_unrepresentable_mean_or_inverse(value):
    mask = torch.zeros((3, 4, 5), dtype=torch.bool)
    mask[0, 0, 0] = True
    with pytest.raises(ValueError, match="spectral"):
        solver._SpectralPreconditioner(torch.full(mask.shape, value, dtype=torch.float64), mask)


@pytest.mark.parametrize("mutation", ["mask", "weights", "policy", "structure", "shape", "dtype"])
def test_prepared_spectral_invalidates_changed_inputs(mutation):
    shape = (3, 4, 5)
    target, weight = np.zeros((*shape, 3)), np.ones(shape)
    fixed = fixed_for(shape, "point")
    c = rows_for(shape)
    prepared = solver.PreparedField()
    kwargs = dict(dtype="float64", preconditioner="spectral", fixed=fixed, constraints=c, prepared=prepared)
    solver.solve_field(target, weight, **kwargs)
    if mutation == "mask":
        fixed[:] = np.nan
        fixed.flat[0] = 1  # Same count, different complete mask.
    elif mutation == "weights":
        torch.from_numpy(weight).numpy().flat[0] = 2
    elif mutation == "policy":
        kwargs["preconditioner"] = "jacobi"
    elif mutation == "structure":
        c.groups[0] = 0
    elif mutation == "shape":
        target, weight = target.transpose(1, 0, 2, 3), weight.transpose(1, 0, 2)
        kwargs["fixed"] = fixed.transpose(1, 0, 2)
    else:
        kwargs.update(dtype="float32", preconditioner="jacobi")
    with pytest.raises(ValueError, match="prepared"):
        solver.solve_field(target, weight, **kwargs)


@pytest.mark.parametrize("kind", ["point", "faces"])
def test_prepared_targets_fixed_values_are_live_and_retention_counts_spectral_arrays(kind):
    shape = (3, 4, 5)
    target, weight = np.zeros((*shape, 3)), np.ones(shape)
    fixed = fixed_for(shape, kind)
    c = rows_for(shape)
    prepared = solver.PreparedField()
    spectral = None
    for value in [0, 0.7, -0.9]:
        c.targets[:] = value
        fixed[np.isfinite(fixed)] = value
        kwargs = dict(dtype="float64", preconditioner="spectral", fixed=fixed, constraints=c, rtol=1e-11)
        out, info = solver.solve_field(target, weight, prepared=prepared, **kwargs)
        expected, _ = solver.solve_field(target, weight, **kwargs)
        np.testing.assert_array_equal(out, expected)
        assert info["converged"]
        if spectral is not None:
            assert spectral is prepared.spectral
        spectral = prepared.spectral
    retained = prepared.retained_bytes
    spectral.extra_retained_test_array = np.empty(1000, dtype=np.float64)
    assert prepared.retained_bytes >= retained + 8000


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cpu_prepared_rows_detect_all_structural_alias_mutations(index, dtype):
    shape = (2, 3, 4)
    c = rows_for(shape)
    alias = c._structural_arrays()[index]
    c = c.with_targets(c.targets)
    workspace = solver.PreparedField()
    target, weights = np.zeros((*shape, 3)), np.ones(shape)
    solver.solve_field(target, weights, constraints=c, prepared=workspace, dtype=dtype)
    alias.setflags(write=True)
    alias.flat[0] += 1
    with pytest.raises(ValueError, match="prepared"):
        solver.solve_field(target, weights, constraints=c, prepared=workspace, dtype=dtype)


def test_spectral_applied_to_initial_and_recomputed_restart_residual(monkeypatch):
    shape = (3, 4, 5)
    rng = np.random.default_rng(46)
    calls = []
    apply = solver._SpectralPreconditioner.apply

    def counted(self, residual):
        calls.append(residual.clone())
        return apply(self, residual)

    monkeypatch.setattr(solver._SpectralPreconditioner, "apply", counted)
    _, info = solver.solve_field(
        rng.normal(size=(*shape, 3)), np.ones(shape), dtype="float64", preconditioner="spectral", rtol=1e-11
    )
    assert info["converged"] and info["iterations"] == 1
    assert len(calls) == 2
    assert torch.linalg.vector_norm(calls[1]) < 1e-11 * torch.linalg.vector_norm(calls[0])


@pytest.mark.parametrize("maxiter", [0, 1, 7, 100])
def test_spectral_forced_restart_continues_with_true_residual(monkeypatch, maxiter):
    rng = np.random.default_rng(92)
    shape = (3, 4, 5)
    target = rng.normal(size=(*shape, 3))
    weights = rng.uniform(0.2, 5, shape)
    fixed = fixed_for(shape, "point")
    mask = np.isfinite(fixed).ravel()
    matrix, rhs = dense_system(target, weights, (2, 3, 4))
    reduced = matrix[np.ix_(~mask, ~mask)]
    reduced_rhs = rhs[~mask] - matrix[np.ix_(~mask, mask)] @ fixed.ravel()[mask]
    scale = np.linalg.norm(reduced_rhs)
    rtol = 1e-11
    norms, applications, events = [], [], []
    original_norm = solver._norm
    original_apply = solver._SpectralPreconditioner.apply
    original_matvec = solver._FieldOperator.matvec

    def force_first_restart(value):
        norm = original_norm(value)
        norms.append(norm)
        # Simulate a falsely small recurrence norm, only after the first step.
        return 0.0 if len(norms) == 3 else norm

    def record_apply(self, residual):
        result = original_apply(self, residual)
        applications.append((residual.numpy().copy(), result.numpy().copy()))
        events.append(("precondition", result.numpy().copy()))
        return result

    def record_matvec(self, value, **kwargs):
        events.append(("matvec", value.numpy().copy()))
        return original_matvec(self, value, **kwargs)

    monkeypatch.setattr(solver, "_norm", force_first_restart)
    monkeypatch.setattr(solver._SpectralPreconditioner, "apply", record_apply)
    monkeypatch.setattr(solver._FieldOperator, "matvec", record_matvec)
    field, info = solver.solve_field(
        target,
        weights,
        (2, 3, 4),
        fixed=fixed,
        dtype="float64",
        preconditioner="spectral",
        rtol=rtol,
        maxiter=maxiter,
    )
    np.testing.assert_array_equal(field.ravel()[mask], fixed.ravel()[mask])
    true_residual = np.linalg.norm(reduced_rhs - reduced @ field.ravel()[~mask]) / scale
    assert info["relative_residual"] == pytest.approx(true_residual, rel=1e-10, abs=2e-15)
    assert info["converged"] == (true_residual <= rtol)
    if maxiter == 0:
        assert info["iterations"] == 0
        np.testing.assert_array_equal(field.ravel()[~mask], 0)
        return
    assert norms[2] > rtol * scale  # The forced restart happened before convergence.
    z = applications[0][1].ravel()[~mask]
    first_step = z * (reduced_rhs @ z) / (z @ reduced @ z)
    np.testing.assert_allclose(applications[1][0].ravel()[~mask], reduced_rhs - reduced @ first_step, atol=2e-14)
    if maxiter > 1:
        assert info["iterations"] > 1
        second_apply = [i for i, (kind, _) in enumerate(events) if kind == "precondition"][1]
        kind, direction = events[second_apply + 1]
        assert kind == "matvec"
        np.testing.assert_array_equal(direction, applications[1][1])
        assert true_residual < np.linalg.norm(reduced_rhs - reduced @ first_step) / scale
    if maxiter == 100:
        assert info["converged"]
        np.testing.assert_allclose(field.ravel()[~mask], np.linalg.solve(reduced, reduced_rhs), atol=2e-10)
