"""AMG preconditioning changes iteration counts only, never the solved system."""

import weakref

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import field_amg
from field_constraints import QuadraticConstraints
from field_solver import PreparedField, _FieldOperator, solve_field


def _problem(n=16, seed=0):
    rng = np.random.default_rng(seed)
    shape = (n, n + 3, n + 1)
    z, y, x = np.indices(shape)
    r = np.hypot(y - shape[1] / 2, x - shape[2] / 2)
    sheets = (np.abs(r / 3 - np.round(r / 3)) < 0.3) & (rng.random(shape) > 0.2)
    weights = np.where(sheets, 1.0, 1e-3)
    target = np.zeros((*shape, 3))
    target[..., 0] = (x - shape[2] / 2) / np.maximum(r, 1e-6) / 3
    target[..., 1] = (y - shape[1] / 2) / np.maximum(r, 1e-6) / 3
    target += 0.05 * rng.standard_normal(target.shape)
    flat = np.flatnonzero(sheets)
    pairs = np.stack([flat[:-1:7], flat[1::7]], 1)
    return target, weights, pairs


def _constraints(shape, seed=1, groups=False):
    rng = np.random.default_rng(seed)
    n, rows = int(np.prod(shape)), 60
    cols = rng.integers(0, n, (rows, 2))
    matrix = sp.csr_matrix((np.tile([1.0, -1.0], rows), (np.repeat(np.arange(rows), 2), cols.ravel())), (rows, n))
    group = np.repeat(np.arange(rows // 6), 6) if groups else None
    return QuadraticConstraints(matrix, weights=50.0, targets=rng.normal(0, 0.1, rows), groups=group)


@pytest.mark.parametrize("groups", [False, True])
def test_assembled_matrix_matches_matrix_free_operator(groups):
    target, weights, pairs = _problem(8)
    shape = weights.shape
    constraints = _constraints(shape, groups=groups)
    mask = np.zeros(shape, bool)
    mask[0] = True
    rw = np.random.default_rng(2).random(shape)
    matrix = field_amg.assemble_free_operator(
        weights, mask, pairs=pairs, lam=3.0, constraints=constraints, reference_weight=rw
    )
    w = torch.tensor(weights)
    op = _FieldOperator(w, torch.tensor(pairs), 3.0)
    extra = constraints.operator("cpu", np.float64, layout="csr")
    v = torch.tensor(np.random.default_rng(3).standard_normal(shape))
    v[torch.tensor(mask)] = 0
    expected = op.matvec(v) + extra.matvec(v).reshape(shape) + torch.tensor(rw) * v
    expected[torch.tensor(mask)] = 0
    np.testing.assert_allclose(matrix @ v.numpy().ravel(), expected.numpy().ravel(), rtol=1e-10, atol=1e-10)
    assert abs(matrix - matrix.T).max() < 1e-12


@pytest.mark.parametrize("setup,smoother", [("torch", "chebyshev"), ("torch", "l1-jacobi"), ("pyamg", "chebyshev")])
def test_preconditioner_is_symmetric_and_positive(setup, smoother):
    _, weights, pairs = _problem(10)
    mask = np.zeros(weights.shape, bool)
    mask.flat[0] = True
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    amg = field_amg.AMGPreconditioner(
        matrix, weights.shape, torch.tensor(mask), max_coarse=50, smoother=smoother, setup=setup
    )
    rng = np.random.default_rng(4)
    u, v = (torch.tensor(rng.standard_normal(weights.shape)) for _ in range(2))
    u[torch.tensor(mask)] = v[torch.tensor(mask)] = 0
    mu, mv = amg.apply(u), amg.apply(v)
    assert float((u * mv).sum()) == pytest.approx(float((v * mu).sum()), rel=1e-9)
    assert float((u * mu).sum()) > 0
    assert amg.metadata()["levels"] >= 2
    assert amg.metadata()["smoother"] == smoother
    assert amg.metadata()["setup"] == setup
    with pytest.raises(ValueError, match="smoother must be"):
        field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), smoother="sor")
    with pytest.raises(ValueError, match="setup must be"):
        field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), setup="ruge")


def test_chebyshev_bounds_cover_every_level_spectrum():
    # An lmax below the top eigenvalue of D^-1 A amplifies those modes and can make the
    # V-cycle indefinite (an occasional nonpositive CG curvature on real 175A blocks).
    _, weights, pairs = _problem(10)
    mask = np.zeros(weights.shape, bool)
    mask.flat[0] = True
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    amg = field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), max_coarse=50)
    assert len(amg.bounds) == len(amg.levels) >= 1
    for (a, _, _, scale), (lmin, lmax) in zip(amg.levels, amg.bounds):
        dense = (scale[:, None] * a.to_dense()).double().numpy()
        top = float(np.max(np.linalg.eigvals(dense).real))
        assert top <= lmax * (1 + 1e-6)
        assert lmax <= 1.5 * top
        assert lmin == pytest.approx(lmax / 30)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_amg_solution_matches_jacobi_with_far_fewer_iterations(dtype):
    target, weights, pairs = _problem(18)
    constraints = _constraints(weights.shape)
    # float32 CG floors near 3e-5 on this lam=1000 system with either preconditioner.
    rtol = 1e-8 if dtype == "float64" else 1e-4
    kwargs = dict(pairs=pairs, lam=1000.0, constraints=constraints, rtol=rtol, maxiter=50000, dtype=dtype)
    jacobi, ji = solve_field(target, weights, preconditioner="jacobi", **kwargs)
    amg, ai = solve_field(target, weights, preconditioner="amg", **kwargs)
    assert ji["converged"] and ai["converged"]
    assert ai["relative_residual"] <= rtol
    assert ai["preconditioner"] == "amg" and ai["preconditioner_metadata"]["kind"] == "amg"
    assert ai["iterations"] * 4 < ji["iterations"]
    np.testing.assert_allclose(amg, jacobi, atol=2e-3 if dtype == "float32" else 1e-5)


def test_fixed_values_and_prepared_hierarchy_reuse():
    target, weights, _ = _problem(12)
    constraints = _constraints(weights.shape)
    fixed = np.full(weights.shape, np.nan)
    fixed[0], fixed[-1] = 1.0, 2.0
    kwargs = dict(constraints=constraints, fixed=fixed, rtol=1e-9, maxiter=50000, dtype="float64")
    reference, _ = solve_field(target, weights, **kwargs)
    prepared = PreparedField()
    first, info = solve_field(target, weights, preconditioner="amg", prepared=prepared, **kwargs)
    hierarchy = prepared.amg
    second, _ = solve_field(target, weights, preconditioner="amg", prepared=prepared, initial=first, **kwargs)
    assert prepared.amg is hierarchy and info["converged"]
    np.testing.assert_allclose(first, reference, atol=1e-6)
    np.testing.assert_allclose(second, reference, atol=1e-6)
    assert np.all(first[0] == 1.0) and np.all(first[-1] == 2.0)


def test_spectral_and_unknown_policies_still_rejected_for_amg_mismatch():
    target, weights, _ = _problem(6)
    with pytest.raises(ValueError, match="jacobi, spectral or amg"):
        solve_field(target, weights, preconditioner="ilu")
    prepared = PreparedField()
    solve_field(target, weights, preconditioner="amg", prepared=prepared, dtype="float64")
    with pytest.raises(ValueError, match="policy changed"):
        solve_field(target, weights, preconditioner="jacobi", prepared=prepared, dtype="float64")


def test_blockwise_galerkin_matches_direct_product():
    _, weights, pairs = _problem(8)
    mask = np.zeros(weights.shape, bool)
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    rng = np.random.default_rng(5)
    p = sp.random(matrix.shape[0], 40, density=0.05, random_state=rng, format="csr") + sp.eye(matrix.shape[0], 40)
    a_t = field_amg._torch_csr(matrix, "cpu", torch.float64)
    p_t = field_amg._torch_csr(p, "cpu", torch.float64)
    blocked = field_amg._galerkin(a_t, p_t, block_nnz=97).to_dense().numpy()
    np.testing.assert_allclose(blocked, (p.T @ matrix @ p).toarray(), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("budget", [1, 7, 1 << 24])
def test_chunked_expansion_matches_sparse_product(budget):
    rng = np.random.default_rng(6)
    left = sp.random(30, 25, density=0.2, random_state=rng, format="coo")
    right = sp.random(25, 20, density=0.3, random_state=rng, format="csr")
    out = field_amg._expand(
        torch.as_tensor(left.row, dtype=torch.int64),
        torch.as_tensor(left.col, dtype=torch.int64),
        torch.as_tensor(left.data),
        field_amg._torch_csr(right, "cpu", torch.float64),
        (30, 20),
        budget=budget,
    )
    np.testing.assert_allclose(out.to_dense().numpy(), (left @ right).toarray(), rtol=1e-12, atol=1e-12)


def test_fixed_faces_are_eliminated_before_coarsening():
    # Fixed nodes are isolated identity rows; kept in the hierarchy they never aggregate and a
    # block with fixed faces ended in a dense coarsest level of thousands of nodes.
    rng = np.random.default_rng(3)
    shape = (24, 24, 24)
    weights = np.where(rng.random(shape) > 0.5, 1.0, 1e-3)
    mask = np.zeros(shape, bool)
    mask[[0, -1]] = True
    mask[:, [0, -1]] = True
    mask[:, :, [0, -1]] = True
    matrix = field_amg.assemble_free_operator(weights, mask)
    amg = field_amg.AMGPreconditioner(matrix, shape, torch.as_tensor(mask), max_coarse=200)
    assert amg.coarse.shape[0] <= 200
    r = torch.as_tensor(rng.standard_normal(shape))
    out = amg.apply(r)
    assert bool((out[torch.as_tensor(mask)] == 0).all()) and bool(torch.isfinite(out).all())
    everything = field_amg.AMGPreconditioner(
        field_amg.assemble_free_operator(weights, np.ones(shape, bool)), shape, torch.ones(shape, dtype=torch.bool)
    )
    assert bool((everything.apply(r) == 0).all())


def _serial_reference(weights, mask, pairs=None, lam=0.0, constraints=None, reference_weight=0.0):
    """The single-threaded assembly that the row-block assembly must reproduce bit for bit."""
    w = np.asarray(weights, dtype=np.float64)
    shape, n = w.shape, w.size
    index = np.arange(n, dtype=np.int32).reshape(shape)
    rows, cols, vals = [], [], []
    diagonal = np.zeros(n)
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = index[tuple(lo)].ravel(), index[tuple(hi)].ravel()
        edge = (0.5 * w[tuple(lo)] + 0.5 * w[tuple(hi)]).ravel()
        rows += [a, b]
        cols += [b, a]
        vals += [-edge, -edge]
        for ends in (a, b):
            diagonal += np.bincount(ends, weights=edge, minlength=n)
    if pairs is not None and lam:
        p = np.asarray(pairs).reshape(-1, 2).astype(np.int32)
        p = p[p[:, 0] != p[:, 1]]
        coupling = np.full(len(p), float(lam))
        rows += [p[:, 0], p[:, 1]]
        cols += [p[:, 1], p[:, 0]]
        vals += [-coupling, -coupling]
        for ends in (p[:, 0], p[:, 1]):
            diagonal += np.bincount(ends, weights=coupling, minlength=n)
    rows.append(index.ravel())
    cols.append(index.ravel())
    vals.append(diagonal + np.broadcast_to(np.asarray(reference_weight, dtype=np.float64), shape).ravel())
    matrix = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), (n, n)).tocsr()
    if constraints is not None:
        c = sp.csr_matrix(constraints.matrix, dtype=np.float64)
        matrix = matrix + (c.T @ sp.diags(constraints.weights) @ c).tocsr()
        if (constraints.groups >= 0).any():
            means = (constraints.members @ c).tocsr()
            inverse = np.divide(
                1, constraints.group_sum, out=np.zeros_like(constraints.group_sum), where=constraints.group_sum > 0
            )
            matrix = matrix - (means.T @ sp.diags(inverse) @ means).tocsr()
    fixed = np.asarray(mask, dtype=bool).ravel()
    if fixed.any():
        keep = sp.diags((~fixed).astype(np.float64))
        matrix = (keep @ matrix @ keep + sp.diags(fixed.astype(np.float64))).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _bitwise_equal(a, b):
    return (
        a.shape == b.shape
        and np.array_equal(a.indptr, b.indptr)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.data.view(np.int64), b.data.view(np.int64))
    )


@pytest.mark.parametrize("groups", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_threaded_assembly_is_bitwise_serial(groups, masked):
    _, weights, pairs = _problem(12)
    shape = weights.shape
    constraints = _constraints(shape, groups=groups)
    mask = np.zeros(shape, bool)
    if masked:
        mask[0] = mask[:, -1] = True
    rw = np.random.default_rng(2).random(shape)
    kwargs = dict(pairs=pairs, lam=1000.0, constraints=constraints, reference_weight=rw)
    expected = _serial_reference(weights, mask, **kwargs)
    for threads in (1, 2, 5):
        got = field_amg.assemble_free_operator(weights, mask, threads=threads, **kwargs)
        assert _bitwise_equal(got, expected)
        keep = np.flatnonzero(~mask.ravel())
        assert _bitwise_equal(field_amg._free_submatrix(got, keep, threads), expected[keep][:, keep].tocsr())
    with pytest.raises(ValueError, match="threads must be positive"):
        field_amg.assemble_free_operator(weights, mask, threads=0)


def test_threaded_setup_gives_the_same_preconditioner():
    _, weights, pairs = _problem(10)
    mask = np.zeros(weights.shape, bool)
    mask[0] = True
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    one, four = (
        field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), max_coarse=50, threads=t) for t in (1, 4)
    )
    v = torch.tensor(np.random.default_rng(4).standard_normal(weights.shape))
    assert torch.equal(one.apply(v), four.apply(v))


def _csr_bitwise_equal(a, b):
    return (
        a.shape == b.shape
        and torch.equal(a.crow_indices(), b.crow_indices())
        and torch.equal(a.col_indices(), b.col_indices())
        and a.values().dtype == b.values().dtype
        and torch.equal(_bits(a.values()), _bits(b.values()))
    )


def _bits(values):
    return values.view(torch.int64 if values.element_size() == 8 else torch.int32)


def test_galerkin_blocks_give_the_same_coarse_operator():
    _, weights, pairs = _problem(12)
    mask = np.zeros(weights.shape, bool)
    mask[0] = True
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    levels, _, _ = field_amg._torch_hierarchy(matrix, "cpu", max_coarse=50, max_levels=2, dense=False)
    a, p = levels[0]
    dense = (p.to_dense().T @ a.to_dense() @ p.to_dense()).numpy()
    reference = None
    for block_nnz in (1 << 30, 4096, 256):
        coarse = field_amg._galerkin(a, p, block_nnz=block_nnz)
        assert coarse.layout == torch.sparse_csr
        rows = torch.repeat_interleave(torch.arange(coarse.shape[0]), coarse.crow_indices().diff())
        keys = rows * coarse.shape[1] + coarse.col_indices()
        assert bool((keys[1:] > keys[:-1]).all())  # sorted and unique, as a coalesced product
        np.testing.assert_allclose(coarse.to_dense().numpy(), dense, rtol=1e-12, atol=1e-9)
        if block_nnz == 4096:
            reference = coarse
    assert reference is not None and _csr_bitwise_equal(field_amg._galerkin(a, p, block_nnz=4096), reference)


@pytest.mark.parametrize("block", [97, 1000, 4096])
def test_row_blocks_build_the_one_block_hierarchy_bit_for_bit(monkeypatch, block):
    # torch's own coalesce uses an unstable sort below ~32k entries; with every coalesce on
    # the stable host path, each entry of a blocked build must be the same left-to-right sum.
    monkeypatch.setattr(field_amg, "_HOST_COALESCE_MIN", 0)
    _, weights, pairs = _problem(12)
    mask = np.zeros(weights.shape, bool)
    mask[0] = True
    matrix = field_amg._free_submatrix(
        field_amg.assemble_free_operator(
            weights, mask, pairs=pairs, lam=1000.0, constraints=_constraints(weights.shape)
        ),
        np.flatnonzero(~mask.ravel()),
    )
    kwargs = dict(max_coarse=20, max_levels=5, dense=False)
    levels, coarsest, complexity = field_amg._torch_hierarchy(matrix, "cpu", block=1 << 30, **kwargs)
    blocked, coarsest_b, complexity_b = field_amg._torch_hierarchy(matrix, "cpu", block=block, **kwargs)
    assert len(levels) == len(blocked) >= 2 and complexity == complexity_b
    assert _csr_bitwise_equal(coarsest, coarsest_b)
    for (a, p), (a_b, p_b) in zip(levels, blocked):
        assert _csr_bitwise_equal(a, a_b) and _csr_bitwise_equal(p, p_b)


def test_blocked_sums_run_left_to_right_at_any_block_size(monkeypatch):
    # One coalesce of a whole large operator sums every entry left to right in storage order
    # (stable host path). Small blocks must do the same, although torch's own coalesce of
    # fewer than ~32k entries sorts unstably.
    _, weights, pairs = _problem(12)
    matrix = field_amg.assemble_free_operator(
        weights, np.zeros(weights.shape, bool), pairs=pairs, lam=1000.0, constraints=_constraints(weights.shape)
    )
    a = field_amg._torch_csr(matrix, "cpu", torch.float64)
    aggregates = field_amg._aggregate(a, 0.25, 2, seed=0)
    size = int(aggregates.max()) + 1
    default = field_amg._HOST_COALESCE_MIN
    monkeypatch.setattr(field_amg, "_HOST_COALESCE_MIN", 0)
    reduced = field_amg._galerkin_aggregate(a, aggregates, size, block=1 << 30)
    prolong = field_amg._smoothed_prolongator(a, aggregates, size, 0.2, block=1 << 30)
    monkeypatch.setattr(field_amg, "_HOST_COALESCE_MIN", default)
    for block in (97, 1000):
        assert _csr_bitwise_equal(field_amg._galerkin_aggregate(a, aggregates, size, block=block), reduced)
        assert _csr_bitwise_equal(field_amg._smoothed_prolongator(a, aggregates, size, 0.2, block=block), prolong)


def _reference_matching(a, theta, seed, rounds=6):
    """The one-shot matching that the blocked one must reproduce, run on one thread."""
    n = a.shape[0]
    crow = a.crow_indices()
    rows = torch.repeat_interleave(torch.arange(n), crow.diff())
    cols, vals = a.col_indices(), a.values()
    diagonal = torch.zeros(n, dtype=vals.dtype).index_add_(0, rows[rows == cols], vals[rows == cols])
    keep = (rows != cols) & (vals < 0)
    rows, cols = rows[keep], cols[keep]
    strength = -vals[keep] / torch.sqrt(diagonal[rows] * diagonal[cols])
    rowmax = torch.zeros(n, dtype=strength.dtype).scatter_reduce_(0, rows, strength, "amax")
    keep = strength >= theta * rowmax[rows]
    rows, cols, strength = rows[keep], cols[keep], strength[keep]
    jitter = (torch.minimum(rows, cols) * 2654435761 + torch.maximum(rows, cols) * 97531 + seed * 7919) % 1000003
    score = (1.0 + jitter.to(strength.dtype) * (0.1 / 1000003.0)) * strength
    nodes = torch.arange(n)
    partner = torch.full((n,), -1)
    for _ in range(rounds):
        free = partner < 0
        active = free[rows] & free[cols]
        if not bool(active.any()):
            break
        r, c, sc = rows[active], cols[active], score[active]
        best = torch.full((n,), -1.0, dtype=sc.dtype).scatter_reduce_(0, r, sc, "amax")
        pick = sc == best[r]
        choice = torch.full((n,), -1)
        choice[r[pick]] = c[pick]  # serial: the last pick of a row wins
        chosen = torch.nonzero(choice >= 0).squeeze(1)
        chosen = chosen[choice[choice[chosen]] == chosen]
        partner[chosen] = choice[chosen]
    root = torch.where(partner >= 0, torch.minimum(nodes, partner), nodes)
    single = partner < 0
    edge = single[rows] & ~single[cols]
    r, c, sc = rows[edge], cols[edge], strength[edge]
    best = torch.full((n,), -1.0, dtype=sc.dtype).scatter_reduce_(0, r, sc, "amax")
    pick = sc == best[r]
    root[r[pick]] = root[c[pick]]
    return torch.unique(root, return_inverse=True)[1]


@pytest.mark.parametrize("uniform", [False, True])
def test_blocked_matching_is_the_serial_one_shot_matching(uniform):
    # Uniform weights make every coupling of a row equally strong: ties go to the last entry.
    _, weights, pairs = _problem(12)
    if uniform:
        weights = np.ones_like(weights)
    matrix = field_amg.assemble_free_operator(weights, np.zeros(weights.shape, bool), pairs=pairs, lam=3.0)
    a = field_amg._torch_csr(matrix, "cpu", torch.float64)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = [_reference_matching(a, 0.25, seed) for seed in (0, 7)]
    finally:
        torch.set_num_threads(threads)
    for block in (1 << 30, 500, 61):
        for seed, reference in zip((0, 7), expected):
            assert torch.equal(field_amg._pairwise_aggregates(a, 0.25, seed, block=block), reference)


def test_compact_transpose_is_the_coalesced_transpose():
    rng = np.random.default_rng(8)
    p = field_amg._torch_csr(sp.random(300, 70, density=0.05, random_state=rng, format="csr"), "cpu", torch.float64)
    expected = field_amg._compact(field_amg._coalesce(p.to_sparse_coo().t()).to_sparse_csr(), torch.float32, "cpu")
    for block in (1 << 30, 50, 7):
        got = field_amg._compact_transpose(p, torch.float32, "cpu", block=block)
        assert got.crow_indices().dtype == torch.int32 and _csr_bitwise_equal(got, expected)


@pytest.mark.parametrize("smoother", ["chebyshev", "l1-jacobi"])
def test_levels_handed_over_during_setup_match_the_kept_hierarchy(smoother):
    _, weights, pairs = _problem(12)
    mask = np.zeros(weights.shape, bool)
    mask[0] = True
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=1000.0)
    amg = field_amg.AMGPreconditioner(
        matrix, weights.shape, torch.tensor(mask), max_coarse=50, dtype=torch.float32, smoother=smoother
    )
    free = field_amg._free_submatrix(matrix, np.flatnonzero(~mask.ravel()))
    levels, coarsest, complexity = field_amg._torch_hierarchy(free, "cpu", max_coarse=50, max_levels=12)
    assert len(amg.levels) == len(levels) >= 2 and amg.complexity == complexity
    for k, ((a, p), (a_c, p_c, pt_c, scale_c)) in enumerate(zip(levels, amg.levels)):
        pt = field_amg._coalesce(p.to_sparse_coo().t()).to_sparse_csr()
        for full, compact in ((a, a_c), (p, p_c), (pt, pt_c)):
            assert _csr_bitwise_equal(field_amg._compact(full, torch.float32, "cpu"), compact)
        if smoother == "chebyshev":
            scale = 1.0 / field_amg._diagonal(a)
            lmax = field_amg._chebyshev_bound(a, scale)
            assert amg.bounds[k] == (lmax / 30.0, lmax)
        else:
            scale = 1.0 / field_amg._absolute_row_sums(a)
        assert torch.equal(_bits(scale.float()), _bits(scale_c))
    assert torch.equal(amg.coarse, torch.linalg.pinv(0.5 * (coarsest + coarsest.T), hermitian=True).float())


def test_out_of_memory_fallback_releases_the_failed_attempt(monkeypatch):
    real = field_amg._torch_hierarchy
    held: list = []

    def flaky(matrix, device, **kwargs):
        if not held:
            workspace = torch.empty(16)  # stands in for the failed attempt's device tensors
            held.append(weakref.ref(workspace))
            raise torch.OutOfMemoryError("simulated")
        assert held[0]() is None, "the failed attempt is still referenced during the host rebuild"
        held.append(device)
        return real(matrix, device, **kwargs)

    monkeypatch.setattr(field_amg, "_torch_hierarchy", flaky)
    _, weights, pairs = _problem(8)
    mask = np.zeros(weights.shape, bool)
    matrix = field_amg.assemble_free_operator(weights, mask, pairs=pairs, lam=10.0)
    amg = field_amg.AMGPreconditioner(matrix, weights.shape, torch.tensor(mask), max_coarse=50)
    assert held[1] == "cpu"
    assert amg.metadata()["levels"] >= 2


def test_host_coalesce_is_bitwise_torch_coalesce():
    generator = torch.Generator().manual_seed(0)
    size = 4 * field_amg._HOST_COALESCE_MIN
    for rows, cols in ((40, 30), (5000, 4000)):  # long runs of duplicates, then mostly unique
        index = torch.stack(
            [torch.randint(0, rows, (size,), generator=generator), torch.randint(0, cols, (size,), generator=generator)]
        )
        scale = torch.pow(10.0, torch.randint(-8, 9, (size,), generator=generator).double())
        values = torch.randn(size, generator=generator, dtype=torch.float64) * scale
        values[::97] = -0.0
        raw = torch.sparse_coo_tensor(index, values, (rows, cols))
        expected, got = raw.coalesce(), field_amg._coalesce(raw)
        assert got.is_coalesced()
        assert torch.equal(got.indices(), expected.indices())
        assert torch.equal(got.values().view(torch.int64), expected.values().view(torch.int64))
