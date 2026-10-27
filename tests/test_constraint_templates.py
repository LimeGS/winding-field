import gc
import importlib.util
import io
import weakref
import zipfile

import numpy as np
import pytest
import torch

from field_constraints import consensus_constraints, pair_constraints


def implementation():
    assert importlib.util.find_spec("constraint_templates"), "bounded physical stencil templates are missing"
    from constraint_templates import StencilConstraints

    return StencilConstraints


ORIGIN = np.array([10.0, -7.0, 20.0])
SPACING = np.array([2.0, 3.0, 5.0])
SHAPE = (4, 5, 6)
LO, HI = np.array([1, 1, 2]), np.array([4, 4, 5])


def local_nodes():
    return np.arange(np.prod(SHAPE)).reshape(SHAPE)[1:4, 1:4, 2:5].ravel()


def test_consensus_keeps_exterior_only_rows_and_matches_global_restriction():
    points = ORIGIN + SPACING * [[2.1, 1.2, 1.1], [4.7, 2.6, 2.2], [0.2, 4.0, 0.3]]
    weights = np.array([1.2, 0.7, 3.0])
    compiled = implementation().consensus(points, [7, 7, 7], weights, ORIGIN, SPACING, SHAPE, cache_bytes=10000)
    full = consensus_constraints(points, [7, 7, 7], weights, ORIGIN, SPACING, SHAPE)
    values = np.arange(np.prod(SHAPE), dtype=float) / 7
    calls = []

    def read(nodes):
        calls.append(nodes.copy())
        return values[nodes]

    for update in (0.0, 3.0):
        values += update
        expected = full.restrict(local_nodes(), values)
        actual = compiled.restrict(LO, HI, read)
        np.testing.assert_array_equal(actual.matrix.toarray(), expected.matrix.toarray())
        np.testing.assert_array_equal(actual.weights, expected.weights)
        np.testing.assert_array_equal(actual.groups, expected.groups)
        np.testing.assert_allclose(actual.targets, expected.targets, atol=1e-13)
        np.testing.assert_array_equal(actual.diagonal, expected.diagonal)
        assert actual.matrix.getrow(2).nnz == 0 and len(actual.targets) == 3
    assert len(calls) == 2  # One exterior batch for each restriction, including warm reuse.
    assert compiled.stats["hits"] == 1 and compiled.stats["bytes"] <= 10000


def test_pair_templates_preserve_same_node_cancellation_and_current_rhs():
    a = ORIGIN + SPACING * [[2.1, 1.2, 1.1], [3.2, 2.3, 1.4]]
    b = ORIGIN + SPACING * [[5.0, 3.1, 2.9], [3.2, 2.3, 1.4]]
    compiled = implementation().pairs(
        a, b, ORIGIN, SPACING, SHAPE, weight=[2.0, 3.0], targets=[0.2, 0.0], cache_bytes=10000
    )
    reference = pair_constraints(a, b, ORIGIN, SPACING, SHAPE, weight=[2.0, 3.0], targets=[0.2, 0.0])
    values = np.random.default_rng(4).normal(size=np.prod(SHAPE))
    result = compiled.restrict(LO, HI, lambda nodes: values[nodes])
    expected = reference.restrict(local_nodes(), values)
    np.testing.assert_array_equal(result.matrix.toarray(), expected.matrix.toarray())
    np.testing.assert_allclose(result.targets, expected.targets, atol=1e-14)
    x = torch.arange(len(local_nodes()), dtype=torch.float64)
    got = result.operator("cpu", np.float64)
    want = expected.operator("cpu", np.float64)
    np.testing.assert_allclose((got.matvec(x) - got.rhs()).numpy(), (want.matvec(x) - want.rhs()).numpy(), atol=1e-12)
    assert result.matrix.getrow(1).nnz == 0


def test_repeated_slot_reduction_is_bit_exact_and_inputs_are_owned():
    nodes = np.array([[0, 1, 1, 7], [0, 7, 7, 1]])
    coefficients = np.array([[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]])
    field = np.array([1e12, 0.3, 0, 0, 0, 0, 0, -1e12])
    compiled = implementation()(nodes, coefficients, (2, 2, 2), groups=[0, 0], cache_bytes=10000)
    identity = compiled.digest
    expected = -np.array(
        [[field[0] * 0.1, 0, 0, field[7] * 0.4], [field[0] * -0.1, field[7] * -0.2, field[7] * -0.3, 0]]
    ).sum(1)
    nodes[:] = 5
    coefficients[:] = 0
    result = compiled.restrict([0, 0, 1], [1, 1, 2], lambda n: field[n])
    assert compiled.digest == identity
    np.testing.assert_array_equal(result.targets, expected)


def test_empty_touched_region_and_cached_none_do_not_read_exterior():
    compiled = implementation()([[0]], [[1.0]], (2, 2, 2), cache_bytes=5000)
    for _ in range(2):
        assert compiled.restrict([1, 1, 1], [2, 2, 2], lambda n: pytest.fail("unneeded read")) is None
    assert compiled.stats["hits"] == 1


@pytest.mark.parametrize("budget", [0, 1, 5000])
def test_cache_eviction_and_large_diagonal_are_bounded(budget):
    compiled = implementation()([[0, 7]], [[1.0, -1.0]], (2, 2, 2), cache_bytes=budget)
    one = compiled.restrict([0, 0, 0], [1, 1, 1], lambda n: np.ones(n.shape))
    diagonal = weakref.ref(one.diagonal)
    del one
    gc.collect()
    assert diagonal() is None, "cache retained a live dense tile diagonal"
    compiled.restrict([1, 1, 1], [2, 2, 2], lambda n: np.ones(n.shape))
    assert compiled.stats["bytes"] <= budget
    if budget:
        assert compiled.stats["evictions"] == (1 if budget == 5000 else 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nodes": [[-1]]},
        {"nodes": [[8]]},
        {"nodes": [[True]]},
        {"nodes": [[0.2]]},
        {"coefficients": [[np.nan]]},
        {"coefficients": [[1j]]},
        {"coefficients": [[1, 2]]},
        {"shape_zyx": (2, 2, 0)},
        {"shape_zyx": (2, 2, True)},
        {"weights": -1},
        {"targets": np.inf},
        {"groups": [-2]},
        {"groups": [True]},
        {"cache_bytes": True},
        {"cache_bytes": -1},
    ],
)
def test_invalid_stencil_inputs_are_rejected(kwargs):
    arguments = dict(nodes=[[0]], coefficients=[[1.0]], shape_zyx=(2, 2, 2))
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        implementation()(**arguments)


@pytest.mark.parametrize(
    "lo,hi",
    [
        ([-1, 0, 0], [1, 1, 1]),
        ([0.0, 0, 0], [1, 1, 1]),
        ([True, 0, 0], [1, 1, 1]),
        ([0, 0], [1, 1, 1]),
        ([0, 0, 0], [3, 1, 1]),
        ([0, 0, 0], [0, 1, 1]),
    ],
)
def test_invalid_restriction_bounds_are_rejected(lo, hi):
    compiled = implementation()([[0]], [[1.0]], (2, 2, 2))
    with pytest.raises(ValueError):
        compiled.restrict(lo, hi, lambda n: np.zeros(n.shape))


@pytest.mark.parametrize("returned", [np.array([np.nan]), np.zeros((1, 1)), np.array([1j])])
def test_invalid_exterior_values_fail_instead_of_using_stale_targets(returned):
    compiled = implementation()([[0, 7]], [[1.0, -1.0]], (2, 2, 2), cache_bytes=5000)
    with pytest.raises(ValueError):
        compiled.restrict([0, 0, 0], [1, 1, 1], lambda n: returned)


def test_unsupported_points_and_pair_shape_mismatch_fail():
    cls = implementation()
    with pytest.raises(ValueError):
        cls.pairs([[0, 0, 0]], [[0, 0, 0], [1, 1, 1]], [0] * 3, [1] * 3, [2] * 3)
    with pytest.raises(ValueError):
        cls.consensus([[9, 0, 0]], [0], [1.0], [0] * 3, [1] * 3, [2] * 3)


def stencil_archive_arrays():
    return {
        "nodes": np.array([[0, 2], [1, 2], [3, 3], [0, 1], [2, 3]], np.int64),
        "coefficients": np.array([[1, -1], [2, 0], [1, 0], [1, -1], [1, -1]], np.float64),
        "shape_zyx": np.array([1, 1, 4], np.int64),
        "weights": np.array([2, 3, 1, 4, 5], np.float64),
        "targets": np.array([1, 2, 3, 4, 5], np.float64),
        "groups": np.array([7, 7, 7, -1, -1], np.int64),
    }


@pytest.mark.parametrize("cache_bytes", [0, 10000])
def test_stencil_archive_roundtrip_preserves_digest_group_means_and_current_exterior(tmp_path, cache_bytes):
    arrays = stencil_archive_arrays()
    original = implementation()(**arrays, cache_bytes=10000)
    digest = original.digest
    for array in arrays.values():
        array[:] = 0  # Saving must use the constructor's immutable snapshots.
    path = tmp_path / "constraints.prepared"
    assert original.save(path) is None
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(stencil_archive_arrays())
        for key, expected in stencil_archive_arrays().items():
            np.testing.assert_array_equal(archive[key], expected)
            assert archive[key].dtype == expected.dtype
    restored = implementation().load(str(path), cache_bytes=cache_bytes)
    assert restored.shape == (1, 1, 4) and restored.digest == digest
    assert restored.max_cache_bytes == cache_bytes
    assert restored.stats == dict(bytes=0, hits=0, misses=0, evictions=0)
    for array in (restored._nodes, restored._coeff, restored._weights, restored._targets, restored._groups):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    for outside, targets, rhs in [
        ([30.0, 40.0], [31.0, 2.0, -37.0, 4.0], [203 / 3, -35.0]),
        ([40.0, 50.0], [41.0, 2.0, -47.0, 4.0], [253 / 3, -45.0]),
    ]:
        values = np.array([10.0, 20.0, *outside])
        for compiled in (original, restored):
            result = compiled.restrict([0, 0, 0], [1, 1, 2], lambda n: values[n])
            np.testing.assert_array_equal(result.matrix.toarray(), [[1, 0], [0, 2], [0, 0], [1, -1]])
            np.testing.assert_array_equal(result.weights, [2, 3, 1, 4])
            np.testing.assert_array_equal(result.targets, targets)
            np.testing.assert_array_equal(result.groups, [0, 0, 0, -1])
            np.testing.assert_array_equal(result.group_sum, [6])
            np.testing.assert_allclose(result.diagonal, [16 / 3, 10], rtol=0, atol=1e-14)
            operator = result.operator("cpu", np.float64)
            np.testing.assert_allclose(operator.rhs().numpy(), rhs, rtol=0, atol=3e-14)
            np.testing.assert_allclose(
                operator.matvec(torch.tensor([2.0, 3.0], dtype=torch.float64)).numpy(),
                [-22 / 3, 18],
                rtol=0,
                atol=1e-14,
            )
    assert restored.digest == digest
    assert restored.stats["hits"] == (1 if cache_bytes else 0)
    path.unlink()
    assert restored.restrict([0, 0, 0], [1, 1, 2], lambda n: values[n]) is not None


def test_stencil_archive_load_defaults_to_no_cache_and_supports_empty_rows(tmp_path):
    compiled = implementation()(np.empty((0, 2), np.int64), np.empty((0, 2)), (1, 1, 4))
    path = tmp_path / "empty.npz"
    compiled.save(path)
    restored = implementation().load(path)
    assert restored.digest == compiled.digest and restored.max_cache_bytes == 0
    assert restored.restrict([0, 0, 0], [1, 1, 2], lambda n: pytest.fail("unneeded read")) is None


@pytest.mark.parametrize("existing", ["file", "symlink", "dangling", "directory"])
def test_stencil_archive_save_never_overwrites_existing_path(tmp_path, existing):
    compiled = implementation()(**stencil_archive_arrays())
    path = tmp_path / "constraints.npz"
    target = tmp_path / "target"
    if existing == "directory":
        path.mkdir()
    elif existing in ("symlink", "dangling"):
        if existing == "symlink":
            target.write_bytes(b"keep target")
        path.symlink_to(target)
    else:
        path.write_bytes(b"keep original")
    with pytest.raises(FileExistsError):
        compiled.save(path)
    if existing == "directory":
        assert path.is_dir()
    elif existing == "dangling":
        assert path.is_symlink() and not target.exists()
    else:
        assert path.read_bytes() == (b"keep target" if existing == "symlink" else b"keep original")


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_stencil_archive_failed_save_removes_only_its_new_partial_file(tmp_path, monkeypatch, error):
    compiled = implementation()(**stencil_archive_arrays())
    path = tmp_path / "constraints.npz"

    def fail_write(stream, **arrays):
        stream.write(b"partial archive")
        raise error("injected write failure")

    monkeypatch.setattr(np, "savez_compressed", fail_write)
    with pytest.raises(error, match="injected write failure"):
        compiled.save(path)
    assert not path.exists()


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "npy"])
def test_stencil_archive_requires_exact_npz_members(tmp_path, change):
    arrays = stencil_archive_arrays()
    path = tmp_path / "constraints.npz"
    if change == "npy":
        with path.open("wb") as stream:
            np.save(stream, np.zeros(3))
    else:
        if change == "missing":
            del arrays["groups"]
        elif change == "extra":
            arrays["metadata"] = np.array([1])
        np.savez(path, **arrays)
        if change == "duplicate":
            raw = io.BytesIO()
            np.save(raw, arrays["nodes"], allow_pickle=False)
            with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("nodes.npy", raw.getvalue())
    with pytest.raises(ValueError, match="NPZ|member|keys|schema"):
        implementation().load(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("nodes", np.array([[0.0, 1.0]])),
        ("nodes", np.array([[0, 4]])),
        ("nodes", np.array([[0, 2**64 - 1]], np.uint64)),
        ("nodes", np.array([0, 1])),
        ("coefficients", np.array([[1, -1]], dtype="U2")),
        ("coefficients", np.ones((5, 2), np.complex128)),
        ("coefficients", np.ones((5, 2), bool)),
        ("coefficients", np.full((5, 2), np.nan)),
        ("coefficients", np.ones((5, 1))),
        ("shape_zyx", np.array([1, 1, 0])),
        ("shape_zyx", np.array([1, 1])),
        ("weights", np.array([1, 2, 3, 4, 5], dtype=object)),
        ("weights", np.array(["1"] * 5)),
        ("weights", np.array([1, 1, -1, 1, 1])),
        ("targets", np.full(5, np.inf)),
        ("targets", np.array(["1"] * 5)),
        ("targets", np.zeros(5, dtype=[("value", "f8")])),
        ("groups", np.array([0.0] * 5)),
        ("groups", np.array([0, 0, 0, -2, -1])),
        ("groups", np.array([2**64 - 1] * 5, np.uint64)),
        ("groups", np.array([0, 0])),
    ],
)
def test_stencil_archive_rejects_unsafe_dtypes_and_invalid_constructor_inputs(tmp_path, key, value):
    arrays = stencil_archive_arrays()
    arrays[key] = value
    path = tmp_path / "constraints.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        implementation().load(path)


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_stencil_archive_rejects_invalid_cache_budget(tmp_path, budget):
    path = tmp_path / "constraints.npz"
    np.savez(path, **stencil_archive_arrays())
    with pytest.raises(ValueError, match="budget"):
        implementation().load(path, cache_bytes=budget)
