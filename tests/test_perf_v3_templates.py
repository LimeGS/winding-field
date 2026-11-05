import numpy as np
import pytest

from constraint_templates import StencilConstraints
from field_store import BlockStore


def equal(a, b):
    if a is None or b is None:
        assert a is b
        return
    for name in ("targets", "weights", "groups", "diagonal"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(a.matrix, name), getattr(b.matrix, name))


@pytest.mark.parametrize("budget", [0, 1, 100000])
def test_index_original_slots_group_zero_exterior_and_half_open_boxes(budget):
    nodes = [[0, 0, 23], [59, 59, 59], [1, 2, 3], [32, 33, 34], [59, 0, 0]]
    coeff = [[1, -1, 0], [0.1, 0.2, 0.3], [0, 0, 0], [1, -1, 1], [1, 0, 0]]
    kwargs = dict(groups=[0, 0, -1, -1, -1], targets=[3, 4, 5, 6, 7], weights=[0, 2, 3, 4, 5])
    scan = StencilConstraints(nodes, coeff, (3, 4, 5), **kwargs)
    indexed = StencilConstraints(nodes, coeff, (3, 4, 5), spatial_index_bytes=budget, cache_bytes=100000, **kwargs)
    assert scan.digest == indexed.digest
    assert indexed.spatial_index_nbytes <= budget
    assert (indexed._postings is not None) == (budget == 100000)
    rng = np.random.default_rng(81)
    boxes = [([0, 0, 0], [1, 1, 1]), ([0, 0, 1], [1, 1, 2]), ([2, 3, 4], [3, 4, 5])]
    for _ in range(30):
        lo = rng.integers([0, 0, 0], [3, 4, 5])
        boxes.append((lo, rng.integers(lo + 1, [4, 5, 6])))
    for _ in range(2):
        values = rng.normal(size=60)
        for lo, hi in boxes:
            equal(scan.restrict(lo, hi, lambda n: values[n]), indexed.restrict(lo, hi, lambda n: values[n]))
    result = indexed.restrict([0, 0, 0], [1, 1, 1], lambda n: np.ones(n.shape))
    assert result.matrix.shape == (2, 1) and result.matrix.nnz == 0


@pytest.mark.parametrize("policy,expected", [("stable", (4, 5, 0)), ("lru", (0, 9, 7))])
def test_three_sweeps_two_entry_capacity(policy, expected):
    probe = StencilConstraints([[0], [1], [2]], [[1], [1], [1]], (1, 1, 3), cache_bytes=100000)
    probe.restrict([0, 0, 0], [1, 1, 1], lambda n: n)
    capacity = probe.stats["bytes"] * 2
    rows = StencilConstraints(
        [[0], [1], [2]], [[1], [1], [1]], (1, 1, 3), cache_bytes=capacity, template_cache_policy=policy
    )
    for _ in range(3):
        for x in range(3):
            rows.restrict([0, 0, x], [1, 1, x + 1], lambda n: n)
            assert rows.stats["bytes"] <= capacity
    assert tuple(rows.stats[k] for k in ("hits", "misses", "evictions")) == expected


def test_gather_plan_reused_but_generation_is_live(tmp_path, monkeypatch):
    rows = StencilConstraints([[0, 7, 7]], [[1, -0.25, -0.75]], (2, 2, 2), cache_bytes=100000)
    with BlockStore(tmp_path / "field", (2, 2, 2), 1, {}, cache_bytes=10000) as store:
        store.commit((1, 1, 1), np.array([[[2.0]]]), {})
        first = rows.restrict([0, 0, 0], [1, 1, 1], store.nodes)
        np.testing.assert_array_equal(first.targets, [2])
        template = next(iter(rows._entries.values()))[0]
        assert template.gather_plan is not None
        assert rows.stats["bytes"] >= template.gather_plan.nbytes
        monkeypatch.setattr(store, "prepare_nodes", lambda n: pytest.fail("rebuilt gather plan"))
        store.commit((1, 1, 1), np.array([[[5.0]]]), {})
        np.testing.assert_array_equal(rows.restrict([0, 0, 0], [1, 1, 1], store.nodes).targets, [5])
        np.testing.assert_array_equal(rows.restrict([0, 0, 0], [1, 1, 1], lambda n: n * 2).targets, [14])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spatial_index_bytes": True},
        {"spatial_index_bytes": -1},
        {"spatial_index_bytes": 1.2},
        {"template_cache_policy": "bad"},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        StencilConstraints([[0]], [[1]], (1, 1, 1), **kwargs)


def test_options_forwarded_without_archive_identity_changes(tmp_path):
    options = dict(spatial_index_bytes=100000, template_cache_policy="stable", cache_bytes=10000)
    pairs = StencilConstraints.pairs([[0, 0, 0]], [[1, 0, 0]], [0] * 3, [1] * 3, [2] * 3, **options)
    consensus = StencilConstraints.consensus([[0, 0, 0]], [0], [1], [0] * 3, [1] * 3, [2] * 3, **options)
    for rows in (pairs, consensus):
        path = tmp_path / rows.digest
        rows.save(path)
        loaded = StencilConstraints.load(path, **options)
        assert loaded.digest == rows.digest
        assert loaded.template_cache_policy == "stable" and loaded.spatial_index_nbytes > 0


def test_index_prunes_far_rows_and_charges_backing_storage(monkeypatch):
    rows = StencilConstraints(
        np.arange(1000).reshape(-1, 1), np.ones((1000, 1)), (10, 10, 10), spatial_index_bytes=100000
    )
    assert rows.spatial_index_nbytes >= sum(a.nbytes for a in rows._postings) + 4096
    for array in rows._postings:
        with pytest.raises(ValueError):
            array.setflags(write=True)
    original = np.unravel_index
    sizes = []

    def counted(indices, *args, **kwargs):
        sizes.append(np.asarray(indices).size)
        return original(indices, *args, **kwargs)

    monkeypatch.setattr(np, "unravel_index", counted)
    result = rows.restrict([5, 5, 5], [5 + 1, 5 + 1, 5 + 1], lambda n: n)
    np.testing.assert_array_equal(result.matrix.toarray(), [[1]])
    assert max(sizes) == 1


def test_empty_index_and_stable_admission_after_oversized_entry():
    empty = StencilConstraints(np.empty((0, 2), int), np.empty((0, 2)), (1, 1, 3), spatial_index_bytes=4096)
    assert empty.restrict([0, 0, 0], [1, 1, 1], lambda n: n) is None
    rows = StencilConstraints([[0]], [[1]], (1, 1, 3), cache_bytes=4096, template_cache_policy="stable")
    assert rows.restrict([0, 0, 0], [1, 1, 1], lambda n: n) is not None
    assert rows.stats["bytes"] == 0
    for _ in range(2):
        assert rows.restrict([0, 0, 1], [1, 1, 2], lambda n: n) is None
    assert rows.stats == dict(bytes=4096, hits=1, misses=2, evictions=0)
