"""The dense historical selector is an oracle only for small test inputs."""

import numpy as np
import pytest

import coherence_pipeline as pipeline


def legacy_pairs(s, fid):
    i, j = np.triu_indices(len(s), 1)
    eligible = np.flatnonzero(s[j] - s[i] >= min(0.3, s[-1]))
    if len(eligible) > 8:
        eligible = np.sort(np.random.default_rng(int(fid)).choice(eligible, 8, replace=False))
    return i[eligible], j[eligible]


def test_selector_matches_dense_order_and_rng():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        for size in range(1, 65):
            s = np.r_[0.0, np.cumsum(rng.choice([0.0, 0.01, 0.1, 0.3, 1.0], size - 1))]
            actual = pipeline._sample_arclength_pairs(s, seed)
            expected = legacy_pairs(s, seed)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9])
def test_candidate_population_boundaries(count, monkeypatch):
    s = np.r_[np.zeros(count), 0.3]
    if count <= 8:

        def forbidden(*args):
            pytest.fail("RNG must not run for at most eight candidates")

        monkeypatch.setattr(np.random, "default_rng", forbidden)
    np.testing.assert_array_equal(pipeline._sample_arclength_pairs(s, 43), legacy_pairs(s, 43))


@pytest.mark.parametrize(
    "s",
    [
        [0.0, 0.4, 0.7],
        [0.0, 0.0, 0.0],
        [0.0, 0.1, 0.2],
        [0.0, 0.4, np.nextafter(0.7, 0.0), 0.7, np.nextafter(0.7, 1.0)],
        [0.0, np.nextafter(0.3, 0.0), 0.3, np.nextafter(0.3, 1.0)],
    ],
)
def test_subtraction_thresholds(s):
    s = np.asarray(s)
    np.testing.assert_array_equal(pipeline._sample_arclength_pairs(s, 7), legacy_pairs(s, 7))


def test_hundred_thousand_samples_without_dense_enumeration(monkeypatch):
    def forbidden(*args):
        pytest.fail("quadratic enumeration")

    monkeypatch.setattr(np, "triu_indices", forbidden)
    s = np.arange(100_000, dtype=float)
    i, j = pipeline._sample_arclength_pairs(s, 9)
    ranks = np.sort(np.random.default_rng(9).choice(100_000 * 99_999 // 2, 8, replace=False))
    np.testing.assert_array_equal(i * (199_999 - i) // 2 + j - i - 1, ranks)
    assert len(i) == 8 and np.all(i < j)


def test_empty_and_overflow_population_before_allocation():
    i, j = pipeline._sample_arclength_pairs(np.empty(0), 0)
    assert i.dtype == j.dtype == np.int64 and not len(i)

    class TooLarge:
        def __len__(self):
            return 2**33

    with pytest.raises(OverflowError, match="int64"):
        pipeline._sample_arclength_pairs(TooLarge(), 0)


def test_compile_exact_integration_with_multirun_and_backtracking(monkeypatch):
    fibres = [
        dict(
            fibre_id=fid,
            points_xyz=np.array(
                [
                    [0.1, 1, 1],
                    [2.1, 1, 1],
                    [0.1, 1, 1],
                    [0.2, 2, 1],
                    [2.2, 2, 1],
                    [0.2, 2, 1],
                ]
            ),
            break_before=[0, 0, 0, 1, 0, 0],
        )
        for fid in (13, 2)
    ]
    args = (fibres, [0] * 3, [1] * 3, [4] * 3)
    actual = pipeline.compile_fibres(*args, voxel_mm=1, pitch_mm=0.05)
    monkeypatch.setattr(pipeline, "_sample_arclength_pairs", legacy_pairs)
    expected = pipeline.compile_fibres(*args, voxel_mm=1, pitch_mm=0.05)
    for a, b in zip(actual[:2], expected[:2]):
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(a.matrix, name), getattr(b.matrix, name))
        for name in ("weights", "targets", "groups"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert a.digest == b.digest and a.trace == b.trace
    for name in actual[2]:
        np.testing.assert_equal(actual[2][name], expected[2][name])
