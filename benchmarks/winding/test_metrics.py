import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def implementation():
    path = Path(__file__).with_name("metrics.py")
    assert path.exists(), "benchmark implementation is missing"
    spec = importlib.util.spec_from_file_location("fibre_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkTests(unittest.TestCase):
    def test_pair_uses_arclength_and_cannot_bridge_outside_run(self):
        b = implementation()
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        pair = b.choose_pair(points, np.ones(4, bool), 2.5, 3.5, 7)
        self.assertEqual(pair, (0, 3, 3.0))
        self.assertIsNone(b.choose_pair(points, np.array([1, 0, 1, 1], bool), 2.5, 3.5, 7))

    def test_pair_upper_bin_boundary_is_excluded(self):
        b = implementation()
        p = np.array([[0, 0, 0], [1.5, 0, 0]])
        self.assertIsNone(b.choose_pair(p, np.ones(2, bool), 0.5, 1.5, 4))
        self.assertEqual(b.choose_pair(p, np.ones(2, bool), 1.5, 2.5, 4), (0, 1, 1.5))

    def test_sampler_preserves_xyz_non_cubic_spacing_and_missing(self):
        b = implementation()
        z, y, x = np.indices((5, 6, 7))
        f = (2 * x + 3 * y + 5 * z).astype(np.float32)
        m = dict(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4], shape_zyx=list(f.shape))
        p = np.array([[13, 27.5, 40], [9, 25, 40], [10, 20, 30]])
        values = b.sample_checked(f, m, p, margin=1)
        self.assertAlmostEqual(values[0], 23)
        self.assertTrue(np.isnan(values[1:]).all())
        support = np.ones(f.shape, bool)
        support[2, 2, 1] = False
        self.assertTrue(np.isnan(b.sample_checked(f, m, p[:1], support, margin=1)[0]))

    def test_summary_keeps_missing_in_coverage_and_uses_strict_thresholds(self):
        b = implementation()
        rows = [dict(bin=0, fiber_id=i, error=e, seam="same") for i, e in enumerate([0.0, 0.5, 1.0, 2.0, None])]
        r = b.summarize(rows)[0]
        self.assertEqual((r["selected"], r["covered"], r["fibers"]), (5, 4, 4))
        self.assertEqual(r["coverage_pct"], 80)
        self.assertEqual(r["median"], 0.75)
        self.assertEqual((r["lt0_5_pct"], r["lt1_pct"], r["lt2_pct"]), (25, 50, 75))
        self.assertIsNone(b.summarize([])[0]["median"])

    def test_lineage_excludes_fragments_but_not_unrelated_collection(self):
        b = implementation()
        a = {"sourceCollectionUUID": "A", "sourceFiberIDs": [1, 2]}
        c = {"sourceCollectionUUID": "A", "sourceFiberIDs": [2, 3]}
        d = {"sourceCollectionUUID": "B", "sourceFiberIDs": [2]}
        self.assertTrue(b.lineage(a) & b.lineage(c))
        self.assertFalse(b.lineage(a) & b.lineage(d))

    def test_selection_is_order_independent_one_pair_per_fibre_bin(self):
        b = implementation()
        fibres = [(i, np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float), {}) for i in range(1, 9)]
        scopes = {"a": ([0, 0, 0], [3, 1, 1])}
        x, _ = b.select_pairs(iter(fibres), scopes, {2}, set(), limit=3, native_mm=1)
        y, _ = b.select_pairs(iter(reversed(fibres)), scopes, {2}, set(), limit=3, native_mm=1)
        self.assertEqual(x, y)
        rows = x["a"]
        self.assertFalse(any(r["fiber_id"] == 2 for r in rows))
        self.assertEqual(len({(r["fiber_id"], r["bin"]) for r in rows}), len(rows))
        self.assertLessEqual(sum(r["bin"] == 0 for r in rows), 3)

    def test_field_hash_mismatch_is_rejected(self):
        b = implementation()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "f.npy"
            np.save(p, np.zeros((3, 4, 5), np.float32))
            meta = dict(
                field_sha256="0" * 64, shape_zyx=[3, 4, 5], step=1, frame="test/world", solver={"converged": True}
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                b.open_field(p, meta)

    def test_unavailable_fine_pair_keeps_geometric_core_stratum(self):
        b = implementation()
        field = np.zeros((5, 6, 7), np.float32)
        meta = dict(origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1], step=1)
        rows = [dict(bin=0, fiber_id=1, a_native_xyz=[0, 8, 8], b_native_xyz=[8, 8, 8])]
        result = b.evaluate(
            field,
            meta,
            rows,
            ownership=np.zeros(field.shape, np.int8),
            transform=dict(scale_xyz=[0.25] * 3, offset_xyz=[0, 0, 0]),
        )
        self.assertIsNone(result[0]["error"])
        self.assertEqual(result[0]["seam"], "same")
        table = b.summarize([r for r in result if r["seam"] == "same"])
        self.assertEqual((table[0]["selected"], table[0]["covered"]), (1, 0))
        self.assertEqual(table[0]["coverage_pct"], 0)

    def test_uncapped_sampling_retains_every_eligible_fibre(self):
        b = implementation()
        fibres = [(i, np.array([[0, 0, 0], [1, 0, 0]], float), {}) for i in range(8)]
        rows, _ = b.select_pairs(iter(fibres), {"a": ([0, 0, 0], [2, 1, 1])}, set(), set(), limit=None, native_mm=1)
        self.assertEqual({r["fiber_id"] for r in rows["a"]}, set(range(8)))

    def test_independent_interpolator_has_known_answer_and_missing_support(self):
        path = Path(__file__).with_name("interpolation.py")
        self.assertTrue(path.exists(), "independent verifier is missing")
        spec = importlib.util.spec_from_file_location("verify_benchmark", path)
        v = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v)
        z, y, x = np.indices((5, 6, 7))
        field = (2 * x + 3 * y + 5 * z).astype(float)
        meta = dict(origin_xyz=[10, 20, 30], spacing_xyz=[2, 3, 4])
        p = np.array([[13, 27.5, 40], [9, 25, 40]])
        values = v.interpolate(field, meta, p)
        self.assertAlmostEqual(values[0], 23)
        self.assertTrue(np.isnan(values[1]))
        support = np.ones(field.shape, bool)
        support[2, 2, 1] = False
        self.assertTrue(np.isnan(v.interpolate(field, meta, p, support)[0]))


if __name__ == "__main__":
    unittest.main()
