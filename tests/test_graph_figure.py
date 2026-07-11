import unittest

import numpy as np

from build_field import match
from figures.make_field_figure import match_vec, per_level_coverage


class GraphFigureTest(unittest.TestCase):
    def test_vector_match_matches_reference(self):
        a = np.array([10.0, 20.0, 30.0, 30.2])
        b = np.array([10.1, 19.8, 30.1])
        ii, jj = match_vec(a, b, tol=0.5)
        self.assertEqual(
            sorted(zip(ii.tolist(), jj.tolist())),
            sorted(match(a, b, tol=0.5)),
        )

    def test_per_level_coverage(self):
        labels = np.array([0, 0, 1, 0, 1, 1])
        counts = np.array([[2, 1], [1, 2]])
        np.testing.assert_allclose(per_level_coverage(labels, counts), [2 / 3, 1 / 3])


if __name__ == "__main__":
    unittest.main()
