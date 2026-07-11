import unittest

import numpy as np

from build_field import match


class MatchTest(unittest.TestCase):
    def test_empty_inputs_return_no_matches(self):
        self.assertEqual(match(np.array([]), np.array([1.0]), 0.2), [])
        self.assertEqual(match(np.array([1.0]), np.array([]), 0.2), [])

    def test_matches_nearest_sorted_wraps_within_tolerance(self):
        pairs = match(
            np.array([1.0, 2.1, 4.0]),
            np.array([1.05, 3.95, 9.0]),
            0.2,
        )
        self.assertEqual(sorted(pairs), [(0, 0), (2, 1)])

    def test_one_to_one_match_keeps_closest_source(self):
        pairs = match(np.array([10.0, 10.06]), np.array([10.05]), 0.2)
        self.assertEqual(pairs, [(1, 0)])

    def test_tolerance_is_strict(self):
        self.assertEqual(match(np.array([1.0]), np.array([1.25]), 0.2), [])


if __name__ == "__main__":
    unittest.main()
