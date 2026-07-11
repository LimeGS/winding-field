import unittest

import numpy as np

from validate_field import (
    main_component_coverage,
    main_component_id,
    merged_sheet_exposure,
)


class ValidateFieldTest(unittest.TestCase):
    def test_main_component_id_uses_non_negative_labels(self):
        labels = np.array(
            [
                [[1, 1, -1], [1, 2, -1]],
                [[1, 1, -1], [2, 2, -1]],
            ]
        )
        self.assertEqual(main_component_id(labels, stride=1), 1)

    def test_main_component_coverage_is_per_z_level(self):
        labels = np.array(
            [
                [[1, 1, -1], [1, 2, -1]],
                [[1, 1, -1], [2, 2, -1]],
            ]
        )
        np.testing.assert_allclose(
            main_component_coverage(labels, main_component=1),
            np.array([0.75, 0.5]),
        )

    def test_empty_component_sample_returns_none(self):
        labels = np.full((2, 2, 2), -1, dtype=np.int64)
        self.assertIsNone(main_component_id(labels, stride=1))
        np.testing.assert_allclose(main_component_coverage(labels), np.array([0.0, 0.0]))

    def test_merged_sheet_exposure_counts_double_spacing_gaps(self):
        radii = np.array([[[1.0, 2.0, 3.0, 5.1, 6.1, 7.1]]])
        anomaly, total, pct = merged_sheet_exposure(radii, z_stride=1, theta_stride=1)
        self.assertEqual((anomaly, total), (1, 5))
        self.assertAlmostEqual(pct, 20.0)


if __name__ == "__main__":
    unittest.main()
