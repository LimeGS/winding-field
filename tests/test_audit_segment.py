import tempfile
import unittest

import numpy as np

from audit_segment import (
    aggregate_metrics,
    load_field,
    main_component_id,
    parse_ppm_header,
    row_metrics,
    shuffle_winding,
)


class AuditSegmentTest(unittest.TestCase):
    def test_load_field_accepts_corrected_output_without_labels(self):
        with tempfile.NamedTemporaryFile(suffix=".npz") as output:
            shape = (2, 3, 4)
            np.savez(
                output.name,
                w=np.zeros(shape),
                r=np.ones(shape),
                zs=np.array([0, 8]),
            )
            winding, labels, radii, levels = load_field(output.name)
        self.assertEqual(winding.shape, shape)
        self.assertIsNone(labels)
        np.testing.assert_array_equal(radii, np.ones(shape))
        np.testing.assert_array_equal(levels, np.array([0, 8]))

    def test_main_component_id_is_optional(self):
        self.assertIsNone(main_component_id(None))
        labels = np.array([[[3, 3, -1], [4, 3, -1]]])
        self.assertEqual(main_component_id(labels, stride=1), 3)

    def test_shuffle_preserves_each_rays_finite_values(self):
        winding = np.array([[[0.0, 1.0, 2.0, np.nan], [5.0, 6.0, np.nan, np.nan]]])
        shuffled = shuffle_winding(winding, seed=7)
        for theta in range(winding.shape[1]):
            before = np.sort(winding[0, theta][np.isfinite(winding[0, theta])])
            after = np.sort(shuffled[0, theta][np.isfinite(shuffled[0, theta])])
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(np.isnan(winding), np.isnan(shuffled))

    def test_row_metrics_use_transition_denominator(self):
        theta = np.linspace(0, 2 * np.pi, 5)
        winding = np.array([0.0, 0.25, 0.5, 2.0, 1.0])
        metrics = row_metrics(winding, theta)
        self.assertEqual(metrics["transitions"], 4)
        self.assertEqual(metrics["jumps"], 2)
        self.assertAlmostEqual(metrics["jump_rate"], 0.5)
        self.assertAlmostEqual(metrics["endpoint_residual"], 0.0)

    def test_aggregate_metrics_reports_numerator_and_denominator(self):
        rows = [
            {"jumps": 2, "transitions": 4, "endpoint_residual": 1.0},
            {"jumps": 1, "transitions": 6, "endpoint_residual": 3.0},
        ]
        result = aggregate_metrics(rows)
        self.assertEqual((result["jumps"], result["transitions"]), (3, 10))
        self.assertAlmostEqual(result["jump_rate"], 0.3)
        self.assertAlmostEqual(result["median_endpoint_residual"], 2.0)

    def test_parse_ppm_header(self):
        header = b"width: 10\nheight: 20\nordered: true\n<>\n"
        self.assertEqual(parse_ppm_header(header), (10, 20, len(header)))


if __name__ == "__main__":
    unittest.main()
