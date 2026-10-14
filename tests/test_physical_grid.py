import numpy as np


def test_grid_samples_fractional_non_cubic_positions():
    from physical_grid import grid_shape, sample_volume

    z, y, x = np.indices((9, 13, 17))
    volume = 100 * z + 10 * y + x
    origin = [1.25, 2.5, 3.75]
    shape = grid_shape(origin, [5.25, 8.5, 7.75], [2, 3, 2])
    assert shape == (2, 2, 2)
    sampled = sample_volume(volume, origin, [2, 3, 2], shape, [1, 1, 1])
    assert sampled[0, 0, 0] == 401.25
    assert sampled[1, 1, 1] == 633.25


def test_overlap_correction_does_not_double_offset():
    from physical_grid import offset_correction

    assert offset_correction(np.ones((2, 2)) * 10, np.ones((2, 2)) * 13) == -3


def test_grid_shape_can_be_written_as_json_receipt():
    import json

    from physical_grid import grid_shape

    shape = grid_shape([0, 0, 0], [10, 20, 30], [2, 2, 2])
    assert json.loads(json.dumps({"shape_zyx": shape})) == {"shape_zyx": [15, 10, 5]}
