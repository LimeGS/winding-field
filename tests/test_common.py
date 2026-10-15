"""Fibre decoding, read-only IO, calibration and reporting on synthetic data."""

import sqlite3
import struct

import numpy as np
import pytest

import common


def test_fibre_reader_requires_explicit_path_even_when_data_environment_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("WREV_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="explicit"):
        common.load_fibers(None, dict(x0=0, x1=1, y0=0, y1=1, z0=0, z1=1))


def test_fibre_blocks_are_ordered_then_clipped_with_exclusive_upper_bound(paris_inputs, monkeypatch):
    box = dict(x0=64, x1=256, y0=0, y1=100, z0=0, z1=100)
    rows = common.load_fibers(paris_inputs.afv, box)
    assert [fid for fid, _ in rows] == [1]
    np.testing.assert_array_equal(rows[0][1], [[64, 64, 64], [128, 64, 64], [192, 64, 64]])
    assert common.load_fibers(paris_inputs.afv, box, min_pts=4) == []
    full = common.load_fibers(paris_inputs.afv, box, clip=False)
    np.testing.assert_array_equal(full[0][1], paris_inputs.points)
    with pytest.raises((TypeError, ValueError)):
        common.load_fibers(box)


def test_missing_fibre_database_is_not_created(tmp_path):
    path = tmp_path / "missing.afv"
    with pytest.raises(sqlite3.OperationalError):
        common.load_fibers(path, dict(x0=0, x1=1, y0=0, y1=1, z0=0, z1=1))
    assert not path.exists()


def test_malformed_fibre_blob_closes_the_connection(paris_inputs, monkeypatch):
    with sqlite3.connect(paris_inputs.afv) as db:
        db.execute("UPDATE blocks SET points=? WHERE fiber_id=1", (b"x",))
    db.close()
    connect = sqlite3.connect
    opened = []

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(common.sqlite3, "connect", track_connection)
    with pytest.raises(struct.error):
        common.load_fibers(paris_inputs.afv, dict(x0=0, x1=300, y0=0, y1=100, z0=0, z1=100))
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_streamed_records_preserve_lineage_and_read_only_source(paris_inputs):
    before = paris_inputs.afv.read_bytes()
    records = list(common.iter_fibre_records(paris_inputs.afv))
    assert [r["fibre_id"] for r in records] == [1, 2]
    assert records[0]["annotation"] == {"sourceCollectionUUID": "A", "sourceFiberIDs": [1]}
    np.testing.assert_array_equal(records[0]["points_xyz"], paris_inputs.points)
    assert paris_inputs.afv.read_bytes() == before


def test_pair_sampling_uses_arclength_instead_of_endpoint_chord():
    points = np.array([[0, 0, 0], [3, 0, 0], [0, 0, 0]], float)
    lengths = common.arclength(points, 1)
    np.testing.assert_array_equal(lengths, [0, 3, 6])
    sampled = common.sample_pairs(points, lengths, np.random.default_rng(9), per_fiber=100, span_lo=5, span_hi=6)
    assert sampled
    for span, a, b in sampled:
        assert span == 6
        np.testing.assert_array_equal(a, b)
    assert common.sample_pairs(points, lengths / 100, np.random.default_rng(9)) == []


def test_trilinear_preserves_a_non_cubic_linear_field():
    z, y, x = np.indices((3, 4, 5))
    assert common.trilinear(100 * z + 10 * y + x, 0.25, 1.5, 2.75) == pytest.approx(42.75)


def test_gap_calibration_counts_low_runs_and_handles_no_evidence():
    detection = np.full((12, 12, 12), 255, np.uint8)
    detection[0:2] = detection[4:6] = detection[10:12] = 0
    assert common.calib_gap(detection, n=300, seed=3) == 2
    assert common.calib_gap(np.zeros((3, 4, 5), np.uint8)) == 10
    assert common.calib_gap(np.full((3, 4, 5), 255, np.uint8)) == 10


@pytest.mark.parametrize("extra_p99", [False, True])
def test_report_preserves_bin_boundaries_and_empty_denominators(capsys, extra_p99):
    common.report([(0.5, 0), (1.0, 2), (1.5, 4)], bins=[(0.5, 1.5), (1.5, 2.5), (3, 4)], extra_p99=extra_p99)
    lines = capsys.readouterr().out.splitlines()
    assert ("p99" in lines[0]) is extra_p99
    assert lines[1].split()[1:3] == ["2", "1.000"]
    assert lines[2].split()[1:3] == ["1", "4.000"]
    assert lines[3].endswith("-")
