import numpy as np
import pytest


def test_radial_orientation_uses_physical_xyz_every_axis():
    from paris_data import orient_radial

    normals = np.zeros((2, 3, 4, 3), np.float32)
    normals[..., 0] = -1
    got = orient_radial(normals, [10, 20, 30], [2, 3, 4], np.array([[0, 0, 0], [0, 0, 100]]))
    np.testing.assert_allclose(got[..., 0], 1)
    np.testing.assert_allclose(got[..., 1:], 0)


def test_fibre_runs_do_not_join_outside_excursions():
    from paris_data import contiguous_inside

    points = np.array([[1, 1, 1], [2, 1, 1], [99, 1, 1], [3, 1, 1], [4, 1, 1]], float)
    runs = contiguous_inside(points, [0, 0, 0], [10, 10, 10])
    assert [p[:, 0].tolist() for p in runs] == [[1, 2], [3, 4]]


def test_invalid_metadata_is_rejected_before_field_write(tmp_path):
    import pytest

    from paris_data import save_field

    path = tmp_path / "field.npy"
    with pytest.raises(TypeError):
        save_field(path, np.zeros((2, 2, 2)), {"unsupported": {1, 2}}, {"converged": True})
    assert not path.exists()


@pytest.mark.parametrize("with_fibres", [False, True])
def test_load_box_samples_physical_frame_and_binds_observation_hashes(paris_inputs, with_fibres):
    from conftest import FRAME

    from paris_data import load_paris_box
    from patch_check import sha256

    target, weights, pairs, meta = load_paris_box(
        [16, 16, 16],
        [80, 64, 48],
        spacing=16,
        dataset=paris_inputs.dataset,
        afv_path=paris_inputs.afv if with_fibres else None,
        pair_limit=1,
    )
    assert target.shape == (2, 3, 4, 3) and weights.shape == (2, 3, 4)
    np.testing.assert_allclose(target[..., 0], np.broadcast_to([0.18, 0.26, 0.34, 0.42], weights.shape))
    np.testing.assert_allclose(target[..., 1:], 0)
    np.testing.assert_array_equal(weights, np.ones(weights.shape))
    assert meta["frame"] == FRAME and meta["spacing_xyz"] == [16, 16, 16]
    assert meta["umbi_sha256"] == sha256(paris_inputs.root / "umbi.json")
    assert len(meta["source_receipts"]) == 4
    assert all(len(r["crop_sha256"]) == 64 and "url" in r for r in meta["source_receipts"])
    assert pairs.shape == ((1, 2) if with_fibres else (0, 2))
    assert meta["afv_sha256"] == (sha256(paris_inputs.afv) if with_fibres else None)
    assert meta["constraint_fibre_ids"] == ([1] if with_fibres else [])
    if with_fibres:
        assert pairs.min() >= 0 and pairs.max() < weights.size and pairs[0, 0] != pairs[0, 1]


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "out_of_bounds"])
def test_incomplete_source_observations_fail_instead_of_becoming_zero(paris_inputs, problem):
    from conftest import LB

    from paris_data import load_paris_box

    key = f"{LB}_grad_mag.ome.zarr/4"
    if problem == "missing":
        del paris_inputs.volumes[key]
    elif problem == "nonfinite":
        paris_inputs.volumes[key].fill(np.nan)
    hi = [800, 64, 48] if problem == "out_of_bounds" else [80, 64, 48]
    with pytest.raises(KeyError if problem == "missing" else ValueError):
        load_paris_box([16, 16, 16], hi, spacing=16, dataset=paris_inputs.dataset)


def test_saved_field_is_float32_hash_bound_and_cannot_be_overwritten(tmp_path):
    import json

    from paris_data import save_field
    from patch_check import sha256

    path = tmp_path / "field.npy"
    field = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    info = {"converged": True, "iterations": 4}
    meta = {"source": "synthetic"}
    receipt = save_field(path, field, meta, info, step=2)
    np.testing.assert_array_equal(np.load(path), field)
    assert np.load(path).dtype == np.float32
    assert receipt == json.loads((tmp_path / "field.npy.json").read_text())
    assert receipt["field_sha256"] == sha256(path) and receipt["step"] == 2
    assert meta == {"source": "synthetic"}
    with pytest.raises(FileExistsError):
        save_field(path, field + 1, meta, info)
    np.testing.assert_array_equal(np.load(path), field)


@pytest.mark.parametrize(
    "converged,step,value", [(False, 1, 0), (True, 0, 0), (True, float("nan"), 0), (True, 1, float("inf"))]
)
def test_invalid_field_publication_leaves_no_artifacts(tmp_path, converged, step, value):
    from paris_data import save_field

    path = tmp_path / "field.npy"
    with pytest.raises(ValueError, match="cannot publish"):
        save_field(path, np.full((2, 3, 4), value), {}, {"converged": converged}, step=step)
    assert list(tmp_path.iterdir()) == []


def test_configured_split_is_respected_by_shared_local_observation_loader(paris_inputs):
    import json

    from dataset_config import load_dataset
    from evaluation_split import build_fibre_split
    from paris_data import load_paris_box
    from patch_check import sha256

    split = build_fibre_split(
        [dict(fibre_id=i, length_mm=1, source_lineage=[["A", str(i)]]) for i in (1, 2)], {2}, reserved_fraction=0.5
    )
    split["afv_sha256"] = sha256(paris_inputs.afv)
    path = paris_inputs.root / "split.json"
    path.write_text(json.dumps(split))
    data = paris_inputs.raw_config
    data["artifacts"] = dict(split=dict(path=path.name, sha256=sha256(path)))
    paris_inputs.config.write_text(json.dumps(data))
    _, _, pairs, meta = load_paris_box(
        [16, 16, 16],
        [80, 64, 48],
        spacing=16,
        afv_path=paris_inputs.afv,
        dataset=load_dataset(paris_inputs.config),
        pair_limit=1,
    )
    assert len(pairs) == 1 and meta["constraint_fibre_ids"] == [2]
    assert meta["fibre_split_sha256"] == sha256(path)
