import numpy as np
import pytest
import torch
from scipy import ndimage

import field_observations as observations


@pytest.mark.parametrize("length", [9, 17, 49])
def test_direct_correlation_matches_scipy(length):
    rng = np.random.default_rng(length)
    value = rng.normal(size=(11, 1, length + 15)).astype(np.float32)
    weights = rng.normal(size=length)
    weights /= np.abs(weights).sum()
    actual = observations._sum_correlation(torch.from_numpy(value), weights.tolist(), 16)
    expected = ndimage.correlate1d(value, weights, axis=-1)[..., length // 2 : -(length // 2)]
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-7, rtol=3e-6)


def test_analytic_eigh_matches_numpy_scales_axes_and_degenerate_tensors():
    rng = np.random.default_rng(17)
    rotations, _ = np.linalg.qr(rng.normal(size=(384, 3, 3)))
    matrices = (rotations * [0.03, 0.2, 1.3]) @ rotations.swapaxes(-1, -2)
    matrices *= np.geomspace(1e-20, 1e20, len(rotations))[:, None, None]
    matrices = np.concatenate(
        [matrices, [np.diag(v) for v in ([3, 1, 0], [0, 3, 1], [1, 0, 3], [0, 0, 0], [1, 1, 1])]]
    ).astype(np.float32)
    expected_values, expected_vectors = np.linalg.eigh(matrices)
    packed = matrices[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]
    values, normals = observations._eigh(torch.from_numpy(packed))
    scale = np.maximum(expected_values[:, -1], 1e-30)
    np.testing.assert_allclose(values.numpy() / scale[:, None], expected_values / scale[:, None], atol=2e-6, rtol=2e-5)
    alignment = np.abs((normals.numpy()[:-2] * expected_vectors[:-2, :, -1]).sum(-1))
    assert alignment.min() > 0.99999
    assert not normals[-2:].any()


def test_eigh_processes_the_partial_final_batch():
    packed = torch.tensor([3.0, 0, 0, 1, 0, 0]).repeat(262145, 1)
    values, normals = observations._eigh(packed)
    torch.testing.assert_close(values, torch.tensor([0.0, 1, 3]).expand_as(values), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(normals.abs(), torch.tensor([1.0, 0, 0]).expand_as(normals))


@pytest.mark.parametrize(
    "sigma,order",
    [(1.0, (1, 0, 0)), (1.0, (0, 1, 0)), (1.0, (0, 0, 1)), (2.0, (0, 0, 0)), (6.0, (0, 0, 0))],
)
def test_separable_gaussian_matches_scipy_reflected_edges(sigma, order):
    material = np.random.default_rng(4).random((7, 8, 9), dtype=np.float32)
    actual = observations._gaussian(torch.from_numpy(material), sigma, order)
    expected = ndimage.gaussian_filter(material, sigma, order=order)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-7, rtol=2e-6)


@pytest.mark.parametrize("value", [0, 127, 255])
def test_constant_input_has_no_support_or_invented_normal(value):
    volume = torch.full((13, 15, 17), value, dtype=torch.uint8)
    density, normal, supported = observations.density_normals(volume, halo=0)
    assert density.shape == supported.shape == volume.shape
    assert normal.shape == (*volume.shape, 3)
    assert density.dtype == normal.dtype == torch.float32
    assert supported.dtype == torch.bool
    assert all(t.device == torch.device("cpu") for t in (density, normal, supported))
    assert not density.any() and not normal.any() and not supported.any()
    assert torch.all(volume == value)


@pytest.mark.parametrize("axis_xyz", [0, 1, 2])
def test_zyx_input_produces_xyz_surface_directors(axis_xyz):
    coordinates = np.indices((17, 19, 21))
    volume = np.where(coordinates[2 - axis_xyz] % 8 < 3, 255, 0).astype(np.uint8)
    density, normal, supported = observations.density_normals(volume, halo=np.int64(2))
    assert density.shape == supported.shape == (13, 15, 17)
    assert normal.shape == (13, 15, 17, 3)
    valid = normal.norm(dim=-1) > 0.5
    assert valid.sum() > 100
    torch.testing.assert_close(normal[..., axis_xyz][valid].abs(), torch.ones_like(density[valid]), atol=1e-5, rtol=0)
    assert torch.all(normal[..., (axis_xyz + 1) % 3][valid].abs() < 1e-5)
    assert torch.all(normal[..., (axis_xyz + 2) % 3][valid].abs() < 1e-5)


@pytest.mark.parametrize("normal_xyz,core_x", [((1.0, 0, 0), 32), ((0.8, 0, 0.6), 40)])
def test_aligned_and_oblique_sheets_integrate_in_input_voxels(normal_xyz, core_x):
    core = (5, 7, core_x)
    z, y, x = np.indices(tuple(n + 80 for n in core), dtype=np.float32)
    phase = normal_xyz[0] * x + normal_xyz[1] * y + normal_xyz[2] * z + 0.37
    volume = np.where(phase % 16 < 4, 255, 0).astype(np.uint8)
    density, normal, supported = observations.density_normals(volume)
    assert density.shape == core
    assert supported.all()
    assert density.mean(dim=(0, 1)).sum().item() * normal_xyz[0] == pytest.approx(2.0, rel=0.02)
    valid = normal.norm(dim=-1) > 0.5
    assert (normal[valid] * torch.tensor(normal_xyz)).sum(-1).abs().mean() > 0.995


def test_full_halo_matches_inline_scipy_reference_at_support_boundary():
    core = (13, 5, 7)
    z, y, x = np.indices(tuple(n + 80 for n in core), dtype=np.float32)
    phase = 0.8 * x + 0.6 * z + 0.003 * (y - 42) ** 2 + 0.37
    volume = np.where((phase % 13 < 3) & (z < 40), 128, 127).astype(np.uint8)
    density, normal, supported = observations.density_normals(torch.from_numpy(volume))

    material = (volume > 127).astype(np.float32)
    gradients = [ndimage.gaussian_filter(material, 1.0, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    matrix = np.empty((*material.shape, 3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            matrix[..., i, j] = matrix[..., j, i] = ndimage.gaussian_filter(gradients[i] * gradients[j], 2.0)
    values, vectors = np.linalg.eigh(matrix)
    coherent = (ndimage.gaussian_filter(material, 2.0) > 0.02) & (
        (values[..., 2] - values[..., 1]) / np.maximum(values[..., 2], 1e-12) > 0.3
    )
    coordinates = np.indices(material.shape, dtype=np.float32)
    director = np.moveaxis(vectors[..., -1], -1, 0)
    backward = ndimage.map_coordinates(material, coordinates - director, order=1, mode="nearest")
    forward = ndimage.map_coordinates(material, coordinates + director, order=1, mode="nearest")
    crossings = (np.abs(material - backward) + np.abs(material - forward)) * 0.25 * coherent
    expected_support = ndimage.gaussian_filter(coherent.astype(np.float32), 6.0) > 0.2
    expected_density = np.where(expected_support, ndimage.gaussian_filter(crossings, 6.0), 0)
    sl = tuple(slice(40, 40 + n) for n in core)
    np.testing.assert_allclose(density.numpy(), expected_density[sl], atol=2e-6, rtol=2e-4)
    np.testing.assert_array_equal(supported.numpy(), expected_support[sl])
    assert supported.any() and not supported.all()
    valid = coherent[sl] & expected_support[sl]
    assert valid.any()
    alignment = np.abs((normal.numpy()[valid] * vectors[sl][..., ::-1, -1][valid]).sum(-1))
    assert alignment.min() > 0.9999
    assert not normal.numpy()[~valid].any()


def test_halo_only_trims_and_autocast_cannot_change_fp32_filters():
    volume = np.where(np.indices((15, 17, 19))[2] % 8 < 3, 255, 0).astype(np.uint8)
    expected = observations.density_normals(volume, halo=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = observations.density_normals(volume, halo=3)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b[3:-3, 3:-3, 3:-3], atol=0, rtol=0)
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize(
    "volume,halo,error",
    [
        ([], 0, TypeError),
        (np.zeros((3, 4, 5), np.float32), 0, TypeError),
        (np.zeros((3, 4), np.uint8), 0, ValueError),
        (np.zeros((3, 4, 5), np.uint8), -1, ValueError),
        (np.zeros((3, 4, 5), np.uint8), 1.5, ValueError),
        (np.zeros((3, 4, 5), np.uint8), True, ValueError),
        (np.zeros((3, 4, 5), np.uint8), np.bool_(False), ValueError),
        (np.zeros((3, 4, 5), np.uint8), 2, ValueError),
        (np.zeros((0, 4, 5), np.uint8), 0, ValueError),
    ],
)
def test_invalid_volume_and_halo_are_rejected(volume, halo, error):
    with pytest.raises(error):
        observations.density_normals(volume, halo=halo)
