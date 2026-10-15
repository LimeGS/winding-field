"""CPU synthetic gates for one common-reference, four-tile refinement problem."""

import hashlib
import json

import numpy as np
import pytest


def pilot_command(inputs, output):
    return [
        "--output",
        str(output),
        "--origin",
        "0",
        "0",
        "0",
        "--size",
        "40",
        "40",
        "3",
        "--step",
        "1",
        "--device",
        "cpu",
        "--config",
        str(inputs.config),
        "--sweeps",
        "1",
    ]


def test_non_cubic_affine_pilot_preserves_physical_endpoints_and_zero_gauge():
    from coarse_reference import run_pilot_arrays

    shape, spacing = (7, 13, 17), np.array([1.3, 2.5, 0.75])
    z, y, x = np.indices(shape)
    target = np.broadcast_to([0.15, 0.2, -0.1], (*shape, 3))
    expected = 0.15 * x * spacing[0] + 0.2 * y * spacing[1] - 0.1 * z * spacing[2]
    mono, tiled, coarse, receipt = run_pilot_arrays(target, np.ones(shape), spacing, halo=2, sweeps=2, rtol=1e-6)
    assert receipt["status"] == "PASS"
    np.testing.assert_allclose(mono, expected, atol=1e-3)
    np.testing.assert_allclose(tiled, expected, atol=1e-3)
    assert coarse[0, 0, 0] == 0
    assert mono.mean() > 1  # no independent median subtraction
    geometry = receipt["coarse_grid"]
    expected_shape = np.ceil((np.array(shape) - 1) / 4).astype(int) + 1
    np.testing.assert_array_equal(coarse.shape, expected_shape)
    np.testing.assert_allclose(
        (np.array(coarse.shape[::-1]) - 1) * geometry["spacing_xyz"], (np.array(shape[::-1]) - 1) * spacing, atol=1e-12
    )
    assert len(receipt["passes"]) == 2 and len(receipt["tiles"]) == 4
    assert len(receipt["passes"][-1]["overlaps"]) == 6  # cyclic overlap graph, not a chain
    assert all(info["converged"] for info in receipt["solvers"].values())
    assert receipt["solvers"]["coarse"]["gauge"]["kind"] == "zero"
    assert receipt["solvers"]["monolithic"]["gauge"]["kind"] == "fixed"
    json.dumps(receipt, allow_nan=False)


def test_pilot_cli_runs_real_solver_and_records_transport_failure(paris_inputs):
    from conftest import LB

    from coarse_reference import main
    from patch_check import sha256

    output = paris_inputs.root / "pilot"
    assert main(pilot_command(paris_inputs, output)) == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "PASS" and all(receipt["gates"].values())
    assert len(receipt["tiles"]) == 4 and len(receipt["passes"]) == 1
    assembled = np.load(output / "assembled.npy")
    assert assembled.shape == (3, 40, 40) and np.isfinite(assembled).all()
    assert receipt["artifacts"]["assembled"]["field_sha256"] == sha256(output / "assembled.npy")
    del paris_inputs.volumes[f"{LB}_nx.ome.zarr/4"]
    failed = paris_inputs.root / "missing-data"
    assert main(pilot_command(paris_inputs, failed)) == 2
    error = json.loads((failed / "receipt.json").read_text())
    assert error["status"] == "ERROR" and error["error"]["type"] == "KeyError"
    assert not (failed / "assembled.npy").exists()


@pytest.mark.parametrize(
    "tamper", [None, "frame", "file_hash", "digest", "units", "constraint_units", "constraint_step"]
)
def test_pilot_cli_checks_continuous_constraint_identity(paris_inputs, tamper):
    from conftest import FRAME

    from coarse_reference import main
    from field_constraints import pair_constraints
    from paris_data import save_field
    from patch_check import sha256

    shape = (3, 40, 40)
    metadata = dict(frame=FRAME, voxel_um=9.6, origin_xyz=[0, 0, 0], spacing_xyz=[1, 1, 1], shape_zyx=list(shape))
    reference = paris_inputs.root / "reference.npy"
    save_field(
        reference, np.broadcast_to(0.1 * np.arange(40, dtype=float), shape), metadata, {"converged": True}, step=2
    )
    constraints = pair_constraints([[8, 20, 1]], [[32, 20, 1]], [0, 0, 0], [1, 1, 1], shape)
    path = paris_inputs.root / "constraints.npz"
    constraints.save(path)
    sidecar = dict(metadata, file_sha256=sha256(path), constraint_digest=constraints.digest)
    if tamper == "frame":
        sidecar["frame"] = "another-scan"
    elif tamper == "file_hash":
        sidecar["file_sha256"] = "0" * 64
    elif tamper == "digest":
        sidecar["constraint_digest"] = "0" * 64
    elif tamper == "units":
        ref_meta_path = reference.with_suffix(".npy.json")
        ref_meta = json.loads(ref_meta_path.read_text())
        ref_meta["voxel_um"] = 12
        ref_meta_path.write_text(json.dumps(ref_meta))
    elif tamper == "constraint_units":
        sidecar["voxel_um"] = 96
    elif tamper == "constraint_step":
        sidecar["step"] = 99
    path.with_suffix(".npz.json").write_text(json.dumps(sidecar))
    output = paris_inputs.root / "pilot"
    code = main(
        pilot_command(paris_inputs, output)
        + [
            "--reference-field",
            str(reference),
            "--reference-metadata",
            str(reference) + ".json",
            "--continuous-constraints",
            str(path),
        ]
    )
    receipt = json.loads((output / "receipt.json").read_text())
    if tamper is None:
        assert code == 0 and receipt["status"] == "PASS"
        assert receipt["input_hashes"]["continuous_constraints"] == constraints.digest
    else:
        assert code == 2 and receipt["status"] == "ERROR"
        assert receipt["error"]["type"] == "ValueError"
        assert not (output / "assembled.npy").exists()


def test_pilot_cli_requires_both_reference_files_before_creating_output(paris_inputs):
    from coarse_reference import main

    output = paris_inputs.root / "invalid"
    with pytest.raises(SystemExit) as failure:
        main(pilot_command(paris_inputs, output) + ["--reference-field", "missing.npy"])
    assert failure.value.code == 2 and not output.exists()


def test_curved_gradient_uses_same_outer_faces_and_satisfies_fine_interior_equations():
    from coarse_reference import run_pilot_arrays

    shape, h = (9, 13, 17), np.array([1.0, 1.5, 2.0])
    z, y, x = np.indices(shape)
    a = np.array([0.012, -0.009, 0.006])
    target = np.stack([2 * a[0] * x * h[0], 2 * a[1] * y * h[1], 2 * a[2] * z * h[2]], -1)
    mono, tiled, _, receipt = run_pilot_arrays(target, np.ones(shape), h, halo=2, sweeps=4, rtol=1e-6)
    assert receipt["status"] == "PASS"
    interior_laplacian = sum(
        np.diff(mono, n=2, axis=axis)[tuple(slice(None) if k == axis else slice(1, -1) for k in range(3))]
        for axis in range(3)
    )
    np.testing.assert_allclose(interior_laplacian, 2 * np.sum(a * h * h), atol=2e-3)
    for axis in range(3):
        for side in (0, -1):
            face = [slice(None)] * 3
            face[axis] = side
            np.testing.assert_array_equal(tiled[tuple(face)], mono[tuple(face)])
    assert receipt["configuration"]["sweeps"] == 4


def test_parent_pairs_are_subselected_and_reindexed_without_resampling():
    from coarse_reference import tile_pairs

    shape = (5, 7, 9)
    coords = np.array([[[1, 2, 3], [3, 5, 6]], [[1, 2, 3], [3, 6, 8]], [[0, 0, 0], [4, 6, 8]], [[1, 2, 3], [3, 5, 6]]])
    parent = np.ravel_multi_index(coords.reshape(-1, 3).T, shape).reshape(-1, 2)
    local, counts = tile_pairs(parent, shape, [[0, 5], [1, 6], [2, 7]])
    expected_coords = coords[[0, 3]] - [0, 1, 2]
    expected = np.ravel_multi_index(expected_coords.reshape(-1, 3).T, (5, 5, 5)).reshape(-1, 2)
    np.testing.assert_array_equal(local, expected)
    assert counts == dict(parent=4, kept=2, crossdomain_pair_drops=1, outside_pair_drops=1)


def test_coarse_pair_remapping_removes_collapses_and_undirected_duplicates():
    from coarse_reference import coarse_pairs

    shape, coarse_shape = (9, 13, 17), (3, 4, 5)
    coords = np.array(
        [[[0, 0, 0], [8, 12, 16]], [[8, 12, 16], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]], [[4, 4, 4], [8, 8, 8]]]
    )
    pairs = np.ravel_multi_index(coords.reshape(-1, 3).T, shape).reshape(-1, 2)
    got, counts = coarse_pairs(pairs, shape, coarse_shape)
    expected = np.array([[0, 59], [26, 52]])
    np.testing.assert_array_equal(got, expected)
    assert counts == dict(parent=4, kept=2, collapsed=1, duplicates=1)


def test_metrics_reject_absolute_offsets_without_fitting_them_away():
    from coarse_reference import measure_pass, tile_layout

    shape = (5, 9, 11)
    layout = tile_layout(shape, 1)
    fields = [np.full(tuple(b - a for a, b in tile["bounds_zyx"]), i, np.float32) for i, tile in enumerate(layout)]
    metrics = measure_pass(np.zeros(shape), np.full(shape, 5.0), fields, layout, 2.0)
    assert metrics["mono_p95_sheets"] == 2.5
    assert metrics["raw_overlap_p95_sheets"] == 1.5


def test_one_sweep_cannot_claim_pass_for_incoherent_random_gradient(tmp_path):
    from coarse_reference import run_pilot_arrays

    target = np.random.default_rng(2).normal(size=(9, 17, 21, 3)) * 10
    _, _, _, receipt = run_pilot_arrays(
        target, np.ones(target.shape[:3]), 1.0, halo=1, sweeps=1, rtol=1e-6, output=tmp_path
    )
    assert receipt["status"] == "FAIL"
    assert len(receipt["passes"]) == 1
    assert not all(receipt["gates"].values())
    assert receipt["gates"]["all_converged"]
    assert (tmp_path / "assembled.npy").exists()
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "FAIL"
    assert json.loads((tmp_path / "assembled.npy.json").read_text())["pilot_gate_status"] == "FAIL"


def test_failed_solver_is_recorded_and_stops_without_more_sweeps(tmp_path):
    from coarse_reference import run_pilot_arrays

    target = np.broadcast_to([1.0, 2.0, 3.0], (5, 9, 11, 3))
    mono, tiled, coarse, receipt = run_pilot_arrays(
        target, np.ones((5, 9, 11)), 1.0, halo=1, maxiter=0, output=tmp_path
    )
    assert receipt["status"] == "ERROR"
    assert receipt["error"]["stage"] == "coarse"
    assert not receipt["solvers"]["coarse"]["converged"]
    assert mono is tiled is coarse is None
    assert receipt["passes"] == []
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "ERROR"
    assert not (tmp_path / "coarse.npy").exists()
    json.dumps(receipt, allow_nan=False)


def test_persisted_fields_keep_metadata_support_and_core_ownership(tmp_path):
    from coarse_reference import run_pilot_arrays

    shape = (5, 9, 11)
    target = np.broadcast_to([0.1, 0.2, 0.3], (*shape, 3))
    mono, tiled, coarse, receipt = run_pilot_arrays(
        target,
        np.ones(shape),
        [2, 3, 4],
        halo=1,
        sweeps=1,
        rtol=1e-6,
        output=tmp_path,
        origin_xyz=[100, 200, 300],
        metadata={"frame": "synthetic:L0"},
        sheet_step=1.784599,
    )
    assert receipt["status"] == "PASS"
    for name, expected in [("coarse", coarse), ("monolithic", mono), ("assembled", tiled)]:
        np.testing.assert_array_equal(np.load(tmp_path / f"{name}.npy"), expected)
        meta = json.loads((tmp_path / f"{name}.npy.json").read_text())
        assert meta["frame"] == "synthetic:L0"
        assert meta["origin_xyz"] == [100, 200, 300]
        assert meta["step"] == 1.784599
        assert meta["solver"]["converged"]
    ownership = np.load(tmp_path / "core_ownership.npy")
    assert ownership.shape == shape and set(np.unique(ownership)) == {0, 1, 2, 3}
    assert np.load(tmp_path / "support.npy").all()
    assert len(list(tmp_path.glob("tile_*_sweep_1.npy"))) == 4
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "PASS"


@pytest.mark.parametrize(
    "kwargs",
    [{"sweeps": 0}, {"halo": 0}, {"sheet_step": 0}, {"spacing": [1, -1, 1]}, {"pairs": [[0, 10000]]}, {"halo": 16}],
)
def test_invalid_pilot_configuration_is_rejected(kwargs):
    from coarse_reference import run_pilot_arrays

    args = dict(target=np.zeros((5, 9, 11, 3)), weights=np.ones((5, 9, 11)), spacing=1.0, halo=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        run_pilot_arrays(**args)


@pytest.fixture
def external_reference(tmp_path):
    shape = (12, 13, 14)
    z, y, x = np.indices(shape)
    field = (123 + 0.25 * x * 4 + 0.5 * y * 4 + 0.125 * z * 4).astype(np.float32)
    path = tmp_path / "reference.npy"
    np.save(path, field)
    meta = dict(
        frame="synthetic:L0",
        origin_xyz=[0, 0, 0],
        spacing_xyz=[4, 4, 4],
        shape_zyx=list(shape),
        step=2.5,
        solver={"converged": True, "iterations": 17, "gauge": {"kind": "fixed", "value": 123}},
        field_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    metadata_path = tmp_path / "reference.npy.json"
    metadata_path.write_text(json.dumps(meta))
    return path, metadata_path, meta


def test_external_reference_preserves_global_offset_step_and_shifted_control(external_reference):
    from coarse_reference import run_pilot_arrays

    path, metadata_path, metadata = external_reference
    shape, origin = (5, 9, 11), np.array([8.0, 12.0, 16.0])
    target = np.broadcast_to([0.25, 0.5, 0.125], (*shape, 3))
    z, y, x = np.indices(shape)
    expected = 123 + 0.25 * (x + origin[0]) + 0.5 * (y + origin[1]) + 0.125 * (z + origin[2])
    runs = []
    for split_offset in (0, 1):
        mono, tiled, coarse, receipt = run_pilot_arrays(
            target,
            np.ones(shape),
            1.0,
            origin_xyz=origin,
            halo=1,
            sweeps=2,
            rtol=1e-6,
            split_offset=split_offset,
            external_reference=path,
            external_metadata=metadata_path,
            metadata={"frame": metadata["frame"]},
        )
        assert receipt["status"] == "PASS"
        np.testing.assert_allclose(mono, expected, atol=1e-3)
        np.testing.assert_allclose(tiled, expected, atol=1e-3)
        assert coarse[0, 0, 0] == expected[0, 0, 0] > 123
        assert "coarse" not in receipt["solvers"]  # imported, not locally solved
        assert receipt["reference"]["kind"] == "external_resampled_reference"
        assert receipt["reference"]["field_sha256"] == metadata["field_sha256"]
        assert receipt["reference"]["source_solver"]["gauge"]["value"] == 123
        assert receipt["configuration"]["sheet_step"] == 2.5
        assert receipt["configuration"]["split_offset"] == split_offset
        runs.append((mono, tiled, receipt))
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert runs[0][2]["input_hashes"] == runs[1][2]["input_hashes"]


@pytest.mark.parametrize("problem", ["frame", "hash", "coverage", "step", "unconverged", "unpaired"])
def test_invalid_external_reference_is_rejected(external_reference, problem):
    from coarse_reference import run_pilot_arrays

    path, metadata_path, meta = external_reference
    args = dict(
        origin_xyz=[8, 12, 16],
        halo=1,
        external_reference=path,
        external_metadata=metadata_path,
        metadata={"frame": "synthetic:L0"},
    )
    if problem == "frame":
        args["metadata"] = {"frame": "another-volume:L0"}
    elif problem == "hash":
        meta["field_sha256"] = "0" * 64
    elif problem == "coverage":
        args["origin_xyz"] = [100, 100, 100]
    elif problem == "step":
        args["sheet_step"] = 1.0
    elif problem == "unconverged":
        meta["solver"]["converged"] = False
    else:
        args["external_metadata"] = None
    metadata_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        run_pilot_arrays(np.zeros((5, 9, 11, 3)), np.ones((5, 9, 11)), 1.0, **args)


@pytest.mark.parametrize("offset", [-2, 0, 3])
def test_shifted_cores_own_every_voxel_exactly_once(offset):
    from coarse_reference import tile_layout

    shape = (7, 17, 19)
    coverage = np.zeros(shape, int)
    tiles = tile_layout(shape, halo=2, split_offset=offset)
    for tile in tiles:
        core = tuple(slice(a, b) for a, b in tile["core_bounds_zyx"])
        coverage[core] += 1
        for (a, b), (c, d) in zip(tile["bounds_zyx"], tile["core_bounds_zyx"]):
            assert a <= c < d <= b
    assert tiles[0]["core_bounds_zyx"][1][1] == shape[1] // 2 + offset
    assert tiles[0]["core_bounds_zyx"][2][1] == shape[2] // 2 + offset
    np.testing.assert_array_equal(coverage, np.ones(shape))


@pytest.mark.parametrize("offset", [-7, 7, 1.5])
def test_invalid_shifted_cuts_fail(offset):
    from coarse_reference import tile_layout

    with pytest.raises(ValueError):
        tile_layout((7, 17, 19), halo=2, split_offset=offset)


def test_retile_artifact_comparison_verifies_identity_support_and_never_aligns(external_reference, tmp_path):
    from coarse_reference import compare_pilot_outputs, run_pilot_arrays

    path, metadata_path, _ = external_reference
    dirs = [tmp_path / "normal", tmp_path / "shifted"]
    shape = (5, 9, 11)
    target = np.broadcast_to([0.25, 0.5, 0.125], (*shape, 3))
    for shift, output in enumerate(dirs):
        output.mkdir()
        run_pilot_arrays(
            target,
            np.ones(shape),
            1.0,
            origin_xyz=[8, 12, 16],
            halo=1,
            sweeps=1,
            split_offset=shift,
            output=output,
            external_reference=path,
            external_metadata=metadata_path,
            metadata={"frame": "synthetic:L0"},
        )
    result = compare_pilot_outputs(*dirs)
    assert result["status"] == "PASS"
    assert result["interior_p95_sheets"] < 0.001
    # A genuine offset must remain visible even with matching, refreshed hashes.
    p = dirs[1] / "assembled.npy"
    np.save(p, np.load(p) + 5)
    mpath = dirs[1] / "assembled.npy.json"
    meta = json.loads(mpath.read_text())
    meta["field_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    mpath.write_text(json.dumps(meta))
    receipt_path = dirs[1] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["assembled"]["field_sha256"] = meta["field_sha256"]
    receipt["artifacts"]["assembled"]["metadata_sha256"] = hashlib.sha256(mpath.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    result = compare_pilot_outputs(*dirs)
    assert result["status"] == "FAIL"
    assert result["interior_p95_sheets"] == pytest.approx(2.0, abs=0.001)
    spiked = np.load(p) - 5
    spiked[2, 3, 4] += 5
    np.save(p, spiked)
    meta["field_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    mpath.write_text(json.dumps(meta))
    receipt["artifacts"]["assembled"]["field_sha256"] = meta["field_sha256"]
    receipt["artifacts"]["assembled"]["metadata_sha256"] = hashlib.sha256(mpath.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    result = compare_pilot_outputs(*dirs)
    assert result["interior_p95_sheets"] < 0.001
    assert result["interior_count_gt_1_sheet"] == 1
    assert result["status"] == "FAIL"  # a good p95 cannot hide this tail
    support = np.load(dirs[1] / "support.npy")
    support[2, 3, 4] = False
    np.save(dirs[1] / "support.npy", support)
    with pytest.raises(ValueError, match="support"):
        compare_pilot_outputs(*dirs)


def test_crossing_fibres_reproduce_restricted_full_matrix_without_extra_pins():
    from test_field_solver import dense_system

    from coarse_reference import _faces, tile_fibre_terms
    from field_solver import solve_field

    rng = np.random.default_rng(21)
    shape = (3, 4, 6)
    target, weights = rng.normal(size=(*shape, 3)), rng.uniform(0.2, 2, shape)
    previous = rng.normal(size=shape) + 10
    # Internal, crossing in both directions, duplicate, fixed-face crossing,
    # and fully external constraints must all retain their correct roles.
    coords = np.array(
        [
            [[1, 1, 2], [1, 2, 3]],
            [[1, 1, 2], [1, 1, 5]],
            [[1, 2, 0], [1, 1, 2]],
            [[1, 1, 2], [1, 1, 5]],
            [[1, 1, 1], [1, 2, 5]],
            [[1, 2, 0], [1, 1, 5]],
        ]
    )
    pairs = np.ravel_multi_index(coords.reshape(-1, 3).T, shape).reshape(-1, 2)
    bounds, sl = [[0, 3], [0, 4], [1, 5]], (slice(None), slice(None), slice(1, 5))
    internal, ref, rw, counts = tile_fibre_terms(pairs, shape, bounds, previous, lam=3.0)
    assert counts["kept"] == 1 and counts["crossing_pairs"] == 4
    assert counts["outside_pair_drops"] == 1 and counts["crossdomain_pair_drops"] == 0
    assert rw[1, 1, 1] == 9  # three soft terms at the free inside endpoint
    assert rw[1, 1, 0] == 3  # the fixed-face constraint remains exact below
    np.testing.assert_allclose(ref[1, 1, 1], (2 * previous[1, 1, 5] + previous[1, 2, 0]) / 3, atol=1e-6)
    local_fixed = _faces(previous[sl])
    field, info = solve_field(
        target[sl],
        weights[sl],
        pairs=internal,
        lam=3.0,
        fixed=local_fixed,
        reference=ref,
        reference_weight=rw,
        rtol=1e-6,
    )
    assert info["converged"]
    matrix, rhs = dense_system(target, weights, (1, 1, 1), pairs, 3.0)
    free = np.zeros(shape, bool)
    free[sl] = np.isnan(local_fixed)
    f = free.ravel()
    expected = np.linalg.solve(matrix[np.ix_(f, f)], rhs[f] - matrix[np.ix_(f, ~f)] @ previous.ravel()[~f])
    np.testing.assert_allclose(field[np.isnan(local_fixed)], expected, atol=1e-4)
    np.testing.assert_array_equal(field[np.isfinite(local_fixed)], local_fixed[np.isfinite(local_fixed)])
    assert not np.allclose(field[1, 1, 1], ref[1, 1, 1])  # soft, not a hard inside pin


@pytest.mark.parametrize("mutation", ["both_masks", "both_steps"])
def test_comparison_rejects_joint_mutation_of_committed_support_or_scale(tmp_path, mutation):
    from coarse_reference import compare_pilot_outputs, run_pilot_arrays

    shape = (5, 9, 11)
    target = np.broadcast_to([0.1, 0.2, 0.3], (*shape, 3))
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
        run_pilot_arrays(target, np.ones(shape), 1, halo=1, sweeps=1, output=root)
        if mutation == "both_masks":
            support = np.load(root / "support.npy")
            support[2, 3, 4] = False
            np.save(root / "support.npy", support)
        else:
            path = root / "assembled.npy.json"
            meta = json.loads(path.read_text())
            meta["step"] *= 100
            path.write_text(json.dumps(meta))
            receipt_path = root / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["artifacts"]["assembled"]["metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="support|metadata|calibration"):
        compare_pilot_outputs(*roots)


def test_long_crossing_pair_is_preserved_by_pilot_sweeps(external_reference):
    from coarse_reference import run_pilot_arrays

    path, metadata_path, _ = external_reference
    shape = (5, 9, 15)
    target = np.broadcast_to([0.25, 0.5, 0.125], (*shape, 3))
    pairs = np.ravel_multi_index(np.array([[2, 4, 3], [2, 4, 11]]).T, shape).reshape(1, 2)
    mono, tiled, _, receipt = run_pilot_arrays(
        target,
        np.ones(shape),
        1.0,
        pairs,
        origin_xyz=[8, 12, 16],
        halo=1,
        sweeps=4,
        rtol=1e-6,
        external_reference=path,
        external_metadata=metadata_path,
        metadata={"frame": "synthetic:L0"},
    )
    assert receipt["status"] == "PASS"
    np.testing.assert_allclose(tiled.ravel()[pairs], mono.ravel()[pairs], atol=0.05)
    assert all(tile["pairs"]["crossdomain_pair_drops"] == 0 for tile in receipt["tiles"])
    assert any(tile["pairs"]["crossing_pairs"] for tile in receipt["tiles"])
    assert all(np.isfinite(p["global_residual_norm"]) for p in receipt["passes"])
    assert receipt["passes"][-1]["global_residual_norm"] < receipt["passes"][0]["global_residual_norm"]


def masked_reference(root, support, spacing=(1, 1, 1)):
    from paris_data import save_field
    from patch_check import sha256

    field = root / "masked-reference.npy"
    metadata = dict(
        frame="masked/world",
        voxel_um=12,
        origin_xyz=[0, 0, 0],
        spacing_xyz=list(spacing),
        shape_zyx=list(support.shape),
        support="reference-support.npy",
    )
    np.save(root / metadata["support"], support)
    metadata["support_sha256"] = sha256(root / metadata["support"])
    values = np.broadcast_to(np.arange(support.shape[2], dtype=float) * spacing[0], support.shape)
    save_field(field, values, metadata, {"converged": True}, step=1)
    return field, field.with_suffix(".npy.json")


@pytest.mark.parametrize("problem", ["unsupported", "hash", "unneeded_hole"])
def test_reference_mask_identity_and_requested_stencils_are_enforced(tmp_path, problem):
    from coarse_reference import load_external_reference

    support = np.ones((5, 5, 5), bool)
    support[
        2 if problem == "unsupported" else 3, 2 if problem == "unsupported" else 3, 2 if problem == "unsupported" else 3
    ] = False
    field, metadata = masked_reference(tmp_path, support)
    if problem == "hash":
        data = json.loads(metadata.read_text())
        data["support_sha256"] = "0" * 64
        metadata.write_text(json.dumps(data))
    args = (field, metadata, "masked/world", np.array([1, 1, 1]), np.ones(3), (2, 2, 2), 1)
    if problem == "unneeded_hole":
        load_external_reference(*args, voxel_um=12)
    else:
        with pytest.raises(ValueError, match="support"):
            load_external_reference(*args, voxel_um=12)


def test_reference_support_is_checked_on_coarse_as_well_as_fine_samples(tmp_path):
    from coarse_reference import run_pilot_arrays

    support = np.ones((5, 15, 9), bool)
    support[0, 7, 0] = False  # y=3.5: used by coarse sampling, not integer fine samples.
    field, metadata = masked_reference(tmp_path, support, [0.5] * 3)
    shape = (3, 8, 5)
    _, assembled, _, receipt = run_pilot_arrays(
        np.broadcast_to([1.0, 0, 0], (*shape, 3)),
        np.ones(shape),
        1,
        halo=1,
        sweeps=1,
        external_reference=field,
        external_metadata=metadata,
        metadata=dict(frame="masked/world", voxel_um=12),
        sheet_step=1,
    )
    assert assembled is None and receipt["status"] == "ERROR"
    assert "support" in receipt["error"]["message"].lower()


def test_coarse_cli_honors_configured_solver_budget(paris_inputs):
    from coarse_reference import main

    data = paris_inputs.raw_config
    data["solver"] = dict(maxiter=1, rtol=1e-10)
    paris_inputs.config.write_text(json.dumps(data))
    output = paris_inputs.root / "limited-pilot"
    assert main(pilot_command(paris_inputs, output)) == 2
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["configuration"]["maxiter"] == 1
    assert receipt["solvers"]["coarse"]["iterations"] == 1
