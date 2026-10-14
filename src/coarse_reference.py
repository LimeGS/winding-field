"""Connected 2x2-XY common-reference pilot; numerical agreement is not scroll validity."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import affine_transform

from field_solver import _FieldOperator, _norm, _scalar, solve_field
from physical_grid import sample_volume


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def load_external_reference(field_path, metadata_path, frame, origin, h, shape, sheet_step=None):
    """Verify a sealed NPY/JSON pair before sampling; retain raw values and gauge."""
    meta = json.loads(Path(metadata_path).read_text())
    digest = _file_hash(field_path)
    if digest != meta.get("field_sha256"):
        raise ValueError("external reference field hash mismatch")
    if not frame or frame != meta.get("frame"):
        raise ValueError("external reference frame mismatch")
    if meta.get("solver", {}).get("converged") is not True:
        raise ValueError("external reference source solver is not converged")
    if meta.get("chart_local_potential"):
        raise ValueError("chart-local potential requires an explicit lifted reference, not scalar smoothing")
    step = float(meta.get("step", 0))
    if (
        not np.isfinite(step)
        or step <= 0
        or (sheet_step is not None and not np.isclose(float(sheet_step), step, rtol=1e-12, atol=0))
    ):
        raise ValueError("sheet_step must match the external reference calibration")
    field = np.load(field_path, mmap_mode="r", allow_pickle=False)
    sh, so = np.asarray(meta["spacing_xyz"], float), np.asarray(meta["origin_xyz"], float)
    if (
        field.ndim != 3
        or list(field.shape) != meta["shape_zyx"]
        or not np.isfinite(field).all()
        or sh.shape != (3,)
        or so.shape != (3,)
        or not np.isfinite([sh, so]).all()
        or (sh <= 0).any()
    ):
        raise ValueError("invalid external reference grid/values")
    last = origin + (np.asarray(shape[::-1]) - 1) * h
    if (origin < so).any() or (last > so + (np.asarray(field.shape[::-1]) - 1) * sh).any():
        raise ValueError("pilot extends outside external reference support")
    provenance = dict(
        kind="external_resampled_reference",
        field_path=str(Path(field_path).resolve()),
        metadata_path=str(Path(metadata_path).resolve()),
        field_sha256=digest,
        metadata_sha256=_file_hash(metadata_path),
        frame=frame,
        source_solver=meta["solver"],
        source_origin_xyz=so.tolist(),
        source_spacing_xyz=sh.tolist(),
        source_shape_zyx=list(field.shape),
        step=step,
        resampling="physical_grid.sample_volume order=1; no gauge/scale fitting",
    )
    return field, meta, provenance


def _pairs(pairs, shape):
    pairs = np.asarray([] if pairs is None else pairs)
    if not pairs.size:
        return np.empty((0, 2), np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("pairs must be integer (M,2) flat ZYX indices")
    if (pairs < 0).any() or (pairs >= np.prod(shape)).any():
        raise ValueError("pair index outside parent grid")
    return pairs.astype(np.int64)


def _coordinates(pairs, shape):
    return np.array(np.unravel_index(pairs.ravel(), shape)).T.reshape(-1, 2, 3)


def coarse_pairs(pairs, shape, coarse_shape):
    pairs = _pairs(pairs, shape)
    coords = np.rint(_coordinates(pairs, shape) * ((np.asarray(coarse_shape) - 1) / (np.asarray(shape) - 1))).astype(
        int
    )
    mapped = np.ravel_multi_index(coords.reshape(-1, 3).T, coarse_shape).reshape(-1, 2)
    keep = mapped[:, 0] != mapped[:, 1]
    unique = np.unique(np.sort(mapped[keep], axis=1), axis=0)
    return unique, dict(
        parent=len(pairs), kept=len(unique), collapsed=int((~keep).sum()), duplicates=int(keep.sum()) - len(unique)
    )


def tile_pairs(pairs, shape, bounds):
    pairs = _pairs(pairs, shape)
    low, high = np.asarray(bounds).T
    coords = _coordinates(pairs, shape)
    inside = ((coords >= low) & (coords < high)).all(-1)
    keep = inside.all(1)
    local = np.ravel_multi_index((coords[keep] - low).reshape(-1, 3).T, high - low).reshape(-1, 2)
    return local, dict(
        parent=len(pairs),
        kept=len(local),
        crossdomain_pair_drops=int((inside[:, 0] != inside[:, 1]).sum()),
        outside_pair_drops=int((~inside.any(1)).sum()),
    )


def tile_fibre_terms(pairs, shape, bounds, previous, lam=3.0):
    """Restrict global fibre energy: internal pairs plus soft crossing penalties.

    Each crossing adds lam to the inside diagonal and lam*previous[outside]
    to its RHS. Repeated pairs retain multiplicity; fixed inside nodes remain
    the solver's responsibility, not hard pins to the outside reference.
    """
    lam = _scalar(lam, "lam")
    if previous.shape != tuple(shape) or not np.isfinite(previous).all():
        raise ValueError("previous assembled field must match the finite parent grid")
    pairs = _pairs(pairs, shape)
    internal, counts = tile_pairs(pairs, shape, bounds)
    low, high = np.asarray(bounds).T
    coords = _coordinates(pairs, shape)
    inside = ((coords >= low) & (coords < high)).all(-1)
    cross = inside[:, 0] != inside[:, 1]
    endpoint = inside[cross, 1].astype(int)
    row = np.arange(int(cross.sum()))
    local = np.ravel_multi_index((coords[cross][row, endpoint] - low).T, high - low)
    outside = pairs[cross][row, 1 - endpoint]
    nodes, inverse, multiplicity = np.unique(local, return_inverse=True, return_counts=True)
    reference, rw = np.zeros(high - low, np.float32), np.zeros(high - low, np.float32)
    if len(nodes):
        sums = np.bincount(inverse, weights=previous.ravel()[outside].astype(np.float64))
        reference.ravel()[nodes] = sums / multiplicity
        rw.ravel()[nodes] = lam * multiplicity
    counts.update(crossdomain_pair_drops=0, crossing_pairs=int(cross.sum()), crossing_anchor_nodes=len(nodes))
    return internal, reference, rw, counts


def tile_layout(shape, halo, split_offset=0):
    nz, ny, nx = shape
    if not isinstance(split_offset, (int, np.integer)):
        raise ValueError("split_offset must be an integer number of fine voxels")
    yc, xc = ny // 2 + split_offset, nx // 2 + split_offset
    if not (0 < halo < min(yc, ny - yc, xc, nx - xc)):
        raise ValueError("invalid cuts/halo: require four distinct overlapping XY subdomains")
    tiles = []
    for y, x in itertools.product(((0, yc), (yc, ny)), ((0, xc), (xc, nx))):
        tiles.append(
            dict(
                id=len(tiles),
                bounds_zyx=[
                    [0, nz],
                    [max(0, y[0] - halo), min(ny, y[1] + halo)],
                    [max(0, x[0] - halo), min(nx, x[1] + halo)],
                ],
                core_bounds_zyx=[[0, nz], list(y), list(x)],
            )
        )
    return tiles


def _slices(bounds):
    return tuple(slice(int(a), int(b)) for a, b in bounds)


def _sample(field, source_h, shape, h):
    if not np.allclose(
        (np.asarray(field.shape[::-1]) - 1) * source_h, (np.asarray(shape[::-1]) - 1) * h, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("resampling grids must share first and last physical positions")
    # Only coincident endpoints may round beyond support; nearest handles that ULP.
    return affine_transform(
        field, np.diag((h / source_h)[::-1]), output_shape=tuple(shape), order=1, mode="nearest", prefilter=False
    )


def _faces(field):
    fixed = np.full(field.shape, np.nan, np.float32)
    for axis in range(3):
        for side in (0, -1):
            face = [slice(None)] * 3
            face[axis] = side
            fixed[tuple(face)] = field[tuple(face)]
    return fixed


def measure_pass(monolithic, assembled, fields, tiles, sheet_step):
    overlaps = []
    for a, b in itertools.combinations(range(len(tiles)), 2):
        ba, bb = np.array(tiles[a]["bounds_zyx"]), np.array(tiles[b]["bounds_zyx"])
        low, high = np.maximum(ba[:, 0], bb[:, 0]), np.minimum(ba[:, 1], bb[:, 1])
        if (high <= low).any():
            continue
        va = fields[a][_slices(np.stack([low - ba[:, 0], high - ba[:, 0]], 1))]
        vb = fields[b][_slices(np.stack([low - bb[:, 0], high - bb[:, 0]], 1))]
        delta = (va - vb) / sheet_step
        overlaps.append(
            dict(
                a=a,
                b=b,
                samples=delta.size,
                signed_median_sheets=float(np.median(delta)),
                p95_sheets=float(np.percentile(np.abs(delta), 95)),
            )
        )
    delta = np.abs(assembled[1:-1, 1:-1, 1:-1] - monolithic[1:-1, 1:-1, 1:-1]) / sheet_step
    return dict(
        mono_p95_sheets=float(np.percentile(delta, 95)),
        core_interior_samples=delta.size,
        mono_p99_sheets=float(np.percentile(delta, 99)),
        mono_max_sheets=float(delta.max()),
        mono_count_gt_1_sheet=int((delta > 1).sum()),
        raw_overlap_p95_sheets=max(o["p95_sheets"] for o in overlaps),
        overlaps=overlaps,
    )


def run_pilot_arrays(
    target,
    weights,
    spacing,
    pairs=None,
    *,
    origin_xyz=(0, 0, 0),
    halo=16,
    sweeps=4,
    sheet_step=None,
    lam=3.0,
    device="cpu",
    rtol=1e-5,
    maxiter=4000,
    output=None,
    metadata=None,
    external_reference=None,
    external_metadata=None,
    split_offset=0,
    constraints=None,
):
    """Return monolithic-refinement, assembled, coarse and JSON-safe receipts.

    output, if supplied, must be an empty existing directory. Solver failures
    return ERROR and possibly None fields; metric failures retain solved fields
    with FAIL. Every declared sweep runs, with no adaptive retries or alignment.
    External reference and metadata are paired file paths. Their calibration is
    inherited unless explicitly asserted equal; local-reference default step is 1.
    """
    if np.iscomplexobj(target) or np.iscomplexobj(weights):
        raise ValueError("real target and weights required")
    target, weights = np.asarray(target, np.float32), np.asarray(weights, np.float32)
    shape = weights.shape
    h, origin = np.broadcast_to(np.asarray(spacing, float), (3,)).copy(), np.asarray(origin_xyz, float)
    if (
        len(shape) != 3
        or min(shape) < 3
        or min(shape[1:]) < 4
        or target.shape != (*shape, 3)
        or not np.isfinite(target).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
        or origin.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(h).all()
        or (h <= 0).any()
    ):
        raise ValueError("invalid finite physical grid, target or positive weights")
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in (halo, sweeps)):
        raise ValueError("halo and sweeps must be positive integers")
    tiles = tile_layout(shape, halo, split_offset)
    if (external_reference is None) != (external_metadata is None):
        raise ValueError("external_reference and external_metadata must be supplied together")
    external = refmeta = reference = None
    if external_reference is not None:
        external, refmeta, reference = load_external_reference(
            external_reference, external_metadata, (metadata or {}).get("frame"), origin, h, shape, sheet_step
        )
        sheet_step = reference["step"]
    elif sheet_step is None:
        sheet_step = 1.0
    if not np.isfinite(sheet_step) or sheet_step <= 0:
        raise ValueError("sheet_step must be positive and fixed independently of agreement")
    pairs = _pairs(pairs, shape)
    if constraints is not None and (constraints.matrix.shape[1] != int(np.prod(shape)) or external is None):
        raise ValueError("continuous pilot constraints require matching grid and an explicit shared reference")
    output = None if output is None else Path(output)
    if output is not None and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("output must be an empty existing directory")
    cshape = np.ceil((np.asarray(shape) - 1) / 4).astype(int) + 1
    ch = h * (np.asarray(shape[::-1]) - 1) / (cshape[::-1] - 1)
    receipt = dict(
        schema="common_reference_pilot_v2_cross_fibres",
        scope="connected_2x2_XY_pilot_not_whole_scroll",
        configuration=dict(
            shape_zyx=list(shape),
            spacing_xyz=h.tolist(),
            origin_xyz=origin.tolist(),
            halo=int(halo),
            sweeps=int(sweeps),
            coarse_factor=4,
            sheet_step=float(sheet_step),
            lam=float(lam),
            device=str(device),
            rtol=float(rtol),
            maxiter=int(maxiter),
            split_offset=int(split_offset),
        ),
        crossing_fibre_policy="lam*count nodal penalty to mean previous-global outside value; no dropped crossing pairs",
        coarse_grid=dict(shape_zyx=cshape.tolist(), spacing_xyz=ch.tolist()),
        tiles=tiles,
        passes=[],
        solvers={},
        artifacts={},
        status="ERROR",
        input_hashes=dict(
            target_xyz=_array_hash(target), weights_zyx=_array_hash(weights), parent_pairs=_array_hash(pairs)
        ),
        reference=reference,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        thresholds=dict(mono_p95_sheets=0.1, raw_overlap_p95_sheets=0.25),
        limitations=[
            "Default sheet_step=1 is an unvalidated unit step; no agreement fitting.",
            "Monolithic control fixes coarse outer faces; it is not an unconstrained solve.",
            "Support is computational, not tissue certification; topology/sign remain unproven.",
            "Crossing fibre terms use previous-sweep outside values; finite sweeps may not reach the global solution.",
        ],
        metric_domain="Owned cores excluding global outer faces; worst pairwise raw-overlap p95.",
    )
    mono = assembled = coarse = None
    stage = "setup"
    meta = dict(metadata or {})
    meta.setdefault("frame", "synthetic")
    if constraints is not None:
        receipt["input_hashes"]["continuous_constraints"] = constraints.digest
        receipt["continuous_constraints"] = dict(
            rows=constraints.matrix.shape[0],
            trace=constraints.trace,
            policy="global rows and group weights retained, external contributions moved into targets",
        )

    def store(name, field, info, grid_h=h, grid_origin=origin, **extra):
        if output is None:
            return
        from paris_data import save_field

        saved = save_field(
            output / f"{name}.npy",
            field,
            {
                **meta,
                "origin_xyz": grid_origin.tolist(),
                "spacing_xyz": grid_h.tolist(),
                "shape_zyx": list(field.shape),
                "pilot_scope": receipt["scope"],
                "problem": name,
                "constraint_pairs": None,
                "coarse_reference_array_sha256": receipt.get("coarse_reference_array_sha256"),
                "reference": reference,
                **extra,
            },
            info,
            step=sheet_step,
        )
        receipt["artifacts"][name] = dict(
            field=f"{name}.npy",
            metadata=f"{name}.npy.json",
            field_sha256=saved["field_sha256"],
            metadata_sha256=_file_hash(output / f"{name}.npy.json"),
        )

    def solve(
        name, g, w, ps, hs, fixed=None, initial=None, nodal_reference=None, nodal_weight=0.0, local_constraints=None
    ):
        nonlocal stage
        stage = name
        print(f"SOLVE {name} shape={w.shape} pairs={len(ps)}", flush=True)
        field, info = solve_field(
            g,
            w,
            hs,
            ps,
            lam,
            fixed=fixed,
            initial=initial,
            reference=nodal_reference,
            reference_weight=nodal_weight,
            device=device,
            rtol=rtol,
            maxiter=maxiter,
            constraints=local_constraints,
        )
        receipt["solvers"][name] = info
        if not info["converged"]:
            raise RuntimeError(f"{name}: solver did not converge ({info.get('reason')})")
        return field, info

    try:
        if external is None:
            cp, counts = coarse_pairs(pairs, shape, cshape)
            receipt["coarse_grid"]["pairs"] = counts
            cg = np.stack([_sample(target[..., k], h, cshape, ch) for k in range(3)], -1)
            cw = _sample(weights, h, cshape, ch)
            coarse, info = solve("coarse", cg, cw, cp, ch)
            base = _sample(coarse, ch, shape, h)
        else:
            stage = "external_reference_resampling"
            samples = []

            def sample_external(hs, sh):
                return sample_volume(
                    external, origin, hs, sh, refmeta["spacing_xyz"], refmeta["origin_xyz"], order=1, receipts=samples
                )

            coarse, base = sample_external(ch, cshape), sample_external(h, shape)
            receipt["coarse_grid"].update(kind="external_resampled_reference", sample_receipts=samples)
            info = dict(
                converged=True,
                kind="external_resampled_reference",
                iterations=None,
                relative_residual=None,
                gauge=dict(kind="inherited", source=refmeta["solver"].get("gauge")),
            )
        receipt["coarse_reference_array_sha256"] = _array_hash(coarse)
        store(
            "coarse",
            coarse,
            info,
            ch,
            constraint_pairs=len(cp) if external is None else None,
            pair_remapping=counts if external is None else None,
        )
        mono, info = solve("monolithic", target, weights, pairs, h, _faces(base), base, local_constraints=constraints)
        store(
            "monolithic",
            mono,
            info,
            constraint_pairs=len(pairs),
            boundary_problem="coarse_outer_faces_refinement_control",
        )
        global_op = _FieldOperator(torch.from_numpy(weights), torch.from_numpy(pairs), lam)
        global_rhs = global_op.rhs(torch.from_numpy(target), h)
        continuous_op = None if constraints is None else constraints.operator("cpu")
        if continuous_op is not None:
            global_rhs += continuous_op.rhs().reshape(shape)
        interior = (slice(1, -1),) * 3
        boundary = torch.from_numpy(np.nan_to_num(_faces(base), nan=0.0))
        boundary_term = global_op.matvec(boundary)
        if continuous_op is not None:
            boundary_term += continuous_op.matvec(boundary)
        residual_scale = _norm((global_rhs - boundary_term)[interior]) or 1.0
        del boundary
        ownership = np.full(shape, -1, np.int8)
        for tile in tiles:
            ownership[_slices(tile["core_bounds_zyx"])] = tile["id"]
        if output is not None:
            np.save(output / "support.npy", np.ones(shape, bool))
            np.save(output / "core_ownership.npy", ownership)
            receipt.update(support="support.npy", core_ownership="core_ownership.npy")
            receipt["support_sha256"] = _file_hash(output / "support.npy")
            receipt["core_ownership_sha256"] = _file_hash(output / "core_ownership.npy")
        previous = base
        for sweep in range(1, sweeps + 1):
            fields, infos = [], []
            for tile in tiles:
                bounds = np.array(tile["bounds_zyx"])
                sl = _slices(bounds)
                ps, fibre_ref, fibre_weight, tile["pairs"] = tile_fibre_terms(pairs, shape, bounds, previous, lam)
                fixed = _faces(previous[sl])
                for axis in range(3):
                    for side, touches in ((0, bounds[axis, 0] == 0), (-1, bounds[axis, 1] == shape[axis])):
                        if touches:
                            face = [slice(None)] * 3
                            face[axis] = side
                            fixed[tuple(face)] = base[sl][tuple(face)]
                name = f"tile_{tile['id']}_sweep_{sweep}"
                local_constraints = None
                if constraints is not None:
                    coords = np.indices(tuple(bounds[:, 1] - bounds[:, 0])).reshape(3, -1) + bounds[:, 0, None]
                    nodes = np.ravel_multi_index(coords, shape)
                    local_constraints = constraints.restrict(nodes, previous)
                field, info = solve(
                    name,
                    target[sl],
                    weights[sl],
                    ps,
                    h,
                    fixed,
                    previous[sl],
                    fibre_ref,
                    fibre_weight,
                    local_constraints,
                )
                fields.append(field)
                infos.append(info)
                store(
                    name,
                    field,
                    info,
                    grid_origin=origin + bounds[:, 0][::-1] * h,
                    parent_bounds_zyx=tile["bounds_zyx"],
                    parent_core_bounds_zyx=tile["core_bounds_zyx"],
                    constraint_pairs=len(ps),
                    pair_selection=tile["pairs"],
                    boundary_problem="previous_sweep_faces_global_outer_faces_fixed_to_coarse",
                )
            assembled = np.empty(shape, np.float32)
            for tile, field in zip(tiles, fields):
                core, low = np.array(tile["core_bounds_zyx"]), np.array(tile["bounds_zyx"])[:, 0]
                assembled[_slices(core)] = field[_slices(core - low[:, None])]
            metrics = measure_pass(mono, assembled, fields, tiles, sheet_step)
            residual = global_op.matvec(torch.from_numpy(assembled)) - global_rhs
            if continuous_op is not None:
                residual += continuous_op.matvec(torch.from_numpy(assembled))
            metrics["global_residual_norm"] = _norm(residual[interior])
            metrics["global_relative_residual"] = metrics["global_residual_norm"] / residual_scale
            del residual
            receipt["passes"].append(dict(sweep=sweep, **metrics))
            assembly_info = dict(
                converged=all(i["converged"] for i in infos), kind="assembly_of_converged_tiles_not_a_global_solve"
            )
            store(
                f"assembled_sweep_{sweep}",
                assembled,
                assembly_info,
                support="support.npy",
                core_ownership="core_ownership.npy",
            )
            previous = assembled
        receipt["gates"] = dict(
            all_converged=all(i["converged"] for i in receipt["solvers"].values()),
            mono_p95_below_0_1=metrics["mono_p95_sheets"] < 0.1,
            raw_overlap_p95_below_0_25=metrics["raw_overlap_p95_sheets"] < 0.25,
        )
        receipt["status"] = "PASS" if all(receipt["gates"].values()) else "FAIL"
        store(
            "assembled",
            assembled,
            assembly_info,
            support="support.npy",
            core_ownership="core_ownership.npy",
            pilot_gate_status=receipt["status"],
        )
    except Exception as exc:
        receipt["status"] = "ERROR"
        receipt["error"] = dict(stage=stage, type=type(exc).__name__, message=str(exc))
    if output is not None:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return mono, assembled, coarse, receipt


def compare_pilot_outputs(first, second):
    """Hash-verified, unaligned comparison on identical physical support and inputs."""
    roots = [Path(first), Path(second)]
    receipts = [json.loads((root / "receipt.json").read_text()) for root in roots]
    fields, metas, support = [], [], []
    for root, receipt in zip(roots, receipts):
        if _file_hash(root / "assembled.npy.json") != receipt["artifacts"]["assembled"].get("metadata_sha256"):
            raise ValueError("assembled metadata is not bound to the original run receipt")
        if _file_hash(root / "support.npy") != receipt.get("support_sha256"):
            raise ValueError("support is not bound to the original run receipt")
        meta = json.loads((root / "assembled.npy.json").read_text())
        if meta.get("step") != receipt["configuration"]["sheet_step"]:
            raise ValueError("calibration differs from the original run configuration")
        digest = _file_hash(root / "assembled.npy")
        if digest != meta["field_sha256"] or digest != receipt["artifacts"]["assembled"]["field_sha256"]:
            raise ValueError("assembled field hash mismatch")
        field = np.load(root / "assembled.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(root / "support.npy", allow_pickle=False)
        if (
            list(field.shape) != meta["shape_zyx"]
            or mask.shape != field.shape
            or mask.dtype != bool
            or not np.isfinite(field).all()
        ):
            raise ValueError("invalid field/support grid")
        fields.append(field)
        metas.append(meta)
        support.append(mask)
    keys = ("frame", "origin_xyz", "spacing_xyz", "shape_zyx", "step", "reference")
    if any(metas[0].get(k) != metas[1].get(k) for k in keys):
        raise ValueError("different frames, grids, calibration or reference identity")
    if receipts[0]["input_hashes"] != receipts[1]["input_hashes"]:
        raise ValueError("different fine inputs or parent fibre pairs")
    if not np.array_equal(*support):
        raise ValueError("different support masks")
    mask = support[0].copy()
    for axis in range(3):
        for side in (0, -1):
            face = [slice(None)] * 3
            face[axis] = side
            mask[tuple(face)] = False
    if not mask.any():
        raise ValueError("empty common interior support")
    step = float(metas[0]["step"])
    if not np.isfinite(step) or step <= 0:
        raise ValueError("invalid shared sheet step")
    delta = (fields[1][mask].astype(np.float64) - fields[0][mask]) / step
    p95 = float(np.percentile(np.abs(delta), 95))
    tail_count = int((np.abs(delta) > 1).sum())
    gates = dict(
        cases_pass=all(r["status"] == "PASS" for r in receipts),
        interior_p95_below_0_1=p95 < 0.1,
        no_retile_samples_gt_1_sheet=tail_count == 0,
    )
    return dict(
        status="PASS" if all(gates.values()) else "FAIL",
        gates=gates,
        paths=[str(p.resolve()) for p in roots],
        field_sha256=[m["field_sha256"] for m in metas],
        receipt_sha256=[_file_hash(root / "receipt.json") for root in roots],
        frame=metas[0]["frame"],
        origin_xyz=metas[0]["origin_xyz"],
        spacing_xyz=metas[0]["spacing_xyz"],
        shape_zyx=metas[0]["shape_zyx"],
        sheet_step=step,
        reference=metas[0]["reference"],
        input_hashes=receipts[0]["input_hashes"],
        support_array_sha256=_array_hash(support[0]),
        samples=int(mask.sum()),
        interior_p95_sheets=p95,
        interior_max_sheets=float(np.abs(delta).max()),
        interior_p99_sheets=float(np.percentile(np.abs(delta), 99)),
        interior_count_gt_1_sheet=tail_count,
        signed_median_sheets=float(np.median(delta)),
        alignment="none",
        case_gates=[r["gates"] for r in receipts],
        case_final_metrics=[r["passes"][-1] for r in receipts],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--origin", nargs=3, type=float, default=[4088, 2500, 10096])
    parser.add_argument("--size", nargs=3, type=float, default=[512, 512, 512])
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="/root/wrev")
    parser.add_argument("--afv")
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--halo", type=int, default=16, choices=[16, 32])
    parser.add_argument(
        "--sheet-step", type=float, help="default: external reference step, otherwise 1; must match reference"
    )
    parser.add_argument("--continuous-constraints", type=Path, help="NPZ plus JSON physical frame/grid/digest sidecar")
    parser.add_argument("--reference-field", type=Path)
    parser.add_argument("--reference-metadata", type=Path)
    parser.add_argument("--split-offset", type=int, default=0, help="shift both XY core cuts in fine-grid voxels")
    args = parser.parse_args(argv)
    if (args.reference_field is None) != (args.reference_metadata is None):
        parser.error("--reference-field and --reference-metadata are required together")
    args.output.mkdir(exist_ok=False)
    try:
        from paris_data import load_paris_box

        target, weights, pairs, meta = load_paris_box(
            args.origin,
            np.array(args.origin) + args.size,
            spacing=args.step,
            afv_path=args.afv,
            data_root=args.data_root,
        )
        continuous = None
        if args.continuous_constraints:
            from field_constraints import QuadraticConstraints

            cm = json.loads(Path(str(args.continuous_constraints) + ".json").read_text())
            for key in ("frame", "origin_xyz", "spacing_xyz", "shape_zyx"):
                if cm[key] != meta[key]:
                    raise ValueError("continuous constraint physical metadata mismatch")
            if _file_hash(args.continuous_constraints) != cm["file_sha256"]:
                raise ValueError("constraint file hash mismatch")
            continuous = QuadraticConstraints.load(args.continuous_constraints)
            if continuous.digest != cm["constraint_digest"]:
                raise ValueError("constraint rows changed")
        _, _, _, receipt = run_pilot_arrays(
            target,
            weights,
            args.step,
            pairs,
            origin_xyz=args.origin,
            sweeps=args.sweeps,
            halo=args.halo,
            sheet_step=args.sheet_step,
            device=args.device,
            output=args.output,
            metadata=meta,
            external_reference=args.reference_field,
            external_metadata=args.reference_metadata,
            split_offset=args.split_offset,
            constraints=continuous,
        )
    except Exception as exc:
        receipt = dict(status="ERROR", error=dict(stage="input", type=type(exc).__name__, message=str(exc)))
    receipt["command"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=receipt["status"], gates=receipt.get("gates"), error=receipt.get("error"))))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
