#!/usr/bin/env python3
"""Run approved coherence features on synthetic truth or hash-bound real data.

synthetic: all five stages, analytic truth/negative controls.
global: balanced split and full-domain continuous/consensus ablations on fixed cache.
local: native geometry/separation and unsigned arrival, bounded observation crop.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "benchmarks" / "winding"))
import metrics as benchmark

from anisotropic_reference import exterior_seeds, solve_arrival
from coherence_pipeline import compile_fibres, compile_separation, match_trace
from dataset_config import artifact_path as artifact
from dataset_config import load_dataset, mode_constraints, native_to_world, observation_recipe, validate_afv
from evaluation_split import build_fibre_split, validate_fibre_split
from field_constraints import QuadraticConstraints, solve_with_relations
from field_topology import build_transversal_inventory, geometry_charts
from paris_data import black_value, exterior_faces, load_paris_box, sample_source, save_field
from patch_check import sha256


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_hashes():
    return {
        name: sha256(BASE / "src" / name)
        for name in (
            "coherence_pipeline.py",
            "field_constraints.py",
            "field_solver.py",
            "evaluation_split.py",
            "dataset_config.py",
            "paris_data.py",
            "common.py",
            "physical_grid.py",
        )
    }


def finish(out, result):
    write(out / "RESULTS.json", result)
    write(out / "MANIFEST.json", {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()})
    return result


def validate_cached_reference(path, plan, shape):
    metadata_path = Path(str(path) + ".json")
    m = json.loads(metadata_path.read_text())
    if (
        sha256(path) != plan["baseline_sha256"]
        or sha256(metadata_path) != plan["baseline_metadata_sha256"]
        or m["step"] != plan["calibration_step"]
        or list(shape) != m["shape_zyx"]
        or m["field_sha256"] != plan["baseline_sha256"]
        or not m["solver"]["converged"]
    ):
        raise ValueError("reference bytes, calibration or grid differs from frozen cache provenance")
    origin = np.asarray(m["origin_xyz"], float)
    h = np.asarray(m["spacing_xyz"], float)
    if origin.shape != (3,) or h.shape != (3,) or not np.isfinite([origin, h]).all() or (h <= 0).any():
        raise ValueError("invalid reference coordinate metadata")
    return m


def nodal_control(a, b, origin, h, shape, weight):
    indices = np.rint((np.stack([a, b], axis=1) - origin) / h).astype(int)[:, :, ::-1]
    pairs = np.ravel_multi_index(indices.reshape(-1, 3).T, shape).reshape(-1, 2)
    n = len(pairs)
    rows = np.repeat(np.arange(n), 2)
    matrix = coo_matrix((np.tile([1.0, -1.0], n), (rows, pairs.ravel())), shape=(n, int(np.prod(shape))))
    return QuadraticConstraints(matrix, weight)


def run_gradient_cases(
    out,
    target,
    weights,
    meta,
    fibres,
    device="cpu",
    fixed=None,
    relations=(),
    stages=None,
    chart_volume=None,
    parameters=None,
):
    parameters = parameters or {}
    fibres = list(fibres) if chart_volume is not None else fibres
    pair, consensus, audit = compile_fibres(
        fibres,
        meta["origin_xyz"],
        meta["spacing_xyz"],
        weights.shape,
        voxel_mm=meta["voxel_um"] / 1000,
        pitch_mm=parameters.get("pitch_mm", 0.15),
        pair_weight=parameters.get("pair_weight", 0.1),
    )
    a, b = audit.pop("a_xyz"), audit.pop("b_xyz")
    np.savez_compressed(out / "constraint_points.npz", a_xyz=a, b_xyz=b)
    nodal = nodal_control(
        a,
        b,
        np.array(meta["origin_xyz"]),
        np.array(meta["spacing_xyz"]),
        weights.shape,
        parameters.get("pair_weight", 0.1),
    )
    consensus, normalization = match_trace(consensus, nodal.trace)
    charted_available = True
    if chart_volume is not None:
        try:
            charted_pair, charted_consensus, charted = compile_fibres(
                fibres,
                meta["origin_xyz"],
                meta["spacing_xyz"],
                weights.shape,
                voxel_mm=meta["voxel_um"] / 1000,
                pitch_mm=parameters.get("pitch_mm", 0.15),
                pair_weight=parameters.get("pair_weight", 0.1),
                chart_volume=chart_volume,
            )
            charted.pop("a_xyz")
            charted.pop("b_xyz")
            consensus, normalization = match_trace(charted_consensus, charted_pair.trace)
            audit["charted_observations"] = charted
        except ValueError as exc:
            if str(exc) != "no usable fibre observations":
                raise
            charted_available = False
            audit["charted_status"] = "UNAVAILABLE_NO_UNAMBIGUOUS_STENCILS"
    if "trace_target" in parameters:
        pair, pair_norm = match_trace(pair, parameters["trace_target"])
        consensus, consensus_norm = match_trace(consensus, parameters["trace_target"])
        nodal, nodal_norm = match_trace(nodal, parameters["trace_target"])
        audit["common_trace_matching"] = dict(pair=pair_norm, consensus=consensus_norm, nodal=nodal_norm)
    write(
        out / "constraint-audit.json",
        dict(audit, normalization=normalization, pair_digest=pair.digest, consensus_digest=consensus.digest),
    )
    pair.save(out / "pairs.npz")
    consensus.save(out / "consensus.npz")
    for name, c in (("pairs", pair), ("consensus", consensus)):
        write(
            Path(str(out / (name + ".npz")) + ".json"),
            dict(
                frame=meta["frame"],
                voxel_um=meta["voxel_um"],
                origin_xyz=meta["origin_xyz"],
                spacing_xyz=meta["spacing_xyz"],
                shape_zyx=list(weights.shape),
                step=meta["step"],
                constraint_digest=c.digest,
                file_sha256=sha256(out / (name + ".npz")),
            ),
        )
    compiled_separation = (
        compile_separation(relations, meta["origin_xyz"], meta["spacing_xyz"], weights.shape, meta["step"], weight=3)
        if relations
        else None
    )
    cases = {}
    fields = {}
    settings = [
        ("B1", nodal, None, None),
        ("B2", pair, None, None),
        ("B3", consensus, None, None),
        ("B3-R", consensus, 0.5 * meta["step"], None),
        ("B4", consensus, 0.5 * meta["step"], compiled_separation),
    ]
    for name, c, robust, separation in settings:
        if stages is not None and name not in stages:
            continue
        if not charted_available and name in ("B3", "B3-R"):
            cases[name] = dict(converged=False, status="UNAVAILABLE_NO_UNAMBIGUOUS_STENCILS")
            continue
        if not charted_available and name == "B4":
            c = None
            robust = None  # Separation-only diagnostic; never reuse uncharted consensus silently.
        started = time.time()
        print(
            "SOLVE",
            name,
            "rows",
            c.matrix.shape[0] if c is not None else 0,
            "trace",
            c.trace if c is not None else 0,
            flush=True,
        )
        field, info = solve_with_relations(
            target,
            weights,
            meta["spacing_xyz"],
            constraints=c,
            separation=separation,
            robust_delta=robust,
            outer_iterations=100 if robust is not None else 20,
            fixed=fixed,
            device=device,
            rtol=parameters.get("rtol", 1e-5 if robust is not None else 0.001),
            maxiter=parameters.get("maxiter", 4000),
        )
        # Per-sample weights may be large: saved as arrays, not millions of JSON floats.
        factors = np.asarray(info.pop("robust_factors"), np.float32)
        np.save(out / (name + "-robust-factors.npy"), factors)
        info.update(
            constraint_digest=c.digest if c is not None else None,
            separation_digest=separation.digest if separation is not None else None,
        )
        write(out / (name + "-solver.json"), info)
        cases[name] = dict(
            info,
            wall_s=time.time() - started,
            geometry_relations=len(relations),
            field_status="computed" if info["converged"] else "failed",
        )
        fields[name] = field
        if info["converged"]:
            save_field(
                out / (name + ".npy"),
                field,
                dict(
                    meta,
                    backend="signed_gradient_coherence",
                    stage=name,
                    constraint_pairs=c.matrix.shape[0] if c is not None and name in ("B1", "B2") else None,
                    constraint_observations=c.matrix.shape[0] if c is not None else 0,
                    constraint_digest=c.digest if c is not None else None,
                    constraint_trace=c.trace if c is not None else 0,
                    relation_authority="prediction_hypothesis" if relations else "none",
                ),
                info,
                meta["step"],
            )
        else:
            np.save(out / (name + "-FAILED.npy"), field)
        print("RESULT", name, info["converged"], info.get("relative_residual"), flush=True)
    return fields, cases


def synthetic_run(out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    shape = (5, 7, 9)
    z, y, x = np.indices(shape)
    target = np.zeros((*shape, 3), np.float32)
    target[..., 0] = 1
    weights = np.ones(shape, np.float32)
    fibres = [dict(fibre_id=i, points_xyz=np.array([[i, 0.2, 2.0], [i, 5.8, 2.0]])) for i in (1, 3, 5, 7)]
    meta = dict(
        frame="synthetic_parallel_sheets",
        origin_xyz=[0, 0, 0],
        spacing_xyz=[1, 1, 1],
        voxel_um=1000,
        step=1,
        shape_zyx=list(shape),
    )
    relations = [
        dict(
            a_xyz=[1, 3, 2],
            b_xyz=[2, 3, 2],
            min_difference_sheets=0.8,
            authority="controlled_synthetic_truth",
            order_authority="controlled_synthetic_truth",
        )
    ]
    fields, cases = run_gradient_cases(out, target, weights, meta, fibres, relations=relations)
    n = np.zeros_like(target)
    n[..., 0] = 1
    seeds = np.zeros(shape, bool)
    seeds[:, :, 0] = True
    arrival, info = solve_arrival(n, [1] * 3, seeds, c_parallel=0.002, c_perp=1, maxiter=100)
    np.save(out / "B5.npy", arrival)
    cases["B5"] = info
    truth = x.astype(float)
    measured = {k: float(np.max(abs(f - truth))) for k, f in fields.items()}
    measured["B5"] = float(np.max(abs(arrival - truth)))
    separation = all(abs(f[2, 3, 2] - f[2, 3, 1]) > 0.79 for f in fields.values())
    collapse = abs(np.zeros(shape)[2, 3, 2] - np.zeros(shape)[2, 3, 1]) < 0.79
    return finish(
        out,
        dict(
            cases=cases,
            synthetic_max_errors=measured,
            synthetic_separation_control=separation,
            synthetic_collapse_control_rejected=bool(collapse),
            authority="controlled_synthetic_truth_not_real_scroll_validation",
        ),
    )


def afv_records(dataset):
    from common import arclength

    return [
        dict(
            fibre_id=fid,
            length_mm=float(arclength(native_to_world(p, dataset["afv"]), dataset["voxel_um"] / 1000)[-1]),
            source_lineage=sorted(benchmark.lineage(annotation)),
        )
        for fid, p, annotation in benchmark.iter_fibres(dataset["afv"]["path"])
    ]


def real_fibres(db, train, transform, lo=None, hi=None):
    for fid, p, annotation in benchmark.iter_fibres(db):
        if train is not None and fid not in train:
            continue
        p = native_to_world(p, transform)
        if lo is not None and not ((p.max(0) >= lo).all() and (p.min(0) <= hi).all()):
            continue
        # Unprovided break/cut annotations are not invented. Geometry policy remains declared.
        yield dict(fibre_id=fid, points_xyz=p)


def make_split(dataset, out):
    db = dataset["afv"]["path"]
    supplied = artifact(dataset, "split")
    if supplied is not None:
        manifest = json.loads(supplied.read_text())
        validate_fibre_split(manifest, sha256(db))
    elif "calibration_ids" in dataset.get("context", {}):
        context = dataset["context"]
        manifest = build_fibre_split(
            afv_records(dataset), context["calibration_ids"], seed=context.get("split_seed", 0)
        )
        manifest["afv_sha256"] = sha256(db)
    else:
        return None
    write(out / "split.json", manifest)
    return manifest


def score_fields(dataset, out, fields, metas, split, scope):
    case = artifact(dataset, "evaluation_case")
    if case is None:
        return {name: dict(status="NOT_EVALUATED", reason="no evaluation case supplied") for name in fields}
    import evaluate as evaluator

    producer_path = out / "receipt.json"
    producer = json.loads(producer_path.read_text()) if producer_path.is_file() else {}
    result = {}
    for name, field in fields.items():
        path = out / (name + ".npy")
        if not path.is_file():
            result[name] = dict(status="NOT_EVALUATED", reason="candidate was not published")
            continue
        metadata_path = Path(str(path) + ".json")
        metadata = json.loads(metadata_path.read_text())
        emitted = producer.get("artifacts", {}).get(name)
        if emitted is not None and (
            sha256(path) != emitted["field_sha256"] or sha256(metadata_path) != emitted["metadata_sha256"]
        ):
            raise ValueError("candidate differs from producer receipt")
        destination = out / (name + "-evaluation")
        command = [
            "--case",
            str(case),
            "--field",
            str(path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(destination),
        ]
        if metadata.get("support") is not None:
            if not isinstance(metadata["support"], str):
                raise ValueError("producer support must name a mask artifact")
            support = path.parent / metadata["support"]
            expected = metadata.get("support_sha256", producer.get("support_sha256"))
            if producer.get("support_sha256") is not None and expected != producer["support_sha256"]:
                raise ValueError("support binding differs from producer receipt")
            command += ["--support", str(support)]
            if expected is not None:
                command += ["--support-sha256", expected]
        if evaluator.main(command) != 0:
            error = (
                json.loads((destination / "ERROR.json").read_text()) if (destination / "ERROR.json").is_file() else {}
            )
            raise ValueError(f"candidate evaluation failed: {error.get('message', destination)}")
        result[name] = json.loads((destination / "SUMMARY.json").read_text())
    return result


def observations(dataset):
    grid = dataset["grid"]
    lo, h = np.array(grid["origin_xyz"]), np.array(grid["spacing_xyz"])
    shape = tuple(grid["shape_zyx"])
    hi = lo + h * np.array(shape[::-1])
    cache = dataset.get("cache")
    if cache:
        for name in ("target", "weights", "fixed", "metadata"):
            if sha256(cache[name]["path"]) != cache[name]["sha256"]:
                raise ValueError(f"cached input changed: {name}")
        m = json.loads(Path(cache["metadata"]["path"]).read_text())
        declared = observation_recipe(dataset)
        bound = m.get("observation_config", {})
        for group, entries in declared.items():
            if not isinstance(bound.get(group), dict) or any(
                bound[group].get(name) != value for name, value in entries.items()
            ):
                raise ValueError(
                    "cache observation provenance does not match declared inputs; use a bound cache or cache-only configuration"
                )
        expected = dict(grid, frame=dataset["frame"], voxel_um=dataset["voxel_um"], step=dataset["sheet_step"])
        if any(m.get(k) != v for k, v in expected.items()):
            raise ValueError("cache frame/grid/calibration differs from dataset")
        target, weights, fixed = [
            np.load(cache[k]["path"], mmap_mode="r", allow_pickle=False) for k in ("target", "weights", "fixed")
        ]
        if target.shape != (*shape, 3) or weights.shape != shape or fixed.shape != shape:
            raise ValueError("cache array/grid mismatch")
    else:
        target, weights, _, m = load_paris_box(lo, hi, spacing=h, dataset=dataset, shape_zyx=shape)
        fixed = np.full(shape, np.nan, np.float32)
        declared = exterior_faces(dataset)
        if declared:
            reads = []
            ct = sample_source(dataset, "ct", lo, h, shape, order=0, receipts=reads)
            black = ct == black_value(dataset)
            for axis, side in declared:
                ix = [slice(None)] * 3
                ix[axis] = side
                ix = tuple(ix)
                fixed[ix] = np.where(black[ix], 0.0, fixed[ix])
            if not np.isfinite(fixed).any():
                raise ValueError("declared exterior faces have no black CT anchors")
            m.update(
                anchor_source_receipts=reads,
                exterior_authority=dataset["context"]["exterior_authority"],
                exterior_faces=dataset["context"]["exterior_faces"],
            )
        else:
            m["exterior_authority"] = "none; numerical node-zero gauge only"
    m = dict(
        m,
        step=dataset["sheet_step"],
        calibration=dataset["calibration"],
        dataset_config_sha256=dataset["config_sha256"],
    )
    return target, weights, fixed, m


def global_run(dataset, out, device, production=False, stages=None):
    mode_constraints(dataset, {"pair_weight", "pitch_mm", "trace_target"}, "global coherence")
    out.mkdir(exist_ok=False)
    afv_identity = validate_afv(dataset)
    target, weights, fixed, m = observations(dataset)
    split = make_split(dataset, out)
    train = None if split is None or production else {r["fibre_id"] for r in split["fibres"] if r["split"] == "train"}
    afv_hash = afv_identity["sha256"]
    write(out / "inputs.json", dict(dataset=dataset, afv=afv_identity, source_sha256=source_hashes()))
    m.update(
        afv_sha256=afv_hash,
        split_manifest_sha256=sha256(out / "split.json") if split is not None else None,
        production=train is None,
    )
    fields, cases = run_gradient_cases(
        out,
        target,
        weights,
        m,
        real_fibres(dataset["afv"]["path"], train, dataset["afv"]),
        device,
        fixed,
        stages=stages or ["B2"],
        parameters={**dataset.get("constraints", {}), **dataset.get("solver", {})},
    )
    metas = {k: m for k in fields}
    scores = score_fields(dataset, out, fields, metas, split, "whole_coarse")
    if train is None:
        for value in scores.values():
            value["reserved"] = None
            value["population_authority"] = "all fibres used; no held-out claim"
    return finish(
        out,
        dict(
            cases=cases,
            scores=scores,
            production=train is None,
            split_sha256=sha256(out / "split.json") if split is not None else None,
            limitation="Global stages1-3; native transversal geometry and unsigned arrival are exercised separately in the local run.",
        ),
    )


def continuous_pilot_run(dataset, out, device):
    from coarse_reference import load_external_reference, run_pilot_arrays

    controls = mode_constraints(dataset, {"pair_weight", "pitch_mm", "trace_target"}, "continuous pilot")
    out.mkdir(exist_ok=False)
    validate_afv(dataset)
    split = make_split(dataset, out)
    reference_path = artifact(dataset, "reference_field")
    if reference_path is None:
        raise ValueError("continuous-pilot requires artifacts.reference_field")
    reference_metadata = artifact(dataset, "reference_metadata") or Path(str(reference_path) + ".json")
    grid = dataset["grid"]
    lo, h = np.array(grid["origin_xyz"]), np.array(grid["spacing_xyz"])
    hi = lo + h * np.array(grid["shape_zyx"][::-1])
    target, weights, _, m = load_paris_box(lo, hi, spacing=h, dataset=dataset, shape_zyx=grid["shape_zyx"])
    _, ref_meta, identity = load_external_reference(
        reference_path,
        reference_metadata,
        m["frame"],
        lo,
        h,
        np.array(weights.shape),
        dataset["sheet_step"],
        voxel_um=dataset["voxel_um"],
    )
    train = None if split is None else {r["fibre_id"] for r in split["fibres"] if r["split"] == "train"}
    pair, _, audit = compile_fibres(
        real_fibres(dataset["afv"]["path"], train, dataset["afv"], lo, hi),
        lo,
        h,
        weights.shape,
        voxel_mm=dataset["voxel_um"] / 1000,
        pair_weight=controls.get("pair_weight", 3),
        pitch_mm=controls.get("pitch_mm", 0.15),
    )
    if "trace_target" in controls:
        pair, normalization = match_trace(pair, controls["trace_target"])
        audit["common_trace_matching"] = normalization
    a, b = audit.pop("a_xyz"), audit.pop("b_xyz")
    np.savez_compressed(out / "physical_pairs.npz", a_xyz=a, b_xyz=b)
    pair.save(out / "continuous.npz")
    write(
        out / "audit.json",
        dict(
            audit,
            constraint_digest=pair.digest,
            reference=identity,
            afv_sha256=sha256(dataset["afv"]["path"]),
            split_sha256=sha256(out / "split.json") if split is not None else None,
        ),
    )
    m.update(
        afv_sha256=sha256(dataset["afv"]["path"]),
        continuous_audit_sha256=sha256(out / "audit.json"),
        step=ref_meta["step"],
    )
    results = out / "pilot"
    results.mkdir()
    mono, assembled, _, receipt = run_pilot_arrays(
        target,
        weights,
        h,
        origin_xyz=lo,
        halo=dataset.get("context", {}).get("halo", 16),
        sweeps=dataset.get("context", {}).get("sweeps", 4),
        sheet_step=ref_meta["step"],
        device=device,
        rtol=dataset.get("solver", {}).get("rtol", 1e-5),
        maxiter=dataset.get("solver", {}).get("maxiter", 4000),
        output=results,
        metadata=m,
        external_reference=reference_path,
        external_metadata=reference_metadata,
        constraints=pair,
    )
    scores = {}
    if assembled is not None:
        sm = {
            k: dict(m, origin_xyz=lo.tolist(), spacing_xyz=h.tolist(), shape_zyx=list(weights.shape))
            for k in ("assembled", "monolithic")
        }
        scores = score_fields(dataset, results, dict(assembled=assembled, monolithic=mono), sm, split, "fine_assembled")
    return finish(
        out,
        dict(
            cases={"continuous_pilot": dict(converged=receipt["status"] == "PASS", status=receipt["status"])},
            scores=scores,
            scope="configured pilot, continuous rows and verified global reference",
        ),
    )


def arrival_boundary_run(dataset, out, device=None):
    """Arrival on the declared grid using explicit fixed/CT evidence."""
    mode_constraints(dataset, set(), "arrival-boundary")
    if "rtol" in dataset.get("solver", {}):
        raise ValueError("CG rtol is unsupported for arrival-boundary; use context.arrival_tolerance")
    if (device or dataset.get("solver", {}).get("device", "cpu")) != "cpu":
        raise ValueError("arrival-boundary currently supports only the CPU backend")
    out.mkdir(exist_ok=False)
    target, weights, fixed, m = observations(dataset)
    g = np.asarray(target)
    norm = np.linalg.norm(g, axis=-1)
    origin, h = np.array(m["origin_xyz"]), np.array(m["spacing_xyz"])
    ct = sample_source(dataset, "ct", origin, h, norm.shape, order=0)
    air = ct == black_value(dataset)
    material = (
        sample_source(dataset, "surface", origin, h, norm.shape, order=0) > dataset["prediction"]["surface_threshold"]
    )
    contradictory = air & material
    valid = ((norm > 1e-6) & ~contradictory) | (air & ~material)
    air &= ~material
    n = np.divide(g, norm[..., None], out=np.zeros_like(g), where=norm[..., None] > 1e-6)
    # Missing material directors remain unsupported; in observed air, isotropic cost makes direction immaterial.
    n[(norm <= 1e-6) & air] = [1, 0, 0]
    cp = dataset.get("context", {}).get("air_sheets_per_mm", 0.1 / 45) * dataset["voxel_um"] / 1000 * m["step"]
    cn = np.maximum(norm, cp)
    cn[air & ~material] = cp
    verified = np.isfinite(fixed)
    boundary = np.zeros(verified.shape, bool)
    for axis, side in exterior_faces(dataset):
        index = [slice(None)] * 3
        index[axis] = side
        boundary[tuple(index)] = True
    verified &= boundary
    seeds = exterior_seeds(material, air, verified_exterior=verified)
    field, info = solve_arrival(
        n,
        h,
        seeds,
        support=valid,
        c_parallel=cp,
        c_perp=cn,
        maxiter=dataset.get("solver", {}).get("maxiter", 500),
        tolerance=dataset.get("context", {}).get("arrival_tolerance", 1e-4),
    )
    np.save(out / "arrival.npy", field)
    np.save(out / "support.npy", np.isfinite(field))
    np.save(out / "seeds.npy", seeds)
    info.update(
        origin_xyz=origin.tolist(),
        spacing_xyz=h.tolist(),
        shape_zyx=list(field.shape),
        step=m["step"],
        seed_authority="explicitly declared exterior faces AND fixed anchors AND observed black CT",
        seed_mask_sha256=sha256(out / "seeds.npy"),
        dataset_config_sha256=dataset["config_sha256"],
        black_value=black_value(dataset),
        contradictory_material_black_ct_nodes=int(contradictory.sum()),
        scope="bounded exterior-boundary probe, not a whole-scroll winding field",
    )
    return finish(out, dict(cases={"B5": info}))


def local_run(dataset, out, device):
    """Geometry/separation and bounded arrival on the configured physical grid."""
    controls = mode_constraints(dataset, {"pair_weight", "pitch_mm", "trace_target"}, "local coherence")
    out.mkdir(exist_ok=False)
    validate_afv(dataset)
    split = make_split(dataset, out)
    grid = dataset["grid"]
    lo, h = np.array(grid["origin_xyz"]), np.array(grid["spacing_xyz"])
    hi = lo + h * np.array(grid["shape_zyx"][::-1])
    target, weights, pairs, m = load_paris_box(lo, hi, spacing=h, dataset=dataset, shape_zyx=grid["shape_zyx"])
    m["step"] = dataset["sheet_step"]
    p = sample_source(dataset, "surface", lo, h, weights.shape) * dataset["prediction"]["surface_scale"]
    if (p < 0).any() or (p > 1).any():
        raise ValueError("surface_scale must yield probabilities in [0,1]")
    ct = sample_source(dataset, "ct", lo, h, weights.shape, order=0)
    norms = np.linalg.norm(target, axis=-1)
    directors = np.divide(target, norms[..., None], out=np.zeros_like(target), where=norms[..., None] > 1e-6)
    directors[norms < 1e-6] = [1, 0, 0]
    from coarse_reference import load_external_reference, sample_reference

    path = artifact(dataset, "reference_field")
    if path is None:
        raise ValueError("local mode requires artifacts.reference_field")
    reference_metadata = artifact(dataset, "reference_metadata") or Path(str(path) + ".json")
    source, reference_meta, reference_identity = load_external_reference(
        path, reference_metadata, m["frame"], lo, h, np.array(weights.shape), m["step"], voxel_um=dataset["voxel_um"]
    )
    reference = sample_reference(source, reference_meta, lo, h, weights.shape)
    write(out / "reference.json", reference_identity)
    fixed = np.full(weights.shape, np.nan, np.float32)
    for axis in range(3):
        for side in (0, -1):
            ix = [slice(None)] * 3
            ix[axis] = side
            fixed[tuple(ix)] = reference[tuple(ix)]
    inventory = build_transversal_inventory(
        p,
        directors,
        lo,
        h,
        dataset["voxel_um"] / 1000,
        reference,
        max_profiles=500,
        support=norms > 1e-6,
        profile_step_mm=dataset.get("context", {}).get(
            "profile_step_mm", min(dataset["sources"]["surface"]["spacing_xyz"]) * dataset["voxel_um"] / 1000
        ),
    )
    write(out / "topology.json", inventory)
    charts, chart_info = geometry_charts(p, directors, h * dataset["voxel_um"] / 1000, observed_support=norms > 1e-6)
    np.save(out / "charts.npy", charts)
    write(out / "charts.json", chart_info)
    train = None if split is None else {r["fibre_id"] for r in split["fibres"] if r["split"] == "train"}
    fields, cases = run_gradient_cases(
        out,
        target,
        weights,
        m,
        real_fibres(dataset["afv"]["path"], train, dataset["afv"], lo, hi),
        device,
        fixed,
        inventory["relations"],
        chart_volume=charts,
        parameters={**controls, **dataset.get("solver", {})},
    )
    # Bound metric-ray storage without changing the configured physical grid.
    crop = (slice(0, dataset.get("context", {}).get("arrival_max_nodes_per_axis", 32)),) * 3
    n = directors[crop]
    material = p[crop] > 0.5
    air = ct[crop] == black_value(dataset)
    # Material predictions in black CT are contradictions, not zero-distance seeds.
    air &= ~material
    verified = np.zeros_like(air)
    for axis, side in exterior_faces(dataset):
        if side == -1 and air.shape[axis] != weights.shape[axis]:
            continue  # A truncated leading crop does not retain the parent's maximum face.
        index = [slice(None)] * 3
        index[axis] = side
        verified[tuple(index)] = True
    verified &= np.isfinite(fixed[crop])
    seeds = exterior_seeds(material, air, verified_exterior=verified)
    if not seeds.any():
        cases["B5"] = dict(converged=False, status="UNAVAILABLE_NO_OBSERVED_EXTERIOR", known_nodes=0)
    else:
        valid = (norms[crop] > 1e-6) | air
        costs = np.maximum(norms[crop], 1e-4)
        cp = dataset.get("context", {}).get("air_sheets_per_mm", 0.1 / 45) * dataset["voxel_um"] / 1000 * m["step"]
        costs[air] = cp
        arrival, info = solve_arrival(
            n,
            h,
            seeds,
            support=valid,
            c_parallel=cp,
            c_perp=costs,
            maxiter=dataset.get("solver", {}).get("maxiter", 500),
            tolerance=dataset.get("context", {}).get("arrival_tolerance", 1e-4),
        )
        np.save(out / "B5.npy", arrival)
        np.save(out / "B5-seeds.npy", seeds)
        np.save(out / "B5-support.npy", np.isfinite(arrival))
        write(out / "B5-solver.json", info)
        cases["B5"] = dict(
            info,
            origin_xyz=lo.tolist(),
            spacing_xyz=h.tolist(),
            shape_zyx=list(arrival.shape),
            step=m["step"],
            black_value=black_value(dataset),
            seed_mask_sha256=sha256(out / "B5-seeds.npy"),
            seed_authority="declared parent exterior faces retained in this crop, fixed support and observed CT",
            scope="bounded_observation_crop_not_whole_reference",
        )
    inputs = {
        name: array
        for name, array in dict(target=target, weights=weights, probability=p, ct=ct, directors=directors).items()
    }
    for name, array in inputs.items():
        np.save(out / (name + "-input.npy"), array)
    return finish(
        out,
        dict(
            cases=cases,
            charts=chart_info,
            topology=dict(accepted=len(inventory["relations"]), ambiguous=inventory["ambiguous_profiles"]),
            limitation="Conservative prediction relations, not verified laminae; bounded arrival support is explicitly retained.",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("synthetic", "global", "local", "arrival-boundary", "continuous-pilot"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, help="explicit dataset JSON; required for real-data modes")
    p.add_argument("--device", help="override configured solver device (otherwise CPU)")
    p.add_argument("--production", action="store_true")
    p.add_argument("--stages", nargs="+", choices=("B1", "B2", "B3", "B3-R", "B4"))
    a = p.parse_args()
    if a.mode == "synthetic":
        result = synthetic_run(a.output)
    else:
        if a.config is None:
            p.error("real-data modes require --config; no dataset is selected implicitly")
        dataset = load_dataset(a.config)
        device = a.device or dataset.get("solver", {}).get("device", "cpu")
        if a.mode == "global":
            result = global_run(dataset, a.output, device, a.production, a.stages)
        elif a.mode == "arrival-boundary":
            result = arrival_boundary_run(dataset, a.output, device)
        elif a.mode == "continuous-pilot":
            result = continuous_pilot_run(dataset, a.output, device)
        else:
            result = local_run(dataset, a.output, device)
    print("DONE", json.dumps({k: v.get("converged") for k, v in result["cases"].items()}), flush=True)
    return 0 if all(case.get("converged") is True for case in result["cases"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
