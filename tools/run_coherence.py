#!/usr/bin/env python3
"""Run approved coherence features on synthetic truth or hash-bound real data.

synthetic: all five stages, analytic truth/negative controls.
global: balanced split and full-domain continuous/consensus ablations on fixed cache.
local: native geometry/separation and unsigned arrival, bounded observation crop.
"""

import argparse
import gzip
import json
import sqlite3
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
from evaluation_split import build_fibre_split, validate_fibre_split
from field_constraints import QuadraticConstraints, solve_with_relations
from field_topology import build_transversal_inventory, geometry_charts
from paris_data import CTB, MB, load_paris_box, save_field
from patch_check import sample_field, sha256
from physical_grid import sample_volume


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_hashes():
    return {
        name: sha256(BASE / "src" / name)
        for name in ("coherence_pipeline.py", "field_constraints.py", "field_solver.py", "evaluation_split.py")
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
    out, target, weights, meta, fibres, device="cpu", fixed=None, relations=(), stages=None, chart_volume=None
):
    fibres = list(fibres) if chart_volume is not None else fibres
    pair, consensus, audit = compile_fibres(
        fibres,
        meta["origin_xyz"],
        meta["spacing_xyz"],
        weights.shape,
        voxel_mm=meta["voxel_um"] / 1000,
        pitch_mm=0.15,
        pair_weight=0.1,
    )
    a, b = audit.pop("a_xyz"), audit.pop("b_xyz")
    np.savez_compressed(out / "constraint_points.npz", a_xyz=a, b_xyz=b)
    nodal = nodal_control(a, b, np.array(meta["origin_xyz"]), np.array(meta["spacing_xyz"]), weights.shape, 0.1)
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
                pitch_mm=0.15,
                pair_weight=0.1,
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
            rtol=1e-5 if robust is not None else 0.001,
            maxiter=4000,
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


def afv_records(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [
            dict(
                fibre_id=int(fid),
                length_mm=float(length) * 0.0024,
                source_lineage=sorted(benchmark.lineage(json.loads(annotation))),
            )
            for fid, length, annotation in con.execute("SELECT id,length,annotation FROM fibers")
        ]
    finally:
        con.close()


def real_fibres(db, train, lo=None, hi=None):
    for fid, p, annotation in benchmark.iter_fibres(db):
        if fid not in train:
            continue
        if lo is not None and not (((p.max(0) / 4) >= lo).all() and ((p.min(0) / 4) <= hi).all()):
            continue
        # Unprovided break/cut annotations are not invented. Geometry policy remains declared.
        yield dict(fibre_id=fid, points_xyz=p / 4)


def make_split(root, out):
    db = root / "Paris4_v1.afv"
    forced = set()
    for p in (root / "disclosure-20261011/baseline-audit").glob("slab*_constraints.npz"):
        with np.load(p, allow_pickle=False) as f:
            forced.update(map(int, f["fiber_ids"]))
    if not forced:
        raise ValueError("calibration constraint antecedents unavailable")
    manifest = build_fibre_split(afv_records(db), forced)
    manifest["afv_sha256"] = sha256(db)
    write(out / "split.json", manifest)
    return manifest


def score_fields(root, out, fields, metas, split, scope):
    r = root / "fibre-benchmark-20261012-all/results"
    with gzip.open(r / "selected-pairs.json.gz", "rt") as f:
        selected = json.load(f)[scope]
    reserved = {r["fibre_id"] for r in split["fibres"] if r["split"] == "reserved"}
    source = json.loads((root / "disclosure-20261011/holdout-predictions-v3/test_results.json").read_text())["methods"][
        "whole_coarse_secondary"
    ]["rows"]
    result = {}
    for name, field in fields.items():
        m = metas[name]
        rows = benchmark.evaluate(field, m, selected)
        values = np.array([row["error"] if row["error"] is not None else np.nan for row in rows])
        np.save(out / (name + "-errors.npy"), values)
        human = sample_field(field, np.array([r["a_xyz"] for r in source]), m["origin_xyz"], m["spacing_xyz"])
        human_b = sample_field(field, np.array([r["b_xyz"] for r in source]), m["origin_xyz"], m["spacing_xyz"])
        known = np.isfinite(human) & np.isfinite(human_b)
        prediction = abs(human - human_b) / m["step"]
        truth = np.array([r["dw"] for r in source])
        result[name] = dict(
            all=benchmark.summarize(rows),
            reserved=benchmark.summarize([r for r in rows if r["fiber_id"] in reserved]),
            selection_sha256=sha256(r / "selected-pairs.json.gz"),
            human=dict(
                covered=int(known.sum()),
                eligible=len(source),
                exact=int(np.sum(known & (np.rint(prediction) == truth))),
                mae=float(np.mean(abs(prediction[known] - truth[known]))) if known.any() else None,
            ),
        )
    return result


def global_run(root, cache, out, device, production=False, stages=None):
    out.mkdir(exist_ok=False)
    split = make_split(root, out)
    target = np.load(cache / "target.npy", allow_pickle=False)
    weights = np.load(cache / "weights.npy", allow_pickle=False)
    fixed = np.load(cache / "fixed.npy", allow_pickle=False)
    frozen_plan = json.loads((cache.parent / "PLAN_FROZEN.json").read_text())
    m = validate_cached_reference(
        root / "disclosure-20261011/whole-reference-v3/coarse.npy", frozen_plan, weights.shape
    )
    if target.shape != (*weights.shape, 3) or fixed.shape != weights.shape:
        raise ValueError("cache array/grid mismatch")
    declared = frozen_plan["input_sha256"]
    for name in ("target.npy", "weights.npy", "fixed.npy"):
        if sha256(cache / name) != declared[name]:
            raise ValueError("cached input changed")
    write(
        out / "inputs.json",
        dict(
            cache={k: declared[k] for k in ("target.npy", "weights.npy", "fixed.npy")},
            source_sha256=source_hashes(),
        ),
    )
    train = {r["fibre_id"] for r in split["fibres"] if production or r["split"] == "train"}
    m.update(
        afv_sha256=sha256(root / "Paris4_v1.afv"),
        split_manifest_sha256=sha256(out / "split.json"),
        production=production,
    )
    fields, cases = run_gradient_cases(
        out, target, weights, m, real_fibres(root / "Paris4_v1.afv", train), device, fixed, stages=stages
    )
    metas = {k: m for k in fields}
    scores = score_fields(root, out, fields, metas, split, "whole_coarse")
    if production:
        for value in scores.values():
            value["reserved"] = None
            value["population_authority"] = "all fibres used; no held-out claim"
    return finish(
        out,
        dict(
            cases=cases,
            scores=scores,
            production=production,
            split_sha256=sha256(out / "split.json"),
            limitation="Global stages1-3; native transversal geometry and unsigned arrival are exercised separately in the local run.",
        ),
    )


def continuous_pilot_run(root, split_path, reference_path, out, device):
    from coarse_reference import load_external_reference, run_pilot_arrays

    out.mkdir(exist_ok=False)
    split = json.loads(split_path.read_text())
    validate_fibre_split(split, sha256(root / "Paris4_v1.afv"))
    lo = np.array([4088, 2500, 10096])
    hi = lo + 512
    target, weights, _, m = load_paris_box(lo, hi, spacing=2, afv_path=None, data_root=str(root))
    _, ref_meta, identity = load_external_reference(
        reference_path, Path(str(reference_path) + ".json"), m["frame"], lo, np.array([2] * 3), np.array(weights.shape)
    )
    train = {r["fibre_id"] for r in split["fibres"] if r["split"] == "train"}
    pair, _, audit = compile_fibres(
        real_fibres(root / "Paris4_v1.afv", train, lo, hi), lo, np.array([2] * 3), weights.shape, pair_weight=3
    )
    a, b = audit.pop("a_xyz"), audit.pop("b_xyz")
    np.savez_compressed(out / "physical_pairs.npz", a_xyz=a, b_xyz=b)
    pair.save(out / "continuous.npz")
    write(
        out / "audit.json",
        dict(
            audit,
            constraint_digest=pair.digest,
            reference=identity,
            afv_sha256=split["afv_sha256"],
            split_sha256=sha256(split_path),
        ),
    )
    m.update(afv_sha256=split["afv_sha256"], continuous_audit_sha256=sha256(out / "audit.json"), step=ref_meta["step"])
    results = out / "pilot"
    results.mkdir()
    mono, assembled, _, receipt = run_pilot_arrays(
        target,
        weights,
        2,
        origin_xyz=lo,
        halo=16,
        sweeps=4,
        sheet_step=ref_meta["step"],
        device=device,
        output=results,
        metadata=m,
        external_reference=reference_path,
        external_metadata=Path(str(reference_path) + ".json"),
        constraints=pair,
    )
    scores = {}
    if assembled is not None:
        sm = {
            k: dict(m, origin_xyz=lo.tolist(), spacing_xyz=[2] * 3, shape_zyx=list(weights.shape))
            for k in ("assembled", "monolithic")
        }
        scores = score_fields(root, out, dict(assembled=assembled, monolithic=mono), sm, split, "fine_assembled")
    return finish(
        out,
        dict(
            cases={"continuous_pilot": dict(converged=receipt["status"] == "PASS", status=receipt["status"])},
            scores=scores,
            scope="same 4.9mm pilot, continuous rows and candidate global reference",
        ),
    )


def arrival_boundary_run(root, cache, out):
    """Measured coarse crop TOUCHING original X=0 anchors, never interior-rim seeds."""
    import zarr

    out.mkdir(exist_ok=False)
    plan = json.loads((cache.parent / "PLAN_FROZEN.json").read_text())
    weights = np.load(cache / "weights.npy", mmap_mode="r", allow_pickle=False)
    m = validate_cached_reference(root / "disclosure-20261011/whole-reference-v3/coarse.npy", plan, weights.shape)
    if sha256(cache / "target.npy") != plan["input_sha256"]["target.npy"]:
        raise ValueError("director input cache changed")
    target = np.load(cache / "target.npy", mmap_mode="r", allow_pickle=False)
    z = weights.shape[0] // 2
    y = weights.shape[1] // 2
    hits = np.flatnonzero((weights[z, y - 8 : y + 8, :] > 0.5).any(0))
    if not len(hits):
        raise ValueError("no material in declared lateral probe")
    nx = min(weights.shape[2], int(hits[0]) + 12)
    region = (slice(z - 8, z + 8), slice(y - 8, y + 8), slice(0, nx))
    g = np.asarray(target[region])
    norm = np.linalg.norm(g, axis=-1)
    origin = np.array([0, y - 8, z - 8]) * 32
    ct = sample_volume(zarr.open(CTB + "/5", mode="r"), origin, [32] * 3, norm.shape, [8] * 3)
    fixed = np.load(cache / "fixed.npy", mmap_mode="r", allow_pickle=False)[region]
    air = ct == 0
    material = np.asarray(weights[region]) > 0.5
    contradictory = air & material
    valid = ((norm > 1e-6) & ~contradictory) | (air & ~material)
    air &= ~material
    n = np.divide(g, norm[..., None], out=np.zeros_like(g), where=norm[..., None] > 1e-6)
    # Missing material directors remain unsupported; in observed air, isotropic cost makes direction immaterial.
    n[(norm <= 1e-6) & air] = [1, 0, 0]
    cp = 0.1 / 45 * 0.0096 * m["step"]
    cn = np.maximum(norm, cp)
    cn[air & ~material] = cp
    verified = np.isfinite(fixed)
    verified[:, :, 1:] = False  # Only the inherited original X=0 face.
    seeds = exterior_seeds(material, air, verified_exterior=verified)
    field, info = solve_arrival(
        n, [32] * 3, seeds, support=valid, c_parallel=cp, c_perp=cn, maxiter=500, tolerance=1e-4
    )
    np.save(out / "arrival.npy", field)
    np.save(out / "support.npy", np.isfinite(field))
    np.save(out / "seeds.npy", seeds)
    info.update(
        origin_xyz=origin.tolist(),
        spacing_xyz=[32] * 3,
        shape_zyx=list(field.shape),
        step=m["step"],
        seed_authority="hash-bound original black exterior X=0 anchors AND current observed black CT",
        seed_mask_sha256=sha256(out / "seeds.npy"),
        fixed_cache_sha256=plan["input_sha256"]["fixed.npy"],
        contradictory_material_black_ct_nodes=int(contradictory.sum()),
        scope="bounded exterior-boundary probe, not a whole-scroll winding field",
    )
    return finish(out, dict(cases={"B5": info}))


def local_run(root, split_path, out, device):
    """Real 19.2um geometry at the existing pilot; arrival on a bounded subgrid."""
    import zarr

    out.mkdir(exist_ok=False)
    split = json.loads(split_path.read_text())
    validate_fibre_split(split, sha256(root / "Paris4_v1.afv"))
    lo = np.array([4088, 2500, 10096])
    hi = lo + 128  # Bounded1.23mm geometry crop, same pilot center provenance.
    target, weights, pairs, m = load_paris_box(lo, hi, spacing=2, afv_path=None, data_root=str(root))
    m["step"] = 1.192497988541921
    p = sample_volume(zarr.open(MB + "/0", mode="r"), lo, [2] * 3, weights.shape, [1] * 3) / 255
    ct = sample_volume(zarr.open(CTB + "/2", mode="r"), lo, [2] * 3, weights.shape, [1] * 3)
    norms = np.linalg.norm(target, axis=-1)
    directors = np.divide(target, norms[..., None], out=np.zeros_like(target), where=norms[..., None] > 1e-6)
    directors[norms < 1e-6] = [1, 0, 0]
    from coarse_reference import load_external_reference

    path = root / "disclosure-20261011/whole-reference-v3/coarse.npy"
    source, reference_meta, reference_identity = load_external_reference(
        path, Path(str(path) + ".json"), m["frame"], lo, np.array([2] * 3), np.array(weights.shape), m["step"]
    )
    reference = sample_volume(
        source, lo, [2] * 3, weights.shape, reference_meta["spacing_xyz"], reference_meta["origin_xyz"]
    )
    write(out / "reference.json", reference_identity)
    fixed = np.full(weights.shape, np.nan, np.float32)
    for axis in range(3):
        for side in (0, -1):
            ix = [slice(None)] * 3
            ix[axis] = side
            fixed[tuple(ix)] = reference[tuple(ix)]
    inventory = build_transversal_inventory(
        p, directors, lo, np.array([2] * 3), 0.0096, reference, max_profiles=500, support=norms > 1e-6
    )
    write(out / "topology.json", inventory)
    charts, chart_info = geometry_charts(p, directors, np.array([2] * 3) * 0.0096, observed_support=norms > 1e-6)
    np.save(out / "charts.npy", charts)
    write(out / "charts.json", chart_info)
    train = {r["fibre_id"] for r in split["fibres"] if r["split"] == "train"}
    fields, cases = run_gradient_cases(
        out,
        target,
        weights,
        m,
        real_fibres(root / "Paris4_v1.afv", train, lo, hi),
        device,
        fixed,
        inventory["relations"],
        chart_volume=charts,
    )
    # Arrival crop32^3 keeps metric-ray storage bounded; preserves the physical2-voxel grid.
    crop = (slice(0, 32),) * 3
    n = directors[crop]
    material = p[crop] > 0.5
    air = ct[crop] == 0
    # Material predictions in black CT are contradictions, not zero-distance seeds.
    air &= ~material
    # This is an interior crop. Its boundary is not the scroll exterior; no unverified seeds.
    seeds = exterior_seeds(material, air, verified_exterior=np.zeros_like(air))
    if not seeds.any():
        cases["B5"] = dict(converged=False, status="UNAVAILABLE_NO_OBSERVED_EXTERIOR", known_nodes=0)
    else:
        valid = (norms[crop] > 1e-6) | air
        costs = np.maximum(norms[crop], 1e-4)
        costs[air] = 0.1 / 45 * 0.0096 * m["step"]
        arrival, info = solve_arrival(
            n,
            [2] * 3,
            seeds,
            support=valid,
            c_parallel=0.1 / 45 * 0.0096 * m["step"],
            c_perp=costs,
            maxiter=500,
            tolerance=1e-4,
        )
        np.save(out / "B5.npy", arrival)
        np.save(out / "B5-support.npy", np.isfinite(arrival))
        write(out / "B5-solver.json", info)
        cases["B5"] = dict(
            info,
            origin_xyz=lo.tolist(),
            spacing_xyz=[2] * 3,
            shape_zyx=list(arrival.shape),
            step=m["step"],
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
    p.add_argument("--root", type=Path, default=Path("/root/wrev"))
    p.add_argument("--cache", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference", type=Path)
    p.add_argument("--production", action="store_true")
    p.add_argument("--stages", nargs="+", choices=("B1", "B2", "B3", "B3-R", "B4"))
    a = p.parse_args()
    if a.mode == "synthetic":
        result = synthetic_run(a.output)
    elif a.mode in ("global", "arrival-boundary"):
        if a.cache is None:
            p.error("global requires frozen cache")
        if a.mode == "global":
            result = global_run(a.root, a.cache, a.output, a.device, a.production, a.stages)
        else:
            result = arrival_boundary_run(a.root, a.cache, a.output)
    else:
        if a.split is None:
            p.error("local requires identity-bound split")
        if a.mode == "continuous-pilot":
            if a.reference is None:
                p.error("continuous-pilot requires a verified reference")
            result = continuous_pilot_run(a.root, a.split, a.reference, a.output, a.device)
        else:
            result = local_run(a.root, a.split, a.output, a.device)
    print("DONE", json.dumps({k: v.get("converged") for k, v in result["cases"].items()}), flush=True)


if __name__ == "__main__":
    main()
