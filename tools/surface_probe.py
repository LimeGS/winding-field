#!/usr/bin/env python3
"""CPU-only frozen prior-seed product probe; human patches are evaluation only.

Run through run_watch.py. --work and --output must be new directories.
--reuse-inputs OLD_WORK reuses hash-verified frozen seed/CT after an adapter fix;
it does not select another seed, level, or physical chart.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from surface_export import _metadata, _seed_distances, _sha, mesh_geometry

RULE = (
    "central Z index=nz//2; prior>0; exclude 16 grid voxels at every boundary; "
    "nearest positive prior pixel to XY center ((ny-1)/2,(nx-1)/2); "
    "row-major tie break; level=assembled raw field at selected grid sample; no GT"
)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def choose_seed(field, metadata, prior_plane, margin=16, support=None):
    origin, spacing, _ = _metadata(metadata, field.shape)
    z = field.shape[0] // 2
    ny, nx = field.shape[1:]
    prior = np.asarray(prior_plane)
    if prior.shape != (ny, nx) or margin < 0 or z < margin or z >= field.shape[0] - margin:
        raise ValueError("invalid prior plane or interior margin")
    yy, xx = np.indices((ny, nx))
    eligible = (
        (prior > 0)
        & np.isfinite(prior)
        & np.isfinite(field[z])
        & (yy >= margin)
        & (yy < ny - margin)
        & (xx >= margin)
        & (xx < nx - margin)
    )
    if support is not None:
        if support.shape != field.shape or support.dtype != np.bool_:
            raise ValueError("invalid field support")
        eligible &= support[z]
    if not eligible.any():
        raise ValueError("no positive prior pixel within frozen interior")
    distance = (yy - (ny - 1) / 2) ** 2 + (xx - (nx - 1) / 2) ** 2
    y, x = np.unravel_index(np.argmin(np.where(eligible, distance, np.inf)), prior.shape)
    return dict(
        rule=RULE,
        margin_grid_voxels=margin,
        eligible_pixels=int(eligible.sum()),
        seed_index_zyx=[int(z), int(y), int(x)],
        seed_xyz=(origin + np.array([x, y, z]) * spacing).tolist(),
        prior_value=float(prior[y, x]),
        level=float(field[z, y, x]),
        human_labels_used=False,
        frozen_utc=datetime.now(UTC).isoformat(),
    )


def owners_at(points, metadata, ownership):
    origin, spacing, _ = _metadata(metadata, ownership.shape)
    q = (np.asarray(points) - origin) / spacing
    inside = np.isfinite(q).all(1) & (q >= 0).all(1) & (q <= np.array(ownership.shape[::-1]) - 1).all(1)
    result = np.full(len(q), -1, dtype=np.int64)
    indices = np.rint(q[inside]).astype(int)
    result[inside] = ownership[tuple(indices[:, ::-1].T)]
    return result


def seam_registration(vertices, faces, metadata, ownership):
    owner = owners_at(vertices, metadata, ownership)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    pairs = np.sort(owner[edges], axis=1)
    crossing = (pairs[:, 0] >= 0) & (pairs[:, 1] != pairs[:, 0])
    unique, counts = np.unique(pairs[crossing], axis=0, return_counts=True)
    return dict(
        tile_vertex_counts={str(int(i)): int((owner == i).sum()) for i in np.unique(owner) if i >= 0},
        unowned_vertices=int((owner < 0).sum()),
        seam_edge_counts={f"{a}:{b}": int(n) for (a, b), n in zip(unique, counts)},
        crosses_core_seams=bool(crossing.any()),
        measurement="nearest field-grid ownership; mesh edge transitions",
        physical_sheet_identity="UNMEASURED",
        self_intersection="UNMEASURED",
    )


def summary(values):
    values = np.asarray(values)
    return dict(
        n=len(values),
        **({f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)} if len(values) else {}),
        max=float(values.max()) if len(values) else None,
    )


def exact_point_to_mesh(points, vertices, faces, max_candidates=10_000_000):
    """Exact point/triangle minimum over a provably sufficient sphere-bound set."""
    points, vertices, faces = np.asarray(points, float), np.asarray(vertices, float), np.asarray(faces)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or mesh_geometry(vertices, faces)["basic_status"] != "PASS"
    ):
        raise ValueError("invalid point/triangle distance input")
    triangles = vertices[faces]
    centers = triangles.mean(1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(1)
    tree = cKDTree(centers)
    upper, _ = cKDTree(vertices[np.unique(faces)]).query(points)
    distances, nearest = np.empty(len(points)), np.empty(len(points), dtype=np.int64)
    count, radius = 0, float(radii.max())
    for i, point in enumerate(points):
        tolerance = 1e-9 * max(1, float(upper[i]), radius)
        ids = np.asarray(tree.query_ball_point(point, upper[i] + radius + tolerance), dtype=np.int64)
        ids = ids[np.linalg.norm(centers[ids] - point, axis=1) - radii[ids] <= upper[i] + tolerance]
        count += len(ids)
        if count > max_candidates:
            raise ValueError("exact distance candidate budget exceeded")
        if not len(ids):
            raise ValueError("empty certified distance candidate set")
        d = _seed_distances(point, triangles[ids])
        winner = int(d.argmin())
        distances[i], nearest[i] = d[winner], ids[winner]
    return (
        distances,
        nearest,
        dict(
            complete=True,
            candidate_pairs=count,
            method="exact_point_triangle_minimum_over_centroid_radius_bound",
            bound="distance(point,centroid)-triangle_radius <= nearest_used_vertex_distance",
            scope="piecewise-linear artifact surface; floating-point arithmetic",
        ),
    )


def evaluate_patch(patch, vertices, faces, metadata, ownership, *, exclude_uv_rc=None, exact=False):
    import tifffile

    xyz = np.stack([tifffile.imread(patch / f"{axis}.tif") for axis in "xyz"], axis=-1).astype(float)
    origin, spacing, um = _metadata(metadata, metadata["shape_zyx"])
    valid = np.isfinite(xyz).all(-1) & (xyz >= 0).all(-1)
    grid = (xyz - origin) / spacing
    covered = valid & (grid >= 16).all(-1) & (grid <= np.array(metadata["shape_zyx"][::-1]) - 1 - 16).all(-1)
    excluded = np.zeros(valid.shape, bool)
    if exclude_uv_rc is not None:
        rr, cc = np.indices(valid.shape)
        excluded = (np.abs(rr - exclude_uv_rc[0]) <= 2) & (np.abs(cc - exclude_uv_rc[1]) <= 2)
    excluded_covered = int((covered & excluded).sum())
    covered &= ~excluded
    points = xyz[covered]
    result = dict(
        patch=str(patch),
        frame=metadata["frame"],
        evaluated_after_mesh_frozen=True,
        valid_human_points=int(valid.sum()),
        covered_human_points=len(points),
        coverage_fraction=float(covered.sum() / max((valid & ~excluded).sum(), 1)),
        margin_grid_voxels=16,
        excluded_seed_neighborhood_points=excluded_covered,
        exclusion_uv_radius=2 if exclude_uv_rc is not None else None,
        validation_valid_points=int((valid & ~excluded).sum()),
        patch_sha256={f: _sha(patch / f) for f in ("meta.json", "x.tif", "y.tif", "z.tif")},
        distance_kind="nearest vertex distances are UPPER BOUNDS, not exact point-to-triangle distances",
        normal_kind="nearest mesh vertex normal absolute cosine; no writing-face claim",
    )
    if not len(points):
        return {**result, "status": "NO_COVERAGE"}
    distance, nearest = cKDTree(vertices).query(points)
    result["human_to_mesh_vertex_upper_bound_um"] = summary(distance * um)
    result["within_distance_um"] = {
        str(threshold): int((distance * um <= threshold).sum()) for threshold in (19.2, 48, 96)
    }
    in_bbox = ((vertices >= points.min(0)) & (vertices <= points.max(0))).all(1)
    back, _ = cKDTree(points).query(vertices[in_bbox])
    result["mesh_to_human_vertex_upper_bound_um"] = summary(back * um)
    result["reverse_scope"] = "mesh vertices in covered human XYZ bbox; not human UV coverage"
    result["mesh_vertices_in_human_bbox"] = int(in_bbox.sum())
    owners = owners_at(points, metadata, ownership)
    result["human_by_tile"] = {str(int(i)): summary(distance[owners == i] * um) for i in np.unique(owners)}
    if exact:
        try:
            d, _, measure = exact_point_to_mesh(points, vertices, faces)
            result["human_to_mesh_surface_distance_um"] = summary(d * um)
            result["human_to_mesh_surface_distance_receipt"] = measure
            result["surface_distance_by_tile_um"] = {
                str(int(i)): summary(d[owners == i] * um) for i in np.unique(owners)
            }
            result["surface_points_within_um"] = {str(t): int((d * um <= t).sum()) for t in (19.2, 48, 96)}
            idx = np.arange(valid.size).reshape(valid.shape)
            a, b, c, d0 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
            hf = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d0, c], -1).reshape(-1, 3)])
            hf = hf[covered.ravel()[hf].all(1)]
            hp = xyz.reshape(-1, 3)
            lengths = np.linalg.norm(hp[hf] - np.roll(hp[hf], 1, axis=1), axis=2)
            hf = hf[(lengths <= 100).all(1)]
            reverse, _, measure = exact_point_to_mesh(
                vertices[in_bbox], hp[np.unique(hf)], np.unique(hf, return_inverse=True)[1].reshape(-1, 3)
            )
            result["mesh_to_human_surface_distance_um"] = summary(reverse * um)
            result["mesh_to_human_surface_distance_receipt"] = measure
            result["surface_distance_status"] = "MEASURED"
            result["reverse_surface_scope"] = (
                "covered human triangles excluding seed neighborhood; XYZ-bbox mesh queries"
            )
        except ValueError as error:
            result["surface_distance_status"] = "UNMEASURED_OR_PARTIAL"
            result["surface_distance_error"] = str(error)
    dr, dc = np.zeros_like(xyz), np.zeros_like(xyz)
    dr[1:-1] = xyz[2:] - xyz[:-2]
    dc[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    stencil = covered.copy()
    stencil[1:-1] &= valid[2:] & valid[:-2]
    stencil[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    stencil[[0, -1]] = False
    stencil[:, [0, -1]] = False
    n = np.cross(dr, dc)
    norm = np.linalg.norm(n, axis=-1)
    stencil &= (norm > 1e-12) & (np.linalg.norm(dr, axis=-1) <= 200) & (np.linalg.norm(dc, axis=-1) <= 200)
    tri = vertices[faces]
    mesh_n = np.zeros_like(vertices)
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for col in range(3):
        np.add.at(mesh_n, faces[:, col], raw)
    mesh_n /= np.maximum(np.linalg.norm(mesh_n, axis=1)[:, None], 1e-12)
    nn = nearest[stencil[covered]]
    human_n = n[stencil] / norm[stencil, None]
    result["normal_absolute_cosine"] = summary(np.abs(np.sum(human_n * mesh_n[nn], axis=1)))
    result["status"] = "MEASURED_NOT_CERTIFIED"
    return result


def context_image(path, ct, ct_meta, selection, vertices):
    from PIL import Image, ImageDraw, PngImagePlugin

    origin = np.array(ct_meta["origin_xyz"])
    seed = np.array(selection["seed_xyz"])
    z = int(round(seed[2] - origin[2]))
    plane_z = float(origin[2] + z)
    image = ct[z].astype(float)
    lo, hi = np.percentile(image, [1, 99])
    gray = np.rint(255 * np.clip((image - lo) / max(hi - lo, 1), 0, 1)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    nearby = vertices[np.abs(vertices[:, 2] - plane_z) <= 2]
    xy = np.rint(nearby[:, :2] - origin[:2]).astype(int)
    inside = ((xy >= 0) & (xy < np.array(image.shape[::-1]))).all(1)
    rgb[xy[inside, 1], xy[inside, 0]] = [255, 60, 60]
    out = Image.fromarray(rgb)
    draw = ImageDraw.Draw(out)
    x, y = seed[:2] - origin[:2]
    draw.line((x - 6, y, x + 6, y), fill="cyan", width=1)
    draw.line((x, y - 6, x, y + 6), fill="cyan", width=1)
    draw.text((5, 5), f"CT /2 z={plane_z:g} L2; mesh +/-2 L2; NOT INK", fill="yellow")
    png = PngImagePlugin.PngInfo()
    kind = "human seed" if selection.get("human_seed") else "prior seed"
    png.add_text(
        "Description",
        f"Actual CT /2 plane; red mesh vertices within 2 L2 units; cyan frozen {kind} projected to plane; not ink",
    )
    png.add_text("CT_z_L2", str(plane_z))
    out.save(path, pnginfo=png)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--app-run", type=Path, required=True)
    p.add_argument("--patch", type=Path, required=True)
    p.add_argument("--reuse-inputs", type=Path)
    args = p.parse_args(argv)
    args.work.mkdir(exist_ok=False)
    result = dict(
        status="FAILED",
        source_sha256=_sha(__file__),
        exporter_sha256=_sha(Path(__file__).with_name("surface_export.py")),
    )
    code = 2
    try:
        meta = json.loads(args.metadata.read_text())
        field = np.load(args.field, mmap_mode="r", allow_pickle=False)
        origin, spacing, um = _metadata(meta, field.shape)
        if meta["frame"] != "PHercParis4/20260411134726:L2" or um != 9.6:
            raise ValueError("this probe is frozen to the selected Paris4 L2 frame")
        field_hash = _sha(args.field)
        if field_hash != meta["field_sha256"]:
            raise ValueError("assembled field hash differs from its metadata")
        support_path = args.field.parent / meta["support"]
        support = np.load(support_path, allow_pickle=False)
        if args.reuse_inputs:
            inputs = json.loads((args.reuse_inputs / "inputs.json").read_text())
            if inputs["field_sha256"] != field_hash or inputs["metadata_sha256"] != _sha(args.metadata):
                raise ValueError("reuse field/metadata mismatch")
            for key in ("ct", "ct_metadata", "seed"):
                if _sha(inputs[key]) != inputs[key + "_sha256"]:
                    raise ValueError("reused input hash mismatch: " + key)
            selection = json.loads(Path(inputs["seed"]).read_text())
            if selection["rule"] != RULE:
                raise ValueError("reused seed rule differs")
            write_json(args.work / "seed.json", selection)
            inputs["reused_from"] = str(args.reuse_inputs)
        else:
            import zarr

            from paris_data import CTB, MB
            from physical_grid import sample_volume

            write_json(
                args.work / "selection_rule.json",
                dict(rule=RULE, field_sha256=field_hash, metadata_sha256=_sha(args.metadata)),
            )
            plane_origin = origin.copy()
            plane_origin[2] += (field.shape[0] // 2) * spacing[2]
            print("READ prior central plane", flush=True)
            read_receipts = []
            prior = sample_volume(
                zarr.open(MB + "/0", mode="r"),
                plane_origin,
                spacing,
                (1,) + field.shape[1:],
                [1, 1, 1],
                order=0,
                receipts=read_receipts,
            )[0]
            np.save(args.work / "prior_plane.npy", prior)
            selection = choose_seed(field, meta, prior, support=support)
            selection.update(
                field_sha256=field_hash,
                prior_source=MB + "/0",
                prior_read=read_receipts,
                prior_array_sha256=_sha(args.work / "prior_plane.npy"),
            )
            write_json(args.work / "seed.json", selection)
            print("FROZEN SEED", json.dumps(selection), flush=True)
            ct_lo = origin - 16
            ct_hi = origin + np.array(field.shape[::-1]) * spacing + 16
            if not np.equal(ct_lo, np.floor(ct_lo)).all() or not np.equal(ct_hi, np.floor(ct_hi)).all():
                raise ValueError("CT /2 crop bounds must be integer L2 coordinates")
            print("READ CT /2", ct_lo.tolist(), ct_hi.tolist(), flush=True)
            volume = zarr.open(CTB + "/2", mode="r")
            lo, hi = ct_lo.astype(int)[::-1], ct_hi.astype(int)[::-1]
            if (lo < 0).any() or (hi > np.array(volume.shape)).any() or str(volume.dtype) != "uint8":
                raise ValueError("CT source bounds/dtype mismatch")
            ct = np.asarray(volume[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))])
            if tuple(ct.shape) != tuple(hi - lo):
                raise ValueError("CT crop truncated")
            np.save(args.work / "ct.npy", ct)
            ct_meta = dict(
                frame=meta["frame"],
                origin_xyz=ct_lo.tolist(),
                spacing_xyz=[1, 1, 1],
                voxel_um=9.6,
                shape_zyx=list(ct.shape),
                source=CTB + "/2",
                source_spacing_xyz=[1, 1, 1],
                source_shape_zyx=list(volume.shape),
                source_chunks=list(volume.chunks),
                dtype=str(ct.dtype),
                published_missing_chunk_coverage="NOT_INDEPENDENTLY_CERTIFIED",
            )
            write_json(args.work / "ct_metadata.json", ct_meta)
            inputs = dict(
                field_sha256=field_hash,
                metadata_sha256=_sha(args.metadata),
                ct=str((args.work / "ct.npy").resolve()),
                ct_metadata=str((args.work / "ct_metadata.json").resolve()),
                seed=str((args.work / "seed.json").resolve()),
            )
            for key in ("ct", "ct_metadata", "seed"):
                inputs[key + "_sha256"] = _sha(inputs[key])
            del ct
        write_json(args.work / "inputs.json", inputs)
        command = [
            sys.executable,
            str(Path(__file__).with_name("surface_export.py")),
            "--field",
            str(args.field),
            "--metadata",
            str(args.metadata),
            "--support",
            str(support_path),
            "--level",
            str(selection["level"]),
            "--seed-xyz",
            *map(str, selection["seed_xyz"]),
            "--output",
            str(args.output),
            "--pitch",
            "1",
            "--depth-offsets",
            "-8",
            "-4",
            "0",
            "4",
            "8",
            "--mesh-step",
            "2",
            "--app-run",
            str(args.app_run),
            "--flatten-iters",
            "200",
            "--flatten-timeout",
            "120",
            "--ct",
            inputs["ct"],
            "--ct-metadata",
            inputs["ct_metadata"],
        ]
        print("EXPORT", " ".join(command), flush=True)
        with (args.work / "export.log").open("x") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        result.update(command=command, returncode=run.returncode, seed=selection, inputs=inputs)
        receipt = json.loads((args.output / "receipt.json").read_text())
        result["export_receipt"] = receipt
        if (args.output / "surface.npz").is_file():
            with np.load(args.output / "surface.npz") as mesh:
                vertices, faces = mesh["vertices_xyz"], mesh["faces"]
            ownership_path = args.field.parent / meta["core_ownership"]
            ownership = np.load(ownership_path, mmap_mode="r", allow_pickle=False)
            result["core_ownership_sha256"] = _sha(ownership_path)
            result["seams"] = seam_registration(vertices, faces, meta, ownership)
            print("EVALUATE frozen mesh against held-out patch", flush=True)
            result["heldout_patch"] = evaluate_patch(args.patch, vertices, faces, meta, ownership)
            ct = np.load(inputs["ct"], mmap_mode="r", allow_pickle=False)
            context_image(
                args.work / "ct_context.png",
                ct,
                json.loads(Path(inputs["ct_metadata"]).read_text()),
                selection,
                vertices,
            )
        result["status"] = receipt["status"]
        code = run.returncode
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["small_artifacts"] = {
            f.name: _sha(f) for f in args.work.iterdir() if f.is_file() and f.suffix in (".json", ".png", ".log")
        }
        write_json(args.work / "result.json", result)
        print(
            "PROBE RESULT",
            json.dumps(dict(status=result["status"], error=result.get("error"), result=str(args.work / "result.json"))),
            flush=True,
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
