"""Executable five-stage field APIs over explicit geometry and verified observations."""

import numpy as np

from common import arclength
from field_constraints import QuadraticConstraints, consensus_constraints, pair_constraints
from field_topology import sample_segment, split_segments
from physical_grid import point_stencils


def observed_points(points, origin, spacing, shape, support=None, charts=None):
    if charts is not None and (charts.shape != tuple(shape) or charts.dtype.kind not in "iu"):
        raise ValueError("chart/grid type or shape mismatch")
    nodes, coeff, valid = point_stencils(points, origin, spacing, shape, support)
    labels = np.zeros(len(nodes), int)
    if charts is not None:
        values = charts.ravel()[nodes]
        for i in range(len(nodes)):
            used = values[i][coeff[i] > 0]
            if not len(used) or (used < 0).any() or np.any(used != used[0]):
                valid[i] = False
                labels[i] = -1
            else:
                labels[i] = used[0]
    return valid, labels


def observed_edges(points, origin, spacing, shape, support=None, charts=None):
    """DDA cell-boundary/midpoint probes cover every interpolated polyline cell."""
    good = np.ones(len(points), bool)
    if support is None and charts is None:
        return good
    grid = (np.asarray(points) - origin) / spacing
    for i in range(1, len(points)):
        a, b = grid[i - 1], grid[i]
        delta = b - a
        if not np.isfinite([a, b]).all():
            good[i] = False
            continue
        fractions = [0.0, 1.0]
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                continue
            planes = np.arange(np.floor(min(a[axis], b[axis])) + 1, np.ceil(max(a[axis], b[axis])))
            fractions.extend(((planes - a[axis]) / delta[axis]).tolist())
        f = np.unique(np.clip(fractions, 0, 1))
        mid = (f[:-1] + f[1:]) / 2
        f = np.sort(np.r_[f, mid])
        xyz = origin + (a + f[:, None] * delta) * spacing
        valid, labels = observed_points(xyz, origin, spacing, shape, support, charts)
        good[i] = valid.all() and (charts is None or np.all(labels == labels[0]))
    return good


def nodal_pair_constraints(a, b, origin, spacing, shape, weight):
    from scipy.sparse import coo_matrix

    q = np.rint((np.stack([a, b], axis=1) - origin) / spacing).astype(int)[:, :, ::-1]
    indices = np.ravel_multi_index(q.reshape(-1, 3).T, shape).reshape(-1, 2)
    rows = np.repeat(np.arange(len(indices)), 2)
    return QuadraticConstraints(
        coo_matrix(
            (np.tile([1.0, -1.0], len(indices)), (rows, indices.ravel())), shape=(len(indices), int(np.prod(shape)))
        ),
        weight,
    )


def match_trace(constraints, target_trace):
    if not np.isfinite(target_trace) or target_trace < 0 or constraints.trace <= 0:
        raise ValueError("positive observed constraint trace required")
    factor = target_trace / constraints.trace
    result = QuadraticConstraints(
        constraints.matrix, constraints.weights * factor, constraints.targets, constraints.groups
    )
    return result, dict(
        scale=factor,
        trace_before=constraints.trace,
        trace_after=result.trace,
        rule="geometry-only matched total diagonal trace, not outcome fitting",
    )


def compile_fibres(
    fibres, origin, spacing, shape, voxel_mm=0.0096, pitch_mm=0.15, support=None, chart_volume=None, pair_weight=0.1
):
    a = []
    b = []
    samples = []
    mass = []
    groups = []
    kept = []
    lost = 0
    segments = 0
    max_xyz = np.full(3, -np.inf)
    for fibre in fibres:
        p = np.asarray(fibre["points_xyz"], float)
        fid = fibre["fibre_id"]
        valid, charts = observed_points(p, origin, spacing, shape, support, chart_volume)
        if chart_volume is not None:
            if chart_volume.shape != tuple(shape):
                raise ValueError("chart/grid mismatch")
        breaks = (
            np.zeros(len(p), bool)
            if fibre.get("break_before") is None
            else np.asarray(fibre["break_before"], bool).copy()
        )
        breaks |= ~observed_edges(p, np.asarray(origin), np.asarray(spacing), shape, support, chart_volume)
        selected = split_segments(p, valid, charts, breaks)
        used = False
        for ix in selected:
            segment = p[ix]
            if arclength(segment, voxel_mm)[-1] <= 0:
                continue
            uniform, w = sample_segment(segment, voxel_mm, pitch_mm)
            known, uniform_charts = observed_points(uniform, origin, spacing, shape, support, chart_volume)
            lost += int((~known).sum())
            edge_known = observed_edges(uniform, np.asarray(origin), np.asarray(spacing), shape, support, chart_volume)
            for run in split_segments(uniform, known, uniform_charts, ~edge_known):
                points = uniform[run]
                weights = w[run]
                samples.append(points)
                mass.append(weights)
                groups.append(np.full(len(points), segments, np.int64))
                max_xyz = np.maximum(max_xyz, points.max(0))
                segments += 1
                used = True
                s = arclength(points, voxel_mm)
                i, j = np.triu_indices(len(points), 1)
                eligible = np.flatnonzero(s[j] - s[i] >= min(0.3, s[-1]))
                if len(eligible):
                    if len(eligible) > 8:
                        eligible = np.sort(np.random.default_rng(int(fid)).choice(eligible, 8, replace=False))
                    a.append(points[i[eligible]])
                    b.append(points[j[eligible]])
        if used:
            kept.append(int(fid))
    if not samples or not a:
        raise ValueError("no usable fibre observations")
    pair = pair_constraints(np.concatenate(a), np.concatenate(b), origin, spacing, shape, pair_weight, support=support)
    budget = nodal_pair_constraints(
        np.concatenate(a), np.concatenate(b), np.asarray(origin), np.asarray(spacing), shape, pair_weight
    ).trace
    if budget > 0 and pair.trace > 0:
        pair, pair_normalization = match_trace(pair, budget)
    else:
        pair_normalization = dict(scale=1.0, rule="degenerate nodal budget; retain measured continuous trace")
    consensus = consensus_constraints(
        np.concatenate(samples),
        np.concatenate(groups),
        np.concatenate(mass),
        origin,
        spacing,
        shape,
        weight=1,
        support=support,
    )
    audit = dict(
        fibre_ids=kept,
        segments=segments,
        pair_rows=pair.matrix.shape[0],
        samples=consensus.matrix.shape[0],
        unsupported_samples=lost,
        pitch_mm=pitch_mm,
        pair_weight=pair_weight,
        max_sample_xyz=max_xyz.tolist(),
        trace_budget=budget,
        pair_trace_normalization=pair_normalization,
        a_xyz=np.concatenate(a),
        b_xyz=np.concatenate(b),
        limitation="Declared geometry/support/chart breaks, not proof of pretrained trace or single-turn correctness.",
    )
    return pair, consensus, audit


def compile_separation(relations, origin, spacing, shape, sheet_step, weight=1.0, support=None):
    if not np.isfinite(sheet_step) or sheet_step <= 0:
        raise ValueError("positive fixed sheet scale required")
    for r in relations:
        if (
            r.get("authority") not in ("prediction_geometry_not_physical_truth", "controlled_synthetic_truth")
            or r.get("order_authority") not in ("declared_reference_prior_not_truth", "controlled_synthetic_truth")
            or not np.isfinite(r["min_difference_sheets"])
            or r["min_difference_sheets"] <= 0
        ):
            raise ValueError("missing supported relation provenance or signed margin")
    successors = {}
    incoming = {}
    for r in relations:
        a, b = tuple(r["a_xyz"]), tuple(r["b_xyz"])
        successors.setdefault(a, set()).add(b)
        successors.setdefault(b, set())
    for a, values in successors.items():
        incoming.setdefault(a, 0)
        for b in values:
            incoming[b] = incoming.get(b, 0) + 1
    queue = [a for a in incoming if incoming[a] == 0]
    visited = 0
    for a in queue:
        visited += 1
        for b in successors[a]:
            incoming[b] -= 1
            if incoming[b] == 0:
                queue.append(b)
    if visited != len(successors):
        raise ValueError("contradictory separation order cycle")
    a = np.array([r["a_xyz"] for r in relations], float).reshape(-1, 3)
    b = np.array([r["b_xyz"] for r in relations], float).reshape(-1, 3)
    margins = np.array([r["min_difference_sheets"] for r in relations]) * sheet_step
    return pair_constraints(b, a, origin, spacing, shape, weight, targets=margins, support=support)
