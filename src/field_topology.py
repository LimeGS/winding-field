"""Geometry observations with explicit unknowns, cuts and prediction-only authority."""

import itertools

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import arclength


def chart_domains(material_charts, observed):
    """One-node interpolation halo; competing observations remain unresolved.

    This extends a coordinate domain, not the observed material or sheet labels.
    The radius is dictated by the eight-node trilinear stencil, not field errors.
    """
    labels = np.asarray(material_charts)
    observed = np.asarray(observed)
    if (
        labels.ndim != 3
        or labels.dtype.kind not in "iu"
        or observed.shape != labels.shape
        or observed.dtype.kind != "b"
    ):
        raise ValueError("invalid chart domain observations")
    minimum = np.full(labels.shape, np.iinfo(np.int64).max)
    maximum = np.full(labels.shape, -1, np.int64)
    for delta in itertools.product((-1, 0, 1), repeat=3):
        a = tuple(slice(max(0, -v), min(length, length - v)) for v, length in zip(delta, labels.shape))
        b = tuple(slice(max(0, v), min(length, length + v)) for v, length in zip(delta, labels.shape))
        values = labels[a]
        known = (values >= 0) & observed[a]
        np.minimum(minimum[b], values, out=minimum[b], where=known)
        np.maximum(maximum[b], values, out=maximum[b], where=known)
    result = labels.copy()
    fill = (labels < 0) & observed & (minimum == maximum) & (maximum >= 0)
    result[fill] = maximum[fill]
    result[~observed] = -1
    return result


def geometry_charts(probability, directors, spacing_mm, observed_support=None, patch_size_mm=5.0):
    """Local tangent-continuation regions, NOT sheet or winding identity labels.

    Nonmaterial/uncertain nodes stay -1. Edges across gaps, normal discontinuities
    or physical patch boundaries are explicit cuts; no connectivity is forced.
    """
    p = np.asarray(probability, float)
    n = np.asarray(directors, float)
    h = np.asarray(spacing_mm, float)
    if (
        p.ndim != 3
        or n.shape != (*p.shape, 3)
        or h.shape != (3,)
        or not np.isfinite(p).all()
        or not np.isfinite(n).all()
        or (p < 0).any()
        or (p > 1).any()
        or (h <= 0).any()
        or not np.isfinite(h).all()
        or not np.isfinite(patch_size_mm)
        or patch_size_mm <= 0
    ):
        raise ValueError("invalid native chart observations")
    observed = np.ones(p.shape, bool) if observed_support is None else np.asarray(observed_support)
    if observed.shape != p.shape or observed.dtype.kind != "b":
        raise ValueError("invalid observed support")
    material = p >= 0.8
    norm = np.linalg.norm(n, axis=-1)
    eligible = material & observed & (norm > 1e-6)
    n = np.divide(n, norm[..., None], out=np.zeros_like(n), where=norm[..., None] > 1e-6)
    nodes = np.flatnonzero(eligible)
    labels = np.full(p.shape, -1, np.int32)
    if not len(nodes):
        return labels, dict(
            chart_count=0,
            material_nodes=int(material.sum()),
            unresolved_material_nodes=int(material.sum()),
            authority="prediction_geometry_not_winding_identity",
        )
    lookup = np.full(p.shape, -1, np.int64)
    lookup.ravel()[nodes] = np.arange(len(nodes))
    rows = []
    cols = []
    cuts = 0
    for delta in itertools.product((-1, 0, 1), repeat=3):
        if delta == (0, 0, 0) or next(v for v in delta if v) != 1:
            continue
        a = tuple(slice(max(0, -v), min(length, length - v)) for v, length in zip(delta, p.shape))
        b = tuple(slice(max(0, v), min(length, length + v)) for v, length in zip(delta, p.shape))
        both = eligible[a] & eligible[b]
        positions = np.argwhere(both)
        if not len(positions):
            continue
        start = np.array([s.start for s in a])
        end = np.array([s.start for s in b])
        ia = positions + start
        ib = positions + end
        na = n[tuple(ia.T)]
        nb = n[tuple(ib.T)]
        direction = np.asarray(delta)[::-1] * h
        direction /= np.linalg.norm(direction)
        tangent = (abs(na @ direction) <= 0.26) & (abs(nb @ direction) <= 0.26)
        smooth = abs(np.sum(na * nb, axis=1)) >= 0.8
        midpoint = (ia + ib) / 2
        substrate = map_coordinates(p, midpoint.T, order=1, prefilter=False, mode="constant", cval=np.nan) >= 0.8
        same_patch = np.all(
            np.floor(ia[:, ::-1] * h / patch_size_mm) == np.floor(ib[:, ::-1] * h / patch_size_mm), axis=1
        )
        keep = tangent & smooth & substrate & same_patch
        cuts += int((~keep).sum())
        rows.extend(lookup[tuple(ia[keep].T)].tolist())
        cols.extend(lookup[tuple(ib[keep].T)].tolist())
    graph = coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(nodes), len(nodes)))
    count, components = connected_components(graph, directed=False)
    labels.ravel()[nodes] = components
    domains = chart_domains(labels, observed)
    return domains, dict(
        chart_count=int(count),
        material_nodes=int(material.sum()),
        eligible_nodes=len(nodes),
        interpolation_domain_nodes=int((domains >= 0).sum()),
        interpolation_halo_nodes=int(((domains >= 0) & (labels < 0)).sum()),
        unresolved_material_nodes=int((material & ~eligible).sum()),
        candidate_edges=len(rows),
        cut_edges=cuts,
        patch_size_mm=patch_size_mm,
        authority="prediction_geometry_not_winding_identity",
        limitation="Bounded tangent regions do not establish winding transitions; missing/cross-chart stencils remain unresolved.",
    )


def split_segments(points, valid=None, charts=None, break_before=None):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("XYZ polyline required")
    n = len(p)
    good = np.isfinite(p).all(1)
    if valid is not None:
        good &= np.broadcast_to(np.asarray(valid, bool), (n,))
    c = np.zeros(n, int) if charts is None else np.asarray(charts)
    breaks = np.zeros(n, bool) if break_before is None else np.asarray(break_before, bool)
    if c.shape != (n,) or breaks.shape != (n,) or c.dtype.kind not in "iu":
        raise ValueError("bad segment metadata")
    good &= c >= 0
    out = []
    start = None
    for i in range(n):
        disconnected = not good[i] or (i > 0 and (breaks[i] or c[i] != c[i - 1]))
        if disconnected and start is not None:
            if i - start >= 2:
                out.append(np.arange(start, i))
            start = None
        if good[i] and start is None:
            start = i
    if start is not None and n - start >= 2:
        out.append(np.arange(start, n))
    return out


def sample_segment(points, voxel_mm, pitch_mm=0.15):
    p = np.asarray(points, float)
    if not np.isfinite([voxel_mm, pitch_mm]).all() or min(voxel_mm, pitch_mm) <= 0 or not np.isfinite(p).all():
        raise ValueError("finite geometry and positive sampling pitch required")
    s = arclength(p, voxel_mm)
    if len(p) < 2 or s[-1] <= 0:
        raise ValueError("nondegenerate segment required")
    positions = np.linspace(0, s[-1], int(np.ceil(s[-1] / pitch_mm)) + 1)
    sampled = np.stack([np.interp(positions, s, p[:, k]) for k in range(3)], axis=1)
    edges = np.diff(positions)
    w = np.zeros(len(positions))
    w[:-1] += edges / 2
    w[1:] += edges / 2
    return sampled, w


def chart_offsets(transitions, tolerance=1e-8):
    """Consistent potential differences; no forced choice for inconsistent cycles."""
    graph = {}
    for a, b, d in transitions:
        if type(a) is not int or type(b) is not int or not np.isfinite(d):
            raise ValueError("bad chart transition")
        graph.setdefault(a, []).append((b, float(d)))
        graph.setdefault(b, []).append((a, -float(d)))
    result = {}
    for root in sorted(graph):
        if root in result:
            continue
        result[root] = 0.0
        queue = [root]
        for a in queue:
            for b, d in graph[a]:
                value = result[a] + d
                if b in result:
                    if abs(result[b] - value) > tolerance:
                        raise ValueError("contradictory chart transition cycle")
                else:
                    result[b] = value
                    queue.append(b)
    return result


def profile_relations(offsets_mm, probability, min_gap_mm=0.12, max_gap_mm=0.30):
    """Conservative peak/gap hypothesis, not a certificate that peaks are laminae."""
    x = np.asarray(offsets_mm, float)
    p = np.asarray(probability, float)
    if (
        x.ndim != 1
        or p.shape != x.shape
        or len(x) < 3
        or not np.isfinite(x).all()
        or (np.diff(x) <= 0).any()
        or min_gap_mm <= 0
        or max_gap_mm < min_gap_mm
    ):
        raise ValueError("invalid profile")
    tolerance = 8 * np.finfo(float).eps
    if ((p[np.isfinite(p)] < -tolerance) | (p[np.isfinite(p)] > 1 + tolerance)).any():
        raise ValueError("probabilities outside [0,1]")
    # Trilinear floating-point sums may exceed the convex bound by a few ULPs.
    clipped = int(np.sum(np.isfinite(p) & ((p < 0) | (p > 1))))
    p = np.clip(p, 0, 1)
    peaks, _ = find_peaks(np.where(np.isfinite(p), p, -1), height=0.8, prominence=0.3)
    accepted = []
    ambiguous = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        gap = x[b] - x[a]
        interval = p[a : b + 1]
        if (
            min_gap_mm <= gap <= max_gap_mm
            and np.isfinite(interval).all()
            and np.count_nonzero(interval[1:-1] < 0.2) >= 2
        ):
            accepted.append((int(a), int(b)))
        else:
            ambiguous.append(
                dict(indices=[int(a), int(b)], gap_mm=float(gap), reason="faces_delamination_unknown_or_unresolved_gap")
            )
    return dict(
        accepted=accepted,
        ambiguous=ambiguous,
        roundoff_clipped=clipped,
        authority="prediction_geometry_not_physical_truth",
    )


def build_transversal_inventory(
    probability, directors, origin, spacing, voxel_mm, reference=None, max_profiles=2000, seed=20261012, support=None
):
    """Observe native prediction profiles before mapping relations to solver nodes."""
    from patch_check import sample_field
    from physical_grid import point_stencils

    p = np.asarray(probability, float)
    n = np.asarray(directors, float)
    if (
        p.ndim != 3
        or n.shape != (*p.shape, 3)
        or not np.isfinite(n).all()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
        or voxel_mm <= 0
    ):
        raise ValueError("invalid observation arrays")
    if support is None:
        support = np.ones(p.shape, bool)
    candidates = np.flatnonzero((p > 0.8) & support)
    if len(candidates) > max_profiles:
        candidates = np.random.default_rng(seed).choice(candidates, max_profiles, replace=False)
    offsets = np.arange(-0.35, 0.3500001, 0.0096)
    result = []
    ambiguous = 0
    unresolved_order = 0
    for flat in candidates:
        zyx = np.array(np.unravel_index(flat, p.shape))
        center = np.asarray(origin) + zyx[::-1] * spacing
        direction = n[tuple(zyx)]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction = direction / norm
        xyz = center + offsets[:, None] / voxel_mm * direction
        values = sample_field(p, xyz, origin, spacing)
        _, _, valid = point_stencils(xyz, origin, spacing, p.shape, support)
        values[~valid] = np.nan
        rel = profile_relations(offsets, values)
        ambiguous += len(rel["ambiguous"])
        for a, b in rel["accepted"]:
            endpoints = xyz[[a, b]]
            if reference is None:
                unresolved_order += 1
                continue
            labels = sample_field(reference, endpoints, origin, spacing)
            if not np.isfinite(labels).all() or abs(labels[1] - labels[0]) < 1e-6:
                unresolved_order += 1
                continue
            if labels[1] < labels[0]:
                endpoints = endpoints[::-1]
            result.append(
                dict(
                    a_xyz=endpoints[0].tolist(),
                    b_xyz=endpoints[1].tolist(),
                    physical_gap_mm=float(offsets[b] - offsets[a]),
                    authority=rel["authority"],
                    order_authority="declared_reference_prior_not_truth",
                    min_difference_sheets=0.5,
                    source_profile_node=int(flat),
                )
            )
    return dict(
        relations=result,
        ambiguous_profiles=ambiguous,
        unresolved_order=unresolved_order,
        sampled_profiles=len(candidates),
        seed=seed,
        scope="native_prediction_profile_hypotheses",
        limitation="Two peaks with a gap are conservative prediction hypotheses; no physical lamina certification.",
    )
