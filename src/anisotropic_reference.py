"""Monotone semi-Lagrangian unsigned arrival, explicit support/charts and residual."""

import itertools
import time

import numpy as np
from scipy.ndimage import binary_propagation

from field_topology import chart_offsets
from physical_grid import point_stencils


def exterior_seeds(material, observed_air, verified_exterior=None):
    material = np.asarray(material, bool)
    air = np.asarray(observed_air, bool)
    if material.ndim != 3 or air.shape != material.shape or (material & air).any():
        raise ValueError("invalid material/observed-air masks")
    if verified_exterior is not None:
        verified = np.asarray(verified_exterior)
        if verified.shape != air.shape or verified.dtype.kind != "b":
            raise ValueError("invalid verified exterior mask")
        return air & verified & ~material
    # This default is for a declared full domain, never an interior crop.
    rim = np.zeros_like(air)
    rim[:, 0, :] = rim[:, -1, :] = True
    rim[:, :, 0] = rim[:, :, -1] = True
    return binary_propagation(rim & air, mask=air) & ~material


def solve_arrival(
    directors,
    spacing,
    seeds,
    support=None,
    charts=None,
    transitions=(),
    c_parallel=0.002,
    c_perp=1.0,
    maxiter=500,
    tolerance=1e-5,
    chunk=32768,
    max_cache_bytes=256 * 1024**2,
):
    """Positive local costs on metric-aligned rays with positive interpolation.

    Transitions encode consistent chart potentials; incompatible cycles fail.
    Output is a chart-local potential, with offsets explicit in the receipt.
    Unknown or unreachable nodes are NaN, never zero. No implicit periodic edges.
    """
    started = time.time()
    n = np.asarray(directors, float)
    h = np.asarray(spacing, float)
    seeds = np.asarray(seeds)
    shape = seeds.shape
    if (
        seeds.ndim != 3
        or min(shape) < 2
        or seeds.dtype.kind != "b"
        or n.shape != (*shape, 3)
        or h.shape != (3,)
        or not np.isfinite(h).all()
        or (h <= 0).any()
        or not np.isfinite(n).all()
        or type(maxiter) is not int
        or maxiter < 1
        or type(chunk) is not int
        or chunk < 1
        or type(max_cache_bytes) is not int
        or max_cache_bytes < 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("invalid arrival geometry or solver parameters")
    valid = np.ones(shape, bool) if support is None else np.asarray(support)
    labels = np.zeros(shape, np.int64) if charts is None else np.asarray(charts)
    if valid.shape != shape or valid.dtype.kind != "b" or labels.shape != shape or labels.dtype.kind not in "iu":
        raise ValueError("support/chart shape or type mismatch")
    valid = valid & (labels >= 0)
    if (seeds & ~valid).any() or not seeds.any():
        raise ValueError("seeds must lie in observed support")
    offsets = chart_offsets(transitions)
    allowed = {(a, b) for a, b, _ in transitions} | {(b, a) for a, b, _ in transitions}
    cp = np.broadcast_to(np.asarray(c_parallel, float), shape).ravel()
    cn = np.broadcast_to(np.asarray(c_perp, float), shape).ravel()
    if not np.isfinite(cp).all() or not np.isfinite(cn).all() or (cp <= 0).any() or (cn <= 0).any():
        raise ValueError("arrival costs must be finite and positive")
    norms = np.linalg.norm(n, axis=-1)
    if (norms[valid] < 1e-8).any():
        raise ValueError("missing director on supported node")
    n = np.divide(n, norms[..., None], out=np.zeros_like(n), where=norms[..., None] > 0).reshape(-1, 3)
    # Canonical representative makes n and -n produce identical discretization.
    pivot = np.argmax(abs(n), axis=1)
    sign = n[np.arange(len(n)), pivot] < 0
    n[sign] *= -1
    axes = np.eye(3)[np.argmin(abs(n), axis=1)]
    t1 = np.cross(n, axes)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(n, t1)
    directions = [v for v in itertools.product((-1, 0, 1), repeat=3) if any(v)]
    ids = np.flatnonzero(valid)
    grid = np.column_stack(np.unravel_index(ids, shape))[:, ::-1] * h
    flat_labels = labels.ravel()
    all_offsets = np.array([offsets.get(int(c), 0.0) for c in flat_labels])
    field = np.full(int(np.prod(shape)), np.inf)
    field[seeds.ravel()] = 0.0

    # Build bounded block stencils once; a ray is invalid if its interior crosses unknown or a forbidden cut.
    def make_block(start):
        block = ids[start : start + chunk]
        centers = grid[start : start + chunk]
        stencils = []
        rays = [
            (u * t1[block] + v * t2[block] + w * n[block]) * h.min() / np.sqrt(u * u + v * v + w * w)
            for u, v, w in directions
        ]
        # Grid-axis rays make seeded faces causally reachable for oblique directors.
        rays.extend(
            [
                np.broadcast_to(sign * np.eye(3)[axis] * h[axis], (len(block), 3))
                for axis in range(3)
                for sign in (-1, 1)
            ]
        )
        for ray in rays:
            distance = np.linalg.norm(ray, axis=1)
            direction = ray / distance[:, None]
            endpoint = centers + ray
            nodes, coeff, good = point_stencils(endpoint, [0, 0, 0], h, shape, valid)
            costs = np.zeros(len(block))
            for fraction in (0.25, 0.5, 0.75, 1.0):
                q, weights, ok = point_stencils(centers + fraction * ray, [0, 0, 0], h, shape, valid)
                good &= ok
                # Integrate the metric along the ray. A frozen tangent Euler step
                # drifts outward on curved sheets at artificially near-zero cost.
                projection = np.clip(np.sum(weights * np.sum(n[q] * direction[:, None, :], axis=2) ** 2, axis=1), 0, 1)
                parallel = np.sum(weights * cp[q], axis=1)
                perpendicular = np.sum(weights * cn[q], axis=1)
                costs += distance / 4 * np.sqrt(parallel**2 * (1 - projection) + perpendicular**2 * projection)
                for col in range(8):
                    source = flat_labels[block]
                    destination = flat_labels[q[:, col]]
                    across = (weights[:, col] > 0) & (source != destination)
                    permitted = np.zeros(len(block), bool)
                    for a, b in allowed:
                        permitted |= (source == a) & (destination == b)
                    good &= ~across | permitted
            self_coeff = np.where(nodes == block[:, None], coeff, 0).sum(1)
            coeff = np.where(nodes == block[:, None], 0, coeff)
            good &= self_coeff < 1 - 1e-6
            stencils.append((nodes.astype(np.int32), coeff.astype(np.float32), good, costs, 1 - self_coeff))
        return block, stencils

    blocks = []
    cache_bytes = 0
    for start in range(0, len(ids), chunk):
        block, stencils = make_block(start)
        size = block.nbytes + sum(a.nbytes for row in stencils for a in row)
        if cache_bytes + size > max_cache_bytes:
            blocks = None
            cache_bytes = 0
            break
        blocks.append((block, stencils))
        cache_bytes += size
    stable = False
    residual = None
    for iteration in range(maxiter):
        old = field
        updated = field.copy()
        stream = blocks if blocks is not None else (make_block(start) for start in range(0, len(ids), chunk))
        for block, stencils in stream:
            best = old[block].copy()
            for nodes, coeff, good, costs, denominator in stencils:
                values = old[nodes]
                finite = np.logical_or(coeff == 0, np.isfinite(values)).all(1) & good
                candidate = np.full(len(block), np.inf)
                if finite.any():
                    candidate[finite] = (
                        np.sum(np.where(coeff[finite] > 0, values[finite], 0) * coeff[finite], axis=1) + costs[finite]
                    ) / denominator[finite]
                best = np.minimum(best, candidate)
            updated[block] = best
        updated[seeds.ravel()] = 0
        finite = np.isfinite(old) & np.isfinite(updated)
        changed_finiteness = not np.array_equal(np.isfinite(old), np.isfinite(updated))
        residual = float(np.max(abs(updated[finite] - old[finite]), initial=0))
        field = updated
        if not changed_finiteness and residual <= tolerance:
            stable = True
            break
    # One fixed-point update below tolerance was observed, including reachability.
    known = np.isfinite(field) & valid.ravel()
    result = np.full(len(field), np.nan)
    result[known] = field[known] - all_offsets[known]
    info = dict(
        converged=stable,
        iterations=iteration + 1,
        bellman_update_residual=residual,
        tolerance=tolerance,
        known_nodes=int(known.sum()),
        unreachable_nodes=int(np.sum(valid.ravel() & ~known)),
        unknown_nodes=int(np.sum(~valid)),
        chart_offsets={str(k): v for k, v in offsets.items()},
        transitions=[list(v) for v in transitions],
        elapsed_s=time.time() - started,
        stencil_cache_bytes=cache_bytes,
        stencils_rebuilt_per_iteration=blocks is None,
        discretization="metric-integrated positive trilinear rays plus grid axes, self-coefficient eliminated",
        limitation="Finite-stencil reference and explicit chart hypotheses, not topology or lamina certification.",
    )
    return result.reshape(shape).astype(np.float32), info
