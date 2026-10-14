"""Physical sample constraints, eliminated segment means, exact tile restriction."""

import hashlib

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, vstack

from physical_grid import point_stencils


class QuadraticConstraints:
    """Energy sum w_i*(A_i phi-target_i-c_group)^2; group=-1 means c=0."""

    def __init__(self, matrix, weights=1.0, targets=0.0, groups=None):
        if np.iscomplexobj(matrix.data):
            raise ValueError("real constraint matrix required")
        self.matrix = csr_matrix(matrix, dtype=np.float64, copy=True)
        self.matrix.sum_duplicates()
        self.matrix.eliminate_zeros()
        n = self.matrix.shape[0]
        self.weights = np.broadcast_to(np.asarray(weights, float), (n,)).copy()
        self.targets = np.broadcast_to(np.asarray(targets, float), (n,)).copy()
        raw = np.full(n, -1, np.int64) if groups is None else np.asarray(groups)
        if (
            raw.shape != (n,)
            or raw.dtype.kind not in "iu"
            or (raw < -1).any()
            or not np.isfinite(self.matrix.data).all()
            or not np.isfinite(self.weights).all()
            or not np.isfinite(self.targets).all()
            or (self.weights < 0).any()
        ):
            raise ValueError("invalid constraint coefficients, weights, targets or groups")
        self.groups = raw.astype(np.int64).copy()
        use = self.groups >= 0
        self.groups[use] = np.unique(self.groups[use], return_inverse=True)[1]
        ng = int(self.groups.max() + 1) if use.any() else 0
        self.group_sum = np.bincount(self.groups[use], weights=self.weights[use], minlength=ng)
        self.members = coo_matrix((self.weights[use], (self.groups[use], np.flatnonzero(use))), shape=(ng, n)).tocsr()
        # Exact diagonal after coalescing shared interpolation nodes and eliminating means.
        diagonal = np.asarray(self.matrix.power(2).T @ self.weights).ravel()
        if ng:
            means = self.members @ self.matrix
            inverse = np.divide(1, self.group_sum, out=np.zeros(ng), where=self.group_sum > 0)
            diagonal -= np.asarray(means.power(2).T @ inverse).ravel()
        if np.min(diagonal, initial=0) < -1e-8:
            raise ValueError("negative quadratic diagonal")
        self.diagonal = np.maximum(diagonal, 0)

    @property
    def trace(self):
        return float(self.diagonal.sum())

    @property
    def digest(self):
        h = hashlib.sha256()
        for a in (
            np.array(self.matrix.shape, np.int64),
            self.matrix.indptr.astype(np.int64),
            self.matrix.indices.astype(np.int64),
            self.matrix.data,
            self.weights,
            self.targets,
            self.groups,
        ):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez_compressed(
            path,
            data=self.matrix.data,
            indices=self.matrix.indices,
            indptr=self.matrix.indptr,
            shape=np.array(self.matrix.shape),
            weights=self.weights,
            targets=self.targets,
            groups=self.groups,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            shape = f["shape"]
            if shape.shape != (2,) or shape.dtype.kind not in "iu" or (shape < 0).any():
                raise ValueError("invalid matrix shape")
            data, indices, indptr = f["data"], f["indices"], f["indptr"]
            if (
                data.ndim != 1
                or indices.shape != data.shape
                or indices.dtype.kind not in "iu"
                or indptr.ndim != 1
                or indptr.dtype.kind not in "iu"
                or len(indptr) != int(shape[0]) + 1
                or indptr[0] != 0
                or indptr[-1] != len(data)
                or (np.diff(indptr.astype(np.int64)) < 0).any()
                or (indices < 0).any()
                or (indices >= shape[1]).any()
            ):
                raise ValueError("invalid CSR coordinate inventory")
            a = csr_matrix((data, indices, indptr), shape=tuple(map(int, shape)))
            a.check_format(full_check=True)
            return cls(a, f["weights"], f["targets"], f["groups"])

    def residual(self, field, center_weights=None):
        r = np.asarray(self.matrix @ np.asarray(field).ravel()).ravel() - self.targets
        use = self.groups >= 0
        if use.any():
            w = self.weights if center_weights is None else np.asarray(center_weights, float)
            if w.shape != self.weights.shape or not np.isfinite(w).all() or (w < 0).any():
                raise ValueError("invalid robust center weights")
            sums = np.bincount(self.groups[use], weights=w[use] * r[use], minlength=len(self.group_sum))
            total = np.bincount(self.groups[use], weights=w[use], minlength=len(self.group_sum))
            mean = np.divide(sums, total, out=np.zeros_like(sums), where=total > 0)
            r[use] -= mean[self.groups[use]]
        return r

    def restrict(self, local_nodes, previous):
        local_nodes = np.asarray(local_nodes)
        if (
            local_nodes.ndim != 1
            or local_nodes.dtype.kind not in "iu"
            or len(np.unique(local_nodes)) != len(local_nodes)
            or (local_nodes < 0).any()
            or (local_nodes >= self.matrix.shape[1]).any()
        ):
            raise ValueError("invalid local node selection")
        previous = np.asarray(previous, float).ravel()
        if len(previous) != self.matrix.shape[1] or not np.isfinite(previous).all():
            raise ValueError("finite global previous field required")
        outside = previous.copy()
        outside[local_nodes] = 0
        # Zero-column outside rows stay: their mass and target affect eliminated means.
        return QuadraticConstraints(
            self.matrix[:, local_nodes], self.weights, self.targets - self.matrix @ outside, self.groups
        )

    def robust_weights(self, field, delta, center_weights=None):
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("positive robust delta required")
        r = abs(self.residual(field, center_weights))
        factors = np.minimum(1, delta / np.maximum(r, delta))
        return QuadraticConstraints(self.matrix, self.weights * factors, self.targets, self.groups), factors

    def operator(self, device, dtype=np.float32):
        return _QuadraticOperator(self, device)


class _QuadraticOperator:
    def __init__(self, c, device):
        def sparse(a):
            a = a.tocoo()
            indices = torch.as_tensor(np.stack([a.row, a.col]), dtype=torch.long, device=device)
            return torch.sparse_coo_tensor(
                indices, torch.as_tensor(a.data, dtype=torch.float32, device=device), a.shape, check_invariants=True
            ).coalesce()

        with torch.sparse.check_sparse_tensor_invariants(True):
            self.a = sparse(c.matrix)
            self.at = self.a.transpose(0, 1).coalesce()
        self.w = torch.as_tensor(c.weights, dtype=torch.float32, device=device)
        self.target = torch.as_tensor(c.targets, dtype=torch.float32, device=device)
        self.groups = torch.as_tensor(c.groups, dtype=torch.long, device=device)
        self.use = self.groups >= 0
        self.sum = torch.as_tensor(c.group_sum, dtype=torch.float32, device=device)
        self.diagonal = torch.as_tensor(c.diagonal, dtype=torch.float32, device=device)

    def center_weight(self, r):
        v = self.w * r
        if len(self.sum):
            total = torch.zeros_like(self.sum)
            total.index_add_(0, self.groups[self.use], v[self.use])
            mean = total / self.sum.clamp_min(torch.finfo(torch.float32).tiny)
            v = v.clone()
            v[self.use] -= self.w[self.use] * mean[self.groups[self.use]]
        return v

    def matvec(self, x):
        y = torch.sparse.mm(self.a, x.reshape(-1, 1)).reshape(-1)
        return torch.sparse.mm(self.at, self.center_weight(y).reshape(-1, 1)).reshape(x.shape)

    def rhs(self):
        return torch.sparse.mm(self.at, self.center_weight(self.target).reshape(-1, 1)).reshape(-1)


def sample_matrix(points, origin, spacing, shape, support=None):
    nodes, w, valid = point_stencils(points, origin, spacing, shape, support)
    if not valid.all():
        raise ValueError("constraint point outside valid support")
    n = len(nodes)
    a = coo_matrix((w.ravel(), (np.repeat(np.arange(n), 8), nodes.ravel())), shape=(n, int(np.prod(shape)))).tocsr()
    a.eliminate_zeros()
    return a


def pair_constraints(a, b, origin, spacing, shape, weight=1.0, targets=0.0, support=None):
    if np.asarray(a).shape != np.asarray(b).shape:
        raise ValueError("pair endpoints differ")
    return QuadraticConstraints(
        sample_matrix(a, origin, spacing, shape, support) - sample_matrix(b, origin, spacing, shape, support),
        weight,
        targets,
    )


def consensus_constraints(points, segments, length_weights, origin, spacing, shape, weight=1.0, support=None):
    return QuadraticConstraints(
        sample_matrix(points, origin, spacing, shape, support),
        np.asarray(length_weights) * weight,
        groups=np.asarray(segments, dtype=np.int64),
    )


def combine_constraints(terms):
    terms = [c for c in terms if c is not None]
    if not terms:
        return None
    groups = []
    offset = 0
    for c in terms:
        g = c.groups.copy()
        g[g >= 0] += offset
        offset += len(c.group_sum)
        groups.append(g)
    return QuadraticConstraints(
        vstack([c.matrix for c in terms]),
        np.concatenate([c.weights for c in terms]),
        np.concatenate([c.targets for c in terms]),
        np.concatenate(groups),
    )


def solve_with_relations(
    target,
    weights,
    spacing=(1, 1, 1),
    constraints=None,
    separation=None,
    robust_delta=None,
    outer_iterations=20,
    **kwargs,
):
    """IRLS consensus and squared lower-bound hinge; finite penalties are not hard order."""
    from field_solver import solve_field

    if type(outer_iterations) is not int or outer_iterations < 1:
        raise ValueError("invalid outer iterations")
    if separation is not None and (separation.groups >= 0).any():
        raise ValueError("separation cannot have consensus groups")
    current = constraints
    initial = kwargs.pop("initial", None)
    if initial is None:
        initial = np.zeros(weights.shape, np.float32)
    field = np.asarray(initial, np.float32)
    factors = np.ones(0 if constraints is None else len(constraints.weights))
    history = []
    info = {}
    stable = False
    for iteration in range(outer_iterations):
        active = None
        hinge = None
        if separation is not None:
            active = np.asarray(separation.matrix @ field.ravel()).ravel() < separation.targets
            hinge = QuadraticConstraints(separation.matrix, separation.weights * active, separation.targets)
        field, info = solve_field(
            target, weights, spacing, constraints=combine_constraints([current, hinge]), initial=field, **kwargs
        )
        if not info["converged"]:
            break
        change = 0
        if robust_delta is not None and constraints is not None:
            current, newfactors = constraints.robust_weights(field, robust_delta, center_weights=current.weights)
            change = float(np.max(abs(newfactors - factors), initial=0))
            factors = newfactors
        newactive = (
            None if separation is None else np.asarray(separation.matrix @ field.ravel()).ravel() < separation.targets
        )
        stable = change <= 1e-3 and (active is None or np.array_equal(active, newactive))
        history.append(
            dict(
                iteration=iteration + 1,
                weight_change=change,
                active_separation=int(newactive.sum()) if newactive is not None else 0,
            )
        )
        if stable:
            break
    violations = (
        0
        if separation is None
        else int(np.sum(np.asarray(separation.matrix @ field.ravel()).ravel() < separation.targets))
    )
    info = dict(
        info,
        converged=bool(info.get("converged") and stable),
        outer_converged=stable,
        outer_history=history,
        separation_violations=violations,
        robust_factors=factors.tolist(),
        limitation="Soft order penalties and numerical convergence do not certify physical identity.",
    )
    return field, info
