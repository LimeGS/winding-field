"""Independent NumPy corner interpolation for benchmark verification."""

import itertools

import numpy as np


def interpolate(field, meta, xyz, support=None):
    q = (np.asarray(xyz, float) - meta["origin_xyz"]) / meta["spacing_xyz"]
    good = np.isfinite(q).all(1) & (q >= 1).all(1) & (q <= np.array(field.shape[::-1]) - 2).all(1)
    result = np.full(len(q), np.nan)
    ids = np.flatnonzero(good)
    base = np.floor(q[ids]).astype(int)
    fraction = q[ids] - base
    total = np.zeros(len(ids))
    valid = np.ones(len(ids), bool)
    for corner in itertools.product((0, 1), repeat=3):
        ix = base + corner
        weight = np.prod(np.where(np.array(corner), fraction, 1 - fraction), axis=1)
        total += field[ix[:, 2], ix[:, 1], ix[:, 0]].astype(float) * weight
        if support is not None:
            valid &= support[ix[:, 2], ix[:, 1], ix[:, 0]].astype(bool)
    result[ids[valid]] = total[valid]
    return result
