"""Audit a segment mesh against a winding-field artifact.

Each sampled PPM row is read as a continuous physical path. Candidate wraps
come from the field at the corresponding (z, theta) ray, and a Viterbi pass
balances radial distance against winding-coordinate continuity. The reported
jump rate is the fraction of consecutive path transitions whose winding
increment differs from the continuous angular increment by more than 0.5.

The corrected method artifacts contain ``w``, ``r``, and ``zs``. Artifacts
produced by ``build_field.py`` additionally contain ``labels``; when labels
are present the audit restricts candidates to the largest component, and
when they are absent it uses all finite field candidates. This matches the
as-run audit used for the published corrected table.

Usage:
    python audit_segment.py --field field.npz --axis axis.npy \
        --ppm-url <segment.ppm URL> --transform-url <transform.json URL>
"""
import argparse
from collections import Counter
import json
import urllib.request as urlrequest

import numpy as np


def load_field(path):
    """Load and validate a winding-field artifact."""
    with np.load(path) as data:
        required = {"w", "r", "zs"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"field artifact is missing arrays: {', '.join(missing)}")
        winding = np.array(data["w"], copy=True)
        radii = np.array(data["r"], copy=False)
        z_levels = np.array(data["zs"], copy=False)
        labels = np.array(data["labels"], copy=False) if "labels" in data.files else None

    if winding.shape != radii.shape:
        raise ValueError("field arrays 'w' and 'r' must have identical shapes")
    if winding.ndim != 3 or len(z_levels) != winding.shape[0]:
        raise ValueError("expected w/r shape (z, theta, wrap) and one zs value per z")
    if labels is not None and labels.shape != winding.shape:
        raise ValueError("field array 'labels' must have the same shape as 'w'")
    if len(z_levels) < 2:
        raise ValueError("field must contain at least two z levels")
    return winding, labels, radii, z_levels


def main_component_id(labels, stride=7):
    """Return the dominant non-negative component label, or None."""
    if labels is None:
        return None
    sample = labels[::stride, ::stride]
    valid = sample[sample >= 0]
    if valid.size == 0:
        return None
    return int(Counter(valid.tolist()).most_common(1)[0][0])


def shuffle_winding(winding, seed=0):
    """Shuffle finite winding values independently within each ray."""
    shuffled = np.array(winding, copy=True)
    rng = np.random.default_rng(seed)
    for zi in range(shuffled.shape[0]):
        for theta in range(shuffled.shape[1]):
            finite = np.isfinite(shuffled[zi, theta])
            if finite.sum() > 1:
                values = shuffled[zi, theta, finite].copy()
                rng.shuffle(values)
                shuffled[zi, theta, finite] = values
    return shuffled


def choose_transform(transform):
    """Choose the affine direction with the smaller landmark residual."""
    matrix = np.asarray(transform["transformation_matrix"], dtype=np.float64)
    affine = np.eye(4)
    affine[:3, :3] = matrix[:3, :3]
    affine[:3, 3] = matrix[:3, 3]
    fixed = np.asarray(transform["fixed_landmarks"], dtype=np.float64)
    moving = np.asarray(transform["moving_landmarks"], dtype=np.float64)
    candidates = [affine, np.linalg.inv(affine)]
    errors = [
        float(np.abs(fixed @ candidate[:3, :3].T + candidate[:3, 3] - moving).mean())
        for candidate in candidates
    ]
    selected = candidates[int(np.argmin(errors))]
    return selected[:3, :3], selected[:3, 3], min(errors)


def row_metrics(winding_series, unwrapped_theta, threshold=0.5):
    """Compute jump and endpoint metrics for one Viterbi path."""
    winding_series = np.asarray(winding_series, dtype=np.float64)
    unwrapped_theta = np.asarray(unwrapped_theta, dtype=np.float64)
    if winding_series.ndim != 1 or winding_series.shape != unwrapped_theta.shape:
        raise ValueError("winding series and theta must be equal-length vectors")
    if len(winding_series) < 2:
        raise ValueError("a row needs at least two path points")

    angular_step = np.diff(unwrapped_theta) / (2 * np.pi)
    winding_step = np.diff(winding_series)
    deviation = np.abs(winding_step - angular_step)
    jump_indices = np.flatnonzero(deviation > threshold)
    loops = float((unwrapped_theta[-1] - unwrapped_theta[0]) / (2 * np.pi))
    delta_w = float(winding_series[-1] - winding_series[0])
    transitions = len(winding_series) - 1
    return {
        "points": len(winding_series),
        "transitions": transitions,
        "jumps": len(jump_indices),
        "jump_rate": len(jump_indices) / transitions,
        "jump_indices": jump_indices,
        "winding_step": winding_step,
        "loops": loops,
        "delta_w": delta_w,
        "endpoint_residual": abs(abs(delta_w) - abs(loops)),
    }


def aggregate_metrics(rows):
    """Aggregate jump arithmetic over successfully scored rows."""
    if not rows:
        return {
            "rows": 0,
            "jumps": 0,
            "transitions": 0,
            "jump_rate": float("nan"),
            "median_endpoint_residual": float("nan"),
        }
    jumps = sum(row["jumps"] for row in rows)
    transitions = sum(row["transitions"] for row in rows)
    return {
        "rows": len(rows),
        "jumps": jumps,
        "transitions": transitions,
        "jump_rate": jumps / transitions,
        "median_endpoint_residual": float(
            np.median([row["endpoint_residual"] for row in rows])
        ),
    }


def parse_ppm_header(header):
    """Parse the VC PPM text header and return width, height, byte offset."""
    marker = header.find(b"<>\n")
    if marker < 0:
        raise ValueError("PPM header terminator '<>\\n' was not found")
    fields = {}
    for line in header[:marker].decode().strip().splitlines():
        key, value = line.split(": ", 1)
        fields[key] = value
    return int(fields["width"]), int(fields["height"]), marker + 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--axis", required=True)
    parser.add_argument("--ppm-url", required=True)
    parser.add_argument("--transform-url", required=True)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--ustep", type=int, default=8)
    parser.add_argument("--qscale", type=float, default=4.0)
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--spacing", type=float, default=17.0,
                        help="nominal wrap spacing in field voxels (as-run: 17)")
    parser.add_argument("--candidate-radius", type=float, default=2.5,
                        help="candidate search radius in multiples of --spacing")
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--show-jumps", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    winding, labels, radii, z_levels = load_field(args.field)
    main_component = main_component_id(labels)
    if labels is None:
        print("component_filter=all-finite (field has no labels)", flush=True)
    elif main_component is None:
        raise ValueError("field labels contain no non-negative component ids")
    else:
        print(f"component_filter=largest label {main_component}", flush=True)

    if args.shuffle:
        winding = shuffle_winding(winding)
    axis = np.load(args.axis)

    with urlrequest.urlopen(args.transform_url, timeout=60) as response:
        transform = json.load(response)
    rotation, translation, landmark_error = choose_transform(transform)
    print(f"transform landmark_err={landmark_error:.1f}", flush=True)

    request = urlrequest.Request(args.ppm_url, headers={"Range": "bytes=0-199"})
    header = urlrequest.urlopen(request, timeout=30).read()
    ppm_width, ppm_height, data_offset = parse_ppm_header(header)
    ntheta = winding.shape[1]
    row_results = []

    for v in np.linspace(ppm_height * 0.1, ppm_height * 0.9, args.rows).astype(int):
        start = data_offset + int(v) * ppm_width * 48
        end = start + ppm_width * 48 - 1
        request = urlrequest.Request(
            args.ppm_url, headers={"Range": f"bytes={start}-{end}"}
        )
        buffer = urlrequest.urlopen(request, timeout=300).read()
        expected_bytes = ppm_width * 48
        if len(buffer) != expected_bytes:
            raise ValueError(
                f"PPM row {v} returned {len(buffer)} bytes; expected {expected_bytes}"
            )
        positions = np.frombuffer(buffer, np.float64).reshape(ppm_width, 6)
        positions = positions[::args.ustep, :3]
        positions = positions[(positions > 0).all(axis=1)]
        points = positions @ rotation.T + translation

        zq, yq, xq = points[:, 2] / args.qscale, points[:, 1] / args.qscale, points[:, 0] / args.qscale
        z_index = np.clip(
            np.round((zq - z_levels[0]) / (z_levels[1] - z_levels[0])).astype(int),
            0,
            len(z_levels) - 1,
        )
        steps = []
        for index in range(len(points)):
            z_full = int(z_levels[int(z_index[index])])
            cy, cx = axis[min(z_full, len(axis) - 1)]
            dy, dx = yq[index] - cy, xq[index] - cx
            radius = float(np.hypot(dy, dx))
            theta = float(np.arctan2(dy, dx))
            theta_index = int((theta % (2 * np.pi)) / (2 * np.pi) * ntheta) % ntheta
            rr = radii[int(z_index[index]), theta_index]
            ww = winding[int(z_index[index]), theta_index]
            candidate = np.isfinite(rr) & np.isfinite(ww)
            if labels is not None:
                candidate &= labels[int(z_index[index]), theta_index] == main_component
            candidate &= np.abs(rr - radius) < args.candidate_radius * args.spacing
            if not candidate.any():
                continue
            candidates = [
                (float(abs(rr[j] - radius) / args.spacing) ** 2, float(ww[j]))
                for j in np.flatnonzero(candidate)
            ]
            steps.append((theta, candidates))

        if len(steps) < args.min_points:
            print(f"row v={v}: too few usable points ({len(steps)})", flush=True)
            continue

        theta_unwrapped = np.unwrap([step[0] for step in steps])
        costs = [np.array([cost for cost, _ in steps[0][1]])]
        winding_values = [np.array([value for _, value in steps[0][1]])]
        backpointers = []
        for index in range(1, len(steps)):
            emission = np.array([cost for cost, _ in steps[index][1]])
            values = np.array([value for _, value in steps[index][1]])
            angular_step = (theta_unwrapped[index] - theta_unwrapped[index - 1]) / (2 * np.pi)
            transition = args.lam * (
                values[None, :] - winding_values[-1][:, None] - angular_step
            ) ** 2
            total = costs[-1][:, None] + transition
            previous = np.argmin(total, axis=0)
            costs.append(total[previous, np.arange(len(values))] + emission)
            backpointers.append(previous)
            winding_values.append(values)

        state = int(np.argmin(costs[-1]))
        path = [state]
        for previous in reversed(backpointers):
            state = int(previous[state])
            path.append(state)
        path.reverse()
        winding_series = np.array(
            [winding_values[index][path[index]] for index in range(len(steps))]
        )
        metrics = row_metrics(winding_series, theta_unwrapped)
        metrics["row"] = int(v)
        row_results.append(metrics)

        if args.show_jumps:
            for jump in metrics["jump_indices"]:
                theta_position = float(theta_unwrapped[jump] % (2 * np.pi))
                print(
                    f"  jump u{jump}: dw={metrics['winding_step'][jump]:+.1f} "
                    f"theta={theta_position / (2 * np.pi) * 360:.0f}deg",
                    flush=True,
                )
        print(
            f"row v={v}: points={metrics['points']} transitions={metrics['transitions']} "
            f"loops={metrics['loops']:+.2f} dw={metrics['delta_w']:+.2f} "
            f"endpoint_residual={metrics['endpoint_residual']:.3f} "
            f"jumps={metrics['jumps']} jump_rate={metrics['jump_rate'] * 100:.2f}%",
            flush=True,
        )

    aggregate = aggregate_metrics(row_results)
    mode = "SHUFFLE" if args.shuffle else "REAL"
    print(
        f"AUDIT_DONE mode={mode} rows={aggregate['rows']} "
        f"jumps={aggregate['jumps']} transitions={aggregate['transitions']} "
        f"jump_rate={aggregate['jump_rate'] * 100:.3f}% "
        f"median_endpoint_residual={aggregate['median_endpoint_residual']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
