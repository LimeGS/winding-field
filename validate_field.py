"""Field self-checks for the validated wrap graph.

Reports per-level coverage of the main connected component and the
corridor-anomaly census. Double-spacing gaps are treated as merged- or
missing-sheet signatures inherited from the surface predictions.

Usage: python validate_field.py --field field.npz
"""
import argparse
from collections import Counter

import numpy as np


def main_component_id(labels, stride=7):
    """Return the most common non-negative component id from a sparse sample."""
    sample = labels[::stride, ::stride]
    valid = sample[sample >= 0]
    if valid.size == 0:
        return None
    return int(Counter(valid.tolist()).most_common(1)[0][0])


def main_component_coverage(labels, main_component=None, stride=7):
    """Fraction of valid wraps in each z-level belonging to the main component."""
    if main_component is None:
        main_component = main_component_id(labels, stride=stride)
    coverage = []
    for zi in range(labels.shape[0]):
        level = labels[zi]
        finite = level >= 0
        if finite.any() and main_component is not None:
            coverage.append(float((level[finite] == main_component).mean()))
        else:
            coverage.append(0.0)
    return np.asarray(coverage, dtype=np.float64)


def merged_sheet_exposure(radii, z_stride=4, theta_stride=4):
    """Count corridor gaps with a double-spacing signature."""
    anomaly = total = 0
    for zi in range(0, radii.shape[0], z_stride):
        for theta in range(0, radii.shape[1], theta_stride):
            rr = radii[zi, theta]
            rr = rr[np.isfinite(rr)]
            if len(rr) < 6:
                continue
            gaps = np.diff(rr)
            median_gap = np.median(gaps)
            anomaly += int(((gaps > 1.6 * median_gap) & (gaps < 2.7 * median_gap)).sum())
            total += len(gaps)
    pct = anomaly / max(total, 1) * 100.0
    return anomaly, total, pct


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = np.load(args.field)
    if "labels" not in data.files:
        raise ValueError(
            "field artifact has no 'labels' array; validate_field.py expects "
            "the output of build_field.py. Use figures/make_field_figure.py "
            "to recompute connectivity from wfield_multi.npz."
        )
    labels, radii = data["labels"], data["r"]

    coverage = main_component_coverage(labels)
    print(
        "main-component coverage per z-level: "
        f"median {np.median(coverage):.2f} "
        f"p10 {np.percentile(coverage, 10):.2f} | "
        f"levels >=80%: {(coverage >= .8).mean() * 100:.0f}%"
    )

    anomaly, total, pct = merged_sheet_exposure(radii)
    print(f"merged/missing-sheet exposure: {anomaly}/{total} corridors ({pct:.2f}%)")
    print("VALIDATE_DONE")


if __name__ == "__main__":
    main()
