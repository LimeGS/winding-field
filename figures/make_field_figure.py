"""Generate figures/field_figure.png: wrap-adjacency GRAPH CONNECTIVITY.

WHAT THIS FIGURE SHOWS (and what it does not)
---------------------------------------------
This figure visualizes the *connected components of the candidate
wrap-adjacency graph* — the reusable POSITIVE sub-result of this repo
(largest component = 96.67% of all wraps; see README.md and
docs/NEGATIVE_RESULT.md). It does NOT show a winding-number
assignment: solving for a consistent integer winding number ON TOP of this
graph is the separate, FAILED sub-result (43.8-45.8% GATE3 disagreement
across three solver architectures plus a tolerance sensitivity test — the
negative result documented in
docs/NEGATIVE_RESULT.md). A mostly-green picture here means "the graph is
connected", nothing more.

PROVENANCE / WHY THIS SCRIPT EXISTS
-----------------------------------
An earlier figures/field_figure.png was an orphaned artifact: no generator
script, unreferenced by any doc, and produced by an exploratory run whose
component labels do not match the validated numbers. This script replaces
it with a fully reproducible figure computed from scratch.

The components are RECOMPUTED here with the same algorithm as
build_field.py pass 1: 1-to-1 radius-proximity matching between adjacent
rays (theta-adjacent including the theta seam, and z-adjacent), with
tol = 0.45 x median wrap spacing, then connected components over all
matched pairs. Two deliberate implementation differences, neither of which
changes the resulting components:
  * match() is vectorized with numpy (verified exactly equivalent to
    build_field.match by the built-in self-test below, which runs on every
    invocation);
  * the union-find is scipy.sparse.csgraph.connected_components instead of
    build_field.py's manual loop (same equivalence classes, much faster at
    ~68M nodes / ~127M edges).

IMPORTANT: the input artifact wfield_multi.npz also contains precomputed
`w` and `comp` arrays from an earlier exploratory run. They do NOT match
the validated pass-1 union-find numbers and are deliberately ignored here;
only the raw wrap radii `r` (and `zs`) are read.

Input : wfield_multi.npz — the release artifact described in README.md
        (kept out of git; place it at repo root or pass --npz). Keys used:
        r (nz, NTH, MAXW) float32 wrap radii, NaN-padded; zs (nz,) int64.
Output: figures/field_figure.png

Usage : python figures/make_field_figure.py [--npz PATH] [--out PATH]
Runtime: ~2 minutes on a 12-core laptop (numpy + scipy required).
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from build_field import match as match_ref  # noqa: E402  (self-test oracle)

# The documented, repeatedly-cited connectivity number this figure must
# reproduce (README.md table; docs/NEGATIVE_RESULT.md). If the recomputation
# lands away from it, the script refuses to render rather than produce a
# figure that contradicts the docs.
EXPECTED_LARGEST_PCT = 96.7
EXPECTED_CHAINED_PCT = 99.75
GATE_TOLERANCE_PCT = 1.0
CHAIN_GATE_TOLERANCE_PCT = 0.05

_EMPTY = np.empty(0, np.int64)


def match_vec(a, b, tol):
    """Vectorized exact equivalent of build_field.match().

    Same semantics: for each wrap radius in `a`, consider the two nearest
    candidates in sorted `b` (searchsorted neighbors, first candidate wins
    ties), accept if within tol, then dedupe to 1-to-1 keeping the closest
    source per target (earliest source wins ties). Returns (ii, jj) index
    arrays instead of a list of tuples.
    """
    na, nb = len(a), len(b)
    if not na or not nb:
        return _EMPTY, _EMPTY
    pos = np.searchsorted(b, a)
    j1 = pos - 1
    j2 = pos
    d1 = np.where(j1 >= 0, np.abs(b[np.clip(j1, 0, nb - 1)] - a), np.inf)
    d2 = np.where(j2 < nb, np.abs(b[np.clip(j2, 0, nb - 1)] - a), np.inf)
    take2 = d2 < d1  # j1 is tried first; j2 replaces only if strictly closer
    best = np.where(take2, j2, j1)
    bd = np.where(take2, d2, d1)
    ok = bd < tol
    if not ok.any():
        return _EMPTY, _EMPTY
    ii = np.nonzero(ok)[0]
    jj = best[ii]
    bd = bd[ii]
    # 1-to-1 dedupe per target j: keep smallest distance, ties -> earliest i
    order = np.lexsort((ii, bd, jj))
    js = jj[order]
    first = np.ones(len(js), bool)
    first[1:] = js[1:] != js[:-1]
    sel = order[first]
    sel.sort()
    return ii[sel], jj[sel]


def self_test():
    """Verify match_vec is exactly build_field.match on random inputs."""
    rng = np.random.default_rng(42)
    for _ in range(500):
        a = np.sort(rng.uniform(0, 50, rng.integers(0, 12)))
        b = np.sort(rng.uniform(0, 50, rng.integers(0, 12)))
        tol = rng.uniform(0.1, 8.0)
        ref = sorted(match_ref(a, b, tol))
        ii, jj = match_vec(a, b, tol)
        if ref != sorted(zip(ii.tolist(), jj.tolist())):
            raise AssertionError("match_vec diverged from build_field.match")
    for _ in range(200):  # realistic wrap spacing ~17
        n = int(rng.integers(2, 60))
        a = np.cumsum(rng.uniform(10, 24, n)).astype(np.float32)
        b = np.sort(a + rng.normal(0, 4, n)).astype(np.float32)
        ref = sorted(match_ref(a, b, 7.65))
        ii, jj = match_vec(a, b, 7.65)
        if ref != sorted(zip(ii.tolist(), jj.tolist())):
            raise AssertionError("match_vec diverged from build_field.match")
    print("self-test: match_vec == build_field.match on 700 random cases OK",
          flush=True)


def compute_components(Rf):
    """build_field.py pass 1 over the NaN-padded (nz, NTH, MAXW) array.

    Returns (labels, counts, tol, n_edges, chained): int32 component label
    per valid wrap (flattened in ray order), per-ray wrap counts, the
    tolerance used, the number of matched edges, and a boolean vector saying
    whether each wrap participates in at least one edge.
    """
    nz, NTH, _ = Rf.shape
    fin = np.isfinite(Rf)
    cnt = fin.sum(axis=-1).astype(np.int64)  # valid wraps form a sorted prefix
    N = int(cnt.sum())
    print(f"nodes {N}", flush=True)

    # tolerance exactly as build_field.py: median wrap spacing at the middle
    # z-level, over rays with >3 wraps
    zmid = nz // 2
    diffs = [np.diff(Rf[zmid, t, :cnt[zmid, t]])
             for t in range(NTH) if cnt[zmid, t] > 3]
    med = float(np.median(np.concatenate(diffs)))
    tol = 0.45 * med
    print(f"median wrap spacing {med:.1f} -> tol {tol:.2f}", flush=True)

    base = np.zeros((nz, NTH), np.int64)
    base.flat[1:] = np.cumsum(cnt.flat)[:-1]

    # collect all matched pairs (theta-adjacent incl. seam, then z-adjacent)
    eu = np.empty(2 * N, np.int32)  # upper bound: <=1 theta + <=1 z match/node
    ev = np.empty(2 * N, np.int32)
    chained = np.zeros(N, dtype=bool)
    k = 0
    t0 = time.time()
    for zi in range(nz):
        Rz, cz, bz = Rf[zi], cnt[zi], base[zi]
        for t in range(NTH):
            t2 = (t + 1) % NTH
            ii, jj = match_vec(Rz[t, :cz[t]], Rz[t2, :cz[t2]], tol)
            m = len(ii)
            if m:
                eu[k:k + m] = bz[t] + ii
                ev[k:k + m] = bz[t2] + jj
                chained[bz[t] + ii] = True
                chained[bz[t2] + jj] = True
                k += m
        if zi % 200 == 0:
            print(f"theta edges z {zi}/{nz}", flush=True)
    for zi in range(nz - 1):
        Rz, cz, bz = Rf[zi], cnt[zi], base[zi]
        Rz2, cz2, bz2 = Rf[zi + 1], cnt[zi + 1], base[zi + 1]
        for t in range(NTH):
            ii, jj = match_vec(Rz[t, :cz[t]], Rz2[t, :cz2[t]], tol)
            m = len(ii)
            if m:
                eu[k:k + m] = bz[t] + ii
                ev[k:k + m] = bz2[t] + jj
                chained[bz[t] + ii] = True
                chained[bz2[t] + jj] = True
                k += m
        if zi % 200 == 0:
            print(f"z edges z {zi}/{nz}", flush=True)
    print(f"{k} edges in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    g = coo_matrix((np.ones(k, np.int8), (eu[:k], ev[:k])), shape=(N, N))
    ncomp, labels = connected_components(g, directed=False)
    print(f"connected_components: {ncomp} components in "
          f"{time.time() - t0:.0f}s", flush=True)
    return labels, cnt, tol, k, chained


def per_level_coverage(labels, counts):
    """Fraction of wraps in the largest component for every z level."""
    sizes = np.bincount(labels)
    largest = int(np.argmax(sizes))
    level_sizes = counts.sum(axis=1).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(level_sizes[:-1])])
    coverage = []
    for start, size in zip(starts, level_sizes):
        level = labels[start:start + size]
        coverage.append(float((level == largest).mean()) if size else 0.0)
    return np.asarray(coverage, dtype=np.float64)


def render(Rf, zs, labels, cnt, largest_pct, chained_pct, coverage_median,
           n_nodes, n_comp, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    nz, NTH, _ = Rf.shape
    base = np.zeros((nz, NTH), np.int64)
    base.flat[1:] = np.cumsum(cnt.flat)[:-1]
    sizes = np.bincount(labels)
    giant = int(sizes.argmax())
    in_giant = labels == giant

    # ---- panel B data: per-ray majority membership over the full volume ----
    # (an "any smaller-component wrap" criterion would flood: nearly every
    # ray carries ~1 such wrap, typically its outermost one, so we color each
    # ray by where the MAJORITY of its wraps live)
    ray_idx = np.repeat(np.arange(nz * NTH, dtype=np.int64), cnt.ravel())
    off_per_ray = np.bincount(ray_idx[~in_giant], minlength=nz * NTH)
    tot_per_ray = cnt.ravel()
    nonempty = tot_per_ray > 0
    majority_off = nonempty & (2 * off_per_ray > tot_per_ray)
    img = np.zeros(nz * NTH, np.uint8)              # 0 = no wraps
    img[nonempty] = 1                               # 1 = majority in largest
    img[majority_off] = 2                           # 2 = majority in smaller
    img = img.reshape(nz, NTH)
    pct_maj_off = majority_off.sum() / max(nonempty.sum(), 1) * 100
    print(f"rays whose majority of wraps is outside the largest component: "
          f"{int(majority_off.sum())} ({pct_maj_off:.1f}% of non-empty rays)",
          flush=True)

    # ---- panel A data: one z-slice, polar -> cartesian ---------------------
    zi = nz // 2
    th = np.arange(NTH) / NTH * 2 * np.pi
    xs, ys, gg = [], [], []
    for t in range(NTH):
        c = cnt[zi, t]
        if not c:
            continue
        rr = Rf[zi, t, :c]
        xs.append(rr * np.cos(th[t]))
        ys.append(rr * np.sin(th[t]))
        gg.append(in_giant[base[zi, t]:base[zi, t] + c])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    gg = np.concatenate(gg)

    # ---- layout -------------------------------------------------------------
    BG, GREEN, MAGENTA, DIM = "#000000", "#00e070", "#ff40ff", "#1c1c1c"
    fig = plt.figure(figsize=(13.4, 9.2), facecolor=BG)
    gs = fig.add_gridspec(1, 2, left=0.065, right=0.97, top=0.735,
                          bottom=0.20, wspace=0.18, width_ratios=[1.0, 1.05])

    fig.text(0.5, 0.975,
             "Wrap-adjacency GRAPH CONNECTIVITY (validated result)\n"
             "— NOT a winding-number assignment —",
             ha="center", va="top", color="w", fontsize=16, fontweight="bold",
             linespacing=1.4)
    fig.text(0.5, 0.885,
             "Union-find over radius-proximity wrap matches (build_field.py "
             "pass 1, recomputed from scratch by figures/make_field_figure.py)"
             f" on Scroll 3, quarter scale:\n{chained_pct:.2f}% of wraps have "
             f"an edge; largest connected component = {largest_pct:.2f}% "
             f"of {n_nodes:,} wraps; median per-z coverage = "
             f"{coverage_median:.2f}% ({n_comp:,} components total)",
             ha="center", va="top", color="#cccccc", fontsize=10,
             linespacing=1.5)
    fig.text(0.5, 0.822,
             "Connectivity is the POSITIVE sub-result. Assigning an integer "
             "winding number ON TOP of this graph FAILED\n(43.8–45.8% "
             "GATE3 disagreement after the sign correction) — see "
             "docs/NEGATIVE_RESULT.md",
             ha="center", va="top", color="#ffb340", fontsize=11,
             fontweight="bold", linespacing=1.5)

    # panel A: cross-section scatter
    axA = fig.add_subplot(gs[0, 0], facecolor=BG)
    axA.scatter(xs[gg], ys[gg], s=0.35, c=GREEN, lw=0, rasterized=True)
    if (~gg).any():
        axA.scatter(xs[~gg], ys[~gg], s=0.9, c=MAGENTA, lw=0, rasterized=True)
    axA.set_aspect("equal")
    axA.set_title(f"One z-slice (z = {int(zs[zi])}): wraps in the\n"
                  "scroll cross-section, per-wrap component", color="w",
                  fontsize=10.5)
    axA.set_xlabel("x (voxels from axis)", color="#aaaaaa", fontsize=9)
    axA.set_ylabel("y (voxels from axis)", color="#aaaaaa", fontsize=9)
    axA.tick_params(colors="#888888", labelsize=8)
    for s in axA.spines.values():
        s.set_color("#444444")

    # panel B: full-volume theta x z ray map
    axB = fig.add_subplot(gs[0, 1], facecolor=BG)
    cmap = ListedColormap([DIM, GREEN, MAGENTA])
    axB.imshow(img, cmap=cmap, vmin=0, vmax=2, aspect="auto",
               origin="lower", interpolation="nearest",
               extent=[0, 360, float(zs[0]), float(zs[-1])])
    axB.set_title("Full volume (θ × z): per-ray MAJORITY component\n"
                  f"({pct_maj_off:.1f}% of non-empty rays majority-magenta)",
                  color="w", fontsize=10.5)
    axB.set_xlabel("ray angle θ (degrees)", color="#aaaaaa", fontsize=9)
    axB.set_ylabel("z (voxels)", color="#aaaaaa", fontsize=9)
    axB.tick_params(colors="#888888", labelsize=8)
    for s in axB.spines.values():
        s.set_color("#444444")

    legend_items = [
        Line2D([], [], marker="o", ls="", ms=7, mfc=GREEN, mec=GREEN,
               label=f"LARGEST connected component ({largest_pct:.1f}% of "
                     "wraps) — connected in the graph, winding number still "
                     "UNSOLVED"),
        Line2D([], [], marker="o", ls="", ms=7, mfc=MAGENTA, mec=MAGENTA,
               label="smaller components, all pooled "
                     f"({100 - largest_pct:.1f}% of wraps; mostly each ray's "
                     "outermost wrap)"),
        Line2D([], [], marker="s", ls="", ms=7, mfc=DIM, mec="#555555",
               label="no wrap detected (empty ray / outside scroll)"),
    ]
    leg = fig.legend(handles=legend_items, loc="lower center",
                     bbox_to_anchor=(0.5, 0.052), ncol=1, frameon=False,
                     fontsize=10)
    for t in leg.get_texts():
        t.set_color("#dddddd")

    fig.text(0.5, 0.022,
             "Green means CONNECTED, not SOLVED: this figure shows graph "
             "connectivity only and must not be read as a working winding "
             "field w(x).",
             ha="center", va="bottom", color="#ffb340", fontsize=10,
             style="italic")

    fig.savefig(out_path, dpi=150, facecolor=BG)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", default=os.path.join(REPO_ROOT,
                                                  "wfield_multi.npz"),
                    help="path to wfield_multi.npz (release artifact; "
                         "see README.md)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "field_figure.png"))
    a = ap.parse_args()

    self_test()
    D = np.load(a.npz)
    Rf, zs = D["r"], D["zs"]  # stored w/comp are exploratory; ignored (above)
    print(f"r {Rf.shape} zs {zs.shape}", flush=True)

    labels, cnt, tol, n_edges, chained = compute_components(Rf)
    n_nodes = len(labels)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    largest_pct = sizes[order[0]] / n_nodes * 100
    chained_pct = float(chained.mean() * 100)
    coverage = per_level_coverage(labels, cnt)
    coverage_median = float(np.median(coverage) * 100)
    coverage_p10 = float(np.percentile(coverage, 10) * 100)
    print("top-5 component sizes:",
          [f"{int(sizes[c])} ({sizes[c] / n_nodes * 100:.2f}%)"
           for c in order[:5]], flush=True)
    print(f"RESULT chained {chained_pct:.4f}% | largest component "
          f"{largest_pct:.4f}% of {n_nodes} wraps | per-z coverage median "
          f"{coverage_median:.4f}% p10 {coverage_p10:.4f}% | "
          f"{len(sizes)} components | {n_edges} edges", flush=True)

    if abs(largest_pct - EXPECTED_LARGEST_PCT) > GATE_TOLERANCE_PCT:
        print(f"VALIDATION GATE FAILED: recomputed largest component "
              f"{largest_pct:.2f}% is not within {GATE_TOLERANCE_PCT} pt of "
              f"the documented {EXPECTED_LARGEST_PCT}%. Refusing to render a "
              "figure that contradicts README.md / docs/NEGATIVE_RESULT.md — "
              "investigate first.", flush=True)
        sys.exit(1)
    print(f"validation gate OK (documented {EXPECTED_LARGEST_PCT}%)",
          flush=True)

    if abs(chained_pct - EXPECTED_CHAINED_PCT) > CHAIN_GATE_TOLERANCE_PCT:
        print(f"VALIDATION GATE FAILED: recomputed chained fraction "
              f"{chained_pct:.4f}% is not within "
              f"{CHAIN_GATE_TOLERANCE_PCT} pt of the documented "
              f"{EXPECTED_CHAINED_PCT}%.", flush=True)
        sys.exit(1)
    print(f"chained validation gate OK (documented "
          f"{EXPECTED_CHAINED_PCT}%)", flush=True)

    render(Rf, zs, labels, cnt, largest_pct, chained_pct, coverage_median,
           n_nodes, len(sizes), a.out)


if __name__ == "__main__":
    main()
