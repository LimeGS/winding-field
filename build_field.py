"""Build the global winding field from a winding map r(z, theta, wrap).

Two passes, both over the same wrap graph (radius-proximity matches between
adjacent rays in theta and z):
  1. Union-find -> connected components. This pass backs the reproducible
     connectivity claims in docs/NEGATIVE_RESULT.md (96.6671% in the largest
     component on the release artifact). Connectivity does not establish
     that every candidate edge preserves physical sheet identity.
  2. Per component, one plain FIFO BFS assigns w with a +1/-1 branch cut per
     revolution (chirality is not assumed: both are tried on redundant edges
     and the conflict-free one wins).

IMPORTANT: pass 2's plain-FIFO `assign()` is a simple reference
implementation, NOT the confidence-priority method described as "Method 1"
in docs/NEGATIVE_RESULT.md's 5-method table -- that method (a real
priority-queue spanning tree, quality-guided by edge confidence) is
`method1_confidence_mst.py` in this same directory, added 2026-07-08 after
an adversarial review found this file's `assign()` didn't match its own
prior docstring's description. Use this script for the graph (pass 1); use
the `method*_corrected.py` scripts to reproduce the corrected winding-
assignment numbers. The un-suffixed method scripts are retained only to
reproduce the original sign-bugged report.

Input : rmap.npy  (Z, NTH, MAXW) float32, wrap radii per ray, -1 = empty
Output: field.npz (w, labels, r, zs)

Usage:  python build_field.py --rmap rmap_full.npy --zstep 8 --out field.npz
"""
import argparse
from collections import Counter, deque

import numpy as np


def match(a, b, tol):
    """1-to-1 radius-proximity matching between two sorted wrap lists."""
    if not len(a) or not len(b):
        return []
    jj = np.searchsorted(b, a)
    out = {}
    for i, r in enumerate(a):
        best, bd = -1, tol
        for j in (jj[i] - 1, jj[i]):
            if 0 <= j < len(b) and abs(b[j] - r) < bd:
                best, bd = j, abs(b[j] - r)
        if best >= 0 and (best not in out or bd < out[best][1]):
            out[best] = (i, bd)
    return [(v[0], j) for j, v in out.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmap", required=True)
    ap.add_argument("--zstep", type=int, default=8)
    ap.add_argument("--out", default="field.npz")
    a = ap.parse_args()

    rmap = np.load(a.rmap, mmap_mode="r")
    NTH = rmap.shape[1]
    ZS = list(range(0, rmap.shape[0], a.zstep))
    rays = {}
    for k, z in enumerate(ZS):
        sl = np.asarray(rmap[z])
        rays[z] = [row[row > 0] for row in sl]
        if k % 200 == 0:
            print(f"load {k}/{len(ZS)}", flush=True)
    med = np.median(np.concatenate(
        [np.diff(r) for r in rays[ZS[len(ZS) // 2]] if len(r) > 3]))
    tol = 0.45 * med
    print(f"median wrap spacing {med:.1f}", flush=True)

    lens = {(z, t): len(rays[z][t]) for z in ZS for t in range(NTH)}
    base = {}
    n = 0
    for z in ZS:
        for t in range(NTH):
            base[(z, t)] = n
            n += lens[(z, t)]
    print("nodes", n, flush=True)

    # ---- pass 1: union-find over theta+z edges -----------------------------
    p = np.arange(n, dtype=np.int64)

    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    edges = []          # kept for pass 2 (theta edges carry the seam step)
    for z in ZS:
        for t in range(NTH):
            t2 = (t + 1) % NTH
            for i, j in match(rays[z][t], rays[z][t2], tol):
                edges.append((base[(z, t)] + i, base[(z, t2)] + j,
                              1.0 if t == NTH - 1 else 0.0))
    zidx = {z: i for i, z in enumerate(ZS)}
    for zi in range(len(ZS) - 1):
        z, z2 = ZS[zi], ZS[zi + 1]
        for t in range(NTH):
            for i, j in match(rays[z][t], rays[z2][t], tol):
                edges.append((base[(z, t)] + i, base[(z2, t)] + j, 0.0))
    for x, y, _ in edges:
        rx, ry = find(x), find(y)
        if rx != ry:
            p[ry] = rx
    roots = np.array([find(x) for x in range(n)], dtype=np.int64)
    sizes = Counter(roots.tolist())
    top = sizes.most_common(5)
    print(f"components {len(sizes)} | largest {top[0][1]} "
          f"({top[0][1] / n * 100:.1f}%)", flush=True)

    # ---- pass 2: w assignment with chirality selection ---------------------
    adj = [[] for _ in range(n)]
    for x, y, s in edges:
        adj[x].append((y, s))
        adj[y].append((x, -s))

    def assign(chir):
        w = np.full(n, np.nan, np.float32)
        conf = tot = 0
        for seed in [top[0][0]] + [r for r, _ in top[1:]]:
            if np.isfinite(w[seed]):
                continue
            w[seed] = 0.0
            q = deque([seed])
            while q:
                x = q.popleft()
                for y, s in adj[x]:
                    cand = w[x] + chir * s
                    tot += 1
                    if np.isnan(w[y]):
                        w[y] = cand
                        q.append(y)
                    elif abs(w[y] - cand) > 0.01:
                        conf += 1
        return w, conf, tot

    best = None
    for chir in (+1.0, -1.0):
        w, conf, tot = assign(chir)
        print(f"chirality {chir:+.0f}: conflicts {conf}/{tot} "
              f"({conf / max(tot, 1) * 100:.3f}%)", flush=True)
        if best is None or conf < best[1]:
            best = (w, conf, chir)
    w, conf, chir = best
    print(f"selected chirality {chir:+.0f}", flush=True)

    # remaining components: seed each (local w only)
    todo = np.nonzero(np.isnan(w))[0]
    for seed in todo:
        if np.isfinite(w[seed]):
            continue
        w[seed] = 0.0
        q = deque([int(seed)])
        while q:
            x = q.popleft()
            for y, s in adj[x]:
                if np.isnan(w[y]):
                    w[y] = w[x] + chir * s
                    q.append(y)

    # ---- pack back to (Z, NTH, MAXW) ---------------------------------------
    mx = max(lens.values())
    A = np.full((len(ZS), NTH, mx), np.nan, np.float32)
    L = np.full((len(ZS), NTH, mx), -1, np.int64)
    R = np.full((len(ZS), NTH, mx), np.nan, np.float32)
    for z in ZS:
        for t in range(NTH):
            k = lens[(z, t)]
            if not k:
                continue
            b = base[(z, t)]
            A[zidx[z], t, :k] = w[b:b + k]
            L[zidx[z], t, :k] = roots[b:b + k]
            R[zidx[z], t, :k] = rays[z][t]
    np.savez_compressed(a.out, w=A, labels=L, r=R, zs=np.array(ZS))
    print(f"BUILD_FIELD_DONE {a.out} (assigned "
          f"{np.isfinite(A).sum() / max(n, 1) * 100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
