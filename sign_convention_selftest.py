"""Synthetic self-test for the sign-convention bug in the winding-assignment
propagation (method1_confidence_mst.py and siblings), 2026-07-10.

Bug under test: per-edge modal offset m = mode(IP_dest - IP_origin) over
matched pairs, propagated as off[v] = off[u] + m + chir*s (original code,
SIGN=+1). Algebra for a matched same-sheet pair requires
off[v] = off[u] - m + chir*s (corrected, SIGN=-1), since W = IP + off and
W_v(j) = W_u(i) + chir*s  =>  off[v] = off[u] + (IP_u(i) - IP_v(j)) + chir*s.

Dataset: clean Archimedean spiral r(k,theta) = 30 + 17*(k + theta/2pi),
k = 0..K-1, identical across z, small jitter << tol. Failure trigger: rays
tt in [30,60) at all z have the INNERMOST wrap (k=0) deleted, so their IP
indices shift by -1 vs neighbors and the modal offset at the block
boundaries is m = -/+1 (gap correction cannot restore a missing innermost
wrap; it only fixes interior gaps).

Pipeline below is copied verbatim from
release/winding-field/method1_confidence_mst.py (cleanup, gap-correct,
match(), NB build, solve_mst, chirality choice, W = IP + off, GATE3),
parameterized only by SIGN in the two propagation lines.

Assertions:
  A1 corrected recovers w_true up to a global additive constant (>99%,
     including inside the deleted-wrap block)
  A2 original shows magnitude-2 W errors on rays reached through the block
     boundary (the block sits at mode(e_full) - 2 or +2)
  A3 GATE1 purity identical between runs (it precedes propagation)
  A4 GATE3 kill-rate ~0% corrected, clearly >0% original
"""
import numpy as np, heapq
from collections import Counter

# ---------- synthetic dataset ----------
nz, NTH, K = 8, 96, 12
SP = 17.0
BLOCK_LO, BLOCK_HI = 30, 60          # rays with innermost wrap deleted
rng = np.random.default_rng(42)
Rf = np.full((nz, NTH, K), np.nan, np.float32)
WT = np.full((nz, NTH, K), np.nan, np.float32)   # ground-truth winding index
for zi in range(nz):
    for tt in range(NTH):
        ks = np.arange(K)
        if BLOCK_LO <= tt < BLOCK_HI:
            ks = ks[1:]              # delete innermost wrap k=0
        r = 30.0 + SP * (ks + tt / NTH) + rng.uniform(-0.2, 0.2, len(ks))
        Rf[zi, tt, :len(ks)] = r
        WT[zi, tt, :len(ks)] = ks
ZS = np.arange(nz)

def run_pipeline(SIGN):
    """SIGN=+1 reproduces the original code; SIGN=-1 is the corrected algebra.
    Everything else is verbatim from method1_confidence_mst.py."""
    fin = np.isfinite(Rf)
    CL = [[None] * NTH for _ in range(nz)]
    for zi in range(nz):
        for tt in range(NTH):
            rr = Rf[zi, tt][fin[zi, tt]].copy()
            if len(rr) >= 4:
                d = np.diff(rr); med = np.median(d)
                keep = [0]
                for k in range(1, len(rr)):
                    if rr[k] - rr[keep[-1]] < 0.55 * med:
                        rr[keep[-1]] = (rr[keep[-1]] + rr[k]) / 2
                    else: keep.append(k)
                rr = rr[keep]
            CL[zi][tt] = rr
    lens = np.array([[len(CL[zi][tt]) for tt in range(NTH)] for zi in range(nz)], dtype=np.int64)
    base = np.zeros((nz, NTH), np.int64); base.flat[1:] = np.cumsum(lens.flat)[:-1]
    N = int(lens.sum())
    Rflat = np.empty(N, np.float32); IP = np.empty(N, np.float32)
    for zi in range(nz):
        for tt in range(NTH):
            n = lens[zi, tt]
            if not n: continue
            b = base[zi, tt]; rr = CL[zi][tt]
            Rflat[b:b+n] = rr
            if n >= 3:
                d = np.diff(rr)
                med = np.array([np.median(d[max(0, k-2):k+3]) for k in range(len(d))])
                gaps = np.clip(np.round(d / np.maximum(med, 1e-3)) - 1, 0, 2)
                g = np.concatenate([[0], np.cumsum(gaps)])
            else:
                g = np.zeros(n)
            IP[b:b+n] = np.arange(n) + g
    tol = 0.45 * 17.0
    def match(a, b):
        if not len(a) or not len(b): return []
        jj = np.searchsorted(b, a); out = {}
        for i, r in enumerate(a):
            best, bd = -1, tol
            for j in (jj[i]-1, jj[i]):
                if 0 <= j < len(b) and abs(b[j]-r) < bd: best, bd = j, abs(b[j]-r)
            if best >= 0 and (best not in out or bd < out[best][1]): out[best] = (i, bd)
        return [(v[0], j) for j, v in out.items()]
    def ray(zi, tt):
        n = lens[zi, tt]; b = base[zi, tt]; return Rflat[b:b+n], IP[b:b+n], b
    NB = {}; purities = []
    eu = []; ev = []; es = []
    for zi in range(nz):
        for tt in range(NTH):
            t2 = (tt+1) % NTH
            a, ia, ba = ray(zi, tt); b2, ib, bb = ray(zi, t2)
            ms = match(a, b2)
            s = 1.0 if tt == NTH-1 else 0.0
            for i, j in ms:
                eu.append(ba+i); ev.append(bb+j); es.append(s)
            if len(ms) >= 5:
                vs = [round(float(ib[j]-ia[i])) for i, j in ms]
                c = Counter(vs); m, nm = c.most_common(1)[0]
                pu = nm/len(vs); purities.append(pu)
                NB.setdefault(ba, []).append((bb, m, s, pu))
                NB.setdefault(bb, []).append((ba, -m, -s, pu))
    for zi in range(nz-1):
        for tt in range(NTH):
            a, ia, ba = ray(zi, tt); b2, ib, bb = ray(zi+1, tt)
            ms = match(a, b2)
            for i, j in ms:
                eu.append(ba+i); ev.append(bb+j); es.append(0.0)
            if len(ms) >= 5:
                vs = [round(float(ib[j]-ia[i])) for i, j in ms]
                c = Counter(vs); m, nm = c.most_common(1)[0]
                pu = nm/len(vs); purities.append(pu)
                NB.setdefault(ba, []).append((bb, m, 0.0, pu))
                NB.setdefault(bb, []).append((ba, -m, 0.0, pu))
    purities = np.array(purities)

    def solve_mst(chir):
        seed = base[nz//2, 0]
        off = {seed: 0.0}
        h = []
        for (v, m, s, pu) in NB.get(seed, []): heapq.heappush(h, (-pu, seed, v, m, s))
        treeedges = set()
        while h:
            negpu, u, v, m, s = heapq.heappop(h)
            if v in off: continue
            off[v] = off[u] + SIGN*m + chir*s          # <-- the line under test
            treeedges.add((u, v)); treeedges.add((v, u))
            for (w, m2, s2, pu2) in NB.get(v, []):
                if w not in off: heapq.heappush(h, (-pu2, v, w, m2, s2))
        conf = tot = 0
        for u, edges in NB.items():
            if u not in off: continue
            for (v, m, s, pu) in edges:
                if v not in off or (u, v) in treeedges: continue
                tot += 1
                cand = off[u] + SIGN*m + chir*s        # <-- the line under test
                if abs(off[v]-cand) > 0.5: conf += 1
        return off, conf, tot

    res = {}
    gate2 = {}
    for chir in (1, -1):
        off, conf, tot = solve_mst(chir)
        res[chir] = (off, conf, tot)
        gate2[chir] = (conf, tot)
    chir = min(res, key=lambda c: res[c][1] / max(res[c][2], 1)); off, _, _ = res[chir]
    Wseed = np.full(N, np.nan, np.float32)
    for zi in range(nz):
        for tt in range(NTH):
            n = lens[zi, tt]
            if not n: continue
            b = base[zi, tt]
            if b in off: Wseed[b:b+n] = IP[b:b+n] + off[b]
    eu = np.array(eu, np.int64); ev = np.array(ev, np.int64); es = np.array(es, np.float32) * chir
    val = np.isfinite(Wseed[eu]) & np.isfinite(Wseed[ev])
    kill = np.abs(Wseed[ev] - (Wseed[eu] + es)) > 0.5
    kr = float((kill & val).sum()) / max(int(val.sum()), 1)
    return dict(purities=purities, gate2=gate2, chir=chir, W=Wseed, kill=kr,
                lens=lens, base=base, N=N, nval=int(val.sum()))

print("=== running ORIGINAL propagation (SIGN=+1, off[v]=off[u]+m+chir*s) ===", flush=True)
orig = run_pipeline(+1)
print("=== running CORRECTED propagation (SIGN=-1, off[v]=off[u]-m+chir*s) ===", flush=True)
fixd = run_pipeline(-1)

# ---------- flatten ground truth using the same base/lens layout ----------
lens, base, N = fixd["lens"], fixd["base"], fixd["N"]
# cleanup must not have merged/dropped anything for the truth mapping to hold
exp = np.full((nz, NTH), K, np.int64); exp[:, BLOCK_LO:BLOCK_HI] = K - 1
assert np.array_equal(lens, exp), "cleanup altered detection counts; truth mapping invalid"
wt = np.empty(N, np.float32)
is_block = np.zeros(N, bool)
for zi in range(nz):
    for tt in range(NTH):
        n = lens[zi, tt]; b = base[zi, tt]
        wt[b:b+n] = WT[zi, tt, :n]
        if BLOCK_LO <= tt < BLOCK_HI: is_block[b:b+n] = True

def mode_of(x):
    v, c = np.unique(np.round(x).astype(int), return_counts=True)
    return int(v[np.argmax(c)])

def recovery(W, name):
    """Fraction of detections whose W equals sign*w_true + const, best sign."""
    fin = np.isfinite(W)
    best = None
    for sgn in (+1, -1):
        e = W[fin] - sgn * wt[fin]
        c = mode_of(e)
        frac = float((np.abs(e - c) < 0.25).mean())
        eb = W[fin & is_block] - sgn * wt[fin & is_block]
        fb = float((np.abs(eb - c) < 0.25).mean()) if (fin & is_block).any() else 0.0
        if best is None or frac > best[1]:
            best = (sgn, frac, fb, c)
    sgn, frac, fb, c = best
    print(f"{name}: best sign {sgn:+d}, const {c:+d} | global exact-recovery "
          f"{frac*100:.2f}% | inside deleted-wrap block {fb*100:.2f}%")
    return best

print()
print(f"GATE1 purity: orig median {np.median(orig['purities']):.3f}  "
      f"fixed median {np.median(fixd['purities']):.3f}")
print(f"GATE2 orig : chir +1 {orig['gate2'][1]}  chir -1 {orig['gate2'][-1]}  -> chose {orig['chir']:+d}")
print(f"GATE2 fixed: chir +1 {fixd['gate2'][1]}  chir -1 {fixd['gate2'][-1]}  -> chose {fixd['chir']:+d}")
sg_f, rec_f, recb_f, c_f = recovery(fixd["W"], "W corrected")
sg_o, rec_o, recb_o, c_o = recovery(orig["W"], "W original ")

# block-specific error mode for the original run
fin_o = np.isfinite(orig["W"])
e_full = orig["W"][fin_o & ~is_block] - sg_o * wt[fin_o & ~is_block]
e_blk  = orig["W"][fin_o &  is_block] - sg_o * wt[fin_o &  is_block]
m_full, m_blk = mode_of(e_full), mode_of(e_blk)
blk_at_mode = float((np.abs(e_blk - m_blk) < 0.25).mean())
print(f"original: mode(err full rays) {m_full:+d}, mode(err block rays) {m_blk:+d} "
      f"(block at its mode: {blk_at_mode*100:.2f}%)")
print(f"GATE3 kill-rate: orig {orig['kill']*100:.3f}% (over {orig['nval']} pairs)  "
      f"fixed {fixd['kill']*100:.3f}% (over {fixd['nval']} pairs)")
print()

# ---------- assertions ----------
ok = True
def check(name, cond, detail):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")
    ok = ok and cond

check("A1 corrected recovers w_true (global)",
      rec_f > 0.99, f"{rec_f*100:.2f}% > 99% required")
check("A1b corrected recovers w_true inside deleted-wrap block",
      recb_f > 0.99, f"{recb_f*100:.2f}% > 99% required")
check("A2 original has magnitude-2 W errors on block rays",
      abs(m_blk - m_full) == 2 and blk_at_mode > 0.99,
      f"|mode(block)-mode(full)| = {abs(m_blk-m_full)} (need 2), block at mode {blk_at_mode*100:.2f}%")
check("A3 GATE1 purity identical between runs",
      np.array_equal(orig["purities"], fixd["purities"]),
      f"medians {np.median(orig['purities']):.3f} vs {np.median(fixd['purities']):.3f}")
check("A4a GATE3 ~0% for corrected", fixd["kill"] < 0.001, f"{fixd['kill']*100:.3f}% < 0.1% required")
check("A4b GATE3 clearly >0% for original", orig["kill"] > 0.005, f"{orig['kill']*100:.3f}% > 0.5% required")

print()
print("SYNTH_TEST", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
