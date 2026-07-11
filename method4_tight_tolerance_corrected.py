# SIGN-CORRECTED RERUN 2026-07-10: fixes the confirmed sign-convention bug in offset propagation.
# Changed lines vs original: solve_mst `off[v]=off[u]-m+chir*s` and GATE2 check `cand=off[u]-m+chir*s`
# (both were `+m`); output renamed to ./wfield_final_signfix_m4.npz. No other changes.
"""PROVENANCE (added 2026-07-08): this is the script behind "Method 4:
Tightened tolerance (0.25x vs 0.45x)" in docs/NEGATIVE_RESULT.md's 5-method
table (68.5% disagreement rate, 69% mesh-audit jump rate) -- originally
`data/wfield/wfield15_tol.py` in the parent project repo. It is identical
to method1_confidence_mst.py (v10.4) except for one constant:
`tol = 0.25 * 17.0` here vs `tol = 0.45 * 17.0` there (line ~66) -- that
single-line diff is the entire experiment. Logic, structure, and every
other constant/number are unchanged from the original research script;
comments, docstrings, and log/print strings have been translated from
Spanish to English for publication (the original Spanish-language version
remains in the parent project repo at the path above if byte-for-byte
provenance is needed). Hardcoded local paths reflect the original
research-machine layout; adjust before rerunning.

Original docstring (translated):

v10.4: same preprocessing (doublet-merge + gap-correct) as v10.3, but
replaces the hard purity>=0.75 gate + naive BFS with a MAX-CONFIDENCE
SPANNING TREE (quality-guided path following, the standard InSAR/
phase-unwrapping method for this exact problem): ALL edges with >=5 votes
are used, weighted by purity, propagating first along the
highest-confidence ones (Prim-style priority queue). GATE2 now measures
loop closure over ALL non-tree edges (previously it only averaged over
the 4-84 edges the naive BFS reached -- now over thousands)."""
import numpy as np, heapq
from collections import Counter
D=np.load("./wfield_multi.npz")
Rf,ZS=D["r"],D["zs"]; NTH=Rf.shape[1]; nz=len(ZS)
fin=np.isfinite(Rf)
CL=[[None]*NTH for _ in range(nz)]
for zi in range(nz):
    for tt in range(NTH):
        rr=Rf[zi,tt][fin[zi,tt]].copy()
        if len(rr)>=4:
            d=np.diff(rr); med=np.median(d)
            keep=[0]
            for k in range(1,len(rr)):
                if rr[k]-rr[keep[-1]]<0.55*med:
                    rr[keep[-1]]=(rr[keep[-1]]+rr[k])/2
                else: keep.append(k)
            rr=rr[keep]
        CL[zi][tt]=rr
    if zi%300==0: print("cleanup",zi,flush=True)
lens=np.array([[len(CL[zi][tt]) for tt in range(NTH)] for zi in range(nz)],dtype=np.int64)
base=np.zeros((nz,NTH),np.int64); base.flat[1:]=np.cumsum(lens.flat)[:-1]
N=int(lens.sum()); print("nodes after cleanup",N,flush=True)
Rflat=np.empty(N,np.float32); IP=np.empty(N,np.float32)
for zi in range(nz):
    for tt in range(NTH):
        n=lens[zi,tt]
        if not n: continue
        b=base[zi,tt]; rr=CL[zi][tt]
        Rflat[b:b+n]=rr
        if n>=3:
            d=np.diff(rr)
            med=np.array([np.median(d[max(0,k-2):k+3]) for k in range(len(d))])
            gaps=np.clip(np.round(d/np.maximum(med,1e-3))-1,0,2)
            g=np.concatenate([[0],np.cumsum(gaps)])
        else:
            g=np.zeros(n)
        IP[b:b+n]=np.arange(n)+g
    if zi%300==0: print("gaps",zi,flush=True)
print("corrected indices ready",flush=True)
tol=0.25*17.0
def match(a,b):
    if not len(a) or not len(b): return []
    jj=np.searchsorted(b,a); out={}
    for i,r in enumerate(a):
        best,bd=-1,tol
        for j in (jj[i]-1,jj[i]):
            if 0<=j<len(b) and abs(b[j]-r)<bd: best,bd=j,abs(b[j]-r)
        if best>=0 and (best not in out or bd<out[best][1]): out[best]=(i,bd)
    return [(v[0],j) for j,v in out.items()]
def ray(zi,tt):
    n=lens[zi,tt]; b=base[zi,tt]; return Rflat[b:b+n],IP[b:b+n],b
NB={}; purities=[]
eu=[];ev=[];es=[]
for zi in range(nz):
    for tt in range(NTH):
        t2=(tt+1)%NTH
        a,ia,ba=ray(zi,tt); b2,ib,bb=ray(zi,t2)
        ms=match(a,b2)
        s=1.0 if tt==NTH-1 else 0.0
        for i,j in ms:
            eu.append(ba+i); ev.append(bb+j); es.append(s)
        if len(ms)>=5:
            vs=[round(float(ib[j]-ia[i])) for i,j in ms]
            c=Counter(vs); m,nm=c.most_common(1)[0]
            pu=nm/len(vs); purities.append(pu)
            NB.setdefault(ba,[]).append((bb,m,s,pu))
            NB.setdefault(bb,[]).append((ba,-m,-s,pu))
    if zi%200==0: print("th",zi,flush=True)
for zi in range(nz-1):
    for tt in range(NTH):
        a,ia,ba=ray(zi,tt); b2,ib,bb=ray(zi+1,tt)
        ms=match(a,b2)
        for i,j in ms:
            eu.append(ba+i); ev.append(bb+j); es.append(0.0)
        if len(ms)>=5:
            vs=[round(float(ib[j]-ia[i])) for i,j in ms]
            c=Counter(vs); m,nm=c.most_common(1)[0]
            pu=nm/len(vs); purities.append(pu)
            NB.setdefault(ba,[]).append((bb,m,0.0,pu))
            NB.setdefault(bb,[]).append((ba,-m,0.0,pu))
    if zi%200==0: print("z",zi,flush=True)
purities=np.array(purities)
print(f"GATE1 purity: median {np.median(purities):.3f} p10 {np.percentile(purities,10):.3f}",flush=True)
print(f"rays with edges: {len(NB)} / {(lens>0).sum()}",flush=True)

def solve_mst(chir):
    seed=base[nz//2,0]
    off={seed:0.0}
    h=[]
    for (v,m,s,pu) in NB.get(seed,[]): heapq.heappush(h,(-pu,seed,v,m,s))
    treeedges=set()
    while h:
        negpu,u,v,m,s=heapq.heappop(h)
        if v in off: continue
        off[v]=off[u]-m+chir*s
        treeedges.add((u,v)); treeedges.add((v,u))
        for (w,m2,s2,pu2) in NB.get(v,[]):
            if w not in off: heapq.heappush(h,(-pu2,v,w,m2,s2))
    conf=tot=0
    for u,edges in NB.items():
        if u not in off: continue
        for (v,m,s,pu) in edges:
            if v not in off or (u,v) in treeedges: continue
            tot+=1
            cand=off[u]-m+chir*s
            if abs(off[v]-cand)>0.5: conf+=1
    return off,conf,tot

res={}
for chir in (1,-1):
    off,conf,tot=solve_mst(chir)
    res[chir]=(off,conf,tot)
    print(f"GATE2 chir {chir:+d}: conflicts {conf}/{tot} ({conf/max(tot,1)*100:.2f}%) rays_reached {len(off)}/{(lens>0).sum()}",flush=True)
chir=min(res,key=lambda c:res[c][1]/max(res[c][2],1)); off,_,_=res[chir]
print(f"chirality chosen {chir:+d}",flush=True)
Wseed=np.full(N,np.nan,np.float32)
for zi in range(nz):
    for tt in range(NTH):
        n=lens[zi,tt]
        if not n: continue
        b=base[zi,tt]
        if b in off: Wseed[b:b+n]=IP[b:b+n]+off[b]
print(f"nodes with w assigned: {np.isfinite(Wseed).sum()}/{N}",flush=True)
eu=np.array(eu,np.int64); ev=np.array(ev,np.int64); es=np.array(es,np.float32)*chir
val=np.isfinite(Wseed[eu])&np.isfinite(Wseed[ev])
kill=np.abs(Wseed[ev]-(Wseed[eu]+es))>0.5
kr=float((kill&val).sum())/max(int(val.sum()),1)
print(f"GATE3 kill-rate: {kr*100:.1f}% (over {int(val.sum())} valid edges)",flush=True)
keep=val&~kill
eu,ev,es=eu[keep],ev[keep],es[keep]
W=Wseed.copy()
for sweep in range(4):
    acc=np.zeros(N,np.float64); cnt=np.zeros(N,np.float64)
    np.add.at(acc,ev,W[eu]+es); np.add.at(cnt,ev,1)
    np.add.at(acc,eu,W[ev]-es); np.add.at(cnt,eu,1)
    fin_w=np.isfinite(W)
    acc[fin_w]+=W[fin_w]; cnt[fin_w]+=1
    Wn=np.where(cnt>0,(acc/np.maximum(cnt,1)).astype(np.float32),W)
    d=float(np.nanmean(np.abs(Wn[fin_w]-W[fin_w])))
    W=Wn
    print(f"sweep {sweep}: delta {d:.4f}",flush=True)
    if d<0.002: break
mx=int(lens.max())
A=np.full((nz,NTH,mx),np.nan,np.float32)
for zi in range(nz):
    for tt in range(NTH):
        n=lens[zi,tt]
        if n: A[zi,tt,:n]=W[base[zi,tt]:base[zi,tt]+n]
Rc=np.full((nz,NTH,mx),np.nan,np.float32)
for zi in range(nz):
    for tt in range(NTH):
        n=lens[zi,tt]
        if n: Rc[zi,tt,:n]=CL[zi][tt]
np.savez_compressed("./wfield_final_signfix_m4.npz",w=A,r=Rc,zs=ZS)
zi=nz//2; wz=A[zi][np.isfinite(A[zi])]
print(f"w z-mid: unique {len(np.unique(np.round(wz)))} range {wz.min():.0f}..{wz.max():.0f}" if len(wz) else "w z-mid: EMPTY",flush=True)
print("WFIELD13_DONE",flush=True)
