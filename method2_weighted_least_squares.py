"""PROVENANCE (added 2026-07-08): this is the script behind "Method 2:
Global weighted least-squares" in docs/NEGATIVE_RESULT.md's 5-method table
(70.5% disagreement rate, 65% mesh-audit jump rate) -- originally
`data/wfield/wfield14.py` (internal version tag v10.5) in the parent
project repo. Logic, structure, and every constant/number are unchanged
from that original research script; comments, docstrings, and log/print
strings have been translated from Spanish to English for publication (the
original Spanish-language version remains in the parent project repo at
the path above if byte-for-byte provenance is needed). Requires `scipy`
(sparse least-squares solver) -- see requirements.txt. Hardcoded local
paths reflect the original research-machine layout; adjust before
rerunning. See method1_confidence_mst.py's header for the fuller
provenance note (applies identically here).

Original docstring (translated):

v10.5 (Track A - cheap idea): same preprocessing as v10.3/v10.4
(doublet-merge + gap-correct), but instead of a max-confidence spanning
tree (greedy, one single commitment per ray) does a GLOBAL WEIGHTED
LEAST-SQUARES over ALL edges simultaneously (Ghiglia-Pritt style
weighted-LS phase unwrapping): solves for each ray's offset by minimizing
the purity-weighted disagreement of ALL edges at once, instead of
committing greedily edge by edge. Tests whether the SNR ceiling
(70% disagreement in v10.4) is a problem with the CONSENSUS METHOD (fixable
with global averaging) or with the INPUT SIGNAL (not fixable without a
different source)."""
import numpy as np
from collections import Counter
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr
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
tol=0.45*17.0
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
rayids={}
next_id=0
for zi in range(nz):
    for tt in range(NTH):
        if lens[zi,tt]: rayids[base[zi,tt]]=next_id; next_id+=1
NR=next_id
print(f"rays with data: {NR}",flush=True)
EU=[];EV=[];EM=[];ES=[];EW=[]
eu=[];ev=[];es=[]
purities=[]
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
            EU.append(rayids[ba]); EV.append(rayids[bb]); EM.append(m); ES.append(s); EW.append(pu)
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
            EU.append(rayids[ba]); EV.append(rayids[bb]); EM.append(m); ES.append(0.0); EW.append(pu)
    if zi%200==0: print("z",zi,flush=True)
purities=np.array(purities)
print(f"GATE1 purity: median {np.median(purities):.3f} p10 {np.percentile(purities,10):.3f}",flush=True)
EU=np.array(EU);EV=np.array(EV);EM=np.array(EM,np.float64);ES=np.array(ES,np.float64);EW=np.array(EW,np.float64)
NE=len(EU); print(f"ray-to-ray edges: {NE}",flush=True)

def solve_lsq(chir):
    seed_ray=rayids[base[nz//2,0]]
    keep=np.arange(NE)!=-1
    rows=np.arange(NE)
    w=np.sqrt(EW)
    Au=sp.coo_matrix((-w,(rows,EU)),shape=(NE,NR))
    Av=sp.coo_matrix((w,(rows,EV)),shape=(NE,NR))
    A=(Au+Av).tocsr()
    b=w*(EM+chir*ES)
    mask=np.ones(NR,bool); mask[seed_ray]=False
    A2=A[:,mask]
    sol=lsmr(A2,b,atol=1e-6,btol=1e-6,maxiter=400)
    x=sol[0]
    O=np.zeros(NR)
    O[mask]=x
    resid=A@O-b
    return O,float(np.sqrt(np.mean(resid**2)))

res={}
for chir in (1,-1):
    O,rms=solve_lsq(chir)
    res[chir]=(O,rms)
    print(f"GATE2 chir {chir:+d}: weighted_rms_residual {rms:.4f}",flush=True)
chir=min(res,key=lambda c:res[c][1]); O,_=res[chir]
print(f"chirality chosen {chir:+d}",flush=True)
Wseed=np.full(N,np.nan,np.float32)
for zi in range(nz):
    for tt in range(NTH):
        n=lens[zi,tt]
        if not n: continue
        b=base[zi,tt]; rid=rayids[b]
        Wseed[b:b+n]=IP[b:b+n]+O[rid]
print(f"nodes with w assigned: {np.isfinite(Wseed).sum()}/{N}",flush=True)
eu=np.array(eu,np.int64); ev=np.array(ev,np.int64); es=np.array(es,np.float32)*chir
val=np.isfinite(Wseed[eu])&np.isfinite(Wseed[ev])
kill=np.abs(Wseed[ev]-(Wseed[eu]+es))>0.5
kr=float((kill&val).sum())/max(int(val.sum()),1)
print(f"GATE3 kill-rate: {kr*100:.1f}% (over {int(val.sum())} valid edges)",flush=True)
keep=val&~kill
eu2,ev2,es2=eu[keep],ev[keep],es[keep]
W=Wseed.copy()
for sweep in range(4):
    acc=np.zeros(N,np.float64); cnt=np.zeros(N,np.float64)
    np.add.at(acc,ev2,W[eu2]+es2); np.add.at(cnt,ev2,1)
    np.add.at(acc,eu2,W[ev2]-es2); np.add.at(cnt,eu2,1)
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
np.savez_compressed("./wfield_final_v105.npz",w=A,r=Rc,zs=ZS)
zi=nz//2; wz=A[zi][np.isfinite(A[zi])]
print(f"w z-mid: unique {len(np.unique(np.round(wz)))} range {wz.min():.0f}..{wz.max():.0f}" if len(wz) else "w z-mid: EMPTY",flush=True)
print("WFIELD14_DONE",flush=True)
