"""PROVENANCE (added 2026-07-08): this is the script behind "Method 3:
Triangle-confirmed spanning tree" in docs/NEGATIVE_RESULT.md's 5-method table
(69.2% disagreement / 31.4% ray-level conflict, 67% mesh-audit jump rate) --
originally `data/wfield/wfield_trackB.py` (internal version tag v11,
"Track B") in the parent project repo. Logic, structure, and every
constant/number are unchanged from that original research script;
comments, docstrings, and log/print strings have been translated from
Spanish to English for publication (the original Spanish-language version
remains in the parent project repo at the path above if byte-for-byte
provenance is needed). Hardcoded local paths reflect the original
research-machine layout; adjust before rerunning. See
method1_confidence_mst.py's header for the fuller provenance note
(applies identically here).

Original docstring (translated):

v11 (Track B - new signal): instead of indexing differently, adds an
INDEPENDENT consistency signal never tried before: TRIANGLE verification.
For every 1-hop edge (tt->tt+1), also compute the DIRECT 2-hop jump
(tt->tt+2) and check whether m(tt->tt+1)+m(tt+1->tt+2) == m(tt->tt+2)
(same in z). This is INDEPENDENT evidence of whether a single ray-pair's
majority vote is reliable -- it isn't another indexing scheme, it's a
genuine redundancy check (like loop closure in SLAM). If the
triangle-confirmation rate is also low (~50%), it confirms the ceiling is
corridor-detection SNR, not the consensus method. If it's high, it gives a
new, better filter to feed the same max-confidence spanning tree (MST)
that v10.4 already validated as the correct architecture."""
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
def mode_delta(zi,tt,zi2,tt2):
    a,ia,ba=ray(zi,tt); b2,ib,bb=ray(zi2,tt2)
    ms=match(a,b2)
    if len(ms)<5: return None,0.0,ba,bb,ms
    vs=[round(float(ib[j]-ia[i])) for i,j in ms]
    c=Counter(vs); m,nm=c.most_common(1)[0]
    return m,nm/len(vs),ba,bb,ms

# --- THETA: 1-hop and direct 2-hop, triangle check ---
M1={}   # (zi,tt) -> (m, purity, ba, bb)
M2={}   # (zi,tt) -> direct m for tt->tt+2
eu=[];ev=[];es=[]
for zi in range(nz):
    for tt in range(NTH):
        t2=(tt+1)%NTH
        m,pu,ba,bb,ms=mode_delta(zi,tt,zi,t2)
        if m is not None: M1[(zi,tt)]=(m,pu,ba,bb)
        if ms:
            s=1.0 if tt==NTH-1 else 0.0
            for i,j in ms: eu.append(ba+i); ev.append(bb+j); es.append(s)
    if zi%300==0: print("th1",zi,flush=True)
for zi in range(nz):
    for tt in range(NTH):
        t3=(tt+2)%NTH
        m,pu,ba,bb,_=mode_delta(zi,tt,zi,t3)
        if m is not None: M2[(zi,tt)]=m
    if zi%300==0: print("th2",zi,flush=True)
conf_th=0; tot_th=0
NB={}
for zi in range(nz):
    for tt in range(NTH):
        t2=(tt+1)%NTH
        if (zi,tt) not in M1 or (zi,t2) not in M1: continue
        m1,pu1,ba1,bb1=M1[(zi,tt)]; m2,pu2,ba2,bb2=M1[(zi,t2)]
        direct=M2.get((zi,tt))
        tot_th+=1
        ok = (direct is not None) and (m1+m2==direct)
        if ok:
            conf_th+=1
            s=1.0 if tt==NTH-1 else 0.0
            NB.setdefault(ba1,[]).append((bb1,m1,s,pu1))
            NB.setdefault(bb1,[]).append((ba1,-m1,-s,pu1))
print(f"TRIANGLE theta: confirmed {conf_th}/{tot_th} ({100*conf_th/max(tot_th,1):.1f}%)",flush=True)

# --- Z: 1-hop and direct 2-hop, triangle check ---
M1z={}; M2z={}
for zi in range(nz-1):
    for tt in range(NTH):
        m,pu,ba,bb,ms=mode_delta(zi,tt,zi+1,tt)
        if m is not None: M1z[(zi,tt)]=(m,pu,ba,bb)
        if ms:
            for i,j in ms: eu.append(ba+i); ev.append(bb+j); es.append(0.0)
    if zi%200==0: print("z1",zi,flush=True)
for zi in range(nz-2):
    for tt in range(NTH):
        m,pu,ba,bb,_=mode_delta(zi,tt,zi+2,tt)
        if m is not None: M2z[(zi,tt)]=m
    if zi%200==0: print("z2",zi,flush=True)
conf_z=0; tot_z=0
for zi in range(nz-2):
    for tt in range(NTH):
        if (zi,tt) not in M1z or (zi+1,tt) not in M1z: continue
        m1,pu1,ba1,bb1=M1z[(zi,tt)]; m2,pu2,ba2,bb2=M1z[(zi+1,tt)]
        direct=M2z.get((zi,tt))
        tot_z+=1
        ok=(direct is not None) and (m1+m2==direct)
        if ok:
            conf_z+=1
            NB.setdefault(ba1,[]).append((bb1,m1,0.0,pu1))
            NB.setdefault(bb1,[]).append((ba1,-m1,0.0,pu1))
print(f"TRIANGLE z: confirmed {conf_z}/{tot_z} ({100*conf_z/max(tot_z,1):.1f}%)",flush=True)
print(f"TRIANGLE total: confirmed {conf_th+conf_z}/{tot_th+tot_z} ({100*(conf_th+conf_z)/max(tot_th+tot_z,1):.1f}%)",flush=True)
print(f"rays with triangle-confirmed edges: {len(NB)}",flush=True)
print(f"corridor-level edges (eu/ev, unfiltered, reused from 1-hop): {len(eu)}",flush=True)

def solve_mst(chir):
    seed=base[nz//2,0]
    off={seed:0.0}
    h=[]
    for (v,m,s,pu) in NB.get(seed,[]): heapq.heappush(h,(-pu,seed,v,m,s))
    treeedges=set()
    while h:
        negpu,u,v,m,s=heapq.heappop(h)
        if v in off: continue
        off[v]=off[u]+m+chir*s
        treeedges.add((u,v)); treeedges.add((v,u))
        for (w,m2,s2,pu2) in NB.get(v,[]):
            if w not in off: heapq.heappush(h,(-pu2,v,w,m2,s2))
    conf=tot=0
    for u,edges in NB.items():
        if u not in off: continue
        for (v,m,s,pu) in edges:
            if v not in off or (u,v) in treeedges: continue
            tot+=1
            cand=off[u]+m+chir*s
            if abs(off[v]-cand)>0.5: conf+=1
    return off,conf,tot

res={}
for chir in (1,-1):
    off,conf,tot=solve_mst(chir)
    res[chir]=(off,conf,tot)
    print(f"GATE2 chir {chir:+d}: conflicts {conf}/{tot} ({conf/max(tot,1)*100:.2f}%) rays_reached {len(off)}/{int((lens>0).sum())}",flush=True)
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
np.savez_compressed("./wfield_trackB.npz",w=A,r=Rc,zs=ZS)
zi=nz//2; wz=A[zi][np.isfinite(A[zi])]
print(f"w z-mid: unique {len(np.unique(np.round(wz)))} range {wz.min():.0f}..{wz.max():.0f}" if len(wz) else "w z-mid: EMPTY",flush=True)
print("WFIELD_TRACKB_DONE",flush=True)
