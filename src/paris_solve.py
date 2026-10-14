#!/usr/bin/env python3
"""Paris4 absolute validation: per-slab phase solve, score vs human pairs.

Slabs group nearby collections. Working grid 19.2um (two L2 voxels).
Umbilicus-radial orientation. Step calibrated from ridge counts per slab.
"""

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import VOX_MM, arclength, data_dir, decode_lasagna_normals
from patch_check import sample_field, sha256

DEV = os.environ.get("WREV_DEVICE", "cuda:0")
L2B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4"
CTB = L2B + "/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
MB = L2B + "/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr"
LB = (
    L2B
    + "/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c"
)
MARGIN = 100
CAP = 40_000_000


def working_shape(lo_xyz, hi_xyz):
    """Two L2 voxels per grid sample; array dimensions are ZYX, not XYZ."""
    return tuple(int(v) for v in ((np.asarray(hi_xyz) - lo_xyz) // 2)[::-1])


def run_config():
    orient = os.environ.get("ORIENT", "umbi")
    if orient != "umbi" or os.environ.get("RGT"):
        raise ValueError("Paris4 implements only ORIENT=umbi; RGT/v6 is not connected")
    config = dict(
        orient=orient,
        lam=float(os.environ.get("LAM", "3")),
        afv=os.environ["AFV"] if "AFV" in os.environ else data_dir() + "/Paris4_v1.afv",
        fiber_limit=int(os.environ.get("FIBER_LIMIT", "400")),
        pair_limit=int(os.environ.get("PAIR_LIMIT", "2000")),
        device=DEV,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        normal_encoding="hemisphere_nx_ny_nz_positive",
    )
    if not np.isfinite(config["lam"]) or config["lam"] < 0 or config["fiber_limit"] < 1 or config["pair_limit"] < 1:
        raise ValueError("invalid fibre weight or limits")
    return config


def load_pairs(zlo=10000, zhi=11000):
    with open(data_dir() + "/relwind.json") as stream:
        d = json.load(stream)
    pairs, colls = [], {}
    for k, v in d["collections"].items():
        pts = [(pid, np.array(p["p"], float), p.get("wind_a", 0)) for pid, p in v["points"].items()]
        if len(pts) < 2:
            continue
        P = np.array([p for _, p, _ in pts])
        if not ((P[:, 2] >= zlo - 200).any() and (P[:, 2] < zhi + 200).any()):
            continue
        colls[k] = P
        for (i, a, wa), (j, b, wb) in itertools.combinations(pts, 2):
            dw = abs(wa - wb)
            if 1 <= dw <= 6 and zlo <= a[2] < zhi and zlo <= b[2] < zhi:
                pairs.append((a, b, dw, k))
    return pairs, colls


def group_slabs(colls):
    items = sorted(colls.items(), key=lambda kv: kv[1][:, 2].mean())
    slabs, cur, curbox = [], [], None
    for k, P in items:
        lo, hi = P.min(0) - MARGIN, P.max(0) + MARGIN
        if curbox is None:
            cur, curbox = [(k, P)], [lo, hi]
            continue
        nlo = np.minimum(curbox[0], lo)
        nhi = np.maximum(curbox[1], hi)
        if np.prod((nhi - nlo) // 2) > CAP:
            slabs.append((cur, curbox))
            cur, curbox = [(k, P)], [lo, hi]
        else:
            cur.append((k, P))
            curbox = [nlo, nhi]
    if cur:
        slabs.append((cur, curbox))
    return slabs


def main():
    import zarr

    config = run_config()
    print("effective config: " + json.dumps(config), flush=True)
    t0 = time.time()
    predict_only = os.environ.get("PREDICT_ONLY") == "1"
    if predict_only and (not os.environ.get("SLAB") or os.environ.get("AUDIT_DIR")):
        raise ValueError("prediction-only requires explicit SLAB and no label audit")
    pairs, colls = ([], {}) if predict_only else load_pairs()
    config["prediction_only"] = predict_only
    audit = os.environ.get("AUDIT_DIR")
    audit_rows = []
    if audit:
        from paris_pair_check import load_pair_records, score_pairs, summarize_rows

        audit = Path(audit)
        audit.mkdir(exist_ok=False)
        pair_records, _, selection_exclusions = load_pair_records(Path(data_dir()) / "relwind.json")
        if len(pair_records) != len(pairs):
            raise ValueError("legacy/new label eligibility differs; investigate before scoring")
        (audit / "selection.json").write_text(json.dumps(dict(records=pair_records, exclusions=selection_exclusions)))
    print(f"pairs: {len(pairs)} collections: {len(colls)}", flush=True)
    _sl = os.environ.get("SLAB", "")
    if _sl:
        _v = [int(v) for v in _sl.split(",")]
        slabs = [
            (list(colls.items()), (np.array([_v[0], _v[2], _v[4]], float), np.array([_v[1], _v[3], _v[5]], float)))
        ]
        print(f"manual slab {_sl}", flush=True)
    else:
        slabs = group_slabs(colls)
        print(f"slabs: {len(slabs)}", flush=True)
    _out = os.environ.get("OUT", "")
    if _out and (len(slabs) != 1 or Path(_out).exists() or Path(_out + ".json").exists()):
        raise ValueError("OUT requires a single slab and new field/metadata paths")
    with open(data_dir() + "/umbi.json") as stream:
        umbi = json.load(stream)
    cp = np.array([[p["x"], p["y"], p["z"]] for p in umbi["control_points"]])
    o = np.argsort(cp[:, 2])
    cp = cp[o]

    def umbi_xy(z):
        i = np.clip(np.searchsorted(cp[:, 2], z) - 1, 0, len(cp) - 2)
        t = (z - cp[i, 2]) / max(cp[i + 1, 2] - cp[i, 2], 1e-6)
        return (1 - t) * cp[i, :2] + t * cp[i + 1, :2]

    m7 = zarr.open(MB + "/0", mode="r")
    laz = lambda ch, lv: zarr.open(f"{LB}_{ch}.ome.zarr/{lv}", mode="r")
    import sqlite3
    import struct

    FDB = sqlite3.connect(Path(config["afv"]).resolve().as_uri() + "?mode=ro", uri=True)
    LAM = config["lam"]
    config["afv_sha256"] = sha256(config["afv"])
    res = []
    for si, (members, (lo, hi)) in enumerate(slabs):
        if audit:
            _out = str(audit / f"slab{si}.npy")
        lo = np.maximum(np.floor(lo).astype(int), 0)
        hi = lo + np.maximum(np.ceil(hi - lo).astype(int), 8)
        det = np.asarray(m7[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]) > 128
        nz, ny, nx = working_shape(lo, hi)

        # lasagna at working grid
        def grab(ch, lv, sc):
            a = np.asarray(
                laz(ch, lv)[
                    int(lo[2] * sc) : int(hi[2] * sc),
                    int(lo[1] * sc) : int(hi[1] * sc),
                    int(lo[0] * sc) : int(hi[0] * sc),
                ]
            ).astype(np.float32)
            t = torch.from_numpy(a[None, None])
            return F.interpolate(t, size=(nz, ny, nx), mode="trilinear").numpy()[0, 0]

        nxx = grab("nx", "4", 0.25)
        nyy = grab("ny", "4", 0.25)
        gm = grab("grad_mag", "4", 0.25)
        n = decode_lasagna_normals(nxx, nyy)
        n = torch.from_numpy(n).to(DEV)
        ridge = torch.from_numpy(det).to(DEV)
        ridge = F.interpolate(ridge[None, None].float(), size=(nz, ny, nx), mode="nearest")[0, 0] > 0.3
        # umbilicus-radial orientation
        zz = (np.arange(nz) + 0.5) * 2 + lo[2]
        cx = np.array([umbi_xy(z)[0] for z in zz])
        cy = np.array([umbi_xy(z)[1] for z in zz])
        ys, xs = np.mgrid[0:ny, 0:nx]
        ox = xs[None] * 2 + lo[0] - cx[:, None, None]
        oy = ys[None] * 2 + lo[1] - cy[:, None, None]
        on = np.sqrt(ox * ox + oy * oy) + 1e-6
        ox, oy = torch.from_numpy(ox / on).to(DEV), torch.from_numpy(oy / on).to(DEV)
        s = torch.sign(n[..., 0] * ox + n[..., 1] * oy)
        s = torch.where(s == 0, torch.ones_like(s), s)
        n = n * s[..., None]
        dens = torch.from_numpy(gm / 1000.0 * 2).to(DEV)  # sheets per working vox (19.2um)
        tgt = (n * dens[..., None]).float()
        # --- fibres in slab (L0 2.4um coords = L2 x4) ---
        q = """SELECT DISTINCT b.fiber_id FROM block_bounds r JOIN blocks b ON b.id=r.id
               WHERE r.max_x>=:x0 AND r.min_x<=:x1 AND r.max_y>=:y0 AND r.min_y<=:y1
                 AND r.max_z>=:z0 AND r.min_z<=:z1"""
        bb4 = dict(
            x0=int(lo[0] * 4),
            x1=int(hi[0] * 4),
            y0=int(lo[1] * 4),
            y1=int(hi[1] * 4),
            z0=int(lo[2] * 4),
            z1=int(hi[2] * 4),
        )
        fids = [r[0] for r in FDB.execute(q, bb4)][: config["fiber_limit"]]
        frng = np.random.default_rng(si)

        def ti(pt):
            z = min(max(int(round((pt[2] - lo[2]) / 2)), 0), nz - 1)
            y = min(max(int(round((pt[1] - lo[1]) / 2)), 0), ny - 1)
            x = min(max(int(round((pt[0] - lo[0]) / 2)), 0), nx - 1)
            return (z * ny + y) * nx + x

        pairs_f = []
        for fid in fids:
            rows = FDB.execute("SELECT points FROM blocks WHERE fiber_id=? ORDER BY first_segment", (fid,))
            try:
                P = np.concatenate(
                    [
                        np.array(struct.unpack(f"<{3 * (len(b) // 24)}d", b), dtype=np.float64).reshape(-1, 3)
                        for (b,) in rows
                    ]
                )
            except Exception:
                continue
            keep = (
                (P[:, 0] >= bb4["x0"])
                & (P[:, 0] < bb4["x1"])
                & (P[:, 1] >= bb4["y0"])
                & (P[:, 1] < bb4["y1"])
                & (P[:, 2] >= bb4["z0"])
                & (P[:, 2] < bb4["z1"])
            )
            P = P[keep] / 4.0  # -> L2 frame
            if len(P) < 3:
                continue
            s = arclength(P, VOX_MM["PHercParis4"] * 4)
            for _ in range(8):
                i, j = frng.integers(0, len(P), 2)
                if abs(s[j] - s[i]) >= 0.3 and len(pairs_f) < config["pair_limit"]:
                    a, b2 = ti(P[i]), ti(P[j])
                    if a != b2:
                        pairs_f.append((a, b2))
        FP = torch.tensor([a for a, _ in pairs_f], device=DEV)
        FQ = torch.tensor([b for _, b in pairs_f], device=DEV)
        print(f"slab{si}: fibres {len(fids)} constraint pairs {len(pairs_f)}", flush=True)
        # --- orientation validation: umbilicus-radial vs peel gradient ---
        from scipy import ndimage as _ndi

        _st = np.ones((1, 3, 3), bool)
        _rem = det.copy()
        _cnt = np.zeros(det.shape, np.float32)
        _r = 0
        while _rem.any() and _r < 120:
            _r += 1
            _er = _ndi.binary_erosion(_rem, structure=_st)
            _cnt[_rem & ~_er] = _r
            _rem = _er
        _pg = np.stack(np.gradient(_cnt.astype(np.float32)), -1)  # zyx comps, L2 grid
        _pg = torch.from_numpy(_pg).to(DEV)
        _pg = F.interpolate(_pg.permute(3, 0, 1, 2)[None], size=(nz, ny, nx), mode="trilinear")[0].permute(1, 2, 3, 0)
        _pgn = (_pg / (_pg.norm(dim=-1, keepdim=True) + 1e-9))[..., [2, 1, 0]]
        _in = torch.zeros_like(ridge)
        _in[12:-12, 12:-12, 12:-12] = True
        _m = ridge & _in & (_pg.norm(dim=-1) > 1e-6)
        _agree = ((n * _pgn).sum(-1)[_m]).sign()
        _agr = float((_agree > 0).float().mean()) if _m.any() else float("nan")
        print(f"slab{si}: orient agreement umbi-vs-peelgrad {_agr:.3f}", flush=True)
        N = nz * ny * nx
        W = (ridge.float() * 0.9 + 0.1).ravel()

        def shift(x, ax, d):
            return torch.roll(x.reshape(nz, ny, nx), d, ax).ravel()

        def noborder(t, ax, last):
            v = t.reshape(nz, ny, nx)
            idx = [slice(None)] * 3
            idx[ax] = (nz, ny, nx)[ax] - 1 if last else 0
            v[tuple(idx)] = 0
            return v.ravel()

        def matvec(p):
            out = torch.zeros_like(p)
            for ax in range(3):
                wf = noborder(0.5 * (W + shift(W, ax, -1)), ax, True)
                wb = noborder(0.5 * (W + shift(W, ax, 1)), ax, False)
                out += wf * (p - shift(p, ax, -1)) + wb * (p - shift(p, ax, 1))
            if len(pairs_f):
                out = out + LAM * torch.bincount(
                    torch.cat([FP, FQ]), torch.cat([p[FP] - p[FQ], p[FQ] - p[FP]]), minlength=N
                )
            return out

        b = torch.zeros(N, device=DEV)
        for ax in range(3):
            ga = tgt.reshape(-1, 3)[:, 2 - ax]
            gf = noborder(0.5 * (ga + shift(ga, ax, -1)), ax, True)
            wf = noborder(0.5 * (W + shift(W, ax, -1)), ax, True)
            gb = noborder(0.5 * (ga + shift(ga, ax, 1)), ax, False)
            wb = noborder(0.5 * (W + shift(W, ax, 1)), ax, False)
            b += wf * gf - wb * gb
        diag = torch.zeros(N, device=DEV)
        for ax in range(3):
            diag += noborder(0.5 * (W + shift(W, ax, -1)), ax, True) + noborder(0.5 * (W + shift(W, ax, 1)), ax, False)
        if len(pairs_f):
            diag += LAM * torch.bincount(torch.cat([FP, FQ]), torch.ones(2 * len(pairs_f), device=DEV), minlength=N)
        Minv = 1.0 / diag.clamp_min(1e-6)
        x = torch.zeros(N, device=DEV)
        r = b - matvec(x)
        z = Minv * r
        p = z.clone()
        rz = (r * z).sum()
        for it in range(1, 3001):
            Ap = matvec(p)
            al = rz / (p * Ap).sum()
            x += al * p
            r -= al * Ap
            if r.norm() < 1e-3 * b.norm():
                break
            z = Minv * r
            rz2 = (r * z).sum()
            p = z + (rz2 / rz) * p
            rz = rz2
        phi = (x.reshape(nz, ny, nx) - x.median()).cpu().numpy()
        # calibrate step: ridge runs along stacking axis (model-free count)
        _ax = int(
            np.argmax(
                np.linalg.eigh(np.cov(np.stack(np.gradient(det.astype(np.float32)), -1).reshape(-1, 3).T))[1][:, -1]
                ** 2
            )
        )
        _sy, _sx = det.shape[1], det.shape[2]
        _sz = det.shape[0]
        if _ax == 0:
            _lines = [det[:, _sy // 2 + (k % 5) - 2, _sx // 2 + (k // 5) - 2] for k in range(25)]
        elif _ax == 1:
            _lines = [det[_sz // 2 + (k % 5) - 2, :, _sx // 2 + (k // 5) - 2] for k in range(25)]
        else:
            _lines = [det[_sz // 2 + (k % 5) - 2, _sy // 2 + (k // 5) - 2, :] for k in range(25)]
        _cnt = []
        for _ln in _lines:
            _d = np.diff(_ln.astype(np.int8))
            _cnt.append(int((_d == 1).sum()) + (1 if _ln[0] else 0))
        _cnt = np.array(_cnt)
        _nsh = float(np.median(_cnt[_cnt > 0])) if (_cnt > 0).any() else 0.0
        _vv = phi[ridge.cpu().numpy()]
        _rng = float(np.percentile(_vv, 99) - np.percentile(_vv, 1)) if _vv.size else 0.0
        step = _rng / max(_nsh, 1.0) if _nsh >= 2 else 1.0
        print(f"slab{si}: axis {_ax} sheets {_nsh:.0f} phirange {_rng:.2f} step {step:.3f}", flush=True)
        if _out:
            np.save(_out, phi.astype(np.float32))
            metadata = dict(
                frame="PHercParis4/20260411134726:L2",
                origin_xyz=lo.tolist(),
                spacing_xyz=[2, 2, 2],
                shape_zyx=list(phi.shape),
                step=step,
                config=config,
                fibres=len(fids),
                constraint_pairs=len(pairs_f),
                ridge_count=_nsh,
                ridge_axis=_ax,
                field_range_98pct=_rng,
                iterations=it,
                relative_residual=float(r.norm() / b.norm()),
                source_sha256=sha256(__file__),
                field_sha256=sha256(_out),
                normal_encoding="hemisphere_nx_ny_nz_positive",
                common_sha256=sha256(Path(__file__).with_name("common.py")),
                ridge_line_counts=_cnt.tolist(),
                collection_ids=[str(k) for k, _ in members],
                human_patches_used_as_constraints=False,
            )
            Path(_out + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"saved {_out}", flush=True)
        if audit:
            records = [r for r in pair_records if r["collection_id"] in metadata["collection_ids"]]
            rows, summary = score_pairs(phi, metadata, records)
            for row in rows:
                row["slab_id"] = si
                row["field_sha256"] = metadata["field_sha256"]
            audit_rows.extend(rows)
            (audit / f"slab{si}_pairs.json").write_text(json.dumps(dict(rows=rows, summary=summary), indent=2))
            np.savez(audit / f"slab{si}_constraints.npz", pairs=np.asarray(pairs_f, dtype=np.int64), fiber_ids=fids)
            (audit / "progress.json").write_text(
                json.dumps(dict(completed=si + 1, total=len(slabs), summary=summarize_rows(audit_rows)))
            )
        ok = tot = 0
        excluded = 0
        per = {}
        keys = set(m[0] for m in members)
        for a, b2, dw, k in pairs:
            if k not in keys:
                continue
            values = sample_field(phi, np.array([a, b2]), lo, [2, 2, 2])
            if not np.isfinite(values).all():
                excluded += 1
                continue
            pred = int(round(abs(values[0] - values[1]) / step))
            tot += 1
            ok += pred == dw
            per.setdefault(dw, [0, 0])
            per[dw][1] += 1
            per[dw][0] += pred == dw
        print(
            f"slab{si}: members {len(members)} grid {(nz, ny, nx)} step {step:.3f} "
            f"exact {ok}/{tot} excluded {excluded} "
            + " ".join(f"dw{dw}={a}/{b}" for dw, (a, b) in sorted(per.items())),
            flush=True,
        )
        res.append((ok, tot))
    ok_total, tot_total = sum(a for a, _ in res), sum(b for _, b in res)
    score = f"{ok_total / tot_total:.3f}" if tot_total else "n/a"
    print(f"TOTAL exact {ok_total}/{tot_total} = {score} ({time.time() - t0:.0f}s)", flush=True)
    if audit:
        (audit / "pairs.json").write_text(
            json.dumps(dict(rows=audit_rows, summary=summarize_rows(audit_rows)), indent=2)
        )
        (audit / "DONE").write_text("Fields, constraints and identifiable pair receipts saved.\n")


if __name__ == "__main__":
    main()
