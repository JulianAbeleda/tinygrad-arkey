#!/usr/bin/env python3
"""Short-ctx failure matrix for the owned tile (docs/owned-amdgcn-tile-short-ctx-scope-20260623.md).
Standalone: detect over-split / empty-split NaN at short ctx (S=48) vs a ctx-aware S clamp, vs numpy ref."""
import json, pathlib, sys
import numpy as np
from tinygrad import Tensor, UOp, dtypes
from extra.qk_owned_flash_decode_graph_node import amdgcn_flash_decode, Hq, Hkv, Hd, G, SCALE
MAXC=4608
OUT=pathlib.Path(__file__).resolve().parents[1]/"bench/qk-owned-amdgcn-tile-short-ctx"
def empty_splits(n_valid,S):
    per=-(-n_valid//S); cnt=0
    for s in range(S):
        t0=s*per; t1=min(t0+per,n_valid)
        if t0>=n_valid or t1<=t0: cnt+=1
    return per,cnt
def ctx_S(n_valid,PER=43,SMIN=4): 
    S=max(SMIN,min(48,n_valid//PER))
    # shrink until no empty split
    while S>SMIN and empty_splits(n_valid,S)[1]>0: S-=1
    return S
def run(n_valid,S):
    rng=np.random.default_rng(0); sp=n_valid-1
    Q=(rng.standard_normal((Hq,Hd))*8).astype(np.float16)
    K=(rng.standard_normal((Hkv,MAXC,Hd))*200).astype(np.float16); V=(rng.standard_normal((Hkv,MAXC,Hd))*0.5).astype(np.float16)
    K[:,n_valid:]=0; V[:,n_valid:]=0
    vsp=UOp.variable("start_pos",0,MAXC-1)
    out=amdgcn_flash_decode(Tensor(Q),Tensor(K.reshape(-1)),Tensor(V.reshape(-1)),vsp,S,MAXC).reshape(Hq*Hd)
    carry=Tensor.ones(MAXC,dtype=dtypes.float32)[0:vsp.bind(sp)].sum().reshape(1)*0.0
    on=(out+carry).reshape(Hq,Hd).numpy()
    ref=np.zeros((Hq,Hd),np.float32)
    for h in range(Hq):
        kvh=h//G; sc=(Q[h:h+1].astype(np.float32)@K[kvh,:n_valid].astype(np.float32).T)[0]*SCALE
        p=np.exp(sc-sc.max()); p/=p.sum(); ref[h]=p@V[kvh,:n_valid].astype(np.float32)
    fin=bool(np.all(np.isfinite(on))); rmse=float(np.sqrt(((on-ref)**2).mean())/(np.sqrt((ref**2).mean())+1e-9)) if fin else -1.0
    per,emp=empty_splits(n_valid,S)
    return {"n_valid":n_valid,"S":S,"per":per,"empty_splits":emp,"finite":fin,"rel_rmse":rmse,"ok":fin and rmse<=1e-3}
res={"date":"2026-06-23","phase":"OWNED_TILE_SHORT_CTX_FAILURE_MATRIX","gpu":"gfx1100","rows":[]}
for nv in (513,1025,2049,4097):
    r48=run(nv,48); rsc=run(nv,ctx_S(nv))
    res["rows"].append({"ctx":nv-1,"S48":r48,"ctxS":rsc})
    print(f"ctx{nv-1}: S=48 empty={r48['empty_splits']} finite={r48['finite']} rmse={r48['rel_rmse']:.1e} ok={r48['ok']}  ||  ctxS={rsc['S']} empty={rsc['empty_splits']} finite={rsc['finite']} rmse={rsc['rel_rmse']:.1e} ok={rsc['ok']}", file=sys.stderr)
overspill=any(not r["S48"]["ok"] for r in res["rows"] if r["ctx"]<2048)
fixed=all(r["ctxS"]["ok"] for r in res["rows"])
res["verdict"]=("SHORT_CTX_EMPTY_SPLIT_NAN" if overspill else "SHORT_CTX_FAILURE_NOT_REPRODUCED")
res["ctx_scaled_S_fixes_all"]=fixed
OUT.mkdir(parents=True,exist_ok=True); (OUT/"failure_matrix.json").write_text(json.dumps(res,indent=2))
print("verdict:",res["verdict"],"| ctx_scaled_S_fixes_all:",fixed, file=sys.stderr)
