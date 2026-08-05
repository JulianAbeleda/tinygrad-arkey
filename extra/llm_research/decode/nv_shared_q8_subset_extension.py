#!/usr/bin/env python3
"""Resumable blocks-19..35 shared-Q8 precision-budget extension.

This is research-only orchestration. Every GPU child installs an explicit
default-off subset lease in a fresh process. The rank phase is CPU-only and
scores all 2^17 tail subsets from signed full-logit singleton deltas.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys
import numpy as np

from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison

BASE_INDICES=tuple(range(1,13))+(14,15,16,17,18)
TAIL_INDICES=tuple(range(19,36))
DEFAULT_PRIOR_ROOT=pathlib.Path("/tmp/nv-q4-subset-search")
QUALIFIER=pathlib.Path(__file__).with_name("nv_shared_q8_progressive_qualification.py")


def _atomic_json(path:pathlib.Path, payload:dict) -> None:
  tmp=path.with_name(path.name+".tmp")
  tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
  os.replace(tmp,path)


def _valid_child(path:pathlib.Path, indices:tuple[int,...], count:int) -> bool:
  npz=path.with_suffix(".npz")
  if not path.exists() or not npz.exists(): return False
  try:
    row=json.loads(path.read_text()); arr=np.load(npz)["logits"]
  except Exception: return False
  expected=2*len(indices)
  return (tuple(row.get("fused_indices",())) == indices and row.get("count") == count and
          row.get("fused_rmsnorm_q8_provider_count") == expected and
          row.get("q8_provider_count") == expected and row.get("legacy_q4_shared_consumer_count") == 0 and
          row.get("cooperative_q4_consumer_count",0) >= 2*expected and
          arr.shape == (count,1,151936) and np.isfinite(arr).all())


def singleton_sweep(args) -> dict:
  root=pathlib.Path(args.out); root.mkdir(parents=True,exist_ok=True)
  rows=[]
  for block in TAIL_INDICES:
    indices=BASE_INDICES+(block,); out=root/f"base17-b{block}.json"
    if _valid_child(out,indices,args.count):
      rows.append({"block":block,"status":"resumed","path":str(out)})
      continue
    cmd=["timeout",f"{args.timeout}s","flock","-w",str(args.lock_wait),args.lock,sys.executable,str(QUALIFIER),
         "--mode","child","--model",args.model,"--depth",str(args.depth),"--count",str(args.count),
         "--max-context",str(args.max_context),"--groups","0","--fused-groups","0",
         "--fused-indices",",".join(map(str,indices)),"--cooperative-q4","--composed","--out",str(out)]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out.with_suffix(".stdout").write_text(run.stdout); out.with_suffix(".stderr").write_text(run.stderr)
    status="complete" if run.returncode == 0 and _valid_child(out,indices,args.count) else "failed"
    rows.append({"block":block,"status":status,"returncode":run.returncode,"path":str(out)})
    _atomic_json(root/"progress.json",{"schema":"tinygrad.nv_shared_q8_subset_extension.progress.v1","rows":rows})
    if status != "complete": break
  result={"schema":"tinygrad.nv_shared_q8_subset_extension.progress.v1","base_indices":list(BASE_INDICES),
          "tail_indices":list(TAIL_INDICES),"rows":rows,"complete":len(rows)==len(TAIL_INDICES) and
          all(row["status"] in ("complete","resumed") for row in rows)}
  _atomic_json(root/"progress.json",result); return result


def additive_subset_scores(base_delta:np.ndarray, singleton_deltas:np.ndarray, denominator:float) -> tuple[np.ndarray,np.ndarray]:
  """Return cardinality and relative-L2 prediction for every tail bitmask."""
  if singleton_deltas.ndim != 2 or base_delta.ndim != 1 or singleton_deltas.shape[1] != base_delta.size:
    raise ValueError("delta matrix shape mismatch")
  vecs=singleton_deltas.astype(np.float64,copy=False); base=base_delta.astype(np.float64,copy=False)
  gram=vecs@vecs.T; cross=vecs@base; base2=float(base@base)
  masks=np.arange(1<<len(vecs),dtype=np.uint32)
  bits=((masks[:,None]>>np.arange(len(vecs),dtype=np.uint32))&1).astype(np.float64)
  norm2=base2+2*(bits@cross)+np.sum((bits@gram)*bits,axis=1)
  return bits.sum(axis=1).astype(np.int16),np.sqrt(np.maximum(norm2,0))/max(float(denominator),1e-30)


def _load_authority(prior:pathlib.Path) -> tuple[dict,np.ndarray,dict,np.ndarray]:
  g0_path=prior/"g0.json"; base_path=prior/"quint_add18.json"
  for path in (g0_path,base_path,g0_path.with_suffix(".npz"),base_path.with_suffix(".npz")):
    if not path.exists(): raise FileNotFoundError(path)
  g0_row,base_row=json.loads(g0_path.read_text()),json.loads(base_path.read_text())
  if tuple(base_row.get("fused_indices",())) != BASE_INDICES: raise ValueError("prior base is not accepted 17-block subset")
  return g0_row,np.load(g0_path.with_suffix(".npz"))["logits"],base_row,np.load(base_path.with_suffix(".npz"))["logits"]


def rank(args) -> dict:
  root,prior=pathlib.Path(args.root),pathlib.Path(args.prior_root)
  g0_row,g0,base_row,base=_load_authority(prior)
  singleton_rows,singletons=[],[]
  for block in TAIL_INDICES:
    path=root/f"base17-b{block}.json"
    if not _valid_child(path,BASE_INDICES+(block,),args.count): raise RuntimeError(f"missing/invalid singleton {block}: {path}")
    row=json.loads(path.read_text()); arr=np.load(path.with_suffix(".npz"))["logits"]
    absolute=_semantic_comparison(g0,arr,g0_row,row)
    inc=(arr-base).reshape(-1).astype(np.float32)
    singleton_rows.append({"block":block,"absolute":absolute,
      "incremental_relative_l2":float(np.linalg.norm(inc.astype(np.float64))/max(np.linalg.norm(base.astype(np.float64)),1e-30))})
    singletons.append(inc)
  base_delta=(base-g0).reshape(-1).astype(np.float32)
  singleton_matrix=np.stack(singletons)
  denom=float(np.linalg.norm(g0.astype(np.float64).ravel()))
  cards,scores=additive_subset_scores(base_delta,singleton_matrix,denom)
  masks=np.arange(1<<len(TAIL_INDICES),dtype=np.uint32)
  # Cardinality is the primary objective because each admitted block owns one
  # provider family; relative L2 ranks equal-cardinality candidates.
  order=np.lexsort((scores,-cards))
  predicted=[]
  for mask in order[:args.top]:
    chosen=[TAIL_INDICES[i] for i in range(len(TAIL_INDICES)) if int(mask)&(1<<i)]
    predicted.append({"mask":int(mask),"tail_indices":chosen,"total_indices":list(BASE_INDICES)+chosen,
                      "cardinality":len(BASE_INDICES)+len(chosen),"predicted_relative_l2":float(scores[mask]),
                      "predicted_l2_pass":bool(scores[mask] <= 1e-3)})
  # Reconstruct only the best few candidates for predicted argmax/top-10 gates.
  semantic_predictions=[]
  promising=[m for m in order if scores[m] <= args.rank_ceiling][:args.semantic_top]
  for mask in promising:
    selected=[i for i in range(len(TAIL_INDICES)) if int(mask)&(1<<i)]
    pred=base.copy()
    for i in selected: pred += singleton_matrix[i].reshape(base.shape)
    fake=dict(base_row); fake["tokens"]=base_row["tokens"]
    sem=_semantic_comparison(g0,pred,g0_row,fake)
    semantic_predictions.append({"mask":int(mask),"tail_indices":[TAIL_INDICES[i] for i in selected],
                                 "total_indices":list(BASE_INDICES)+[TAIL_INDICES[i] for i in selected],**sem})
  result={"schema":"tinygrad.nv_shared_q8_subset_extension.rank.v1","base_indices":list(BASE_INDICES),
    "tail_indices":list(TAIL_INDICES),"singleton_rows":singleton_rows,"search":{"subsets":1<<len(TAIL_INDICES),
    "method":"signed-full-logit additive Gram search","authority_relative_l2_max":1e-3},
    "ranked":predicted,"semantic_predictions":semantic_predictions}
  _atomic_json(pathlib.Path(args.out),result); return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("singleton-sweep","rank"),required=True)
  ap.add_argument("--out",required=True); ap.add_argument("--root",default="/tmp/nv-q4-subset-extension")
  ap.add_argument("--prior-root",default=str(DEFAULT_PRIOR_ROOT)); ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=8); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--timeout",type=int,default=600); ap.add_argument("--lock-wait",type=int,default=300); ap.add_argument("--lock",default="/tmp/gpu-bench.lock")
  ap.add_argument("--top",type=int,default=64); ap.add_argument("--semantic-top",type=int,default=12); ap.add_argument("--rank-ceiling",type=float,default=1.1e-3)
  args=ap.parse_args()
  result=singleton_sweep(args) if args.mode == "singleton-sweep" else rank(args)
  print(json.dumps(result,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
