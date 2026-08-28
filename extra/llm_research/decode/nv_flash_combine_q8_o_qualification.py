#!/usr/bin/env python3
"""Full-logit qualification for Flash-combine-owned Q8 O inputs."""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL, LOCK, PYTHON
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison


def child(arm:str, depth:int, count:int, max_context:int, blocks:int, out:pathlib.Path):
  os.environ.update(DEV="NV",PROFILE="0")
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  model=_load(MODEL,max_context)
  geometry={"split_count":6,"llama_vec_wide":True,"token_bound":768,"combine_register_weights":True}
  model._flash_decode_tile_geometry_lease=geometry
  for index,block in enumerate(model.blk):
    block_geometry=dict(geometry)
    if arm == "candidate" and index < blocks: block_geometry["o_q8_owned"]=True
    block._flash_decode_tile_geometry_lease=block_geometry
  model._decode_direct_greedy_promoted=True; model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0,diagnostic_full_logits=True)
  tokens=[]; logits=[]
  try:
    for _ in range(count):
      token,full=next(gen)
      if full is None: continue
      arr=full.numpy().reshape(-1)
      if int(token) != int(arr.argmax()): raise RuntimeError("sample/logit binding mismatch")
      tokens.append(int(token)); logits.append(arr)
  finally: gen.close()
  values=np.stack(logits)
  row={"schema":"tinygrad.nv_flash_combine_q8_o_qualification.v1","arm":arm,"depth":depth,
       "count":len(tokens),"blocks":blocks if arm == "candidate" else 0,"tokens":tokens,
       "finite":bool(np.isfinite(values).all()),"shape":list(values.shape)}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
  np.savez_compressed(out.with_suffix(".npz"),logits=values)
  return row


def driver(args):
  root=args.out.with_suffix(""); root.mkdir(parents=True,exist_ok=True)
  rows={}; arrays={}
  for arm in ("control","candidate"):
    out=root/f"{arm}.json"
    cmd=["timeout",str(args.timeout),"flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
         "--arm",arm,"--depth",str(args.depth),"--count",str(args.count),"--max-context",str(args.max_context),
         "--blocks",str(args.blocks),"--out",str(out)]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,"PYTHONPATH":str(ROOT)})
    if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-6000:]}")
    rows[arm]=json.loads(out.read_text()); arrays[arm]=np.load(out.with_suffix(".npz"))["logits"]
  comparison=_semantic_comparison(arrays["control"],arrays["candidate"],rows["control"],rows["candidate"])
  result={"schema":"tinygrad.nv_flash_combine_q8_o_qualification.v1","blocks":args.blocks,
          "rows":rows,"comparison":comparison,"verdict":"PASS" if comparison["semantic_pass"] else "FAIL"}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main():
  ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--arm",choices=("control","candidate"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=12)
  ap.add_argument("--max-context",type=int,default=768); ap.add_argument("--blocks",type=int,default=8)
  ap.add_argument("--timeout",type=int,default=1800); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args(); result=child(args.arm,args.depth,args.count,args.max_context,args.blocks,args.out) if args.arm else driver(args)
  print(json.dumps(result,indent=2,sort_keys=True))

if __name__ == "__main__": main()
