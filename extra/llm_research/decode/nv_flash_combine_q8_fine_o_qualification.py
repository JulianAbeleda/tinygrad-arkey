#!/usr/bin/env python3
"""Recurrent full-logit quality gate for scale-per-16 Flash->Q4 O activations.

This is quality-only research.  It preserves Flash combine's fp16 rounding
point and changes the signed-Q8 activation scale group from 32 values to 16.
No timing or promotion credit is produced by this harness.
"""
from __future__ import annotations

import argparse,json,os,pathlib,subprocess,sys
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL,LOCK,PYTHON
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _semantic_comparison


def child(arm:str,depth:int,count:int,max_context:int,blocks:int,start_block:int,out:pathlib.Path):
  os.environ.update(DEV="NV",PROFILE="0")
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  model=_load(MODEL,max_context)
  model._flash_decode_tile_geometry_lease={"split_count":6,"llama_vec_wide":True,"token_bound":768,"combine_register_weights":True}
  model._flash_decode_block_geometry_overrides={i:{"o_q8_fine_owned":True} for i in range(start_block,start_block+blocks)} if arm=="candidate" else {}
  model._decode_direct_greedy_promoted=True;model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0,diagnostic_full_logits=True);tokens=[];logits=[]
  try:
    for _ in range(count):
      token,full=next(gen)
      if full is None:continue
      arr=full.numpy().reshape(-1)
      if not np.isfinite(arr).all():raise RuntimeError(f"non-finite logits step={len(tokens)}")
      if int(token)!=int(arr.argmax()):raise RuntimeError(f"sample/logit binding mismatch step={len(tokens)}")
      tokens.append(int(token));logits.append(arr)
  finally:gen.close()
  values=np.stack(logits);row={"schema":"tinygrad.nv_flash_combine_q8_fine_o_qualification.v1","arm":arm,"depth":depth,
    "count":len(tokens),"blocks":blocks if arm=="candidate" else 0,"start_block":start_block,"tokens":tokens,
    "finite":bool(np.isfinite(values).all()),"shape":list(values.shape)}
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n");np.savez_compressed(out.with_suffix(".npz"),logits=values);return row


def driver(a):
  root=a.out.with_suffix("");root.mkdir(parents=True,exist_ok=True);rows={};arrays={}
  for arm in ("control","candidate"):
    out=root/f"{arm}.json";cmd=["timeout",str(a.timeout),"flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
      "--arm",arm,"--depth",str(a.depth),"--count",str(a.count),"--max-context",str(a.max_context),"--blocks",str(a.blocks),
      "--start-block",str(a.start_block),"--out",str(out)]
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,"PYTHONPATH":str(ROOT)})
    if run.returncode:raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-8000:]}")
    rows[arm]=json.loads(out.read_text());arrays[arm]=np.load(out.with_suffix(".npz"))["logits"]
  comparison=_semantic_comparison(arrays["control"],arrays["candidate"],rows["control"],rows["candidate"])
  result={"schema":"tinygrad.nv_flash_combine_q8_fine_o_qualification.v1","blocks":a.blocks,"start_block":a.start_block,
    "representation":{"activation_values":4096,"quantized_bytes":4096,"scale_group":16,"metadata_groups":256,
      "metadata_bytes":1024,"packet_bytes":5120,"current_q8_1_packet_bytes":4608,"metadata_growth_bytes":512,
      "packet_vs_fp16_bytes_saved":3072,"fp16_rounding_point_preserved":True},"rows":rows,"comparison":comparison,
    "verdict":"PASS" if comparison["semantic_pass"] else "FAIL","note":"quality gate only; timing deliberately forbidden"}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result


def main():
  ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--arm",choices=("control","candidate"));ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--count",type=int,default=12);ap.add_argument("--max-context",type=int,default=768);ap.add_argument("--blocks",type=int,default=36)
  ap.add_argument("--start-block",type=int,default=0);ap.add_argument("--timeout",type=int,default=1800);ap.add_argument("--out",type=pathlib.Path,required=True);a=ap.parse_args()
  if a.start_block<0 or a.start_block+a.blocks>36:raise ValueError("selected block range is outside 0..35")
  result=child(a.arm,a.depth,a.count,a.max_context,a.blocks,a.start_block,a.out) if a.arm else driver(a);print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":main()
