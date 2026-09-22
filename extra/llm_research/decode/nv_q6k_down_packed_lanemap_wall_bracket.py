#!/usr/bin/env python3
"""Reverse production wall bracket for the Q6_K FFN-down packed lane map."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL, LOCK, _gpu_state


def _set_packed_lanemap(model, enabled:bool) -> list[int]:
  from tinygrad.llm.q6k_ffn_down_mmvq import Q6KFFNDownMMVQAdmission
  from tinygrad.llm.qk_primitives import Q6KPrimitiveLinear
  indices=[]
  for index, block in enumerate(model.blk):
    ffn=getattr(block, "ffn_down", None)
    if not isinstance(ffn, Q6KPrimitiveLinear) or getattr(ffn, "route_role", "") != "ffn_down": continue
    if (getattr(ffn, "out_features", None), getattr(ffn, "in_features", None)) != (4096, 12288): continue
    current=getattr(ffn, "_q6k_ffn_down_mmvq_admission", None)
    if current is None or not getattr(current, "fp16_fma", False):
      raise RuntimeError(f"block {index} lacks the promoted fp16 Q6-down admission")
    ffn._q6k_ffn_down_mmvq_admission=Q6KFFNDownMMVQAdmission(
      index, fp16_fma=True, rows_per_block=1, packed_lanemap=enabled)
    indices.append(index)
  if len(indices) != 18: raise RuntimeError(f"expected 18 Q6-down blocks, got {indices}")
  return indices


def timing_child(candidate:bool, composed:bool, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path)->dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL, max_context); indices=_set_packed_lanemap(model, candidate)
  model._decode_direct_greedy_promoted=composed; model._decode_feedback_pingpong_promoted=composed
  gen=model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled=_settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally: gen.close()
  result={"schema":"tinygrad.nv_q6k_down_packed_lanemap_wall_bracket.v1","mode":"timing-child",
    "arm":"candidate" if candidate else "control","packed_lanemap":candidate,"composed":composed,"q6_down_blocks":indices,
    "gpu_state":_gpu_state(),"depth":depth,"count":count,"reps":reps,"settled_continuous":True,**settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def bracket(composed:bool, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path)->dict:
  root=pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  def child(candidate:bool,label:str)->dict:
    arm_out=root/f"{label}.json"
    cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,str(pathlib.Path(__file__).resolve()),
      "--mode","timing-child","--depth",str(depth),"--count",str(count),"--max-context",str(max_context),
      "--reps",str(reps),"--out",str(arm_out)]
    if candidate: cmd.append("--candidate")
    if composed: cmd.append("--composed")
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
      env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"})
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
    return json.loads(arm_out.read_text())
  arms=[child(False,"control_a"),child(True,"candidate"),child(False,"control_c")]
  midpoint=statistics.median((arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]))
  candidate=arms[1]["median_ms_per_token"]; hashes={x["token_stream_hash"] for x in arms}
  recovery=(midpoint-candidate)*1000.0
  strict_wall=len(hashes)==1 and candidate<min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"])
  endpoint_us=4515.39571875
  result={"schema":"tinygrad.nv_q6k_down_packed_lanemap_wall_bracket.v1","mode":"reverse-bracket",
    "depth":depth,"count":count,"reps":reps,"composed":composed,"population":"18 Q6_K FFN-down projections",
    "control_a_ms_per_token":arms[0]["median_ms_per_token"],"candidate_ms_per_token":candidate,
    "control_c_ms_per_token":arms[2]["median_ms_per_token"],"control_midpoint_ms_per_token":midpoint,
    "recovery_us_per_token":recovery,"candidate_speedup_pct":(midpoint/candidate-1)*100,
    "all_token_hashes_equal":len(hashes)==1,"token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "projection":{"conservative_endpoint_us":endpoint_us,"conservative_endpoint_tok_s":1e6/endpoint_us,
      "projected_us":endpoint_us-recovery,"projected_tok_s":1e6/(endpoint_us-recovery),
      "remaining_to_227_us":endpoint_us-recovery-1e6/227.0},
    "verdict":"WALL_PASS" if strict_wall else "NO_GO_WALL","arms":arms}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main()->int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("timing","timing-child"),default="timing")
  ap.add_argument("--candidate",action="store_true"); ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--composed",action="store_true")
  ap.add_argument("--count",type=int,default=32); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  result=(timing_child(args.candidate,args.composed,args.depth,args.count,args.max_context,args.reps,args.out)
          if args.mode=="timing-child" else bracket(args.composed,args.depth,args.count,args.max_context,args.reps,args.out))
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
