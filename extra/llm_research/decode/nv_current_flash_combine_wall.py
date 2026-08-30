#!/usr/bin/env python3
"""Current composed-token reverse bracket for flash-combine width."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL, LOCK, _gpu_state
from extra.llm_research.decode.nv_flash_combine_width_profile import _install_combine_width


def child(width:int|None, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context); _install_combine_width(model,width)
  model._decode_direct_greedy_promoted=True; model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally: gen.close()
  result={"schema":"tinygrad.nv_current_flash_combine_wall.v1","width":width,"composed":True,
    "gpu_state":_gpu_state(),"depth":depth,"count":count,"reps":reps,**settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def bracket(width:int, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root=pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  def arm(candidate:bool,label:str) -> dict:
    dst=root/f"{label}.json"; cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,str(pathlib.Path(__file__).resolve()),
      "--mode","child","--width",str(width),"--depth",str(depth),"--count",str(count),"--max-context",str(max_context),
      "--reps",str(reps),"--out",str(dst)]
    if candidate: cmd.append("--candidate")
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"})
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    return json.loads(dst.read_text())
  arms=[arm(False,"control_a"),arm(True,"candidate"),arm(False,"control_c")]
  midpoint=(arms[0]["median_ms_per_token"]+arms[2]["median_ms_per_token"])/2
  candidate=arms[1]["median_ms_per_token"]; hashes={a["token_stream_hash"] for a in arms}
  result={"schema":"tinygrad.nv_current_flash_combine_wall.v1","mode":"reverse-bracket","width":width,
    "control_a_ms_per_token":arms[0]["median_ms_per_token"],"candidate_ms_per_token":candidate,
    "control_c_ms_per_token":arms[2]["median_ms_per_token"],"control_midpoint_ms_per_token":midpoint,
    "recovery_us_per_token":(midpoint-candidate)*1000,"all_token_hashes_equal":len(hashes)==1,
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate < min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]) else "NO_GO_WALL",
    "arms":arms}; out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("timing","child"),default="timing"); ap.add_argument("--candidate",action="store_true")
  ap.add_argument("--width",type=int,choices=(64,128),default=128); ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--count",type=int,default=16); ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--out",type=pathlib.Path,required=True); a=ap.parse_args()
  result=child(a.width if a.candidate else None,a.depth,a.count,a.max_context,a.reps,a.out) if a.mode=="child" else bracket(a.width,a.depth,a.count,a.max_context,a.reps,a.out)
  print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
