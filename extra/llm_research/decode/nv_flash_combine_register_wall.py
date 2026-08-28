#!/usr/bin/env python3
"""Reverse token-wall bracket for register-broadcast Flash combine weights."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL, LOCK, PYTHON


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path, installed:bool=False,
              splits6:bool=False, o_prefetch_groups:int=0, o_q8_owned:bool=False, o_q8_blocks:int=36) -> dict:
  os.environ.update(DEV="NV", PROFILE="0")
  os.environ.pop("TINYGRAD_FLASH_LOAD_SCHEDULE_DISABLE", None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

  dev=Device["NV"]; model=_load(MODEL,max_context)
  candidate=arm.startswith("candidate")
  if installed:
    if candidate: os.environ.pop("TINYGRAD_FLASH_COMBINE_REGISTER_DISABLE", None)
    else: os.environ["TINYGRAD_FLASH_COMBINE_REGISTER_DISABLE"]="1"
  else:
    geometry={"split_count":6 if splits6 else 8,"llama_vec_wide":True,"token_bound":768 if splits6 else 1024}
    if candidate or o_prefetch_groups or o_q8_owned: geometry["combine_register_weights"]=True
    if candidate and o_prefetch_groups: geometry["o_successor_prefetch_groups"]=o_prefetch_groups
    model._flash_decode_tile_geometry_lease=geometry
    model._flash_decode_block_geometry_overrides={index:{"o_q8_owned":True}
      for index in range(o_q8_blocks)} if candidate and o_q8_owned else {}
  model._decode_direct_greedy_promoted=True; model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,dev,count,reps)
  finally: gen.close()
  result={"schema":"tinygrad.nv_flash_combine_register_wall.v1","arm":arm,"depth":depth,"count":count,
    "reps":reps,"max_context":max_context,"matched_explicit_geometry":not installed,"installed_route":installed,
    "split_count":6 if splits6 else 8,
    "combine_register_weights":candidate or bool(o_prefetch_groups) or o_q8_owned,
    "o_successor_prefetch_groups":o_prefetch_groups if candidate else 0,
    "o_q8_owned":bool(candidate and o_q8_owned),"o_q8_blocks":o_q8_blocks if candidate and o_q8_owned else 0,"settled":settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  return result


def _child(arm:str,root:pathlib.Path,args)->dict:
  out=root/f"{arm}.json"
  cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
    "--arm",arm,"--depth",str(args.depth),"--count",str(args.count),"--max-context",str(args.max_context),
    "--reps",str(args.reps),"--out",str(out)] + (["--installed"] if args.installed else []) + \
    (["--splits6"] if args.splits6 else []) + (["--o-prefetch-groups",str(args.o_prefetch_groups)] if args.o_prefetch_groups else []) + \
    (["--o-q8-owned","--o-q8-blocks",str(args.o_q8_blocks)] if args.o_q8_owned else [])
  env={**os.environ,"PYTHONPATH":str(ROOT),"PROFILE":"0"}; env.pop("TINYGRAD_FLASH_LOAD_SCHEDULE_DISABLE",None)
  env.pop("TINYGRAD_FLASH_COMBINE_REGISTER_DISABLE",None)
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-6000:]}")
  return json.loads(out.read_text())


def driver(args)->dict:
  root=pathlib.Path(str(args.out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  names=("candidate_a","control","candidate_c") if args.candidate_controls else ("control_a","candidate","control_c")
  arms=[_child(name,root,args) for name in names]
  def wall(x): return float(x["settled"]["median_ms_per_token"])*1000.0
  a,b,c=map(wall,arms); hashes={x["settled"]["token_stream_hash"] for x in arms}
  control,candidate=(b,statistics.median((a,c))) if args.candidate_controls else (statistics.median((a,c)),b)
  result={"schema":"tinygrad.nv_flash_combine_register_wall.v1","mode":"reverse-bracket-unprofiled",
    "depth":args.depth,"count":args.count,"reps":args.reps,"max_context":args.max_context,"arm_order":list(names),
    "installed_route":args.installed,"split_count":6 if args.splits6 else 8,
    "o_successor_prefetch_groups":args.o_prefetch_groups,
    "o_q8_owned":args.o_q8_owned,
    "o_q8_blocks":args.o_q8_blocks,
    "all_token_hashes_equal":len(hashes)==1,"token_stream_hashes":sorted(hashes),
    "walls_us_per_token":{"arm_a":a,"arm_b":b,"arm_c":c,"control":control,"candidate":candidate,
                           "candidate_delta":candidate-control},
    "tokens_per_second":{"control":1e6/control,"candidate":1e6/candidate,
                         "candidate_delta":1e6/candidate-1e6/control},"arms":arms}
  passed=(control>max(a,c) if args.candidate_controls else candidate<min(a,c))
  result["verdict"]="COMBINE_REGISTER_WALL_PASS" if len(hashes)==1 and passed else "NO_GO_WALL"
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main()->int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--arm",choices=("control_a","candidate","control_c","candidate_a","control","candidate_c"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=24)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--candidate-controls",action="store_true"); ap.add_argument("--installed",action="store_true")
  ap.add_argument("--splits6",action="store_true")
  ap.add_argument("--o-prefetch-groups",type=int,choices=(0,1,2,4),default=0)
  ap.add_argument("--o-q8-owned",action="store_true")
  ap.add_argument("--o-q8-blocks",type=int,default=36)
  ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.depth+7+args.count*args.reps>args.max_context: raise ValueError("timed continuation must fit max-context")
  result=run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.out,args.installed,args.splits6,args.o_prefetch_groups,args.o_q8_owned,args.o_q8_blocks) if args.arm else driver(args)
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
