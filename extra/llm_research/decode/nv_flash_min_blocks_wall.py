#!/usr/bin/env python3
"""Reverse token-wall bracket for the program-scoped NV Flash min-blocks lease."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_flash_llama_vec_wide_qualification import MODEL, LOCK, PYTHON

POLICY = "prefix:flash_vec_llama_score_pv_"


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path,
              profile_jsonl:pathlib.Path|None=None, jit_batch_size:int=32) -> dict:
  os.environ.update(DEV="NV", PROFILE="1" if profile_jsonl else "0")
  os.environ["JIT_BATCH_SIZE"] = str(jit_batch_size)
  if profile_jsonl:
    profile_jsonl.parent.mkdir(parents=True,exist_ok=True); profile_jsonl.unlink(missing_ok=True)
    os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(profile_jsonl)
    from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import _install_graph_tracker
    _install_graph_tracker()
  else: os.environ.pop("HCQ_GRAPH_PROFILE_JSON", None)
  # Exercise the installed load-schedule admission, including its capture cap,
  # rather than recreating half of the policy through research environment
  # switches.  Control uses the explicit load-time rollback.
  os.environ.pop("NV_MIN_BLOCKS_PROGRAMS", None)
  if arm.startswith("candidate"): os.environ.pop("TINYGRAD_FLASH_LOAD_SCHEDULE_DISABLE", None)
  else: os.environ["TINYGRAD_FLASH_LOAD_SCHEDULE_DISABLE"] = "1"
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  dev=Device["NV"]; model=_load(MODEL,max_context)
  model._decode_direct_greedy_promoted=True; model._decode_feedback_pingpong_promoted=True
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,dev,count,reps)
  finally: gen.close()
  profile_summary=None
  if profile_jsonl:
    from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
      _complete_replays,_flush_final_timestamps,_per_name_table,_replay_metrics)
    dev.synchronize(); _flush_final_timestamps(); dev.synchronize()
    lines=[json.loads(line) for line in profile_jsonl.read_text().splitlines() if line.strip()]
    replays=_complete_replays(lines); steady=replays[3:]
    if not steady: raise RuntimeError(f"no steady replays in {profile_jsonl}")
    metrics=[_replay_metrics(x) for x in steady]
    profile_summary={
      "complete_replay_count":len(replays),"steady_replay_count":len(steady),
      "ledger":{key:round(statistics.median(float(x[key]) for x in metrics),3) for key in
                ("node_sum_us","union_us","overlap_us","span_us")},
      "flash_rows":{name:row for name,row in _per_name_table(steady).items()
                    if name.startswith("flash_vec_llama") or name.startswith("flash_fused_gmax")}}
  result={"schema":"tinygrad.nv_flash_min_blocks_wall.v1","arm":arm,"depth":depth,"count":count,
    "reps":reps,"max_context":max_context,"jit_batch_size":jit_batch_size,
    "min_blocks_policy":"installed_flash_load_schedule" if arm.startswith("candidate") else None,
    "settled":settled,"profile_jsonl":str(profile_jsonl) if profile_jsonl else None,"profile_summary":profile_summary}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
  return result


def _child(arm:str,root:pathlib.Path,args) -> dict:
  out=root/f"{arm}.json"
  cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
    "--arm",arm,"--depth",str(args.depth),"--count",str(args.count),"--max-context",str(args.max_context),
    "--reps",str(args.reps),"--jit-batch-size",str(args.jit_batch_size),"--out",str(out)]
  env={**os.environ,"PYTHONPATH":str(ROOT),"PROFILE":"0"}; env.pop("NV_MIN_BLOCKS_PROGRAMS",None)
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-6000:]}")
  return json.loads(out.read_text())


def driver(args) -> dict:
  root=pathlib.Path(str(args.out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  names=("candidate_a","control","candidate_c") if args.candidate_controls else ("control_a","candidate","control_c")
  arms=[_child(name,root,args) for name in names]
  def wall(x): return float(x["settled"]["median_ms_per_token"])*1000.0
  ca,cb,cc=map(wall,arms); hashes={x["settled"]["token_stream_hash"] for x in arms}
  control,candidate=(cb,statistics.median((ca,cc))) if args.candidate_controls else (statistics.median((ca,cc)),cb)
  result={"schema":"tinygrad.nv_flash_min_blocks_wall.v1","mode":"reverse-bracket-unprofiled",
    "depth":args.depth,"count":args.count,"reps":args.reps,"max_context":args.max_context,"arm_order":list(names),
    "jit_batch_size":args.jit_batch_size,
    "all_token_hashes_equal":len(hashes)==1,"token_stream_hashes":sorted(hashes),
    "walls_us_per_token":{"arm_a":ca,"arm_b":cb,"arm_c":cc,"control":control,"candidate":candidate,
                          "candidate_delta":candidate-control},
    "tokens_per_second":{"control":1e6/control,"candidate":1e6/candidate,
                         "candidate_delta":1e6/candidate-1e6/control},
    "verdict":"MIN_BLOCKS_WALL_PASS" if len(hashes)==1 and
      (control > max(ca,cc) if args.candidate_controls else candidate < min(ca,cc)) else "NO_GO_WALL","arms":arms}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main() -> int:
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--arm",choices=("control_a","candidate","control_c","candidate_a","control","candidate_c"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=16)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=9)
  ap.add_argument("--jit-batch-size",type=int,default=32)
  ap.add_argument("--candidate-controls",action="store_true")
  ap.add_argument("--profile-jsonl",type=pathlib.Path,help="child-only graph profile destination")
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.depth+7+args.count*args.reps > args.max_context: raise ValueError("timed continuation must fit max-context")
  result=run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.out,args.profile_jsonl,args.jit_batch_size) if args.arm else driver(args)
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
