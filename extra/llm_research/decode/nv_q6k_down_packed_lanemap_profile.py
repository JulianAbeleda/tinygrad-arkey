#!/usr/bin/env python3
"""Production profile closure for scalar versus packed-lanemap Q6_K down."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from collections import Counter

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, _flush_final_timestamps, _gpu_state,
  _install_graph_tracker, _per_name_table, _replay_metrics)
from extra.llm_research.decode.nv_q6k_down_packed_lanemap_wall_bracket import _set_packed_lanemap

CONTROL="q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"
CANDIDATE="q6k_fp16_packed_lanemap_4096_12288_epi_ffnresadd"


def _current_decode_replays(lines:list[dict])->list[list[dict]]:
  """Recover current 462-node token replays: 32 + 64 + 128 + 238.

  The older profile helper expects a 256-node fourth group. Paired K/V and
  native-tail landings changed the current graph family, while prefill still
  produces 256/394 groups. Select the most common tail after the stable
  32/64/128 prefix so the parser follows the current decode population.
  """
  sizes=[len(x.get("entries",[])) for x in lines]; tails=Counter()
  for i in range(len(sizes)-3):
    if tuple(sizes[i:i+3])==(32,64,128): tails[sizes[i+3]]+=1
  if not tails: return []
  group=(32,64,128,tails.most_common(1)[0][0]); replays=[]; i=0
  while i+4<=len(lines):
    if tuple(sizes[i:i+4])==group:
      replays.append([e for row in lines[i:i+4] for e in row.get("entries",[])]); i+=4
    else: i+=1
  return replays


def _reparse(row:dict)->dict:
  path=pathlib.Path(row["profile_jsonl"])
  lines=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  replays=_current_decode_replays(lines); steady=replays[3:]
  if not steady: raise RuntimeError(f"no current decode replays in {path}")
  metrics=[_replay_metrics(x) for x in steady]
  row["complete_replay_count"]=len(replays); row["steady_replay_count"]=len(steady)
  row["ledger"]={key:round(statistics.median(float(x[key]) for x in metrics),3) for key in
                 ("node_sum_us","union_us","overlap_us","span_us")}
  row["per_name_table"]=_per_name_table(steady)
  return row


def run_child(arm:str,depth:int,count:int,max_context:int,reps:int,profile_jsonl:pathlib.Path,out:pathlib.Path)->dict:
  os.environ["DEV"]="NV"; os.environ["PROFILE"]="1"; os.environ["HCQ_GRAPH_PROFILE_JSON"]=str(profile_jsonl)
  profile_jsonl.unlink(missing_ok=True); _install_graph_tracker()
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  dev=Device["NV"]; model=_load(MODEL,max_context); indices=_set_packed_lanemap(model,arm=="candidate")
  model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,dev,count,reps)
  finally: gen.close()
  dev.synchronize(); _flush_final_timestamps(); dev.synchronize()
  lines=[json.loads(line) for line in profile_jsonl.read_text().splitlines() if line.strip()]
  sizes=[len(x.get("entries",[])) for x in lines]; replays=_current_decode_replays(lines); steady=replays[3:]
  metrics=[_replay_metrics(x) for x in steady]
  ledger={key:round(statistics.median(float(x[key]) for x in metrics),3) for key in
          ("node_sum_us","union_us","overlap_us","span_us")}
  result={"schema":"tinygrad.nv_q6k_down_packed_lanemap_profile.v1","arm":arm,"depth":depth,"count":count,
    "reps":reps,"max_context":max_context,"gpu_state":_gpu_state(),"q6_down_blocks":indices,"settled":settled,
    "profile_jsonl":str(profile_jsonl),"group_size_histogram":dict(sorted(Counter(sizes).items())),
    "complete_replay_count":len(replays),"steady_replay_count":len(steady),"ledger":ledger,
    "per_name_table":_per_name_table(steady)}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def child(arm:str,root:pathlib.Path,depth:int,count:int,max_context:int,reps:int)->dict:
  out,profile=root/f"{arm}.json",root/f"{arm}.profile.jsonl"
  cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
    "--arm",arm,"--depth",str(depth),"--count",str(count),"--max-context",str(max_context),"--reps",str(reps),
    "--profile-jsonl",str(profile),"--out",str(out)]
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"})
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())


def driver(depth:int,count:int,max_context:int,reps:int,out:pathlib.Path,reuse_existing:bool=False)->dict:
  root=pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  arms=([_reparse(json.loads((root/f"{arm}.json").read_text())) for arm in ("control_a","candidate","control_c")]
        if reuse_existing else
        [child("control_a",root,depth,count,max_context,reps),child("candidate",root,depth,count,max_context,reps),
         child("control_c",root,depth,count,max_context,reps)])
  def wall(x): return float(x["settled"]["median_ms_per_token"])*1000
  control_device=statistics.median(float(arms[i]["per_name_table"][CONTROL]["median_us"]) for i in (0,2))
  candidate_device=float(arms[1]["per_name_table"][CANDIDATE]["median_us"])
  control_wall=statistics.median((wall(arms[0]),wall(arms[2]))); candidate_wall=wall(arms[1])
  control_ledger={key:statistics.median(float(arms[i]["ledger"][key]) for i in (0,2)) for key in arms[0]["ledger"]}
  candidate_ledger={key:float(arms[1]["ledger"][key]) for key in arms[1]["ledger"]}
  hashes={x["settled"]["token_stream_hash"] for x in arms}
  result={"schema":"tinygrad.nv_q6k_down_packed_lanemap_profile.v1","mode":"reverse-bracket-profile",
    "depth":depth,"count":count,"reps":reps,"all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "walls_us_per_token":{"control_a":wall(arms[0]),"candidate":candidate_wall,"control_c":wall(arms[2]),
      "control_midpoint":control_wall,"candidate_minus_control":candidate_wall-control_wall},
    "q6_down_device":{"calls_per_token":18,"control_us_per_token":control_device,
      "candidate_us_per_token":candidate_device,"delta_us_per_token":candidate_device-control_device,
      "control_us_per_call":control_device/18,"candidate_us_per_call":candidate_device/18},
    "ledger_control":control_ledger,"ledger_candidate":candidate_ledger,
    "closure":{"node_sum_delta_us":candidate_ledger["node_sum_us"]-control_ledger["node_sum_us"],
      "union_delta_us":candidate_ledger["union_us"]-control_ledger["union_us"]},
    "arms":[{k:v for k,v in x.items() if k!="per_name_table"} for x in arms]}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main()->int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=("control_a","candidate","control_c"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=32)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=3)
  ap.add_argument("--reuse-existing",action="store_true")
  ap.add_argument("--profile-jsonl",type=pathlib.Path); ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.arm:
    if args.profile_jsonl is None: raise SystemExit("--profile-jsonl required")
    result=run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.profile_jsonl,args.out)
  else: result=driver(args.depth,args.count,args.max_context,args.reps,args.out,args.reuse_existing)
  print(json.dumps(result if not args.arm else {k:v for k,v in result.items() if k!="per_name_table"},indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
