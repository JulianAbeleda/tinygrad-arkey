#!/usr/bin/env python3
"""Production wall bracket for targeted shared-Q8 Q || paired-K/V placement.

Candidate is a harness-only HCQGraph placement override.  It pins the exact
shared-Q8 provider and Q projection to compute GPFIFO 0 and its installed
Q4/Q4 paired K/V projection to GPFIFO 1.  All other nodes retain the current
ready-placement policy and ordinary dependency resolution.  Control is the
unmodified installed scheduler.  No runtime file or route policy is changed.
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, statistics, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL,LOCK,_gpu_state

PROVIDER="rmsnorm_q8_1_llama_provider_4096"
QPROJ="q4k_warp_coop_q8_dp4a_direct_4096_4096"
KVPAIR="q4k_warp_coop_q8_dp4a_pair_direct_1024_4096"
TARGETS=(PROVIDER,QPROJ,KVPAIR)


def _install_targeted_placement() -> None:
  from tinygrad.runtime.graph.hcq import HCQGraph
  original=HCQGraph._pick_compute_queue
  def targeted(self,dev,runtime,graph_idx=-1,rdeps_peek=None):
    queues=self.compute_queues[dev]
    if dev.device.split(":",1)[0]=="NV" and len(queues)>1:
      if runtime.name in (PROVIDER,QPROJ): return queues[0]
      if runtime.name==KVPAIR: return queues[1]
    return original(self,dev,runtime,graph_idx,rdeps_peek)
  HCQGraph._pick_compute_queue=targeted


def _census(path:pathlib.Path) -> dict:
  lines=[json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
  counts={name:collections.Counter() for name in TARGETS}
  graph_hits=collections.Counter()
  for row in lines:
    touched=False
    for rec in row.get("assignments",()):
      if rec["name"] in counts:
        counts[rec["name"]][int(rec["queue"])]+=1; touched=True
    if touched: graph_hits["with_target"]+=1
  return {"graphs":len(lines),"graphs_with_target":graph_hits["with_target"],
          "target_queue_counts":{name:{str(k):v for k,v in sorted(c.items())} for name,c in counts.items()}}


def _install_aux_first_submit() -> None:
  """Reverse only replay submission order after the installed graph is built.

  HCQGraph construction, queue identities, dependency signals, the primary
  timeline owner, and queue binding all remain untouched.  Reordering the
  already-built list here changes the ``__call__`` submit loop from q0,q1 to
  q1,q0, matching the independently positive live-HCQ microgate.
  """
  from tinygrad.runtime.graph.hcq import HCQGraph
  original=HCQGraph.__init__
  def init(self,*args,**kwargs):
    original(self,*args,**kwargs)
    for dev,queues in self.compute_queues.items():
      if dev.device.split(":",1)[0]=="NV" and len(queues)>1: self.compute_queues[dev]=[queues[1],queues[0],*queues[2:]]
  HCQGraph.__init__=init


def timing_child(candidate:bool,variant:str,depth:int,count:int,max_context:int,reps:int,census:pathlib.Path,out:pathlib.Path)->dict:
  if candidate and variant=="target-placement": _install_targeted_placement()
  if candidate and variant=="aux-first": _install_aux_first_submit()
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context)
  model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,Device[Device.DEFAULT],count,reps)
  finally: gen.close()
  result={"schema":"tinygrad.nv_qkv_targeted_placement_wall_bracket.v1","mode":"timing-child",
    "arm":"candidate" if candidate else "control","variant":variant,
    "placement":("shared_q4q4_targeted" if variant=="target-placement" else "installed_ready_aux_first_submit") if candidate else "installed_ready",
    "gpu_state":_gpu_state(),"depth":depth,"count":count,"reps":reps,"settled_continuous":True,
    "census":_census(census),**settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def bracket(variant:str,depth:int,count:int,max_context:int,reps:int,out:pathlib.Path)->dict:
  root=pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True,exist_ok=True)
  def child(candidate:bool,label:str)->dict:
    arm_out,census=root/f"{label}.json",root/f"{label}.census.jsonl"
    census.unlink(missing_ok=True)
    cmd=["timeout","1800","flock","-w","600",LOCK,sys.executable,str(pathlib.Path(__file__).resolve()),
      "--mode","timing-child","--depth",str(depth),"--count",str(count),"--max-context",str(max_context),
      "--reps",str(reps),"--variant",variant,"--census-jsonl",str(census),"--out",str(arm_out)]
    if candidate: cmd.append("--candidate")
    env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV","HCQ_MULTI_QUEUE_CENSUS_JSON":str(census)}
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
    return json.loads(arm_out.read_text())
  arms=[child(False,"control_a"),child(True,"candidate"),child(False,"control_c")]
  midpoint=statistics.median((arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]))
  candidate=arms[1]["median_ms_per_token"]; hashes={x["token_stream_hash"] for x in arms}
  topology_changed=(variant!="target-placement" or
    arms[1]["census"]["target_queue_counts"]!=arms[0]["census"]["target_queue_counts"] or
    arms[1]["census"]["target_queue_counts"]!=arms[2]["census"]["target_queue_counts"])
  strict_wall=len(hashes)==1 and candidate<min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"])
  result={"schema":"tinygrad.nv_qkv_targeted_placement_wall_bracket.v1","mode":"reverse-bracket",
    "variant":variant,"depth":depth,"count":count,"reps":reps,
    "population":"installed two-GPFIFO decode graphs" if variant=="aux-first" else "nine shared-Q8 Q4/Q4 attention groups",
    "control_a_ms_per_token":arms[0]["median_ms_per_token"],"candidate_ms_per_token":candidate,
    "control_c_ms_per_token":arms[2]["median_ms_per_token"],"control_midpoint_ms_per_token":midpoint,
    "recovery_us_per_token":(midpoint-candidate)*1000.0,"candidate_speedup_pct":(midpoint/candidate-1)*100,
    "all_token_hashes_equal":len(hashes)==1,"token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "topology_changed":topology_changed,
    "verdict":"WALL_PASS" if strict_wall and topology_changed else "NO_OP_TOPOLOGY" if strict_wall else "NO_GO_WALL","arms":arms}
  out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main()->int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("timing","timing-child"),default="timing")
  ap.add_argument("--candidate",action="store_true"); ap.add_argument("--variant",choices=("target-placement","aux-first"),default="target-placement")
  ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--count",type=int,default=32); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--reps",type=int,default=7); ap.add_argument("--census-jsonl",type=pathlib.Path)
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.mode=="timing-child":
    if args.census_jsonl is None: raise SystemExit("--census-jsonl required")
    result=timing_child(args.candidate,args.variant,args.depth,args.count,args.max_context,args.reps,args.census_jsonl,args.out)
  else: result=bracket(args.variant,args.depth,args.count,args.max_context,args.reps,args.out)
  print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
