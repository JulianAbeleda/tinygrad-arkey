#!/usr/bin/env python3
"""Profile and reverse-wall gate for shared-Q8 Q4/Q4 or Q4/Q6 K/V producers."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from collections import Counter
from dataclasses import replace

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL,LOCK,PYTHON,_flush_final_timestamps,_gpu_state,_install_graph_tracker,_per_name_table,_replay_metrics)


def _install(model,candidate:bool,mixed:bool=False,triple:bool=False) -> list[int]:
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear,Q6KPrimitiveLinear
  eligible=[]
  for index,block in enumerate(model.blk):
    current=getattr(block,"_shared_q8_attention_admission",None)
    if current is None: continue
    v_type=Q6KPrimitiveLinear if mixed else Q4KPrimitiveLinear
    if not (isinstance(getattr(block,"attn_k",None),Q4KPrimitiveLinear) and isinstance(getattr(block,"attn_v",None),v_type)): continue
    if not (current.cooperative_q4 and current.q4_direct_output):
      raise RuntimeError(f"block {index} lacks the promoted cooperative direct-output prerequisite")
    if triple and not mixed and not isinstance(getattr(block,"attn_v",None),Q4KPrimitiveLinear):
      raise RuntimeError(f"triple QKV requires Q4/Q4 or Q4/Q6 block, got {index}")
    if triple and not mixed:
      k_words=block.attn_k.q4k_storage.words
      v_words=block.attn_v.q4k_storage.words
      # Equalize persistent allocation/address topology in both arms. The
      # control arm does not consume this view, but allocating it only for the
      # candidate would make a production wall comparison confounded by memory
      # placement and allocator pressure.
      block.attn_q._shared_q8_qkv_words=k_words.cat(v_words,dim=0).contiguous().realize()
    block._shared_q8_attention_admission=replace(current,q4_kv_pair_output=False if (mixed or triple) else candidate,
      q4_q6_kv_pair_output=candidate if mixed and not triple else False,
      q4_qkv_triple_output=candidate if triple and not mixed else False,
      q4_q6_qkv_triple_output=candidate if triple and mixed else False)
    eligible.append(index)
  expected=8 if mixed else 9
  if len(eligible)!=expected: raise RuntimeError(f"expected {expected} shared-Q8 {'Q4/Q6' if mixed else 'Q4/Q4'} pairs, got {eligible}")
  return eligible


def _replays(lines:list[dict]) -> tuple[tuple[int,...],list[list[dict]]]:
  sizes=[len(x.get("entries",())) for x in lines]; plen=4
  candidates=Counter(tuple(sizes[i:i+plen]) for i in range(len(sizes)-plen+1)
    if sizes[i]<sizes[i+1]<sizes[i+2]<sizes[i+3])
  if not candidates: raise RuntimeError(f"no stable four-graph decode cycle in {Counter(sizes)}")
  pattern=candidates.most_common(1)[0][0]; out=[]; i=0
  while i+plen<=len(lines):
    if tuple(sizes[i:i+plen])==pattern:
      out.append([entry for row in lines[i:i+plen] for entry in row.get("entries",())]); i+=plen
    else: i+=1
  if len(out)<4: raise RuntimeError(f"only {len(out)} complete replays for {pattern}")
  return pattern,out


def _run_tokens(candidate:bool,depth:int,count:int,max_context:int,reps:int,mixed:bool=False,triple:bool=False):
  os.environ["DEV"]="NV"
  os.environ.pop("TINYGRAD_Q4K_KV_PAIR_DISABLE",None)
  os.environ.pop("TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE",None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load,_prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context); eligible=_install(model,candidate,mixed,triple)
  model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,Device["NV"],count,reps)
  finally: gen.close()
  return settled,eligible


def profile_child(candidate:bool,depth:int,count:int,max_context:int,reps:int,profile:pathlib.Path,out:pathlib.Path,mixed:bool=False,triple:bool=False)->dict:
  os.environ.update(PROFILE="1",HCQ_GRAPH_PROFILE_JSON=str(profile)); profile.unlink(missing_ok=True); _install_graph_tracker()
  from tinygrad import Device
  settled,eligible=_run_tokens(candidate,depth,count,max_context,reps,mixed,triple)
  Device["NV"].synchronize(); _flush_final_timestamps(); Device["NV"].synchronize()
  lines=[json.loads(x) for x in profile.read_text().splitlines() if x.strip()]; pattern,replays=_replays(lines); steady=replays[3:]
  metrics=[_replay_metrics(x) for x in steady]
  ledger={key:round(statistics.median(float(row[key]) for row in metrics),3) for key in
    ("node_count","node_sum_us","union_us","overlap_us","span_us")}
  result={"schema":"tinygrad.nv_shared_q8_kv_pair_qualification.v1","mode":"profile-child",
    "arm":"candidate" if candidate else "control","variant":"q4q6" if mixed else "q4q4","triple":triple,
    "depth":depth,"count":count,"reps":reps,"max_context":max_context,
    "gpu_state":_gpu_state(),"eligible_blocks":eligible,"settled":settled,"group_pattern":pattern,
    "complete_replay_count":len(replays),"steady_replay_count":len(steady),"ledger":ledger,"per_name_table":_per_name_table(steady)}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def timing_child(candidate:bool,depth:int,count:int,max_context:int,reps:int,out:pathlib.Path,mixed:bool=False,triple:bool=False)->dict:
  settled,eligible=_run_tokens(candidate,depth,count,max_context,reps,mixed,triple)
  result={"schema":"tinygrad.nv_shared_q8_kv_pair_qualification.v1","mode":"timing-child",
    "arm":"candidate" if candidate else "control","variant":"q4q6" if mixed else "q4q4","triple":triple,
    "depth":depth,"count":count,"reps":reps,"max_context":max_context,
    "gpu_state":_gpu_state(),"eligible_blocks":eligible,**settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def _child(mode:str,candidate:bool,label:str,root:pathlib.Path,args)->dict:
  out=root/f"{label}.json"; cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
    "--mode",f"{mode}-child","--depth",str(args.depth),"--count",str(args.count),"--max-context",str(args.max_context),
    "--reps",str(args.reps),"--out",str(out)]
  if candidate: cmd.append("--candidate")
  if args.mixed: cmd.append("--mixed")
  if args.triple: cmd.append("--triple")
  if mode=="profile": cmd += ["--profile-jsonl",str(root/f"{label}.profile.jsonl")]
  env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"}; env.pop("TINYGRAD_Q4K_KV_PAIR_DISABLE",None); env.pop("TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE",None)
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env)
  if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())


def profile_driver(args)->dict:
  root=args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True,exist_ok=True)
  control=_child("profile",False,"control",root,args); candidate=_child("profile",True,"candidate",root,args)
  hashes={control["settled"]["token_stream_hash"],candidate["settled"]["token_stream_hash"]}
  pair_names={n:v for n,v in candidate["per_name_table"].items() if "pair_direct_1024_4096" in n}
  result={"schema":"tinygrad.nv_shared_q8_kv_pair_qualification.v1","mode":"profile","variant":"q4q6" if args.mixed else "q4q4","triple":args.triple,
    "all_token_hashes_equal":len(hashes)==1,
    "ledger_delta":{k:round(float(candidate["ledger"][k])-float(control["ledger"][k]),3) for k in control["ledger"]},
    "structural":{"control_nodes":control["ledger"]["node_count"],"candidate_nodes":candidate["ledger"]["node_count"],
      "eligible_blocks":candidate["eligible_blocks"],"pair_names":pair_names,"control_pattern":control["group_pattern"],
      "candidate_pattern":candidate["group_pattern"]},
    "arms":[{k:v for k,v in row.items() if k!="per_name_table"} for row in (control,candidate)]}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def timing_driver(args)->dict:
  root=args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True,exist_ok=True)
  arms=[_child("timing",False,"control_a",root,args),_child("timing",True,"candidate",root,args),
    _child("timing",False,"control_c",root,args)]
  midpoint=statistics.median((arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]))
  candidate=arms[1]["median_ms_per_token"]; hashes={row["token_stream_hash"] for row in arms}
  result={"schema":"tinygrad.nv_shared_q8_kv_pair_qualification.v1","mode":"timing","variant":"q4q6" if args.mixed else "q4q4","triple":args.triple,
    "depth":args.depth,"count":args.count,"reps":args.reps,
    "control_a_ms_per_token":arms[0]["median_ms_per_token"],"candidate_ms_per_token":candidate,
    "control_c_ms_per_token":arms[2]["median_ms_per_token"],"control_midpoint_ms_per_token":midpoint,
    "recovery_us_per_token":(midpoint-candidate)*1000.0,"all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate<min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]) else "NO_GO_WALL",
    "arms":arms}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main()->int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("profile","profile-child","timing","timing-child"),default="profile")
  ap.add_argument("--candidate",action="store_true"); ap.add_argument("--mixed",action="store_true"); ap.add_argument("--triple",action="store_true")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=32)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=3)
  ap.add_argument("--profile-jsonl",type=pathlib.Path); ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.mode=="profile-child":
    if args.profile_jsonl is None: raise SystemExit("--profile-jsonl is required")
    result=profile_child(args.candidate,args.depth,args.count,args.max_context,args.reps,args.profile_jsonl,args.out,args.mixed,args.triple)
  elif args.mode=="timing-child": result=timing_child(args.candidate,args.depth,args.count,args.max_context,args.reps,args.out,args.mixed,args.triple)
  elif args.mode=="profile": result=profile_driver(args)
  else: result=timing_driver(args)
  print(json.dumps(result if "per_name_table" not in result else {k:v for k,v in result.items() if k!="per_name_table"},indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
