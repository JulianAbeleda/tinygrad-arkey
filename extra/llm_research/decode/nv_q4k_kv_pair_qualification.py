#!/usr/bin/env python3
"""Profile and reverse-wall gate for ordinary Q4/Q4 K/V pair producers.

Both arms disable the already-promoted terminal K/V cache sink so this phase
isolates projection fusion from the paired-output/cache composition problem.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, _flush_final_timestamps, _gpu_state, _install_graph_tracker,
  _per_name_table, _replay_metrics)


def _install(model, candidate:bool, triple:bool=False) -> list[int]:
  if not candidate and not triple: return []
  from tinygrad.llm.q4k_kv_pair import Q4KKVPairAdmission, Q4KQKVAdmission
  admitted=[]
  for index, block in enumerate(model.blk):
    # This first landing deliberately excludes the separate shared-Q8 triple.
    if getattr(block, "_shared_q8_attention_admission", None) is not None: continue
    if not (hasattr(getattr(block, "attn_k", None), "q4k_storage") and
            hasattr(getattr(block, "attn_v", None), "q4k_storage")): continue
    if triple:
      # Both arms own the same packed K-then-V allocation so allocator/address
      # topology cannot masquerade as a producer win.
      block.attn_q._q4k_qkv_words=block.attn_k.q4k_storage.words.cat(
        block.attn_v.q4k_storage.words,dim=0).contiguous().realize()
      if candidate: block._q4k_qkv_admission=Q4KQKVAdmission(index)
    elif candidate: block._q4k_kv_pair_admission=Q4KKVPairAdmission(index)
    admitted.append(index)
  if triple and len(admitted)!=9: raise RuntimeError(f"expected 9 ordinary Q4/Q4/Q4 blocks, got {admitted}")
  return admitted


def _compose_producer_sink() -> bool: return os.environ.get("NV_Q4KV_COMPOSE_PRODUCER_SINK", "") not in ("", "0")


def _pattern_and_replays(lines:list[dict], triple_candidate:bool=False) -> tuple[tuple[int, ...], list[list[dict]]]:
  """Discover the stable decode cycle (four graphs with producer sink, else five)."""
  sizes=[len(x.get("entries",())) for x in lines]
  plen=4 if (_compose_producer_sink() or triple_candidate) else 5
  occurrences=[(i,tuple(sizes[i:i+plen])) for i in range(len(sizes)-plen+1)
    if sizes[i] < sizes[i+1] < sizes[i+2] < sizes[i+3] and
    (plen == 4 or 0 < sizes[i+4] < sizes[i+3])]
  candidates=Counter(pattern for _i,pattern in occurrences)
  if not candidates: raise RuntimeError(f"no stable decode pattern in histogram {Counter(sizes)}")
  # Compilation/warmup may repeat an older, larger graph pattern exactly as
  # often as the final steady pattern. Break count ties toward the latest
  # occurrence so the census follows the graph that actually reaches wall.
  repeated=[p for p,n in candidates.items() if n>=4]
  pattern=max(repeated or list(candidates),key=lambda p:max(i for i,q in occurrences if q==p))
  out=[]; i=0
  while i+len(pattern)<=len(lines):
    if tuple(sizes[i:i+len(pattern)])==pattern:
      out.append([e for row in lines[i:i+len(pattern)] for e in row.get("entries",())]); i+=len(pattern)
    else: i+=1
  if len(out)<4: raise RuntimeError(f"only {len(out)} complete replays for pattern {pattern}")
  return pattern,out


def _run_tokens(candidate:bool, depth:int, count:int, max_context:int, reps:int, triple:bool=False):
  # Phase A-C isolate projection fusion with the legacy cache-store chain.
  # The explicit composition phase leaves the promoted producer sink enabled.
  os.environ["DEV"]="NV"
  if candidate or triple: os.environ.pop("TINYGRAD_Q4K_KV_PAIR_DISABLE",None)
  else: os.environ["TINYGRAD_Q4K_KV_PAIR_DISABLE"]="1"
  if _compose_producer_sink(): os.environ.pop("TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE",None)
  else: os.environ["TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE"]="1"
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model=_load(MODEL,max_context); admitted=_install(model,candidate,triple)
  model._decode_direct_greedy_promoted=False; model._decode_feedback_pingpong_promoted=False
  gen=model.generate(_prompt(MODEL,depth),chunk_size=32,temperature=0.0)
  try: settled=_settled_continuous_windows(gen,Device["NV"],count,reps)
  finally: gen.close()
  return settled,admitted


def profile_child(candidate:bool, depth:int, count:int, max_context:int, reps:int,
                  profile_jsonl:pathlib.Path, out:pathlib.Path, triple:bool=False) -> dict:
  os.environ.update(PROFILE="1",HCQ_GRAPH_PROFILE_JSON=str(profile_jsonl))
  profile_jsonl.unlink(missing_ok=True); _install_graph_tracker()
  from tinygrad import Device
  settled,admitted=_run_tokens(candidate,depth,count,max_context,reps,triple)
  Device["NV"].synchronize(); _flush_final_timestamps(); Device["NV"].synchronize()
  lines=[json.loads(x) for x in profile_jsonl.read_text().splitlines() if x.strip()]
  pattern,replays=_pattern_and_replays(lines,triple and candidate)
  # Use the final bounded run. Earlier occurrences with the same graph-size
  # signature can be compile/warmup captures with incomparable timestamps.
  steady=replays[-min(9,len(replays)):]
  metrics=[_replay_metrics(x) for x in steady]
  ledger={key:round(statistics.median(float(x[key]) for x in metrics),3) for key in
          ("node_count","node_sum_us","union_us","overlap_us","span_us")}
  table=_per_name_table(steady)
  result={"schema":"tinygrad.nv_q4k_kv_pair_qualification.v1","mode":"profile-child",
    "arm":"candidate" if candidate else "control","triple":triple,"depth":depth,"count":count,"reps":reps,
    "max_context":max_context,"gpu_state":_gpu_state(),"admitted_blocks":admitted,"settled":settled,
    "group_pattern":pattern,"complete_replay_count":len(replays),"steady_replay_count":len(steady),
    "ledger":ledger,"per_name_table":table}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def timing_child(candidate:bool, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path, triple:bool=False) -> dict:
  settled,admitted=_run_tokens(candidate,depth,count,max_context,reps,triple)
  result={"schema":"tinygrad.nv_q4k_kv_pair_qualification.v1","mode":"timing-child",
    "arm":"candidate" if candidate else "control","triple":triple,"depth":depth,"count":count,"reps":reps,
    "max_context":max_context,"gpu_state":_gpu_state(),"admitted_blocks":admitted,**settled}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def _child(mode:str,candidate:bool,label:str,root:pathlib.Path,args) -> dict:
  out=root/f"{label}.json"; cmd=["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
    "--mode",f"{mode}-child","--depth",str(args.depth),"--count",str(args.count),"--max-context",str(args.max_context),
    "--reps",str(args.reps),"--out",str(out)]
  if candidate: cmd.append("--candidate")
  if args.triple: cmd.append("--triple")
  if mode=="profile": cmd += ["--profile-jsonl",str(root/f"{label}.profile.jsonl")]
  child_env={**os.environ,"PYTHONPATH":str(ROOT),"DEV":"NV"}
  if candidate or args.triple: child_env.pop("TINYGRAD_Q4K_KV_PAIR_DISABLE",None)
  else: child_env["TINYGRAD_Q4K_KV_PAIR_DISABLE"]="1"
  if _compose_producer_sink(): child_env.pop("TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE",None)
  else: child_env["TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE"]="1"
  run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=child_env)
  if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())


def profile_driver(args) -> dict:
  root=args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True,exist_ok=True)
  control=_child("profile",False,"control",root,args); candidate=_child("profile",True,"candidate",root,args)
  hashes={control["settled"]["token_stream_hash"],candidate["settled"]["token_stream_hash"]}
  pair_names={n:v for n,v in candidate["per_name_table"].items() if n.startswith("q4k_g3_lanemap_gemv_pair_vec_1024_4096")}
  result={"schema":"tinygrad.nv_q4k_kv_pair_qualification.v1","mode":"profile",
    "all_token_hashes_equal":len(hashes)==1,
    "ledger_delta":{k:round(float(candidate["ledger"][k])-float(control["ledger"][k]),3) for k in control["ledger"]},
    "structural":{"control_nodes":control["ledger"]["node_count"],"candidate_nodes":candidate["ledger"]["node_count"],
      "admitted_blocks":candidate["admitted_blocks"],"pair_names":pair_names,
      "control_pattern":control["group_pattern"],"candidate_pattern":candidate["group_pattern"]},
    "arms":[{k:v for k,v in row.items() if k!="per_name_table"} for row in (control,candidate)]}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def timing_driver(args) -> dict:
  root=args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True,exist_ok=True)
  arms=[_child("timing",False,"control_a",root,args),_child("timing",True,"candidate",root,args),
        _child("timing",False,"control_c",root,args)]
  midpoint=statistics.median((arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"]))
  candidate=arms[1]["median_ms_per_token"]; hashes={x["token_stream_hash"] for x in arms}
  result={"schema":"tinygrad.nv_q4k_kv_pair_qualification.v1","mode":"timing","depth":args.depth,
    "count":args.count,"reps":args.reps,"control_a_ms_per_token":arms[0]["median_ms_per_token"],
    "candidate_ms_per_token":candidate,"control_c_ms_per_token":arms[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token":midpoint,"recovery_us_per_token":(midpoint-candidate)*1000.0,
    "all_token_hashes_equal":len(hashes)==1,"token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate<min(arms[0]["median_ms_per_token"],arms[2]["median_ms_per_token"])
      else "NO_GO_WALL","arms":arms}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("profile","profile-child","timing","timing-child"),default="profile")
  ap.add_argument("--candidate",action="store_true"); ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--triple",action="store_true")
  ap.add_argument("--count",type=int,default=32); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--reps",type=int,default=3); ap.add_argument("--profile-jsonl",type=pathlib.Path)
  ap.add_argument("--out",type=pathlib.Path,required=True); args=ap.parse_args()
  if args.mode=="profile-child":
    if args.profile_jsonl is None: raise SystemExit("--profile-jsonl is required")
    result=profile_child(args.candidate,args.depth,args.count,args.max_context,args.reps,args.profile_jsonl,args.out,args.triple)
  elif args.mode=="timing-child": result=timing_child(args.candidate,args.depth,args.count,args.max_context,args.reps,args.out,args.triple)
  elif args.mode=="profile": result=profile_driver(args)
  else: result=timing_driver(args)
  print(json.dumps(result if "per_name_table" not in result else {k:v for k,v in result.items() if k!="per_name_table"},indent=2,sort_keys=True))
  return 0


if __name__=="__main__": raise SystemExit(main())
