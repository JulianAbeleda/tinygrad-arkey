#!/usr/bin/env python3
"""Profile control versus semantic REDUCE_OUTPUT Q/K norm+RoPE fusion."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, _complete_replays, _gpu_state, _per_name_table, _replay_metrics,
  _install_graph_tracker, _flush_final_timestamps)


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int, profile_jsonl:pathlib.Path, out:pathlib.Path) -> dict:
  os.environ.update(DEV="NV", PROFILE="1", HCQ_GRAPH_PROFILE_JSON=str(profile_jsonl))
  profile_jsonl.unlink(missing_ok=True)
  _install_graph_tracker()
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _set_admission
  dev = Device["NV"]
  model = _load(MODEL, max_context)
  if arm == "production":
    admitted = [idx for idx, block in enumerate(model.blk)
                if getattr(block, "_decode_qk_norm_rope_promoted", False)]
    if not getattr(model, "_decode_qk_norm_rope_promoted", False) or len(admitted) != len(model.blk):
      raise RuntimeError(f"production Q/K norm+RoPE policy did not promote every block: {admitted}")
  else:
    admitted = _set_admission(model, arm == "candidate")
  model._decode_direct_greedy_promoted = False
  model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, dev, count, reps)
  finally: gen.close()
  dev.synchronize(); _flush_final_timestamps(); dev.synchronize()
  lines = [json.loads(x) for x in profile_jsonl.read_text().splitlines() if x.strip()]
  sizes = [len(x.get("entries", ())) for x in lines]
  replays = _complete_replays(lines)
  steady = replays[3:] if len(replays) > 3 else replays
  metrics = [_replay_metrics(x) for x in steady]
  ledger = {k: round(statistics.median([m[k] for m in metrics]), 3) if metrics else None
            for k in ("node_count", "node_sum_us", "union_us", "overlap_us", "span_us")}
  result = {"schema":"tinygrad.nv_qk_norm_rope_semantic_profile.v1", "arm":arm, "depth":depth,
            "count":count, "reps":reps, "max_context":max_context, "gpu_state":_gpu_state(),
            "route_source":"production-policy" if arm == "production" else "research-override",
            "admitted_blocks":admitted if arm in ("candidate", "production") else [], "settled":settled,
            "group_size_histogram":dict(sorted(Counter(sizes).items())), "complete_replay_count":len(replays),
            "steady_replay_count":len(steady), "ledger":ledger, "per_name_table":_per_name_table(steady)}
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def child(arm:str, root:pathlib.Path, args) -> dict:
  out, profile = root/f"{arm}.json", root/f"{arm}.profile.jsonl"
  cmd = ["timeout","1800","flock","-w","600",LOCK,str(PYTHON),str(pathlib.Path(__file__).resolve()),
         "--arm",arm,"--depth",str(args.depth),"--count",str(args.count),"--max-context",str(args.max_context),
         "--reps",str(args.reps),"--profile-jsonl",str(profile),"--out",str(out)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ,"PYTHONPATH":str(ROOT)})
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm",choices=("driver","control","candidate","production"),default="driver")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=8)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=2)
  ap.add_argument("--profile-jsonl",type=pathlib.Path); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.arm != "driver":
    print(json.dumps(run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.profile_jsonl,args.out),indent=2)); return 0
  root=args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True,exist_ok=True)
  rows=[child("control",root,args),child("candidate",root,args)]
  result={"schema":"tinygrad.nv_qk_norm_rope_semantic_profile.v1","arms":rows}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
