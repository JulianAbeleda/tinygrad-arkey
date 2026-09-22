#!/usr/bin/env python3
"""Reverse-bracket the closed llama-style S6/wide-KV flash research lease."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, _complete_replays, _flush_final_timestamps, _gpu_state, _install_graph_tracker,
  _per_name_table, _replay_metrics,
)


def _install(model, candidate:bool, max_context:int) -> None:
  if candidate and max_context % 128: raise ValueError("wide flash qualification requires max-context divisible by 128")
  # Keep the control explicitly on the scalar route after the wide route is installed by policy.
  geometry = {"split_count":max_context//128, "llama_vec_wide":True} if candidate else {"llama_vec_wide":False}
  setattr(model, "_flash_decode_tile_geometry_lease", geometry)
  for block in model.blk: setattr(block, "_flash_decode_tile_geometry_lease", geometry)


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int,
              profile_jsonl:pathlib.Path, out:pathlib.Path) -> dict:
  os.environ.update(DEV="NV", PROFILE="1", HCQ_GRAPH_PROFILE_JSON=str(profile_jsonl))
  profile_jsonl.unlink(missing_ok=True)
  _install_graph_tracker()
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

  dev = Device["NV"]
  model = _load(MODEL, max_context)
  _install(model, arm == "candidate", max_context)
  model._decode_direct_greedy_promoted = False
  model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, dev, count, reps)
  finally: gen.close()
  dev.synchronize(); _flush_final_timestamps(); dev.synchronize()

  lines = [json.loads(line) for line in profile_jsonl.read_text().splitlines() if line.strip()]
  replays = _complete_replays(lines)
  steady = replays[3:]
  if not steady: raise RuntimeError(f"no steady replays in {profile_jsonl}")
  metrics = [_replay_metrics(x) for x in steady]
  ledger = {key:round(statistics.median(float(x[key]) for x in metrics), 3) for key in
            ("node_sum_us", "union_us", "overlap_us", "span_us")}
  table = _per_name_table(steady)
  result = {"schema":"tinygrad.nv_flash_llama_vec_wide_qualification.v1", "arm":arm,
    "depth":depth, "count":count, "reps":reps, "max_context":max_context, "gpu_state":_gpu_state(),
    "settled":settled, "ledger":ledger, "complete_replay_count":len(replays),
    "steady_replay_count":len(steady), "per_name_table":table, "profile_jsonl":str(profile_jsonl)}
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def _child(arm:str, root:pathlib.Path, depth:int, count:int, max_context:int, reps:int) -> dict:
  out, profile = root/f"{arm}.json", root/f"{arm}.profile.jsonl"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--arm", arm, "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
    "--reps", str(reps), "--profile-jsonl", str(profile), "--out", str(out)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH":str(ROOT)})
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-6000:]}")
  return json.loads(out.read_text())


def driver(depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  arms = [_child("control_a", root, depth, count, max_context, reps),
          _child("candidate", root, depth, count, max_context, reps),
          _child("control_c", root, depth, count, max_context, reps)]
  def wall(row): return float(row["settled"]["median_ms_per_token"])*1000.0
  control_wall = statistics.median((wall(arms[0]), wall(arms[2])))
  candidate_wall = wall(arms[1])
  hashes = {x["settled"]["token_stream_hash"] for x in arms}
  control_ledger = {key:statistics.median(float(arms[i]["ledger"][key]) for i in (0, 2)) for key in arms[0]["ledger"]}
  candidate_ledger = {key:float(arms[1]["ledger"][key]) for key in arms[1]["ledger"]}
  result = {"schema":"tinygrad.nv_flash_llama_vec_wide_qualification.v1", "mode":"reverse-bracket-profile",
    "depth":depth, "count":count, "reps":reps, "max_context":max_context,
    "all_token_hashes_equal":len(hashes)==1, "token_stream_hashes":sorted(hashes),
    "walls_us_per_token":{"control_a":wall(arms[0]), "candidate":candidate_wall,
      "control_c":wall(arms[2]), "control_midpoint":control_wall,
      "candidate_delta":candidate_wall-control_wall},
    "ledger_control":control_ledger, "ledger_candidate":candidate_ledger,
    "ledger_delta":{key:candidate_ledger[key]-control_ledger[key] for key in control_ledger},
    "candidate_flash_rows":{name:row for name,row in arms[1]["per_name_table"].items()
                            if name.startswith("flash_vec_llama") or name.startswith("flash_fused_gmax")},
    "arms":[{k:v for k,v in row.items() if k != "per_name_table"} for row in arms]}
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm", choices=("control_a","candidate","control_c"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=32)
  ap.add_argument("--max-context",type=int,default=768); ap.add_argument("--reps",type=int,default=7)
  ap.add_argument("--profile-jsonl",type=pathlib.Path); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.arm:
    if args.profile_jsonl is None: raise SystemExit("--profile-jsonl required")
    result=run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.profile_jsonl,args.out)
  else: result=driver(args.depth,args.count,args.max_context,args.reps,args.out)
  print(json.dumps(result if not args.arm else {k:v for k,v in result.items() if k != "per_name_table"},indent=2,sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
