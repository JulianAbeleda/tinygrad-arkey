#!/usr/bin/env python3
"""Production device-time closure for Q4_K FFN-down scalar versus vector loads."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, _complete_replays, _flush_final_timestamps, _gpu_state,
  _install_graph_tracker, _per_name_table, _replay_metrics)
from extra.llm_research.decode.nv_q4k_down_vector_load_wall_bracket import _set_load_style

SCALAR = "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"
VECTOR = "q4k_fp16_mmvq_direct_vec_4096_12288_epi_ffnresadd"


def run_child(arm:str, depth:int, count:int, max_context:int, reps:int,
              profile_jsonl:pathlib.Path, out:pathlib.Path) -> dict:
  os.environ["DEV"] = "NV"
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(profile_jsonl)
  profile_jsonl.unlink(missing_ok=True)
  _install_graph_tracker()

  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows

  dev = Device["NV"]
  model = _load(MODEL, max_context)
  indices = _set_load_style(model, arm == "candidate")
  model._decode_direct_greedy_promoted = False
  model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, dev, count, reps)
  finally: gen.close()
  dev.synchronize()
  _flush_final_timestamps()
  dev.synchronize()

  lines = [json.loads(line) for line in profile_jsonl.read_text().splitlines() if line.strip()]
  sizes = [len(x.get("entries", [])) for x in lines]
  replays = _complete_replays(lines)
  steady = replays[3:] if len(replays) > 3 else replays
  metrics = [_replay_metrics(x) for x in steady]
  ledger = {key: round(statistics.median(float(x[key]) for x in metrics), 3) for key in
            ("node_sum_us", "union_us", "overlap_us", "span_us")}
  result = {"schema":"tinygrad.nv_q4k_down_vector_load_profile.v1", "arm":arm,
    "depth":depth, "count":count, "reps":reps, "max_context":max_context,
    "gpu_state":_gpu_state(), "q4_down_blocks":indices, "settled":settled,
    "profile_jsonl":str(profile_jsonl), "group_size_histogram":dict(sorted(Counter(sizes).items())),
    "complete_replay_count":len(replays), "steady_replay_count":len(steady),
    "ledger":ledger, "per_name_table":_per_name_table(steady)}
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def child(arm:str, root:pathlib.Path, depth:int, count:int, max_context:int, reps:int) -> dict:
  out, profile = root/f"{arm}.json", root/f"{arm}.profile.jsonl"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--arm", arm, "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
    "--reps", str(reps), "--profile-jsonl", str(profile), "--out", str(out)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env={**os.environ, "PYTHONPATH":str(ROOT)})
  if run.returncode: raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def driver(depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  arms = [child("control_a", root, depth, count, max_context, reps),
          child("candidate", root, depth, count, max_context, reps),
          child("control_c", root, depth, count, max_context, reps)]

  def wall(x): return float(x["settled"]["median_ms_per_token"])*1000.0
  def replay_table(x):
    lines = [json.loads(line) for line in pathlib.Path(x["profile_jsonl"]).read_text().splitlines() if line.strip()]
    return _complete_replays(lines)[3:]
  ctrl_table = _per_name_table(replay_table(arms[0])+replay_table(arms[2]))
  cand_table = _per_name_table(replay_table(arms[1]))
  scalar_us = float(ctrl_table[SCALAR]["median_us"])
  vector_us = float(cand_table[VECTOR]["median_us"])
  control_wall = statistics.median((wall(arms[0]), wall(arms[2])))
  candidate_wall = wall(arms[1])
  control_ledger = {key:statistics.median(float(arms[i]["ledger"][key]) for i in (0,2)) for key in arms[0]["ledger"]}
  candidate_ledger = {key:float(arms[1]["ledger"][key]) for key in arms[1]["ledger"]}
  hashes = {x["settled"]["token_stream_hash"] for x in arms}
  result = {"schema":"tinygrad.nv_q4k_down_vector_load_profile.v1", "mode":"reverse-bracket-profile",
    "depth":depth, "count":count, "reps":reps, "all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "walls_us_per_token":{"control_a":wall(arms[0]), "candidate":candidate_wall,
      "control_c":wall(arms[2]), "control_midpoint":control_wall},
    "q4_down_device":{"calls_per_token":18, "scalar_us_per_token":scalar_us,
      "vector_us_per_token":vector_us, "delta_us_per_token":round(vector_us-scalar_us,3),
      "scalar_us_per_call":round(scalar_us/18,3), "vector_us_per_call":round(vector_us/18,3)},
    "ledger_control":control_ledger, "ledger_candidate":candidate_ledger,
    "closure":{"wall_delta_us":round(candidate_wall-control_wall,3),
      "node_sum_delta_us":round(candidate_ledger["node_sum_us"]-control_ledger["node_sum_us"],3),
      "union_delta_us":round(candidate_ledger["union_us"]-control_ledger["union_us"],3),
      "overlap_delta_us":round(candidate_ledger["overlap_us"]-control_ledger["overlap_us"],3)},
    "arms":[{k:v for k,v in x.items() if k != "per_name_table"} for x in arms]}
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--arm", choices=("control_a","candidate","control_c"))
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=32)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=6)
  ap.add_argument("--profile-jsonl",type=pathlib.Path); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  if args.arm:
    if args.profile_jsonl is None: raise SystemExit("--profile-jsonl is required for child arms")
    result=run_child(args.arm,args.depth,args.count,args.max_context,args.reps,args.profile_jsonl,args.out)
  else: result=driver(args.depth,args.count,args.max_context,args.reps,args.out)
  print(json.dumps(result if not args.arm else {k:v for k,v in result.items() if k != "per_name_table"},indent=2,sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
