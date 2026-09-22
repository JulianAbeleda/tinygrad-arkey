#!/usr/bin/env python3
"""Profile/census and reverse-wall qualification for the producer K/V cache sink."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, DECODE_GROUP_PREFIX, _flush_final_timestamps, _gpu_state, _install_graph_tracker,
  _per_name_table, _replay_metrics)


def _install(model, candidate:bool) -> list[int]:
  if not candidate: return []
  from tinygrad.llm.producer_kv_cache_sink import ProducerKVCacheSinkAdmission
  existing = [index for index, block in enumerate(model.blk)
              if isinstance(getattr(block, "_producer_kv_cache_sink_admission", None), ProducerKVCacheSinkAdmission)]
  if len(existing) == len(model.blk): return existing
  for index, block in enumerate(model.blk):
    block._producer_kv_cache_sink_admission = ProducerKVCacheSinkAdmission(index)
  return list(range(len(model.blk)))


def _complete_replays(lines:list[dict], candidate:bool) -> list[list[dict]]:
  """Parse the exact graph sequence for this topology.

  Control owns the ordinary 36-node cache-store graph after the common
  32/64/128/256 prefix. Candidate deletes that entire graph, so reusing the
  generic tail-inference parser mistakes the next token's 32-node graph for a
  tail and reports 512 nodes. Requiring the next row to begin a new prefix also
  excludes prefill compile groups whose prefix is followed by a 394-node tail.
  """
  pattern = tuple(DECODE_GROUP_PREFIX) + (() if candidate else (36,))
  sizes = [len(x.get("entries", ())) for x in lines]
  out, i = [], 0
  while i + len(pattern) <= len(lines):
    end = i + len(pattern)
    if tuple(sizes[i:end]) == pattern and (end == len(lines) or sizes[end] == DECODE_GROUP_PREFIX[0]):
      out.append([e for row in lines[i:end] for e in row.get("entries", ())]); i = end
    else: i += 1
  return out


def _run_tokens(candidate:bool, depth:int, count:int, max_context:int, reps:int):
  if candidate: os.environ.pop("TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE", None)
  else: os.environ["TINYGRAD_PRODUCER_KV_CACHE_SINK_DISABLE"] = "1"
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model = _load(MODEL, max_context); admitted = _install(model, candidate)
  model._decode_direct_greedy_promoted = False; model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, Device["NV"], count, reps)
  finally: gen.close()
  return settled, admitted


def profile_child(candidate:bool, depth:int, count:int, max_context:int, reps:int,
                  profile_jsonl:pathlib.Path, out:pathlib.Path) -> dict:
  os.environ.update(DEV="NV", PROFILE="1", HCQ_GRAPH_PROFILE_JSON=str(profile_jsonl))
  profile_jsonl.unlink(missing_ok=True); _install_graph_tracker()
  from tinygrad import Device
  settled, admitted = _run_tokens(candidate, depth, count, max_context, reps)
  Device["NV"].synchronize(); _flush_final_timestamps(); Device["NV"].synchronize()
  lines = [json.loads(x) for x in profile_jsonl.read_text().splitlines() if x.strip()]
  sizes = [len(x.get("entries", ())) for x in lines]
  replays = _complete_replays(lines, candidate); steady = replays[3:] if len(replays) > 3 else replays
  metrics = [_replay_metrics(x) for x in steady]
  ledger = {key:round(statistics.median(float(x[key]) for x in metrics), 3) for key in
            ("node_count", "node_sum_us", "union_us", "overlap_us", "span_us")}
  table = _per_name_table(steady)
  producer_names = sorted(name for name in table if name.startswith("reduce_output_rmsnorm_rope_kv_cache_8_128"))
  generic_store_names = sorted(name for name in table if name.startswith("E_8_8_16_2"))
  result = {"schema":"tinygrad.nv_producer_kv_cache_sink_qualification.v1", "mode":"profile-child",
    "arm":"candidate" if candidate else "control", "depth":depth, "count":count, "reps":reps,
    "max_context":max_context, "gpu_state":_gpu_state(), "admitted_blocks":admitted, "settled":settled,
    "group_size_histogram":dict(sorted(Counter(sizes).items())), "complete_replay_count":len(replays),
    "steady_replay_count":len(steady), "ledger":ledger, "producer_names":producer_names,
    "generic_store_names":generic_store_names, "per_name_table":table}
  out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def timing_child(candidate:bool, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  os.environ["DEV"] = "NV"
  settled, admitted = _run_tokens(candidate, depth, count, max_context, reps)
  result = {"schema":"tinygrad.nv_producer_kv_cache_sink_qualification.v1", "mode":"timing-child",
    "arm":"candidate" if candidate else "control", "depth":depth, "count":count, "reps":reps,
    "max_context":max_context, "gpu_state":_gpu_state(), "admitted_blocks":admitted, **settled}
  out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def _child(mode:str, candidate:bool, label:str, root:pathlib.Path, args) -> dict:
  out = root/f"{label}.json"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON), str(pathlib.Path(__file__).resolve()),
    "--mode", f"{mode}-child", "--depth", str(args.depth), "--count", str(args.count),
    "--max-context", str(args.max_context), "--reps", str(args.reps), "--out", str(out)]
  if candidate: cmd.append("--candidate")
  if mode == "profile": cmd += ["--profile-jsonl", str(root/f"{label}.profile.jsonl")]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH":str(ROOT), "DEV":"NV"})
  if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())


def profile_driver(args) -> dict:
  root = args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True, exist_ok=True)
  control, candidate = _child("profile", False, "control", root, args), _child("profile", True, "candidate", root, args)
  hashes = {control["settled"]["token_stream_hash"], candidate["settled"]["token_stream_hash"]}
  result = {"schema":"tinygrad.nv_producer_kv_cache_sink_qualification.v1", "mode":"profile",
    "all_token_hashes_equal":len(hashes) == 1,
    "ledger_delta":{key:round(float(candidate["ledger"][key])-float(control["ledger"][key]), 3)
                    for key in control["ledger"]},
    "structural": {"control_nodes":control["ledger"]["node_count"], "candidate_nodes":candidate["ledger"]["node_count"],
      "candidate_producer_names":candidate["producer_names"], "candidate_generic_store_names":candidate["generic_store_names"]},
    "arms":[{k:v for k,v in row.items() if k != "per_name_table"} for row in (control, candidate)]}
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); return result


def timing_driver(args) -> dict:
  root = args.out.parent/(args.out.stem+"_arms"); root.mkdir(parents=True, exist_ok=True)
  arms = [_child("timing", False, "control_a", root, args), _child("timing", True, "candidate", root, args),
          _child("timing", False, "control_c", root, args)]
  midpoint = statistics.median((arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"]))
  candidate = arms[1]["median_ms_per_token"]; hashes = {x["token_stream_hash"] for x in arms}
  result = {"schema":"tinygrad.nv_producer_kv_cache_sink_qualification.v1", "mode":"timing",
    "depth":args.depth, "count":args.count, "reps":args.reps,
    "control_a_ms_per_token":arms[0]["median_ms_per_token"], "candidate_ms_per_token":candidate,
    "control_c_ms_per_token":arms[2]["median_ms_per_token"], "control_midpoint_ms_per_token":midpoint,
    "recovery_us_per_token":(midpoint-candidate)*1000.0, "all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate < min(arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"])
      else "NO_GO_WALL", "arms":arms}
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); return result


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("profile", "profile-child", "timing", "timing-child"), default="profile")
  ap.add_argument("--candidate", action="store_true"); ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32); ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=3); ap.add_argument("--profile-jsonl", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path, required=True); args = ap.parse_args()
  if args.mode == "profile-child":
    if args.profile_jsonl is None: raise SystemExit("--profile-jsonl is required")
    result = profile_child(args.candidate, args.depth, args.count, args.max_context, args.reps, args.profile_jsonl, args.out)
  elif args.mode == "timing-child": result = timing_child(args.candidate, args.depth, args.count, args.max_context, args.reps, args.out)
  elif args.mode == "profile": result = profile_driver(args)
  else: result = timing_driver(args)
  print(json.dumps(result if "per_name_table" not in result else {k:v for k,v in result.items() if k != "per_name_table"}, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
