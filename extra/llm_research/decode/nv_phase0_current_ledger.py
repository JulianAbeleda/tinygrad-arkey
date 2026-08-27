#!/usr/bin/env python3
"""Rebuild the current composed production decode ledger.

This harness profiles the current no-rollback route population, verifies the
promoted admissions at model load, and closes:

    wall        = device_union + host_gap
    device_union = node_sum - overlap

in the same PROFILE=1 domain, and emits a disjoint per-name current census.

The ``useful_body = command_interval - dependency_spin`` identity requires the
separate ``%globaltimer`` wait-exit machinery and is produced in Phase 1.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, statistics, subprocess, sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  MODEL, LOCK, PYTHON, canon, _replay_metrics, _per_name_table,
  _gpu_state, _install_graph_tracker, _flush_final_timestamps)
from extra.llm_research.decode.nv_q6k_down_packed_lanemap_profile import _current_decode_replays

SCHEMA = "tinygrad.nv_phase0_current_ledger.v1"


def _verify_production_routes(model) -> dict:
  from tinygrad.llm.qk_primitives import Q6KPrimitiveLinear
  qk = [idx for idx, block in enumerate(model.blk) if getattr(block, "_decode_qk_norm_rope_promoted", False)]
  q6 = [idx for idx, block in enumerate(model.blk)
        if isinstance((ffn := getattr(block, "ffn_down", None)), Q6KPrimitiveLinear)
        and getattr(ffn, "route_role", "") == "ffn_down"
        and getattr(ffn, "out_features", None) == 4096 and getattr(ffn, "in_features", None) == 12288
        and getattr(ffn, "_q6k_ffn_down_mmvq_admission", None) is not None]
  if not getattr(model, "_decode_qk_norm_rope_promoted", False) or len(qk) != len(model.blk):
    raise RuntimeError(f"production Q/K norm+RoPE policy not installed on every block: {qk}")
  if not getattr(model, "_decode_q6k_ffn_down_fp16_geometry_promoted", False) or not q6:
    raise RuntimeError(f"production Q6 FFN-down policy not installed: {q6}")
  native_argmax_threads = int(getattr(model, "_decode_native_argmax_threads", 0))
  if native_argmax_threads != 1024:
    raise RuntimeError(f"production native argmax policy not installed: threads={native_argmax_threads}")
  q6_packed = [idx for idx in q6 if getattr(getattr(model.blk[idx].ffn_down,
    "_q6k_ffn_down_mmvq_admission", None), "packed_lanemap", False)]
  gateup = [idx for idx, block in enumerate(model.blk)
    if getattr(getattr(block, "ffn_gate", None), "_q4k_gate_up_four_warp_admission", None) is not None]
  shared = [getattr(block, "_shared_q8_attention_admission", None) for block in model.blk]
  q4q4_shared = [x.block_index for x in shared if x is not None and x.q4_kv_pair_output]
  q4q6_shared = [x.block_index for x in shared if x is not None and x.q4_q6_kv_pair_output]
  ordinary_pairs = [idx for idx, block in enumerate(model.blk)
    if getattr(block, "_q4k_kv_pair_admission", None) is not None]
  if len(q6_packed) != 18 or len(gateup) != 36 or len(q4q4_shared) != 10 or len(q4q6_shared) != 8 or len(ordinary_pairs) != 8:
    raise RuntimeError(f"current route census mismatch q6_packed={q6_packed} gateup={gateup} "
      f"shared_q4q4={q4q4_shared} shared_q4q6={q4q6_shared} ordinary={ordinary_pairs}")
  if not getattr(model, "_decode_producer_kv_cache_sink_promoted", False):
    raise RuntimeError("production K/V cache sink policy not installed")
  return {"qk_norm_rope_blocks": qk, "q6_ffn_down_blocks": q6,
          "q6_ffn_down_packed_blocks": q6_packed, "gateup_fourwarp_blocks": gateup,
          "shared_q4q4_pair_blocks": q4q4_shared, "shared_q4q6_pair_blocks": q4q6_shared,
          "ordinary_q4q4_pair_blocks": ordinary_pairs, "producer_kv_cache_sink": True,
          "native_argmax_threads": native_argmax_threads}


def run_child(depth: int, count: int, max_context: int, reps: int,
              profile_jsonl: pathlib.Path, out: pathlib.Path) -> dict:
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
  routes = _verify_production_routes(model)
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = True
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try:
    settled = _settled_continuous_windows(gen, dev, count, reps)
  finally:
    gen.close()
  dev.synchronize()
  _flush_final_timestamps()
  dev.synchronize()

  result = {
    "schema": SCHEMA, "arm": "production", "depth": depth, "count": count, "reps": reps,
    "max_context": max_context, "gpu_state": _gpu_state(), "routes": routes, "settled": settled,
    "profile_jsonl": str(profile_jsonl),
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def analyze(profile_jsonl: pathlib.Path, settled: dict, warmup: int) -> dict:
  lines = [json.loads(line) for line in profile_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
  sizes = [len(x.get("entries", [])) for x in lines]
  replays = _current_decode_replays(lines)
  steady = replays[warmup:] if len(replays) > warmup else replays
  # Compute raw medians first; round only at the end so the node_sum/union/
  # overlap identity closes to exactly zero rather than accumulating rounding.
  raw = {
    "node_sum_us": [sum(float(e["duration"]) for e in r) for r in steady],
    "union_us": [_replay_metrics(r)["union_us"] for r in steady],
    "overlap_us": [_replay_metrics(r)["overlap_us"] for r in steady],
    "span_us": [_replay_metrics(r)["span_us"] for r in steady],
  }
  node_sum = round(statistics.median(raw["node_sum_us"]), 3)
  union = round(statistics.median(raw["union_us"]), 3)
  overlap = round(node_sum - union, 3)
  span = round(statistics.median(raw["span_us"]), 3)
  wall_us = round(float(settled["median_ms_per_token"]) * 1000.0, 3)
  host_gap_us = round(wall_us - union, 3)

  table = _per_name_table(steady)
  # Order rows by median wall contribution (count * median_us per replay).
  rows = [{"name": name, "per_replay_count": int(v["per_replay_count"]),
           "replay_samples": int(v["replay_samples"]),
           "steady_replay_fraction": round(v["replay_samples"] / len(steady), 6),
           "median_us_per_replay": v["median_us"],
           "mean_us_per_replay": v["mean_us"],
           "median_us_per_call": round(v["median_us"] / v["per_replay_count"], 6),
           "wall_us_per_replay": round(v["median_us"], 3)}
          for name, v in table.items()]
  rows.sort(key=lambda r: -r["wall_us_per_replay"])

  closure = {
    "wall_us_per_token": wall_us,
    "node_sum_us": node_sum,
    "union_us": union,
    "overlap_us": overlap,
    "span_us": span,
    "host_gap_us": host_gap_us,
    "identity_union_node_sum_overlap_residual_us": round((node_sum - overlap) - union, 6),
    "identity_wall_union_host_gap_residual_us": round(wall_us - union - host_gap_us, 6),
    "target_240_us": 4166.666667,
    "remaining_to_240_us": round(wall_us - 4166.666667, 3),
    "tok_per_s": round(1e6 / wall_us, 3) if wall_us else None,
  }

  return {
    "schema": SCHEMA, "mode": "analyze", "warmup": warmup,
    "group_size_histogram": dict(sorted(Counter(sizes).items())),
    "complete_replay_count": len(replays), "steady_replay_count": len(steady),
    "closure": closure, "rows": rows,
    "token_stream_hash": settled.get("token_stream_hash"),
    "samples_ms_per_token": settled.get("samples_ms_per_token"),
  }


def sha256_file(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("driver", "child", "analyze"), default="driver")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=8)
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--profile-jsonl", type=pathlib.Path)
  ap.add_argument("--child-json", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  if args.mode == "child":
    if args.profile_jsonl is None:
      raise SystemExit("--profile-jsonl is required for child mode")
    row = run_child(args.depth, args.count, args.max_context, args.reps, args.profile_jsonl, args.out)
    print(json.dumps({k: v for k, v in row.items() if k != "settled"}, indent=2, sort_keys=True))
    return 0

  if args.mode == "analyze":
    if args.profile_jsonl is None or args.child_json is None:
      raise SystemExit("--profile-jsonl and --child-json are required for analyze mode")
    child = json.loads(args.child_json.read_text())
    result = analyze(args.profile_jsonl, child["settled"], args.warmup)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["closure"], indent=2))
    return 0

  # driver
  root = pathlib.Path(str(args.out).removesuffix(".json"))
  root.mkdir(parents=True, exist_ok=True)
  profile = root / "production.profile.jsonl"
  child_json = root / "production.child.json"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON),
         str(pathlib.Path(__file__).resolve()), "--mode", "child",
         "--depth", str(args.depth), "--count", str(args.count),
         "--max-context", str(args.max_context), "--reps", str(args.reps),
         "--profile-jsonl", str(profile), "--out", str(child_json)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
  if run.returncode:
    raise RuntimeError(f"production child failed rc={run.returncode}: {run.stderr[-4000:]}")

  child = json.loads(child_json.read_text())
  result = analyze(profile, child["settled"], args.warmup)
  result["gpu_state"] = child.get("gpu_state")
  result["routes"] = child.get("routes")
  result["commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                             text=True).strip()
  result["sha256"] = {
    "profile_jsonl": sha256_file(profile),
    "child_json": sha256_file(child_json),
  }
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  manifest = root / "sha256.txt"
  manifest.write_text("\n".join(
    f"{sha256_file(root / name)}  {name}" for name in sorted(p.name for p in root.iterdir())) + "\n")
  print(json.dumps(result["closure"], indent=2))
  print(json.dumps(result["rows"][:30], indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
