#!/usr/bin/env python3
"""Reverse-bracket wall for the Q4_K attention-K four-warp exact route.

Control and candidate differ only by the Q4 K admission: control runs the
installed ``q4k_g3_lanemap_gemv_1024_4096`` one-warp-per-row route, candidate
runs the exact group-factorized four-warp direct consumer on the same Q4 K
blocks.  Both arms keep every other promoted route, so the delta isolates the
Q4 K change.  Each arm is a fresh model load under its own
``flock -w 600 /tmp/gpu-bench.lock``.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LOCK = "/tmp/gpu-bench.lock"


def _set_admission(model, enabled: bool) -> list[int]:
  from tinygrad.llm.q4k_k_four_warp import Q4KKFourWarpAdmission
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  out = []
  for idx, block in enumerate(model.blk):
    k = getattr(block, "attn_k", None)
    if not isinstance(k, Q4KPrimitiveLinear) or getattr(k, "route_role", "") != "attn_kv":
      continue
    if (getattr(k, "out_features", None), getattr(k, "in_features", None)) != (1024, 4096):
      continue
    if enabled:
      k._q4k_k_four_warp_admission = Q4KKFourWarpAdmission(idx)
    elif hasattr(k, "_q4k_k_four_warp_admission"):
      delattr(k, "_q4k_k_four_warp_admission")
    out.append(idx)
  return out


def timing_child(depth: int, count: int, max_context: int, reps: int, enabled: bool, composed: bool,
                 out: pathlib.Path) -> dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model = _load(MODEL, max_context)
  admitted = _set_admission(model, enabled)
  model._decode_direct_greedy_promoted = composed
  model._decode_feedback_pingpong_promoted = composed
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try:
    settled = _settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally:
    gen.close()
  result = {"schema": "tinygrad.q4k_k_four_warp_wall_bracket.v1",
    "enabled": enabled, "composed": composed, "q4_k_blocks": admitted,
    "depth": depth, "count": count, "reps": reps,
    "included_cost": True, "settled_continuous": True, **settled}
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def bracket(depth: int, count: int, max_context: int, reps: int, composed: bool, out: pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json"))
  root.mkdir(parents=True, exist_ok=True)

  def child(enabled: bool, label: str):
    o = root / f"{label}.json"
    cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, sys.executable,
      str(pathlib.Path(__file__).resolve()), "--mode", "timing-child",
      "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
      "--reps", str(reps), "--out", str(o)]
    if enabled:
      cmd.append("--enabled")
    if composed:
      cmd.append("--composed")
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      env={**os.environ, "PYTHONPATH": "/home/ubuntu/tinygrad-arkey"})
    if run.returncode:
      raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    return json.loads(o.read_text())

  rows = [child(False, "control_a"), child(True, "candidate"), child(False, "control_c")]
  control_mid = statistics.median((rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"]))
  candidate = rows[1]["median_ms_per_token"]
  hashes = {r["token_stream_hash"] for r in rows}
  result = {"schema": "tinygrad.q4k_k_four_warp_wall_bracket.v1", "mode": "reverse-bracket",
    "depth": depth, "count": count, "reps": reps, "composed": composed,
    "control_a_ms_per_token": rows[0]["median_ms_per_token"],
    "candidate_ms_per_token": candidate,
    "control_c_ms_per_token": rows[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token": control_mid,
    "candidate_minus_control_ms_per_token": candidate - control_mid,
    "candidate_speedup_pct": (control_mid / candidate - 1) * 100,
    "all_token_hashes_equal": len(hashes) == 1,
    "token_stream_hash": sorted(hashes)[0] if len(hashes) == 1 else sorted(hashes),
    "verdict": "WALL_PASS" if len(hashes) == 1 and candidate < control_mid else "NO_GO_WALL",
    "arms": rows}
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("timing-child", "timing"), default="timing")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--enabled", action="store_true")
  ap.add_argument("--composed", action="store_true")
  ap.add_argument("--out", default="/tmp/q4k_k_four_warp_wall_bracket.json")
  args = ap.parse_args()
  if args.mode == "timing-child":
    row = timing_child(args.depth, args.count, args.max_context, args.reps, args.enabled, args.composed,
      pathlib.Path(args.out))
    print(json.dumps(row, indent=2, sort_keys=True))
    return 0
  result = bracket(args.depth, args.count, args.max_context, args.reps, args.composed, pathlib.Path(args.out))
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
