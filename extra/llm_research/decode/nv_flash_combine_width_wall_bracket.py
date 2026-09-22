#!/usr/bin/env python3
"""Reverse production wall bracket for an independent flash-combine lane width."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL, LOCK, _gpu_state
from extra.llm_research.decode.nv_flash_combine_width_profile import _install_combine_width


def timing_child(depth:int, count:int, max_context:int, reps:int, width:int|None, out:pathlib.Path) -> dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model = _load(MODEL, max_context); _install_combine_width(model, width)
  model._decode_direct_greedy_promoted = False; model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally: gen.close()
  result = {"schema":"tinygrad.nv_flash_combine_width_wall_bracket.v1", "width":width,
    "gpu_state":_gpu_state(), "depth":depth, "count":count, "reps":reps, "settled_continuous":True, **settled}
  out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def bracket(width:int, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  def child(candidate:bool, label:str) -> dict:
    arm_out = root/f"{label}.json"
    cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, sys.executable, str(pathlib.Path(__file__).resolve()),
      "--mode", "timing-child", "--width", str(width), "--depth", str(depth), "--count", str(count),
      "--max-context", str(max_context), "--reps", str(reps), "--out", str(arm_out)]
    if candidate: cmd.append("--candidate")
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env={**os.environ, "PYTHONPATH":str(ROOT), "DEV":"NV"})
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    return json.loads(arm_out.read_text())
  arms = [child(False, "control_a"), child(True, "candidate"), child(False, "control_c")]
  midpoint = statistics.median((arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"]))
  candidate = arms[1]["median_ms_per_token"]; hashes = {x["token_stream_hash"] for x in arms}
  result = {"schema":"tinygrad.nv_flash_combine_width_wall_bracket.v1", "mode":"reverse-bracket", "width":width,
    "depth":depth, "count":count, "reps":reps, "control_a_ms_per_token":arms[0]["median_ms_per_token"],
    "candidate_ms_per_token":candidate, "control_c_ms_per_token":arms[2]["median_ms_per_token"],
    "control_midpoint_ms_per_token":midpoint, "recovery_us_per_token":(midpoint-candidate)*1000.0,
    "candidate_speedup_pct":(midpoint/candidate-1)*100.0, "all_token_hashes_equal":len(hashes)==1,
    "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
    "verdict":"WALL_PASS" if len(hashes)==1 and candidate < min(arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"])
      else "NO_GO_WALL", "arms":arms}
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n"); return result


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=("timing", "timing-child"), default="timing")
  ap.add_argument("--width", type=int, choices=(64, 128), default=64); ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32); ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=9); ap.add_argument("--candidate", action="store_true")
  ap.add_argument("--out", type=pathlib.Path, required=True); args = ap.parse_args()
  result = timing_child(args.depth, args.count, args.max_context, args.reps, args.width if args.candidate else None, args.out) \
    if args.mode == "timing-child" else bracket(args.width, args.depth, args.count, args.max_context, args.reps, args.out)
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
