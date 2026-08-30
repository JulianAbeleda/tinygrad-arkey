#!/usr/bin/env python3
"""Reverse bracket for the composed booked NV decode routes.

Controls explicitly disable both the Q6_K FFN-down four-warp admission and
the semantic Q/K REDUCE_OUTPUT RMSNorm+RoPE epilogue.  The candidate leaves
the production policy untouched and fails unless both routes are installed.
Every arm is a fresh process under the shared GPU benchmark lock.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL, LOCK, _gpu_state


def _q6_active_blocks(model) -> list[int]:
  from tinygrad.llm.qk_primitives import Q6KPrimitiveLinear
  return [idx for idx, block in enumerate(model.blk)
          if isinstance((ffn := getattr(block, "ffn_down", None)), Q6KPrimitiveLinear)
          and getattr(ffn, "route_role", "") == "ffn_down"
          and getattr(ffn, "out_features", None) == 4096 and getattr(ffn, "in_features", None) == 12288
          and getattr(ffn, "_q6k_ffn_down_mmvq_admission", None) is not None]


def _qk_active_blocks(model) -> list[int]:
  return [idx for idx, block in enumerate(model.blk)
          if getattr(block, "_decode_qk_norm_rope_promoted", False)]


def _configure(model, enabled:bool) -> tuple[list[int], list[int]]:
  if enabled:
    qk, q6 = _qk_active_blocks(model), _q6_active_blocks(model)
    if not getattr(model, "_decode_qk_norm_rope_promoted", False) or len(qk) != len(model.blk):
      raise RuntimeError(f"production Q/K norm+RoPE policy is not installed on every block: {qk}")
    if not getattr(model, "_decode_q6k_ffn_down_fp16_geometry_promoted", False) or not q6:
      raise RuntimeError(f"production Q6 FFN-down policy is not installed: {q6}")
    return qk, q6
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _set_admission as set_qk
  from extra.llm_research.decode.q6k_ffn_down_four_warp_wall_bracket import _set_admission as set_q6
  set_qk(model, False)
  set_q6(model, False)
  qk, q6 = _qk_active_blocks(model), _q6_active_blocks(model)
  if qk or q6: raise RuntimeError(f"composed control did not close both routes: qk={qk} q6={q6}")
  return qk, q6


def timing_child(depth:int, count:int, max_context:int, reps:int, enabled:bool, out:pathlib.Path) -> dict:
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model = _load(MODEL, max_context)
  qk, q6 = _configure(model, enabled)
  model._decode_direct_greedy_promoted = False
  model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally: gen.close()
  result = {"schema":"tinygrad.nv_booked_composition_wall_bracket.v1", "enabled":enabled,
            "route_source":"production-policy" if enabled else "research-override-disabled",
            "qk_norm_rope_blocks":qk, "q6_ffn_down_blocks":q6, "gpu_state":_gpu_state(),
            "depth":depth, "count":count, "reps":reps, "included_cost":True,
            "settled_continuous":True, **settled}
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def bracket(depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  def child(enabled:bool, label:str) -> dict:
    arm_out = root/f"{label}.json"
    cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, sys.executable, str(pathlib.Path(__file__).resolve()),
           "--mode", "timing-child", "--depth", str(depth), "--count", str(count),
           "--max-context", str(max_context), "--reps", str(reps), "--out", str(arm_out)]
    if enabled: cmd.append("--enabled")
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env={**os.environ, "PYTHONPATH":"/home/ubuntu/tinygrad-arkey", "DEV":"NV"})
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    return json.loads(arm_out.read_text())
  arms = [child(False, "control_a"), child(True, "candidate"), child(False, "control_c")]
  control_mid = statistics.median((arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"]))
  candidate = arms[1]["median_ms_per_token"]
  hashes = {arm["token_stream_hash"] for arm in arms}
  result = {"schema":"tinygrad.nv_booked_composition_wall_bracket.v1", "mode":"reverse-bracket",
            "depth":depth, "count":count, "reps":reps,
            "control_a_ms_per_token":arms[0]["median_ms_per_token"],
            "candidate_ms_per_token":candidate,
            "control_c_ms_per_token":arms[2]["median_ms_per_token"],
            "control_midpoint_ms_per_token":control_mid,
            "candidate_minus_control_ms_per_token":candidate-control_mid,
            "recovery_us_per_token":(control_mid-candidate)*1000,
            "candidate_speedup_pct":(control_mid/candidate-1)*100,
            "all_token_hashes_equal":len(hashes)==1,
            "token_stream_hash":sorted(hashes)[0] if len(hashes)==1 else sorted(hashes),
            "verdict":"WALL_PASS" if len(hashes)==1 and candidate < control_mid else "NO_GO_WALL",
            "arms":arms}
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def main() -> int:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("timing","timing-child"),default="timing")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--count",type=int,default=32)
  ap.add_argument("--max-context",type=int,default=1024); ap.add_argument("--reps",type=int,default=5)
  ap.add_argument("--enabled",action="store_true"); ap.add_argument("--out",type=pathlib.Path,required=True)
  args=ap.parse_args()
  result = (timing_child(args.depth,args.count,args.max_context,args.reps,args.enabled,args.out)
            if args.mode == "timing-child" else bracket(args.depth,args.count,args.max_context,args.reps,args.out))
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
