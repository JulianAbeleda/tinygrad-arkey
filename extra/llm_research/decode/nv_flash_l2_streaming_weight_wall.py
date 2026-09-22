#!/usr/bin/env python3
"""Reverse token-wall bracket for evict-first Q/K/V projection weight loads."""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode.qk_norm_rope_wall_bracket import MODEL, LOCK, _gpu_state

BOOKING_BAR_US = 50.0
QKV_TARGETS = ",".join((
  "q4k_g3_lanemap_gemv_vec_4096_4096",
  "q4k_g3_lanemap_gemv_vec_1024_4096",
  "q6k_v_four_warp_fp16_direct_1024_4096",
))
WHOLE_DENSE_TARGETS = ",".join((
  "q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096@1",
  "q4k_gate_up_four_warp_vec_fp16_12288_4096@1+2",
  "q4k_g3_lanemap_gemv_vec_4096_4096@1",
  "q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd@1",
  "q4k_fp16_mmvq_direct_vec_4096_12288_epi_ffnresadd@1",
  "q4k_warp_coop_q8_dp4a_direct_4096_4096@1",
  "q4k_g3_lanemap_gemv_vec_1024_4096@1",
  "q6k_v_four_warp_fp16_direct_1024_4096@1",
  "q4k_warp_coop_q8_dp4a_pair_direct_1024_4096@2+3",
  "q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096@2+3",
  "q4k_g3_lanemap_gemv_pair_vec_1024_4096@2+3",
  "q6k_gen_coop_151936_4096_inkernel@1",
))
Q6_REUSE_CLASS_TARGET = "q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd_splitstream@2"


def _targets(scope:str) -> str:
  return QKV_TARGETS if scope == "qkv" else Q6_REUSE_CLASS_TARGET if scope == "q6_payload" else WHOLE_DENSE_TARGETS


def child(candidate:bool, scope:str, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  targets = _targets(scope)
  if candidate:
    os.environ["NV_L2_STREAMING_WEIGHT_PROGRAMS"] = targets
    if scope == "q6_payload": os.environ["NV_Q6_FFN_DOWN_SPLIT_WEIGHT_STREAM"] = "1"
  else: os.environ.pop("NV_L2_STREAMING_WEIGHT_PROGRAMS", None)
  if not candidate: os.environ.pop("NV_Q6_FFN_DOWN_SPLIT_WEIGHT_STREAM", None)
  os.environ.update(DEV="NV", PROFILE="0")
  os.environ.pop("HCQ_GRAPH_PROFILE_JSON", None)
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  model = _load(MODEL, max_context)
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = True
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try: settled = _settled_continuous_windows(gen, Device["NV"], count, reps)
  finally: gen.close()
  result = {"schema":"tinygrad.nv_flash_l2_streaming_weight_wall.v1", "candidate":candidate,
    "scope":scope, "targets":targets.split(",") if candidate else [], "gpu_state":_gpu_state(),
    "depth":depth, "count":count, "reps":reps, "max_context":max_context, **settled}
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def bracket(scope:str, depth:int, count:int, max_context:int, reps:int, out:pathlib.Path) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json")); root.mkdir(parents=True, exist_ok=True)
  def arm(candidate:bool, label:str) -> dict:
    dst = root/f"{label}.json"
    cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, sys.executable, str(pathlib.Path(__file__).resolve()),
      "--mode", "child", "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
      "--reps", str(reps), "--scope", scope, "--out", str(dst)]
    if candidate: cmd.append("--candidate")
    env = {**os.environ, "PYTHONPATH":str(ROOT), "DEV":"NV", "PROFILE":"0"}
    env.pop("NV_L2_STREAMING_WEIGHT_PROGRAMS", None)
    env.pop("NV_Q6_FFN_DOWN_SPLIT_WEIGHT_STREAM", None)
    run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if run.returncode: raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-6000:]}")
    return json.loads(dst.read_text())
  arms = [arm(False, "control_a"), arm(True, "candidate"), arm(False, "control_c")]
  control = statistics.median((arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"]))
  cand = arms[1]["median_ms_per_token"]
  hashes = {x["token_stream_hash"] for x in arms}
  recovery = (control-cand)*1000
  result = {"schema":"tinygrad.nv_flash_l2_streaming_weight_wall.v1", "mode":"reverse-bracket",
    "scope":scope, "targets":_targets(scope).split(","), "depth":depth, "count":count, "reps":reps, "max_context":max_context,
    "all_token_hashes_equal":len(hashes)==1, "token_stream_hashes":sorted(hashes),
    "walls_us_per_token":{"control_a":arms[0]["median_ms_per_token"]*1000, "candidate":cand*1000,
      "control_c":arms[2]["median_ms_per_token"]*1000, "control_midpoint":control*1000,
      "candidate_delta":-recovery},
    "recovery_us_per_token":recovery, "booking_bar_us_per_token":BOOKING_BAR_US,
    "tokens_per_second":{"control_midpoint":1000/control, "candidate":1000/cand,
      "candidate_delta":1000/cand-1000/control},
    "verdict":("BOOKING_PASS" if len(hashes)==1 and recovery >= BOOKING_BAR_US and
      cand < min(arms[0]["median_ms_per_token"], arms[2]["median_ms_per_token"])
      else "MECHANISM_ONLY_BELOW_BOOKING_BAR" if len(hashes)==1 and recovery > 0 else "NO_GO_WALL"),
    "arms":arms}
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  return result


def main() -> None:
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("timing","child"),default="timing")
  ap.add_argument("--candidate",action="store_true"); ap.add_argument("--scope",choices=("qkv","whole_dense","q6_payload"),default="qkv")
  ap.add_argument("--depth",type=int,default=512)
  ap.add_argument("--count",type=int,default=16); ap.add_argument("--max-context",type=int,default=1024)
  ap.add_argument("--reps",type=int,default=7); ap.add_argument("--out",type=pathlib.Path,required=True); a=ap.parse_args()
  if a.depth + 7 + a.count*a.reps > a.max_context: raise ValueError("depth + warmup + count*reps must fit max-context")
  result = child(a.candidate,a.scope,a.depth,a.count,a.max_context,a.reps,a.out) if a.mode=="child" else bracket(a.scope,a.depth,a.count,a.max_context,a.reps,a.out)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
