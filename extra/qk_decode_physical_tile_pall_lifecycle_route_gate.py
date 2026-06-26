#!/usr/bin/env python3
"""Route gate for the generated PALL physical lifecycle decode route."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
TARGET = "flash_pall_score_state_pv_lifecycle_32_128"
CLEAR_FLAGS = (
  "DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
  "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
  "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
  "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
  "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
  "DECODE_ATTN_PHYSICAL_TILE_P1_SCORE", "DECODE_ATTN_PHYSICAL_TILE_PALL_SCORE",
  "DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE", "V_DOT2_LOWERING", "WARP_REDUCE_LOWERING",
)

def _env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_PALL_LIFECYCLE_ROUTE_CHILD": "1", "QK_PALL_LIFECYCLE_ROUTE_ARM": arm}
  for k in CLEAR_FLAGS: env[k] = "0"
  if arm == "pall_lifecycle":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    env["DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE"] = "1"
    env["V_DOT2_LOWERING"] = "1"
  return env

def _signature(route: dict[str, Any]) -> dict[str, Any]:
  gen = [n for n in route["route_fire"]["program_node_names"] if n.startswith("flash_")]
  return {
    "generated_attention_programs": gen,
    "has_target": any(n.startswith(TARGET) for n in gen),
    "has_score_chain": any(n.startswith(("flash_score_whole_cache", "flash_pall_lds_crosslane_score", "flash_prob", "flash_partial_coop_vec", "flash_den", "flash_combine")) for n in gen),
    "has_state_tail": any(n == "flash_state_gmax_32_128" for n in gen) and any(n.startswith("flash_state_combine_32_128") for n in gen),
    "has_owned": route["route_counts"]["owned_flash_tile_gqa_whole"] or route["route_counts"]["owned_flash_combine"],
  }

def _child_route(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  route = capture("a2" if arm == "pall_lifecycle" else "baseline")
  return {"arm": arm, "route": route, "signature": _signature(route)}

def _run_child(arm: str) -> dict[str, Any]:
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_env(arm),
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if p.returncode != 0: return {"arm": arm, "failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "returncode": 0, "error": "no json", "output_tail": (p.stdout or "")[-8000:]}

def build() -> dict[str, Any]:
  baseline = _run_child("baseline")
  lifecycle = _run_child("pall_lifecycle")
  route_gate = {"checked": True, "pass": False, "baseline": baseline, "pall_lifecycle": lifecycle}
  if not baseline.get("failed") and not lifecycle.get("failed"):
    route = lifecycle["route"]; sig = lifecycle["signature"]
    route_gate["pass"] = bool(route["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
      sig["has_target"] and sig["has_state_tail"] and not sig["has_score_chain"] and not sig["has_owned"] and
      baseline["route"]["tokens_sample"] == route["tokens_sample"] and not route["materialization"]["E_49152_present"])
  verdict = "PALL_LIFECYCLE_ROUTE_CLEAN__WD_NEXT" if route_gate["pass"] else "PALL_LIFECYCLE_ROUTE_FAIL"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_pall_lifecycle", "verdict": verdict,
          "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE": "1", "V_DOT2_LOWERING": "1"},
          "route_gate": route_gate,
          "decision": "Route is ready for a bounded W==D falsification run, but known q.k-per-output-column recompute makes speed risk high."}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_PALL_LIFECYCLE_ROUTE_CHILD") == "1":
    print(json.dumps(_child_route(os.environ.get("QK_PALL_LIFECYCLE_ROUTE_ARM", "baseline"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "pall_lifecycle_route_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"pall-lifecycle-route-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "PALL_LIFECYCLE_ROUTE_CLEAN__WD_NEXT" else 1

if __name__ == "__main__": raise SystemExit(main())
