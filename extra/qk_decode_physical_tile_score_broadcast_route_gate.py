#!/usr/bin/env python3
"""Route gate for score-broadcast PALL lifecycle decode route."""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-space"
CLEAR_FLAGS = (
  "DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
  "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
  "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
  "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
  "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
  "DECODE_ATTN_PHYSICAL_TILE_P1_SCORE", "DECODE_ATTN_PHYSICAL_TILE_PALL_SCORE",
  "DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE",
  "DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS",
  "V_DOT2_LOWERING", "WARP_REDUCE_LOWERING",
)

def _env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_SCORE_BROADCAST_ROUTE_CHILD": "1", "QK_SCORE_BROADCAST_ROUTE_ARM": arm}
  for k in CLEAR_FLAGS: env[k] = "0"
  if arm == "score_broadcast":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    env["DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE"] = "1"
    env["V_DOT2_LOWERING"] = "1"
    chunks = os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "4")
    env["DECODE_ATTN_SCORE_BROADCAST_CHUNKS"] = chunks
    if chunks != "4": env["DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS"] = "1"
  return env

def _signature(route: dict[str, Any]) -> dict[str, Any]:
  gen = [n for n in route["route_fire"]["program_node_names"] if n.startswith("flash_")]
  pv_targets = [f"flash_pall_score_broadcast_pv_cols_{i}_32_32_128" for i in (0, 32, 64, 96)]
  expected = ["flash_pall_score_once_state_32_128", *pv_targets, "flash_pall_score_broadcast_combine4_32_128"]
  unexpected = [n for n in gen if not any(n.startswith(e) for e in expected)]
  return {
    "generated_attention_programs": gen,
    "has_state": any(n.startswith("flash_pall_score_once_state_32_128") for n in gen),
    "pv_targets_present": {t: any(n.startswith(t) for n in gen) for t in pv_targets},
    "has_combine": any(n.startswith("flash_pall_score_broadcast_combine4_32_128") for n in gen),
    "generated_program_count_ok": len(gen) == len(expected) and not unexpected,
    "unexpected_generated_programs": unexpected,
    "has_old_score_chain": any(n.startswith(("flash_score_whole_cache", "flash_prob", "flash_partial_coop_vec", "flash_den", "flash_combine_32_128")) for n in gen),
    "has_owned": route["route_counts"]["owned_flash_tile_gqa_whole"] or route["route_counts"]["owned_flash_combine"],
  }

def _child_route(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  route = capture("a2" if arm == "score_broadcast" else "baseline")
  return {"arm": arm, "route": route, "signature": _signature(route)}

def _run_child(arm: str) -> dict[str, Any]:
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_env(arm),
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if p.returncode != 0: return {"arm": arm, "failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "returncode": 0, "error": "no json", "output_tail": (p.stdout or "")[-12000:]}

def build() -> dict[str, Any]:
  chunks = int(os.environ.get("DECODE_ATTN_SCORE_BROADCAST_CHUNKS", "4"))
  baseline = _run_child("baseline")
  route = _run_child("score_broadcast")
  diagnostic_chunks = chunks != 4
  gate = {"checked": True, "pass": False, "chunks": chunks, "diagnostic_chunks": diagnostic_chunks, "baseline": baseline, "score_broadcast": route}
  if not baseline.get("failed") and not route.get("failed"):
    r = route["route"]; sig = route["signature"]
    mat = r["materialization"]
    gate["pass"] = bool(not diagnostic_chunks and r["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
      sig["has_state"] and all(sig["pv_targets_present"].values()) and sig["has_combine"] and
      sig["generated_program_count_ok"] and
      not sig["has_old_score_chain"] and not sig["has_owned"] and
      baseline["route"]["tokens_sample"] == r["tokens_sample"] and
      not mat["E_49152_present"] and not mat["full_maxc_copy_kernels"] and
      mat.get("selected_route_buffer_identity", mat["buffer_identity_inputs"]))
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": "SCORE_BROADCAST_ROUTE_CLEAN__WD_NEXT" if gate["pass"] else "SCORE_BROADCAST_ROUTE_FAIL",
          "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
                    "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": str(chunks),
                    "DECODE_ATTN_SCORE_BROADCAST_DIAGNOSTIC_CHUNKS": "1" if diagnostic_chunks else "0",
                    "V_DOT2_LOWERING": "1"},
          "route_gate": gate,
          "decision": "Run bounded W==D only if route is clean with all four chunks; reduced chunks are liveness diagnostics only."}

def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_SCORE_BROADCAST_ROUTE_CHILD") == "1":
    print(json.dumps(_child_route(os.environ.get("QK_SCORE_BROADCAST_ROUTE_ARM", "baseline"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "score_broadcast_route_latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"score-broadcast-route-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "SCORE_BROADCAST_ROUTE_CLEAN__WD_NEXT" else 1

if __name__ == "__main__": raise SystemExit(main())
