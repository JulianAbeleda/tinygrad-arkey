#!/usr/bin/env python3
"""In-model route gate for the fused x-lane score+PV decode tile.

Captures the owned baseline and the DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE route under full-model decode
and checks: tokens match owned, route is buffer-identity clean (no full-MAXC copy), owned kernels absent,
the fused-xlane tile + state gmax + state combine fire, and the generated whole-cache route is clean.
Reports the occupancy proxy for the configured split count.

Run: DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_attention_fused_xlane_score_pv_route_gate.py
Scope: docs/decode-fused-tile-occupancy-roofline-baseline.md
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-fused-xlane-score-pv-route"
TARGET = "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128" if os.environ.get("DECODE_ATTN_BLOCK_TILE", "0") != "0" else \
  "flash_fused_xlane_score_pv_tile_whole_cache_32_128"
CU_COUNT = 96
HKV = 8

_ZERO = ("DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
         "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
         "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
         "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
         "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
         "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", "DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE")


def _route_env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_FUSED_XLANE_SCORE_PV_CHILD": "1", "QK_FUSED_XLANE_SCORE_PV_ARM": arm}
  for k in _ZERO: env[k] = "0"
  if arm == "xlane":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    env["DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE"] = "1"
  return env


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  return {"arm": arm, "route": capture("a2" if arm == "xlane" else "baseline")}


def _run_child(arm: str) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_route_env(arm),
                     capture_output=True, text=True)
  if r.returncode != 0:
    return {"arm": arm, "failed": True, "returncode": r.returncode, "stderr_tail": r.stderr[-6000:]}
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "error": "no json", "stdout_tail": r.stdout[-6000:], "stderr_tail": r.stderr[-6000:]}


def build() -> dict[str, Any]:
  baseline = _run_child("baseline")
  xlane = _run_child("xlane")
  s_cfg = int(os.environ.get("DECODE_ATTN_FUSED_XLANE_SCORE_PV_S", "48"))
  occ = {"split_count": s_cfg, "tile_workgroups": HKV * s_cfg, "wg_per_cu": round(HKV * s_cfg / CU_COUNT, 2)}
  if baseline.get("failed") or xlane.get("failed"):
    verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__CHILD"
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
            "occupancy": occ, "baseline": baseline, "xlane": xlane}
  rb, rx = baseline["route"], xlane["route"]
  names = list(rx["route_fire"]["program_node_names"])
  generated = [n for n in names if n.startswith("flash_")]
  mat = rx["materialization"]
  token_match = rb["tokens_sample"] == rx["tokens_sample"]
  materialization_clean = (not mat["E_49152_present"]) and len(mat.get("full_maxc_copy_kernels", [])) == 0 and bool(mat["selected_route_buffer_identity"])
  owned_absent = rx["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and rx["route_counts"]["owned_flash_combine"] == 0
  generated_clean = rx["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN"
  has_target = any(n.startswith(TARGET) for n in generated)
  has_gmax = any(n == "flash_state_gmax_32_128" for n in generated)
  has_combine = any(n.startswith("flash_state_combine_32_128") for n in generated)
  passed = token_match and materialization_clean and owned_absent and generated_clean and has_target and has_gmax and has_combine
  if not token_match: verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__TOKEN_MISMATCH"
  elif not materialization_clean: verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__MATERIALIZATION"
  elif not owned_absent: verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__OWNED_ROUTE_PRESENT"
  elif not has_target: verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__TARGET_PROGRAM_MISSING"
  elif not (has_gmax and has_combine): verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__INCOMPLETE_LIFECYCLE"
  elif not generated_clean: verdict = "FUSED_XLANE_SCORE_PV_ROUTE_FAIL__CAPTURE_NOT_CLEAN"
  else: verdict = "FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict, "pass": passed,
          "occupancy": occ, "token_match": token_match, "materialization_clean": materialization_clean,
          "owned_absent": owned_absent, "generated_clean": generated_clean,
          "has_target": has_target, "has_gmax": has_gmax, "has_combine": has_combine,
          "generated_attention_programs": generated,
          "tokens_baseline": rb["tokens_sample"], "tokens_xlane": rx["tokens_sample"],
          "decision": "If clean, run the attribution economics pre-gate (expect has_v_dot2/has_lds true, wg/CU~4) then W==D."}


def main() -> int:
  if os.environ.get("QK_FUSED_XLANE_SCORE_PV_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_FUSED_XLANE_SCORE_PV_ARM", "baseline"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"fused-xlane-score-pv-route-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "FUSED_XLANE_SCORE_PV_ROUTE_CLEAN__ECONOMICS_NEXT" else 1


if __name__ == "__main__":
  raise SystemExit(main())
