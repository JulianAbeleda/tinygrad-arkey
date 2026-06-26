#!/usr/bin/env python3
"""P7 structural gate for token-sharded x-lane online-state+PV decode-attention tile."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-xlane"
MANIFEST = ROOT / "bench/qk-search-spaces/decode_attention_online_softmax_pv_tile_v1.json"


def _env_for_arm(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_ONLINE_STATE_PV_XLANE_CHILD": "1", "QK_ONLINE_STATE_PV_XLANE_ARM": arm}
  for k in ("DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
            "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_LDS_TILE", "DECODE_ATTN_TILE_PLACEHOLDER",
            "DECODE_ATTN_TILE_SCORE_MAX", "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV",
            "DECODE_ATTN_TILE_PROB_PARTIAL_PV", "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE",
            "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE", "V_DOT2_LOWERING", "WARP_REDUCE_LOWERING"):
    env[k] = "0"
  if arm == "xlane":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    env["DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE"] = "1"
    env["DECODE_ATTN_TILE_COMBINE_BUNDLE"] = "1"
  return env


def _run_child(arm: str, allow_failure: bool=False) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=_env_for_arm(arm), capture_output=True, text=True)
  if r.returncode != 0:
    if allow_failure:
      return {"arm": arm, "child_failed": True, "returncode": r.returncode, "stdout_tail": r.stdout[-4000:], "stderr_tail": r.stderr[-12000:]}
    raise RuntimeError(f"{arm} child failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  raise RuntimeError(f"{arm} child did not emit JSON\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def _signature(route: dict[str, Any]) -> dict[str, Any]:
  names = list(route["route_fire"]["program_node_names"])
  generated = [n for n in names if n.startswith("flash_")]
  src_join = "\n".join(str(x) for x in generated)
  return {
    "generated_attention_programs": generated,
    "has_xlane_tile": any(n.startswith("flash_online_state_pv_tile_xlane_whole_cache") for n in generated),
    "has_state_gmax": any(n.startswith("flash_state_gmax") for n in generated),
    "has_state_combine": any(n.startswith("flash_state_combine") for n in generated),
    "has_score": any(n.startswith("flash_score_whole_cache") for n in generated),
    "has_stale_scalar_state_tile": any(n.startswith("flash_online_state_pv_tile_whole_cache") for n in generated),
    "has_external_max": any(n == "flash_max_32" or n.startswith("flash_tile_score_max") for n in generated),
    "has_external_den": any(n == "flash_den_32" for n in generated),
    "has_old_prob_or_partial": any("prob" in n or "partial" in n for n in generated),
    "name_has_xlane": "xlane" in src_join,
  }


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  mode = "a2" if arm == "xlane" else "baseline"
  route = capture(mode)
  return {"arm": arm, "route": route, "signature": _signature(route)}


def build() -> dict[str, Any]:
  manifest = json.loads(MANIFEST.read_text())
  xlane = _run_child("xlane", allow_failure=True)
  if xlane.get("child_failed"):
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_XLANE_FAIL__CAPTURE", "search_space_id": manifest["search_space_id"],
            "candidate_id": "decode_attention_online_state_pv_xlane_p7", "xlane": xlane,
            "decision": "Do not proceed; classify/fix token-sharded x-lane tile capture first."}
  owned = _run_child("owned")
  route = xlane["route"]
  sig = xlane["signature"]
  token_match = owned["route"]["tokens_sample"] == route["tokens_sample"]
  route_clean = (
    route["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
    route["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and
    route["route_counts"]["owned_flash_combine"] == 0 and
    not route["materialization"]["E_49152_present"] and
    bool(route["materialization"]["selected_route_buffer_identity"]) and
    token_match
  )
  if not token_match:
    verdict = "ONLINE_STATE_PV_XLANE_FAIL__TOKEN_MISMATCH"
  elif not route_clean:
    verdict = "ONLINE_STATE_PV_XLANE_FAIL__ROUTE_OR_MATERIALIZATION"
  elif not (sig["has_xlane_tile"] and sig["has_state_gmax"] and sig["has_state_combine"] and sig["has_score"]):
    verdict = "ONLINE_STATE_PV_XLANE_FAIL__PROGRAM_NOT_BOUND"
  elif sig["has_stale_scalar_state_tile"] or sig["has_external_max"] or sig["has_external_den"] or sig["has_old_prob_or_partial"]:
    verdict = "ONLINE_STATE_PV_XLANE_FAIL__STALE_STAGE_PRESENT"
  else:
    verdict = "ONLINE_STATE_PV_XLANE_STRUCTURAL_ROUTE_CLEAN"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "search_space_id": manifest["search_space_id"],
    "candidate_id": "decode_attention_online_state_pv_xlane_p7",
    "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE": "1"},
    "token_match": token_match,
    "route_clean": route_clean,
    "owned": owned,
    "xlane": xlane,
    "decision": "If structural route is clean, next run ISA/resource/W==D attribution; otherwise fix x-lane route first."
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_ONLINE_STATE_PV_XLANE_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_ONLINE_STATE_PV_XLANE_ARM", "owned"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-xlane-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ONLINE_STATE_PV_XLANE_STRUCTURAL_ROUTE_CLEAN" else 1


if __name__ == "__main__":
  raise SystemExit(main())
