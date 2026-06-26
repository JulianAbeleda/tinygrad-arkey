#!/usr/bin/env python3
"""A3.2 cross-lane lowering gate for generated whole-cache decode attention."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-2-cross-lane"
CTXS = (512, 1024, 2048, 4096)


def _env_for_arm(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_A32_XLANE_CHILD": "1", "QK_A32_XLANE_ARM": arm}
  env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  env["DECODE_ATTN_SCORE_VDOT2"] = "0"
  env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1" if arm in ("a2", "a32") else "0"
  if arm == "a32":
    env["WARP_REDUCE_LOWERING"] = "1"
    env["DECODE_ATTN_CROSS_LANE"] = "1"
  else:
    env["WARP_REDUCE_LOWERING"] = "0"
    env["DECODE_ATTN_CROSS_LANE"] = "0"
  return env


def _run_child(arm: str, allow_failure: bool=False) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=_env_for_arm(arm),
                     capture_output=True, text=True)
  if r.returncode != 0:
    if allow_failure:
      return {
        "arm": arm,
        "child_failed": True,
        "returncode": r.returncode,
        "stdout_tail": r.stdout[-4000:],
        "stderr_tail": r.stderr[-8000:],
      }
    raise RuntimeError(f"{arm} child failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  raise RuntimeError(f"{arm} child did not emit JSON\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  from extra.qk_decode_search_gate import _setup_model, run_wd

  mode = "a2" if arm in ("a2", "a32") else "baseline"
  route = capture(mode)
  m, _tok = _setup_model()
  wd = run_wd(m, ctxs=list(CTXS))
  names = route["route_fire"]["program_node_names"]
  return {
    "arm": arm,
    "route": route,
    "wd": wd,
    "flash_programs": [n for n in names if n.startswith("flash_")],
    "requested_warp_reduce_lowering": os.environ.get("WARP_REDUCE_LOWERING") == "1",
    "axis_mapping_note": "WARP_REDUCE_LOWERING only applies to WARP/GROUP_REDUCE axes; A2 attention reductions are not yet lane-mapped.",
  }


def _rows(owned: dict[str, Any], a2: dict[str, Any], a32: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for ctx in CTXS:
    o, b, x = owned["wd"][str(ctx)]["tok_s"], a2["wd"][str(ctx)]["tok_s"], a32["wd"][str(ctx)]["tok_s"]
    rows.append({
      "ctx": ctx,
      "owned_tok_s": o,
      "a2_tok_s": b,
      "a32_tok_s": x,
      "a32_vs_a2_pct": round(100.0 * x / b, 1) if b else None,
      "a32_vs_owned_pct": round(100.0 * x / o, 1) if o else None,
      "delta_vs_a2_tok_s": round(x - b, 1),
      "a2_spread_pct": a2["wd"][str(ctx)]["spread_pct"],
      "a32_spread_pct": a32["wd"][str(ctx)]["spread_pct"],
    })
  return rows


def _route_clean(owned: dict[str, Any], arm: dict[str, Any]) -> bool:
  r = arm["route"]
  return (
    r["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
    r["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and
    r["route_counts"]["owned_flash_combine"] == 0 and
    not r["materialization"]["E_49152_present"] and
    bool(r["materialization"]["selected_route_buffer_identity"]) and
    owned["route"]["tokens_sample"] == r["tokens_sample"]
  )


def build() -> dict[str, Any]:
  a32 = _run_child("a32", allow_failure=True)
  if a32.get("child_failed"):
    return {
      "date": "2026-06-25",
      "timestamp": time.strftime("%Y%m%d-%H%M%S"),
      "verdict": "A3_2_BLOCKED_BY_CODEGEN_GLOBAL_WARP_REDUCE",
      "candidate": "decode_attention_generated_wholecache_cross_lane_requested",
      "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "WARP_REDUCE_LOWERING": "1", "DECODE_ATTN_CROSS_LANE": "1"},
      "route_clean": False,
      "rows": [],
      "a32": a32,
      "decision": "Do not promote. WARP_REDUCE_LOWERING is not safely scoped to the attention candidate; next work is explicit attention lane-axis mapping or local scoped lowering.",
    }
  owned = _run_child("owned")
  a2 = _run_child("a2")
  rows = _rows(owned, a2, a32)
  route_clean = _route_clean(owned, a32)
  transfer_rows = [r for r in rows if r["a32_tok_s"] > r["a2_tok_s"] and (r["a32_tok_s"] - r["a2_tok_s"]) > max(r["a2_spread_pct"], r["a32_spread_pct"]) * r["a2_tok_s"] / 100.0]
  if not route_clean:
    verdict = "A3_2_FAIL__ROUTE_OR_TOKEN_OR_MATERIALIZATION"
  elif len(transfer_rows) >= 2:
    verdict = "A3_2_CROSS_LANE_TRANSFERS"
  else:
    verdict = "A3_2_CROSS_LANE_NO_TRANSFER__AXIS_MAPPING_NOT_EXPOSED"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "candidate": "decode_attention_generated_wholecache_cross_lane_requested",
    "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "WARP_REDUCE_LOWERING": "1", "DECODE_ATTN_CROSS_LANE": "1"},
    "route_clean": route_clean,
    "rows": rows,
    "owned": owned,
    "a2": a2,
    "a32": a32,
    "decision": "Do not promote. If no transfer, next work is explicit lane-axis mapping or LDS tile layout.",
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_A32_XLANE_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_A32_XLANE_ARM", "owned"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-2-cross-lane-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("A3_2_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
