#!/usr/bin/env python3
"""A3 decode-attention baseline profiler.

Compares the shipped owned attention route against the A2 generated whole-cache
skeleton. This is measurement-only and does not promote anything.
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-baseline"
CTXS = (512, 1024, 2048, 4096)


def _run_child(arm: str) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_A3_BASELINE_CHILD": "1", "QK_A3_BASELINE_ARM": arm}
  if arm == "a2":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  else:
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "0"
    env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=env,
                     capture_output=True, text=True)
  if r.returncode != 0:
    raise RuntimeError(f"{arm} child failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  raise RuntimeError(f"{arm} child did not emit JSON\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  from extra.qk_decode_search_gate import _setup_model, run_wd

  if arm == "a2":
    os.environ["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    os.environ["DECODE_ATTN_GENERATED_SKELETON"] = "0"
    mode = "a2"
  else:
    os.environ["DECODE_ATTN_GENERATED_WHOLECACHE"] = "0"
    os.environ["DECODE_ATTN_GENERATED_SKELETON"] = "0"
    mode = "baseline"
  route = capture(mode)
  m, _tok = _setup_model()
  wd = run_wd(m, ctxs=list(CTXS))
  names = route["route_fire"]["program_node_names"]
  flash_programs = [n for n in names if n.startswith("flash_")]
  return {
    "arm": arm,
    "mode": mode,
    "route": route,
    "wd": wd,
    "flash_programs": flash_programs,
    "program_presence": {
      "owned_flash_tile_gqa_whole": route["route_counts"]["owned_flash_tile_gqa_whole"],
      "owned_flash_combine": route["route_counts"]["owned_flash_combine"],
      "generated_attention_programs": route["route_counts"]["generated_attention_programs"],
      "flash_score_whole_cache": any(n.startswith("flash_score_whole_cache") for n in flash_programs),
      "flash_partial_whole_cache": any(n.startswith("flash_partial_coop_vec_whole_cache") for n in flash_programs),
    },
  }


def _delta_rows(owned: dict[str, Any], a2: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for ctx in CTXS:
    o = owned["wd"][str(ctx)]["tok_s"]
    g = a2["wd"][str(ctx)]["tok_s"]
    rows.append({
      "ctx": ctx,
      "owned_tok_s": o,
      "a2_tok_s": g,
      "a2_vs_owned_pct": round(100.0 * g / o, 1) if o else None,
      "delta_tok_s": round(g - o, 1),
      "owned_spread_pct": owned["wd"][str(ctx)]["spread_pct"],
      "a2_spread_pct": a2["wd"][str(ctx)]["spread_pct"],
    })
  return rows


def build() -> dict[str, Any]:
  owned = _run_child("owned")
  a2 = _run_child("a2")
  rows = _delta_rows(owned, a2)
  route_clean = (
    a2["route"]["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
    a2["route"]["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and
    a2["route"]["route_counts"]["owned_flash_combine"] == 0 and
    not a2["route"]["materialization"]["E_49152_present"] and
    bool(a2["route"]["materialization"]["selected_route_buffer_identity"]) and
    owned["route"]["tokens_sample"] == a2["route"]["tokens_sample"]
  )
  verdict = "DECODE_ATTENTION_A3_BASELINE_CAPTURED" if route_clean else "DECODE_ATTENTION_A3_BASELINE_FAIL"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "candidate": "decode_attention_generated_wholecache_skeleton",
    "authority": "W==D via extra.qk_decode_search_gate.run_wd; route/materialization via extra.qk_decode_attention_purity_capture",
    "ctxs": list(CTXS),
    "route_clean": route_clean,
    "token_byte_identical": owned["route"]["tokens_sample"] == a2["route"]["tokens_sample"],
    "rows": rows,
    "owned": owned,
    "a2": a2,
    "interpretation": (
      "A2 is lifecycle-clean. Use rows to identify the speed gap before primitive lowering."
      if route_clean else
      "A2 route cleanliness failed; do not start primitive lowering."
    ),
    "next": "A3.1 whole-cache score v_dot2 lowering if route_clean is true.",
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_A3_BASELINE_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_A3_BASELINE_ARM", "owned"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-baseline-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "DECODE_ATTENTION_A3_BASELINE_CAPTURED" else 1


if __name__ == "__main__":
  raise SystemExit(main())
