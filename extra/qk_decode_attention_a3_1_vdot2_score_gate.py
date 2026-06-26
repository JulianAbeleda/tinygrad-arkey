#!/usr/bin/env python3
"""A3.1 score-vdot2 gate for generated whole-cache decode attention."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-1-vdot2-score"
CTXS = (512, 1024, 2048, 4096)


def _env_for_arm(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_A31_SCORE_CHILD": "1", "QK_A31_SCORE_ARM": arm}
  env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  if arm in ("a2", "a31"):
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
  else:
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "0"
  if arm == "a31":
    env["DECODE_ATTN_SCORE_VDOT2"] = "1"
    env["V_DOT2_LOWERING"] = "1"
  else:
    env["DECODE_ATTN_SCORE_VDOT2"] = "0"
    env.pop("V_DOT2_LOWERING", None)
  return env


def _run_child(arm: str) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=_env_for_arm(arm),
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

  mode = "a2" if arm in ("a2", "a31") else "baseline"
  route = capture(mode)
  m, _tok = _setup_model()
  wd = run_wd(m, ctxs=list(CTXS))
  names = route["route_fire"]["program_node_names"]
  return {
    "arm": arm,
    "route": route,
    "wd": wd,
    "score_programs": [n for n in names if n.startswith("flash_score_whole_cache")],
    "score_vdot2_program_present": any(n.startswith("flash_score_whole_cache_vdot2") for n in names),
  }


def _rows(owned: dict[str, Any], a2: dict[str, Any], a31: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for ctx in CTXS:
    o, b, v = owned["wd"][str(ctx)]["tok_s"], a2["wd"][str(ctx)]["tok_s"], a31["wd"][str(ctx)]["tok_s"]
    rows.append({
      "ctx": ctx,
      "owned_tok_s": o,
      "a2_tok_s": b,
      "a31_tok_s": v,
      "a31_vs_a2_pct": round(100.0 * v / b, 1) if b else None,
      "a31_vs_owned_pct": round(100.0 * v / o, 1) if o else None,
      "delta_vs_a2_tok_s": round(v - b, 1),
      "a2_spread_pct": a2["wd"][str(ctx)]["spread_pct"],
      "a31_spread_pct": a31["wd"][str(ctx)]["spread_pct"],
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
  owned = _run_child("owned")
  a2 = _run_child("a2")
  a31 = _run_child("a31")
  rows = _rows(owned, a2, a31)
  route_clean = _route_clean(owned, a31)
  vdot2_named = a31["score_vdot2_program_present"]
  transfer_rows = [r for r in rows if r["a31_tok_s"] > r["a2_tok_s"] and (r["a31_tok_s"] - r["a2_tok_s"]) > max(r["a2_spread_pct"], r["a31_spread_pct"]) * r["a2_tok_s"] / 100.0]
  if not route_clean:
    verdict = "A3_1_FAIL__ROUTE_OR_TOKEN_OR_MATERIALIZATION"
  elif not vdot2_named:
    verdict = "A3_1_FAIL__VDOT2_SCORE_ROUTE_NOT_CAPTURED"
  elif len(transfer_rows) >= 2:
    verdict = "A3_1_VDOT2_SCORE_TRANSFERS"
  elif len(transfer_rows) == 0:
    verdict = "A3_1_VDOT2_SCORE_NO_TRANSFER"
  else:
    verdict = "A3_1_VDOT2_SCORE_INCONCLUSIVE"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "candidate": "flash_score_whole_cache_vdot2_32_128",
    "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_SCORE_VDOT2": "1", "V_DOT2_LOWERING": "1"},
    "route_clean": route_clean,
    "vdot2_named_score_program": vdot2_named,
    "rows": rows,
    "owned": owned,
    "a2": a2,
    "a31": a31,
    "decision": "Continue to A3.2 if transfer is insufficient; promote nothing from A3.1 alone.",
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_A31_SCORE_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_A31_SCORE_ARM", "owned"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-1-vdot2-score-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("A3_1_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
