#!/usr/bin/env python3
"""A3.8 route/delta attribution audit for generated decode attention stages."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-8-stage-attribution"
CTXS = (512, 1024, 2048, 4096)

ARMS = {
  "a2": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1"},
  "a36_tile_max": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_TILE_SCORE_MAX": "1"},
  "a37_tile_prob": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_TILE_PROB": "1"},
}


def _base_env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_A38_ATTR_CHILD": "1", "QK_A38_ATTR_ARM": arm}
  for k in ("DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
            "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_LDS_TILE", "DECODE_ATTN_TILE_PLACEHOLDER",
            "DECODE_ATTN_TILE_SCORE_MAX", "DECODE_ATTN_TILE_PROB", "WARP_REDUCE_LOWERING"):
    env[k] = "0"
  env.update(ARMS[arm])
  return env


def _run_child(arm: str) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=_base_env(arm), capture_output=True, text=True)
  if r.returncode != 0:
    return {"arm": arm, "child_failed": True, "returncode": r.returncode, "stdout_tail": r.stdout[-4000:], "stderr_tail": r.stderr[-8000:]}
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "child_failed": True, "returncode": 0, "stdout_tail": r.stdout[-4000:], "stderr_tail": r.stderr[-8000:], "error": "no json"}


def _classify_programs(names: list[str]) -> dict[str, Any]:
  generated = [n for n in names if n.startswith("flash_")]
  classes = {
    "score": [n for n in generated if "score" in n],
    "tile": [n for n in generated if "tile" in n],
    "max": [n for n in generated if "max" in n],
    "prob": [n for n in generated if "prob" in n],
    "partial": [n for n in generated if "partial" in n],
    "gmax": [n for n in generated if "gmax" in n],
    "den": [n for n in generated if "den" in n],
    "combine": [n for n in generated if "combine" in n],
  }
  return {
    "generated_programs": generated,
    "generated_program_count": len(generated),
    "classes": classes,
    "stage_counts": {k: len(v) for k, v in classes.items()},
    "has_flash_max": "flash_max_32" in generated,
    "has_flash_prob": "flash_prob_32" in generated,
    "has_tile_score_max": any(n.startswith("flash_tile_score_max") for n in generated),
    "has_tile_prob": any(n.startswith("flash_tile_prob") for n in generated),
    "has_partial_pv": any(n.startswith("flash_partial") for n in generated),
  }


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  from extra.qk_decode_search_gate import _setup_model, run_wd

  route = capture("a2")
  names = list(route["route_fire"]["program_node_names"])
  m, _tok = _setup_model()
  return {
    "arm": arm,
    "env": {k: os.environ.get(k, "") for k in os.environ if k.startswith("DECODE_ATTN") or k == "WARP_REDUCE_LOWERING"},
    "route": route,
    "signature": _classify_programs(names),
    "wd": run_wd(m, ctxs=list(CTXS)),
  }


def _route_clean(ref_tokens: list[int], arm: dict[str, Any]) -> bool:
  if arm.get("child_failed"): return False
  r = arm["route"]
  return (
    r["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN" and
    r["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and
    r["route_counts"]["owned_flash_combine"] == 0 and
    not r["materialization"]["E_49152_present"] and
    bool(r["materialization"]["selected_route_buffer_identity"]) and
    r["tokens_sample"] == ref_tokens
  )


def _diff_programs(base: dict[str, Any], arm: dict[str, Any]) -> dict[str, list[str]]:
  b = set(base["signature"]["generated_programs"])
  a = set(arm["signature"]["generated_programs"])
  return {"added": sorted(a - b), "removed": sorted(b - a), "kept": sorted(a & b)}


def _rows(base: dict[str, Any], arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  rows = []
  for ctx in CTXS:
    b = base["wd"][str(ctx)]["tok_s"]
    row: dict[str, Any] = {"ctx": ctx, "a2_tok_s": b}
    for name, arm in arms.items():
      if name == "a2": continue
      x = arm["wd"][str(ctx)]["tok_s"]
      row[f"{name}_tok_s"] = x
      row[f"{name}_vs_a2_pct"] = round(100.0 * x / b, 1) if b else None
      row[f"{name}_delta_tok_s"] = round(x - b, 1)
    rows.append(row)
  return rows


def _diagnose(arms: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[str, str]:
  a36 = arms["a36_tile_max"]
  a37 = arms["a37_tile_prob"]
  a36_clean = a36["route_clean"] and "flash_max_32" in a36["program_diff_vs_a2"]["removed"]
  a37_clean = a37["route_clean"] and {"flash_max_32", "flash_prob_32"}.issubset(set(a37["program_diff_vs_a2"]["removed"]))
  a36_deltas = [r["a36_tile_max_delta_tok_s"] for r in rows]
  a37_deltas = [r["a37_tile_prob_delta_tok_s"] for r in rows]
  a36_material = sum(1 for x in a36_deltas if x > 0.5)
  a37_material = sum(1 for x in a37_deltas if x > 0.5)
  if not a36_clean or not a37_clean:
    return "A3_8_ATTRIBUTION_INCONCLUSIVE__ROUTE_PAYLOAD_NOT_CLEAN", "Route or payload activation failed; do not infer stage bottleneck."
  if a36_material == 0 and a37_material <= 1:
    return "A3_8_ATTRIBUTION_READY__PARTIAL_PV_NEXT", "Max/prob metadata replacement did not materially transfer; next meaningful stage is partial PV."
  if a36_material >= 2 or a37_material >= 2:
    return "A3_8_ATTRIBUTION_READY__REPEAT_METADATA_OR_PROMOTE_CANDIDATE", "Metadata path may transfer; repeat before promoting or moving on."
  return "A3_8_ATTRIBUTION_INCONCLUSIVE__NEEDS_KERNEL_TIMING", "Route deltas are mixed; add per-kernel timing or isolated stage microbench."


def build() -> dict[str, Any]:
  raw = {arm: _run_child(arm) for arm in ARMS}
  if any(v.get("child_failed") for v in raw.values()):
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": "A3_8_FAIL__CHILD_CAPTURE", "arms": raw, "rows": [], "decision": "Fix child capture before attribution."}
  ref_tokens = raw["a2"]["route"]["tokens_sample"]
  for name, arm in raw.items():
    arm["route_clean"] = _route_clean(ref_tokens, arm)
  for name, arm in raw.items():
    if name != "a2": arm["program_diff_vs_a2"] = _diff_programs(raw["a2"], arm)
  rows = _rows(raw["a2"], raw)
  if not all(arm["route_clean"] for arm in raw.values()):
    verdict, diagnosis = "A3_8_FAIL__ROUTE_OR_TOKEN", "At least one arm failed route/materialization/token hygiene."
  else:
    verdict, diagnosis = _diagnose(raw, rows)
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "arms": raw,
    "rows": rows,
    "diagnosis": diagnosis,
    "decision": "Proceed to A3.8/A3.9 partial PV tile payload if verdict is PARTIAL_PV_NEXT; otherwise follow diagnosis.",
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_A38_ATTR_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_A38_ATTR_ARM", "a2"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-8-stage-attribution-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("A3_8_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
