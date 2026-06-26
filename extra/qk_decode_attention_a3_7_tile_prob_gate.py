#!/usr/bin/env python3
"""A3.7 generated tile probability gate for decode attention."""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-7-tile-prob"
MANIFEST = ROOT / "bench/qk-search-spaces/decode_attention_tile_combine_a3_4.json"
CTXS = (512, 1024, 2048, 4096)


def _env_for_arm(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_A37_TILE_CHILD": "1", "QK_A37_TILE_ARM": arm}
  env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  env["DECODE_ATTN_SCORE_VDOT2"] = "0"
  env["DECODE_ATTN_SCORE_XLANE"] = "0"
  env["DECODE_ATTN_LDS_TILE"] = "0"
  env["DECODE_ATTN_TILE_PLACEHOLDER"] = "0"
  env["DECODE_ATTN_TILE_SCORE_MAX"] = "0"
  env["WARP_REDUCE_LOWERING"] = "0"
  env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1" if arm in ("a2", "a37") else "0"
  env["DECODE_ATTN_TILE_COMBINE_BUNDLE"] = "1" if arm == "a37" else "0"
  env["DECODE_ATTN_TILE_PROB"] = "1" if arm == "a37" else "0"
  return env


def _run_child(arm: str, allow_failure: bool=False) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=_env_for_arm(arm), capture_output=True, text=True)
  if r.returncode != 0:
    if allow_failure:
      return {"arm": arm, "child_failed": True, "returncode": r.returncode, "stdout_tail": r.stdout[-4000:], "stderr_tail": r.stderr[-8000:]}
    raise RuntimeError(f"{arm} child failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  raise RuntimeError(f"{arm} child did not emit JSON\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def _bundle_signature(names: list[str]) -> dict[str, Any]:
  generated = [n for n in names if n.startswith("flash_")]
  tile = [n for n in generated if "tile" in n.lower()]
  combine = [n for n in generated if "combine" in n.lower()]
  partial = [n for n in generated if "partial" in n.lower()]
  score = [n for n in generated if "score" in n.lower()]
  metadata = [n for n in generated if any(k in n.lower() for k in ("max", "den", "prob", "gmax"))]
  return {
    "generated_attention_programs": generated,
    "tile_programs": tile,
    "combine_programs": combine,
    "partial_programs": partial,
    "score_programs": score,
    "metadata_programs": metadata,
    "has_tile_score_max": any(n.startswith("flash_tile_score_max") for n in generated),
    "has_tile_prob": any(n.startswith("flash_tile_prob") for n in generated),
    "has_separate_flash_max": any(n == "flash_max_32" for n in generated),
    "has_separate_flash_prob": any(n == "flash_prob_32" for n in generated),
    "has_tile_program": bool(tile),
    "has_combine_program": bool(combine),
    "has_partial_or_score": bool(partial or score),
    "has_metadata": bool(metadata),
    "bundle_bound": bool(tile and combine and (partial or score) and metadata),
  }


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  from extra.qk_decode_search_gate import _setup_model, run_wd

  mode = "a2" if arm in ("a2", "a37") else "baseline"
  route = capture(mode)
  names = list(route["route_fire"]["program_node_names"])
  sig = _bundle_signature(names)
  out: dict[str, Any] = {"arm": arm, "route": route, "bundle_signature": sig}
  if arm != "a37" or sig["bundle_bound"]:
    m, _tok = _setup_model()
    out["wd"] = run_wd(m, ctxs=list(CTXS))
  return out


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


def _rows(owned: dict[str, Any], a2: dict[str, Any], a37: dict[str, Any]) -> list[dict[str, Any]]:
  if "wd" not in a37: return []
  rows = []
  for ctx in CTXS:
    o, b, x = owned["wd"][str(ctx)]["tok_s"], a2["wd"][str(ctx)]["tok_s"], a37["wd"][str(ctx)]["tok_s"]
    rows.append({
      "ctx": ctx,
      "owned_tok_s": o,
      "a2_tok_s": b,
      "a37_tok_s": x,
      "a37_vs_a2_pct": round(100.0 * x / b, 1) if b else None,
      "a37_vs_owned_pct": round(100.0 * x / o, 1) if o else None,
      "delta_vs_a2_tok_s": round(x - b, 1),
      "a2_spread_pct": a2["wd"][str(ctx)]["spread_pct"],
      "a37_spread_pct": a37["wd"][str(ctx)]["spread_pct"],
    })
  return rows


def build() -> dict[str, Any]:
  manifest = json.loads(MANIFEST.read_text())
  a37 = _run_child("a37", allow_failure=True)
  if a37.get("child_failed"):
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": "A3_7_FAIL__CAPTURE",
            "manifest": manifest, "route_clean": False, "bundle_bound": False, "rows": [], "a37": a37,
            "decision": "Do not promote. Fix capture failure before lifecycle benchmarking."}
  owned = _run_child("owned")
  a2 = _run_child("a2")
  route_clean = _route_clean(owned, a37)
  sig = a37["bundle_signature"]
  rows = _rows(owned, a2, a37)
  transfer_rows = [r for r in rows if r["a37_tok_s"] > r["a2_tok_s"]]
  if not route_clean:
    verdict = "A3_7_FAIL__ROUTE_OR_TOKEN_OR_MATERIALIZATION"
  elif not sig["bundle_bound"]:
    verdict = "A3_7_FAIL__TILE_PROB_NOT_BOUND"
  elif not sig["has_tile_score_max"] or not sig["has_tile_prob"] or sig["has_separate_flash_max"] or sig["has_separate_flash_prob"]:
    verdict = "A3_7_FAIL__PAYLOAD_NOT_ACTIVE"
  elif len(transfer_rows) >= 2:
    verdict = "A3_7_TILE_PROB_TRANSFERS"
  elif rows:
    verdict = "A3_7_TILE_PROB_NO_TRANSFER"
  else:
    verdict = "A3_7_TILE_PROB_INCONCLUSIVE"
  return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict, "manifest": manifest,
          "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_TILE_PROB": "1"},
          "route_clean": route_clean, "bundle_bound": bool(sig["bundle_bound"]), "rows": rows,
          "owned": owned, "a2": a2, "a37": a37,
          "decision": "Promote only on W==D transfer; otherwise move to partial PV payload."}


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_A37_TILE_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_A37_TILE_ARM", "owned"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-7-tile-prob-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("A3_7_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
