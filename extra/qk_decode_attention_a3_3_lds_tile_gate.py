#!/usr/bin/env python3
"""A3.3 LDS/tile lifecycle gate for generated decode attention.

This is intentionally a route/lifecycle gate first. A standalone generated LDS
flash-attention kernel exists in extra/gemm/amd_flash_attention.py, but it is not
promotable unless the decode route can bind an LDS/tile candidate while keeping
A2's whole-cache/no-E_49152 lifecycle properties.
"""
from __future__ import annotations

import json, os, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-a3-3-lds-tile"
CTXS = (512, 1024, 2048, 4096)


def _env_for_arm(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_A33_LDS_CHILD": "1", "QK_A33_LDS_ARM": arm}
  env["DECODE_ATTN_GENERATED_SKELETON"] = "0"
  env["DECODE_ATTN_SCORE_VDOT2"] = "0"
  env["DECODE_ATTN_SCORE_XLANE"] = "0"
  env["WARP_REDUCE_LOWERING"] = "0"
  env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1" if arm in ("a2", "a33") else "0"
  env["DECODE_ATTN_LDS_TILE"] = "1" if arm == "a33" else "0"
  return env


def _run_child(arm: str, allow_failure: bool=False) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=_env_for_arm(arm),
                     capture_output=True, text=True)
  if r.returncode != 0:
    if allow_failure:
      return {"arm": arm, "child_failed": True, "returncode": r.returncode,
              "stdout_tail": r.stdout[-4000:], "stderr_tail": r.stderr[-8000:]}
    raise RuntimeError(f"{arm} child failed rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  raise RuntimeError(f"{arm} child did not emit JSON\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")


def _standalone_lds_evidence() -> dict[str, Any]:
  p = ROOT / "extra/gemm/amd_flash_attention.py"
  src = p.read_text()
  return {
    "path": str(p.relative_to(ROOT)),
    "exists": p.exists(),
    "has_local_addrspace": "AddrSpace.LOCAL" in src,
    "has_barrier": "barrier" in src,
    "has_wmma": "SHAPED_WMMA" in src,
    "has_cross_lane": "ds_bpermute" in src or "warp_reduce" in src,
    "classification": "standalone_generated_lds_flash_attention_not_decode_bound",
  }


def _attention_programs(names: list[str]) -> list[str]:
  return [n for n in names if n.startswith("flash_")]


def _lds_tile_program_present(names: list[str]) -> bool:
  # Current expected future naming. This gate is intentionally conservative:
  # generic A2 programs are not counted as LDS/tile lifecycle candidates.
  return any(("lds" in n.lower() or "tile" in n.lower()) and n.startswith("flash_") for n in names)


def _child(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  from extra.qk_decode_search_gate import _setup_model, run_wd

  mode = "a2" if arm in ("a2", "a33") else "baseline"
  route = capture(mode)
  names = route["route_fire"]["program_node_names"]
  out: dict[str, Any] = {
    "arm": arm,
    "route": route,
    "attention_programs": _attention_programs(names),
    "lds_tile_program_present": _lds_tile_program_present(names),
  }
  if arm != "a33" or out["lds_tile_program_present"]:
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


def _rows(owned: dict[str, Any], a2: dict[str, Any], a33: dict[str, Any]) -> list[dict[str, Any]]:
  if "wd" not in a33: return []
  rows = []
  for ctx in CTXS:
    o, b, x = owned["wd"][str(ctx)]["tok_s"], a2["wd"][str(ctx)]["tok_s"], a33["wd"][str(ctx)]["tok_s"]
    rows.append({
      "ctx": ctx,
      "owned_tok_s": o,
      "a2_tok_s": b,
      "a33_tok_s": x,
      "a33_vs_a2_pct": round(100.0 * x / b, 1) if b else None,
      "a33_vs_owned_pct": round(100.0 * x / o, 1) if o else None,
      "delta_vs_a2_tok_s": round(x - b, 1),
      "a2_spread_pct": a2["wd"][str(ctx)]["spread_pct"],
      "a33_spread_pct": a33["wd"][str(ctx)]["spread_pct"],
    })
  return rows


def build() -> dict[str, Any]:
  standalone = _standalone_lds_evidence()
  a33 = _run_child("a33", allow_failure=True)
  if a33.get("child_failed"):
    return {
      "date": "2026-06-25",
      "timestamp": time.strftime("%Y%m%d-%H%M%S"),
      "verdict": "A3_3_FAIL__LDS_TILE_CAPTURE",
      "candidate": "decode_attention_generated_lds_tile_lifecycle",
      "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_LDS_TILE": "1"},
      "standalone_lds_evidence": standalone,
      "route_clean": False,
      "rows": [],
      "a33": a33,
      "decision": "Do not promote. Classify capture failure before W==D.",
    }
  owned = _run_child("owned")
  a2 = _run_child("a2")
  route_clean = _route_clean(owned, a33)
  lds_present = bool(a33["lds_tile_program_present"])
  rows = _rows(owned, a2, a33)
  if not route_clean:
    verdict = "A3_3_FAIL__ROUTE_OR_TOKEN_OR_MATERIALIZATION"
  elif not lds_present:
    verdict = "A3_3_BLOCKED_BY_ROUTE_BINDING"
  elif rows and sum(1 for r in rows if r["a33_tok_s"] > r["a2_tok_s"]) >= 2:
    verdict = "A3_3_LDS_TILE_TRANSFERS"
  elif rows:
    verdict = "A3_3_LDS_TILE_NO_TRANSFER"
  else:
    verdict = "A3_3_LDS_TILE_INCONCLUSIVE"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "candidate": "decode_attention_generated_lds_tile_lifecycle",
    "flags": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_LDS_TILE": "1"},
    "standalone_lds_evidence": standalone,
    "route_clean": route_clean,
    "lds_tile_program_present": lds_present,
    "rows": rows,
    "owned": owned,
    "a2": a2,
    "a33": a33,
    "decision": ("Promote nothing. Existing generated LDS flash-attention evidence is standalone; decode route binding for an LDS/tile lifecycle candidate is missing."
                 if verdict == "A3_3_BLOCKED_BY_ROUTE_BINDING" else
                 "Use verdict and rows to decide promotion or next blocker."),
  }


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_A33_LDS_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_A33_LDS_ARM", "owned"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-a3-3-lds-tile-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].startswith("A3_3_FAIL") else 1


if __name__ == "__main__":
  raise SystemExit(main())
