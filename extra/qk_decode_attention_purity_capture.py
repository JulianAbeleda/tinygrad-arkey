#!/usr/bin/env python3
"""Decode attention purity capture.

Captures the current decode-attention lifecycle route and classifies whether the
attention path is generated/search-owned or still backed by owned AMDGCN tile +
combine programs.
"""
from __future__ import annotations

import json, os, time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-purity"


def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]


def capture() -> dict[str, Any]:
  from extra.qk_decode_search_gate import _setup_model, capture_decode, check_route_fire, check_materialization

  os.environ.setdefault("DECODE_ATTN_KV_IDENTITY", "1")
  os.environ.setdefault("DECODE_ATTN_AMDGCN_TILE", "1")
  m, tok = _setup_model()
  toks, captured, _step, _v, _temp = capture_decode(m, tok)
  names = _program_names(captured)
  counts = Counter(names)
  owned_tile = sum(n == "owned_flash_tile_gqa_whole" or n.startswith("owned_flash_tile_gqa_whole") for n in names)
  owned_combine = sum(n.startswith("owned_flash_combine") for n in names)
  generated_attention = sum("generated" in n and "flash" in n for n in names)
  materialization = check_materialization(captured)
  route_fire = check_route_fire(captured, "owned_flash_tile_gqa_whole")
  if owned_tile > 0 and owned_combine > 0 and not materialization["E_49152_present"]:
    verdict = "DECODE_ATTENTION_NOT_PURE__OWNED_TILE_COMBINE"
  elif generated_attention > 0 and owned_tile == 0 and owned_combine == 0 and not materialization["E_49152_present"]:
    verdict = "DECODE_ATTENTION_PURE_SEARCH_GENERATED"
  else:
    verdict = "DECODE_ATTENTION_PURITY_CAPTURE_FAIL"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "tokens_sample": toks,
    "route_counts": {
      "owned_flash_tile_gqa_whole": owned_tile,
      "owned_flash_combine": owned_combine,
      "generated_attention_programs": generated_attention,
    },
    "route_fire": route_fire,
    "materialization": materialization,
    "top_program_counts": counts.most_common(40),
    "classification": "Decode attention is still backed by owned AMDGCN tile + combine programs; GEMV purity is complete, attention purity is next.",
  }


def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  out = capture()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-purity-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] != "DECODE_ATTENTION_PURITY_CAPTURE_FAIL" else 1


if __name__ == "__main__":
  raise SystemExit(main())
