#!/usr/bin/env python3
"""G3.0 codegen-shape capture for Q4_K GEMV purity.

Builds one decode context for owned, lane-partition bridge, and G2 LaneMap arms.
It records captured PROGRAM metadata and source-shape evidence so the next codegen
work targets the actual lowering mismatch rather than rerunning speed probes.
"""
from __future__ import annotations

import json, os, time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-gemv-g3-codegen-capture"
CTX = 512
MAXC = 640
ARMS = {
  "owned": {"Q4K_GEMV_SCHEDULER": "0"},
  "bridge": {"Q4K_GEMV_SCHEDULER": "4"},
  "g2_lanemap": {"Q4K_GEMV_SCHEDULER": "5"},
  "g3_lanemap_codegen": {"Q4K_GEMV_SCHEDULER": "6"},
}
CLEAR_ENV = ("Q4K_GEMV_SCHEDULER", "MV_ROWS_PER_THREAD", "WARP_REDUCE_LOWERING", "BUBBLEBEAM_FUTURESIGHT", "BEAM_COALESCE")


def _program_name(u) -> str:
  return getattr(u.src[0].arg, "name", str(u.src[0].arg))


def _program_source(p) -> str:
  if len(p.src) > 3 and p.src[3].op.name == "SOURCE" and isinstance(p.src[3].arg, str): return p.src[3].arg
  return ""


def _program_record(call) -> dict[str, Any]:
  p = call.src[0]
  info = p.arg
  src = _program_source(p)
  return {
    "name": getattr(info, "name", str(info)),
    "global_size": list(getattr(info, "global_size", ()) or ()),
    "local_size": list(getattr(info, "local_size", ()) or ()),
    "source_len": len(src),
    "source_flags": {
      "has_q4k_name": "q4k" in getattr(info, "name", ""),
      "has_lane_partition_name": "lane_partition" in getattr(info, "name", ""),
      "source_mentions_q4k": "q4k" in src,
      "source_has_shift": ">>" in src or "rshift" in src,
      "source_has_mask_15": "15u" in src or "0xf" in src or "& 15" in src,
      "source_has_local_id": "get_local_id" in src,
      "source_has_barrier": "barrier" in src or "s_barrier" in src,
      "source_has_ds_bpermute": "bpermute" in src,
    },
    "source_keyword_counts": {
      "uint": src.count("unsigned int") + src.count("uint"),
      "load_like": src.count("data") + src.count("*") + src.count("["),
      "shift": src.count(">>"),
      "mask": src.count("&"),
      "for_loop": src.count("for ("),
    },
  }


def _capture_arm(arm:str, env:dict[str, str]) -> dict[str, Any]:
  from tinygrad import Tensor, UOp, TinyJit
  from tinygrad.helpers import getenv
  from extra.qk.harness_contract import DEFAULT_MODEL
  from extra.llm.generate import load_model_and_tokenizer

  for k in CLEAR_ENV: os.environ.pop(k, None)
  for k, v in env.items(): os.environ[k] = v
  getenv.cache_clear()

  m, tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 120)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  step = TinyJit(m.forward); out = Tensor([[int(ids[CTX])]], dtype="int32").contiguous()
  tokens = []
  for i in range(8):
    out = step(out, v.bind(CTX + i), temp).realize()
    try: tokens.append(int(out.item()))
    except Exception: pass

  programs = [_program_record(u) for u in step.captured.linear.toposort()
              if u.op.name == "CALL" and len(u.src) and u.src[0].op.name == "PROGRAM"]
  names = [p["name"] for p in programs]
  name_counts = Counter(names)
  gateup_owned = sum(n.startswith("q4k_gemv_warp_12288") for n in names)
  gateup_bridge = sum(n.startswith("q4k_lane_partition_gemv_12288") for n in names)
  gateup_g3 = sum(n.startswith("q4k_g3_lanemap_gemv_12288") for n in names)
  q4k_named = [n for n in names if "q4k" in n]
  named_gateup = [n for n in names if n.startswith(("q4k_gemv_warp_12288", "q4k_lane_partition_gemv_12288"))]
  source_summary = {
    "programs_with_shift": sum(p["source_flags"]["source_has_shift"] for p in programs),
    "programs_with_mask_15": sum(p["source_flags"]["source_has_mask_15"] for p in programs),
    "programs_with_local_id": sum(p["source_flags"]["source_has_local_id"] for p in programs),
    "programs_with_ds_bpermute": sum(p["source_flags"]["source_has_ds_bpermute"] for p in programs),
  }
  interesting = [p for p in programs if p["name"].startswith(("q4k_gemv_warp_12288", "q4k_lane_partition_gemv_12288"))]
  if arm == "g2_lanemap":
    interesting += [p for p in programs if p["source_flags"]["source_has_shift"] and p["source_flags"]["source_has_mask_15"]][:12]
  return {
    "env": env,
    "program_count": len(programs),
    "unique_program_count": len(name_counts),
    "route_counts": {
      "owned_gateup": gateup_owned,
      "lane_partition_gateup": gateup_bridge,
      "g3_lanemap_gateup": gateup_g3,
      "q4k_named_programs": len(q4k_named),
      "named_gateup_programs": len(named_gateup),
    },
    "top_program_counts": name_counts.most_common(24),
    "q4k_named_programs_sample": q4k_named[:24],
    "source_summary": source_summary,
    "interesting_programs": interesting[:24],
    "tokens_sample": tokens[:8],
  }


def _decide(arms:dict[str, Any]) -> tuple[str, str]:
  owned_ok = arms["owned"]["route_counts"]["owned_gateup"] > 0
  bridge_ok = arms["bridge"]["route_counts"]["lane_partition_gateup"] > 0 and arms["bridge"]["route_counts"]["owned_gateup"] == 0
  g2_clean = arms["g2_lanemap"]["route_counts"]["owned_gateup"] == 0 and arms["g2_lanemap"]["route_counts"]["lane_partition_gateup"] == 0
  g2_has_no_named_gateup = arms["g2_lanemap"]["route_counts"].get("named_gateup_programs", 0) == 0
  g3_ok = arms.get("g3_lanemap_codegen", {}).get("route_counts", {}).get("g3_lanemap_gateup", 0) > 0 and arms["g3_lanemap_codegen"]["route_counts"]["owned_gateup"] == 0 and arms["g3_lanemap_codegen"]["route_counts"]["lane_partition_gateup"] == 0
  if owned_ok and bridge_ok and g3_ok:
    return "G3_ONE_WORD_PER_LANE_LOWERING_HOOK_PRESENT", "G3 LaneMap codegen arm now emits a named wave32 gate/up program without owned warp or lane-partition bridge attribution. W==D decides whether it is promotable or only a generated custom-shape probe."
  if owned_ok and bridge_ok and g2_clean and g2_has_no_named_gateup:
    return "G3_CODEGEN_MISMATCH_CAPTURED", "Owned and bridge arms each have a named wave32 gate/up program, while G2 LaneMap is route-clean but has no named/generated gate/up program shape; it lowers into generic Tensor programs instead of one-word-per-lane in-register dequant/reduce."
  return "G3_CAPTURE_FAIL", "Capture did not observe the expected route attribution pattern."


def main() -> int:
  os.chdir(ROOT)
  OUT.mkdir(parents=True, exist_ok=True)
  arms = {arm: _capture_arm(arm, env) for arm, env in ARMS.items()}
  verdict, interpretation = _decide(arms)
  out = {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "scope": "G3.0 codegen mismatch capture",
    "ctx": CTX,
    "verdict": verdict,
    "arms": arms,
    "interpretation": interpretation,
    "next_action": "Implement G3.1 one-word-per-lane lowering hook; first target is a generated gate/up program shape with wave32 row ownership and packed uint32 word load bound to word_col.",
  }
  stamped = OUT / f"g3-codegen-capture-{out['timestamp']}.json"
  latest = OUT / "latest.json"
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  latest.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if verdict == "G3_CODEGEN_MISMATCH_CAPTURED" else 1


if __name__ == "__main__":
  raise SystemExit(main())
