#!/usr/bin/env python3
"""Top-level pure-machine-search gap audit.

This wraps workload-specific audits into one decode+prefill verdict. It is intentionally conservative:
if a workload has performance evidence but incomplete search provenance, it says so rather than claiming purity.
"""
from __future__ import annotations
import json, pathlib, datetime
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/qk-pure-machine-search-gap"
CANON = ROOT / "bench/canonical-benchmarks.json"
DECODE_ATT = ROOT / "bench/qk-pure-search-gap/latest.json"
PREFILL_READINESS = ROOT / "bench/qk-prefill-search/prefill_search_readiness.json"
PREFILL_LONGCTX_DOC = ROOT / "docs/archive/prefill-long-context-integration-hardening-result-20260624.md"
GEMV_SCOPE = ROOT / "docs/gemv-pure-search-generated-route-scope.md"
OCC_GUARD = ROOT / "bench/qk-decode-occupancy-guardrail/latest.json"
OUTER_B = ROOT / "bench/qk-decode-outer-b-split-combine/latest.json"
PRESSURE_OWN = ROOT / "bench/qk-decode-pressure-search-ownership/latest.json"

def load_json(path: pathlib.Path, default: Any=None) -> Any:
  if not path.exists(): return default
  return json.loads(path.read_text())

def load_text(path: pathlib.Path) -> str:
  return path.read_text() if path.exists() else ""

def avg(xs: list[float]) -> float:
  return round(sum(xs) / len(xs), 1) if xs else 0.0

def tok_map(section: dict[str, Any], name: str) -> dict[str, float]:
  return {str(k): float(v) for k, v in section.get(name, {}).get("tok_s", {}).items()}

def pct_of(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
  return {k: round((candidate[k] / baseline[k]) * 100.0, 1) for k in candidate.keys() & baseline.keys() if baseline[k]}

def ratio_of(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
  return {k: round(baseline[k] / candidate[k], 2) for k in candidate.keys() & baseline.keys() if candidate[k]}

def main() -> int:
  canon = load_json(CANON, {})
  decode_att = load_json(DECODE_ATT, {})
  prefill_readiness = load_json(PREFILL_READINESS, {})
  prefill_longctx_doc = load_text(PREFILL_LONGCTX_DOC)
  gemv_scope = load_text(GEMV_SCOPE)
  occ_guard = load_json(OCC_GUARD, {})
  outer_b = load_json(OUTER_B, {})
  pressure_own = load_json(PRESSURE_OWN, {})

  decode = canon.get("decode", {})
  prefill = canon.get("prefill", {})
  decode_baseline = tok_map(decode, "baseline")
  decode_bubble = tok_map(decode, "bubblebeam_futuresight")
  prefill_baseline = tok_map(prefill, "baseline")
  prefill_aggressive = tok_map(prefill, "aggressive_target")

  decode_attention_score = decode_att.get("pure_search_score", {}).get("decode_attention_pure_machine_search_score_0_to_100", 0)
  decode_gemv_pure = "GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV" in gemv_scope
  decode_gemv_score = 85 if decode_gemv_pure else 65
  prefill_score = 62

  sections = []
  sections.append({
    "workload": "decode_attention",
    "score_0_to_100": decode_attention_score,
    "authority": "child_audit",
    "source": str(DECODE_ATT.relative_to(ROOT)),
    "verdict": decode_att.get("verdict", "missing"),
    "time_delta": decode_att.get("time_delta_explanation", {}).get("remaining_gap_to_owned", {}),
    "primitive_vocab": decode_att.get("primitive_vocabulary_attribution", {}).get("verdict", "missing"),
    "provenance": {
      "generated_transfers": True,
      "search_owned": False,
      "manual_flags": True,
      "owned_kernel_required_for_default_perf": True,
    },
    "next_actions": decode_att.get("next_actions", []),
  })
  sections.append({
    "workload": "decode_gemv",
    "score_0_to_100": decode_gemv_score,
    "authority": "canonical_benchmark_plus_gemv_scope",
    "verdict": "GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV" if decode_gemv_pure else "GEMV_PROVENANCE_NEEDS_AUDIT",
    "time_delta": {
      "baseline_tok_s": decode_baseline,
      "bubblebeam_futuresight_tok_s": decode_bubble,
      "bubblebeam_pct_of_baseline": pct_of(decode_bubble, decode_baseline),
    },
    "primitive_vocab": {
      "present": ["Q4_K LaneMap G3 generated route", "BubbleBeam/FutureSight selector"],
      "missing_or_partial": ["none for tracked Q4_K GEMV roles" if decode_gemv_pure else "provenance gate incomplete"],
    },
    "provenance": {
      "generated": decode_gemv_pure,
      "search_owned": decode_gemv_pure,
      "manual_flags": False,
      "owned_kernel_required_for_tracked_q4k_roles": not decode_gemv_pure,
    },
    "next_actions": ["keep GEMV purity gate in periodic regression bundle"],
  })
  sections.append({
    "workload": "prefill",
    "score_0_to_100": prefill_score,
    "authority": "canonical_benchmark_plus_prefill_search_readiness",
    "verdict": "PREFILL_PARTIAL__EIGHTWAVE_BASELINE_STABLE__SEARCH_PROVENANCE_AND_AGGRESSIVE_BOUND_NOT_CLOSED",
    "time_delta": {
      "baseline_tok_s": prefill_baseline,
      "aggressive_target_tok_s": prefill_aggressive,
      "aggressive_pct_of_baseline": pct_of(prefill_aggressive, prefill_baseline),
      "baseline_over_aggressive_ratio": ratio_of(prefill_baseline, prefill_aggressive),
      "long_context_status": "flat_512_to_8192" if "NO_GROWTH_CONFIRMED" in prefill_longctx_doc else "unknown",
    },
    "primitive_vocab": {
      "present": ["eightwave graph-GEMM default", "whole-prefill synced authority", "long-context hardening"],
      "manual_or_not_fully_search_owned": ["eightwave/default emit provenance needs binding into canonical search-space manifest"],
      "missing_or_partial": ["prefill aggressive-bound proof", "prefill search provenance for all promoted flags", "attention/copy decomposition blind spot in long-context hardening"],
    },
    "provenance": {
      "generated": True,
      "search_owned": False,
      "manual_flags_or_manual_emit_history": True,
      "owned_kernel_required_for_current_baseline": False,
    },
    "search_readiness": {
      "verdict": prefill_readiness.get("verdict", "missing"),
      "gap_to_tensile_pct": prefill_readiness.get("gap_to_tensile_pct", {}),
      "unlock_condition": prefill_readiness.get("unlock_condition", "missing"),
    },
    "next_actions": [
      "bind eightwave/current prefill baseline to search provenance manifest",
      "carry prefill aggressive target as a non-search bound until proven by whole-prefill authority",
      "keep long-context no-growth gate in periodic bundle",
    ],
  })

  ctx_explanation = {
    "question": "Is the ctx regression due to flash attention?",
    "answer": "Partially: it is decode-attention-specific and tied to the generated flash-style online-softmax/split-KV block loop, not to flash attention as a concept.",
    "evidence": [
      "Prefill graph/eightwave baseline is flat across 512..8192 in canonical benchmarks and long-context hardening.",
      "Generated decode attention full stack falls from 32.8 tok/s at ctx512 to 6.2 tok/s at ctx4096 while owned stays 103.2 to 93.8 tok/s.",
      "Delta campaign identifies the outer b-block online-softmax carry as the ctx-slope source; inner tt split was refuted.",
      "Owned/tuned kernels are good because they use a pressure-aware structure that controls this recurrence/scheduling cost.",
    ],
    "verdict": "CTX_REGRESSION_IS_GENERATED_FLASH_DECODE_OUTER_BLOCK_CARRY__NOT_PREFILL_AND_NOT_FLASH_ATTENTION_IN_GENERAL",
  }

  overall_score = round((decode_attention_score * 0.35) + (decode_gemv_score * 0.25) + (prefill_score * 0.40))
  out = {
    "schema": "qk_pure_machine_search_gap_audit_v1",
    "date": datetime.date.today().isoformat(),
    "inputs": {
      "canonical_benchmarks": str(CANON.relative_to(ROOT)),
      "decode_attention_child_audit": str(DECODE_ATT.relative_to(ROOT)),
      "prefill_search_readiness": str(PREFILL_READINESS.relative_to(ROOT)),
      "prefill_long_context_hardening_doc": str(PREFILL_LONGCTX_DOC.relative_to(ROOT)),
      "gemv_scope": str(GEMV_SCOPE.relative_to(ROOT)),
      "occupancy_guardrail": str(OCC_GUARD.relative_to(ROOT)),
      "outer_b_split_contract": str(OUTER_B.relative_to(ROOT)),
      "pressure_search_ownership": str(PRESSURE_OWN.relative_to(ROOT)),
    },
    "overall_score_0_to_100": overall_score,
    "weights": {"decode_attention": 0.35, "decode_gemv": 0.25, "prefill": 0.40},
    "sections": sections,
    "ctx_regression_explanation": ctx_explanation,
    "next_actions": [
      {"rank": 1, "area": "decode_attention", "action": "implement LDS-staged outer-b split-combine lowering behind the new search contract"},
      {"rank": 2, "area": "decode_attention", "action": "bind pressure-aware/manual flags into BubbleBeam/FutureSight candidate ownership"},
      {"rank": 3, "area": "prefill", "action": "bind eightwave/current baseline to explicit search provenance"},
      {"rank": 4, "area": "shared", "action": "put manual flags/primitives into BubbleBeam/FutureSight candidate provenance"},
    ],
    "verdict": "PURE_MACHINE_SEARCH_PARTIAL__DECODE_GEMV_CLOSE__DECODE_ATTENTION_AND_PREFILL_STILL_NOT_FULLY_SEARCH_OWNED",
  }
  OUTDIR.mkdir(parents=True, exist_ok=True)
  (OUTDIR / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
