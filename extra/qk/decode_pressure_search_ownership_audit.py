#!/usr/bin/env python3
"""Audit pressure-aware scheduling/search ownership for decode attention."""
from __future__ import annotations
import json, pathlib, datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
PURE = ROOT / "bench/qk-pure-search-gap/latest.json"
OCC = ROOT / "bench/qk-decode-occupancy-guardrail/latest.json"
OUTER = ROOT / "bench/qk-decode-outer-b-split-combine/latest.json"

def load(p: pathlib.Path): return json.loads(p.read_text()) if p.exists() else {}

def build() -> dict:
  pure, occ, outer = load(PURE), load(OCC), load(OUTER)
  manual = ["DECODE_ATTN_BLOCK_TILE", "DECODE_STAGE_COALESCE", "COALESCED_LOAD_LOWERING", "SCHED_UNROLL", "SCHED_LIST", "DECODE_FAST_EXP2"]
  out = {
    "schema": "qk_decode_pressure_search_ownership_audit_v1",
    "date": datetime.date.today().isoformat(),
    "inputs": {"pure_gap": str(PURE.relative_to(ROOT)), "occupancy": str(OCC.relative_to(ROOT)), "outer_b_contract": str(OUTER.relative_to(ROOT))},
    "manual_winning_flags": manual,
    "search_owned_now": ["Audit.split_aware_hotloop_oracle", "ResourceModel.occupancy_guardrail", "OuterBlockLoop.lds_staged_split_combine.search_contract"],
    "not_search_owned_yet": ["Math.fast_exp2_valid_domain", "Sched.recurrence_unroll_list", "OuterBlockLoop.lds_staged_split_combine.lowering", "Scheduler.pressure_aware_latency_hiding"],
    "resource_guardrail": occ.get("verdict", "missing"),
    "outer_b_vocab": outer.get("verdict", "missing"),
    "decode_attention_verdict": pure.get("verdict", "missing"),
    "promotion_blocker": "manual winning flags and outer-b lowering are not BubbleBeam-owned",
    "verdict": "PRESSURE_SEARCH_OWNERSHIP_PARTIAL__GUARDRAIL_AND_VOCAB_PRESENT__LOWERING_AND_FLAG_BINDING_REMAIN"
  }
  return out

if __name__ == "__main__":
  import sys; sys.path.insert(0, str(ROOT))
  from extra.qk.gate_registry import run
  raise SystemExit(run("pressure_search_ownership"))
