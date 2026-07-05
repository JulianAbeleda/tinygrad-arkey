#!/usr/bin/env python3
"""Synthetic 14B role-shape classifier for the Q4_K/Q8_1 tiled WMMA route.

This gate is intentionally honest: Phase 2 proves one bounded tile, but the full role-shape lowering is not implemented
until a direct tiled scheduler/codegen path exists. Passing this gate means the role shapes are enumerated, bounded RAW
requirements are explicit, and the current blocker is classified rather than silently falling back.
"""
from __future__ import annotations

import json
from typing import Any

from tinygrad.llm.generated_candidates import select_generated_candidate
from tinygrad.llm.quant_specs import activation_spec, quant_spec
from tinygrad.llm.runtime_specs import RuntimeOpSpec

from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill

EXPECTED_CANDIDATE = "quant_linear_prefill.q4k_int8_wmma_tiled_substrate"
ROLE_SHAPES = (
  ("attn_qo", 5120, 5120),
  ("attn_kv", 1024, 5120),
  ("ffn_gate_up", 17408, 5120),
  ("ffn_down", 5120, 17408),
)


def _candidate_selection(role:str, n:int, k:int) -> dict[str, Any]:
  op = RuntimeOpSpec("QuantizedLinear", "prefill", role, {"M": 512, "N": n, "K": k},
                     quant_spec("Q4_K").tensor_spec(), activation_spec("Q8_1").activation_spec(),
                     lowering_strategy="iu8_wmma_tiled_grouped_dot")
  return select_generated_candidate(op, preferred=(EXPECTED_CANDIDATE,)).to_json()


def _row(role:str, n:int, k:int) -> dict[str, Any]:
  spec = describe_q4k_int8_wmma_tiled_prefill(n, k, 512, role=role, m_tile=16, n_tile=16, group_tile=1)
  candidate = _candidate_selection(role, n, k)
  return {"role": role, "m": spec.m, "n": spec.n, "k": spec.k, "groups": spec.groups,
          "candidate": candidate,
          "tile": {"m_tile": spec.m_tile, "n_tile": spec.n_tile, "group_tile": spec.group_tile,
                   "live_raw_elems": spec.live_raw_elems,
                   "forbidden_full_raw_shape": [spec.groups, spec.m, spec.n],
                   "forbidden_full_raw_elems": spec.forbidden_full_raw_elems},
          "class": "blocked.full_route_lowering_missing",
          "reason": "Phase-2 one-tile emitter is correct, but no direct tiled full-role lowering exists yet; route must not fall through."}


def build() -> dict[str, Any]:
  rows = [_row(role, n, k) for role, n, k in ROLE_SHAPES]
  selected_ok = all(r["candidate"]["status"] == "selected" and
                    r["candidate"]["candidate"]["candidate_id"] == EXPECTED_CANDIDATE for r in rows)
  bounded_ok = all(r["tile"]["live_raw_elems"] <= r["tile"]["m_tile"] * r["tile"]["n_tile"] * r["tile"]["group_tile"]
                   for r in rows)
  classified = all(r["class"] == "blocked.full_route_lowering_missing" for r in rows)
  ok = selected_ok and bounded_ok and classified
  return {"schema": "q4k_wmma_tiled_role_shape_gate.v1",
          "scope": "synthetic 14B Q4_K/Q8_1 tiled WMMA role-shape classification",
          "verdict": "Q4K_WMMA_TILED_ROLE_SHAPES_BLOCKED_FULL_ROUTE" if ok else "Q4K_WMMA_TILED_ROLE_SHAPES_FAIL",
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "roles": rows,
          "next_required": "implement direct tiled scheduler/codegen lowering that maps role shapes to bounded tiles without route-local WMMA source/asm or default fallback"}


if __name__ == "__main__":
  out = build()
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPES_BLOCKED_FULL_ROUTE" else 1)
