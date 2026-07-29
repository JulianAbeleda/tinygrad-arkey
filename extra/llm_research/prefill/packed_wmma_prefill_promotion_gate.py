#!/usr/bin/env python3
"""Authority gate for the `packed_wmma_prefill_generated` manifest row.

docs/packed-wmma-14b-machine-search-claim-scope-20260725.md PHASE 1: the 14B packed-WMMA prefill route was
real, reproducible, and beating llama.cpp -- but invisible to the purity system (0 hits in ROUTES/
HOT_FAMILIES). This gate is what makes the registration auditable rather than decorative: it reads ONLY
pre-collected, transcribed evidence from docs/packed-wmma-14b-promotion-evidence-20260725.json, never
computes or invents a number, and FAILS CLOSED on any manifest-admitted shape lacking evidence -- same shape
as extra/llm_research/prefill/prefill_causal_tile_skip_promotion_gate.py and extra/llm_research/prefill/prefill_softmax_reduce_fuse_promotion_gate.py.

TWO THINGS THIS GATE IS SPECIFICALLY GUARDING AGAINST, because both were live risks during registration:

  1. Overclaiming provenance. The kernel genuinely is tinygrad-scheduler-generated (an ordinary
     `(a @ b.transpose()).schedule_linear()` compiled under Opt(OptOps.TC, 0, (-1,2,1)),
     extra/llm_research/prefill/current_prefill_execution_adapter.py:76-87) -- but PACKED_WMMA_GEOM
     (tinygrad/llm/packed_wmma_prefill.py) is a FROZEN table keyed by exact (quant, role, shape) rows whose geometry
     shape term, and its cited source file has never existed in this repo. `machine_authored_generated`
     would assert an emitter derives extents from descriptor fields; this table does not. This gate FAILS
     if the manifest row's provenance is ever anything other than exactly "tinygrad_scheduler_generated" --
     including a silent upgrade to "machine_authored_generated" without the Phase 2 work (in-repo
     reproducible search + shape-keyed geometry) that would actually earn it.
  2. Coverage overclaim. PACKED_WMMA_GEOM has no (Q6_K, attn_qo) or (Q6_K, ffn_gate_up) entry -- those
     combos silently decline to the direct-packed baseline. This gate requires evidence for EVERY
     shape_guards entry the manifest row actually admits, and nothing more; a shape_guards entry with no
     matching evidence is a FAIL, not a pass-by-omission.

This script does not flip any default or touch the manifest. It reports PASS/FAIL for informing a human's
decision only.

Run: PYTHONPATH=. python3 extra/llm_research/prefill/packed_wmma_prefill_promotion_gate.py
"""
from __future__ import annotations

import json
import pathlib
import sys

from extra.llm_research.route_manifest import ROUTES
from extra.llm_research.prefill.promotion_gate_common import load_evidence, fail, build_result, main as _common_main

GATE = "packed_wmma_prefill_promotion"
ROUTE_ID = "packed_wmma_prefill_generated"
FLAG = "TINYGRAD_PREFILL_PACKED_WMMA"
EXPECTED_PROVENANCE = "tinygrad_scheduler_generated"
EVIDENCE_PATH = pathlib.Path(__file__).resolve().parents[3] / "docs" / "packed-wmma-14b-promotion-evidence-20260725.json"
SCHEMA = "packed-wmma-14b-promotion-evidence.v1"


def _required_shapes() -> list[dict]:
  """Every (quant, role, shape) this route's manifest row actually claims -- derived from the manifest,
  never hardcoded, so a shape_guards edit is reflected here automatically."""
  guards = ROUTES[ROUTE_ID]["shape_guards"]
  out = []
  for g in guards:
    if not all(k in g for k in ("role", "M", "N", "K", "quant")): continue
    out.append({"quant": g["quant"], "role": g["role"], "shape": [g["M"], g["N"], g["K"]]})
  return out


def _check_canary(required: list[dict], evidence: dict) -> list[str]:
  fails = []
  canary = evidence.get("canary") or {}
  combos = canary.get("combos")
  if not isinstance(combos, list):
    return [f"shape {r['quant']}/{r['role']}/{r['shape']}: no canary combos recorded" for r in required]
  by_key = {(c.get("quant"), c.get("role"), tuple(c.get("shape") or ())): c for c in combos}
  for r in required:
    key = (r["quant"], r["role"], tuple(r["shape"]))
    entry = by_key.get(key)
    if entry is None:
      fails.append(f"shape {r['quant']}/{r['role']}/{r['shape']}: no matching canary evidence entry")
      continue
    if entry.get("passed") is not True:
      fails.append(f"shape {r['quant']}/{r['role']}/{r['shape']}: canary passed={entry.get('passed')!r}, expected True")
    if entry.get("max_abs") != 0.0:
      fails.append(f"shape {r['quant']}/{r['role']}/{r['shape']}: canary max_abs={entry.get('max_abs')!r}, expected 0.0")
  return fails


def _check_parity(evidence: dict) -> list[str]:
  fails = []
  parity = evidence.get("e2e_token_parity_14b") or {}
  if parity.get("status") != "PASS":
    fails.append(f"e2e_token_parity_14b: status={parity.get('status')!r}, expected PASS")
  if parity.get("route_live") is not True:
    fails.append("e2e_token_parity_14b: route_live is not True (must be measured with the packed-WMMA route ON, not TINYGRAD_PREFILL_PACKED_WMMA=0)")
  sdpa, fused = parity.get("sdpa"), parity.get("fused")
  if sdpa is None or fused is None or sdpa != fused:
    fails.append(f"e2e_token_parity_14b: sdpa={sdpa!r} fused={fused!r} do not match")
  return fails


def _check_throughput(evidence: dict) -> list[str]:
  fails = []
  tp = evidence.get("throughput") or {}
  by_ctx = tp.get("14b_tok_s_by_context") or {}
  if not by_ctx:
    fails.append("throughput: no 14b_tok_s_by_context recorded")
  delta = tp.get("llama_cpp_same_session_delta_pct") or {}
  for ctx in ("pp512", "pp1024", "pp2048", "pp4096"):
    if by_ctx.get(ctx) is None:
      fails.append(f"throughput: missing 14b_tok_s_by_context[{ctx}]")
    val = delta.get(ctx)
    if val is None or val <= 0:
      fails.append(f"throughput: llama_cpp_same_session_delta_pct[{ctx}]={val!r}, expected a positive measured margin")
  return fails


def evaluate() -> dict:
  route_row = ROUTES.get(ROUTE_ID)
  if route_row is None:
    return fail(GATE, f"manifest has no route {ROUTE_ID!r}")
  if route_row.get("provenance") != EXPECTED_PROVENANCE:
    return fail(GATE, f"{ROUTE_ID} provenance={route_row.get('provenance')!r}, expected exactly {EXPECTED_PROVENANCE!r} "
                       f"(PACKED_WMMA_GEOM is a frozen shape-blind table -- machine_authored_generated is not earned yet)")

  required = _required_shapes()
  if not required:
    return fail(GATE, f"{ROUTE_ID} shape_guards has no exact (role, M, N, K, quant) entries to check")

  evidence = load_evidence(EVIDENCE_PATH, SCHEMA, ROUTE_ID, FLAG)
  if evidence is None:
    return fail(GATE, f"evidence artifact missing or invalid: {EVIDENCE_PATH}", required_shapes=required)

  failures = []
  failures += _check_canary(required, evidence)
  failures += _check_parity(evidence)
  failures += _check_throughput(evidence)

  verdict = "PASS" if not failures else "FAIL"
  return build_result(
    gate=GATE, route_id=ROUTE_ID, flag=FLAG,
    provenance=route_row.get("provenance"),
    required_shapes=required,
    evidence_path=EVIDENCE_PATH,
    failures=failures,
    verdict=verdict,
    note=("all manifest-admitted shapes have complete, passing canary + e2e parity + throughput evidence" if verdict == "PASS"
          else "at least one manifest-admitted shape lacks complete promotion evidence; do NOT treat this row as fully proven"),
  )


def main() -> int:
  return _common_main(evaluate)


if __name__ == "__main__":
  sys.exit(main())
