"""Regression pin for P0-3 (LDS identity replacement / conflation).

# PINS P0-3 (fixed in C1/C2)

Per finding P0-3 in `pure-register-direct-l2-completion-scope-20260712.md`, the
pair generator creates DISTINCT direct-L2 and LDS candidate identities that must
be preserved end to end. Two current defects violate this:

1. `attn_qo_direct_l2_adapter_20260712.make_benchmark_callback` writes the shared
   top-level (direct) `canonical_identity` into BOTH benchmark rows, so the LDS
   row loses its own identity and carries the direct identity instead.
2. `pure_register_direct_l2_decision._blockers` validates only the direct
   (`left`) `canonical_identity` and never requires the LDS identity.

These tests are CPU-only (generation, adapter join, and decision validation only;
no device/runtime import). They FAIL against current code and will pass once the
distinct LDS identity is preserved and required.
"""
from __future__ import annotations

from extra.qk.prefill.attn_qo_l2_lds_pair_generator_20260712 import generate_pair
from extra.qk.prefill.attn_qo_direct_l2_adapter_20260712 import (
  make_benchmark_callback, prepare_exact_pair,
)
from extra.qk.prefill.pure_register_direct_l2_decision import decide

SHAPE = {"m": 512, "n": 4096, "k": 4096}


def _prepared_pair() -> dict:
  pair = generate_pair()
  direct, lds = pair["candidates"]["direct_l2"], pair["candidates"]["lds"]
  assert direct["canonical_identity"] != lds["canonical_identity"], (
    "generator precondition: direct and LDS candidate identities must be distinct")
  prepared = prepare_exact_pair(
    direct_payload=direct["payload"], lds_payload=lds["payload"],
    direct_binary_sha256="b" * 64, lds_binary_sha256="c" * 64, pair_key=pair["pair_key"])
  assert prepared["status"] == "prepared", prepared.get("blockers")
  return prepared


def test_lds_benchmark_row_keeps_its_own_identity():
  prepared = _prepared_pair()
  direct_id = prepared["candidates"]["direct_l2"]["canonical_identity"]
  lds_id = prepared["candidates"]["lds"]["canonical_identity"]

  callback = make_benchmark_callback(prepared, lambda storage, phase, index: {"samples_ms": [1.0]})
  rows = callback({"canonical_identity": prepared["canonical_identity"]})

  # PINS P0-3: the direct identity must NOT be written into the LDS row.
  assert rows["lds"]["canonical_identity"] != direct_id, (
    "LDS benchmark row was stamped with the DIRECT candidate identity "
    f"({direct_id[:12]}...); the distinct LDS identity was lost")
  assert rows["lds"]["canonical_identity"] == lds_id, (
    "LDS benchmark row must preserve the distinct LDS candidate identity "
    f"{lds_id[:12]}..., got {rows['lds']['canonical_identity'][:12]}...")


def test_decision_gate_requires_the_lds_identity():
  # A pair valid in every respect except a missing/invalid LDS canonical_identity.
  common_env = {"commit": "82f5a586a", "target": "gfx1100"}
  counters = {group: {"status": "live"} for group in ("l2", "memory", "compute")}

  def row(storage, binary, identity):
    return {"role": "attn_qo", "shape": SHAPE, "pair_key": "semantic-v1",
            "environment": common_env, "storage": storage, "binary_sha256": binary,
            "canonical_identity": identity, "artifact": {"status": "pass"},
            "correctness": {"status": "pass"}, "samples_ms": [10.0] * 12, "counters": counters}

  pair = {"direct_l2": row("direct_l2", "b" * 64, "d" * 64),
          "lds": row("lds", "c" * 64, None)}  # LDS identity is absent/invalid.
  report = decide(pair)

  # PINS P0-3: the gate must fail closed on a missing LDS identity.
  assert report["status"] == "blocked", (
    "decision gate accepted a pair with a missing LDS canonical_identity; "
    f"it does not require the LDS identity (decision={report.get('decision')!r})")
  assert any("identit" in reason.lower() for reason in report["blockers"]), report["blockers"]
