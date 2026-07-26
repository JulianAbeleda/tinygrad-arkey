#!/usr/bin/env python3
"""Shared plumbing for the prefill promotion gates (packed_wmma_prefill_promotion_gate.py,
prefill_softmax_reduce_fuse_promotion_gate.py, prefill_causal_tile_skip_promotion_gate.py).

These three gates all encode the SAME evidence-loading and verdict-emitting rule: read one fixed,
pre-collected docs/*.json evidence artifact for a manifest route, fail closed if it is missing,
unparseable, or does not self-identify (via `_schema`/`route_id`/`flag`) as the evidence for exactly
this gate, assemble a result dict recording what was required and what failed, and print/exit on the
verdict. This module owns exactly that shared shell; it does NOT own any gate's route_id, flag,
evidence path, schema string, gate name, numeric thresholds, or shape/evidence-content checks -- those
stay in each gate file as per-route policy, even where two gates currently happen to share a value.

`extra/qk/prefill/pure_register_evaluation_gate.py` is a different rule (compile-artifact provenance,
no route_manifest, no main()) and deliberately does NOT use this module.
"""
from __future__ import annotations

import json
import pathlib


def load_evidence(path: pathlib.Path, schema: str, route_id: str, flag: str) -> dict | None:
  """Read `path` as JSON and return it iff it exists, parses, and self-identifies as the evidence for
  exactly this (schema, route_id, flag). Any failure returns None -- callers fail closed on None."""
  if not path.is_file(): return None
  try:
    data = json.loads(path.read_text())
  except (json.JSONDecodeError, OSError):
    return None
  if data.get("_schema") != schema: return None
  if data.get("route_id") != route_id or data.get("flag") != flag: return None
  return data


def fail(gate: str, reason: str, **extra) -> dict:
  """The early-FAIL result shape shared by every pre-evidence-check bailout (no such route, wrong
  provenance, no evidence artifact, etc): {gate, verdict: FAIL, reason, **whatever else that site adds}."""
  return {"gate": gate, "verdict": "FAIL", "reason": reason, **extra}


def build_result(*, gate: str, route_id: str, flag: str, provenance: str, required_shapes,
                  evidence_path: pathlib.Path, failures, verdict: str, note: str, extra: dict | None = None) -> dict:
  """Assemble the full (non-early-bailout) result dict with the exact key order every gate prints:
  gate, route_id, flag, provenance, required_shapes, [extra fields], evidence_path, failures, verdict, note."""
  result = {
    "gate": gate,
    "route_id": route_id,
    "flag": flag,
    "provenance": provenance,
    "required_shapes": required_shapes,
  }
  if extra:
    result.update(extra)
  result["evidence_path"] = str(evidence_path)
  result["failures"] = failures
  result["verdict"] = verdict
  result["note"] = note
  return result


def main(evaluate) -> int:
  """Shared gate entry point: print the JSON report, print the AUTHORITY_GATE verdict line, exit 0/1."""
  report = evaluate()
  print(json.dumps(report, indent=2))
  print(f"AUTHORITY_GATE: {report['verdict']}")
  return 0 if report["verdict"] == "PASS" else 1
