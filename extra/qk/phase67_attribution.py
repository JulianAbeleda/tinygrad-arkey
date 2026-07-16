"""Fail-closed Phase 6/7 identity join and post-miss timing-tax ledger.

This module is deliberately measurement-only.  Synchronized whole-prefill tok/s
decides promotion; Boltbeam rows only explain a miss after the matched run has
been tied to exact execution identities.
"""
from __future__ import annotations

import hashlib, json
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "tinygrad.prefill_phase67_matched_run.v1"
LEDGER_SCHEMA = "boltbeam.prefill_timing_tax_ledger.v1"
_SHA256_KEYS = ("revision", "model_sha256", "clock_identity")
_BINDING_KEYS = ("candidate_identity", "binary_sha256")
_TAX_CLASSES = ("candidate_roles", "activation_preparation_dequantization", "attention",
                "launch_synchronization", "residual")


def _sha256(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _digest(value: Any, label: str) -> str:
  if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
    raise ValueError(f"{label} must be a lowercase SHA-256 digest")
  return value


def _contexts(rows: Mapping[str, Any], label: str) -> tuple[int, ...]:
  try: contexts = tuple(sorted(int(k) for k in rows))
  except (TypeError, ValueError) as exc: raise ValueError(f"{label} contexts must be integer keys") from exc
  if not contexts or any(c <= 0 for c in contexts): raise ValueError(f"{label} contexts must be positive and non-empty")
  for context in contexts:
    samples = rows[str(context)]
    if not isinstance(samples, list) or not samples or any(not isinstance(x, (int, float)) or x <= 0 for x in samples):
      raise ValueError(f"{label} context {context} requires positive raw tok/s samples")
  return contexts


def _bindings(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, str], ...]:
  out = []
  for row in rows:
    role = row.get("invocation_id") or row.get("role")
    if not isinstance(role, str) or not role: raise ValueError("every route binding requires an invocation_id or role")
    out.append((role, _digest(row.get("candidate_identity"), f"{role} candidate_identity"),
                _digest(row.get("binary_sha256"), f"{role} binary_sha256")))
  if not out or len(out) != len(set(x[0] for x in out)): raise ValueError("route bindings must be non-empty and unique")
  return tuple(sorted(out))


def bind_matched_run(tinygrad: Mapping[str, Any], llama: Mapping[str, Any]) -> dict[str, Any]:
  """Validate and canonically bind one alternating matched-run artifact pair."""
  for side, row in (("tinygrad", tinygrad), ("llama", llama)):
    for key in _SHA256_KEYS: _digest(row.get(key), f"{side}.{key}")
  if tinygrad["revision"] != llama["revision"]: raise ValueError("matched revision identity differs")
  if tinygrad["model_sha256"] != llama["model_sha256"]: raise ValueError("matched model identity differs")
  if tinygrad["clock_identity"] != llama["clock_identity"]: raise ValueError("matched clock identity differs")
  tiny_contexts = _contexts(tinygrad.get("tok_s_by_context"), "tinygrad")
  if tiny_contexts != _contexts(llama.get("tok_s_by_context"), "llama"): raise ValueError("matched context set differs")
  sessions = tinygrad.get("session_order")
  if sessions != llama.get("session_order") or not isinstance(sessions, list) or len(sessions) < 6:
    raise ValueError("matched artifacts require the same complete alternating session order")
  if any(sessions[i] == sessions[i-1] for i in range(1, len(sessions))) or set(sessions) != {"tinygrad", "llama"}:
    raise ValueError("session order must alternate tinygrad and llama")
  bindings = _bindings(tinygrad.get("route_bindings") or ())
  payload = {"revision": tinygrad["revision"], "model_sha256": tinygrad["model_sha256"],
             "clock_identity": tinygrad["clock_identity"], "contexts": list(tiny_contexts),
             "session_order": sessions, "route_bindings": [list(x) for x in bindings]}
  return {"schema": SCHEMA, "matched_run_identity": _sha256(payload), **payload,
          "tinygrad_tok_s_by_context": tinygrad["tok_s_by_context"],
          "llama_tok_s_by_context": llama["tok_s_by_context"],
          "promotion_authority": "synchronized_whole_prefill_tok_s"}


def timing_tax_ledger(matched: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
  """Rank Boltbeam timing gaps, refusing traces not produced for the exact matched run."""
  if matched.get("schema") != SCHEMA: raise ValueError("matched-run schema is invalid")
  if trace.get("matched_run_identity") != matched.get("matched_run_identity"):
    raise ValueError("Boltbeam trace is not bound to this matched run")
  if tuple(sorted(int(x) for x in trace.get("contexts", ()))) != tuple(matched["contexts"]):
    raise ValueError("Boltbeam trace context set differs from matched run")
  if _bindings(trace.get("route_bindings") or ()) != tuple(tuple(x) for x in matched["route_bindings"]):
    raise ValueError("Boltbeam trace candidate/binary bindings differ from matched run")
  rows = trace.get("timing_tax_ms")
  if not isinstance(rows, Mapping) or set(rows) != set(_TAX_CLASSES):
    raise ValueError(f"timing_tax_ms must exactly cover {_TAX_CLASSES!r}")
  ledger = []
  for tax_class in _TAX_CLASSES:
    by_context = rows[tax_class]
    if not isinstance(by_context, Mapping) or set(map(str, matched["contexts"])) != set(by_context):
      raise ValueError(f"{tax_class} must cover every matched context")
    vals = [float(by_context[str(c)]) for c in matched["contexts"]]
    if any(v < 0 for v in vals): raise ValueError("timing tax cannot be negative")
    ledger.append({"tax_class": tax_class, "total_tax_ms": round(sum(vals), 6),
                   "tax_ms_by_context": {str(c): vals[i] for i, c in enumerate(matched["contexts"])}})
  ledger.sort(key=lambda row: (-row["total_tax_ms"], row["tax_class"]))
  return {"schema": LEDGER_SCHEMA, "matched_run_identity": matched["matched_run_identity"],
          "promotion_authority": "synchronized_whole_prefill_tok_s", "attribution_only": True,
          "ranked_timing_tax": ledger}


__all__ = ["SCHEMA", "LEDGER_SCHEMA", "bind_matched_run", "timing_tax_ledger"]
