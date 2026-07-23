from __future__ import annotations
import json
from types import MappingProxyType
from typing import Any, Mapping

_CONCRETE_PREFILL_VALIDATED_M = (512,)

_EXECUTING_STRATEGIES = frozenset(("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES", "DIRECT_PACKED_FALLBACK"))
_TC_ATTN_TARGET_REQUIREMENTS = {"backend": "AMD", "architecture": "gfx1100"}
# Enabling one shared compiler path changes both supported model routes.  A
# synthetic or one-model proof is therefore not enough to select it in either
# model: promotion needs the complete cross-route result, including decode
# protection.
_SHARED_ATTENTION_PROOF_FIELDS = ("correctness", "score_resident", "qk_wmma", "pv_wmma",
                                  "model_8b_prefill", "model_14b_prefill",
                                  "decode_nonregression_8b", "decode_nonregression_14b")

def _requirements_met(requirements:Mapping[str, Any], scanned_device_facts:Any) -> bool:
  """Match an exact candidate target contract against the one load-entry scan."""
  return all(getattr(scanned_device_facts, name, None) == expected for name, expected in requirements.items())

def shared_attention_proven_eligible(value:Mapping[str, Any], scanned_device_facts:Any) -> bool:
  """Admit bounded attention only from a target-bound, complete proof record."""
  proof = value.get("shared_attention_proof")
  if not isinstance(proof, Mapping) or proof.get("status") != "PASS": return False
  if not _requirements_met(_TC_ATTN_TARGET_REQUIREMENTS, scanned_device_facts): return False
  target = proof.get("target")
  geometry = proof.get("geometry")
  artifact = proof.get("artifact")
  artifact_ok = (isinstance(artifact, Mapping) and artifact.get("schema") == "tinygrad.shared_attention_proof.v1" and
                 artifact.get("status") == "PASS" and artifact.get("passed") is True)
  return (isinstance(target, Mapping) and dict(target) == _TC_ATTN_TARGET_REQUIREMENTS and
          isinstance(geometry, Mapping) and bool(geometry) and
          artifact_ok and all(proof.get(field) is True for field in _SHARED_ATTENTION_PROOF_FIELDS))

def select_prefill_runtime_policy(value:Mapping[str, Any], *, scanned_device_facts:Any, workload_reuse:bool,
                                  tc_attn_override:bool|None=None) -> Mapping[str, Any]:
  """Add derived runtime diagnostics without granting them route authority."""
  target_default = shared_attention_proven_eligible(value, scanned_device_facts)
  selected = dict(value)
  selected["routes"] = dict(selected["routes"])
  # An override is a disable switch for diagnosis, never an admission bypass.
  # The proof remains the only authority that can turn this path on.
  selected.update({"workload_reuse": bool(workload_reuse),
                   "prefill_tc_attn": target_default and tc_attn_override is not False})
  return immutable_prefill_policy(selected)

def immutable_prefill_policy(value:Mapping[str, Any]) -> Mapping[str, Any]:
  """Validate and freeze the small runtime authority consumed by model.py."""
  strategy = value.get("strategy")
  if strategy not in _EXECUTING_STRATEGIES: raise ValueError(f"invalid executing prefill strategy {strategy!r}")
  candidate_id = value.get("candidate_id")
  if not isinstance(candidate_id, str) or not candidate_id: raise ValueError("prefill policy requires candidate_id")
  routes = value.get("routes")
  if not isinstance(routes, Mapping) or any(not isinstance(k, str) or not k or
      not isinstance(v, str) or not v for k, v in routes.items()):
    raise ValueError("prefill policy routes must bind invocation IDs to non-empty route IDs")
  # Round-trip first: callers cannot retain mutable nested references after this boundary.
  frozen = json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False))
  frozen["routes"] = MappingProxyType(frozen["routes"])
  return MappingProxyType(frozen)

def prefill_policy_strategy(policy:Mapping[str, Any]|None) -> str:
  return "DIRECT_PACKED_FALLBACK" if policy is None else str(policy["strategy"])

def prefill_policy_uses_overlay(policy:Mapping[str, Any]|None) -> bool:
  return prefill_policy_strategy(policy) == "FULL_RESIDENT_OVERLAY"

def prefill_concrete_kv_auto_decision(workload_reuse:bool, prefill_v2_on:bool) -> tuple[bool, str]:
  # Concrete-KV (the TC-attention fast path, model.py's `isinstance(start_pos, int)` gate) is the
  # default execution mode for every prefill-v2 chunk, not an opt-in: a symbolic start_pos (the
  # `v_start_pos.bind(sp)` continuation-chunk fallback) never satisfies that isinstance check, so
  # every chunk past the first would otherwise take the slow SDPA path regardless of `workload_reuse`.
  # Each per-start_pos jit still compiles lazily on first use (~5s) and is cached on the model instance
  # for its lifetime (see model.py's `prefill_v2_jits.setdefault`), so only the first request to touch a
  # given chunk offset pays the tax; `workload_reuse` (unused here) separately gates EAGER precompile-at-load
  # (model.py's `precompile_concrete_prefill_jits`) for callers who want that tax paid up front instead.
  if not prefill_v2_on: return (False, "selected prefill representation is off -> concrete-KV moot")
  return (True, "prefill-v2 concrete-KV attention is the default execution path; per-start_pos jits compile lazily and cache")

def prefill_v2_validate_ubatch(ubatch:int) -> None:
  if ubatch not in _CONCRETE_PREFILL_VALIDATED_M:
    raise ValueError(f"concrete prefill validates physical M in {_CONCRETE_PREFILL_VALIDATED_M} (got {ubatch}); "
                     f"the warmstart TC schedule is shape-specific. Re-measure per-shape opts for {ubatch} first "
                     f"and add it to _CONCRETE_PREFILL_VALIDATED_M.")
