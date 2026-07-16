from __future__ import annotations
import json
from types import MappingProxyType
from typing import Any, Mapping

_PREFILL_V2_VALIDATED_UBATCH = (512,)

def prefill_plan_uses_overlay(serialized_plan:str|None) -> bool:
  if serialized_plan is None: return False
  return json.loads(serialized_plan).get("decision") == "FULL_RESIDENT_OVERLAY"

_EXECUTING_STRATEGIES = frozenset(("FULL_RESIDENT_OVERLAY", "BOUNDED_PACKED_TILES", "DIRECT_PACKED_FALLBACK"))
_PROMOTED_PREFILL_TARGET_REQUIREMENTS = {"backend": "AMD", "architecture": "gfx1100"}

def _requirements_met(requirements:Mapping[str, Any], scanned_device_facts:Any) -> bool:
  """Match an exact candidate target contract against the one load-entry scan."""
  return all(getattr(scanned_device_facts, name, None) == expected for name, expected in requirements.items())

def select_prefill_runtime_policy(value:Mapping[str, Any], *, scanned_device_facts:Any, workload_reuse:bool,
                                  graph_gemm_override:bool|None=None, tc_attn_override:bool|None=None) -> Mapping[str, Any]:
  """Add size/name-independent workload and target applicability facts, then freeze the load policy."""
  target_default = _requirements_met(_PROMOTED_PREFILL_TARGET_REQUIREMENTS, scanned_device_facts)
  selected = dict(value)
  selected["routes"] = dict(selected["routes"])
  selected.update({"workload_reuse": bool(workload_reuse),
                   "prefill_graph_gemm": target_default if graph_gemm_override is None else graph_gemm_override,
                   "prefill_tc_attn": target_default if tc_attn_override is None else tc_attn_override})
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
  # Precompile pays off only when the caller declares that generated workload will be reused.
  if not prefill_v2_on: return (False, "PREFILL_V2 off -> concrete-KV moot, OFF")
  if workload_reuse: return (True, "workload reuse + PREFILL_V2 on -> precompile concrete jits, ON")
  return (False, "no workload reuse (one-shot assumed) -> OFF; set PREFILL_SERVER_PROFILE=1 or PREFILL_CONCRETE_KV=1")

def prefill_v2_validate_ubatch(ubatch:int) -> None:
  if ubatch not in _PREFILL_V2_VALIDATED_UBATCH:
    raise ValueError(f"PREFILL_V2 only validates PREFILL_UBATCH in {_PREFILL_V2_VALIDATED_UBATCH} (got {ubatch}); "
                     f"the warmstart TC schedule is shape-specific. Re-measure per-shape opts for {ubatch} first "
                     f"and add it to _PREFILL_V2_VALIDATED_UBATCH.")

def prefill_v2_realize_bytes(shapes:list[tuple[int,int]]) -> int:
  return sum(o * i for o, i in shapes) * 2
