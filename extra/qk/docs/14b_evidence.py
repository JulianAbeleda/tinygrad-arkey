"""Exact mixed-quant 14B evidence record and fail-closed runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from extra.qk.layout import format_name, read_metadata, tensor_shape

SCHEMA = "tinygrad.qk.mixed_quant_14b_evidence.v1"
REQUIRED = ("status", "run", "hardware", "roles", "prefill_decode", "route_census", "parity", "memory",
            "q8_prep", "gpu_health", "fallbacks", "direct_packed_comparator", "measurement_definition")
TOP_LEVEL = frozenset(("schema", "research_only", "route_promotion", *REQUIRED))
ROLES = ("embedding", "attn_qo", "attn_kv", "ffn_gate_up", "ffn_down", "lm_head")
PHASES = ("prefill", "decode")
MIXED_ROLE_SUFFIXES = {
  "embedding": ("token_embd.weight",),
  "attn_qo": ("attn_q.weight", "attn_output.weight"),
  "attn_kv": ("attn_k.weight", "attn_v.weight"),
  "ffn_gate_up": ("ffn_gate.weight", "ffn_up.weight"),
  "ffn_down": ("ffn_down.weight",),
  "lm_head": ("output.weight",),
}


class EvidenceError(ValueError): pass


def exact_mixed_quant_inventory(model_path: str | Path) -> dict[str, Any]:
  """Read the model's role quantization without realizing tensor payloads."""
  path = Path(model_path)
  meta = read_metadata(path)
  roles: dict[str, Any] = {}
  for role, suffixes in MIXED_ROLE_SUFFIXES.items():
    tensors = []
    for suffix in suffixes:
      matches = [info for info in meta.infos if info.name.endswith(suffix)]
      if not matches: raise EvidenceError(f"{path}: missing tensor suffix {suffix!r} for role {role}")
      tensors.extend(matches)
    roles[role] = {
      "tensors": [{"name": info.name, "shape": list(tensor_shape(info)), "quantization": format_name(info.typ)}
                  for info in tensors],
      "quantizations": sorted({format_name(info.typ) for info in tensors}),
    }
  return {"model_path": str(path), "role_count": len(roles), "roles": roles}


def validate_exact_mixed_quant_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
  """Fail closed if the known Qwen3-14B mixed-Q4/Q6 boundary is not present."""
  if not isinstance(inventory, Mapping) or set(inventory.get("roles", {})) != set(MIXED_ROLE_SUFFIXES):
    raise EvidenceError("mixed-quant inventory does not cover all evidence roles")
  expected = {
    "embedding": {"Q4_K"}, "attn_qo": {"Q4_K"}, "attn_kv": {"Q4_K", "Q6_K"},
    "ffn_gate_up": {"Q4_K"}, "ffn_down": {"Q4_K", "Q6_K"}, "lm_head": {"Q4_K", "Q6_K"},
  }
  for role, quantizations in expected.items():
    actual = set(inventory["roles"][role].get("quantizations", ()))
    if actual != quantizations:
      raise EvidenceError(f"{role} quantization boundary mismatch: expected {sorted(quantizations)}, got {sorted(actual)}")
  return dict(inventory)


def _required_map(value: Any, name: str) -> Mapping[str, Any]:
  if not isinstance(value, Mapping) or not value: raise EvidenceError(f"{name} must be a non-empty object")
  return value


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
  if not isinstance(record, Mapping): raise EvidenceError("record must be an object")
  missing = [key for key in REQUIRED if key not in record]
  if missing: raise EvidenceError(f"missing required fields: {', '.join(missing)}")
  unknown = set(record) - TOP_LEVEL
  if unknown: raise EvidenceError(f"unknown top-level fields: {sorted(unknown)}")
  if record.get("schema") != SCHEMA: raise EvidenceError(f"schema must be {SCHEMA}")
  if record.get("research_only") is not True or record.get("route_promotion") is not False:
    raise EvidenceError("record must be research_only=true and route_promotion=false")
  if record.get("status") != "PASS": raise EvidenceError("validated evidence must have status=PASS")
  run = _required_map(record["run"], "run")
  for key in ("run_id", "model_id", "quantization", "prompt_tokens", "decode_tokens"):
    if key not in run or run[key] in (None, ""): raise EvidenceError(f"run.{key} is required")
  if run["quantization"] != "mixed_q4_k_m_q6_k_q8_prep":
    raise EvidenceError("run.quantization must identify the exact mixed-quant setup")
  hardware = _required_map(record["hardware"], "hardware")
  for key in ("device", "driver", "runtime", "health_probe_id"):
    if not hardware.get(key): raise EvidenceError(f"hardware.{key} is required")
  roles = record["roles"]
  if not isinstance(roles, Mapping) or set(roles) != set(ROLES): raise EvidenceError(f"roles must contain exactly {ROLES}")
  for role, value in roles.items():
    row = _required_map(value, f"roles.{role}")
    for key in ("route_id", "prefill_ms", "decode_ms", "tokens"):
      if key not in row or row[key] is None: raise EvidenceError(f"roles.{role}.{key} is required")
  phases = _required_map(record["prefill_decode"], "prefill_decode")
  if set(phases) != set(PHASES): raise EvidenceError("prefill_decode must contain prefill and decode")
  for phase in PHASES:
    if not _required_map(phases[phase], f"prefill_decode.{phase}").get("elapsed_ms"):
      raise EvidenceError(f"prefill_decode.{phase}.elapsed_ms is required and non-zero")
  for key in ("route_census", "parity", "memory", "q8_prep", "gpu_health", "fallbacks",
              "direct_packed_comparator", "measurement_definition"):
    _required_map(record[key], key)
  if not isinstance(record["fallbacks"].get("count"), int): raise EvidenceError("fallbacks.count must be an integer")
  if record["gpu_health"].get("status") != "PASS": raise EvidenceError("GPU health must PASS")
  if record["parity"].get("status") != "PASS": raise EvidenceError("output/token parity must PASS")
  return dict(record)


def run_evidence(*, collect: Callable[[], Mapping[str, Any]], hardware_probe: Callable[[], Mapping[str, Any]],
                 output: Path | None = None) -> dict[str, Any]:
  try:
    hardware = hardware_probe()
    record = dict(collect()); record["hardware"] = dict(hardware)
    result = validate_record(record)
  except Exception as exc:
    return {"schema": SCHEMA, "status": "BLOCKED", "research_only": True,
            "route_promotion": False, "blocker": str(exc)}
  if output is not None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result
