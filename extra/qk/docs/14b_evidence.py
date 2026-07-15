"""Exact mixed-quant 14B evidence record and fail-closed runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "tinygrad.qk.mixed_quant_14b_evidence.v1"
REQUIRED = ("status", "run", "hardware", "roles", "prefill_decode", "route_census", "parity", "memory",
            "q8_prep", "gpu_health", "fallbacks", "direct_packed_comparator", "measurement_definition")
TOP_LEVEL = frozenset(("schema", "research_only", "route_promotion", *REQUIRED))
ROLES = ("embedding", "attn_qo", "attn_kv", "ffn_gate_up", "ffn_down", "lm_head")
PHASES = ("prefill", "decode")


class EvidenceError(ValueError): pass


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
