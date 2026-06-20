#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
HIP_RUNNER = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_hip_runner_result.json"
GATEUP_EXTRACT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_gateup_extract_result.json"
SEMANTIC = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_semantic_map_result.json"
NATIVE_LEDGER = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json"
C7D = ROOT / "bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7d_confirmation_result.json"
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_coarse_attribution_result.json"


def load(path: pathlib.Path, default: Any = None) -> Any:
  return json.loads(path.read_text()) if path.exists() else default


def nested(obj: dict[str, Any], *keys: str, default: Any = None) -> Any:
  cur: Any = obj
  for key in keys:
    if not isinstance(cur, dict) or key not in cur:
      return default
    cur = cur[key]
  return cur


def main() -> int:
  hip = load(HIP_RUNNER, {})
  extract = load(GATEUP_EXTRACT, {})
  semantic = load(SEMANTIC, {})
  native_ledger = load(NATIVE_LEDGER, {})
  c7d = load(C7D, {})

  trace_res = nested(hip, "kernel_trace_summary", "resource_fields_first", default={}) or {}
  native_desc = nested(native_ledger, "native", "descriptor", default={}) or {}
  native_grouped = nested(native_ledger, "native", "grouped", default={}) or {}
  oracle_grouped = nested(semantic, "total_grouped", default={}) or {}
  timings_doc = {
    "oracle_prior_us": 93.540,
    "oracle_hip_runner_trace_avg_us": (nested(hip, "kernel_trace_summary", "duration_ns_avg", default=0) or 0) / 1000.0,
    "native_us": 280.247,
    "best_static_us": 270.635,
    "c7c_best_us": 264.628,
  }

  native_alloc_vgpr = native_desc.get("allocated_vgpr_per_workitem")
  oracle_trace_vgpr = trace_res.get("VGPR_Count")
  resource_delta = {
    "oracle_trace_vgpr": oracle_trace_vgpr,
    "oracle_metadata_vgpr": nested(extract, "metadata", "vgpr_count"),
    "native_allocated_vgpr": native_alloc_vgpr,
    "native_minus_oracle_trace_vgpr": (native_alloc_vgpr - oracle_trace_vgpr) if isinstance(native_alloc_vgpr, int) and isinstance(oracle_trace_vgpr, int) else None,
    "oracle_trace_sgpr": trace_res.get("SGPR_Count"),
    "oracle_metadata_sgpr": nested(extract, "metadata", "sgpr_count"),
    "native_static_max_sgpr": nested(native_ledger, "native", "registers", "max_sgpr"),
    "oracle_trace_lds_block_size": trace_res.get("LDS_Block_Size"),
    "oracle_metadata_group_segment_fixed_size": nested(extract, "metadata", "group_segment_fixed_size"),
    "native_group_segment_fixed_size": native_desc.get("group_segment_fixed_size"),
    "oracle_scratch": trace_res.get("Scratch_Size"),
    "native_private_segment_fixed_size": native_desc.get("private_segment_fixed_size"),
  }

  static_count_delta = {
    k: {
      "oracle": oracle_grouped.get(k),
      "native": native_grouped.get(k),
      "native_minus_oracle": (native_grouped.get(k, 0) - oracle_grouped.get(k, 0)),
    }
    for k in sorted(set(oracle_grouped) | set(native_grouped))
    if k in {"global_load", "ds", "valu", "salu", "waitcnt", "dot4", "shuffle", "barrier", "global_store"}
  }

  gates = {
    "hip_runner_kernel_trace_passed": hip.get("gate_pass") is True,
    "oracle_metadata_extracted": extract.get("gate_pass") is True,
    "semantic_map_passed": semantic.get("gate_pass") is True,
    "native_resource_ledger_present": bool(native_desc),
    "oracle_vgpr_now_known": isinstance(oracle_trace_vgpr, int),
    "native_vgpr_now_known": isinstance(native_alloc_vgpr, int),
    "oracle_lower_vgpr_than_native": isinstance(native_alloc_vgpr, int) and isinstance(oracle_trace_vgpr, int) and oracle_trace_vgpr < native_alloc_vgpr,
    "att_pc_timeline_still_absent": True,
  }

  decision = {
    "reopen_native": "scope_only",
    "why": "NINFO-1 is now materially changed: oracle's profiled resource envelope is much smaller in VGPR than native. Without ATT PC attribution, this does not justify another ad hoc schedule rewrite; it justifies a targeted NINFO-4 live-range/resource-compression phase.",
    "target": "native VGPR/live-range reduction toward oracle envelope while preserving the OES-4 S3/S4/S5 semantics",
    "stop_condition": "Do not promote unless a resource/liveness change produces a correct candidate and moves same-run timing materially; static count matching remains refuted.",
    "next_phase": "DNR-4 resource/live-range attribution and compression scope",
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_COARSE_ATTRIBUTION",
    "schema": "decode_oracle_coarse_attribution_v1",
    "verdict": "PASS_DECODE_COARSE_ATTRIBUTION_REOPENS_TARGETED_RESOURCE_LIVENESS_SCOPE" if all(gates.values()) else "BLOCKED_DECODE_COARSE_ATTRIBUTION_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "timings_us": timings_doc,
    "resource_delta": resource_delta,
    "static_count_delta": static_count_delta,
    "decision": decision,
    "gates": gates,
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "resource_delta": resource_delta,
    "decision": decision,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())

