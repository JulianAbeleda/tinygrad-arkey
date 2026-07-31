#!/usr/bin/env python3
"""Compile-only, then one GPU measurement: verify search_provider.py's new
candidate_geometry sidecar reaches build_precontract_lds_stage instead of the
generic TC decline.

Geometry used is AMD's (256,64,32,8,1,1) ffn_gate_up tuple -- known (PG2) to lower on
Metal.  It is used ONLY to verify the wiring; it is not proposed as a default and this
script adds nothing to PACKED_WMMA_ROUTES.
"""
import hashlib, json, sys

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter, ProtocolError

device = Device["METAL"]
arch = getattr(device, "arch", "unknown")

TARGET = {"target_id": "apple_m_wire_check", "backend": "METAL", "arch": arch,
          "subgroup_size": 32, "resolved_target_hash": "0" * 64}

# tm, tn, tk, wm, wn, bc = 256, 64, 32, 8, 1, 1  (AMD ffn_gate_up tuple, PG2-confirmed to lower on Metal)
TM, TN, TK, WM, WN, BC = 256, 64, 32, 8, 1, 1

candidate = {
  "schema_version": "boltbeam.full_kernel_candidate.v2",
  "workload": {
    "profile": "wire_check_profile", "model_sha256": "1" * 64, "phase": "prefill", "role": "ffn_gate_up",
    # M/N deliberately equal the tile exactly (tm=256, tn=64) so outer_m == outer_n == 1: this
    # is a wiring check, not a search over the outer-tile loop trip count.
    "operation": "matmul", "shape": {"m": 256, "n": 64, "k": 5120},
    "operands": {
      "a": {"dtype": "half", "layout": "row_major", "quantization": "none"},
      "b": {"dtype": "uint32", "layout": "packed_q4_k", "quantization": "Q4_K"},
      "c": {"dtype": "float", "layout": "row_major", "quantization": "none"},
    },
    "accumulator_dtype": "float", "target": TARGET,
  },
  "schedule": {
    "plan_kind": "tinygrad_heuristic.v1", "transforms": [],
    "tile": {"m": TM, "n": TN, "k": TK}, "launch": {"threads": WM * WN * 32},
    "mapping": {"lane_policy": "wire_check"},
    "memory": {"a": {"space": "global", "vector_width": 1, "alignment": 1},
               "b": {"space": "global", "vector_width": 1, "alignment": 1},
               "c": {"space": "global", "vector_width": 1, "alignment": 1}},
    "pipeline": {"stage_count": 1},
    "compute": {"family": "wmma"}, "numerical_mode": "default",
  },
  "static_constraints": {"max_local_memory_bytes": None, "max_registers_per_thread": None, "spill_policy": "unknown"},
  "correctness": {"oracle": "canonical_packed_reference", "atol": 1e-3, "rtol": 1e-3},
  "memory_budget": {"status": "unavailable", "bytes": None},
  "provenance": {"generator_id": "wire_check", "generator_revision": "0", "schema_revision": "boltbeam.full_kernel_candidate.v2"},
  "applicability": {"exact_shape": True, "profiles": [], "roles": [], "targets": []},
}
candidate_hash = hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

payload = {
  "candidate": candidate, "candidate_hash": candidate_hash, "target_identity": TARGET,
  "candidate_geometry": {"wm": WM, "wn": WN, "bc": BC},
  "fixture_shape": {"n": 64, "k": 5120, "m": 256},
  "samples": 5, "warmups": 2,
}

adapter = MetalAdapter()

print("=== compile ===")
try:
  result = adapter.compile(payload)
except ProtocolError as exc:
  print("DECLINED", exc.code, str(exc))
  raise SystemExit(1)

source_sha = result["source_sha256"]
print("source_sha256:", source_sha)
print("bound_compiler_opts:", result["bound_compiler_opts"])

from extra.llm_research.search_provider import MetalAdapter as _MA  # noqa
prepared = adapter._prepared(payload)
source = prepared.ast.src[3].arg
has_wmma = "__WMMA_8_8_8_half_float" in source
has_simd = "simdgroup_multiply_accumulate" in source
print("has __WMMA_8_8_8_half_float:", has_wmma)
print("has simdgroup_multiply_accumulate:", has_simd)
if not (has_wmma and has_simd):
  print("FAIL: compiled source does not show the precontract WMMA path")
  raise SystemExit(1)
print("PASS: reached build_precontract_lds_stage, not the generic decline")

print("=== check (correctness) ===")
check_result = adapter.check(payload)
print(json.dumps(check_result, indent=2, default=str))

print("=== measure ===")
Device["METAL"].synchronize()
measure_result = adapter.measure(payload)
Device["METAL"].synchronize()
print("median_ns candidates (samples_ns):", measure_result["samples_ns"])
import statistics
print("median_ns:", statistics.median(measure_result["samples_ns"]))
print("correctness.passed:", check_result.get("correct"))
