#!/usr/bin/env python3
"""M1b phase-3 attempt: run the real isolated GPU canary on Metal for one row.

Row: Q4_K, ffn_gate_up, shape (512, 12288, 4096) [Qwen3-8B], geometry (256, 64, 32, 8, 1, 1)
-- the exact tuple AMD's own production ffn_gate_up row uses (packed_wmma_prefill.py), which
PG2 already proved is LDS-legal on Metal (25600 < 32768, bc=1). Never inserted into
PACKED_WMMA_ROUTES -- a local-only PackedWmmaRoute, exactly as the M1 probe did.

This is a diagnostic script (not committed as production code): it exists to observe, honestly,
where the real (device="METAL")-threaded `run_canary` path lands -- pass, fail, or which phase/
exception -- using the newly-threaded `device` parameter from M1b task 2.
"""
from __future__ import annotations
import sys, tempfile, os, json, traceback
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

import copy
from tinygrad.llm.packed_wmma_prefill import PackedWmmaRoute, PACKED_WMMA_ROUTE_BY_KEY, PACKED_WMMA_ROUTES
from extra.llm_research.prefill.packed_wmma_correctness_canary import candidate_payload, build_artifact, run_canary
from extra.llm_research.runtime_specs import derive_packed_weight_candidate, full_kernel_workload

QUANT, ROLE, SHAPE = "Q4_K", "ffn_gate_up", (512, 12288, 4096)
GEOMETRY = (256, 64, 32, 8, 1, 1)  # AMD's real ffn_gate_up (tm,tn,tk,wm,wn,bc), reused verbatim.

assert (QUANT, ROLE, SHAPE) not in PACKED_WMMA_ROUTE_BY_KEY, "refusing: shape already a production row"
assert GEOMETRY == PACKED_WMMA_ROUTE_BY_KEY[("Q4_K", "ffn_gate_up", (512, 17408, 5120))].geometry


def _payload_for_local_row(profile: str, row: PackedWmmaRoute) -> dict:
  """Same rebind packed_wmma_production_canary._payload_for_production_row does, parameterized
  by profile instead of the hardcoded 14B PROFILE constant (that constant does not describe this
  8B row's shape)."""
  payload = copy.deepcopy(candidate_payload(profile, row.role))
  if tuple(payload["workload"]["shape"][key] for key in ("m", "n", "k")) != row.shape:
    raise ValueError(f"oracle workload does not match row {row}")
  g, schedule = row.geom, payload["schedule"]
  schedule["tile"] = {"m": g["tm"], "n": g["tn"], "k": g["tk"]}
  schedule["waves"] = {"m": g["wm"], "n": g["wn"]}
  schedule["threads"] = g["wm"] * g["wn"] * 32
  a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80
  schedule["lds"]["windows"] = {"a": [0, a_end], "b": [a_end, b_end]}
  schedule["lds"]["strides"] = {"a": 80, "b": 80}
  schedule["pipeline"]["buffer_count"] = g["bc"]
  return payload


def main() -> None:
  local_route = PackedWmmaRoute(QUANT, ROLE, SHAPE, GEOMETRY, canonical_identity="m1b-probe-placeholder")
  assert all(r.canonical_identity != local_route.canonical_identity for r in PACKED_WMMA_ROUTES)

  payload = _payload_for_local_row("qwen3_8b_q4k_m_gfx1100", local_route)
  entry = derive_packed_weight_candidate(payload, QUANT)
  workload = full_kernel_workload(entry.to_json()["payload"])
  print("=== payload built ===")
  print(json.dumps({"profile": workload.profile, "role": workload.role, "shape": workload.shape,
                     "target": workload.target, "canonical_identity": entry.canonical_identity}, sort_keys=True))

  # Compile-only admission sanity BEFORE spending any GPU time.
  from extra.llm_research.prefill.current_prefill_execution_adapter import admit_current_prefill
  try:
    admission = admit_current_prefill(entry.to_json()["payload"], entry.canonical_identity)
    print("=== admission OK ===")
    print("active_lds_bytes:", admission.active_lds_bytes, "capability:", admission.capability)
  except Exception as e:
    print("=== admission FAILED ===")
    traceback.print_exc()
    return

  fd, artifact_path = tempfile.mkstemp(prefix="m1b_metal_qualification_", suffix=".npz")
  os.close(fd)
  try:
    fixture = build_artifact(QUANT, artifact_path, SHAPE)
    print("=== fixture built ===")
    print(json.dumps(fixture, sort_keys=True))
    print("=== calling run_canary(device='METAL') ===")
    outcome = run_canary(QUANT, artifact_path, timeout_seconds=120.0,
                          base_payload=entry.to_json()["payload"], device="METAL")
    print("=== run_canary returned ===")
    print(json.dumps(outcome, sort_keys=True, default=str))
  except Exception as e:
    print("=== run_canary RAISED ===")
    traceback.print_exc()
  finally:
    try: os.remove(artifact_path)
    except OSError: pass


if __name__ == "__main__":
  main()
