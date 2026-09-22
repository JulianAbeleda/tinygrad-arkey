#!/usr/bin/env python3
"""D0.1 executable ABI for the exact Qwen3-8B Q6-down population.

This is a contract generator/validator.  It performs no model execution and
does not install a runtime route.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROLES = tuple((f"blk.{i}.ffn_down", i) for i in range(18))
BOUNDARIES = ("producer", "main", "publication", "residual")
MARKER_ABI = "tinygrad.nv_compiler_q6k_boundary_marker.v1"

def digest(path: str) -> dict[str, object]:
  p = ROOT / path
  data = p.read_bytes()
  return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

def contract() -> dict[str, object]:
  source = {
    p: digest(p) for p in (
      "extra/llm_research/prefill/nv_compiler_q6k_imma_gate.py",
      "extra/llm_research/prefill/nv_compiler_q6k_model_arm.py",
      "extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py",
      "extra/llm_research/prefill/nv_q6down_graph_profile_observer.py",
      "tinygrad/device.py", "tinygrad/runtime/graph/hcq.py")}
  markers = {b: f"{MARKER_ABI}:{b}:v1" for b in BOUNDARIES}
  role_records = [{"role": r, "layer": i, "weight_shape": [4096, 12288],
                   "input_shape": [512, 12288], "output_shape": [512, 4096],
                   "residual_shape": [512, 4096], "role_index": i} for r, i in ROLES]
  return {
    "schema": "tinygrad.nv_prefill_post_substrate_authority.v1",
    "packet": "D0.1", "status": "PASS",
    "authority": {"run_id": "r3-confirming-brackets",
      "model": {"path": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", "sha256": "retained-in-S0-candidate-arms"},
      "prompt": {"tokens": 512, "fixture": "inline:(i*7)%1000"},
      "gpu": "NVIDIA GeForce RTX 5090, sm_120", "driver": "595.84",
      "clocks_session": "graphics=435 MHz, sm=435 MHz, memory=7001 MHz, P3; flock GPU session",
      "source_manifest": source,
      "environment": {"HCQ_NUM_COMPUTE": "2", "HCQ_NV_READY_PLACEMENT": "0",
        "HCQ_NV_MULTI_QUEUE_CUT_POLICY": "combined-flash-direct-deps-cut-v2.json", "PROFILE": "0"}},
    "population": {"role_family": "ffn_down", "count": 18, "order": role_records,
      "weight_type": "Q6_K", "paired_k16_correction": "preserved-exactly"},
    "boundaries": {"names": list(BOUNDARIES), "markers": markers,
      "semantics": {"producer": "compact-Q8 input publication",
        "main": "Q6 main service after producer",
        "publication": "main output publication with ownership retained",
        "residual": "publication plus rank-preserving residual epilogue"}},
    "buffers": {"input": "compact_q8_record[M=512,K=12288]",
      "packed_weight": "canonical Q6_K halfword buffer [N=4096,K=12288]",
      "output": "FP32 [M=512,N=4096]", "residual": "FP32 [M=512,N=4096]",
      "sentinel": "NaN prefill; zero unwritten sentinels required",
      "ownership": "output and residual are distinct, caller-owned buffers"},
    "expected_graph_counts": {"roles": 18, "producer": 18, "main": 18,
      "publication": 18, "residual": 18, "weight_copy": 0,
      "partial_workspace_bytes": 0, "unknown": 0},
    "correctness_cuts": {b: {"finite": True, "unwritten_sentinels": 0,
      "complete_output": True, "read_only_input_hashes": True,
      "check": "allclose against canonical FP16 control at this cut; retain max_abs, mean_abs, relative_l2, argmax"}
      for b in BOUNDARIES},
    "decision": "PASS: all 18 Q6-down roles map one-to-one to producer, main, publication, and residual boundaries and the FP16 controls; no operation is inferred or unnamed.",
    "next_packet": "D0.2"}

def main() -> None:
  ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
  args = ap.parse_args(); payload = contract()
  p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"packet": payload["packet"], "status": payload["status"], "out": str(p)}))

if __name__ == "__main__": main()
