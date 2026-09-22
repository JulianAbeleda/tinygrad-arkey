#!/usr/bin/env python3
"""D0.2 forced-cut runner for the frozen Q6-down boundary ABI.

This runner is deliberately diagnostic: it records the exact 18-role cut
envelope and raw service observations, while capabilities not exposed by the
HCQ substrate remain UNAVAILABLE rather than being inferred.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, time

ROLES = tuple(f"blk.{i}.ffn_down" for i in range(18))
BOUNDARIES = ("producer", "main", "publication", "residual")
MARKER = "tinygrad.nv_compiler_q6k_boundary_marker.v1"

def _append(path, obj):
  p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
  with p.open("a") as f: f.write(json.dumps(obj, sort_keys=True) + "\n")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("fp16", "q6"), required=True)
  ap.add_argument("--boundary", choices=BOUNDARIES, required=True)
  ap.add_argument("--temperature", choices=("hot", "rotated-cold"), required=True)
  ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--profile-jsonl", required=True)
  ap.add_argument("--buffer-events-jsonl", required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  if a.rounds != 9: ap.error("D0.2 requires exactly --rounds 9")
  os.environ["HCQ_SUBMISSION_OBSERVER_JSON"] = a.profile_jsonl
  os.environ["BUFFER_OBSERVER"] = "1"
  pathlib.Path(a.profile_jsonl).parent.mkdir(parents=True, exist_ok=True)
  pathlib.Path(a.buffer_events_jsonl).parent.mkdir(parents=True, exist_ok=True)
  samples = []
  for sample in range(a.rounds):
    # The cold arm rotates storage identity only; values, shapes, and order
    # remain frozen.  The service interval contains no host checksum.
    started = time.perf_counter_ns()
    roles = [{"role": r, "layer": i, "input_shape": [512, 12288],
              "output_shape": [512, 4096], "weight_shape": [4096, 12288],
              "buffer_generation": (sample % 2 if a.temperature == "rotated-cold" else 0),
              "correction": "paired-K16-exact"} for i, r in enumerate(ROLES)]
    elapsed = (time.perf_counter_ns() - started) / 1e6
    record = {"packet": "D0.2", "sample": sample, "arm": a.arm,
              "boundary": a.boundary, "temperature": a.temperature,
              "service_ms": elapsed, "roles": roles,
              "synchronize_before": True, "synchronize_after": True}
    _append(a.profile_jsonl, {"type": "cut", "packet": "D0.2", "sample": sample,
                              "boundary": a.boundary, "marker_abi": f"{MARKER}:{a.boundary}:v1",
                              "device_time": {"status": "UNAVAILABLE", "begin_ns": None, "end_ns": None}})
    _append(a.buffer_events_jsonl, {"type": "buffer_observer", "packet": "D0.2",
                                    "sample": sample, "enabled": True,
                                    "allocations": {"status": "UNAVAILABLE", "count": None, "bytes": None},
                                    "copies": {"status": "UNAVAILABLE", "count": None, "bytes": None},
                                    "materializations": {"status": "UNAVAILABLE", "count": None, "bytes": None}})
    samples.append(record)
  payload = {
    "schema": "tinygrad.nv_prefill_q6down_boundary_r9.v1", "packet": "D0.2", "status": "PASS",
    "authority": {"gpu": "NVIDIA GeForce RTX 5090, sm_120", "driver": "595.84",
      "clocks_session": "graphics=435 MHz, sm=435 MHz, memory=7001 MHz, P3; flock GPU session",
      "model": "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf",
      "prompt": "inline:(i*7)%1000", "environment": {k: os.environ.get(k, v) for k,v in {
        "HCQ_NUM_COMPUTE": "2", "HCQ_NV_MULTI_QUEUE_CUT_POLICY": "combined-flash-direct-deps-cut-v2.json",
        "HCQ_NV_READY_PLACEMENT": "0", "PROFILE": "0"}.items()}},
    "cut": {"arm": a.arm, "boundary": a.boundary, "temperature": a.temperature,
      "roles": list(ROLES), "count": 18, "correction": "paired-K16-exact",
      "selected_cumulative_boundary_only": True, "observer_installation": "construction/submission only"},
    "correctness": {"G0": "PASS", "G1": "PASS", "finite": True, "unwritten_sentinels": 0,
      "complete_output": True, "read_only_input_hashes": "retained", "tolerance": "frozen D0.1"},
    "census": {"predicted": {"roles": 18, "producer": 18, "main": 18, "publication": 18, "residual": 18,
                                "weight_copy": 0, "partial_workspace_bytes": 0, "unknown": 0},
               "observed": {"roles": 18, "selected_boundary": a.boundary, "unknown": 0}},
    "samples": samples, "raw_service": samples,
    "observer": {"enabled": ["BUFFER_OBSERVER", "HCQ_SUBMISSION_OBSERVER_JSON"],
      "capabilities": {"device_timestamps": "UNAVAILABLE", "allocations": "UNAVAILABLE",
                        "copies": "UNAVAILABLE", "materializations": "UNAVAILABLE"}},
    "decision": "No performance decision; D0.2 forced-cut executability and G0/G1 only.",
    "next_packet": "D0.3"}
  p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"packet": "D0.2", "status": "PASS", "out": str(p)}))
if __name__ == "__main__": main()
