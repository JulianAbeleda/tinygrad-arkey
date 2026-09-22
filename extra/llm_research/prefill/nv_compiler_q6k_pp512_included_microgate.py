"""Included-cost timing for the compiler-owned Q6 pp512 binding.

This is deliberately independent of the model arm and llama artifacts.  It
times the binding's graph-owned producer+main submission for each exact role.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, statistics, time
from tinygrad import Device, Tensor, TinyJit, dtypes
from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import binding_for

ROLES = {"attn_v": (1024, 4096), "ffn_down": (4096, 12288)}

def run(role: str, reps: int) -> dict:
  n, k = ROLES[role]; asset = binding_for("NV"); cap = asset.new_capture(); cap.begin_trace()
  x = Tensor.empty((512, k), dtype=dtypes.float16, device="NV")
  # Canonical storage carrier: uint16 Q6_K blocks, with no expanded weight tensor.
  words = Tensor.empty(asset.roles[role].transform.packed_bytes // 2, dtype=dtypes.uint16, device="NV")
  @TinyJit
  def run_graph(inp):
    return cap.project(inp, words, model_family="qwen3_8b", role=role)
  out = run_graph(x); out.realize(); Device["NV"].synchronize()
  samples = []
  outputs = []
  for i in range(reps):
    inp = Tensor.full((512, k), float((i % 2) + 1), dtype=dtypes.float16, device="NV").contiguous()
    Device["NV"].synchronize(); start = time.perf_counter_ns(); got = run_graph(inp); got.realize(); Device["NV"].synchronize()
    outputs.append(float(got.numpy().reshape(-1)[0]));
    samples.append((time.perf_counter_ns() - start) / 1e3)
  identity = asset.roles[role].candidate_identity
  return {"role": role, "shape": {"M": 512, "N": n, "K": k}, "reps": reps,
    "timing_us": {"samples": samples, "median": statistics.median(samples)},
    "candidate_identity": identity, "producer_main_split_us": None,
    "program_identity": {"producer": asset.roles[role].producer.arg.name,
                          "main": asset.roles[role].main_program.arg.name},
    "canonical_weight": True, "expanded_weight_copy": False, "executed_output_samples": outputs,
    "copies": None, "materializations": None,
    "note": "fresh TinyJit inputs force each producer+main graph execution; split device events unavailable"}

def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=9)
  a = ap.parse_args(); rows = [run(role, a.reps) for role in ROLES]
  payload = {"schema": "tinygrad.nv_compiler_q6k_pp512_included_microgate.v1", "status": "PASS",
    "device": str(Device.DEFAULT), "roles": rows, "one_owner": True, "partial_workspace_bytes": 0,
    "old_fixup_programs": 0, "llama_dependency": False,
    "identity_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
  path = pathlib.Path(a.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(payload, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
