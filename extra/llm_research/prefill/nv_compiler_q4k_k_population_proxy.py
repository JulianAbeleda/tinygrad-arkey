#!/usr/bin/env python3
"""Captured 36-real-weight proxy for the occupancy-safe compiler Q4_K K path."""
from __future__ import annotations

import argparse, json, pathlib, statistics, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_compiler_q4k_k_role_gate import M, N, K, _context
from extra.llm_research.prefill.nv_compiler_q4k_production_gate import _activation_carrier, _weight_carrier, _buf


def _call_count(linear:UOp) -> int:
  total = 0
  for call in linear.src:
    program = call.src[0]
    total += _call_count(program.src[0]) if program.op is Ops.CUSTOM_FUNCTION and program.arg == "graph" else 1
  return total


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if args.rounds < 9: raise ValueError("population qualification requires R9 or greater")
  from extra.llm_research.layout import GGML_Q4_K, packed_u32_slice, read_metadata

  model_path = pathlib.Path(args.model)
  metadata = read_metadata(model_path)
  infos = [i for i in metadata.infos if i.name.endswith("attn_k.weight")]
  infos.sort(key=lambda i:int(i.name.split(".")[1]))
  if len(infos) != 36 or any(i.typ != GGML_Q4_K or tuple(reversed(i.dims)) != (N,K) for i in infos):
    raise RuntimeError("proxy requires 36 canonical Q4_K K weights")
  weights = [packed_u32_slice(model_path,metadata,i,device="NV").contiguous().realize() for i in infos]
  weight_samples_before = [weights[i].numpy()[::max(1,weights[i].numel()//4096)].copy() for i in (0,17,35)]

  q8_np = (((np.arange(M*K,dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(M,K)
  groups = np.arange(M*(K//32),dtype=np.int64).reshape(M,K//32)
  scales_np = (2.0**((groups%7)-5)).astype(np.float32)
  sums_np = q8_np.reshape(M,K//32,32).astype(np.int32).sum(2).astype(np.float32)
  record_np = np.frombuffer(q8_np.reshape(-1).tobytes()+scales_np.reshape(-1).tobytes()+sums_np.reshape(-1).tobytes(),
                            np.uint32).copy()
  record = Tensor(record_np,device="NV").contiguous().realize()

  wt, at, geometry, identity, context = _context(64,32,2,2,128)
  key = warmstart_key({M,N},K,wt.storage_dtype)
  @TinyJit
  def batch(record_arg:Tensor):
    return tuple(_activation_carrier(record_arg,at).matmul(_weight_carrier(words,wt).transpose(),dtype=dtypes.int)
      .cast(dtypes.float).contiguous().realize() for words in weights)

  from tinygrad.codegen import to_program_cache
  to_program_cache.clear()
  graph_times = []
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)}, {key:context}):
    for iteration in range(args.rounds+3):
      Device["NV"].synchronize(); started = time.perf_counter_ns(); outputs = batch(record); Device["NV"].synchronize()
      if iteration >= 3: graph_times.append((time.perf_counter_ns()-started)/1e6)
  if batch.captured is None: raise RuntimeError("population graph did not capture")
  sample_indices = np.asarray([0,63,64,4095,16383,M*N-1])
  before = [outputs[i].numpy().reshape(-1)[sample_indices].copy() for i in (0,17,35)]
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)}, {key:context}): outputs = batch(record)
  Device["NV"].synchronize()
  after = [outputs[i].numpy().reshape(-1)[sample_indices] for i in (0,17,35)]
  captured_calls = _call_count(batch.captured.linear)
  # The 36 proxy outputs are independent, so HCQ is allowed to overlap them.
  # Real layer K projections are dependency-ordered.  Re-launch the exact one
  # compiler binary serially on the default queue and synchronize only after
  # all 36 calls; this is the admissible population-service proxy.
  programs = [p for p in to_program_cache.values()
              if getattr(getattr(p.arg,"candidate_context",None),"canonical_identity",None) == identity]
  if len(programs) != 1: raise RuntimeError(f"expected one compiler K program, got {len(programs)}")
  program = programs[0]
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY and isinstance(u.arg,bytes))
  from tinygrad.runtime.ops_nv import NVProgram
  native = NVProgram(Device["NV"],program.arg.name,binary)
  serial_times = []
  for iteration in range(args.rounds+3):
    Device["NV"].synchronize(); started = time.perf_counter_ns()
    for output, words in zip(outputs,weights):
      native(_buf(output),_buf(record),_buf(words),global_size=program.arg.global_size,local_size=program.arg.local_size)
    Device["NV"].synchronize()
    if iteration >= 3: serial_times.append((time.perf_counter_ns()-started)/1e6)
  samples_readonly = all(np.array_equal(weights[i].numpy()[::max(1,weights[i].numel()//4096)],old)
                         for i,old in zip((0,17,35),weight_samples_before))
  hot_min, fp16_population_ms = min(serial_times), 6.3845
  rec = {"schema":"tinygrad.nv_compiler_q4k_k_population_proxy.v1",
         "shape":{"M":M,"N":N,"K":K,"tile":[64,32,64],"ctas_per_call":256},
         "real_weight_buffers":len(weights),"captured_calls":captured_calls,"identity":identity,
         "correctness":{"sample_replay_exact":all(np.array_equal(a,b) for a,b in zip(before,after)),
           "sample_distinct_real_weights":all(not np.array_equal(after[i],after[j]) for i in range(3) for j in range(i+1,3)),
           "sample_finite":bool(all(np.isfinite(x).all() for x in after)),"record_readonly":bool(np.array_equal(record.numpy(),record_np)),
           "weight_samples_readonly":samples_readonly},
         "timing":{"serial_samples_ms":serial_times,"serial_wall_min_ms":hot_min,
           "serial_wall_median_ms":statistics.median(serial_times),
           "independent_graph_samples_ms":graph_times,"independent_graph_min_ms":min(graph_times),
           "independent_graph_median_ms":statistics.median(graph_times),
           "independent_graph_is_not_layer_ordered":True,
           "per_call_wall_us":hot_min*1000/36,"retained_fp16_population_ms":fp16_population_ms,
           "recovery_vs_fp16_ms":fp16_population_ms-hot_min,"speedup_vs_fp16":fp16_population_ms/hot_min,
           "fp16_authority":"nv-prefill-complete-lifecycle-ledger.json, profiled K population"}}
  rec["passed"] = bool(captured_calls == 36 and all(rec["correctness"].values()) and hot_min < fp16_population_ms)
  out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True); out_path.write_text(json.dumps(rec,indent=2)+"\n")
  print(json.dumps(rec,sort_keys=True))
  if not rec["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
