#!/usr/bin/env python3
"""Q6_K staged-WMMA measurement for an experimental 8B-like fixture.

Research-only: this intentionally bypasses route dispatch and compares the
staged candidate with the existing generated direct-packed kernel.  Its
512x4096x12288 shape is not canonical Qwen3-14B authority.
"""
from __future__ import annotations
import json, os, platform, time
from pathlib import Path
from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from extra.qk.layout import Q6_K_BLOCK_BYTES, q6_k_reference
from extra.qk.q6k_prefill_route_spec import describe_q6k_packed_prefill, emit_q6k_packed_prefill_kernel
from extra.qk.q6k_wmma_prefill_spec import describe_q6k_wmma_prefill, emit_q6k_wmma_prefill

M, N, K = 512, 4096, 12288

def timed(fn, rounds=3):
  fn().numpy()  # compile/warmup and synchronize
  vals = []
  for _ in range(rounds):
    start = time.perf_counter()
    result = fn()
    result.numpy()  # host conversion is the synchronization point
    vals.append((time.perf_counter()-start)*1e3)
  return {"samples_ms": vals, "median_ms": sorted(vals)[len(vals)//2], "accounting": "synchronized_wall"}

def compile_facts(out):
  with Context(DEV="AMD:ISA:gfx1100"):
    compiled = compile_linear(out.schedule_linear())
  programs = [u.src[0] for u in compiled.src if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
  sources = [next((u.arg for u in p.src if u.op is Ops.SOURCE), "") for p in programs]
  return {"programs": len(programs), "wmma_programs": sum("wmma_f32_16x16x16_f16" in s for s in sources),
          "source_chars": [len(s) for s in sources], "device": Device.DEFAULT}

def main():
  packed_bytes = N*K//256*Q6_K_BLOCK_BYTES
  raw = Tensor.empty(packed_bytes//2, dtype=dtypes.uint16).contiguous().realize()
  raw8 = raw.bitcast(dtypes.uint8)
  x = Tensor.randn(M, K).cast(dtypes.float16).contiguous().realize()
  staged = describe_q6k_wmma_prefill(M, N, K, role="ffn_down")
  direct = describe_q6k_packed_prefill(N, K, M, role="ffn_down", output_layout="direct_out")
  staged_out = emit_q6k_wmma_prefill(raw8, x, staged)
  direct_out = Tensor.empty(M, N, dtype=dtypes.float32).custom_kernel(
    raw, x.reshape(M*K), fxn=emit_q6k_packed_prefill_kernel(direct))[0]
  staged_compile = compile_facts(staged_out)
  direct_compile = compile_facts(direct_out)
  weight = q6_k_reference(raw8, N*K).reshape(N, K).cast(dtypes.float16).contiguous()
  material = timed(lambda: q6_k_reference(raw8, N*K).reshape(N, K).cast(dtypes.float16).contiguous())
  contraction_t = timed(lambda: x.matmul(weight.transpose(), dtype=dtypes.float32).contiguous())
  # Fresh packed storage is deliberate: reusing raw8 would let the materializer
  # cache turn the combined measurement into contraction-only time.
  combined = timed(lambda: emit_q6k_wmma_prefill(
    Tensor.empty(packed_bytes//2, dtype=dtypes.uint16).bitcast(dtypes.uint8), x, staged))
  direct_t = timed(lambda: Tensor.empty(M, N, dtype=dtypes.float32).custom_kernel(
    raw, x.reshape(M*K), fxn=emit_q6k_packed_prefill_kernel(direct))[0])
  result = {"schema":"q6k_8b_like_fixture_ffn_down_wmma_bench.v1", "research_only":True,
    "authority":"experimental_fixture_not_qwen3_14b",
    "route_default":"direct_packed", "fallback_gated":True, "shape":{"M":M,"N":N,"K":K},
    "hardware":platform.platform(), "device":Device.DEFAULT,
    "memory":{"packed_bytes":N*K//256*Q6_K_BLOCK_BYTES, "fp16_weight_bytes":N*K*2,
               "activation_fp16_bytes":M*K*2, "output_fp32_bytes":M*N*4,
               "staged_live_bytes":N*K*2 + M*K*2 + M*N*4},
    "staged":{"materialization":material, "contraction":contraction_t, "combined":combined,
              "compile":staged_compile},
    "direct_packed":{"contraction":direct_t, "compile":direct_compile},
    "resource_constraints":{"max_vgpr":"not exposed by compile artifact", "lds_bytes":"not exposed by compile artifact",
                             "spills":"not observed by host harness; inspect final code object"}}
  print(json.dumps(result, indent=2, sort_keys=True))
  out = Path(os.environ.get("Q6K_BENCH_OUTPUT", "bench/q6k-8b-like-fixture-ffn-down-wmma/latest.json")); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")

if __name__ == "__main__": main()
