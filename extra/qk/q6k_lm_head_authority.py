"""Isolated Q6_K LM-head prefill authority.

This deliberately compares the generated direct-output route with the older
packed-load partials route on the exact 8B lm_head shape, without constructing
the transformer.  Both consume the same packed weights and activation tensor;
the second route is retained as the independent correctness comparator.
"""
from __future__ import annotations
import argparse, json, os, time
from tinygrad import Tensor, dtypes, Device
from extra.qk.layout import Q6K_HALFWORDS_PER_BLOCK, Q6_K_BLOCK_ELEMS
from extra.qk.q6k_prefill_route_spec import describe_q6k_packed_prefill, emit_q6k_packed_prefill_kernel
from extra.qk.quant.q6_k_gemv_primitive import q6k_gemm_packed_load_kernel

M, N, K = 512, 151936, 4096

def run(*, warmups=4, rounds=3, device=None):
  dev = device or Device.DEFAULT
  blocks = K // Q6_K_BLOCK_ELEMS
  halfs = Tensor.zeros((N * blocks * Q6K_HALFWORDS_PER_BLOCK,), dtype=dtypes.uint16, device=dev).realize()
  x = Tensor.ones((M * K,), dtype=dtypes.float16, device=dev).realize()
  spec = describe_q6k_packed_prefill(N, K, M, role="lm_head", parts=1,
                                     output_layout="direct_out", opts=("UPCAST:1:4",))
  def make_generated():
    return Tensor.empty(M, N, dtype=dtypes.float32, device=dev).custom_kernel(
      halfs, x, fxn=emit_q6k_packed_prefill_kernel(spec))[0]
  generated = make_generated()
  # Independent comparator: legacy packed-load partials route, then reduction.
  partials = Tensor.empty(N, M, 1, dtype=dtypes.float32, device=dev)
  ref = partials.custom_kernel(halfs, x, fxn=q6k_gemm_packed_load_kernel(
    N, K, M, 1, (), name="q6k_lm_head_reference"))[0].sum(axis=2).transpose(0, 1)
  generated.realize(); ref.realize()  # compile excluded from timed rounds
  diff = (generated - ref).abs().max().item()
  times = []
  for _ in range(warmups): make_generated().realize(); Device[dev].synchronize()
  for _ in range(rounds):
    Device[dev].synchronize(); t0 = time.perf_counter(); make_generated().realize(); Device[dev].synchronize()
    times.append((time.perf_counter() - t0) * 1e3)
  return {"schema":"q6k-lm-head-prefill-authority.v1", "shape":{"M":M,"N":N,"K":K},
          "route":{"kernel":spec.kernel_name,"role":"lm_head","output_layout":"direct_out","parts":1},
          "reference":{"kernel":"q6k_lm_head_reference","max_abs":diff,"passed":diff <= 1e-3},
          "timing":{"warmups":warmups,"rounds":rounds,"median_ms":sorted(times)[len(times)//2],"samples_ms":times},
          "resources":{"packed_weight_bytes":int(halfs.nbytes()),"output_bytes":M*N*4}, "device":dev}

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--warmups", type=int, default=4); ap.add_argument("--rounds", type=int, default=3)
  ap.add_argument("--device", default=None); args = ap.parse_args()
  print(json.dumps(run(warmups=args.warmups, rounds=args.rounds, device=args.device), indent=2))
