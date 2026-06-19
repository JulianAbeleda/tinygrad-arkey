#!/usr/bin/env python3
"""Q8L-2 — expressibility spike for decode_q4k_ffn_q8_sidechannel.

Can ONE tinygrad custom_kernel express the fused RMSNorm/apply producer that emits 3 outputs from the raw pre-norm
activation, with TWO reduction granularities?
  inputs:  x[N] fp32 (pre-norm), w[N] fp32 (norm weight)
  outputs: (1) fp normalized y[N]   = x * rsqrt(mean(x^2)+eps) * w   [per-ROW reduce]
           (2) q8 scales s[N/32]    = max(|y| over each 32)/127      [per-32 reduce over the normalized y]
           (3) q8 packed qs[N] int8 = round(y / s[blk]).clip(-128,127)
Oracles: nn.RMSNorm for (1); extra.qk_layout.q8_1_quantize(y) for (2)/(3).
Gate (scope Q8L-2): one kernel, no dense fallback, q8 scales+packed correct, compile/source sane.
This is expressibility only -- no perf, no route.

  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_q8_sidechannel_producer.py
"""
from __future__ import annotations
import sys
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, UOp
from extra.q4_k_gemv_primitive import _kernel_info
from extra.qk_layout import q8_1_quantize, Q8_1_BLOCK_ELEMS as BE

# --- Attempt A: scalar reduce + per-block reduce + multi-store, mirroring the q4k partial idiom ---
def producer_kernel_A(N:int, eps:float):
  nblk = N // BE
  def kernel(out_fp:UOp, out_scale:UOp, out_qs:UOp, x:UOp, w:UOp) -> UOp:
    # rinv via a reduce over all N into out_scale[nblk] (scratch) is messy; instead compute ss inline per element
    # using a REDUCE range and a scalar accumulator stored to a 1-elem slot, then read back.
    # stage 1: ss -> store to out_scale[0] as scratch (then reread). per-row.
    r1 = UOp.range(N, 0, axis_type=AxisType.REDUCE)
    sacc = out_scale[0].set(0.0)
    sacc = out_scale[0].set(sacc.after(r1)[0] + x[r1].cast(dtypes.float32)*x[r1].cast(dtypes.float32), end=r1)
    # rinv from the accumulated ss (read the scratch slot, after the reduce)
    rinv = ((sacc[0] / N) + eps).rsqrt()
    # stage 2: y[j] = x*rinv*w ; store fp ; (per-element, after stage1)
    j = UOp.range(N, 1)
    y = x[j].cast(dtypes.float32) * rinv * w[j].cast(dtypes.float32)
    st_fp = out_fp[j].store(y).end(j)
    # stage 3: per-32 max(|y|) -> scale ; block range + inner reduce
    b = UOp.range(nblk, 2)
    p = UOp.range(BE, 3, axis_type=AxisType.REDUCE)
    yi = x[b*BE+p].cast(dtypes.float32) * rinv * w[b*BE+p].cast(dtypes.float32)
    macc = out_scale[b].set(0.0)
    macc = out_scale[b].set((macc.after(p)[b]).maximum(yi.abs()), end=p)
    st_scale = out_scale[b].set(macc[b] / 127.0).end(b)
    # stage 4: qs[j] = round(y / scale[j//32]).clip ; after scale stored
    j2 = UOp.range(N, 4)
    yq = x[j2].cast(dtypes.float32) * rinv * w[j2].cast(dtypes.float32)
    sc = out_scale[j2 // BE]
    q = (yq / sc).round().clip(-128, 127).cast(dtypes.int8)
    st_qs = out_qs[j2].store(q).end(j2)
    return UOp.group(st_fp, st_scale, st_qs).sink(arg=_kernel_info(f"q8_sidechannel_producer_{N}", "none", ()))
  return kernel

def main():
  N = getenv("N", 4096); eps = 1e-6
  Tensor.manual_seed(7)
  x = Tensor.randn(N, dtype=dtypes.float32).contiguous().realize()
  w = Tensor.randn(N, dtype=dtypes.float32).contiguous().realize()
  # oracle fp norm
  ref_fp = (x * ((x*x).mean() + eps).rsqrt() * w).realize()
  ref_qs, ref_scale = q8_1_quantize(ref_fp)
  ref_qs, ref_scale = ref_qs.realize(), ref_scale.realize()
  out_fp = Tensor.empty(N, dtype=dtypes.float32)
  out_scale = Tensor.empty(N // BE, dtype=dtypes.float32)
  out_qs = Tensor.empty(N, dtype=dtypes.int8)
  try:
    res = Tensor.custom_kernel(out_fp, out_scale, out_qs, x, w, fxn=producer_kernel_A(N, eps))
    gfp, gsc, gqs = res[0].realize(), res[1].realize(), res[2].realize()
    fp_err = (gfp - ref_fp).abs().max().item()
    sc_err = (gsc - ref_scale).abs().max().item()
    qs_err = (gqs.cast(dtypes.int32) - ref_qs.cast(dtypes.int32)).abs().max().item()
    print(f"Q8L-2 expressibility: ONE KERNEL ran.")
    print(f"  fp_norm max|err| = {fp_err:.3e}  (oracle nn.RMSNorm)")
    print(f"  q8_scale max|err| = {sc_err:.3e}  (oracle q8_1_quantize)")
    print(f"  q8_qs max|int err| = {qs_err}  (oracle q8_1_quantize)")
    ok = fp_err < 1e-4 and sc_err < 1e-4 and qs_err <= 1
    print(f"  VERDICT: {'PASS (expressible+correct)' if ok else 'RAN but INCORRECT'}")
  except Exception as e:
    import traceback; traceback.print_exc()
    print(f"\nQ8L-2 expressibility: FAILED at construction/compile -> {type(e).__name__}: {str(e)[:240]}")

if __name__ == "__main__":
  main()
