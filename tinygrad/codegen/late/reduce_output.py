"""Cooperative reduction-to-output UOp bodies.

These are normal UOp SINK programs.  No source strings, inline assembly, or
custom-kernel Tensor transport are used.  Admission is owned by rangeify,
which binds only concrete identity-preserving views into the ordinary CALL.
"""
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo, ReduceOutputSpec
from tinygrad.codegen.late.warp_reduce import _warp_reduce_sum_staged

def emit_reduce_output_rmsnorm(spec:ReduceOutputSpec, x_dtype, weight_dtype):
  if spec.recipe != "sumsq_rsqrt_affine" or not spec.affine: raise ValueError("unsupported reduce-output recipe")
  if spec.rows != 1 or spec.dim < 32 or spec.dim % 512: raise ValueError("reduce-output RMSNorm requires one row and dim divisible by 512")
  if x_dtype not in (dtypes.float16, dtypes.float32) or weight_dtype not in (dtypes.float16, dtypes.float32):
    raise ValueError("reduce-output RMSNorm requires fp16/fp32 inputs")
  dim, lane, warps = spec.dim, 32, 16
  per_lane = dim // (lane * warps)
  def kernel(out:UOp, x:UOp, weight:UOp) -> UOp:
    laneid = UOp.range(lane, 0, AxisType.LOCAL)
    warp = UOp.range(warps, 1, AxisType.LOCAL)
    red = UOp.range(per_lane, 2, AxisType.REDUCE)
    base = warp * (per_lane * lane) + laneid + red * lane
    xv = x[base].cast(dtypes.float32)
    acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(red)[0] + xv*xv).end(red))
    warp_total = _warp_reduce_sum_staged(acc[0], laneid, lane, slot_base=90)
    smem = UOp.placeholder((warps,), dtypes.float32, 230, AddrSpace.LOCAL)
    published = smem[warp].store(warp_total, laneid.eq(0))
    ready = UOp.barrier(UOp.group(published))
    total = UOp.const(dtypes.float32, 0.0)
    for wi in range(warps): total = total + smem.after(ready)[wi]
    scale = (total / UOp.const(dtypes.float32, float(dim)) + UOp.const(dtypes.float32, spec.eps)).sqrt().reciprocal()
    # Lane restoration: reuse the same local ids after the reduction barrier;
    # only the serial per-lane phase changes from REDUCE to LOOP ownership.
    epi = UOp.range(per_lane, 2, AxisType.LOOP)
    obase = warp * (per_lane * lane) + laneid + epi * lane
    normed = (x[obase].cast(dtypes.float32) * scale).cast(x_dtype)
    value = (normed * weight[obase].cast(x_dtype)).cast(spec.out_dtype)
    return out[obase].store(value).end(laneid, warp, epi).sink(
      arg=KernelInfo(name=f"reduce_output_rmsnorm_{spec.rows}_{dim}", opts_to_apply=()))
  return kernel
