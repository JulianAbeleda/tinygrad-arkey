"""Cooperative reduction-to-output UOp bodies.

These are normal UOp SINK programs.  No source strings, inline assembly, or
custom-kernel Tensor transport are used.  Admission is owned by rangeify,
which binds only concrete identity-preserving views into the ordinary CALL.

The body is fully spec-driven and reproduces the ORDINARY reduce association
bitwise: the ordinary r_16_256 kernel is 16 threads, each serially summing 256
CONTIGUOUS elements, then a serial chain over the 16 partials.  The fused body
mirrors that exact order: every lane computes the same per-warp serial chain
over ``per_lane * lane`` contiguous elements (the reduce index is
lane-independent), lane 0 publishes the per-warp partial, and the serial
cross-warp chain plus the elementwise epilogue follow.  The epilogue still
uses all lanes (``per_lane`` elements per lane), so the launch stays at the
full 512-thread width while the reduction is bitwise-equal to the ordinary
program for every dtype mix.  The recipe string selects the per-lane
accumulation and epilogue (``sumsq_rsqrt_affine`` is the shipped legacy
RMSNorm recipe; ``max_affine`` is the MAX-reduce affine variant).  Any
shape/recipe the builder cannot express exactly fails closed with ValueError,
which the rangeify selector maps to the same reject path as the legacy
emitter.
"""
from math import prod
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo, ReduceOutputSpec
from tinygrad.codegen.late.warp_reduce import _LADDER

# recipe -> program-name stem.  The legacy RMSNorm recipe keeps its historical
# name so the 08-05 body pin (reduce_output_rmsnorm_*) does not move.
_RECIPE_STEMS = {"sumsq_rsqrt_affine": "rmsnorm", "max_affine": "max"}

def reduce_output_association(shape: tuple[int, ...], lanes: int = 32) -> tuple[int, int, int]:
  """Derive (warps, lanes, per_lane) from an ordinary reduce loop nest.

  The decode DAG's ordinary reduce programs (r_16_256, r_2_8_4_4_16, r_8_16_8)
  tile one row's reduction as ``(warps, per_warp)``: the outermost extent is
  the per-thread serial chain count and ``per_warp`` is the number of
  CONTIGUOUS elements each thread sums serially (r_16_256 = 16 threads x 256
  contiguous serial).  The fused body keeps that exact serial chain per warp
  and uses ``lanes`` only to widen the epilogue, so the derivation is
  (warps, lanes, per_lane = per_warp // lanes) with the reduce phase summing
  ``per_warp`` contiguous elements serially.
  """
  if len(shape) < 2: raise ValueError(f"reduce-output association needs (warps, per_warp), got {shape!r}")
  warps, per_warp = shape[0], prod(shape[1:])
  if per_warp % lanes: raise ValueError(f"per-warp extent {per_warp} is not divisible by {lanes} lanes")
  return warps, lanes, per_warp // lanes

def emit_reduce_output(spec:ReduceOutputSpec, x_dtype, weight_dtype):
  """Generic cooperative reduction-to-output body derived entirely from ``spec``."""
  if spec.recipe not in _RECIPE_STEMS or not spec.affine: raise ValueError(f"unsupported reduce-output recipe {spec.recipe!r}")
  if spec.reduce_op not in _LADDER: raise ValueError(f"unsupported reduce-output op {spec.reduce_op}")
  if spec.rows != 1 or spec.dim < 32 or spec.dim % 512: raise ValueError("reduce-output requires one row and dim divisible by 512")
  lane, warps, per_lane = spec.lanes, spec.warps, spec.per_lane
  if warps * lane * per_lane != spec.dim: raise ValueError(f"association {warps}x{lane}x{per_lane} does not cover dim {spec.dim}")
  if x_dtype not in (dtypes.float16, dtypes.float32) or weight_dtype not in (dtypes.float16, dtypes.float32):
    raise ValueError("reduce-output requires fp16/fp32 inputs")
  dim, lane, warps, per_lane, sumsq = spec.dim, lane, warps, per_lane, spec.recipe == "sumsq_rsqrt_affine"
  def kernel(out:UOp, x:UOp, weight:UOp) -> UOp:
    laneid = UOp.range(lane, 0, AxisType.LOCAL)
    warp = UOp.range(warps, 1, AxisType.LOCAL)
    # Ordinary r_16_256 association: each thread serially sums per_lane*lane
    # CONTIGUOUS elements, then a serial chain combines the per-warp partials.
    # The reduce index is lane-independent so every lane of a warp computes the
    # identical serial chain; only lane 0 publishes it.  This is bitwise-equal
    # to the ordinary kernel, unlike a strided per-lane split with a shuffle
    # ladder, whose fp32 association differs in the last ulp.
    red = UOp.range(per_lane * lane, 2, AxisType.REDUCE)
    base = warp * (per_lane * lane) + red
    xv = x[base].cast(dtypes.float32)
    acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    if sumsq:
      acc = acc.after(acc[0].store(acc.after(red)[0] + xv*xv).end(red))
    else:
      acc = acc.after(acc[0].store(acc.after(red)[0].maximum(xv.abs())).end(red))
    # Every lane holds the same serial-chain value; no cross-lane ladder.
    warp_total = acc[0]
    smem = UOp.placeholder((warps,), dtypes.float32, 230, AddrSpace.LOCAL)
    published = smem[warp].store(warp_total, laneid.eq(0))
    ready = UOp.barrier(UOp.group(published))
    total = UOp.const(dtypes.float32, 0.0)
    for wi in range(warps):
      total = (total + smem.after(ready)[wi]) if sumsq else total.maximum(smem.after(ready)[wi])
    if sumsq:
      scale = (total / UOp.const(dtypes.float32, float(dim)) + UOp.const(dtypes.float32, spec.eps)).sqrt().reciprocal()
    else:
      # MAX recipe: affine epilogue over the warp-reduced max-abs; |xv| >= 0
      # keeps the zero-initialized accumulator exact.
      scale = (total + UOp.const(dtypes.float32, spec.eps)).reciprocal()
    # Lane restoration: reuse the same local ids after the reduction barrier;
    # only the serial per-lane phase changes from REDUCE to LOOP ownership.
    epi = UOp.range(per_lane, 2, AxisType.LOOP)
    obase = warp * (per_lane * lane) + laneid + epi * lane
    normed = (x[obase].cast(dtypes.float32) * scale).cast(x_dtype)
    value = (normed * weight[obase].cast(x_dtype)).cast(spec.out_dtype)
    return out[obase].store(value).end(laneid, warp, epi).sink(
      arg=KernelInfo(name=f"reduce_output_{_RECIPE_STEMS[spec.recipe]}_{spec.rows}_{dim}", opts_to_apply=()))
  return kernel

def emit_reduce_output_rmsnorm(spec:ReduceOutputSpec, x_dtype, weight_dtype):
  """Legacy entry point: the sumsq/ADD recipe only (08-05 body pin)."""
  if spec.recipe != "sumsq_rsqrt_affine" or spec.reduce_op is not Ops.ADD:
    raise ValueError("emit_reduce_output_rmsnorm is the legacy sumsq ADD recipe only")
  return emit_reduce_output(spec, x_dtype, weight_dtype)
