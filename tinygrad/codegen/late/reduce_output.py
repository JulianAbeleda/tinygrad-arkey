"""Cooperative reduction-to-output UOp bodies.

These are normal UOp SINK programs.  No source strings, inline assembly, or
custom-kernel Tensor transport are used.  Admission is owned by rangeify,
which binds only concrete identity-preserving views into the ordinary CALL.

The body is fully spec-driven: the reduce op composes with the warp/lane
``_LADDER``, the warp/lane/per-lane association mirrors the ordinary reduce
shape, and the recipe string selects the per-lane accumulation and epilogue.
The 08-05 single-recipe body is the r_16_256 special case (ADD, 16 warps / 32
lanes / 8 per lane, ``sumsq_rsqrt_affine``) and stays byte-identical: same
program name and body digest.  Any shape/recipe the builder cannot express
exactly fails closed with ValueError, which the rangeify selector maps to the
same reject path as the legacy emitter.
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
  tile one row's reduction as ``(warps, per_warp)``: the outermost loop is the
  cooperative warp count and the remaining extent is distributed across
  ``lanes`` lanes per warp.  The 08-05 fixed body (16 warps, 32 lanes, 8 per
  lane) is exactly the derivation of r_16_256 for dim 4096.
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
    red = UOp.range(per_lane, 2, AxisType.REDUCE)
    base = warp * (per_lane * lane) + laneid + red * lane
    xv = x[base].cast(dtypes.float32)
    acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    if sumsq:
      acc = acc.after(acc[0].store(acc.after(red)[0] + xv*xv).end(red))
    else:
      acc = acc.after(acc[0].store(acc.after(red)[0].maximum(xv.abs())).end(red))
    warp_total = _LADDER[spec.reduce_op](acc[0], laneid, lane, slot_base=90)
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
