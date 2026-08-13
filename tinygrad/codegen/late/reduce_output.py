"""Cooperative reduction-to-output UOp bodies.

These are normal UOp SINK programs.  No source strings, inline assembly, or
custom-kernel Tensor transport are used.  Admission is owned by rangeify,
which binds only concrete identity-preserving views into the ordinary CALL.

The body is fully spec-driven and reproduces the ORDINARY reduce association
bitwise.  The single-row shapes mirror the ordinary r_16_256 kernel: 16
threads, each serially summing 256 CONTIGUOUS elements, then a serial chain
over the 16 partials.  The multi-row q/k shapes (rows 8/32 x dim 128) mirror
the ordinary r_8_16_8 / r_2_8_4_4_16 tiling instead: one 32-lane block per
row (grid = rows), per-row P lanes each serially sum S elements at
``(t*t_stride + r*s_stride)``, then a serial chain combines the P partials in
t order (see ``_NV_MULTI_ROW_ASSOC``).  That exact fp32 summation order is
what makes the fused logits bitwise-equal on NV; a plain per-row serial chain
differs in the last ulp and flips the full-logit SHA.  The single-row
epilogue keeps all lanes busy (``per_lane`` elements per lane), so that
launch stays at the full 512-thread width while the reduction is
bitwise-equal to the ordinary program for every dtype mix.  The recipe string
selects the per-lane accumulation and epilogue (``sumsq_rsqrt_affine`` is the
shipped legacy RMSNorm recipe; ``max_affine`` is the MAX-reduce affine
variant).  Any shape/recipe the builder cannot express exactly fails closed
with ValueError, which the rangeify selector maps to the same reject path as
the legacy emitter.
"""
from math import prod
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo, ReduceOutputSpec
from tinygrad.codegen.late.warp_reduce import _LADDER

# recipe -> program-name stem.  The legacy RMSNorm recipe keeps its historical
# name so the 08-05 body pin (reduce_output_rmsnorm_*) does not move.
_RECIPE_STEMS = {"sumsq_rsqrt_affine": "rmsnorm", "max_affine": "max"}

# NV ordinary reduce tiling for the fp32 q/k shapes: (P, S, t_stride,
# s_stride).  r_2_8_4_4_16 (q, 32 rows) uses 8 lanes x 16 serial stride-8
# elements; r_8_16_8 (k, 8 rows) uses 16 lanes x 8 serial contiguous elements.
# Both combine the P partials in a serial chain in t order.
_NV_MULTI_ROW_ASSOC = {(32, 128): (8, 16, 1, 8), (8, 128): (16, 8, 8, 1)}

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
  if spec.rows not in (1, 8, 32): raise ValueError(f"reduce-output requires rows in (1, 8, 32), got {spec.rows}")
  if spec.rows == 1:
    # Legacy single-row path (byte-identical): one row, dim divisible by 512.
    if spec.dim < 32 or spec.dim % 512: raise ValueError("reduce-output requires one row and dim divisible by 512")
  else:
    # Row-mode: warp w owns row w over the full dim-contiguous chain.
    if spec.dim not in (128, 4096): raise ValueError(f"reduce-output multi-row requires dim 128 or 4096, got {spec.dim}")
    if spec.warps != spec.rows: raise ValueError(f"reduce-output multi-row requires warps == rows, got warps {spec.warps} != rows {spec.rows}")
  lane, warps, per_lane = spec.lanes, spec.warps, spec.per_lane
  if warps * lane * per_lane != spec.rows * spec.dim: raise ValueError(f"association {warps}x{lane}x{per_lane} does not cover {spec.rows} rows x {spec.dim}")
  if x_dtype not in (dtypes.float16, dtypes.float32) or weight_dtype not in (dtypes.float16, dtypes.float32):
    raise ValueError("reduce-output requires fp16/fp32 inputs")
  dim, lane, warps, per_lane, sumsq = spec.dim, lane, warps, per_lane, spec.recipe == "sumsq_rsqrt_affine"
  nv_assoc = _NV_MULTI_ROW_ASSOC.get((spec.rows, spec.dim)) if spec.rows > 1 else None
  def kernel(out:UOp, x:UOp, weight:UOp) -> UOp:
    laneid = UOp.range(lane, 0, AxisType.LOCAL)
    # Row-mode (rows > 1): one independent block per row, so the row index is
    # a GLOBAL range (grid dim 0) and the block is the 32-lane LOCAL range.
    # The single-row path keeps its LOCAL warp range.
    if spec.rows > 1:
      row = UOp.range(spec.rows, 0, AxisType.GLOBAL)
    else:
      warp = UOp.range(warps, 1, AxisType.LOCAL)
    if nv_assoc is not None:
      # NV ordinary partial-chain association for this row shape.  P lanes
      # each serially sum S elements at (t*t_stride + r*s_stride); idle lanes
      # (laneid >= P) compute an in-bounds duplicate partial and never publish.
      P, S, t_stride, s_stride = nv_assoc
      partial_lane = laneid % P
      red = UOp.range(S, 2, AxisType.REDUCE)
      base = row * dim + partial_lane * t_stride + red * s_stride
      xv = x[base].cast(dtypes.float32)
      acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
      acc = acc.after(acc[0].store(0.0))
      if sumsq:
        acc = acc.after(acc[0].store(acc.after(red)[0] + xv*xv).end(red))
      else:
        acc = acc.after(acc[0].store(acc.after(red)[0].maximum(xv.abs())).end(red))
      # One slot per partial: the row's block owns its own (P,) slots, so no
      # cross-row clobbering is possible (grid-per-row geometry).
      smem = UOp.placeholder((P,), dtypes.float32, 230, AddrSpace.LOCAL)
      published = smem[partial_lane].store(acc[0], laneid < P)
      ready = UOp.barrier(UOp.group(published))
      # Serial combine in t order, exactly like the ordinary kernel's partial
      # chain; every lane of the warp computes the same total.
      total = UOp.const(dtypes.float32, 0.0)
      for ti in range(P):
        total = (total + smem.after(ready)[ti]) if sumsq else total.maximum(smem.after(ready)[ti])
    else:
      # Ordinary r_16_256 association: each thread serially sums per_lane*lane
      # CONTIGUOUS elements, then a serial chain combines the per-warp partials.
      # The reduce index is lane-independent so every lane of a warp computes the
      # identical serial chain; only lane 0 publishes it.  This is bitwise-equal
      # to the ordinary kernel, unlike a strided per-lane split with a shuffle
      # ladder, whose fp32 association differs in the last ulp.
      red = UOp.range(per_lane * lane, 2, AxisType.REDUCE)
      base = (warp if spec.rows == 1 else row) * (per_lane * lane) + red
      xv = x[base].cast(dtypes.float32)
      acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
      acc = acc.after(acc[0].store(0.0))
      if sumsq:
        acc = acc.after(acc[0].store(acc.after(red)[0] + xv*xv).end(red))
      else:
        acc = acc.after(acc[0].store(acc.after(red)[0].maximum(xv.abs())).end(red))
      # Every lane holds the same serial-chain value; no cross-lane ladder.
      warp_total = acc[0]
      # Single-row: one per-warp partial slot.  Row-mode: the block owns one
      # row, so a single readback slot suffices (no cross-row combine exists
      # across independent blocks).
      smem = UOp.placeholder((warps,) if spec.rows == 1 else (1,), dtypes.float32, 230, AddrSpace.LOCAL)
      published = (smem[warp] if spec.rows == 1 else smem[0]).store(warp_total, laneid.eq(0))
      ready = UOp.barrier(UOp.group(published))
      if spec.rows == 1:
        total = UOp.const(dtypes.float32, 0.0)
        for wi in range(warps):
          total = (total + smem.after(ready)[wi]) if sumsq else total.maximum(smem.after(ready)[wi])
      else:
        # Row-mode: the block reads back ONLY its own published row total.
        total = smem.after(ready)[0]
    if sumsq:
      scale = (total / UOp.const(dtypes.float32, float(dim)) + UOp.const(dtypes.float32, spec.eps)).sqrt().reciprocal()
    else:
      # MAX recipe: affine epilogue over the warp-reduced max-abs; |xv| >= 0
      # keeps the zero-initialized accumulator exact.
      scale = (total + UOp.const(dtypes.float32, spec.eps)).reciprocal()
    # Lane restoration: reuse the same local ids after the reduction barrier;
    # only the serial per-lane phase changes from REDUCE to LOOP ownership.
    epi = UOp.range(per_lane, 2, AxisType.LOOP)
    if spec.rows == 1:
      obase = warp * (per_lane * lane) + laneid + epi * lane
      wbase = obase
    else:
      # The weight is (dim,) and shared across rows; index the row-local
      # element (per_lane*lane == dim for the admitted multi-row shapes).
      obase = row * (per_lane * lane) + laneid + epi * lane
      wbase = laneid + epi * lane
    normed = (x[obase].cast(dtypes.float32) * scale).cast(x_dtype)
    value = (normed * weight[wbase].cast(x_dtype)).cast(spec.out_dtype)
    return out[obase].store(value).end(laneid, *((warp,) if spec.rows == 1 else (row,)), epi).sink(
      arg=KernelInfo(name=f"reduce_output_{_RECIPE_STEMS[spec.recipe]}_{spec.rows}_{dim}", opts_to_apply=()))
  return kernel

def emit_reduce_output_rmsnorm(spec:ReduceOutputSpec, x_dtype, weight_dtype):
  """Legacy entry point: the sumsq/ADD recipe only (08-05 body pin)."""
  if spec.recipe != "sumsq_rsqrt_affine" or spec.reduce_op is not Ops.ADD:
    raise ValueError("emit_reduce_output_rmsnorm is the legacy sumsq ADD recipe only")
  return emit_reduce_output(spec, x_dtype, weight_dtype)
