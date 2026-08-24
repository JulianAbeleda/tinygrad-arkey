"""Closed-default fused Q/K norm+rope admission candidate (research only).

The installed decode route emits, per attention block, one cooperative
``reduce_output_rmsnorm_{32,8}_128`` kernel for the Q/K head norm followed by a
separate 4096-thread ``apply_rope`` elementwise kernel.  This candidate folds
the full-head rotary rotation into the reduce-output epilogue, so the norm and
rope become one kernel per head (144 kernels -> 72 kernels across 36 blocks).

The fused body is bit-exact against the installed norm -> apply_rope chain under
NVRTC (the compiler production actually uses); see
``extra/llm_research/decode/nv_qk_norm_rope_nvrtc_bit_exact.py``.

The route is unreachable unless a research harness attaches a
``QKNormRopeAdmission`` to a concrete block.  It is NOT wired into any
production call site.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.reduce_output import _NV_MULTI_ROW_ASSOC, ReduceOutputSpec
from tinygrad.dtype import AddrSpace
from tinygrad.llm.kernel_program import (
  KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program)
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


@dataclass(frozen=True)
class QKNormRopeAdmission:
  block_index: int

  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("Q/K norm+rope block index must be a non-negative integer")


def emit_reduce_output_rope(spec: ReduceOutputSpec, x_dtype, weight_dtype):
  """Fused reduce-output RMSNorm + full-head rotary epilogue.

  The reduction phase is a verbatim copy of ``emit_reduce_output``'s NV
  multi-row association, so the sumsq/scale is bitwise-identical to the
  installed kernel.  Only the epilogue changes: after computing each normed
  value it applies the half-rotate
  ``(v_lo*cos - v_hi*sin, v_hi*cos + v_lo*sin)`` before storing.  per_lane is
  4, so the epilogue is unrolled over the two (lo, hi=lo+2) pairs; the partner
  element lives at the same lane id, two epi slots away, so no cross-lane
  shuffle is required.
  """
  if spec.rows not in (8, 32) or spec.dim != 128:
    raise ValueError("reduce-output rope requires rows in (8,32) and dim 128")
  if spec.recipe != "sumsq_rsqrt_affine" or not spec.affine:
    raise ValueError("reduce-output rope requires the sumsq affine recipe")
  if spec.warps != spec.rows or spec.lanes != 32 or spec.per_lane != 4:
    raise ValueError("reduce-output rope requires warps==rows, 32 lanes, 4 per_lane")
  lane, per_lane, dim = spec.lanes, spec.per_lane, spec.dim
  P, S, t_stride, s_stride = _NV_MULTI_ROW_ASSOC[(spec.rows, dim)]
  half = per_lane // 2

  def kernel(out: UOp, x: UOp, weight: UOp, freqs: UOp) -> UOp:
    laneid = UOp.range(lane, 0, AxisType.LOCAL)
    row = UOp.range(spec.rows, 0, AxisType.GLOBAL)
    partial_lane = laneid % P
    red = UOp.range(S, 2, AxisType.REDUCE)
    base = row * dim + partial_lane * t_stride + red * s_stride
    xv = x[base].cast(dtypes.float32)
    acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(red)[0] + xv * xv).end(red))
    smem = UOp.placeholder((P,), dtypes.float32, 230, AddrSpace.LOCAL)
    published = smem[partial_lane].store(acc[0], laneid < P)
    ready = UOp.barrier(UOp.group(published))
    total = UOp.const(dtypes.float32, 0.0)
    for ti in range(P):
      total = total + smem.after(ready)[ti]
    scale = (total / UOp.const(dtypes.float32, float(dim)) + UOp.const(dtypes.float32, spec.eps)).sqrt().reciprocal()

    epi = UOp.range(half, 2, AxisType.LOOP)
    lo_base = row * dim + laneid + epi * lane
    hi_base = row * dim + laneid + (epi + half) * lane
    w_lo = laneid + epi * lane
    w_hi = laneid + (epi + half) * lane
    v_lo = ((x[lo_base].cast(dtypes.float32) * scale).cast(x_dtype)
            * weight[w_lo].cast(x_dtype)).cast(spec.out_dtype)
    v_hi = ((x[hi_base].cast(dtypes.float32) * scale).cast(x_dtype)
            * weight[w_hi].cast(x_dtype)).cast(spec.out_dtype)
    h = laneid + epi * lane
    cosv = freqs[h].cast(dtypes.float32)
    sinv = freqs[h + dim // 2].cast(dtypes.float32)
    lo_store = out[lo_base].store(v_lo * cosv - v_hi * sinv)
    hi_store = out[hi_base].store(v_hi * cosv + v_lo * sinv)
    return UOp.group(lo_store, hi_store).end(laneid, row, epi).sink(
      arg=KernelInfo(name=f"reduce_output_rmsnorm_rope_{spec.rows}_128", opts_to_apply=()))
  return kernel


def qk_norm_rope_call(admission, block, key: str, pre_norm: Tensor, norm, freqs_cis: Tensor) -> Tensor | None:
  """Return the fused norm+rope head tensor, or None to keep the installed chain."""
  if not isinstance(admission, QKNormRopeAdmission): return None
  if key not in ("q", "k"): return None
  if not str(pre_norm.device).startswith("NV"): return None
  if pre_norm.dtype != dtypes.float32: return None
  shape = pre_norm.shape
  rows = 32 if key == "q" else 8
  if shape != (1, rows, 1, 128): return None
  if freqs_cis.dtype != dtypes.float32 or freqs_cis.shape != (1, 128): return None

  weight = getattr(norm, "_decode_reduce_output_weight", None)
  if weight is None or weight.dtype != dtypes.float16 or weight.shape != (128,): return None

  spec = ReduceOutputSpec(rows=rows, dim=128, eps=float(norm.eps), out_dtype=dtypes.float32,
                          affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                          warps=rows, lanes=32, per_lane=4)
  x_flat = pre_norm.reshape(rows * 128)
  f_flat = freqs_cis.reshape(128)
  program = KernelProgram("decode_qk_norm_rope", f"blk{admission.block_index}.{key}",
                          KernelProgramProvenance.RESEARCH_ONLY,
                          emit_reduce_output_rope(spec, dtypes.float32, dtypes.float16),
                          output_spec=OutputSpec((rows * 128,), dtypes.float32))
  out = execute_research_program(Tensor.empty((rows * 128,), dtype=dtypes.float32, device=pre_norm.device),
                                 x_flat, weight, f_flat, program=program)
  return out.reshape(shape)


__all__ = ["QKNormRopeAdmission", "emit_reduce_output_rope", "qk_norm_rope_call"]
