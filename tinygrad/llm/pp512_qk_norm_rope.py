"""Research-only pp512 Q/K head RMSNorm+RoPE contract.

This is deliberately not a production route.  The installed fused emitter is
an exact per-token head primitive (Q rows=32, K rows=8); pp512 admission must
tile the leading sequence dimension rather than silently treating 512 tokens
as extra heads.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

from tinygrad import dtypes
from tinygrad import Tensor
from tinygrad.codegen.late.reduce_output import ReduceOutputSpec
from tinygrad.llm.qk_norm_rope_mmvq import emit_reduce_output_rope
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tinygrad.uop.ops import Ops


@dataclass(frozen=True)
class PP512QKNormRopeAdmission:
  """Closed research lease for exact Q/K pp512 tensors."""
  sequence: int = 512
  head_dim: int = 128
  q_heads: int = 32
  kv_heads: int = 8

  def __post_init__(self):
    debug = os.getenv("NV_PP512_QK_DEBUG") == "1"
    if not (debug and self.head_dim == 128 and self.sequence in (1, 2, 16, 512) and self.q_heads in (1, 32) and self.kv_heads in (1, 8)) and \
       (self.sequence, self.head_dim, self.q_heads, self.kv_heads) != (512, 128, 32, 8):
      raise ValueError("pp512 Q/K norm+RoPE requires sequence=512, q_heads=32, kv_heads=8, head_dim=128")

  def tile_shape(self, key: str) -> tuple[int, int, int]:
    if key == "q": return (self.sequence, self.q_heads, self.head_dim)
    if key == "k": return (self.sequence, self.kv_heads, self.head_dim)
    raise ValueError(f"unsupported pp512 Q/K key {key!r}")

  def tiles(self, key: str) -> int:
    self.tile_shape(key)
    return self.sequence


def emit_pp512_qk_norm_rope(admission: PP512QKNormRopeAdmission, key: str):
  """Return the production NVRTC UOp emitter for one pp512 sequence tile.

  The caller launches this emitter once per sequence row; keeping the tile
  boundary explicit preserves the installed Q/K head geometry and cache-ready
  contiguous layout.
  """
  if not isinstance(admission, PP512QKNormRopeAdmission): raise TypeError("invalid pp512 Q/K admission")
  rows = admission.q_heads if key == "q" else admission.kv_heads if key == "k" else 0
  if not rows: raise ValueError(f"unsupported pp512 Q/K key {key!r}")
  spec = ReduceOutputSpec(rows=rows, dim=admission.head_dim, eps=1e-6, out_dtype=dtypes.float32,
                          affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                          warps=rows, lanes=32, per_lane=4, epilogue="rope")
  return emit_reduce_output_rope(spec, dtypes.float32, dtypes.float16)


def emit_pp512_qk_norm_rope_sequence(admission: PP512QKNormRopeAdmission, key: str):
  """Emit one sequence-aware launch for the complete contiguous pp512 tensor.

  The first global axis selects the head and the second selects the sequence
  row.  The ABI is head-major ``(1, heads, 512, 128)`` as consumed by Flash.
  """
  if not isinstance(admission, PP512QKNormRopeAdmission): raise TypeError("invalid pp512 Q/K admission")
  rows = admission.q_heads if key == "q" else admission.kv_heads if key == "k" else 0
  if not rows: raise ValueError(f"unsupported pp512 Q/K key {key!r}")
  dim = admission.head_dim
  spec = ReduceOutputSpec(rows=rows, dim=dim, eps=1e-6, out_dtype=dtypes.float32,
                          affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                          warps=rows, lanes=32, per_lane=4, epilogue="rope")
  # Reuse the exact body association, changing only the leading contiguous
  # sequence offset and adding a global sequence axis.
  from tinygrad.codegen.late.reduce_output import _NV_MULTI_ROW_ASSOC
  assoc = _NV_MULTI_ROW_ASSOC.get((rows, dim), (1, 4, 1, 32))
  lane, P, S, t_stride, s_stride = spec.lanes, *assoc
  half = spec.per_lane // 2
  def kernel(out: UOp, x: UOp, weight: UOp, freqs: UOp) -> UOp:
    laneid = UOp.range(lane, 0, AxisType.LOCAL)
    tile = UOp.range(rows * admission.sequence, 0, AxisType.GLOBAL)
    row = tile // admission.sequence
    seq = tile % admission.sequence
    seqbase = row * (admission.sequence * dim) + seq * dim
    partial_lane = laneid % P
    red = UOp.range(S, 2, AxisType.REDUCE)
    base = seqbase + row * dim + partial_lane * t_stride + red * s_stride
    xv = x[base].cast(dtypes.float32)
    acc = UOp.placeholder((1,), dtypes.float32, 20, AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(red)[0] + xv * xv).end(red))
    smem = UOp.placeholder((P,), dtypes.float32, 230, AddrSpace.LOCAL)
    published = smem[partial_lane].store(acc[0], laneid < P)
    ready = UOp.barrier(UOp.group(published))
    total = UOp.const(dtypes.float32, 0.0)
    for ti in range(P): total = total + smem.after(ready)[ti]
    scale = (total / UOp.const(dtypes.float32, float(dim)) + UOp.const(dtypes.float32, spec.eps)).sqrt().reciprocal()
    epi = UOp.range(half, 2, AxisType.LOOP)
    lo_base = seqbase + row * dim + laneid + epi * lane
    hi_base = seqbase + row * dim + laneid + (epi + half) * lane
    w_lo, w_hi = laneid + epi * lane, laneid + (epi + half) * lane
    v_lo = ((x[lo_base].cast(dtypes.float32) * scale).cast(dtypes.float32) * weight[w_lo].cast(dtypes.float32))
    v_hi = ((x[hi_base].cast(dtypes.float32) * scale).cast(dtypes.float32) * weight[w_hi].cast(dtypes.float32))
    h = laneid + epi * lane
    fbase = seq * dim
    cosv, sinv = freqs[fbase + h].cast(dtypes.float32), freqs[fbase + h + dim // 2].cast(dtypes.float32)
    lo_store = out[lo_base].store(v_lo * cosv - v_hi * sinv)
    hi_store = out[hi_base].store(v_hi * cosv + v_lo * sinv)
    return UOp.group(lo_store, hi_store).end(laneid, tile, epi).sink(
      arg=KernelInfo(name=f"pp512_{key}_qk_norm_rope_sequence", opts_to_apply=()))
  return kernel


def pp512_qk_norm_rope_sequence_call(admission, key: str, x: Tensor, weight: Tensor, freqs: Tensor) -> Tensor:
  """Bind the sequence emitter to contiguous head-major [1, heads, 512, 128] buffers."""
  if not isinstance(admission, PP512QKNormRopeAdmission): raise TypeError("invalid pp512 Q/K admission")
  rows = admission.q_heads if key == "q" else admission.kv_heads
  expected = (1, rows, admission.sequence, admission.head_dim)
  if key not in ("q", "k") or tuple(x.shape) != expected:
    raise ValueError(f"pp512 {key!r} tensor must have shape {expected}")
  if tuple(weight.shape) != (admission.head_dim,) or tuple(freqs.shape) != (admission.sequence, admission.head_dim):
    raise ValueError("pp512 Q/K norm+RoPE requires weight [128] and freqs [512,128] buffers")
  if str(x.device) != str(weight.device) or str(x.device) != str(freqs.device):
    raise ValueError("pp512 Q/K norm+RoPE inputs must share a device")
  out_elems = x.numel() * (2 if os.getenv("NV_PP512_QK_DEBUG_PAD") == "1" else 1)
  program = KernelProgram("pp512_qk_norm_rope_sequence", key, KernelProgramProvenance.RESEARCH_ONLY,
                          emit_pp512_qk_norm_rope_sequence(admission, key),
                          output_spec=OutputSpec((out_elems,), dtypes.float32))
  out = execute_research_program(Tensor.empty((out_elems,), dtype=dtypes.float32, device=x.device),
                                 x.reshape(x.numel()), weight, freqs.reshape(admission.sequence * admission.head_dim), program=program)
  if os.getenv("NV_PP512_QK_DEBUG_RETURN_PAD") == "1": return out
  return out[:x.numel()].reshape(x.shape)


def validate_pp512_qk_shapes(admission: PP512QKNormRopeAdmission, q_shape, k_shape) -> None:
  if not isinstance(admission, PP512QKNormRopeAdmission): raise TypeError("invalid pp512 Q/K admission")
  if tuple(q_shape) != admission.tile_shape("q") or tuple(k_shape) != admission.tile_shape("k"):
    raise ValueError(f"pp512 Q/K shapes must be {admission.tile_shape('q')} and {admission.tile_shape('k')}")


__all__ = ["PP512QKNormRopeAdmission", "validate_pp512_qk_shapes", "emit_pp512_qk_norm_rope",
           "emit_pp512_qk_norm_rope_sequence", "pp512_qk_norm_rope_sequence_call"]
