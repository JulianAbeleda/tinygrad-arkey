"""Closed-default producer-owned K/V cache sink qualification route.

Ordinary model loads never attach ``ProducerKVCacheSinkAdmission``. A research
harness may lease the exact Qwen3-8B decode boundary so the terminal K
RMSNorm+RoPE producer writes K and final V directly into the current cache
slot, replacing the generic post-producer store launch.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor, dtypes
from tinygrad.codegen.late.reduce_output import emit_reduce_output_rope_kv_cache
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_promoted_program
from tinygrad.uop.ops import Ops, ReduceOutputSpec
from extra.llm_research.boltbeam_authority import tickets_for_candidate


@dataclass(frozen=True)
class ProducerKVCacheSinkAdmission:
  block_index: int

  def __post_init__(self):
    if not isinstance(self.block_index, int) or isinstance(self.block_index, bool) or self.block_index < 0:
      raise ValueError("producer K/V cache-sink block index must be a non-negative integer")


def _flat_producer_output(value:Tensor) -> Tensor|None:
  """Return an exact offset-zero view of one concrete projection output.

  The opaque-program boundary otherwise materializes arbitrary movement
  expressions. Only equal-span semantic/reshape wrappers over an AFTER or a
  declared precompiled output are admitted here.
  """
  u, numel = value.uop, value.uop.numel()
  # Movement UOps may carry shape-descriptor sources after symbolic lowering;
  # src[0] remains the sole value source. Equal numel and the narrow op set are
  # the physical offset-zero proof, not fixed UOp arity.
  while u.op in (Ops.MEMORY_SEMANTIC, Ops.RESHAPE) and len(u.src) >= 1 and u.numel() == numel:
    u = u.src[0]
  if not (u.op is Ops.AFTER or u.has_precompiled_output_identity()): return None
  return Tensor(u, device=value.device)


def producer_kv_cache_sink_call(admission, cache:Tensor, k_input:Tensor, v_input:Tensor, norm,
                                freqs:Tensor, max_context:int) -> Tensor|None:
  """Return cache AFTER the producer-owned store, or None without graph changes."""
  if not isinstance(admission, ProducerKVCacheSinkAdmission): return None
  if not str(cache.device).startswith("NV") or cache.dtype not in (dtypes.float16, dtypes.float32): return None
  if cache.shape != (2, 1, 8, max_context, 128): return None
  if k_input.dtype != dtypes.float32 or v_input.dtype != dtypes.float32 or k_input.numel() != 1024 or v_input.numel() != 1024: return None
  if freqs.dtype != dtypes.float32 or freqs.shape != (max_context, 128): return None
  weight = getattr(norm, "_decode_reduce_output_weight", None)
  if weight is None or weight.dtype != dtypes.float16 or weight.shape != (128,): return None
  k_producer, v_producer = _flat_producer_output(k_input), _flat_producer_output(v_input)
  if k_producer is None or v_producer is None: return None

  spec = ReduceOutputSpec(rows=8, dim=128, eps=float(norm.eps), out_dtype=dtypes.float32,
                          affine=True, recipe="sumsq_rsqrt_affine", reduce_op=Ops.ADD,
                          warps=8, lanes=32, per_lane=4, epilogue="rope")
  program = KernelProgram("decode_producer_kv_cache_sink", f"blk{admission.block_index}.k_terminal_cache_sink",
                          KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED,
                          emit_reduce_output_rope_kv_cache(spec, k_producer.dtype, weight.dtype,
                                                           cache.dtype, max_context), output_spec=None,
                          boltbeam_ticket=tickets_for_candidate({"family":"qk_norm_rope_cache_sink.v1",
                            "max_context":max_context,"cache_dtype":str(cache.dtype)},
                            (("decode_producer_kv_cache_sink","qk_norm_rope_cache_sink"),)))
  return execute_promoted_program(cache, k_producer, weight, v_producer, freqs, program=program)


__all__ = ["ProducerKVCacheSinkAdmission", "producer_kv_cache_sink_call"]
