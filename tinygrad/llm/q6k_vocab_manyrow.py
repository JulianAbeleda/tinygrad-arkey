"""Research-only many-row Q8-producer + packed Q6_K vocabulary primitive.

This module deliberately has no model or admission-policy integration.  A harness
must attach ``Q6KVocabManyRowAdmission`` and call ``q6k_vocab_manyrow_call``.
The producer owns the Q8 packet; the consumer owns the canonical Q6_K weights
and materializes every vocabulary logit.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.llm.q4k_ffn_down_mmvq import emit_q8_provider
from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, q6k_spec_for_role
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program

ROWS, K = 151936, 4096
Q8_GROUPS = K // 32
Q8_PAYLOAD_WORDS = K // 4
Q8_WORDS = Q8_PAYLOAD_WORDS + Q8_GROUPS

@dataclass(frozen=True)
class Q6KVocabManyRowAdmission:
  """Explicit harness lease; normal model construction never creates this."""
  candidate_id: str = "nv_vocab_manyrow_q8_q6"
  target: str = "NV:sm_120"

  def __post_init__(self):
    if not isinstance(self.candidate_id, str) or not self.candidate_id:
      raise ValueError("candidate_id must be non-empty")

def _unpack_q8_packet(packed: Tensor) -> Tensor:
  """Decode the producer packet for the Q6 consumer without reading source x again."""
  words = packed[:Q8_PAYLOAD_WORDS]
  # Each uint32 owns four consecutive activation values. Keep those byte lanes
  # adjacent when flattening; concatenating one whole shift lane at a time
  # transposes the packet into q[0],q[4],...,q[1],q[5],... .
  shifts = Tensor([0, 8, 16, 24], dtype=dtypes.uint32, device=packed.device)
  q = ((words.reshape(-1, 1) >> shifts).bitwise_and(255).cast(dtypes.uint8).bitcast(dtypes.int8)) \
    .reshape(-1)[:K].cast(dtypes.float32)
  # Q8_1 metadata stores d in the low half of each uint32.  The consumer input
  # is reconstructed from the producer-owned packet, so the source activation
  # is not a second input to the MMVQ program.
  meta = packed[Q8_PAYLOAD_WORDS:Q8_WORDS]
  ds = meta.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
  # Repeat each group's scale across its 32 adjacent q values. Tensor.repeat
  # would tile the whole 128-scale vector and silently rotate scales by group.
  scale = ds.reshape(-1, 1).expand(-1, 32).reshape(-1)
  return (q * scale).cast(dtypes.float16).contiguous()

def q6k_vocab_manyrow_call(admission: object, linear: Any, x: Tensor) -> Tensor | None:
  """Run the complete producer/main lifecycle, returning shape ``(1,1,151936)``."""
  if not isinstance(admission, Q6KVocabManyRowAdmission): return None
  if not str(x.device).startswith("NV") or tuple(x.shape) != (1, 1, K): return None
  storage = getattr(linear, "q6k_storage", None)
  if storage is None or getattr(linear, "out_features", None) != ROWS or getattr(linear, "in_features", None) != K:
    return None
  if getattr(linear, "bias", None) is not None: return None
  source = x.reshape(K).cast(dtypes.float16).contiguous()
  provider = KernelProgram("nv_vocab_manyrow", "q8_provider", KernelProgramProvenance.RESEARCH_ONLY,
    emit_q8_provider(dtypes.float16, k=K), output_spec=OutputSpec((Q8_WORDS,), dtypes.uint32))
  packet = execute_research_program(Tensor.empty((Q8_WORDS,), dtype=dtypes.uint32, device=x.device), source, program=provider)
  q8 = _unpack_q8_packet(packet)
  spec = q6k_spec_for_role(ROWS, K, role="lm_head", parts=1, row_tile=2, use_coop=True,
                           target=admission.target, reduction="in_kernel")
  consumer = KernelProgram("nv_vocab_manyrow", "q6_mmvq", KernelProgramProvenance.RESEARCH_ONLY,
    emit_q6k_gemv_kernel(spec), output_spec=OutputSpec((ROWS,), dtypes.float32))
  out = execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=x.device),
    storage.halfs.to(x.device), q8, program=consumer)
  return out.reshape(1, 1, ROWS)

__all__ = ["ROWS", "K", "Q8_GROUPS", "Q6KVocabManyRowAdmission", "q6k_vocab_manyrow_call"]
