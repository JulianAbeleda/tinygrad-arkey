"""Bounded generated Q6_K prefill matrix-core experiment.

This is intentionally not a production route.  It uses tinygrad's canonical Q6_K
decoder once, materializes fp16 weights, and hands an ordinary fp16 contraction to
the scheduler/tensor-core optimizer.  The direct-packed implementation remains the
rollback path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import ScheduleHints

from extra.qk.layout import GGML_Q6_K, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, q6_k_reference


Q6KPrefillRole = Literal["attn_qo", "attn_kv", "ffn_gate_up", "ffn_down", "lm_head", "test"]
_ROLES = ("attn_qo", "attn_kv", "ffn_gate_up", "ffn_down", "lm_head", "test")


@dataclass(frozen=True)
class Q6KPrefillWMMASpec:
  """Shape/role contract for the bounded generated contraction (X[M,K] @ W[N,K].T)."""
  m: int
  n: int
  k: int
  role: Q6KPrefillRole
  max_m: int = 32
  max_n: int = 32
  max_k: int = 512
  target: str = "amd_gfx1100"
  implementation: str = "dequant_once_fp16_matmul_v0"

  @property
  def packed_bytes(self) -> int: return self.n * self.k // Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES

  @property
  def kernel_name(self) -> str: return f"q6k_wmma_prefill_generated_{self.role}_{self.m}_{self.n}_{self.k}"

  def validate(self) -> None:
    if self.role not in _ROLES: raise ValueError(f"unsupported role={self.role!r}; allowed roles are {_ROLES}")
    if min(self.m, self.n, self.k) <= 0: raise ValueError(f"m/n/k must be positive, got {(self.m, self.n, self.k)}")
    if self.k % Q6_K_BLOCK_ELEMS: raise ValueError(f"k={self.k} must be a multiple of {Q6_K_BLOCK_ELEMS}")
    if self.m % 16 or self.n % 16: raise ValueError(f"m and n must align to fp16 WMMA tiles (16,16), got {(self.m, self.n)}")
    if self.m > self.max_m or self.n > self.max_n or self.k > self.max_k:
      raise ValueError(f"shape {(self.m, self.n, self.k)} exceeds bounded maximum {(self.max_m, self.max_n, self.max_k)}")
    if self.implementation != "dequant_once_fp16_matmul_v0": raise ValueError(f"unsupported implementation={self.implementation!r}")

  def to_json(self) -> dict[str, Any]:
    return {"m": self.m, "n": self.n, "k": self.k, "role": self.role, "target": self.target,
            "implementation": self.implementation, "packed_bytes": self.packed_bytes, "kernel_name": self.kernel_name,
            "bounds": {"m": self.max_m, "n": self.max_n, "k": self.max_k}}


def describe_q6k_wmma_prefill(m:int, n:int, k:int, *, role:Q6KPrefillRole="test") -> Q6KPrefillWMMASpec:
  spec = Q6KPrefillWMMASpec(m=m, n=n, k=k, role=role)
  spec.validate()
  return spec


def emit_q6k_wmma_prefill(packed:Tensor, x:Tensor, spec:Q6KPrefillWMMASpec) -> Tensor:
  """Return fp32 [M,N]; packed is the canonical byte stream and x is fp16-compatible [M,K]."""
  spec.validate()
  if packed.dtype != dtypes.uint8 or packed.numel() != spec.packed_bytes:
    raise ValueError(f"packed must be uint8[{spec.packed_bytes}], got dtype={packed.dtype} shape={packed.shape}")
  if x.shape != (spec.m, spec.k): raise ValueError(f"x must have shape {(spec.m, spec.k)}, got {x.shape}")
  # This contiguous node is the explicit dequant-once boundary.  The contraction is ordinary tinygrad Tensor math.
  weight_f16 = q6_k_reference(packed, spec.n * spec.k).reshape(spec.n, spec.k).cast(dtypes.float16).contiguous()
  out = x.cast(dtypes.float16).matmul(weight_f16.transpose(), dtype=dtypes.float32)
  # Keep the same matmul axis convention as the production fp16 prefill route: automatic TC selection, N, M, K.
  return out.contiguous(arg=ScheduleHints(opts_to_apply=(Opt(OptOps.TC, 0, (-1, 2, 1)),), name=spec.kernel_name))
