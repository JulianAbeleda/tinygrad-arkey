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
Q6K_WMMA_ROUTE = "staged_dequant_then_fp16_wmma"
Q6K_ZERO_POINT = 32


@dataclass(frozen=True)
class Q6KPrefillWMMASpec:
  """Shape/role contract for the bounded generated contraction (X[M,K] @ W[N,K].T)."""
  m: int
  n: int
  k: int
  role: Q6KPrefillRole
  # The bounded contract is one real 14B Q6_K role, not a toy tile.  WMMA
  # still tiles the ordinary fp16 contraction; these are the authority shape
  # limits for ffn_down (M=512, N=4096, K=12288).
  max_m: int = 512
  max_n: int = 4096
  max_k: int = 12288
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

  def admission(self, *, fused: bool = False, lowering_proof: bool = True) -> dict[str, Any]:
    """Return the Q6-only dispatch gate; estimates never admit a fused route."""
    self.validate()
    errors = []
    if self.target != "amd_gfx1100": errors.append("no Q6 WMMA capability record for target")
    if fused: errors.append("Q6 packed operands have no legal gfx1100 WMMA lowering")
    if not lowering_proof: errors.append("staged fp16 WMMA lowering proof missing")
    return {"admitted": not errors, "route": Q6K_WMMA_ROUTE, "errors": errors,
            "quant_correction": "d * scale_i8 * (code_u6 - 32)", "zero_point": Q6K_ZERO_POINT,
            "stage_boundary": "Q6_K bytes -> fp16 weights -> WMMA"}

  def to_json(self) -> dict[str, Any]:
    return {"m": self.m, "n": self.n, "k": self.k, "role": self.role, "target": self.target,
            "implementation": self.implementation, "route": Q6K_WMMA_ROUTE,
            "quant_correction": "d * scale_i8 * (code_u6 - 32)", "zero_point": Q6K_ZERO_POINT,
            "stage_boundary": "Q6_K bytes -> fp16 weights -> WMMA",
            "packed_bytes": self.packed_bytes, "kernel_name": self.kernel_name,
            "bounds": {"m": self.max_m, "n": self.max_n, "k": self.max_k}}


def describe_q6k_wmma_prefill(m:int, n:int, k:int, *, role:Q6KPrefillRole="test") -> Q6KPrefillWMMASpec:
  spec = Q6KPrefillWMMASpec(m=m, n=n, k=k, role=role)
  spec.validate()
  return spec


def emit_q6k_wmma_prefill(packed:Tensor, x:Tensor, spec:Q6KPrefillWMMASpec) -> Tensor:
  """Return fp32 [M,N]; packed is the canonical byte stream and x is fp16-compatible [M,K]."""
  spec.validate()
  admission = spec.admission()
  if not admission["admitted"]: raise ValueError("Q6 WMMA admission failed: " + "; ".join(admission["errors"]))
  if packed.dtype != dtypes.uint8 or packed.numel() != spec.packed_bytes:
    raise ValueError(f"packed must be uint8[{spec.packed_bytes}], got dtype={packed.dtype} shape={packed.shape}")
  if x.shape != (spec.m, spec.k): raise ValueError(f"x must have shape {(spec.m, spec.k)}, got {x.shape}")
  # This contiguous node is the explicit dequant-once boundary.  The contraction is ordinary tinygrad Tensor math.
  weight_f16 = q6_k_reference(packed, spec.n * spec.k).reshape(spec.n, spec.k).cast(dtypes.float16).contiguous()
  out = x.cast(dtypes.float16).matmul(weight_f16.transpose(), dtype=dtypes.float32)
  # Keep the same matmul axis convention as the production fp16 prefill route: automatic TC selection, N, M, K.
  return out.contiguous(arg=ScheduleHints(opts_to_apply=(Opt(OptOps.TC, 0, (-1, 2, 1)),), name=spec.kernel_name))
