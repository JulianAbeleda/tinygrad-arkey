"""Typed ABI for a compiler-owned causal Flash prefill candidate.

This module is deliberately declarative.  It does not load CUDA source/cubins,
compile kernels, import the model, or inspect a device.  F2's emitter should
consume :class:`NativeFlashPP512Spec` and implement ``emit`` below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal, Protocol

Scalar = Literal["float16", "float32"]


@dataclass(frozen=True)
class NativeFlashPP512Spec:
  batch: int = 1
  query_heads: int = 32
  kv_heads: int = 8
  tokens: int = 512
  head_dim: int = 128
  start_pos: int = 0
  q_dtype: Scalar = "float16"
  k_dtype: Scalar = "float16"
  v_dtype: Scalar = "float16"
  accumulation_dtype: Scalar = "float32"
  causal: bool = True
  output_dtype: Scalar = "float16"
  output_layout: str = "B,Hq,T,Hd"

  @property
  def live_tokens(self) -> int:
    return self.start_pos + self.tokens

  @property
  def query_group_size(self) -> int:
    return self.query_heads // self.kv_heads

  def validate(self) -> "NativeFlashPP512Spec":
    positive = ("batch", "query_heads", "kv_heads", "tokens", "head_dim")
    if any(getattr(self, key) <= 0 for key in positive):
      raise ValueError("all dimensions must be positive")
    if self.batch != 1: raise ValueError("only batch=1 is admitted")
    if self.query_heads % self.kv_heads: raise ValueError("query_heads must divide evenly into kv_heads")
    if self.start_pos < 0: raise ValueError("start_pos must be non-negative")
    if not self.causal: raise ValueError("causal attention is required")
    if (self.q_dtype, self.k_dtype, self.v_dtype) != ("float16", "float16", "float16"):
      raise ValueError("Q/K/V must all be float16")
    if self.accumulation_dtype != "float32": raise ValueError("online softmax must accumulate in float32")
    if self.output_dtype != "float16": raise ValueError("output must be float16 at this ABI boundary")
    if self.output_layout != "B,Hq,T,Hd": raise ValueError("unsupported output layout")
    if self.tokens not in (128, 512, 1024): raise ValueError("tokens must be one of 128, 512, 1024")
    if self.start_pos not in (0, 256): raise ValueError("start_pos must be one of 0, 256")
    return self

  def gqa_kv_head(self, query_head: int) -> int:
    self.validate()
    if not 0 <= query_head < self.query_heads: raise ValueError("query head out of range")
    return query_head // self.query_group_size

  def to_dict(self) -> dict[str, object]:
    self.validate()
    return {**asdict(self), "live_tokens": self.live_tokens, "query_group_size": self.query_group_size,
            "abi": "tinygrad.native_flash_pp512.v1"}

  def serialize(self) -> str:
    return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FlashEmitterRequest:
  """Exact interface F2 must implement for one validated ABI instance."""
  spec: NativeFlashPP512Spec
  q_shape: tuple[int, int, int, int]
  k_shape: tuple[int, int, int, int]
  v_shape: tuple[int, int, int, int]

  def validate(self) -> "FlashEmitterRequest":
    self.spec.validate()
    expected_q = (self.spec.batch, self.spec.query_heads, self.spec.tokens, self.spec.head_dim)
    expected_kv = (self.spec.batch, self.spec.kv_heads, self.spec.tokens, self.spec.head_dim)
    if self.q_shape != expected_q or self.k_shape != expected_kv or self.v_shape != expected_kv:
      raise ValueError("Q/K/V shapes do not match the Flash ABI")
    return self


class NativeFlashEmitter(Protocol):
  def emit(self, request: FlashEmitterRequest) -> object:
    """Return a tinygrad-owned KernelProgram/UOp emitter for the request."""


ADMITTED_TEST_MATRIX = tuple(
  NativeFlashPP512Spec(tokens=tokens, start_pos=start_pos)
  for tokens in (128, 512, 1024) for start_pos in (0, 256)
)
