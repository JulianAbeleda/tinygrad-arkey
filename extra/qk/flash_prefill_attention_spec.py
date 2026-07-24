#!/usr/bin/env python3
"""Descriptor scaffolding for the machine-searched fixed-16-WMMA prefill route.

Mirrors extra/qk/flash_decode_attention_spec.py (FlashDecodeTileSpec): the topology
that today lives as inline `emit(...)` glue in tinygrad/llm/fused_attention.py's
custom_kernel_attention is here owned as DATA by a frozen dataclass, so a route can
compose it (machine_authored_generated) instead of importing the hand builder by
name (hand_authored_uop_template). See docs/flash-prefill-pure-search-lift-scope-20260724.md.

P-B1..P-B3 de-literalized the substrate validators, the emitter's index math, and the HIP backend's
drain address math to all derive from head_dim. P4a (this file) closes the last gap: Hd is now THREADED
into the builder call (not just a decorative field the emitter ignored) -- `self.Hd` flows into
`amd_gfx1100_q16_grid_hd128_loop_attention`'s `head_dim=` kwarg, which threads into the
AMDAttentionGridSpec and AMDAttentionOutputDrainSpec it constructs, so the descriptor genuinely owns
head_dim end to end. Hd remains validated up to a ceiling (see validate()): the wave32 VGPR budget
hard-ceilings at Hd=128, so Hd must be a positive 16-multiple <=128 -- form-generic + spec-threaded up
to that ceiling. Only Hd=128 is numerically PROVEN so far (the route admits only ADMITTED_GRIDS-listed
shapes); this validate() bound describes what the emitter can legally construct, not what is proven fast/
correct on real hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad.uop.ops import UOp


@dataclass(frozen=True)
class FlashPrefillAttentionSpec:
  """Data-owned topology for amd_gfx1100_q16_grid_hd128_loop_attention.

  Field names match the builder's kwargs 1:1 (Hq->q_heads, Hkv->kv_heads, Hd->head_dim
  -- Hd IS passed to the builder now (P4a); it is no longer a decorative field the
  emitter ignores).
  """
  Hq: int
  Hkv: int
  q_tokens: int
  kv_tokens: int
  causal: bool
  scale: float
  Hd: int = 128
  valid_kv: int | None = None
  query_start: int | None = None
  # None means "the full accumulator for whatever Hd is" (Hd//16), resolved in __post_init__ --
  # byte-identical to the old literal default 8 at Hd=128 (128//16==8), but stays correct as Hd
  # varies (an explicit value is always respected as given, never reinterpreted).
  acc_blocks: int | None = None
  output_block_base: int = 0
  phase_abi_v1: bool = False
  target: str = "amd_gfx1100"

  def __post_init__(self):
    if self.acc_blocks is None:
      object.__setattr__(self, "acc_blocks", self.Hd // 16)

  def validate(self) -> "FlashPrefillAttentionSpec":
    # Hd is form-generic + spec-threaded (P4a) up to the Hd<=128 wave32 VGPR-budget ceiling: Hd must
    # be a positive 16-multiple <=128. Only Hd=128 is numerically validated/proven so far -- the route
    # admits only ADMITTED_GRIDS-listed shapes (fused_attention.py), a separate, narrower gate.
    if self.Hd <= 0 or self.Hd % 16 or self.Hd > 128:
      raise ValueError(f"FlashPrefillAttentionSpec requires a positive 16-multiple head_dim <=128, got {self.Hd}")
    if self.q_tokens <= 0 or self.q_tokens % 16:
      raise ValueError(f"q_tokens must be a positive multiple of 16, got {self.q_tokens}")
    if self.kv_tokens <= 0 or self.kv_tokens % 16 or self.kv_tokens > 4096:
      raise ValueError(f"kv_tokens must be a positive multiple of 16 and <=4096, got {self.kv_tokens}")
    if self.Hkv <= 0 or self.Hq <= 0 or self.Hq % self.Hkv:
      raise ValueError(f"Hq must be a positive multiple of Hkv, got Hq={self.Hq} Hkv={self.Hkv}")
    if self.acc_blocks not in {1, 2, 4, 8}:
      raise ValueError(f"acc_blocks must be one of {{1,2,4,8}}, got {self.acc_blocks}")
    if (self.output_block_base, self.acc_blocks) != (0, 8):
      if self.output_block_base % self.acc_blocks:
        raise ValueError("output_block_base must be aligned to acc_blocks")
      if not 0 <= self.output_block_base <= 8 - self.acc_blocks:
        raise ValueError("output_block_base is outside the accumulator-slice range")
    return self

  def emit(self, kernel_info=None):
    """Return a custom_kernel-shaped fxn: (out_ph, q_ph, k_ph, v_ph) -> UOp.

    Reproduces amd_gfx1100_q16_grid_hd128_loop_attention's call site EXACTLY as it
    appears inline in tinygrad/llm/fused_attention.py:custom_kernel_attention's
    `emit` closure -- same kwargs, same argument-for-argument threading (the
    closure's `valid_kv=ctx.kv_tokens, query_start=ctx.start_pos` become this
    spec's own `self.valid_kv`/`self.query_start` fields).

    `kernel_info`: optional override for the builder's `kernel_info=` kwarg, so a
    caller that needs to carry forward its own KernelInfo (e.g. postrange.py's
    AST-swap, which must preserve self.ast.arg's existing fields via replace())
    can still route through this SAME emitter seam instead of calling the raw
    builder directly. Default (None) is UNCHANGED from before this override
    existed -- `KernelInfo(name="amd_gfx1100_q16_grid_hd128_loop_attention")`.
    """
    self.validate()
    from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
    from tinygrad.uop.ops import KernelInfo
    ki = kernel_info if kernel_info is not None else KernelInfo(name="amd_gfx1100_q16_grid_hd128_loop_attention")

    def fxn(out_ph: UOp, q_ph: UOp, k_ph: UOp, v_ph: UOp) -> UOp:
      return amd_gfx1100_q16_grid_hd128_loop_attention(
        q_ph, k_ph, v_ph, out_ph, q_tokens=self.q_tokens, q_heads=self.Hq,
        kv_heads=self.Hkv, kv_tokens=self.kv_tokens, scale=self.scale, causal=self.causal,
        valid_kv=self.valid_kv, query_start=self.query_start,
        output_block_base=self.output_block_base, acc_blocks=self.acc_blocks,
        phase_abi_v1=self.phase_abi_v1, head_dim=self.Hd, kernel_info=ki)
    return fxn

  @property
  def emitted_kernel_names(self) -> tuple[str, ...]:
    return ("amd_gfx1100_q16_grid_hd128_loop_attention",)

  def to_json(self) -> dict[str, Any]:
    return {"Hq": self.Hq, "Hkv": self.Hkv, "Hd": self.Hd, "q_tokens": self.q_tokens,
            "kv_tokens": self.kv_tokens, "causal": self.causal, "valid_kv": self.valid_kv,
            "query_start": self.query_start, "acc_blocks": self.acc_blocks,
            "output_block_base": self.output_block_base, "phase_abi_v1": self.phase_abi_v1,
            "scale": self.scale, "target": self.target}


def describe_flash_prefill_attention(Hq: int, Hkv: int, q_tokens: int, kv_tokens: int, *,
                                     causal: bool, scale: float, valid_kv: int | None = None,
                                     query_start: int | None = None, acc_blocks: int = 8,
                                     output_block_base: int = 0, phase_abi_v1: bool = False) -> FlashPrefillAttentionSpec:
  return FlashPrefillAttentionSpec(Hq=Hq, Hkv=Hkv, q_tokens=q_tokens, kv_tokens=kv_tokens, causal=causal,
    scale=scale, valid_kv=valid_kv, query_start=query_start, acc_blocks=acc_blocks,
    output_block_base=output_block_base, phase_abi_v1=phase_abi_v1)


def emit_flash_prefill_attention(spec: FlashPrefillAttentionSpec):
  return spec.emit()
