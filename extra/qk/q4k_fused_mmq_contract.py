"""Backend-neutral descriptor for the bounded fused packed-Q4 MMQ tile."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from extra.qk.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS

FUSED_Q4K_MMQ_CONTRACT = "q4k-fused-mmq-tile-v2"

QWEN3_14B_FUSED_ROLE_SHAPES = (
  (512, 1024, 5120),
  (512, 5120, 5120),
  (512, 5120, 17408),
  (512, 17408, 5120),
)


@dataclass(frozen=True)
class FusedQ4KMMQTileSpec:
  m: int = 16
  n: int = 16
  k: int = 256
  m_tile: int = 16
  n_tile: int = 16
  group_tile: int = 8
  wmma_shape: tuple[int, int, int] = (16, 16, 16)
  lifecycle: Literal["decode_lds_wmma_correct_writeback"] = "decode_lds_wmma_correct_writeback"
  extension: str | None = None

  def validate(self) -> None:
    if (self.m, self.n, self.k) != (16, 16, 256) and (self.m, self.n, self.k) not in QWEN3_14B_FUSED_ROLE_SHAPES:
      raise NotImplementedError("fused MMQ admits only the bounded tile or exact Qwen3-14B role shapes")
    if self.wmma_shape != (16, 16, 16):
      raise ValueError("the canonical fused tile requires WMMA 16x16x16")
    if self.extension is not None:
      raise NotImplementedError(f"fused MMQ extension {self.extension!r} has no lowering")
    if (self.m_tile, self.n_tile) != (16, 16) or self.group_tile != 8:
      raise NotImplementedError("fused MMQ requires bounded 16x16x8 logical M/N/K tiles")
    if self.k % Q8_1_BLOCK_ELEMS or self.k // Q8_1_BLOCK_ELEMS < self.group_tile:
      raise ValueError("K must contain complete Q8_1 groups and one bounded K tile")

  @property
  def words_shape(self) -> tuple[int]:
    return (self.n * (self.k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,)

  @property
  def xq_shape(self) -> tuple[int, int]: return (self.m, self.k)

  @property
  def xscales_shape(self) -> tuple[int, int]: return (self.m, self.k // Q8_1_BLOCK_ELEMS)

  @property
  def live_raw_elems(self) -> int: return self.m_tile * self.n_tile * self.group_tile

  def compiler_geometry(self) -> KernelTileGeometry:
    self.validate()
    return KernelTileGeometry((self.m, self.n, self.wmma_shape[2]), (1, 1), 32, 32,
      (KernelLDSWindow("A", 0, 16 * 16 * 2, 64),
       KernelLDSWindow("B", 16 * 16 * 2, 2 * 16 * 16 * 2, 64)))


__all__ = ["FUSED_Q4K_MMQ_CONTRACT", "QWEN3_14B_FUSED_ROLE_SHAPES", "FusedQ4KMMQTileSpec"]
