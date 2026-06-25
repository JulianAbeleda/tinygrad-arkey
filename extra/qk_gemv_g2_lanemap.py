#!/usr/bin/env python3
"""Minimal Q4_K gate/up LaneMap for GEMV G2.

This is intentionally narrower than the existing lane-partition custom bridge.
It only describes the generated-code representation target: lane ownership and
packed-word indexing for decode FFN gate/up Q4_K GEMV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tinygrad.uop.ops import UOp, AxisType

QK_K = 256
WARP = 32
BLOCK_GROUPS = 4
WORDS_PER_GROUP = 8
Q4K_WORDS_PER_BLOCK = 36
Q4K_QUANT_WORD_BASE = 4
GROUP_PAIRS = 4


@dataclass(frozen=True)
class Q4KGateUpLaneMap:
  k: int = 4096
  n: int = 12288
  qk_k: int = QK_K
  lane_extent: int = WARP
  block_groups: int = BLOCK_GROUPS
  words_per_group: int = WORDS_PER_GROUP
  q4k_words_per_block: int = Q4K_WORDS_PER_BLOCK
  q4k_quant_word_base: int = Q4K_QUANT_WORD_BASE
  group_pairs: int = GROUP_PAIRS
  groups: int = 8

  @property
  def k_blocks(self) -> int: return self.k // self.qk_k
  @property
  def blocks_per_group(self) -> int: return self.k_blocks // self.block_groups

  def validate(self) -> None:
    if self.k % self.qk_k != 0: raise ValueError(f"k={self.k} must divide qk_k={self.qk_k}")
    if self.k_blocks % self.block_groups != 0: raise ValueError(f"k_blocks={self.k_blocks} must divide block_groups={self.block_groups}")
    if self.lane_extent != self.block_groups * self.words_per_group:
      raise ValueError(f"lane extent mismatch: {self.lane_extent} != {self.block_groups} * {self.words_per_group}")
    if self.words_per_group != 8: raise ValueError("Q4_K gate/up G2 lane map currently requires eight packed words per group")
    if self.q4k_words_per_block != 36: raise ValueError("Q4_K block word layout must be 36 uint32 words")

  def axis_uops(self) -> dict[str, UOp]:
    self.validate()
    return {
      "row": UOp.range(self.n, 0, AxisType.GLOBAL),
      "block_group": UOp.range(self.block_groups, 1, AxisType.LOCAL),
      "word_col": UOp.range(self.words_per_group, 2, AxisType.LOCAL),
      "local_block": UOp.range(self.blocks_per_group, 3, AxisType.REDUCE),
      "group_pair": UOp.range(self.group_pairs, 4, AxisType.REDUCE),
    }

  def lane_expr(self, axes:dict[str, UOp]) -> UOp:
    return axes["block_group"] * self.words_per_group + axes["word_col"]

  def block_expr(self, axes:dict[str, UOp]) -> UOp:
    return axes["block_group"] * self.blocks_per_group + axes["local_block"]

  def packed_word_index_expr(self, axes:dict[str, UOp]) -> UOp:
    blk = self.block_expr(axes)
    return (axes["row"] * self.k_blocks + blk) * self.q4k_words_per_block + self.q4k_quant_word_base + axes["group_pair"] * self.words_per_group + axes["word_col"]

  def packed_word_index_ref(self, row:int, block_group:int, local_block:int, group_pair:int, word_col:int) -> int:
    blk = block_group * self.blocks_per_group + local_block
    return (row * self.k_blocks + blk) * self.q4k_words_per_block + self.q4k_quant_word_base + group_pair * self.words_per_group + word_col

  def serialize(self) -> dict[str, Any]:
    d = asdict(self)
    d["k_blocks"] = self.k_blocks
    d["blocks_per_group"] = self.blocks_per_group
    d["lane_formula"] = "lane = block_group * words_per_group + word_col"
    d["packed_word_index_formula"] = "(row * k_blocks + (block_group * blocks_per_group + local_block)) * q4k_words_per_block + q4k_quant_word_base + group_pair * words_per_group + word_col"
    return d
