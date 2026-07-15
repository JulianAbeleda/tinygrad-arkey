"""Exact host-side ABI for research Q4_K x Q8_1 MMQ operands."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS

@dataclass(frozen=True)
class Q4KQ8MMQABI:
  q4_words_per_block: int = Q4K_WORDS_PER_BLOCK
  q8_values_per_block: int = Q8_1_BLOCK_ELEMS
  q8_scale_dtype: str = "float32"
  q8_sum_dtype: str = "float32"
  def validate(self, words: np.ndarray, values: np.ndarray, scales: np.ndarray, sums: np.ndarray, *, k: int, n: int) -> None:
    if k <= 0 or k % Q4_K_BLOCK_ELEMS: raise ValueError("K must be Q4_K block aligned")
    blocks = n * (k // Q4_K_BLOCK_ELEMS); groups = n * (k // Q8_1_BLOCK_ELEMS)
    if words.dtype != np.uint32 or words.size != blocks * self.q4_words_per_block: raise ValueError("Q4 ABI requires 36 uint32 words per 256-value block")
    if values.dtype != np.int8 or values.size != groups * self.q8_values_per_block: raise ValueError("Q8 ABI requires 32 int8 values per block")
    if scales.dtype != np.float32 or scales.size != groups: raise ValueError("Q8 ABI requires one float32 scale per block")
    if sums.dtype != np.float32 or sums.size != groups: raise ValueError("Q8 ABI requires one float32 sum per block")

Q4K_Q8_MMQ_ABI = Q4KQ8MMQABI()
