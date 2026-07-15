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
    if not isinstance(k, int) or not isinstance(n, int) or k <= 0 or k % Q4_K_BLOCK_ELEMS: raise ValueError("K must be a positive Q4_K-block-aligned integer")
    if n <= 0: raise ValueError("N must be a positive integer")
    blocks = n * (k // Q4_K_BLOCK_ELEMS); groups = n * (k // Q8_1_BLOCK_ELEMS)
    expected = ((words, blocks * self.q4_words_per_block, "Q4 ABI requires"),
                (values, groups * self.q8_values_per_block, "Q8 ABI requires"),
                (scales, groups, "Q8 ABI requires"), (sums, groups, "Q8 ABI requires"))
    dtypes = (np.uint32, np.int8, np.float32, np.float32)
    descriptions = (f"{self.q4_words_per_block} uint32 words per 256-value block",
                    f"{self.q8_values_per_block} int8 values per block",
                    "one float32 scale per block", "one float32 sum per block")
    for (array, size, prefix), dtype, description in zip(expected, dtypes, descriptions):
      if array.ndim != 1 or not array.flags.c_contiguous or array.dtype != dtype or array.size != size:
        raise ValueError(f"{prefix} {description}")

Q4K_Q8_MMQ_ABI = Q4KQ8MMQABI()
