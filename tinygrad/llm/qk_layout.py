"""Canonical storage geometry for the GGML K-quant formats used by tinygrad LLM primitives."""
from __future__ import annotations

from dataclasses import dataclass

GGML_Q4_K = 12
GGML_Q6_K = 14
GGML_Q8_0 = 8

QK_BLOCK_ELEMS = 256
Q4_K_BLOCK_ELEMS = QK_BLOCK_ELEMS
Q4_K_BLOCK_BYTES = 144
Q4K_WORDS_PER_BLOCK = Q4_K_BLOCK_BYTES // 4
Q6_K_BLOCK_ELEMS = QK_BLOCK_ELEMS
Q6_K_BLOCK_BYTES = 210
Q6K_HALFWORDS_PER_BLOCK = Q6_K_BLOCK_BYTES // 2
Q8_1_BLOCK_ELEMS = 32
# Q8_0: one fp16 scale d, then 32 int8 values. 34 bytes per 32 values, so a block is 17 uint16 halfwords.
Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34
Q8_0_HALFWORDS_PER_BLOCK = Q8_0_BLOCK_BYTES // 2


@dataclass(frozen=True)
class QuantFormat:
  """Typed identity of a packed GGML K-quant block format.

  A peer of ``DType``, deliberately not a member of ``dtypes``: block formats are
  not element-addressable and not closed under arithmetic, so no ``DType``
  invariant applies to them. Construction accepts the canonical name string at the
  file/metadata boundary and canonicalizes through ``QUANT_FORMATS``.
  """
  name: str
  block_elems: int
  block_bytes: int
  storage_roles: tuple[str, ...]

  def __post_init__(self):
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("quant format name must be a non-empty string")
    for field, value in (("block_elems", self.block_elems), ("block_bytes", self.block_bytes)):
      if not isinstance(value, int) or value <= 0:
        raise ValueError(f"quant format {field} must be positive, got {value!r}")
    if not isinstance(self.storage_roles, tuple) or not self.storage_roles or not all(
        isinstance(role, str) and role for role in self.storage_roles):
      raise ValueError("quant format storage_roles must be a non-empty tuple of non-empty strings")


Q4_K = QuantFormat("Q4_K", Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, ("words",))
Q6_K = QuantFormat("Q6_K", Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, ("halfs",))
Q8_0 = QuantFormat("Q8_0", Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, ("halfs",))
# Q5_0: d fp16, qh 4 bytes (5th bits), 16 qs bytes (two nibbles). Q4_0: d fp16, 16 qs bytes.
Q5_0 = QuantFormat("Q5_0", 32, 22, ("halfs",))
Q4_0 = QuantFormat("Q4_0", 32, 18, ("halfs",))

# Single authority for name -> format at the string/JSON boundary.
QUANT_FORMATS: dict[str, QuantFormat] = {"Q4_K": Q4_K, "Q6_K": Q6_K, "Q8_0": Q8_0, "Q5_0": Q5_0, "Q4_0": Q4_0}
# GGML's type id for each format (ggml.h enum ggml_type), so a file's tensor table reads as a QuantFormat.
GGML_TYPE_FORMATS: dict[int, QuantFormat] = {2: Q4_0, 6: Q5_0, 8: Q8_0, 12: Q4_K, 14: Q6_K}
GGML_TYPE_IDS: dict[str, int] = {fmt.name: typ for typ, fmt in GGML_TYPE_FORMATS.items()}


def packed_nbytes(fmt:QuantFormat, elements:int) -> int:
  """Bytes of `elements` values in `fmt`; refuses a count that is not whole blocks."""
  if elements % fmt.block_elems: raise ValueError(f"{elements} {fmt.name} values are not whole {fmt.block_elems}-element blocks")
  return elements // fmt.block_elems * fmt.block_bytes
