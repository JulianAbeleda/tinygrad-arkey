"""Canonical storage geometry for the GGML K-quant formats used by tinygrad LLM primitives."""
from __future__ import annotations

from dataclasses import dataclass

GGML_Q4_K = 12
GGML_Q6_K = 14

QK_BLOCK_ELEMS = 256
Q4_K_BLOCK_ELEMS = QK_BLOCK_ELEMS
Q4_K_BLOCK_BYTES = 144
Q4K_WORDS_PER_BLOCK = Q4_K_BLOCK_BYTES // 4
Q6_K_BLOCK_ELEMS = QK_BLOCK_ELEMS
Q6_K_BLOCK_BYTES = 210
Q6K_HALFWORDS_PER_BLOCK = Q6_K_BLOCK_BYTES // 2
Q8_1_BLOCK_ELEMS = 32


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

# Single authority for name -> format at the string/JSON boundary.
QUANT_FORMATS: dict[str, QuantFormat] = {"Q4_K": Q4_K, "Q6_K": Q6_K}
