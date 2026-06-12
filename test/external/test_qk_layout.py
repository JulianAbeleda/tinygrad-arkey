import struct, unittest

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor

from extra.qk_layout import (
  GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS,
  packed_byte_range, q4_k_reference, q6_k_reference, quant_weight_bytes, role_from_name, tensor_shape,
)

def _synthetic_blocks(block_bytes:int, nblocks:int, half_offsets:tuple[int, ...]) -> bytearray:
  data = bytearray((i*37 + 11) & 0xff for i in range(block_bytes*nblocks))
  for b in range(nblocks):
    base = b * block_bytes
    for j, off in enumerate(half_offsets):
      data[base+off:base+off+2] = struct.pack("<e", 0.25 + 0.125*b + 0.0625*j)
  return data

class TestQKLayout(unittest.TestCase):
  def test_q4_k_reference_matches_current_gguf_expression(self):
    data = _synthetic_blocks(Q4_K_BLOCK_BYTES, 2, (0, 2))
    t = Tensor(list(data), dtype=dtypes.uint8)
    ref = q4_k_reference(t, 2*Q4_K_BLOCK_ELEMS).numpy()
    got = ggml_data_to_tensor(t, 2*Q4_K_BLOCK_ELEMS, GGML_Q4_K).numpy()
    np.testing.assert_equal(ref, got)

  def test_q6_k_reference_matches_current_gguf_expression(self):
    data = _synthetic_blocks(Q6_K_BLOCK_BYTES, 2, (208,))
    t = Tensor(list(data), dtype=dtypes.uint8)
    ref = q6_k_reference(t, 2*Q6_K_BLOCK_ELEMS).numpy()
    got = ggml_data_to_tensor(t, 2*Q6_K_BLOCK_ELEMS, GGML_Q6_K).numpy()
    np.testing.assert_equal(ref, got)

  def test_metadata_helpers_are_format_aware(self):
    info = GGUFInfo("blk.0.ffn_down.weight", (12288, 4096), GGML_Q4_K, 64)
    meta = GGUFMetadata(128, [info], {})
    self.assertEqual(tensor_shape(info), (4096, 12288))
    self.assertEqual(role_from_name(info.name), "ffn_down")
    self.assertEqual(packed_byte_range(meta, info), (192, quant_weight_bytes(info)))

if __name__ == "__main__":
  unittest.main()
