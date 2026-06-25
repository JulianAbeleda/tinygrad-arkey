#!/usr/bin/env python3
import struct, unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops, UOp
from extra.q4_k_gemv_primitive import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, q4k_gemv_warp_kernel
from extra.qk_q4k_lane_partition_gemv import q4k_lane_partition_gemv_kernel

_DEV_OK = Device.DEFAULT == "AMD"
ROWS, K = 4, 1024
KBLOCKS = K // Q4_K_BLOCK_ELEMS

def _synthetic_words(rows=ROWS, k=K):
  kb = k // Q4_K_BLOCK_ELEMS
  words = np.zeros((rows, kb, Q4K_WORDS_PER_BLOCK), dtype=np.uint32)
  d = struct.unpack('<I', struct.pack('<e', 0.03125) + struct.pack('<e', 0.0))[0]
  # scales/mins: scale low six bits = 1, min = 0 for groups 0..7. high-packed groups keep low nibble scale = 0,
  # but this deterministic payload is enough for old/new equality because both kernels decode the same words.
  sw = np.array([0x01010101, 0x00000000, 0x00000000], dtype=np.uint32)
  rng = np.random.default_rng(123)
  words[..., 0] = d
  words[..., 1:4] = sw
  words[..., 4:36] = rng.integers(0, 0xffffffff, size=(rows, kb, 32), dtype=np.uint32)
  return words.reshape(rows, kb * Q4K_WORDS_PER_BLOCK)

@unittest.skipUnless(_DEV_OK, "q4k lane-partition GEMV is AMD wave32 gated")
class TestQ4KLanePartitionGEMV(unittest.TestCase):
  def test_matches_owned_warp_kernel(self):
    words_np = _synthetic_words()
    x_np = np.random.default_rng(7).standard_normal(K).astype(np.float16)
    words = Tensor(words_np.reshape(-1), dtype=dtypes.uint32).realize()
    x = Tensor(x_np, dtype=dtypes.float16).realize()
    got = Tensor.empty(ROWS, dtype=dtypes.float32).custom_kernel(words, x, fxn=q4k_lane_partition_gemv_kernel(ROWS, K))[0].realize().numpy()
    ref = Tensor.empty(ROWS, dtype=dtypes.float32).custom_kernel(words, x, fxn=q4k_gemv_warp_kernel(ROWS, K))[0].realize().numpy()
    self.assertTrue(np.allclose(got, ref, rtol=1e-5, atol=1e-4), f"lane partition q4k mismatch, max_err={np.abs(got-ref).max()}")

  def test_source_has_lane_partition_and_bpermute(self):
    out = Tensor.empty(ROWS, dtype=dtypes.float32).custom_kernel(
      Tensor(_synthetic_words().reshape(-1), dtype=dtypes.uint32).realize(), Tensor(np.ones(K, dtype=np.float16)).realize(),
      fxn=q4k_lane_partition_gemv_kernel(ROWS, K))[0]
    src = ""
    for call in compile_linear(out.schedule_linear()).src:
      p = call.src[0]
      if p.op is Ops.PROGRAM and "q4k_lane_partition_gemv" in p.arg.name:
        src = next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), "")
        break
    self.assertIn("lidx0", src)
    self.assertTrue("%8" in src.replace(" ", "") or "&7" in src.replace(" ", ""))
    self.assertIn("ds_bpermute", src)

if __name__ == "__main__":
  unittest.main()
