from __future__ import annotations

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, UOp

from extra.qk.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, q4_k_reference
from extra.qk.quant.q4_k_gemv_primitive import w_f16


def _make_q4k_words(n:int, k:int, seed:int=1234) -> tuple[Tensor, Tensor]:
  # Random packed Q4_K bytes with finite fp16 d/dmin scales (bytes 0:2 / 2:4 of each block).
  assert k % Q4_K_BLOCK_ELEMS == 0
  rng = np.random.default_rng(seed)
  nblocks = (n * k) // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=nblocks * Q4_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q4_K_BLOCK_BYTES)
  # realistic super-scales so decoded weights are O(0.1) (per-group scale = d*sc, sc up to 63)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  dmin = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  raw[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = dmin.view(np.uint8).reshape(nblocks, 2)
  byte_t = Tensor(raw.reshape(-1).copy()).realize()
  ref = q4_k_reference(byte_t, n * k).reshape(n, k).cast(dtypes.float32).contiguous().realize()
  words = byte_t.bitcast(dtypes.uint32).contiguous().realize()
  return words, ref


def _decode_all(words:Tensor, rows:int, k:int) -> Tensor:
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, w:UOp) -> UOp:
    stores = tuple(out[nn, kk].store(w_f16(w, nn, kk, k_blocks)) for nn in range(rows) for kk in range(k))
    return UOp.sink(*stores, arg=KernelInfo(name=f"q4k_w_f16_decode_{rows}_{k}"))

  return Tensor.empty(rows, k, dtype=dtypes.float16, device=words.device).custom_kernel(words, fxn=kernel)[0]


def test_w_f16_bit_exact_vs_q4k_reference():
  rows, k = 3, 512  # 2 blocks per row: exercises blk 0/1, all 8 groups (low+high branches), all 32 pos
  words, ref = _make_q4k_words(rows, k, seed=20260706)
  got = _decode_all(words, rows, k).cast(dtypes.float32).numpy()
  ref_np = ref.numpy()
  ref_f16 = ref.cast(dtypes.float16).cast(dtypes.float32).numpy()  # f32 reference rounded to fp16
  # True bit-exact gate: w_f16 must equal the reference weight rounded to fp16, elementwise, with no error.
  exact = float(np.max(np.abs(got - ref_f16)))
  # Also report the raw f32 discrepancy, whose only source is the fp16 cast (rmse ~0, well under 3e-4).
  rmse = float(np.sqrt(np.mean((got - ref_np) ** 2)))
  maxabs = float(np.max(np.abs(got - ref_np)))
  print(f"w_f16 decode rows={rows} k={k} exact_vs_f16={exact:.3e} f32_rmse={rmse:.3e} f32_maxabs={maxabs:.3e}")
  assert exact == 0.0, f"not bit-exact vs fp16-rounded reference: maxabs {exact}"
  assert rmse < 3e-4, f"f32 rmse {rmse} too large (should be pure fp16 rounding)"


if __name__ == "__main__":
  test_w_f16_bit_exact_vs_q4k_reference()
  print("PASS")
