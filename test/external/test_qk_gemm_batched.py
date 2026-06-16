import unittest, pathlib, os
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from extra.qk_layout import (read_metadata, tensor_shape, GGML_Q4_K, GGML_Q6_K,
  Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES)
from extra.q4_k_gemv_primitive import q4k_gemm_kernel, parse_opt
from extra.q6_k_gemv_primitive import q6k_gemm_kernel

GGUF = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
DEV = os.environ.get("DEV", "AMD")

# S3: the batched Q4_K/Q6_K decode-GEMM primitives (one weight read for K activation columns, dequant
# hoisted across the columns by UPCAST'ing the batch axis). Correctness vs the fp dequant reference.
@unittest.skipUnless(GGUF.exists(), "needs the Qwen3-8B Q4_K_M gguf")
class TestQKGemmBatched(unittest.TestCase):
  def setUp(self):
    self.meta = read_metadata(GGUF); self.raw = Tensor(GGUF)

  def _check(self, typ, sub, blockbytes, blockelems, gemm_fxn, store_dt, b=8, rows=128):
    info = next(x for x in self.meta.infos if x.typ == typ and sub in x.name)
    shape = tensor_shape(info); rows = min(rows, shape[0]); k = shape[1]
    bstart = self.meta.data_start + info.off; row_bytes = k // blockelems * blockbytes
    packed = Tensor(GGUF, dtype=store_dt)[bstart//store_dt.itemsize:(bstart+rows*row_bytes)//store_dt.itemsize].to(DEV).contiguous().realize()
    Tensor.manual_seed(0); x = Tensor.randn(b, k, dtype=dtypes.float16, device=DEV).realize()
    parts = 4
    p = Tensor.empty(rows, b, parts, dtype=dtypes.float32, device=DEV)
    up = parse_opt(f"UPCAST:1:{min(b,16)}")
    got = p.custom_kernel(packed, x.reshape(b*k), fxn=gemm_fxn(rows, k, b, parts, (parse_opt("LOCAL:0:32"), up)))[0].sum(axis=2)
    dec = ggml_data_to_tensor(self.raw[bstart:bstart+rows*row_bytes].to(DEV), rows*k, info.typ).reshape(rows, k).cast(dtypes.float32)
    ref = dec @ x.cast(dtypes.float32).T  # [rows, b]
    self.assertLess((got.realize() - ref.realize()).abs().max().item(), 1e-1)

  def test_q4k_gemm(self):
    self._check(GGML_Q4_K, "ffn_gate", Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS,
                lambda r,k,b,p,o: q4k_gemm_kernel(r,k,b,p,"none",o), dtypes.uint32)

  def test_q6k_gemm(self):
    self._check(GGML_Q6_K, "ffn_down", Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, q6k_gemm_kernel, dtypes.uint16)

if __name__ == "__main__":
  unittest.main()
