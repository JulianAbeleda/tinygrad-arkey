import unittest, pathlib, numpy as np
from tinygrad import Tensor
from extra.qk_layout import read_metadata, q4_k_reference, tensor_shape, GGML_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS
from extra.qk_quantize import quantize_q4_k

GGUF = pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")

def _deq_np(words, rows, k):  # independent numpy dequant mirroring q4_k_reference's byte logic
  nb = k//256; B = words.reshape(rows*nb, 36)
  d = (B[:,0]&0xffff).astype(np.uint16).view(np.float16).astype(np.float32)
  dmin = ((B[:,0]>>16)&0xffff).astype(np.uint16).view(np.float16).astype(np.float32)
  by = words.reshape(rows*nb,36).view(np.uint8).reshape(-1,144).astype(np.int32); s = by[:,4:16]
  sc = np.concatenate([s[:,0:4]&63, (s[:,8:12]&0xF)|((s[:,0:4]>>6)<<4)],1)
  mn = np.concatenate([s[:,4:8]&63, (s[:,8:12]>>4)|((s[:,4:8]>>6)<<4)],1)
  qs = by[:,16:144].reshape(-1,4,32); q = np.stack([qs&0xF, qs>>4],2).reshape(-1,8,32)
  return (d[:,None,None]*sc[:,:,None]*q - dmin[:,None,None]*mn[:,:,None]).reshape(rows,k)

@unittest.skipUnless(GGUF.exists(), "needs the Qwen3-8B Q4_K_M gguf")
class TestQ4KQuantizer(unittest.TestCase):
  def test_roundtrip_exact_on_llama_q4(self):
    # re-quantizing a weight llama already stored as Q4_K must land back on the same grid (bit-exact)
    meta = read_metadata(GGUF); raw = Tensor(GGUF)
    info = next(x for x in meta.infos if x.typ == GGML_Q4_K and "ffn_gate" in x.name)
    rows, k = 64, tensor_shape(info)[1]
    bs = meta.data_start + info.off; rb = k//Q4_K_BLOCK_ELEMS*Q4_K_BLOCK_BYTES
    fp = q4_k_reference(raw[bs:bs+rows*rb].to("AMD"), rows*k).reshape(rows, k).numpy().astype(np.float32)
    recon = _deq_np(quantize_q4_k(fp), rows, k)
    self.assertEqual(np.abs(recon - fp).max(), 0.0)  # exact: our make_qkx2 reproduces llama's quantization

if __name__ == "__main__":
  unittest.main()
