import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop import Ops

def _unfused_sdpa(q:Tensor, k:Tensor, v:Tensor, mask:Tensor|None=None, is_causal:bool=False) -> Tensor:
  if k.shape[-3] != q.shape[-3]:
    k, v = k.repeat_interleave(q.shape[-3]//k.shape[-3], dim=-3), v.repeat_interleave(q.shape[-3]//v.shape[-3], dim=-3)
  qk = (q @ k.transpose(-2, -1)) * (1.0 / q.shape[-1] ** 0.5)
  if is_causal: mask = Tensor.ones(qk.shape[-2], qk.shape[-1], dtype=dtypes.bool, device=q.device).tril()
  if mask is not None: qk = qk + (mask.where(0, -float("inf")) if mask.dtype == dtypes.bool else mask)
  return qk.softmax(-1) @ v

class TestAttentionGradient(unittest.TestCase):
  def _check(self, qs=(1,2,5,8), ks=(1,2,6,8), mask=None, is_causal=False, gqa=False):
    rng = np.random.default_rng(0)
    qn, kn, vn = (rng.standard_normal(s).astype(np.float32) for s in (qs, ks, ks))
    gn = rng.standard_normal(qs[:-1]+(ks[-1],)).astype(np.float32)
    def run(fused:bool):
      q, k, v = (Tensor(x, device="CPU").realize() for x in (qn, kn, vn))
      out = q.scaled_dot_product_attention(k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=gqa) if fused else \
            _unfused_sdpa(q, k, v, mask, is_causal)
      if fused: self.assertIn(Ops.ATTENTION, {u.op for u in out.uop.toposort()})
      (out * Tensor(gn, device="CPU")).sum().backward()
      return out.numpy(), q.grad.numpy(), k.grad.numpy(), v.grad.numpy()
    for a, b in zip(run(True), run(False)): np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

  def test_plain(self): self._check()
  def test_causal(self): self._check(qs=(1,2,6,8), ks=(1,2,6,8), is_causal=True)
  def test_bool_mask(self):
    self._check(mask=Tensor(np.random.default_rng(1).random((5,6)) > 0.3, device="CPU").realize())
  def test_gqa(self): self._check(qs=(1,4,5,8), ks=(1,2,6,8), gqa=True)

if __name__ == "__main__":
  unittest.main()
