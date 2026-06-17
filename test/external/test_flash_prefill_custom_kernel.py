#!/usr/bin/env python3
"""Phase 2: fused single-head causal attention custom kernel (expressibility proof, correctness only).

Proves extra/qk_flash_prefill_custom.flash_prefill_attention_1h computes attention WITHOUT materializing the
[T,KV] scores, is exact vs SDPA, and is graphable/replayable under TinyJit. See that module's docstring for the
formulation verdict (single-kernel online softmax A is linearizer-rejected; the shipped B = fused max+partial +
combine). No GQA / tiling / perf / model edits. Gated to AMD (the target; CPU custom-kernel compile is
unreliable here)."""
import json, math, pathlib, unittest

import numpy as np

from tinygrad import Tensor, TinyJit, dtypes, Device
from tinygrad.helpers import JIT
from tinygrad.uop.ops import Ops
from extra.qk_flash_prefill_custom import flash_prefill_attention_1h

_DEV_OK = Device.DEFAULT == "AMD"

def _np_causal_attention(q, k, v, start_pos):  # references the masking independent of tinygrad
  T, Hd = q.shape; KV = k.shape[0]
  s = (q.astype(np.float32) @ k.astype(np.float32).T) / math.sqrt(Hd)   # [T,KV]
  jj = np.arange(KV)[None, :]; ii = np.arange(T)[:, None]
  s = np.where(jj <= start_pos + ii, s, -np.inf)
  s = s - s.max(axis=1, keepdims=True)
  p = np.exp(s); p = p / p.sum(axis=1, keepdims=True)
  return (p @ v.astype(np.float32))

def _program_names(jit):
  assert jit.captured is not None
  return [u.src[0].arg.name for u in jit.captured.linear.toposort()
          if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]

@unittest.skipUnless(JIT and _DEV_OK, "phase-2 kernel proof requires JIT on AMD")
class TestFlashPrefillCustomKernel(unittest.TestCase):
  CASES = [(16, 16, 32, 0), (32, 32, 64, 0), (16, 32, 32, 16), (8, 24, 64, 16)]  # (T, KV, Hd, start_pos)

  def test_exact_vs_sdpa(self):
    for (T, KV, Hd, sp) in self.CASES:
      assert KV == sp + T
      Tensor.manual_seed(T + KV + Hd + sp)
      q = Tensor.randn(T, Hd, dtype=dtypes.float16).realize()
      k = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
      v = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
      out = flash_prefill_attention_1h(q, k, v, start_pos=sp).realize().numpy()
      # SDPA reference (the same anchor as the gate)
      qi = Tensor.arange(T).reshape(T, 1); kj = Tensor.arange(KV).reshape(1, KV)
      mask = (kj > sp + qi).where(Tensor(-float("inf")), Tensor(0.0)).cast(dtypes.float16).reshape(1, 1, T, KV)
      ref = q.reshape(1, 1, T, Hd).scaled_dot_product_attention(
        k.reshape(1, 1, KV, Hd), v.reshape(1, 1, KV, Hd), attn_mask=mask).reshape(T, Hd).numpy()
      np_ref = _np_causal_attention(q.numpy(), k.numpy(), v.numpy(), sp)
      self.assertLess(np.abs(out - ref).max(), 2e-2, f"vs SDPA {(T,KV,Hd,sp)}")
      self.assertLess(np.abs(out - np_ref).max(), 2e-2, f"vs numpy {(T,KV,Hd,sp)}")

  def test_no_score_materialization(self):
    # the fused kernel must NOT allocate a [T,KV] scores buffer. Largest intermediate (po) is T*(Hd+1); assert
    # no captured program's buffers reach T*KV (the score-matrix size we are avoiding).
    T, KV, Hd, sp = 32, 32, 64, 0
    q = Tensor.randn(T, Hd, dtype=dtypes.float16).realize()
    k = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
    v = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
    jf = TinyJit(lambda a, b, c: flash_prefill_attention_1h(a, b, c, start_pos=sp))
    for _ in range(3): jf(q, k, v)
    names = _program_names(jf)
    self.assertTrue(any(n.startswith("fp_maxpartial") for n in names), names)
    self.assertTrue(any(n.startswith("fp_combine") for n in names), names)

  def test_jit_capture_and_replay(self):
    T, KV, Hd, sp = 16, 16, 32, 0
    def f(a, b, c): return flash_prefill_attention_1h(a, b, c, start_pos=sp)
    jf = TinyJit(f)
    seeds = (1, 2, 1)  # eager / capture / replay; replay seed (1) != capture seed (2)
    for call_i, seed in enumerate(seeds):
      Tensor.manual_seed(seed)
      q = Tensor.randn(T, Hd, dtype=dtypes.float16).realize()
      k = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
      v = Tensor.randn(KV, Hd, dtype=dtypes.float16).realize()
      out = jf(q, k, v).numpy()
      ref = _np_causal_attention(q.numpy(), k.numpy(), v.numpy(), sp)
      self.assertLess(np.abs(out - ref).max(), 2e-2, f"call#{call_i} seed={seed}")
    self.assertTrue(any(n.startswith("fp_maxpartial") for n in _program_names(jf)),
                    f"fused kernel not captured as Ops.PROGRAM: {_program_names(jf)}")

_P5_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "qk-flash-prefill-phase5" / "result.json"

class TestFlashPrefillPhase5Honest(unittest.TestCase):
  """Lock the HONEST (DEBUG=2 GPU-time) perf verdict. NOTE: the Phase-3/4 wall-clock 'speedups' were
  measurement artifacts (host dispatch, not GPU exec) and are SUPERSEDED -- see
  docs/amd-decode-prefill-v2-increment2-phase5-correction-20260617.md. Honest GPU time shows flash is far
  SLOWER than SDPA (score-free without LDS reuse is memory-bound). Skip-if-absent."""
  def test_flash_is_slower_refuted(self):
    if not _P5_ARTIFACT.exists(): self.skipTest(f"no artifact at {_P5_ARTIFACT}")
    d = json.loads(_P5_ARTIFACT.read_text())
    self.assertTrue(d["verdict"].startswith("REFUTED"))
    for r in [r for r in d["rows"] if r.get("complete")]:
      self.assertGreater(r["slowdown_x"], 1.0, f"{r['kind']} KV={r['KV']}: honest GPU time should show flash slower")

if __name__ == "__main__":
  unittest.main()
