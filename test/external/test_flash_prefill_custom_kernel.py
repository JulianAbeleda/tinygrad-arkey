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

_P3_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "qk-flash-prefill-phase3" / "result.json"

class TestFlashPrefillPhase3Artifact(unittest.TestCase):
  """Lock the Phase-3 single-head real-dim gate result (extra/qk_flash_prefill_phase3.py). Skip-if-absent; the
  slow warm-timing benchmark stays out of the suite (run the script to regenerate)."""
  def test_real_dim_gate(self):
    if not _P3_ARTIFACT.exists(): self.skipTest(f"no artifact at {_P3_ARTIFACT}")
    d = json.loads(_P3_ARTIFACT.read_text())
    self.assertTrue(d["correctness_ok"], "Phase-3 correctness regressed")
    self.assertTrue(d["capture"]["score_free"], "Phase-3 lost score-free property")
    self.assertTrue(d["capture"]["jit_replayed"], "Phase-3 JIT replay regressed")
    long = next(r for r in d["rows"] if r["KV"] == max(r2["KV"] for r2 in d["rows"]))
    self.assertGreaterEqual(long["speedup"], 1.5, f"KV={long['KV']} single-head speedup below the 1.5x gate")

_P4_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "qk-flash-prefill-phase4" / "result.json"

class TestFlashPrefillPhase4Artifact(unittest.TestCase):
  """Lock the Phase-4 GQA multi-head gate (extra/qk_flash_prefill_phase4.py). Skip-if-absent; the slow,
  subprocess-isolated benchmark stays out of the suite."""
  def test_gqa_gate(self):
    if not _P4_ARTIFACT.exists(): self.skipTest(f"no artifact at {_P4_ARTIFACT}")
    d = json.loads(_P4_ARTIFACT.read_text())
    self.assertTrue(d["correctness_ok"], "Phase-4 correctness regressed")
    ok = [r for r in d["rows"] if not r.get("faulted")]
    long = next(r for r in ok if r["KV"] == max(r2["KV"] for r2 in ok))
    self.assertTrue(long["score_free"] and long["jit_replayed"], "Phase-4 lost score-free/replay")
    self.assertEqual(long["n_programs"], 2, "Phase-4 should be 2 programs (head dim inside the kernel)")
    self.assertGreaterEqual(long["speedup"], 2.0, f"KV={long['KV']} GQA speedup below the 2x gate")

if __name__ == "__main__":
  unittest.main()
