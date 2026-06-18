#!/usr/bin/env python3
"""Locks the cooperative-K Q6_K lm_head GEMV: kernel correctness vs the base path + decode routing/fallback.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python -m pytest test/external/test_q6k_coop.py -q
"""
import os, unittest
import numpy as np

class TestQ6KCoop(unittest.TestCase):
  def _setup(self):
    from tinygrad import Tensor, dtypes
    from extra.llm_generate import load_model_and_tokenizer
    m, _ = load_model_and_tokenizer(os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"), 2048, seed=1)
    return m, Tensor, dtypes

  def _coop_matches(self, lin, Tensor, dtypes, seed):
    from extra.q6_k_gemv_primitive import q6k_gemv_partial_kernel, q6k_coop_partial_kernel, parse_opt
    OUT, IN = lin.out_features, lin.in_features
    halfs = lin.q6k_storage.halfs.realize()
    x = Tensor(np.random.default_rng(seed).standard_normal((IN,)).astype(np.float16)).realize()
    base = Tensor.empty(OUT, 1, dtype=dtypes.float32).custom_kernel(
      halfs, x, fxn=q6k_gemv_partial_kernel(OUT, IN, 1, (parse_opt("LOCAL:0:64"),)))[0].sum(1).numpy()
    for rt in (4, 8):
      coop = Tensor.empty(OUT, 16, dtype=dtypes.float32).custom_kernel(
        halfs, x, fxn=q6k_coop_partial_kernel(OUT, IN, rt))[0].sum(1).numpy()
      rel = np.abs(coop - base).max() / (np.abs(base).max() + 1e-9)
      self.assertLess(rel, 2e-2, f"{OUT}x{IN} coop rt={rt} rel err {rel}")   # fp-reassociation tolerance
      self.assertFalse(np.isnan(coop).any())
      self.assertTrue(OUT % rt == 0)                                          # divisibility invariant (routing guard)

  def test_coop_lm_head_matches_base(self):
    m, Tensor, dtypes = self._setup()
    self._coop_matches(m.output, Tensor, dtypes, 7)

  def test_coop_ffn_down_matches_base(self):
    m, Tensor, dtypes = self._setup()
    ff = [l for l in m._q4k_linears.linears if type(l).__name__ == "Q6KPrimitiveLinear"
          and l.out_features == 4096 and l.in_features == 12288]
    self.assertTrue(ff, "no Q6_K ffn_down found")
    self._coop_matches(ff[0], Tensor, dtypes, 3)

  def test_coop_q4k_attn_qo_matches_base(self):
    m, Tensor, dtypes = self._setup()
    from extra.q4_k_gemv_primitive import q4k_gemv_partial_kernel, q4k_coop_partial_kernel
    qo = [l for l in m._q4k_linears.linears if type(l).__name__ == "Q4KPrimitiveLinear"
          and l.out_features == 4096 and l.in_features == 4096 and l.parts == 1]
    self.assertTrue(qo, "no Q4_K attn_q/o found")
    lin = qo[0]; OUT, IN = lin.out_features, lin.in_features
    words = lin.q4k_storage.words.realize()
    x = Tensor(np.random.default_rng(9).standard_normal((IN,)).astype(np.float16)).realize()
    base = Tensor.empty(OUT, 1, dtype=dtypes.float32).custom_kernel(
      words, x, fxn=q4k_gemv_partial_kernel(OUT, IN, 1, "none", lin.opts))[0].sum(1).numpy()
    for rt in (8, 16):
      coop = Tensor.empty(OUT, 8, dtype=dtypes.float32).custom_kernel(
        words, x, fxn=q4k_coop_partial_kernel(OUT, IN, rt))[0].sum(1).numpy()
      rel = np.abs(coop - base).max() / (np.abs(base).max() + 1e-9)
      self.assertLess(rel, 2e-2, f"q4k attn_qo coop rt={rt} rel err {rel}")
      self.assertFalse(np.isnan(coop).any())
      self.assertTrue(OUT % rt == 0)

  def test_decode_greedy_byte_identical(self):
    # lm_head coop must not change greedy argmax (the shipped default gate)
    m, Tensor, dtypes = self._setup()
    from tinygrad import UOp
    for l in (m._q4k_linears.linears if getattr(m, "_q4k_linears", None) else []): l.decode_enabled = True
    m.output.decode_enabled = True
    x = Tensor.empty(1, 1, m.output.in_features, dtype=dtypes.float16).contiguous().realize()
    def argmax_with(coop):
      os.environ["Q6K_LM_HEAD_COOP"] = coop
      return int(m.output(x).reshape(-1).argmax().item())
    a_off = argmax_with("0"); a_on = argmax_with("1")
    self.assertEqual(a_off, a_on, "coop lm_head changed greedy argmax")

if __name__ == "__main__":
  unittest.main()
