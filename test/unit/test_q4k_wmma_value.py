"""VALUE-semantics gate for the int8/Q4_K WMMA prefill substrate (signed-dot regression protection).

The Q4_K/Q8_1 prefill dot is expressed as `Tensor.matmul(..., dtype=int)` so RDNA3 selects the iu8 WMMA; its
numeric correctness was checked only by standalone scripts (extra/qk/prefill_mmq_parity_gate.py,
extra/qk/q4k_wmma_tiled_microgate.py) that live OUTSIDE pytest. Those scripts already do the right comparison --
`ref = (x_dq @ ref_w.T)` vs the emitted `.numpy()` -- but nothing in the suite runs them, so a signed-dot
regression (the unsigned-dot4 failure mode) would not be caught by the test runner. This file wires that exact
reference comparison into pytest.

The iu8 WMMA is only selected/executed on AMD, and the CPU oracle needs a C compiler this environment lacks, so
these tests are `@skipUnless(Device.DEFAULT == "AMD")`. The skip is machine-enforced and honest: on AMD they RUN
and assert real numeric parity; elsewhere they skip rather than silently pass.
"""
import unittest
import numpy as np

from tinygrad import Tensor, Device, dtypes

from extra.qk.layout import q8_1_quantize
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
from extra.qk.prefill_int8_wmma_spec import (
  describe_q4k_int8_wmma_prefill, emit_q4k_int8_wmma_prefill_tensor,
  describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_prefill_tensor)

IS_AMD = Device.DEFAULT == "AMD"


def _q4k_reference_and_inputs(n: int, k: int, m: int, seed: int):
  """Build synthetic finite Q4_K weights + Q8_1 activation and the fp32 dequant reference (x_dq @ ref_w.T)."""
  words, ref_w = _make_q4k_words(n, k, seed)
  x = Tensor(np.random.default_rng(seed + 1).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) *
          xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  ref_out = (x_dq @ ref_w.T).numpy()
  return words, xq, xscales, ref_out


@unittest.skipUnless(IS_AMD, f"iu8 WMMA numeric gate needs the AMD backend (Device.DEFAULT={Device.DEFAULT})")
class TestQ4KInt8WmmaValueSemantics(unittest.TestCase):
  # The parent DEV=AMD numeric gate for the signed int8 WMMA dot, now runnable by pytest.
  def test_prefill_mmq_parity(self):
    # Same shapes/algebra as extra/qk/prefill_mmq_parity_gate.run: emitted substrate vs (x_dq @ ref_w.T).
    for n, k, m in [(64, 256, 16), (32, 512, 16), (16, 768, 16)]:
      with self.subTest(n=n, k=k, m=m):
        words, xq, xscales, ref_out = _q4k_reference_and_inputs(n, k, m, seed=1337)
        spec = describe_q4k_int8_wmma_prefill(n, k, m, role="parity")
        got = emit_q4k_int8_wmma_prefill_tensor(words, xq, xscales, spec).numpy()
        rel = _rel_rmse(got, ref_out)
        self.assertLess(rel, RTOL, f"MMQ signed-dot parity FAILED n={n} k={k} m={m} rel_rmse={rel:.3e}")

  def test_tiled_one_tile_microgate_parity(self):
    # The bounded one-output-tile lowering (group_tile == groups) from q4k_wmma_tiled_microgate.PROBE. It uses
    # distinct seeds for weights (20260705) and activation (20260706), so build the reference to match exactly.
    n, k, m = 16, 256, 16
    words, ref_w = _make_q4k_words(n, k, 20260705)
    x = Tensor(np.random.default_rng(20260706).standard_normal((m, k)).astype(np.float32)).realize()
    xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
    x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) *
            xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
    ref_out = (x_dq @ ref_w.T).numpy()
    spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="microgate", m_tile=16, n_tile=16, group_tile=8)
    got = emit_q4k_int8_wmma_tiled_prefill_tensor(words, xq, xscales, spec).realize().numpy()
    rel = _rel_rmse(got, ref_out)
    self.assertLess(rel, RTOL, f"tiled one-tile signed-dot parity FAILED rel_rmse={rel:.3e}")
    # the live RAW tensor must stay bounded to the declared tile (no full [groups,M,N] materialization).
    self.assertEqual(spec.live_raw_elems, spec.forbidden_full_raw_elems,
                     "one-tile microgate spec should have live == full for this bounded shape")


if __name__ == "__main__":
  unittest.main()
