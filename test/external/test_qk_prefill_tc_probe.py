#!/usr/bin/env python3
"""Lock the Option B verdict: explicit optimizer-TC Q@Kᵀ + softmax + P@V (GQA via broadcast, materialized
scores) BEATS tinygrad SDPA on the Qwen prefill shape -- so SHAPED_WMMA codegen surgery is avoidable for a
prefill-attention speedup. Skip-if-absent; the DEBUG=2 GPU-time probe (extra/qk_prefill_tc_wr_softmax_probe.py)
stays out of the suite."""
import json, pathlib, unittest

_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "qk-prefill-tc-wr-softmax-probe" / "result.json"

class TestQKPrefillTCProbe(unittest.TestCase):
  def test_option_b_viable(self):
    if not _ARTIFACT.exists(): self.skipTest(f"no artifact at {_ARTIFACT}")
    d = json.loads(_ARTIFACT.read_text())
    self.assertTrue(d["correct"], "explicit TC attention incorrect vs SDPA")
    self.assertTrue(d["tc_fired_qk_or_pv"], "tensor cores did not fire for QK/PV")
    self.assertTrue(d["verdict"].startswith("OPTION B VIABLE"))
    long = next(r for r in d["rows"] if not r.get("faulted") and r["KV"] == max(rr["KV"] for rr in d["rows"] if not rr.get("faulted")))
    self.assertGreaterEqual(long["speedup"], 1.5, "explicit TC attention should beat SDPA >=1.5x at long KV")

if __name__ == "__main__":
  unittest.main()
