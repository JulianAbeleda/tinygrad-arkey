"""Encodes the measurement-authority contract, so which harness is canonical stops being re-litigated.

Two failures this guards against, both of which actually happened:

1. `extra/qk/bench.py` is declared the single sanctioned entry for reported throughput, but it dispatches to
   its measurement cores BY FILE PATH via argv builders. Twice a cleanup commit deleted a core while bench.py
   kept calling it: `decode_runtime_overhead.py` (45cfc399c) and `prefill_boltbeam_trace.py` (0e02a1976).
   `bench.py --decode` therefore failed with file-not-found, which is why decode was never re-measured after
   2026-07-03. A dispatch target that does not exist is a silently dead benchmark.

2. Multiple modules could emit a decode tok/s number under different definitions. `model_e2e_bench.py`
   measures decode from a ONE-token seed (`model.generate([seed])`) over a growing window, so its ctx labels
   describe KV *allocation*, not decode depth. `decode_runtime_overhead.py` prefills to exactly ctx first.
   Numbers from the two are not comparable, and the shallow one produced a physically impossible result
   (8B decode RISING 103.9 -> 107.9 tok/s from ctx512 to ctx4096).

The physics, stated once so it is not re-derived: at batch=1 decode is HBM-bound and every token must read
every weight plus the whole KV cache at the current depth. tok/s therefore MUST fall as depth grows --
about 9% for 8B and 6% for 14B between ctx512 and ctx4096. A harness whose ctx columns do not move in that
direction is not measuring depth, whatever its filename says.
"""
import pathlib, re, unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestBenchDispatchTargetsExist(unittest.TestCase):
  """Every path bench.py shells out to must exist. Catches both historical deletions."""

  def _argv_paths(self) -> list[str]:
    # The argv builders hardcode the core's path as the first element.
    out = []
    for mod in ("extra/qk/decode/decode_harness.py", "extra/qk/prefill/prefill_harness.py"):
      for m in re.finditer(r'return \["(extra/[^"]+\.py)"', (ROOT/mod).read_text()): out.append(m.group(1))
    return out

  def test_argv_builders_reference_existing_files(self):
    paths = self._argv_paths()
    self.assertTrue(paths, "found no dispatch targets -- the argv builders changed shape, update this test")
    missing = [p for p in paths if not (ROOT/p).exists()]
    self.assertEqual(missing, [], f"bench.py dispatches to files that do not exist: {missing}")

  def test_bench_entry_itself_imports(self):
    import importlib
    importlib.import_module("extra.qk.bench")


class TestSingleDecodeAuthority(unittest.TestCase):
  """One decode measurement definition is canonical; others must not masquerade as ctx-varying."""

  CANONICAL = ROOT/"extra/qk/decode/decode_runtime_overhead.py"

  def test_canonical_decode_core_prefills_to_depth(self):
    src = self.CANONICAL.read_text()
    self.assertIn("_prefill", src, "the canonical decode core must prefill to depth before timing")
    self.assertNotIn("generate([seed])", src, "the canonical core must not decode from a bare seed")

  def test_shallow_harness_is_marked_non_authoritative(self):
    # model_e2e_bench decodes from a one-token seed. It may keep existing (it also covers VRAM/correctness),
    # but it must say in-file that its decode number is not a ctx-labelled authority, or the next person
    # will put it in the README again.
    src = (ROOT/"extra/llm/model_e2e_bench.py").read_text()
    self.assertIn("generate([seed])", src, "model_e2e_bench changed shape; re-check this contract")
    self.assertIn("NOT A CTX-LABELLED DECODE AUTHORITY", src,
                  "model_e2e_bench.measure_decode decodes from a 1-token seed, so its ctx columns describe KV "
                  "allocation, not depth. It must be marked non-authoritative in-file.")


class TestDecodeDepthContract(unittest.TestCase):
  """The acceptance test any decode harness must satisfy, derived from memory traffic alone."""

  # (name, weight_bytes, layers, kv_heads, head_dim)
  MODELS = [("8B", 4.68*1024**3, 36, 8, 128), ("14B", 8.38*1024**3, 40, 8, 128)]

  @staticmethod
  def _expected_drop(weights: float, layers: int, kvh: int, hd: int, d0: int, d1: int) -> float:
    kv = lambda d: layers*kvh*hd*2*2*d          # K and V, fp16
    return 1.0 - (weights+kv(d0))/(weights+kv(d1))

  def test_predicted_slowdown_is_material_and_negative(self):
    # If this ever came out ~0, the contract below would be vacuous.
    for name, w, l, k, h in self.MODELS:
      drop = self._expected_drop(w, l, k, h, 512, 4096)
      self.assertGreater(drop, 0.04, f"{name}: depth must cost at least a few percent")
      self.assertLess(drop, 0.20, f"{name}: sanity bound")

  def test_a_rising_decode_curve_is_rejected(self):
    # The 07-03 README numbers: 8B 103.9 -> 107.9 tok/s across ctx512 -> ctx4096.
    self.assertFalse(self.accepts("8B", 103.9, 107.9), "a decode curve that RISES with depth must be rejected")

  def test_todays_measured_curves_are_accepted(self):
    self.assertTrue(self.accepts("8B", 95.11, 87.69), "ours 8B (fixed-depth authority) should satisfy the contract")
    self.assertTrue(self.accepts("8B", 97.56, 88.99), "llama-bench tg128 @ d should satisfy the contract")

  @classmethod
  def accepts(cls, model: str, tok_s_512: float, tok_s_4096: float, tol: float = 0.06) -> bool:
    """True iff the measured drop is within `tol` of what memory traffic requires."""
    w, l, k, h = next((w, l, k, h) for n, w, l, k, h in cls.MODELS if n == model)
    expected = cls._expected_drop(w, l, k, h, 512, 4096)
    measured = 1.0 - tok_s_4096/tok_s_512
    return abs(measured - expected) <= tol


if __name__ == "__main__":
  unittest.main()
