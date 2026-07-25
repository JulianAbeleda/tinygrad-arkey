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
   Numbers from the two are not comparable. The shallow one produced a suspicious result (8B decode RISING
   103.9 -> 107.9 tok/s from ctx512 to ctx4096) -- suspicious because bytes/token strictly grow with depth,
   but NOT impossible, see below.

What the physics does and does NOT say (an earlier draft of this file overreached; do not restore it):
  DOES: memory traffic strictly increases with depth -- every token reads every weight plus the whole KV
        cache, so bytes/token at ctx4096 > bytes/token at ctx512, always.
  DOES NOT: fix the direction of tok/s. Decode here runs at only ~50-57% of HBM peak (8B: 485 GB/s at
        ctx512, 494 GB/s at ctx4096 against ~960 peak), so it is NOT bandwidth-saturated. Throughput is
        traffic / efficiency, and efficiency is not constant -- it measurably rises with depth (8B:
        50.5% -> 51.4% of peak). A larger efficiency gain could leave tok/s flat or rising while bytes grow.
Additionally, none of the measurements to date pinned the GPU clock (bench.py exposes --pin-clock and it was
not used), so run-to-run and depth-to-depth clock variation is uncontrolled and can be several percent.

Therefore this file asserts STRUCTURAL properties only -- what a harness measures -- and deliberately makes
no assertion about the direction or magnitude of the throughput curve. A rising decode curve is a reason to
go and check the harness, not proof on its own that it is wrong.
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


if __name__ == "__main__":
  unittest.main()
