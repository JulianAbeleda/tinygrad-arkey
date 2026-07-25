"""Encodes the invariant behind the SQC (inst) wild-PC fault class.

THE INVARIANT: a `BumpAllocator` with `wrap=True` hands back memory from offset 0 with no notion of whether
the GPU has finished reading what was there. If the GPU reads that region asynchronously -- kernargs (which
contain the dispatch packet), PM4 indirect buffers (which contain the command stream), NV command buffers --
then recycling under an in-flight submission corrupts what the GPU is executing. That is the mechanism
behind the `SQC (inst)` page faults documented in
docs/gpu-page-fault-population-analysis-20260725.md: a wave launched at a wild PC.

Three instances were found and fixed (kernargs, pm4_ib_alloc, nv cmdq_allocator). This test exists so a
FOURTH cannot be added silently. Point fixes do not prevent recurrence; an enumerated invariant does.

If you add a `BumpAllocator(..., wrap=True)`, you must classify it here. There is no default -- an
unclassified site fails this test on purpose, because the failure mode it guards against is a rare,
non-deterministic GPU reset that costs days to trace back.
"""
import pathlib, re, unittest
import tinygrad

ROOT = pathlib.Path(tinygrad.__file__).parent

# Every wrapping BumpAllocator, and WHY it is safe. Keyed by "<path>:<line-content-fragment>".
#   "drained"    -> guarded by a defer-until-drained wait before reuse; name the rollback flag.
#   "cannot-wrap"-> sized so alloc() can never exceed size, so the wrap branch is unreachable.
CLASSIFIED = {
  "runtime/support/hcq.py": ("drained", "KERNARGS_WRAP_DRAIN",
    "kernargs hold the AMD dispatch packet (ops_amd.py kernel_object); proven by A/B, "
    "guard off = [15,14,15,15] reuses-in-flight, guard on = [0,0,0,0]"),
  "runtime/ops_amd.py": ("drained", "PM4_IB_WRAP_DRAIN",
    "PM4 indirect buffers hold the command stream the CP fetches asynchronously"),
  "runtime/ops_nv.py": ("drained", "NV_CMDQ_WRAP_DRAIN",
    "NV command buffer a GPFIFO entry points execution at; UNTESTED - no NVIDIA GPU available"),
  "runtime/graph/hcq.py": ("cannot-wrap", None,
    "sized exactly to the sum of its contents (kernargs_size accumulated per call), so it never wraps"),
}


def _wrapping_sites() -> dict[str, list[str]]:
  """Every BumpAllocator construction that can wrap, grouped by module path relative to tinygrad/."""
  found: dict[str, list[str]] = {}
  for path in ROOT.rglob("*.py"):
    for line in path.read_text().splitlines():
      if "BumpAllocator(" not in line or line.lstrip().startswith(("#", "def ", "class ")): continue
      # wrap defaults to True, so only an explicit wrap=False opts out
      if re.search(r"wrap\s*=\s*False", line): continue
      found.setdefault(str(path.relative_to(ROOT)), []).append(line.strip())
  return found


class TestWrappingAllocatorInvariant(unittest.TestCase):
  def test_every_wrapping_allocator_is_classified(self):
    sites = _wrapping_sites()
    unclassified = sorted(set(sites) - set(CLASSIFIED))
    self.assertEqual(unclassified, [], "\n\nA wrapping BumpAllocator was added without classifying it.\n"
      "If the GPU reads this region asynchronously, recycling it under an in-flight submission corrupts what\n"
      "the GPU is executing -- the SQC (inst) wild-PC fault. Either guard it with a defer-until-drained wait\n"
      "(see fill_kernargs in runtime/support/hcq.py) or show it cannot wrap, then add it to CLASSIFIED.\n"
      f"Unclassified: {unclassified}")

  def test_no_classification_is_stale(self):
    # A classified site that no longer exists means the table is drifting away from the code.
    sites = _wrapping_sites()
    self.assertEqual(sorted(set(CLASSIFIED) - set(sites)), [], "CLASSIFIED lists a site that no longer wraps")

  def test_drained_sites_still_carry_their_guard(self):
    # The guard is the whole protection; if its flag vanishes from the file, the site is unprotected while
    # still looking classified.
    for mod, (kind, flag, why) in CLASSIFIED.items():
      if kind != "drained": continue
      src = (ROOT/mod).read_text()
      self.assertIn(flag, src, f"{mod} is classified 'drained' via {flag} but that flag is gone: {why}")
      self.assertIn(".wraps", src, f"{mod} no longer checks the BumpAllocator wrap counter")

  def test_drain_guards_default_to_enabled(self):
    import tinygrad.device as D
    for flag in ("KERNARGS_WRAP_DRAIN", "PM4_IB_WRAP_DRAIN", "NV_CMDQ_WRAP_DRAIN"):
      self.assertEqual(getattr(D, flag).value, 1, f"{flag} must ship enabled; 0 reopens the fault")

  def test_bump_allocator_still_reports_wraps(self):
    # Every guard is built on this counter.
    from tinygrad.runtime.support.memory import BumpAllocator
    a = BumpAllocator(64, wrap=True)
    a.alloc(64); self.assertEqual(a.wraps, 0)
    a.alloc(64); self.assertEqual(a.wraps, 1)


if __name__ == "__main__":
  unittest.main()
