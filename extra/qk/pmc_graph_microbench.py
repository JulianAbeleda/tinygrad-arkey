"""Minimal graph-PMC reproducer / A-B harness.

Runs a few chained small matmuls wrapped in TinyJit (-> HCQGraph) with PMC enabled, then reports,
per captured ProfilePMCEvent, which scheduled counters came back nonzero. Before the fix only
SQ_BUSY_CYCLES (perfcounter index 0) is nonzero; after the fix the full set should be.

Usage:
  DEV=AMD PROFILE=1 PMC=1 PMC_GRAPH=1 .venv/bin/python extra/qk/pmc_graph_microbench.py
"""
from __future__ import annotations

import itertools
from tinygrad import Tensor, TinyJit, Device
from tinygrad.device import Compiled


def _counter_totals(e) -> dict[str, int]:
  # Mirrors extra/qk/prefill_boltbeam_trace.py::_pmc_stats: sequential read per PMCSample.
  view, ptr = memoryview(e.blob).cast("Q"), 0
  out: dict[str, int] = {}
  for s in e.sched:
    total = 0
    for _ in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa), range(s.wgp)):
      total += int(view[ptr]); ptr += 1
    out[s.name] = out.get(s.name, 0) + total
  return out


def main() -> int:
  dev = Device["AMD"]
  print("pmc_enabled:", getattr(dev, "pmc_enabled", None))

  N = 512
  w1 = Tensor.ones(N, N).contiguous().realize()
  w2 = Tensor.ones(N, N).contiguous().realize()

  @TinyJit
  def step(x: Tensor) -> Tensor:
    return ((x @ w1).relu() @ w2).relu()

  x = Tensor.ones(N, N).contiguous().realize()
  # Loop so the jit lowers to HCQGraph and kernels pipeline; PMC collect fires at kickoff>1 / __del__.
  for _ in range(6):
    x = step(x).realize()
  dev.synchronize()

  pmc = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"]
  print(f"ProfilePMCEvent count: {len(pmc)}")
  if not pmc:
    print("NO PMC EVENTS — graph PMC not active (need PROFILE=1 PMC=1 PMC_GRAPH=1)")
    return 1

  # Aggregate: for each counter name, how many events had it nonzero.
  names = [s.name for s in pmc[0].sched]
  nonzero_events = {n: 0 for n in names}
  for e in pmc:
    tot = _counter_totals(e)
    for n in names:
      if tot.get(n, 0): nonzero_events[n] += 1
  print(f"kerns={len(pmc)}  counters nonzero-in-N-events:")
  for n in names:
    print(f"  {n:24} {nonzero_events[n]}/{len(pmc)}")
  distinct_nonzero = sum(1 for n in names if nonzero_events[n])
  print(f"DISTINCT_NONZERO_COUNTERS={distinct_nonzero}/{len(names)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
