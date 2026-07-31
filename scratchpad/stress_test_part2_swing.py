#!/usr/bin/env python3
"""Part 2: distinguish sustained-load throttling vs cache residency for the campaign's
winning geometry (tm=64, tn=32, tk=32, wm=4, wn=1, bc=1), shape (512, 12288, 4096).

Three sub-experiments, all using the same MetalAdapter/time_call machinery the campaign
used (extra.llm_research.search_provider.MetalAdapter), with Device["METAL"].synchronize()
around every measurement exactly as the campaign scripts did:

  A. Long sustained single-buffer run: hundreds of consecutive single-sample dispatches
     against ONE adapter's buffers. If GFLOPS decays monotonically over the run, that is
     throttling. If it drops once (first call) then stays flat, that is a one-time
     cache-warm effect, not throttling.

  B. Two-buffer alternation: two INDEPENDENT MetalAdapter instances (adapterA, adapterB),
     each with its own ~28MB packed-weight buffer at a distinct physical address. Alternate
     A,B,A,B,... dispatches so consecutive calls never touch the same buffer. If throughput
     stays at the isolated-repeat level even while alternating, cache residency of one
     specific buffer is not the explanation. If it drops toward the sweep level, that
     supports cache residency.

  C. Gap test: for one adapter, measure the very first call, then vary an idle host-side
     gap (no GPU work) before the next call, across several gap lengths, and compare
     "first call after gap" vs "steady-state calls with no gap".
"""
import json, statistics, sys, time

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter

state = json.load(open("scratchpad/_campaign_state.json"))
TARGET = state["target"]  # exact same target identity the campaign used, not retyped
winner = json.load(open("scratchpad/_winner_pair.json"))
CANDIDATE, CHASH, GEOM = winner["candidate"], winner["candidate_hash"], winner["geometry"]
M, N, K = 512, 12288, 4096
OPERATIONS = 2 * M * N * K


def payload():
  return {"candidate": CANDIDATE, "candidate_hash": CHASH, "target_identity": TARGET,
          "candidate_geometry": GEOM, "fixture_shape": {"m": M, "n": N, "k": K},
          "samples": 1, "warmups": 0}


def one_call_ns(adapter: MetalAdapter) -> float:
  """One synchronized dispatch+timing of the prepared call, bypassing measure()'s own
  warmup loop so every call here is individually controllable and individually synced."""
  from tinygrad.engine.realize import time_call
  prepared = adapter._prepared(payload())
  call = prepared.call
  Device["METAL"].synchronize()
  ns = time_call(call) * 1e9
  Device["METAL"].synchronize()
  return ns


def gflops(ns: float) -> float:
  return OPERATIONS / (ns * 1e-9) / 1e9


print("=== setup ===")
print("winning geometry:", GEOM, "tile:", CANDIDATE["schedule"]["tile"])

# ---------------------------------------------------------------------------
# Experiment A: long sustained single-buffer run
# ---------------------------------------------------------------------------
print("\n=== A. long sustained single-buffer run ===")
adapterA_sustained = MetalAdapter()
adapterA_sustained.compile(payload())
adapterA_sustained.check(payload())  # correctness gate, same as campaign
N_SUSTAINED = 400
sustained_ns = []
t_wall0 = time.time()
for i in range(N_SUSTAINED):
  sustained_ns.append(one_call_ns(adapterA_sustained))
t_wall1 = time.time()
print(f"{N_SUSTAINED} consecutive single-buffer dispatches, wall time {t_wall1 - t_wall0:.1f}s")
first_call = sustained_ns[0]
# split remaining calls into 10 contiguous buckets, report median GFLOPS per bucket
rest = sustained_ns[1:]
bucket_size = len(rest) // 10
buckets_gflops = []
for b in range(10):
  chunk = rest[b*bucket_size:(b+1)*bucket_size]
  med_ns = statistics.median(chunk)
  buckets_gflops.append(gflops(med_ns))
print(f"first call: {first_call:.0f} ns -> {gflops(first_call):.1f} GFLOPS")
print("steady-state buckets (median GFLOPS per 1/10 of the remaining run, in order):")
for i, g in enumerate(buckets_gflops):
  print(f"  bucket {i}: {g:.1f} GFLOPS")
overall_first10_med = statistics.median(rest[:bucket_size])
overall_last10_med = statistics.median(rest[-bucket_size:])
print(f"first bucket median GFLOPS: {gflops(statistics.median(rest[:bucket_size])):.1f}")
print(f"last bucket median GFLOPS: {gflops(statistics.median(rest[-bucket_size:])):.1f}")
pct_drift = 100 * (overall_last10_med - overall_first10_med) / overall_first10_med
print(f"drift (first-bucket ns -> last-bucket ns): {pct_drift:+.2f}%  (positive = later calls slower = throttling-consistent)")

# ---------------------------------------------------------------------------
# Experiment B: two-buffer alternation
# ---------------------------------------------------------------------------
print("\n=== B. two-buffer alternation (defeats single-buffer cache residency) ===")
adapterA = MetalAdapter(); adapterA.compile(payload()); adapterA.check(payload())
adapterB = MetalAdapter(); adapterB.compile(payload()); adapterB.check(payload())
prepA, prepB = adapterA._prepared(payload()), adapterB._prepared(payload())
print("adapterA / adapterB distinct buffer identity check:",
      "DIFFERENT" if id(prepA.call.src[1].buffer) != id(prepB.call.src[1].buffer) else "SAME (unexpected)")

N_ALT = 60
# baseline: A alone, back-to-back (single-buffer, matches experiment A's regime)
a_alone_ns = [one_call_ns(adapterA) for _ in range(N_ALT)]
# alternating: A, B, A, B, ...
alt_a_ns, alt_b_ns = [], []
for i in range(N_ALT):
  alt_a_ns.append(one_call_ns(adapterA))
  alt_b_ns.append(one_call_ns(adapterB))

med_a_alone = statistics.median(a_alone_ns)
med_alt_a = statistics.median(alt_a_ns)
med_alt_b = statistics.median(alt_b_ns)
print(f"A alone, back-to-back:      median {med_a_alone:.0f} ns -> {gflops(med_a_alone):.1f} GFLOPS  (n={N_ALT})")
print(f"A while alternating w/ B:   median {med_alt_a:.0f} ns -> {gflops(med_alt_a):.1f} GFLOPS  (n={N_ALT})")
print(f"B while alternating w/ A:   median {med_alt_b:.0f} ns -> {gflops(med_alt_b):.1f} GFLOPS  (n={N_ALT})")
alt_drop_pct = 100 * (med_alt_a - med_a_alone) / med_a_alone
print(f"alternation penalty on A: {alt_drop_pct:+.2f}%  (positive = alternating is slower = cache-residency-consistent)")

# ---------------------------------------------------------------------------
# Experiment C: gap test
# ---------------------------------------------------------------------------
print("\n=== C. gap test (idle host-side sleep before next call, no GPU work in the gap) ===")
adapterC = MetalAdapter(); adapterC.compile(payload()); adapterC.check(payload())
# steady-state reference: 10 calls with zero gap
steady_ns = [one_call_ns(adapterC) for _ in range(10)]
steady_med = statistics.median(steady_ns)
print(f"steady-state (0s gap) reference median: {steady_med:.0f} ns -> {gflops(steady_med):.1f} GFLOPS")

for gap_s in (0.0, 1.0, 5.0, 15.0):
  if gap_s > 0:
    time.sleep(gap_s)
  first_after_gap = one_call_ns(adapterC)
  # immediately follow with 5 more calls at zero gap to see how fast it "recovers"
  followers = [one_call_ns(adapterC) for _ in range(5)]
  print(f"gap={gap_s:>5.1f}s  first_after_gap={first_after_gap:.0f} ns ({gflops(first_after_gap):.1f} GFLOPS)  "
        f"followers_median={statistics.median(followers):.0f} ns ({gflops(statistics.median(followers)):.1f} GFLOPS)")

print("\ndone")
