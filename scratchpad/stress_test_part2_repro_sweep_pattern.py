#!/usr/bin/env python3
"""Reproduce the sweep's actual call pattern faithfully: run the campaign's own run_one()
(fresh MetalAdapter() -> compile -> check -> measure(samples=5, warmups=2), exactly what
happened once per candidate in the 91-pair sweep) repeatedly for ONLY the winning
geometry, back to back. Each call includes a fresh compile + a real correctness check
(full numpy oracle over 512x12288x4096, CPU-bound) before the timed GPU samples -- i.e.
each call reproduces one real "burst after CPU-bound idle" cycle from the sweep, not just
a bare GPU dispatch loop.

Question: does repeating this real burst/idle cycle 15x for one geometry reproduce the
sweep's 2558 GFLOPS figure, or does it stay near the isolated 3610 figure?
"""
import json, statistics, sys, time

sys.path.insert(0, ".")
sys.path.insert(0, "scratchpad")
from campaign_phase_b_measure import run_one

winner = json.load(open("scratchpad/_winner_pair.json"))
CANDIDATE, CHASH, GEOM = winner["candidate"], winner["candidate_hash"], winner["geometry"]

N_REPEATS = 15
results = []
t0 = time.time()
for i in range(N_REPEATS):
  t_call0 = time.time()
  r = run_one(CANDIDATE, CHASH, GEOM)
  t_call1 = time.time()
  gap = t_call1 - t_call0
  results.append((r, gap))
  print(f"[{i+1}/{N_REPEATS}] status={r['status']} median_ns={r.get('median_ns')} "
        f"gflops={r.get('gflops')}  wall_for_this_run_one_call={gap:.2f}s")

gflops_list = [r["gflops"] for r, _ in results if r["status"] == "MEASURED"]
print(f"\ntotal wall time: {time.time()-t0:.1f}s")
print(f"n measured: {len(gflops_list)}")
print(f"median gflops across {len(gflops_list)} fresh-adapter run_one() calls: {statistics.median(gflops_list):.1f}")
print(f"min: {min(gflops_list):.1f}  max: {max(gflops_list):.1f}")
print(f"all values: {[round(g,1) for g in gflops_list]}")
