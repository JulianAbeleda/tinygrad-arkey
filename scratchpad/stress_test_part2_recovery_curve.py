#!/usr/bin/env python3
"""Follow-up to stress_test_part2_swing.py's gap test: how many consecutive dispatches
does it take to recover from a post-idle-gap slowdown back to the ~3610 GFLOPS
steady-state, for the campaign's winning geometry?"""
import json, statistics, sys, time

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter
from tinygrad.engine.realize import time_call

state = json.load(open("scratchpad/_campaign_state.json"))
TARGET = state["target"]
winner = json.load(open("scratchpad/_winner_pair.json"))
CANDIDATE, CHASH, GEOM = winner["candidate"], winner["candidate_hash"], winner["geometry"]
M, N, K = 512, 12288, 4096
OPERATIONS = 2 * M * N * K


def payload():
  return {"candidate": CANDIDATE, "candidate_hash": CHASH, "target_identity": TARGET,
          "candidate_geometry": GEOM, "fixture_shape": {"m": M, "n": N, "k": K},
          "samples": 1, "warmups": 0}


def one_call_ns(adapter):
  prepared = adapter._prepared(payload())
  call = prepared.call
  Device["METAL"].synchronize()
  ns = time_call(call) * 1e9
  Device["METAL"].synchronize()
  return ns


def gflops(ns):
  return OPERATIONS / (ns * 1e-9) / 1e9


adapter = MetalAdapter(); adapter.compile(payload()); adapter.check(payload())
# establish steady state
_ = [one_call_ns(adapter) for _ in range(10)]

for gap_s in (2.0, 5.0, 10.0):
  time.sleep(gap_s)
  curve = [one_call_ns(adapter) for _ in range(25)]
  print(f"\n--- recovery curve after {gap_s:.0f}s idle gap ---")
  for i, ns in enumerate(curve):
    print(f"  call {i:2d}: {ns:>12.0f} ns  {gflops(ns):>7.1f} GFLOPS")
  # re-establish steady state before next gap so each gap test starts from the same place
  _ = [one_call_ns(adapter) for _ in range(10)]

print("\ndone")
