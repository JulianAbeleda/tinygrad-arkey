#!/usr/bin/env python3
"""Part 3, steady-state protocol: the plain compile/check/measure() sweep over m showed
the same burst/idle noise Part 2 found (m=512 relative_mad=0.0795, above the campaign's
own 0.05 stability threshold). This re-measures each m with enough back-to-back single
dispatches, on ONE already-warm adapter, to reach the flat steady state Part 2's
experiment A established (400 consecutive calls, <0.1% drift once past the first call) --
discarding a ramp-up prefix -- so the depth curve is not confounded by that artifact.
"""
import json, statistics, sys

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter, ProtocolError
from tinygrad.engine.realize import time_call

state = json.load(open("scratchpad/_campaign_state.json"))
TARGET = state["target"]
winner = json.load(open("scratchpad/_winner_pair.json"))
CANDIDATE, CHASH, GEOM = winner["candidate"], winner["candidate_hash"], winner["geometry"]
N, K = 12288, 4096
TM = CANDIDATE["schedule"]["tile"]["m"]

M_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
RAMP_DISCARD = 8
STEADY_SAMPLES = 20


def payload(m):
  return {"candidate": CANDIDATE, "candidate_hash": CHASH, "target_identity": TARGET,
          "candidate_geometry": GEOM, "fixture_shape": {"m": m, "n": N, "k": K}}


results = []
for m in M_VALUES:
  operations = 2 * m * N * K
  adapter = MetalAdapter()
  try:
    adapter.compile(payload(m))
    check = adapter.check(payload(m))
  except ProtocolError as exc:
    print(f"m={m:5d}  BLOCKED  {exc.code}  {exc}"); results.append({"m": m, "status": "BLOCKED"}); continue
  if not check.get("correct"):
    print(f"m={m:5d}  INCORRECT"); results.append({"m": m, "status": "INCORRECT"}); continue
  prepared = adapter._prepared(payload(m))
  call = prepared.call
  all_ns = []
  for _ in range(RAMP_DISCARD + STEADY_SAMPLES):
    Device["METAL"].synchronize()
    ns = time_call(call) * 1e9
    Device["METAL"].synchronize()
    all_ns.append(ns)
  ramp_ns, steady_ns = all_ns[:RAMP_DISCARD], all_ns[RAMP_DISCARD:]
  steady_med = statistics.median(steady_ns)
  steady_mad = statistics.median(abs(v - steady_med) for v in steady_ns) / steady_med
  gflops = operations / (steady_med * 1e-9) / 1e9
  results.append({"m": m, "status": "MEASURED", "steady_median_ns": steady_med, "relative_mad": steady_mad,
                   "gflops": gflops, "ramp_ns": ramp_ns, "steady_ns": steady_ns, "correct": check["correct"]})
  print(f"m={m:5d}  steady_median_ns={steady_med:>12.0f}  gflops={gflops:>8.1f}  relative_mad={steady_mad:.4f}  "
        f"ramp_first={ramp_ns[0]:.0f}  ramp_last={ramp_ns[-1]:.0f}  correct={check['correct']}")

json.dump(results, open("scratchpad/_part3_depth_steadystate_result.json", "w"), indent=2)

measured = [r for r in results if r["status"] == "MEASURED"]
if measured:
  gs = [r["gflops"] for r in measured]
  print(f"\ngflops range across legal m values (steady state): {min(gs):.1f} - {max(gs):.1f}")
  print(f"spread (max-min)/median: {(max(gs)-min(gs))/statistics.median(gs)*100:.2f}%")
  base = next((r["gflops"] for r in measured if r["m"] == 512), None)
  print(f"m=512 steady-state: {base:.1f} GFLOPS  (doc's isolated figure: 3609.6)")
  for r in measured:
    print(f"  m={r['m']:5d}: {r['gflops']:.1f} GFLOPS  ({100*(r['gflops']-base)/base:+.2f}% vs m=512)")
