#!/usr/bin/env python3
"""Part 3: GFLOPS vs m (token/prefill-depth dimension) for the winning geometry
(tm=64, tn=32, tk=32, wm=4, wn=1, bc=1), n=12288, k=4096 held fixed (ffn_gate_up's real
n/k), varying only the fixture_shape.m the campaign measured at m=512.

The precontract kernel requires m % tm == 0 (no boundary-tile handling; confirmed by the
campaign's own geometry_legal() check "tm does not divide M"), so only multiples of
tm=64 are legal. This does NOT attempt whole-model prefill -- that needs the route
promoted through QUALIFY/POLICY (blocked per the campaign doc). This stays at the single
GEMM-kernel level and sweeps m the same way the campaign's own harness (MetalAdapter,
canonical_packed_reference oracle, time_call) measured m=512, so every point here is
directly comparable to the 3609.6 GFLOPS isolated figure.

Each m gets: fresh MetalAdapter, compile, correctness check against the full numpy
oracle, then 6 samples / 2 warmups (mirrors the campaign's finalist-repeat protocol),
median-of-median across repeats reported, plus relative_mad.
"""
import json, statistics, sys

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter, ProtocolError

state = json.load(open("scratchpad/_campaign_state.json"))
TARGET = state["target"]
winner = json.load(open("scratchpad/_winner_pair.json"))
CANDIDATE, CHASH, GEOM = winner["candidate"], winner["candidate_hash"], winner["geometry"]
N, K = 12288, 4096
TM = CANDIDATE["schedule"]["tile"]["m"]
print(f"winning geometry: {GEOM}  tile: {CANDIDATE['schedule']['tile']}  tm={TM}")

M_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
illegal_probe = [m for m in M_VALUES if m % TM != 0]
print(f"m values requested: {M_VALUES}")
print(f"m values NOT divisible by tm={TM} (expected illegal): {illegal_probe}")


def payload(m, samples=6, warmups=2):
  return {"candidate": CANDIDATE, "candidate_hash": CHASH, "target_identity": TARGET,
          "candidate_geometry": GEOM, "fixture_shape": {"m": m, "n": N, "k": K},
          "samples": samples, "warmups": warmups}


results = []
for m in M_VALUES:
  operations = 2 * m * N * K
  adapter = MetalAdapter()
  row = {"m": m, "operations": operations}
  try:
    compiled = adapter.compile(payload(m))
    row["global_size"] = compiled["launch"]["global_size"]
    row["local_size"] = compiled["launch"]["local_size"]
  except ProtocolError as exc:
    row.update(status="BLOCKED", code=exc.code, message=str(exc)); results.append(row)
    print(f"m={m:5d}  BLOCKED  {exc.code}  {exc}"); continue
  except Exception as exc:
    row.update(status="EXCEPTION", message=f"{type(exc).__name__}: {exc}"); results.append(row)
    print(f"m={m:5d}  EXCEPTION  {type(exc).__name__}: {exc}"); continue
  try:
    check = adapter.check(payload(m))
  except ProtocolError as exc:
    row.update(status="CHECK_BLOCKED", code=exc.code, message=str(exc)); results.append(row)
    print(f"m={m:5d}  CHECK_BLOCKED  {exc.code}  {exc}"); continue
  row["correct"] = check.get("correct")
  if not row["correct"]:
    row["status"] = "INCORRECT"; results.append(row)
    print(f"m={m:5d}  INCORRECT  {check}"); continue
  Device["METAL"].synchronize()
  try:
    measured = adapter.measure(payload(m))
  except ProtocolError as exc:
    row.update(status="MEASURE_BLOCKED", code=exc.code, message=str(exc)); results.append(row)
    print(f"m={m:5d}  MEASURE_BLOCKED  {exc.code}  {exc}"); continue
  Device["METAL"].synchronize()
  samples_ns = measured["samples_ns"]
  median_ns = statistics.median(samples_ns)
  mad = statistics.median(abs(v - median_ns) for v in samples_ns) / median_ns
  gflops = operations / (median_ns * 1e-9) / 1e9
  row.update(status="MEASURED", median_ns=median_ns, relative_mad=mad, gflops=gflops, samples_ns=samples_ns)
  results.append(row)
  print(f"m={m:5d}  MEASURED  median_ns={median_ns:>12.0f}  gflops={gflops:>8.1f}  relative_mad={mad:.4f}  "
        f"global_size={row['global_size']}  local_size={row['local_size']}  correct={row['correct']}")

json.dump(results, open("scratchpad/_part3_depth_result.json", "w"), indent=2)

measured = [r for r in results if r["status"] == "MEASURED"]
if measured:
  gs = [r["gflops"] for r in measured]
  print(f"\ngflops range across legal m values: {min(gs):.1f} - {max(gs):.1f}")
  print(f"spread (max-min)/median: {(max(gs)-min(gs))/statistics.median(gs)*100:.2f}%")
  base = next((r["gflops"] for r in measured if r["m"] == 512), None)
  if base is not None:
    print(f"m=512 (campaign's measured point): {base:.1f} GFLOPS -- doc reports 3609.6 isolated")
    for r in measured:
      print(f"  m={r['m']:5d}: {r['gflops']:.1f} GFLOPS  ({100*(r['gflops']-base)/base:+.2f}% vs m=512)")
