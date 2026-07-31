#!/usr/bin/env python3
"""Phase A: compile-only sweep over every (candidate, geometry) pair from
scratchpad/_campaign_state.json. Cheap (no numpy oracle, no GPU kernel launch):
just proves which candidates reach build_precontract_lds_stage vs decline vs
BLOCKED/raise, and how long compile takes. Writes scratchpad/_phase_a_result.json.
"""
import json, sys, time, traceback

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter, ProtocolError

state = json.load(open("scratchpad/_campaign_state.json"))
pairs = state["candidates"]
print(f"loaded {len(pairs)} (candidate, geometry) pairs")

adapter = MetalAdapter()
results = []
t0 = time.time()
for i, row in enumerate(pairs):
  candidate, chash, geom = row["candidate"], row["candidate_hash"], row["geometry"]
  payload = {
    "candidate": candidate, "candidate_hash": chash, "target_identity": state["target"],
    "candidate_geometry": geom,
    "fixture_shape": {"m": candidate["workload"]["shape"]["m"], "n": candidate["workload"]["shape"]["n"],
                       "k": candidate["workload"]["shape"]["k"]},
    "samples": 5, "warmups": 2,
  }
  entry = {"index": i, "candidate_hash": chash, "geometry": geom,
           "tile": candidate["schedule"]["tile"], "threads": candidate["schedule"]["launch"]["threads"]}
  t1 = time.time()
  try:
    result = adapter.compile(payload)
    prepared = adapter._prepared(payload)
    source = prepared.ast.src[3].arg
    has_wmma = "__WMMA_8_8_8_half_float" in source
    has_simd = "simdgroup_multiply_accumulate" in source
    entry.update(status="COMPILED", reached_precontract=bool(has_wmma and has_simd),
                 source_sha256=result["source_sha256"], compile_s=time.time() - t1)
  except ProtocolError as exc:
    entry.update(status="BLOCKED", code=exc.code, message=str(exc), compile_s=time.time() - t1)
  except Exception as exc:
    entry.update(status="EXCEPTION", exception_type=type(exc).__name__, message=str(exc), compile_s=time.time() - t1)
  results.append(entry)
  tag = entry["status"] + (":precontract" if entry.get("reached_precontract") else (":GENERIC_DECLINE" if entry["status"]=="COMPILED" else ""))
  print(f"[{i+1}/{len(pairs)}] tile={entry['tile']} geom={geom} -> {tag} ({entry['compile_s']:.2f}s)")

print(f"\ntotal phase A time: {time.time()-t0:.1f}s")
n_precontract = sum(1 for r in results if r.get("reached_precontract"))
n_generic = sum(1 for r in results if r["status"] == "COMPILED" and not r.get("reached_precontract"))
n_blocked = sum(1 for r in results if r["status"] == "BLOCKED")
n_exc = sum(1 for r in results if r["status"] == "EXCEPTION")
print(f"reached precontract WMMA kernel: {n_precontract}")
print(f"compiled but GENERIC decline: {n_generic}")
print(f"BLOCKED (ProtocolError): {n_blocked}")
print(f"EXCEPTION: {n_exc}")

from collections import Counter
print("\nBLOCKED reasons:", Counter(r["code"] for r in results if r["status"] == "BLOCKED"))
print("EXCEPTION types:", Counter(r["exception_type"] for r in results if r["status"] == "EXCEPTION"))
for r in results:
  if r["status"] in ("BLOCKED", "EXCEPTION"):
    print(" ", r["status"], r.get("code", r.get("exception_type")), "-", r["message"][:160], "tile=", r["tile"], "geom=", r["geometry"])

json.dump(results, open("scratchpad/_phase_a_result.json", "w"), indent=2)
print("\nwrote scratchpad/_phase_a_result.json")
