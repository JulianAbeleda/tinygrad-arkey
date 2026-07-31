#!/usr/bin/env python3
"""Phase B: check + measure every (candidate, geometry) pair with a FRESH MetalAdapter
per pair.

IMPORTANT finding from phase A: MetalAdapter._prepared_key/_compile_cache key on
{candidate, exact, fixture} only -- candidate_geometry is NOT part of the cache key.
Since bc (buffer_count) and the wm/wn split never appear in the hashed v2 candidate
schema, two distinct (wm, wn, bc) sidecars that legally share one (tm, tn, tk, threads)
candidate silently collide on the SAME cache entry: the second geometry's
compile()/check()/measure() served the FIRST geometry's compiled source verbatim
(source_sha256 identical), never rebuilding for the requested bc. Confirmed directly:
tile (16,16,32)/threads=32 with geometry wm=1,wn=1,bc=1 and the same tile with bc=2
produced byte-identical source_sha256 when reusing one adapter. A single shared adapter
across geometries would therefore silently under-count how many DISTINCT geometries were
actually measured. This script uses one MetalAdapter() per (candidate, geometry) pair so
every requested geometry is genuinely, independently compiled and measured.
"""
import json, statistics, sys, time, traceback

sys.path.insert(0, ".")
from tinygrad.device import Device
from extra.llm_research.search_provider import MetalAdapter, ProtocolError

state = json.load(open("scratchpad/_campaign_state.json"))
pairs = state["candidates"]
TARGET = state["target"]

M, N, K = 512, 12288, 4096


def build_payload(candidate, chash, geom, samples=5, warmups=2):
  return {"candidate": candidate, "candidate_hash": chash, "target_identity": TARGET,
          "candidate_geometry": geom, "fixture_shape": {"m": M, "n": N, "k": K},
          "samples": samples, "warmups": warmups}


def run_one(candidate, chash, geom, *, samples=5, warmups=2):
  """Fresh adapter; returns a result dict, never raises."""
  payload = build_payload(candidate, chash, geom, samples, warmups)
  adapter = MetalAdapter()
  out = {"candidate_hash": chash, "tile": candidate["schedule"]["tile"],
         "threads": candidate["schedule"]["launch"]["threads"], "geometry": geom}
  try:
    compiled = adapter.compile(payload)
    prepared = adapter._prepared(payload)
    source = prepared.ast.src[3].arg
    out["reached_precontract"] = ("__WMMA_8_8_8_half_float" in source) and ("simdgroup_multiply_accumulate" in source)
    out["source_sha256"] = compiled["source_sha256"]
  except ProtocolError as exc:
    out.update(status="BLOCKED", code=exc.code, message=str(exc)); return out
  except Exception as exc:
    out.update(status="EXCEPTION", exception_type=type(exc).__name__, message=str(exc)); return out
  try:
    check_result = adapter.check(payload)
  except ProtocolError as exc:
    out.update(status="CHECK_BLOCKED", code=exc.code, message=str(exc)); return out
  except Exception as exc:
    out.update(status="CHECK_EXCEPTION", exception_type=type(exc).__name__, message=str(exc)); return out
  out["correct"] = check_result.get("correct")
  out["check_evidence"] = {k: v for k, v in check_result.items() if k not in ("correct",)}
  if not out["correct"]:
    out["status"] = "INCORRECT"
    return out
  Device["METAL"].synchronize()
  try:
    measured = adapter.measure(payload)
  except ProtocolError as exc:
    out.update(status="MEASURE_BLOCKED", code=exc.code, message=str(exc)); return out
  except Exception as exc:
    out.update(status="MEASURE_EXCEPTION", exception_type=type(exc).__name__, message=str(exc)); return out
  Device["METAL"].synchronize()
  samples_ns = measured["samples_ns"]
  median_ns = statistics.median(samples_ns)
  operations = measured["work_bytes"]["operations"]
  gflops = operations / (median_ns * 1e-9) / 1e9
  out.update(status="MEASURED", samples_ns=samples_ns, median_ns=median_ns, operations=operations, gflops=gflops)
  return out


if __name__ == "__main__":
  results = []
  t0 = time.time()
  for i, row in enumerate(pairs):
    candidate, chash, geom = row["candidate"], row["candidate_hash"], row["geometry"]
    r = run_one(candidate, chash, geom)
    results.append(r)
    if r["status"] == "MEASURED":
      print(f"[{i+1}/{len(pairs)}] tile={r['tile']} geom={geom} -> MEASURED median_ns={r['median_ns']:.0f} gflops={r['gflops']:.1f} correct={r['correct']}")
    else:
      print(f"[{i+1}/{len(pairs)}] tile={r['tile']} geom={geom} -> {r['status']} {r.get('code', r.get('exception_type',''))} {r.get('message','')[:120]}")
  print(f"\ntotal phase B time: {time.time()-t0:.1f}s")
  json.dump(results, open("scratchpad/_phase_b_result.json", "w"), indent=2)
  print("wrote scratchpad/_phase_b_result.json")

  from collections import Counter
  print("\nstatus histogram:", Counter(r["status"] for r in results))
