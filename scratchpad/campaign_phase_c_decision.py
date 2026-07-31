#!/usr/bin/env python3
"""Phase C: control row + finalist/control stability repeats + mr9-style decision.

Mirrors boltbeam/search/mr9_semantic_search.py::_decision's policy numbers
(minimum_win_fraction=0.03, maximum_relative_mad=0.05, required_repeats=2,
5 raw samples per run) since _decision itself hard-requires the exact_gguf
oracle string ("exact GGUF tensor + deterministic activation"), which is
mutually exclusive with candidate_geometry (search_provider.py:376-382) --
this campaign's oracle is instead canonical_packed_reference over the real
fixture shape, so the decision logic is reimplemented against that oracle
rather than calling _decision directly.
"""
import json, statistics, sys

sys.path.insert(0, ".")
from tinygrad.device import Device
from campaign_phase_b_measure import run_one, build_payload

state = json.load(open("scratchpad/_campaign_state.json"))
pairs = state["candidates"]
b_results = json.load(open("scratchpad/_phase_b_result.json"))
measured = sorted([r for r in b_results if r["status"] == "MEASURED"], key=lambda r: r["median_ns"])
print(f"legal measured candidates: {len(measured)} / {len(pairs)} pairs")

# ---------------------------------------------------------------------------
# Control row: tinygrad_heuristic.v1, empty transforms, NO candidate_geometry sidecar
# ---------------------------------------------------------------------------
control_candidate = pairs[0]["candidate"]  # any legal candidate JSON; tile/threads are
control_hash = pairs[0]["candidate_hash"]  # inert once candidate_geometry is omitted (opts=None -> full heuristic autotune)
control_payload = {"candidate": control_candidate, "candidate_hash": control_hash, "target_identity": state["target"],
                    "fixture_shape": {"m": 512, "n": 12288, "k": 4096}, "samples": 5, "warmups": 2}
print("\n=== control row (tinygrad_heuristic.v1, no candidate_geometry sidecar) ===")

from extra.llm_research.search_provider import MetalAdapter, ProtocolError
adapter = MetalAdapter()
compiled = adapter.compile(control_payload)
prepared = adapter._prepared(control_payload)
source = prepared.ast.src[3].arg
control_reached_precontract = ("__WMMA_8_8_8_half_float" in source) and ("simdgroup_multiply_accumulate" in source)
print("control reached precontract WMMA path (should be False -- unforced heuristic):", control_reached_precontract)
check = adapter.check(control_payload)
print("control correct:", check["correct"])
Device["METAL"].synchronize()
m = adapter.measure(control_payload)
Device["METAL"].synchronize()
control_median = statistics.median(m["samples_ns"])
control_gflops = m["work_bytes"]["operations"] / (control_median * 1e-9) / 1e9
print(f"control median_ns={control_median:.0f} gflops={control_gflops:.1f} samples_ns={m['samples_ns']}")

# ---------------------------------------------------------------------------
# Finalists: top-2 by initial median_ns among correct MEASURED candidates
# ---------------------------------------------------------------------------
finalists = measured[:2]
print("\n=== top-2 finalists (initial measurement) ===")
for f in finalists:
  print(f"  tile={f['tile']} geom={f['geometry']} median_ns={f['median_ns']:.0f} gflops={f['gflops']:.1f}")

hash_to_row = {p["candidate_hash"]: p for p in pairs}


def repeat_measure(candidate, chash, geom, label, n_repeats=2):
  runs = []
  for i in range(n_repeats):
    r = run_one(candidate, chash, geom)
    print(f"    repeat {i+1}/{n_repeats} [{label}] -> status={r['status']} median_ns={r.get('median_ns')} gflops={r.get('gflops')}")
    runs.append(r)
  return runs


REQUIRED_REPEATS = 2
MAX_RELATIVE_MAD = 0.05
MIN_WIN_FRACTION = 0.03

stability = []
print("\n=== stability repeats (control + top-2 finalists), 2 repeats each ===")
# control
print("  control:")
control_repeats = repeat_measure(control_candidate, control_hash, None, "control", REQUIRED_REPEATS)
control_runs = [control_median] + [r["median_ns"] for r in control_repeats if r["status"] == "MEASURED"]
if len(control_runs) != REQUIRED_REPEATS + 1:
  print("  CONTROL REPEATS INCOMPLETE:", control_repeats)
control_agg = statistics.median(control_runs)
control_mad = statistics.median(abs(v - control_agg) for v in control_runs) / control_agg
stability.append({"label": "control", "candidate_hash": control_hash, "geometry": None,
                   "run_medians_ns": control_runs, "aggregate_median_ns": control_agg,
                   "relative_mad": control_mad, "status": "stable" if control_mad <= MAX_RELATIVE_MAD else "unstable"})

for idx, f in enumerate(finalists):
  print(f"  finalist {idx+1} (tile={f['tile']} geom={f['geometry']}):")
  row = hash_to_row[f["candidate_hash"]]
  # match exact geometry (candidate_hash can be shared across geometries; use this finalist's own)
  reps = repeat_measure(row["candidate"], f["candidate_hash"], f["geometry"], f"finalist{idx+1}", REQUIRED_REPEATS)
  runs = [f["median_ns"]] + [r["median_ns"] for r in reps if r["status"] == "MEASURED"]
  if len(runs) != REQUIRED_REPEATS + 1:
    print("  FINALIST REPEATS INCOMPLETE:", reps)
  agg = statistics.median(runs)
  mad = statistics.median(abs(v - agg) for v in runs) / agg
  stability.append({"label": f"finalist{idx+1}", "candidate_hash": f["candidate_hash"], "geometry": f["geometry"],
                     "tile": f["tile"], "run_medians_ns": runs, "aggregate_median_ns": agg,
                     "relative_mad": mad, "status": "stable" if mad <= MAX_RELATIVE_MAD else "unstable"})

print("\n=== stability summary ===")
for s in stability:
  print(f"  {s['label']:12s} agg_median_ns={s['aggregate_median_ns']:.0f}  relative_mad={s['relative_mad']:.4f}  {s['status']}  runs={s['run_medians_ns']}")

json.dump({"control": {"median_ns": control_median, "gflops": control_gflops, "correct": check["correct"],
                       "reached_precontract": control_reached_precontract},
           "finalists": finalists, "stability": stability},
          open("scratchpad/_phase_c_result.json", "w"), indent=2)

if any(s["status"] != "stable" for s in stability):
  print("\nVERDICT: INCONCLUSIVE_FINALIST_INSTABILITY -- one or more repeats exceeded relative_mad<=0.05")
else:
  by_hash_geom = {s["label"]: s for s in stability}
  ordered = sorted(stability, key=lambda s: (s["aggregate_median_ns"], s["candidate_hash"]))
  best = ordered[0]
  control_s = by_hash_geom["control"]
  improvement = 1 - best["aggregate_median_ns"] / control_s["aggregate_median_ns"]
  machine = best["label"] != "control"
  win = machine and improvement >= MIN_WIN_FRACTION
  print(f"\nbest stable candidate: {best['label']} agg_median_ns={best['aggregate_median_ns']:.0f}")
  print(f"control agg_median_ns={control_s['aggregate_median_ns']:.0f}")
  print(f"improvement_fraction={improvement:.4f}  minimum_win_fraction={MIN_WIN_FRACTION}")
  print(f"VERDICT: {'MACHINE_WINNER' if win else 'REFUTED_NO_MATERIAL_WIN'}")
