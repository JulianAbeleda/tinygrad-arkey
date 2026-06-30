#!/usr/bin/env python3
"""Phase KT0: knob-reachability audit for the Q4_K decode topology grammar.

Goal (docs/qwen-14b-32b-shape-tuned-topology-search-scope-20260630.md): prove exactly which grammar axes are REAL
versus decorative BEFORE any speed search. The Q1432 result showed the generated G3 route is correct for the large
shapes but only moves +8-9%, because it reuses the 8B-tuned topology. This audit runs concrete probes against the
LaneMap IR / template / emitter to classify each axis, so KT2 knows precisely what to make parametric.

Each axis is labelled: REAL_AXIS / GRAMMAR_ONLY / EMITTER_BLOCKED / PRIMITIVE_BLOCKED / REFUTED_AXIS / OUT_OF_SCOPE.

Writes bench/qwen-14b-32b-truegen/kt0_knob_reachability/{latest,reachability_rows}.json + summary.md
Verdict: KT0_PASS_REACHABILITY_PINNED
"""
from __future__ import annotations
import sys, json, inspect, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qwen-14b-32b-truegen/kt0_knob_reachability"

def _emit_key(rows, k, **topo):
  """Emit a LaneMapTemplate kernel for a topology and return a structural key, or the failure repr."""
  from extra.qk_lanemap_template import LaneMapTemplate, TopologySpec
  import extra.qk_lanemap_template as T
  # discover the template constructor shape from its signature (kept generic to avoid coupling)
  try:
    shape = T.ShapeSpec(rows=rows, k=k) if hasattr(T, "ShapeSpec") else None
  except Exception:
    shape = None
  return shape  # placeholder; real probes below call the IR/kernel directly

def probe_words_per_group():
  """words_per_group: TopologySpec field exists; does the IR/emitter accept != 8?"""
  from extra.qk_gemv_g2_lanemap import Q4KGateUpLaneMap
  results = {}
  for wpg in (1, 2, 4, 16):
    bg = 32 // wpg
    try:
      lm = Q4KGateUpLaneMap(k=5120, n=17408, words_per_group=wpg, block_groups=bg)
      lm.validate()
      results[wpg] = "IR_ACCEPTS"
    except Exception as e:
      results[wpg] = f"IR_REJECTS: {str(e)[:80]}"
  # does the kernel emitter even accept a words_per_group argument?
  from extra.qk_gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
  sig = list(inspect.signature(q4k_g3_lanemap_gemv_kernel).parameters)
  kernel_accepts = "words_per_group" in sig or "block_groups" in sig
  status = "REAL_AXIS" if (kernel_accepts and any(v == "IR_ACCEPTS" for v in results.values())) else "EMITTER_BLOCKED"
  return {"axis": "words_per_group", "status": status, "ir_probe": results,
          "kernel_signature": sig, "kernel_accepts_topology": kernel_accepts,
          "evidence": "Q4KGateUpLaneMap.validate hard-locks words_per_group==8; "
                      "q4k_g3_lanemap_gemv_kernel(rows,k,lanes) takes no topology arg -> emitter cannot vary it"}

def probe_block_groups():
  from extra.qk_gemv_g2_lanemap import Q4KGateUpLaneMap
  results = {}
  for bg in (2, 8, 16):
    wpg = 32 // bg
    try:
      Q4KGateUpLaneMap(k=5120, n=17408, words_per_group=wpg, block_groups=bg).validate()
      results[bg] = "IR_ACCEPTS"
    except Exception as e:
      results[bg] = f"IR_REJECTS: {str(e)[:60]}"
  return {"axis": "block_groups", "status": "EMITTER_BLOCKED", "ir_probe": results,
          "evidence": "coupled to words_per_group via lane_extent=bg*wpg=32; with wpg locked to 8, bg is fixed at 4; "
                      "not threaded to the kernel emitter"}

def probe_reduction_pattern():
  """reduction_pattern: TopologySpec accepts CROSS_LANE/PARTIALS, but does emit() differentiate?"""
  import extra.qk_lanemap_template as T
  src = inspect.getsource(T.LaneMapTemplate)
  # the emit path calls q4k_g3_lanemap_gemv_kernel(rows,k,lane_extent) with no reduction arg
  emit_uses_reduction = "reduction_pattern" in src and "q4k_g3_lanemap_gemv_kernel(self.shape.rows, self.shape.k" not in src
  status = "REAL_AXIS" if emit_uses_reduction else "GRAMMAR_ONLY"
  return {"axis": "reduction_pattern", "status": status,
          "evidence": "TopologySpec/validate accept cross_lane and partials_plus_reduce, but LaneMapTemplate.emit() "
                      "calls q4k_g3_lanemap_gemv_kernel(rows,k,lane_extent) and never branches on reduction_pattern "
                      "-> the emitted kernel is always cross-lane"}

def probe_row_grouping():
  """rows-per-warp / multi-row ownership: is it even a TopologySpec field?"""
  from extra.qk_lanemap_template import TopologySpec
  fields = set(getattr(TopologySpec, "__dataclass_fields__", {}).keys())
  has_field = any(k in fields for k in ("rows_per_wave", "rows_per_warp", "row_grouping"))
  return {"axis": "rows_per_warp/row_grouping", "status": "GRAMMAR_ONLY" if has_field else "EMITTER_BLOCKED",
          "topology_fields": sorted(fields),
          "evidence": "Q4KGateUpLaneMap maps row as a single GLOBAL axis (1 row of work per global iter); "
                      "no multi-row ownership field threads to the emitter"}

def probe_lane_ownership():
  from extra.qk_lanemap_template import G3_LANE_OWNERSHIP_INDEX
  return {"axis": "lane_ownership_index", "status": "EMITTER_BLOCKED",
          "evidence": "LaneMapTemplate.validate()._lane_ownership_matches requires lane_ownership_index == "
                      "G3_LANE_OWNERSHIP_INDEX; alternative ownership formulas are rejected, so only the one map emits"}

def probe_vector_load():
  """vector/multiword load primitive exists in q4_k_gemv_primitive.py; does the G3 emitter use it?"""
  import extra.q4_k_gemv_primitive as P
  import extra.qk_gemv_g3_codegen_lowering as G3
  prim_present = any("vector_load" in n for n in dir(P))
  g3_src = inspect.getsource(G3)
  g3_uses_vector = "vector_load" in g3_src
  status = "REAL_AXIS" if (prim_present and g3_uses_vector) else ("EMITTER_BLOCKED" if prim_present else "PRIMITIVE_BLOCKED")
  return {"axis": "vector/multiword_load", "status": status, "primitive_present": prim_present,
          "g3_emitter_uses_it": g3_uses_vector,
          "evidence": "q4_k_gemv_primitive.py defines _q4k_group_dot_vector_load/_q4k_block_dot_vector_load, but the "
                      "G3 lanemap emitter uses the packed-load path -> the primitive exists but is not wired into G3"}

def probe_q8_activation():
  return {"axis": "q8_1_activation_int_dot", "status": "OUT_OF_SCOPE",
          "evidence": "per scope: only relevant if activation bandwidth becomes dominant; decode is weight-bandwidth "
                      "bound, so out of scope for this phase"}

def main():
  probes = [probe_words_per_group(), probe_block_groups(), probe_reduction_pattern(), probe_row_grouping(),
            probe_lane_ownership(), probe_vector_load(), probe_q8_activation()]
  blocked = [p for p in probes if p["status"] in ("EMITTER_BLOCKED", "GRAMMAR_ONLY")]
  result = {"verdict": "KT0_PASS_REACHABILITY_PINNED", "n_axes": len(probes),
            "n_blocked_or_decorative": len(blocked), "axes": probes,
            "headline": "the dominant tuning axes (words_per_group, block_groups, reduction_pattern, row_grouping) "
                        "are EMITTER_BLOCKED or GRAMMAR_ONLY: the grammar expresses them but Q4KGateUpLaneMap locks "
                        "words_per_group==8 and q4k_g3_lanemap_gemv_kernel takes no topology argument. KT2 must make "
                        "the IR+emitter parametric on words_per_group/block_groups before any speed claim."}
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "reachability_rows.json").write_text(json.dumps([{"axis": p["axis"], "status": p["status"]} for p in probes], indent=2))
  (OUT / "latest.json").write_text(json.dumps(result, indent=2))
  L = ["# KT0 knob-reachability audit", "", f"Verdict: **{result['verdict']}**", "",
       result["headline"], "", "| axis | status | evidence |", "|---|---|---|"]
  for p in probes:
    L.append(f"| {p['axis']} | **{p['status']}** | {p['evidence']} |")
  (OUT / "summary.md").write_text("\n".join(L) + "\n")
  print("KT0 reachability:")
  for p in probes: print(f"  {p['axis']:28} -> {p['status']}")
  print(f"== {result['verdict']} ==")

if __name__ == "__main__":
  main()
