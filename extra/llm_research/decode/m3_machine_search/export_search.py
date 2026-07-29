#!/usr/bin/env python3
"""M3 CPU-only finite search exporter for Q4_K/Q6_K decode topology plans.

This deliberately emits symbolic topology candidates for generic compiler primitives.  It neither imports
tinygrad nor calls the incumbent G3/Q6 route emitters, so a selected plan cannot
be a relabelled handwritten builder.  The listed primitives have no generic
lowerer or route-bound executor yet; this is not an executable lowering. Its
score is a deterministic structural ordering, not a GPU performance claim.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).parent
REQUEST = HERE / "search_request.json"
EXPORT = HERE / "m3_search_export.json"

def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"))
def ident(prefix, value): return prefix + ":sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()

def plan_q4(waves):
  return {"plan_schema": "tinygrad.generic_topology_plan.v1", "inputs": ["q4k_words:u16", "x:f16"], "output": "out:f32",
    "launch": {"global": ["rows", waves], "local": [32 * waves]},
    "nodes": [{"op":"grid_axis", "name":"row", "extent":"rows"}, {"op":"local_axis", "name":"wave_lane", "extent":32 * waves},
      {"op":"partition", "axis":"k_blocks", "across":"wave_lane", "parts":waves},
      {"op":"q4k_packed_block_dot", "words":"q4k_words", "vector":"x", "accumulator":"f32"},
      {"op":"wave_reduce", "kind":"add", "width":32},
      {"op":"workgroup_reduce", "kind":"add", "inputs":waves, "when":"waves_per_output>1"},
      {"op":"store", "address":"out[row]", "value":"reduced_dot"}], "constraints": ["k % 256 == 0", "rows > 0"]}

def plan_q6(family, knob):
  if family == "coop":
    return {"plan_schema":"tinygrad.generic_topology_plan.v1", "inputs":["q6k_halfs:u16", "x:f16"], "output":"partials:f32",
      "launch":{"global":["ceildiv(rows,row_tile)", 16], "local":[knob, 16]},
      "nodes":[{"op":"grid_axis","name":"row_group","extent":"ceildiv(rows,row_tile)"},{"op":"local_axis","name":"row_in_group","extent":knob},{"op":"local_axis","name":"pos","extent":16},{"op":"reduce_axis","name":"block","extent":"k/256"},{"op":"q6k_packed_block_dot","position":"pos","accumulator":"f32"},{"op":"store","address":"partials[row,pos]","value":"dot"},{"op":"external_reduce","axis":"pos","kind":"add","output":"out[row]"}], "constraints":["rows % row_tile == 0", "k % 256 == 0"]}
  return {"plan_schema":"tinygrad.generic_topology_plan.v1", "inputs":["q6k_halfs:u16", "x:f16"], "output":"partials:f32",
    "launch":{"global":["rows", knob], "local":[1]}, "nodes":[{"op":"grid_axis","name":"row","extent":"rows"},{"op":"grid_axis","name":"k_part","extent":knob},{"op":"reduce_axis","name":"block_in_part","extent":"ceildiv(k/256,parts)"},{"op":"reduce_axis","name":"pos","extent":16},{"op":"q6k_packed_block_dot","position":"pos","accumulator":"f32","predicate":"block < k/256"},{"op":"store","address":"partials[row,k_part]","value":"dot"},{"op":"external_reduce","axis":"k_part","kind":"add","output":"out[row]"}], "constraints":["k % 256 == 0"]}

def ranked(request):
  routes = {}
  q4 = []
  for w in request["routes"]["decode_q4k_g3_generated"]["finite_space"]["waves_per_output"]:
    plan = plan_q4(w); q4.append({"candidate_id":f"m3.q4k.w{w}", "parameters":{"waves_per_output":w}, "structural_score":[w-1, w], "plan":plan})
  q6 = []
  fs = request["routes"]["decode_q6k_coop_generated"]["finite_space"]
  for family in fs["family"]:
    for knob in (fs["coop_row_tile"] if family == "coop" else fs["partial_parts"]):
      plan = plan_q6(family, knob)
      # Fewer output partials/launch coordination wins the pre-measurement structural ordering.
      score = [0 if family == "coop" else 1, (16 if family == "coop" else knob), knob]
      q6.append({"candidate_id":f"m3.q6k.{family}.{knob}", "parameters":({"family":family, "row_tile":knob} if family=="coop" else {"family":family, "parts":knob}), "structural_score":score, "plan":plan})
  for route, population in (("decode_q4k_g3_generated", q4), ("decode_q6k_coop_generated", q6)):
    population.sort(key=lambda x:(x["structural_score"], x["candidate_id"]))
    for rank, row in enumerate(population, 1): row["rank"] = rank; row["plan_id"] = ident("generic-plan", row["plan"])
    routes[route] = {"population_size":len(population), "selected":population[0], "ranked_population":population}
  return routes

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--request", type=Path, default=REQUEST); ap.add_argument("--out", type=Path, default=EXPORT); args=ap.parse_args()
  request=json.loads(args.request.read_text()); routes=ranked(request)
  missing_lowerers=["q4k_packed_block_dot", "q6k_packed_block_dot", "external_reduce"]
  payload={"schema":"tinygrad.decode_machine_search_export.v1", "search_id":request["search_id"], "request_id":ident("search-request", request), "ranking":{"kind":"deterministic_cpu_structural", "performance_claim":"none", "tie_break":"score_then_candidate_id"}, "run":{"executor":"CPython", "device":"none", "deterministic":True}, "routes":routes, "provenance":{"generated_by":"m3 finite enumerator", "runtime_integration":"none", "selected_plans_are":"symbolic topology plan candidates, not executable lowerings"}, "promotion":{"status":"blocked", "reason":"Generic primitive lowerers and a route-bound topology-plan executor do not exist; GPU numerical and timing evidence therefore cannot be collected.", "missing_primitive_lowerers":missing_lowerers, "missing_executor":"route-bound generic topology-plan executor", "blocked_record_command":"PYTHONPATH=. python3 extra/llm_research/decode/m3_machine_search/record_blocked_inputs.py --export extra/llm_research/decode/m3_machine_search/m3_search_export.json --q4k-fixture /path/to/q4k_decode_fixture.json --q6k-fixture /path/to/q6k_decode_fixture.json --device AMD:gfx1100", "required_inputs":["q4k_decode_fixture.json: packed Q4_K words, fp16 vector, reference output and shape", "q6k_decode_fixture.json: packed Q6_K halfwords, fp16 vector, reference output and shape"], "required_hardware":"AMD gfx1100 through a supported tinygrad AMD path (Linux KFD or macOS PCI+AMD)"}}
  args.out.write_text(canonical(payload)+"\n")
if __name__ == "__main__": main()
