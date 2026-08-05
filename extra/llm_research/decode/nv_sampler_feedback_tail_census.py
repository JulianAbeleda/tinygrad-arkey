#!/usr/bin/env python3
"""Fail-closed census for the native d512 greedy sampler/feedback tail.

This is deliberately a topology classifier, not a performance estimator.  It
only recognizes the qualified 948-node Qwen3-8B decode capture and reports
the serial post-LM-head reduction chain that a future included-cost argmax
primitive must replace as one whole unit.
"""
from __future__ import annotations

import argparse, json, pathlib

SCHEMA = "tinygrad.nv_decode.sampler_feedback_tail_census.v1"
_TAIL_PREFIXES = ("E_1187_32_4", "r_32_4_1187", "r_128_16_8_1187", "r_16_8")
_FEEDBACK_PREFIXES = ("E", "E_2")

def _clean(name:str) -> str:
  # Program hashes are intentionally excluded from the contract.
  return name.rsplit("_", 1)[0] if "_" in name else name

def census(payload:dict) -> dict:
  nodes = payload.get("nodes")
  edges = payload.get("edges")
  if not isinstance(nodes, list) or len(nodes) != 948: raise ValueError("expected qualified 948-node native decode DAG")
  if not isinstance(edges, list): raise ValueError("DAG has no edge list")
  lm_head, tail = nodes[943], nodes[944:948]
  if not str(lm_head.get("name", "")).startswith("q6k_gen_coop_151936_4096_inkernel"):
    raise ValueError("qualified Q6_K d512 LM-head missing at node 943")
  got = tuple(_clean(str(node.get("name", ""))) for node in tail)
  if got != _TAIL_PREFIXES: raise ValueError(f"unexpected sampler-tail template: {got!r}")
  feedback = tuple(_clean(str(nodes[i].get("name", ""))) for i in range(2))
  if feedback != _FEEDBACK_PREFIXES: raise ValueError(f"unexpected feedback prefix: {feedback!r}")
  arcs = {(int(edge["from"]), int(edge["to"])) for edge in edges if "from" in edge and "to" in edge}
  required = {(943, 944), (944, 945), (944, 946), (945, 946), (946, 947)}
  if not required <= arcs: raise ValueError(f"sampler reduction dependencies changed: missing {sorted(required-arcs)!r}")
  durations = [float(node["duration_us"]) for node in tail]
  return {"schema":SCHEMA, "status":"PASS", "qualified_node_count":len(nodes),
    "feedback": {"node_indexes":[0,1], "program_prefixes":list(feedback),
                 "device_us":sum(float(nodes[i]["duration_us"]) for i in range(2)),
                 "host_feedback_required":"sampled.item() supplies the public streaming token; next decode already receives device-resident out"},
    "sampler": {"lm_head_index":943, "tail_indexes":[944,945,946,947], "program_prefixes":list(got),
                "tail_device_us":sum(durations), "durations_us":durations,
                "serial_dependencies":[list(x) for x in sorted(required)],
                "candidate_boundary":"replace the entire four-program greedy argmax chain only; do not remove item() or the alias-safe feedback copy without a separate contract"}}

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dag", type=pathlib.Path, required=True)
  parser.add_argument("--out", type=pathlib.Path)
  args = parser.parse_args()
  result=census(json.loads(args.dag.read_text()))
  text=json.dumps(result, indent=2, sort_keys=True)+"\n"
  if args.out: args.out.write_text(text)
  print(text, end="")
  return 0

if __name__ == "__main__": raise SystemExit(main())
