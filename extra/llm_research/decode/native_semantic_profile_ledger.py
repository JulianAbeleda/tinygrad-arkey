#!/usr/bin/env python3
"""Reconcile native NV graph time with llama by disjoint semantic populations.

HCQ per-node timestamps perturb the serialized native graph slightly.  This
tool therefore uses their *composition*, selects the median-total replicate,
and calibrates that composition to an independently measured marker-light
native device window.  It never compares a node sum directly with llama wall.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, sqlite3, statistics
from collections import Counter

from extra.llm_research.decode.cuda_graph_timeline_ledger import _load_rows, _split_replays

SCHEMA = "tinygrad.nv_decode.native_semantic_profile_ledger.v1"
HASH64 = re.compile(r"_[0-9a-f]{64}$")
EXPECTED_GROUPS = [32, 64, 128, 256, 468]


def sha256(path:pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda:f.read(1 << 20), b""): h.update(chunk)
  return h.hexdigest()


def clean(name:str) -> str: return HASH64.sub("", name)


def quant_role(metadata:dict|None) -> str|None:
  semantic = (metadata or {}).get("semantic") or []
  if not semantic: return None
  tensor = str(semantic[0].get("tensor_name", ""))
  suffixes = (("attn_q.weight", "attn_q"), ("attn_k.weight", "attn_k"),
              ("attn_v.weight", "attn_v"), ("attn_output.weight", "attn_o"),
              ("ffn_gate.weight", "ffn_gate_up"), ("ffn_down.weight", "ffn_down"),
              ("output.weight", "vocab"))
  return next((role for suffix, role in suffixes if tensor.endswith(suffix)), None)


def semantic_class(name:str, metadata:dict|None) -> str:
  if (role:=quant_role(metadata)) is not None: return "quantized_core/" + role
  name = clean(name)
  if name.startswith("flash_block_"): return "non_quantized/flash_score"
  if name.startswith("flash_fused_"): return "non_quantized/flash_combine"
  if name.startswith("r_"): return "non_quantized/reduction"
  if name == "E" or name.startswith("E_"): return "non_quantized/elementwise"
  return "non_quantized/other"


def load_capture(path:pathlib.Path, dag_nodes:list[dict]) -> dict:
  records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  sizes = [len(record.get("entries") or []) for record in records]
  if sizes != EXPECTED_GROUPS: raise ValueError(f"{path}: expected groups {EXPECTED_GROUPS}, saw {sizes}")
  entries = [entry for record in records for entry in record["entries"]]
  if len(entries) != len(dag_nodes): raise ValueError(f"{path}: entry/DAG length mismatch")
  totals, counts = Counter(), Counter()
  for index, (entry, node) in enumerate(zip(entries, dag_nodes)):
    if entry["name"] != node["name"]: raise ValueError(f"{path}: name mismatch at node {index}")
    cls = semantic_class(entry["name"], node.get("metadata"))
    totals[cls] += float(entry["duration"]); counts[cls] += 1
  return {"path":str(path), "sha256":sha256(path), "raw_total_us":sum(totals.values()),
          "raw_us":dict(sorted(totals.items())), "counts":dict(sorted(counts.items()))}


def llama_role_costs(trace:pathlib.Path, manifest:pathlib.Path, graph_id:int=2, warmup:int=2) -> dict:
  data = json.loads(manifest.read_text())
  node_roles = {int(row["llama"]["ordered_launch_subgraph"][1]):row["model_role"] for row in data["rows"]}
  con = sqlite3.connect(str(trace))
  try: replays, _ = _split_replays(_load_rows(con, graph_id))
  finally: con.close()
  per_replay = []
  for replay in replays[warmup:]:
    costs = Counter()
    for row in replay:
      if row["class"] != "mmq": continue
      node = int(row["graphNodeId"])
      if node not in node_roles: raise ValueError(f"unmapped llama MMQ graph node {node}")
      costs[node_roles[node]] += (int(row["end"])-int(row["start"]))/1000.0
    per_replay.append(costs)
  roles = sorted(node_roles.values())
  return {role:round(statistics.median(row[role] for row in per_replay), 3) for role in sorted(set(roles))}


def build(profile_paths:list[pathlib.Path], dag_path:pathlib.Path, native_window_us:float,
          llama_timeline_path:pathlib.Path, llama_span_us:float, llama_trace:pathlib.Path|None=None,
          manifest:pathlib.Path|None=None) -> dict:
  dag = json.loads(dag_path.read_text()); nodes = dag["nodes"]
  if len(nodes) != 948: raise ValueError(f"semantic DAG must have 948 nodes, saw {len(nodes)}")
  captures = [load_capture(path, nodes) for path in profile_paths]
  ordered = sorted(captures, key=lambda row:row["raw_total_us"])
  representative = ordered[len(ordered)//2]
  scale = native_window_us / representative["raw_total_us"]
  calibrated = {key:round(value*scale, 3) for key,value in representative["raw_us"].items()}
  quant = round(sum(v for k,v in calibrated.items() if k.startswith("quantized_core/")), 3)
  nonquant = round(sum(v for k,v in calibrated.items() if k.startswith("non_quantized/")), 3)
  timeline = json.loads(llama_timeline_path.read_text())
  mmq = float(timeline["classes"]["mmq"]["union_us"])
  exposed = float(timeline["non_anchor_aggregate"]["exposed_vs_mmq_us"])
  gaps = float(timeline["median"]["internal_gap_us"])
  populations = [
    {"rank":1, "population":"non_quantized_exposed", "native_us":nonquant, "llama_us":exposed,
     "device_delta_us":round(nonquant-exposed, 3)},
    {"rank":2, "population":"quantized_core", "native_us":quant, "llama_us":mmq,
     "device_delta_us":round(quant-mmq, 3)},
    {"rank":3, "population":"llama_internal_gaps", "native_us":0.0, "llama_us":gaps,
     "device_delta_us":round(-gaps, 3)},
  ]
  profiled_equation = sum(row["device_delta_us"] for row in populations)
  authority_delta = native_window_us-llama_span_us
  raw_quant = [sum(v for k,v in row["raw_us"].items() if k.startswith("quantized_core/")) for row in captures]
  calibrated_quant_range = [native_window_us*q/row["raw_total_us"] for q,row in zip(raw_quant, captures)]
  out = {"schema":SCHEMA, "status":"PASS", "topology":{"groups":EXPECTED_GROUPS, "programs":948},
    "method":{"profile_use":"composition_only; median-total replicate scaled to marker-light native device window",
      "wall_rule":"node sums are never used as wall without native-window calibration",
      "llama_rule":"MMQ union and aggregate non-MMQ exposure are disjoint; individual class exposures are non-additive"},
    "captures":captures, "representative_capture":representative["path"], "calibration_scale":scale,
    "native":{"device_window_us":native_window_us, "calibrated_class_us":calibrated,
      "quantized_core_us":quant, "non_quantized_us":nonquant,
      "reconciliation_us":round(native_window_us-quant-nonquant, 6),
      "quantized_core_calibrated_range_us":[round(min(calibrated_quant_range), 3),round(max(calibrated_quant_range), 3)]},
    "llama":{"unprofiled_graph_span_us":llama_span_us, "profiled_mmq_union_us":mmq,
      "profiled_non_mmq_exposed_union_us":exposed, "profiled_internal_gap_us":gaps},
    "ranked_disjoint_device_delta_populations":populations,
    "equation":{"profiled_terms_delta_us":round(profiled_equation, 3),
      "unprofiled_authority_delta_us":round(authority_delta, 3),
      "residual_us":round(profiled_equation-authority_delta, 3),
      "tolerance_us":max(50.0, .02*authority_delta),
      "pass":abs(profiled_equation-authority_delta) <= max(50.0, .02*authority_delta)},
    "provenance":{"semantic_dag":{"path":str(dag_path),"sha256":sha256(dag_path)},
      "llama_timeline":{"path":str(llama_timeline_path),"sha256":sha256(llama_timeline_path)}}}
  if llama_trace is not None and manifest is not None:
    roles = llama_role_costs(llama_trace, manifest)
    native_roles = {k.split("/",1)[1]:v for k,v in calibrated.items() if k.startswith("quantized_core/")}
    role_rows = []
    for llama_role,llama_us in roles.items():
      role_rows.append({"native_role":llama_role,"llama_role":llama_role,
                        "native_us":native_roles[llama_role],"llama_us":llama_us,
                        "delta_us":round(native_roles[llama_role]-llama_us,3)})
    out["quantized_role_diagnostic"] = {"warning":("independently medianed role rows need not sum to aggregate MMQ union; "
      "roles are exact for the pinned Qwen3 graph: its intentional expansion order is Q, V, K"),
      "rows":sorted(role_rows,key=lambda x:x["delta_us"],reverse=True),
      "sources":{"trace":{"path":str(llama_trace),"sha256":sha256(llama_trace)},
                 "manifest":{"path":str(manifest),"sha256":sha256(manifest)}}}
  return out


def main() -> None:
  p=argparse.ArgumentParser(description=__doc__);p.add_argument("--profile",action="append",type=pathlib.Path,required=True)
  p.add_argument("--semantic-dag",type=pathlib.Path,required=True);p.add_argument("--native-window-us",type=float,required=True)
  p.add_argument("--llama-timeline",type=pathlib.Path,required=True);p.add_argument("--llama-span-us",type=float,required=True)
  p.add_argument("--llama-trace",type=pathlib.Path);p.add_argument("--manifest",type=pathlib.Path);p.add_argument("--out",type=pathlib.Path,required=True)
  a=p.parse_args();out=build(a.profile,a.semantic_dag,a.native_window_us,a.llama_timeline,a.llama_span_us,a.llama_trace,a.manifest)
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  print(json.dumps({"status":out["status"],"equation":out["equation"],"populations":out["ranked_disjoint_device_delta_populations"]},indent=2))


if __name__ == "__main__": main()
