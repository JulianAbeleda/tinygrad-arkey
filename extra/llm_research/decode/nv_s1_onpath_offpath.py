#!/usr/bin/env python3
"""S1 on-path vs off-path census from the retained tinygrad control DAG.

For each layer's S1 window (Q anchor end -> O anchor start), compute the
longest logical dependency path between the two anchors and split the
in-window kernel mass into:

  on_path   - nodes on that Q->O dependency spine
  off_path  - nodes whose work is logically independent of the spine and is a
              candidate for overlap

The S1 exposure ceiling after perfect overlap is the on-path window mass.
This is measurement tooling only; it changes nothing.
"""
from __future__ import annotations

import argparse, json, pathlib
from collections import defaultdict

from extra.llm_research.decode.nv_inter_anchor_analysis import (
  dag_deps, intersection_us, strict_path_cost)

SCHEMA = "tinygrad.nv_s1_onpath_offpath.v1"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--control", type=pathlib.Path,
                  default=pathlib.Path("docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json"))
  ap.add_argument("--out", type=pathlib.Path,
                  default=pathlib.Path("docs/task_workflow/output/nv-s1-onpath-offpath-20260822.json"))
  args = ap.parse_args()

  dag = json.loads(args.control.read_text(encoding="utf-8"))
  nodes = dag["nodes"]
  edges = dag["edges"]
  anchors = dag["anchor_ids"]
  deps, _ = dag_deps(nodes, edges, "id", "id")
  weights = [float(n["duration_us"]) for n in nodes]
  children = [[] for _ in nodes]
  for i, preds in enumerate(deps):
    for p in preds:
      children[p].append(i)

  q_ids, o_ids = anchors["Q"], anchors["O"]
  layers = min(len(q_ids), len(o_ids))
  rows = []
  family = defaultdict(lambda: {"on_path_us": 0.0, "off_path_us": 0.0})

  for layer in range(layers):
    q, o = q_ids[layer], o_ids[layer]
    _, path = strict_path_cost(q, o, weights, children)
    spine = set(path) - {q, o}
    window = (nodes[q]["end_us"], nodes[o]["start_us"])
    exposure = window[1] - window[0]
    on_mass = off_mass = 0.0
    for node in nodes:
      mass = intersection_us([(node["start_us"], node["end_us"])], [window])
      if mass <= 1e-9:
        continue
      key = node["family"]
      if node["id"] in spine:
        family[key]["on_path_us"] += mass
        on_mass += mass
      else:
        family[key]["off_path_us"] += mass
        off_mass += mass
    rows.append({
      "layer": layer,
      "q_id": q,
      "o_id": o,
      "spine_node_ids": sorted(spine),
      "exposure_us": round(exposure, 3),
      "on_path_mass_us": round(on_mass, 3),
      "off_path_mass_us": round(off_mass, 3),
      "dead_us": round(exposure - on_mass - off_mass, 3),
    })

  total_exposure = sum(r["exposure_us"] for r in rows)
  total_on = sum(r["on_path_mass_us"] for r in rows)
  total_off = sum(r["off_path_mass_us"] for r in rows)
  payload = {
    "schema": SCHEMA,
    "source_control": str(args.control),
    "layers": layers,
    "summary": {
      "exposure_total_us": round(total_exposure, 3),
      "on_path_mass_total_us": round(total_on, 3),
      "off_path_mass_total_us": round(total_off, 3),
      "dead_total_us": round(total_exposure - total_on - total_off, 3),
      "overlap_ceiling_us": round(total_off, 3),
      "post_overlap_floor_us": round(total_on, 3),
    },
    "per_family_us": {
      k: {
        "on_path_us": round(v["on_path_us"], 3),
        "off_path_us": round(v["off_path_us"], 3),
        "total_us": round(v["on_path_us"] + v["off_path_us"], 3),
      }
      for k, v in sorted(family.items())
    },
    "per_layer": rows,
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

  print(json.dumps({
    "summary": payload["summary"],
    "per_family_us": payload["per_family_us"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
