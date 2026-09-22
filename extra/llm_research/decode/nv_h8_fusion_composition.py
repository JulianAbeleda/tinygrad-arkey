#!/usr/bin/env python3
"""H8 fusion-only-sufficiency recomposition at HEAD (2026-08-21).

Recompute the weighted critical path of the canonical current-HEAD decode DAG
with each remaining legal fusion applied, and with combinations, so
alternate-path takeover is included.  This is the H8 test from
docs/task_workflow/input/nv-split-phase-pdl-causal-design-review-scope-20260820.md:
the currently legal residual and reduction folds plus Q4 FFN-down must close
the locked gap only after the path is recomputed, never by adding raw
zero-cost ceilings.

Input: docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json
Locked checks: original critical path 4249.216 us, S1 gap 634.334 us,
token wall gap 717.505 us.
"""
from __future__ import annotations

import argparse
import json
import pathlib


TOPO = pathlib.Path("docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json")
LOCKED_CP_US = 4249.216
LOCKED_S1_US = 634.334
LOCKED_WALL_US = 717.505
LLAMA_Q4_DOWN_US = 19.232


def load_dag(path: pathlib.Path) -> tuple[list[dict], list[dict]]:
  doc = json.loads(path.read_text())
  return doc["nodes"], doc["edges"]


def critical_path(nodes: list[dict], edges: list[dict], durations: list[float]) -> float:
  """Longest node-weight path.  Weights are the scenario durations."""
  indeg = [0] * len(nodes)
  adj: list[list[int]] = [[] for _ in nodes]
  for edge in edges:
    adj[edge["from"]].append(edge["to"])
    indeg[edge["to"]] += 1
  # Kahn with the deterministic first-ready order; all 1230 edges participate,
  # matching the locked ledger's logical dependency path.
  ready = sorted((i for i, d in enumerate(indeg) if d == 0), reverse=True)
  best = [0.0] * len(nodes)
  order: list[int] = []
  while ready:
    node = ready.pop()
    order.append(node)
    for nxt in adj[node]:
      best[nxt] = max(best[nxt], best[node] + durations[node])
      indeg[nxt] -= 1
      if indeg[nxt] == 0:
        ready.append(nxt)
        ready.sort(reverse=True)
  if len(order) != len(nodes):
    raise RuntimeError(f"DAG cycle or unreachable node: reached {len(order)}/{len(nodes)}")
  sinks = [i for i in range(len(nodes)) if not adj[i]]
  return max(best[s] + durations[s] for s in sinks)


def scenario(nodes: list[dict], edges: list[dict], zero_families: frozenset[str],
             q4_down_delta_us: float = 0.0) -> tuple[float, float]:
  durations = []
  for node in nodes:
    duration = float(node["duration_us"])
    if node["family"] in zero_families:
      duration = 0.0
    if q4_down_delta_us and node.get("anchor") == "down" and node.get("name", "").startswith(
        ("q4k_fp16_mmvq_direct", "q4k_g3_lanemap_gemv")):
      sem = (node.get("metadata") or {}).get("semantic") or [{}]
      if any(entry.get("source_quant_storage") == "Q4_K" for entry in sem if isinstance(entry, dict)):
        duration = max(0.0, duration - q4_down_delta_us)
    durations.append(duration)
  cp = critical_path(nodes, edges, durations)
  return cp, LOCKED_CP_US - cp


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--topo", default=str(TOPO))
  ap.add_argument("--out", default="docs/task_workflow/output/nv-h8-fusion-composition-20260821.json")
  args = ap.parse_args()
  nodes, edges = load_dag(pathlib.Path(args.topo))

  base, _ = scenario(nodes, edges, frozenset())
  if abs(base - LOCKED_CP_US) > 0.01:
    raise RuntimeError(f"base critical path {base:.3f} does not reproduce the locked {LOCKED_CP_US}; DAG weights differ")

  residual, residual_delta = scenario(nodes, edges, frozenset({"residual"}))
  reduce, reduce_delta = scenario(nodes, edges, frozenset({"reduce"}))
  composed, composed_delta = scenario(nodes, edges, frozenset({"residual", "reduce"}))
  vocab_tail, vocab_delta = scenario(nodes, edges, frozenset({"vocab"}))
  max_fusion, max_fusion_delta = scenario(nodes, edges, frozenset({"residual", "reduce", "vocab"}))

  q4_count = sum(
    1 for node in nodes
    if node.get("anchor") == "down" and node["name"].startswith("q4k_fp16_mmvq_direct")
    and any((e.get("source_quant_storage") == "Q4_K") for e in (node.get("metadata") or {}).get("semantic") or [{}]))
  q4_mean = sum(node["duration_us"] for node in nodes
    if node.get("anchor") == "down" and node["name"].startswith("q4k_fp16_mmvq_direct")
    and any((e.get("source_quant_storage") == "Q4_K") for e in (node.get("metadata") or {}).get("semantic") or [{}])) / q4_count
  per_node_delta = q4_mean - LLAMA_Q4_DOWN_US
  q4_floor, q4_delta = scenario(nodes, edges, frozenset(), q4_down_delta_us=per_node_delta)
  full, full_delta = scenario(nodes, edges, frozenset({"residual", "reduce", "vocab"}),
                              q4_down_delta_us=per_node_delta)

  doc = {
    "schema": "tinygrad.nv_h8_fusion_composition.v1",
    "date": "2026-08-21",
    "source": str(args.topo),
    "locked": {"critical_path_us": LOCKED_CP_US, "s1_gap_us": LOCKED_S1_US, "wall_gap_us": LOCKED_WALL_US},
    "base_critical_path_us": round(base, 3),
    "q4_down": {
      "q4_blocks": q4_count,
      "mean_us": round(q4_mean, 3),
      "llama_floor_us": LLAMA_Q4_DOWN_US,
      "per_node_floor_delta_us": round(per_node_delta, 3),
    },
    "scenarios": {
      "residual_only": {"cp_us": round(residual, 3), "ceiling_us": round(residual_delta, 3)},
      "reduce_only": {"cp_us": round(reduce, 3), "ceiling_us": round(reduce_delta, 3)},
      "residual_plus_reduce": {"cp_us": round(composed, 3), "ceiling_us": round(composed_delta, 3)},
      "vocab_tail_only": {"cp_us": round(vocab_tail, 3), "ceiling_us": round(vocab_delta, 3)},
      "max_fusion_residual_reduce_vocab": {"cp_us": round(max_fusion, 3), "ceiling_us": round(max_fusion_delta, 3)},
      "q4_down_at_llama_floor": {"cp_us": round(q4_floor, 3), "ceiling_us": round(q4_delta, 3)},
      "fusion_plus_q4_floor": {"cp_us": round(full, 3), "ceiling_us": round(full_delta, 3)},
    },
    "accounting": {
      "additive_raw_ceiling_us": round(residual_delta + reduce_delta + vocab_delta + q4_delta, 3),
      "recomputed_composition_us": round(full_delta, 3),
      "takeover_and_interference_us": round(residual_delta + reduce_delta + vocab_delta + q4_delta - full_delta, 3),
      "wall_gap_remaining_after_zero_cost_composition_us": round(LOCKED_WALL_US - full_delta, 3),
      "s1_gap_remaining_after_zero_cost_composition_us": round(LOCKED_S1_US - full_delta, 3),
    },
    "note": ("Zero-cost ceilings only. LLAMA_Q4_DOWN_US uses the corrected 19.232 us/node "
             "floor (nv-gemv-core-deficit-correction-20260813.md; 11.776 us is attention-O). "
             "Measured wall conversion is far lower for every row "
             "(see nv-reduce-output-ffn-residual-bind-outcome-20260813.md, nv-vocab-top1-fusion-head-recheck-20260816.md, "
             "nv-q4-down-dp4a-resadd-18block-gate-20260814.md), so this is the upper envelope, not a bookable recovery."),
  }
  pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  pathlib.Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
  print(json.dumps(doc, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
