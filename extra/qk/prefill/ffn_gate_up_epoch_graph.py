#!/usr/bin/env python3
"""Compose existing S10 evidence into an epoch dependency graph for the 8B gate/up GEMM."""
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

from extra.qk.prefill.dbuf_epoch_lifecycle_checker import DBUFEvent, check_events
from extra.qk.prefill.dbuf_s10_lds_spec_exporter import checker_compatible_events, export_s10_lds_spec
from extra.qk.prefill_schedule_spec import PrefillGEMMScheduleSpec
from extra.qk.wmma_lds_spec import WMMALDSSpec

M, N, K = 512, 12288, 4096


def ffn_gate_up_spec() -> WMMALDSSpec:
  schedule = PrefillGEMMScheduleSpec(
    m=M, n=N, k=K, route_family="lds", tile_m=128, tile_n=128, tile_k=32,
    waves_m=4, waves_n=2, wm=2, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=256,
    dbuf=1, plra=0, plrab=1, pad=16, leanaddr=0, role="ffn_gate_up")
  spec = WMMALDSSpec.from_prefill_schedule(schedule)
  if spec is None: raise RuntimeError("the established ffn_gate_up schedule no longer produces WMMALDSSpec")
  return spec


def build_epoch_graph(*, stage_owner_audit: dict[str, Any] | None = None,
                      lifecycle_trace: dict[str, Any] | None = None) -> dict[str, Any]:
  """Build exact structural edges and fail closed on identities absent from existing evidence."""
  spec = ffn_gate_up_spec()
  exported = export_s10_lds_spec(spec)
  rows = exported["events"]
  checker_rows = checker_compatible_events(rows)
  checker = check_events([DBUFEvent.from_json(row) for row in checker_rows])
  nodes, edges, last_produce, last_consume = [], [], {}, {}
  unresolved: list[dict[str, Any]] = []
  barriers: list[str] = []
  for i, row in enumerate(rows):
    node_id = f"event:{i}"
    node = {"id": node_id, **{k: v for k, v in row.items() if k != "schema"}}
    nodes.append(node)
    if row["op"] == "barrier":
      barriers.append(node_id)
      continue
    key = (row["role"], row["epoch"], row["slot"], row["window"])
    slot_key = (row["role"], row["slot"])
    if row["op"] == "produce":
      prior = last_consume.get(slot_key)
      if prior is not None: edges.append({"from": prior, "to": node_id, "kind": "slot_reuse_after_consume", "identity": "exact"})
      last_produce[key] = node_id
    elif row["op"] == "consume":
      producer = last_produce.get(key)
      if producer is not None:
        edges.append({"from": producer, "to": node_id, "kind": "structural_reaching_definition", "identity": "exact_epoch_slot_byte_window"})
      barrier = next((b for b in reversed(barriers) if int(b.split(":")[1]) < i), None)
      if producer is not None and barrier is not None:
        edges.extend(({"from": producer, "to": barrier, "kind": "synchronizes_before"},
                      {"from": barrier, "to": node_id, "kind": "synchronizes_before"}))
      last_consume[slot_key] = node_id
    if row.get("value_key") is None:
      unresolved.append({"node": node_id, "field": "value_key", "reason": "S10 WMMALDSSpec exporter does not preserve global-tile value identity"})

  stage_summary = None if stage_owner_audit is None else stage_owner_audit.get("summary")
  trace_reaching = None if lifecycle_trace is None else lifecycle_trace.get("lds_reaching_def_map")
  external_gaps = []
  if stage_owner_audit is None:
    external_gaps.append("stage-owner audit not supplied; graph cannot correlate epoch nodes to lowered owner records")
  if lifecycle_trace is None:
    external_gaps.append("lifecycle trace not supplied; graph cannot correlate epoch nodes to final instruction indices")
  return {
    "schema": "ffn-gate-up-epoch-dependency-graph.v1",
    "workload": {"role": "ffn_gate_up", "m": M, "n": N, "k": K, "tile_k": spec.tile_k,
                 "epoch_count": K // spec.tile_k, "active_buffers": spec.dbuf_epoch_primitive.nbuf},
    "ownership": {"classification": spec.ownership_classification(),
                  "epoch_primitive": spec.dbuf_epoch_primitive.to_json()},
    "claims": {
      "structural_reaching_definitions_complete": checker["ok"] and not any(x["field"] != "value_key" for x in unresolved),
      "value_reaching_definitions_complete": not unresolved,
      "lowered_instruction_correlation_complete": trace_reaching is not None and not trace_reaching.get("missing_load_count"),
      "complete": False,
    },
    "identity_loss": {"count": len(unresolved), "records": unresolved, "external_evidence_gaps": external_gaps},
    "dbuf_checker": checker,
    "stage_owner_audit_summary": stage_summary,
    "lowered_reaching_def_summary": None if trace_reaching is None else {k: trace_reaching.get(k) for k in
      ("key_strength", "limitation", "load_count", "covered_load_count", "missing_load_count",
       "wmma_missing_a_count", "wmma_missing_b_count")},
    "nodes": nodes, "edges": edges,
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--stage-owner-audit", type=pathlib.Path)
  ap.add_argument("--lifecycle-trace", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path)
  args = ap.parse_args()
  load = lambda path: None if path is None else json.loads(path.read_text())
  report = build_epoch_graph(stage_owner_audit=load(args.stage_owner_audit), lifecycle_trace=load(args.lifecycle_trace))
  text = json.dumps(report, indent=2) + "\n"
  if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
  else: print(text, end="")
  return 0


if __name__ == "__main__": raise SystemExit(main())
