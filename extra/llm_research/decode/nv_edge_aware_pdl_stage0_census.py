#!/usr/bin/env python3
"""Stage 0 typed RAW-edge census prediction for the edge-aware PDL hook.

This is GPU-free, read-only measurement tooling.  It consumes the canonical
current-HEAD tinygrad DAG capture and predicts what an edge-aware split-phase
arm can express under the rules the native path actually implements today:

- the five graph-group boundaries reset the QMD chain;
- an armed pair must be the same-queue consecutive ``active_qmd`` pair;
- any encoded queue wait clears ``active_qmd`` and therefore blocks arming;
- only RAW forward edges are launch candidates, WAR/WAW stay full-completion.

Two facts are deliberately NOT assumed in this census:

1. The public QMD header has one ``WAIT_ON_LATCH_ID`` field and no documented
   multiple-wait capability.  A consumer with more than one RAW producer is
   classified ``multi_producer_fallback`` until a Stage 2 device probe proves
   same-latch aggregation or a safe merge construction.
2. ``phase_a_control.json`` records each edge kind but not the overlapping
   buffer spans.  Every RAW edge is therefore marked ``span-unverified``; no
   edge is called alias-safe from this capture.

The script reconciles its replay against Phase A (108/144 armed pairs) and
Phase D (329/429 broad-name armed real edges) before writing anything.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import nv_pdl_phase_a_census as phase_a  # noqa: E402
import nv_pdl_phase_d_static_coverage as phase_d  # noqa: E402


REASON_ORDER = ("cross_group", "queue_split", "adjacency", "encoded_wait",
                "multi_producer_fallback", "candidate_armed")


def git_head() -> str:
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
  ).stdout.strip()


def node_index(nodes: list[dict]) -> dict[int, dict]:
  return {int(n["id"]): n for n in nodes}


def raw_producer_map(edges: list[dict]) -> dict[int, list[int]]:
  producers: dict[int, list[int]] = collections.defaultdict(list)
  for edge in edges:
    if edge.get("kind") == "RAW":
      producers[int(edge["to"])].append(int(edge["from"]))
  return {consumer: sorted(set(ids)) for consumer, ids in producers.items()}


def bucket_pair(producer_kind: str, consumer_kind: str) -> str:
  return f"{phase_a.tg_bucket(producer_kind)}->{phase_a.tg_bucket(consumer_kind)}"


def reason_counter_buckets(rows: list[dict], reason: str) -> dict[str, int]:
  counter: collections.Counter = collections.Counter()
  for row in rows:
    if row["reason"] == reason:
      counter[bucket_pair(row["producer_kind"], row["consumer_kind"])] += 1
  return dict(sorted(counter.items()))


def multi_producer_geometry(row: dict, producers: list[int],
                            assignments: dict[int, dict]) -> dict:
  """Describe the other RAW producers without endorsing transitive coverage.

  Same-queue-earlier is a *candidate* transitive ordering only; it is not
  evidence that a latch on the immediate predecessor safely covers the
  earlier producer once PDL overlap is introduced.  The verdict stays
  multi_producer_fallback regardless.
  """
  to_id, other = row["from"], [p for p in producers if p != row["from"]]
  target = assignments[to_id]
  same_earlier = []
  cross_queue = []
  other_case = []
  for producer in other:
    entry = assignments.get(producer)
    if entry is None:
      other_case.append(producer)
    elif entry["queue"] == target["queue"] and entry["queue_pos"] < target["queue_pos"]:
      same_earlier.append(producer)
    elif entry["queue"] != target["queue"]:
      cross_queue.append(producer)
    else:
      other_case.append(producer)
  return {
    "same_queue_earlier": same_earlier,
    "cross_queue": cross_queue,
    "other": other_case,
  }


def classify_raw_edges(dag: dict, broad: dict, producers_by_consumer: dict[int, list[int]]) -> dict:
  nodes = node_index(dag["nodes"])
  static_rows = {int(row["from"]): {} for row in broad["static_edges"]["rows"]}
  for row in broad["static_edges"]["rows"]:
    static_rows[int(row["from"])][int(row["to"])] = row
  assignments = {int(entry["id"]): {**entry, "queue_pos": i}
                 for i, entry in enumerate(broad["assignments"])}

  rows: list[dict] = []
  for row in broad["static_edges"]["rows"]:
    if row.get("kind") != "RAW":
      continue
    producer, consumer = int(row["from"]), int(row["to"])
    producer_node, consumer_node = nodes[producer], nodes[consumer]
    structural = row["reason"]
    producer_ids = producers_by_consumer.get(consumer, [])
    if structural == "armed":
      if len(producer_ids) > 1:
        reason = "multi_producer_fallback"
      else:
        reason = "candidate_armed"
    else:
      reason = structural
    rows.append({
      "from": producer,
      "to": consumer,
      "producer_name": str(producer_node["name"]),
      "consumer_name": str(consumer_node["name"]),
      "producer_family": str(producer_node.get("family", "unknown")),
      "consumer_family": str(consumer_node.get("family", "unknown")),
      "producer_kind": phase_a.tg_kind(str(producer_node["name"])),
      "consumer_kind": phase_a.tg_kind(str(consumer_node["name"])),
      "bucket": bucket_pair(phase_a.tg_kind(str(producer_node["name"])),
                            phase_a.tg_kind(str(consumer_node["name"]))),
      "group": int(consumer_node["group_id"]),
      "from_queue": assignments[producer]["queue"],
      "to_queue": assignments[consumer]["queue"],
      "structural_reason": structural,
      "raw_producer_count": len(producer_ids),
      "other_raw_producers": [p for p in producer_ids if p != producer],
      "other_raw_producer_geometry": multi_producer_geometry(
        {"from": producer, "to": consumer}, producer_ids, assignments),
      "reason": reason,
      "alias_span_status": "unverified",
    })

  by_reason: collections.Counter = collections.Counter(r["reason"] for r in rows)
  unknown = set(by_reason) - set(REASON_ORDER)
  if unknown:
    raise SystemExit(f"unclassified RAW rows: {sorted(unknown)}")
  if by_reason.total() != sum(1 for e in dag.get("edges") or [] if e.get("kind") == "RAW"):
    raise SystemExit("RAW row count mismatch against the input DAG")

  by_bucket: collections.Counter = collections.Counter(r["bucket"] for r in rows)
  by_consumer_family: collections.Counter = collections.Counter(r["consumer_family"] for r in rows)
  by_producer_family: collections.Counter = collections.Counter(r["producer_family"] for r in rows)
  multi_geometry: collections.Counter = collections.Counter()
  for row in rows:
    if row["reason"] == "multi_producer_fallback":
      geom = row["other_raw_producer_geometry"]
      if geom["cross_queue"]:
        multi_geometry["has_cross_queue_other_producer"] += 1
      elif geom["same_queue_earlier"]:
        multi_geometry["all_other_producers_same_queue_earlier"] += 1
      else:
        multi_geometry["other_geometry"] += 1

  return {
    "raw_total": by_reason.total(),
    "raw_total_from_input": sum(1 for e in dag.get("edges") or [] if e.get("kind") == "RAW"),
    "by_reason": {reason: by_reason.get(reason, 0) for reason in REASON_ORDER},
    "candidate_armed_total": by_reason["candidate_armed"],
    "candidate_armed_ignoring_multi_producer": by_reason["candidate_armed"]
        + by_reason["multi_producer_fallback"],
    "span_unverified_total": sum(1 for r in rows if r["alias_span_status"] == "unverified"),
    "alias_safe_total": 0,
    "by_bucket": dict(sorted(by_bucket.items())),
    "by_consumer_family": dict(sorted(by_consumer_family.items())),
    "by_producer_family": dict(sorted(by_producer_family.items())),
    "candidate_by_bucket": reason_counter_buckets(rows, "candidate_armed"),
    "multi_producer_by_bucket": reason_counter_buckets(rows, "multi_producer_fallback"),
    "multi_producer_geometry": dict(sorted(multi_geometry.items())),
    "rows": rows,
  }


def reconcile_phase_a(dag: dict, queues: tuple[int, ...]) -> dict[str, dict]:
  result = {}
  for q in queues:
    control = phase_d.phase_a_control_census(dag, q, phase_a.PDL_ENV)
    result[str(q)] = {
      "queues": q,
      "armed_pair_count": control["armed_pair_count"],
      "armed_by_tinygrad_bucket": control["armed_by_tinygrad_bucket"],
      "armed_by_role": control["armed_by_role"],
    }
  return result


def reconcile_phase_d(dag: dict, artifact: dict, broad: dict[int, dict]) -> dict[str, dict]:
  result = {}
  for q, census in broad.items():
    expected = artifact["broad_census"][str(q)]
    checks = {
      "armed_pair_count": census["armed_pair_count"] == expected["armed_pair_count"],
      "armed_with_forward_static_edge": census["armed_with_forward_static_edge"]
          == expected["armed_with_forward_static_edge"],
      "static_edge_reasons": census["static_edges"]["by_reason"]
          == expected["static_edges"]["by_reason"],
    }
    if not all(checks.values()):
      raise SystemExit(f"Phase D replay mismatch on {q} queues: {checks}")
    result[str(q)] = {
      "queues": q,
      "consecutive_pair_count": census["consecutive_pair_count"],
      "armed_pair_count": census["armed_pair_count"],
      "armed_with_forward_static_edge": census["armed_with_forward_static_edge"],
      "static_edge_reasons": census["static_edges"]["by_reason"],
      "replayed_matches_phase_d_artifact": True,
    }
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--control", required=True, type=pathlib.Path)
  ap.add_argument("--llama", required=True, type=pathlib.Path)
  ap.add_argument("--phase-d", required=True, type=pathlib.Path)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  ap.add_argument("--queues", default="1,2")
  args = ap.parse_args()

  queue_modes = tuple(int(v) for v in args.queues.split(","))
  if any(v not in (1, 2) for v in queue_modes):
    raise SystemExit("--queues must be a comma list of 1 and 2")

  dag = phase_a.load_tinygrad(args.control)
  llama = phase_a.load_llama(args.llama)
  phase_d_artifact = json.loads(args.phase_d.read_text(encoding="utf-8"))

  unique_names = tuple(sorted({str(n["name"]) for n in dag["nodes"]}))
  broad_env = {"producers": unique_names, "consumers": unique_names, "trigger_position": "end"}
  producers_by_consumer = raw_producer_map(dag.get("edges") or [])

  broad = {q: phase_d.broad_census(dag, q, broad_env) for q in queue_modes}
  phase_a_reconcile = reconcile_phase_a(dag, queue_modes)
  phase_d_reconcile = reconcile_phase_d(dag, phase_d_artifact, broad)
  censuses = {str(q): classify_raw_edges(dag, broad[q], producers_by_consumer)
              for q in queue_modes}

  doc = {
    "schema": "tinygrad.nv_edge_aware_pdl_stage0_census.v1",
    "commit": git_head(),
    "generated_utc": subprocess.run(
      ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], check=True, capture_output=True, text=True
    ).stdout.strip(),
    "inputs": {
      "control": str(args.control),
      "llama": str(args.llama),
      "phase_d": str(args.phase_d),
      "queues": list(queue_modes),
    },
    "method": {
      "placement": "HCQGraph._pick_compute_queue replay with HCQ_NV_READY_PLACEMENT=1",
      "wait_encoding": "_resolve_deps replay including NV self-dependency optimization",
      "arm_rule": "same-group, same-queue consecutive active_qmd pair, zero encoded waits, RAW forward static edge, no name filter",
      "classification_precedence": list(REASON_ORDER),
      "multi_producer_policy": "one WAIT_ON_LATCH_ID per consumer; consumers with more than one RAW producer are multi_producer_fallback until Stage 2 proves same-latch aggregation or a safe merge",
      "alias_policy": "edge kind is observed by the capture, but overlapping spans were not retained; every RAW edge is span-unverified and no edge is called alias-safe",
      "edge_kind_authority": "phase_a_control.json edges produced by full_token_dag_capture RecordingDepsTracker",
    },
    "reconciliation": {
      "phase_a_control": phase_a_reconcile,
      "phase_d_broad": phase_d_reconcile,
      "phase_a_expected_counts": {"1": 108, "2": 144},
      "phase_d_expected_armed_real_counts": {"1": 329, "2": 429},
    },
    "llama": {
      "node_count": len(llama["nodes"]),
      "raw_edge_count": len(llama.get("raw_edges") or []),
      "programmatic_edge_count": len(llama.get("raw_edges") or []),
    },
    "raw_producer_degree_distribution": {
      str(degree): count for degree, count in sorted(
        collections.Counter(len(v) for v in producers_by_consumer.values()).items())
    },
    "censuses": censuses,
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(json.dumps({
    "commit": doc["commit"],
    "raw_total": next(iter(censuses.values()))["raw_total"],
    "phase_a_reproduction": {str(q): phase_a_reconcile[str(q)]["armed_pair_count"]
                             for q in queue_modes},
    "phase_d_reproduction": {str(q): phase_d_reconcile[str(q)]["armed_with_forward_static_edge"]
                             for q in queue_modes},
    "census": {
      str(q): {
        "candidate_armed": censuses[str(q)]["candidate_armed_total"],
        "candidate_ignoring_multi": censuses[str(q)]["candidate_armed_ignoring_multi_producer"],
        "by_reason": censuses[str(q)]["by_reason"],
        "alias_safe": censuses[str(q)]["alias_safe_total"],
      } for q in queue_modes
    },
    "out": str(args.out),
  }, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
