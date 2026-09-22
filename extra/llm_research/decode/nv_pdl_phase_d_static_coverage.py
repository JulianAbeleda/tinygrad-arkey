#!/usr/bin/env python3
"""Phase D static coverage census for the broadest name-pinned PDL arm.

This is GPU-free, read-only measurement tooling.  It takes the canonical
current-HEAD tinygrad DAG capture and the weighted llama programmatic-edge DAG
and answers one question: if every unique program name is admitted as both a
producer and a consumer, how much of llama's support chain can the existing
name-pinned split-phase arm express, and what blocks the rest?

The replay follows the same authority as Phase A
(``nv_pdl_phase_a_census.py``):

- ``HCQGraph._pick_compute_queue`` with ``HCQ_NV_READY_PLACEMENT=1``;
- ``_resolve_deps`` wait encoding including the NV self-dependency
  optimization;
- native PDL arming only on a same-queue consecutive ``active_qmd`` pair with
  zero encoded waits and a matching name on both sides;
- graph-group boundaries reset the QMD chain, so no pair crosses groups.

Unlike Phase A, this census records every same-queue consecutive pair, not
just the armed subset, and classifies every static tinygrad edge as armed,
adjacency-blocked, wait-blocked, queue-split, or group-cut.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, subprocess, sys
from collections import Counter, defaultdict

import nv_pdl_phase_a_census as phase_a


def aligned_bucket(kind: str) -> str:
  """Map tinygrad roles onto llama's anchor/gemv/support vocabulary."""
  if kind in ("Q", "O", "G", "D", "vocab"):
    return "anchor"
  if kind in ("K", "V"):
    return "gemv"
  return "support"


def aligned_pair_census(pairs: list[dict], kind_getter) -> dict:
  counter: Counter = Counter()
  for pair in pairs:
    counter[(aligned_bucket(kind_getter(pair["producer_kind"])),
             aligned_bucket(kind_getter(pair["consumer_kind"])))] += 1
  return {
    "count": len(pairs),
    "by_bucket": {f"{a}->{b}": v for (a, b), v in sorted(counter.items())},
    "support_to_support": sum(v for (a, b), v in counter.items() if a == "support" and b == "support"),
    "support_to_anchor": sum(v for (a, b), v in counter.items() if a == "support" and b == "anchor"),
    "anchor_to_support": sum(v for (a, b), v in counter.items() if a == "anchor" and b == "support"),
    "gemv_to_support": sum(v for (a, b), v in counter.items() if a == "gemv" and b == "support"),
    "support_to_gemv": sum(v for (a, b), v in counter.items() if a == "support" and b == "gemv"),
  }


def tinygrad_bucket_pair_census(pairs: list[dict], kind_getter) -> dict:
  counter: Counter = Counter()
  for pair in pairs:
    counter[(phase_a.tg_bucket(kind_getter(pair["producer_kind"])),
             phase_a.tg_bucket(kind_getter(pair["consumer_kind"])))] += 1
  return {
    "count": len(pairs),
    "by_bucket": {f"{a}->{b}": v for (a, b), v in sorted(counter.items())},
  }


def replay(dag: dict, num_queues: int, env: dict) -> tuple[list[int], list[dict]]:
  """Replay placement and wait encoding, retaining every same-queue adjacency."""
  nodes = dag["nodes"]
  by_group: dict[int, list[dict]] = defaultdict(list)
  for n in nodes:
    by_group[int(n["group_id"])].append(n)
  groups = [by_group[gid] for gid in sorted(by_group)]

  assignments: list[int] = [-1] * len(nodes)
  adjacency_rows: list[dict] = []

  for group in groups:
    ids = {n["id"]: i for i, n in enumerate(group)}
    preds: dict[int, list[int]] = {i: [] for i in range(len(group))}
    for edge in dag.get("edges") or []:
      a, b = ids.get(edge["from"]), ids.get(edge["to"])
      if a is not None and b is not None and a not in preds[b]:
        preds[b].append(a)

    last_j = {q: None for q in range(num_queues)}
    loads = {q: 0 for q in range(num_queues)}
    queue_access = {q: defaultdict(lambda: None) for q in range(num_queues)}
    active_qmd: dict[int, dict | None] = {q: None for q in range(num_queues)}

    for j, node in enumerate(group):
      if num_queues == 1:
        q = 0
      else:
        rdeps_peek = sorted(set(preds[j]))
        tail = last_j[0]
        if tail is not None and any(dep == tail for dep in rdeps_peek):
          q = 0
        else:
          q = min(range(num_queues), key=lambda qq: (loads[qq], qq))
      assignments[node["id"]] = q

      rdeps = [(assignments[group[dep]["id"]], dep + 1) for dep in sorted(set(preds[j]))]
      same_prev = [] if last_j[q] is None else [(q, last_j[q] + 1)]
      deps = rdeps + same_prev
      opt_deps: list[tuple[int, int]] = []
      for dep_q, dep_val in sorted(set(deps), key=lambda x: x[1], reverse=True):
        qa = queue_access[q][dep_q]
        if qa is None or qa < dep_val:
          opt_deps.append((dep_q, dep_val))
          queue_access[q][dep_q] = dep_val
      if len(opt_deps) == 1 and opt_deps[0][0] == q:
        opt_deps = []

      prev = active_qmd[q]
      if prev is not None:
        name_match = (phase_a._match(str(prev["name"]), env["producers"])
                      and phase_a._match(str(node["name"]), env["consumers"]))
        adjacency_rows.append({
          "producer_id": prev["id"],
          "consumer_id": node["id"],
          "producer_name": str(prev["name"]),
          "consumer_name": str(node["name"]),
          "producer_kind": phase_a.tg_kind(str(prev["name"])),
          "consumer_kind": phase_a.tg_kind(str(node["name"])),
          "group": int(node["group_id"]),
          "queue": q,
          "wait_encoded": len(opt_deps) > 0,
          "wait_deps": [[int(dq), int(dv)] for dq, dv in opt_deps],
          "name_match": name_match,
          "armed": len(opt_deps) == 0 and name_match,
        })

      active_qmd[q] = {"id": node["id"], "name": str(node["name"])}
      last_j[q] = j
      loads[q] += 1

  edge_forward: set[tuple[int, int]] = set()
  edge_any: set[tuple[int, int]] = set()
  for edge in dag.get("edges") or []:
    edge_forward.add((edge["from"], edge["to"]))
    edge_any.add((edge["from"], edge["to"]))
    edge_any.add((edge["to"], edge["from"]))
  for row in adjacency_rows:
    pair = (row["producer_id"], row["consumer_id"])
    row["has_data_edge"] = pair in edge_forward
    row["has_static_edge_any"] = pair in edge_any

  return assignments, adjacency_rows


def classify_static_edges(dag: dict, assignments: list[int],
                          adjacency_rows: list[dict]) -> dict:
  """Classify every forward static edge against the broad replay."""
  nodes = dag["nodes"]
  group_by_id = {n["id"]: int(n["group_id"]) for n in nodes}
  kind_by_id = {n["id"]: phase_a.tg_kind(str(n["name"])) for n in nodes}
  adjacency_by_pair = {(r["producer_id"], r["consumer_id"]): r for r in adjacency_rows}
  armed_ids = {(r["producer_id"], r["consumer_id"]) for r in adjacency_rows if r["armed"]}

  edges: list[dict] = []
  reason_counter: Counter = Counter()
  armed_kind_counter: Counter = Counter()
  armed_aligned_counter: Counter = Counter()
  raw_kind_counter: Counter = Counter()
  raw_aligned_counter: Counter = Counter()
  missed_by_reason_bucket: dict = defaultdict(Counter)

  for edge in dag.get("edges") or []:
    a, b = edge["from"], edge["to"]
    ka, kb = kind_by_id[a], kind_by_id[b]
    aligned = f"{aligned_bucket(ka)}->{aligned_bucket(kb)}"
    if (a, b) in armed_ids:
      reason = "armed"
    elif group_by_id[a] != group_by_id[b]:
      reason = "cross_group"
    elif assignments[a] != assignments[b]:
      reason = "queue_split"
    else:
      row = adjacency_by_pair.get((a, b))
      if row is None:
        reason = "adjacency"
      elif row["wait_encoded"]:
        reason = "encoded_wait"
      elif not row["name_match"]:
        reason = "name_filter"
      else:
        reason = "unknown"

    reason_counter[reason] += 1
    if reason == "armed":
      armed_kind_counter[(ka, kb)] += 1
      armed_aligned_counter[(aligned_bucket(ka), aligned_bucket(kb))] += 1
    else:
      missed_by_reason_bucket[aligned][reason] += 1
    if edge.get("kind") == "RAW":
      raw_kind_counter[(ka, kb)] += 1
      raw_aligned_counter[(aligned_bucket(ka), aligned_bucket(kb))] += 1
    edges.append({"from": a, "to": b, "kind": edge.get("kind"), "reason": reason})

  return {
    "by_reason": dict(sorted(reason_counter.items())),
    "armed_by_kind": {f"{a}->{b}": v for (a, b), v in sorted(armed_kind_counter.items())},
    "armed_by_aligned_bucket": {f"{a}->{b}": v for (a, b), v in sorted(armed_aligned_counter.items())},
    "raw_by_kind": {f"{a}->{b}": v for (a, b), v in sorted(raw_kind_counter.items())},
    "raw_by_aligned_bucket": {f"{a}->{b}": v for (a, b), v in sorted(raw_aligned_counter.items())},
    "missed_by_reason_bucket": {
      bucket: dict(sorted(reasons.items())) for bucket, reasons in sorted(missed_by_reason_bucket.items())
    },
    "rows": edges,
  }


def broad_census(dag: dict, num_queues: int, env: dict) -> dict:
  assignments, adjacency_rows = replay(dag, num_queues, env)
  armed = [r for r in adjacency_rows if r["armed"]]
  real = [r for r in armed if r["has_data_edge"]]
  any_edge = [r for r in armed if r["has_static_edge_any"]]
  incidental = [r for r in armed if not r["has_data_edge"]]
  wait_rows = [r for r in adjacency_rows if r["wait_encoded"]]
  wait_real = [r for r in wait_rows if r["has_data_edge"]]

  def kind_getter(kind: str) -> str:
    return kind

  return {
    "queues": num_queues,
    "consecutive_pair_count": len(adjacency_rows),
    "armed_pair_count": len(armed),
    "armed_with_forward_static_edge": len(real),
    "armed_with_any_static_edge": len(any_edge),
    "incidental_armed_pair_count": len(incidental),
    "armed_tinygrad_bucket": tinygrad_bucket_pair_census(armed, kind_getter),
    "armed_real_tinygrad_bucket": tinygrad_bucket_pair_census(real, kind_getter),
    "armed_aligned_bucket": aligned_pair_census(armed, kind_getter),
    "armed_real_aligned_bucket": aligned_pair_census(real, kind_getter),
    "incidental_aligned_bucket": aligned_pair_census(incidental, kind_getter),
    "wait_encoded_adjacency_count": len(wait_rows),
    "wait_encoded_with_forward_static_edge": len(wait_real),
    "wait_encoded_aligned_bucket": aligned_pair_census(wait_rows, kind_getter),
    "name_filter_blocked_pair_count": sum(1 for r in adjacency_rows if not r["name_match"]),
    "static_edges": classify_static_edges(dag, assignments, adjacency_rows),
    "assignments": [{"id": n["id"], "name": str(n["name"]), "queue": assignments[n["id"]],
                     "group": int(n["group_id"])} for n in dag["nodes"]],
    "adjacency_rows": adjacency_rows,
  }


def phase_a_control_census(dag: dict, num_queues: int, env: dict) -> dict:
  saved = phase_a.PDL_ENV
  phase_a.PDL_ENV = env
  try:
    schedule = phase_a.tinygrad_schedule(dag, num_queues)
  finally:
    phase_a.PDL_ENV = saved
  census = phase_a.pair_census(
    schedule["armed_pairs"],
    lambda pk, ck: (phase_a.tg_bucket(pk), phase_a.tg_bucket(ck)))
  return {
    "queues": num_queues,
    "armed_pair_count": census["count"],
    "armed_by_tinygrad_bucket": census["by_bucket"],
    "armed_by_role": census["by_role"],
  }


def git_head() -> str:
  return subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
  ).stdout.strip()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--control", required=True, type=pathlib.Path)
  ap.add_argument("--llama", required=True, type=pathlib.Path)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  ap.add_argument("--queues", default="1,2")
  args = ap.parse_args()

  queue_modes = [int(v) for v in args.queues.split(",")]
  if any(v not in (1, 2) for v in queue_modes):
    raise SystemExit("--queues must be a comma list of 1 and 2")

  tg = phase_a.load_tinygrad(args.control)
  ll = phase_a.load_llama(args.llama)
  llama_chain = phase_a.llama_chain_census(ll)

  unique_names = tuple(sorted({str(n["name"]) for n in tg["nodes"]}))
  broad_env = {
    "producers": unique_names,
    "consumers": unique_names,
    "trigger_position": "end",
  }
  names_blob = "\n".join(unique_names).encode("utf-8")

  control = {}
  broad = {}
  for q in queue_modes:
    control[q] = phase_a_control_census(tg, q, phase_a.PDL_ENV)
    broad[q] = broad_census(tg, q, broad_env)
    expected = control[q]["armed_pair_count"]
    # Phase A pairs are keyed by producer/consumer; replay the control env
    # through this script's same machinery for an exact set comparison.
    saved = phase_a.PDL_ENV
    try:
      phase_a_schedule = phase_a.tinygrad_schedule(tg, q)
    finally:
      phase_a.PDL_ENV = saved
    phase_a_pairs = {(p["producer_id"], p["consumer_id"])
                     for p in phase_a_schedule["armed_pairs"]}
    _, control_rows = replay(tg, q, phase_a.PDL_ENV)
    replay_pairs = {(r["producer_id"], r["consumer_id"])
                    for r in control_rows if r["armed"]}
    if replay_pairs != phase_a_pairs:
      raise SystemExit(f"control replay mismatch on {q} queues: "
                       f"{len(replay_pairs)} vs {len(phase_a_pairs)} pairs")
    if expected != len(phase_a_pairs):
      raise SystemExit(f"Phase A control count mismatch on {q} queues: "
                       f"{expected} vs {len(phase_a_pairs)}")

  reconciliation = {}
  for q in queue_modes:
    armed_real = broad[q]["armed_real_aligned_bucket"]["by_bucket"]
    rows = []
    for bucket, llama_count in llama_chain["by_bucket"].items():
      tiny_count = armed_real.get(bucket, 0)
      rows.append({
        "bucket": bucket,
        "llama_programmatic": llama_count,
        "broad_armed_with_forward_static_edge": tiny_count,
        "delta": llama_count - tiny_count,
      })
    reconciliation[q] = rows

  doc = {
    "schema": "tinygrad.nv_pdl_phase_d_static_coverage.v1",
    "commit": git_head(),
    "generated_utc": subprocess.run(
      ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], check=True, capture_output=True, text=True
    ).stdout.strip(),
    "inputs": {
      "control": str(args.control),
      "llama": str(args.llama),
      "queues": queue_modes,
    },
    "method": {
      "placement": "HCQGraph._pick_compute_queue replay with HCQ_NV_READY_PLACEMENT=1",
      "wait_encoding": "_resolve_deps replay including NV self-dependency optimization",
      "arm_rule": "same-queue consecutive active_qmd pair, zero encoded waits, name match on both sides",
      "broad_env": "all unique program names admitted as both producers and consumers",
      "broad_env_name_count": len(unique_names),
      "broad_env_name_sha256": hashlib.sha256(names_blob).hexdigest(),
      "trigger_position": "end",
      "static_edge_authority": "tinygrad capture edges; forward edge means producer->consumer data/schedule dependency",
      "bucket_mapping": {
        "anchor": ["Q", "O", "G", "D", "vocab"],
        "gemv": ["K", "V"],
        "support": ["norm", "elementwise", "reduce", "flash", "quant_provider", "other"],
      },
    },
    "control_reproduction": {str(q): control[q] for q in queue_modes},
    "broad_census": {str(q): broad[q] for q in queue_modes},
    "llama": {
      "node_count": len(ll["nodes"]),
      "raw_edge_count": len(ll.get("raw_edges") or []),
      "programmatic_chain": llama_chain,
    },
    "reconciliation": {str(q): reconciliation[q] for q in queue_modes},
  }

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(json.dumps({
    "commit": doc["commit"],
    "control_reproduction": {str(q): control[q]["armed_pair_count"] for q in queue_modes},
    "broad": {
      str(q): {
        "consecutive": broad[q]["consecutive_pair_count"],
        "armed": broad[q]["armed_pair_count"],
        "armed_with_forward_static_edge": broad[q]["armed_with_forward_static_edge"],
        "real_by_aligned_bucket": broad[q]["armed_real_aligned_bucket"]["by_bucket"],
        "static_edge_reasons": broad[q]["static_edges"]["by_reason"],
      } for q in queue_modes
    },
    "llama_programmatic_by_bucket": llama_chain["by_bucket"],
    "out": str(args.out),
  }, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
