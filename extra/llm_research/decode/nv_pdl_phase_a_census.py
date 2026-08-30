#!/usr/bin/env python3
"""Phase A static census for the tested tinygrad PDL arm versus llama's chain.

This is read-only measurement tooling.  It consumes the canonical current-HEAD
tinygrad DAG capture (``phase_a_control.json``) and the weighted llama real-edge
DAG and reconstructs, without touching the GPU, exactly what the 2026-08-20
PDL environment could arm:

- the five graph-group boundaries reset the QMD chain;
- native PDL is armed only on a same-queue consecutive exec pair
  (``NVComputeQueue.active_qmd``), so any encoded queue wait breaks the chain;
- the tested env names only GEMV producers (q4k_/q6k_) and support consumers,
  therefore support-to-support and support-to-GEMV coverage is zero by
  construction.

Queue placement follows ``HCQGraph._pick_compute_queue`` with
``HCQ_NV_READY_PLACEMENT=1``.  Wait encoding follows ``_resolve_deps``,
including the NV self-dependency optimization.

The output is a construction census, not an execution measurement.  Phase B
records the actually armed pairs on the device.
"""
from __future__ import annotations

import argparse, json, pathlib, sys
from collections import Counter, defaultdict

PDL_ENV = {
  "producers": ("prefix:q4k_", "prefix:q6k_"),
  "consumers": ("prefix:reduce_output_rmsnorm", "prefix:E_", "prefix:r_",
                "prefix:flash_", "prefix:rmsnorm_q8_1_llama_provider"),
  "trigger_position": "end",
}


def _match(name: str, rules: tuple[str, ...]) -> bool:
  return name in rules or any(rule.startswith("prefix:") and name.startswith(rule.removeprefix("prefix:"))
                              for rule in rules)


def tg_kind(name: str) -> str:
  """Map a rendered tinygrad program name to the same role vocabulary as llama."""
  if name.startswith("q4k_g3_lanemap_gemv_epi_resadd_"):
    return "O"
  if name.startswith("q4k_g3_lanemap_gemv_w1w3fused16_"):
    return "G"
  if name.startswith("q4k_g3_lanemap_gemv_w1w3fused_"):
    return "G"
  if "_epi_ffnresadd_" in name or name.endswith("_epi_ffnresadd"):
    return "D"
  if name == "q4k_warp_coop_q8_dp4a_partial_4096_4096":
    return "Q"
  if name.startswith("q4k_g3_lanemap_gemv_") and name.endswith("_4096_4096"):
    return "Q"
  if name.startswith("q4k_g3_lanemap_gemv_") and name.endswith("_1024_4096"):
    return "K"
  if name.startswith("q6k_v_four_warp_fp16") and name.endswith("_1024_4096"):
    return "V"
  if name.endswith("_1024_4096") and name.startswith(("q4k_", "q6k_")):
    return "K"
  if name.startswith("q6k_gen_coop") and "inkernel" in name:
    return "vocab"
  if name.startswith("reduce_output_rmsnorm"):
    return "norm"
  if name.startswith("E_"):
    return "elementwise"
  if name.startswith("r_"):
    return "reduce"
  if name.startswith("flash_"):
    return "flash"
  if name.startswith("rmsnorm_q8_1_llama_provider"):
    return "quant_provider"
  return "other"


def tg_bucket(kind: str) -> str:
  if kind in ("Q", "O", "G", "D", "K", "V", "vocab"):
    return "gemv"
  if kind in ("norm", "elementwise", "reduce", "flash", "quant_provider"):
    return "support"
  return "other"


def llama_kind(role: str) -> str:
  return role


def llama_bucket(kind: str) -> str:
  if kind in ("Q", "O", "G", "D", "vocab"):
    return "anchor"
  if kind in ("K", "V"):
    return "gemv"
  return "support"


def load_tinygrad(path: pathlib.Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def load_llama(path: pathlib.Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def tinygrad_schedule(dag: dict, num_queues: int) -> dict:
  """Replay HCQGraph placement/wait encoding for each graph group."""
  nodes = dag["nodes"]
  by_group: dict[int, list[dict]] = defaultdict(list)
  for n in nodes:
    by_group[int(n["group_id"])].append(n)
  groups = [by_group[gid] for gid in sorted(by_group)]

  assignments: list[int] = [-1] * len(nodes)
  pairs: list[dict] = []
  breaks: list[dict] = []

  for group in groups:
    ids = {n["id"]: i for i, n in enumerate(group)}
    preds: dict[int, list[int]] = {i: [] for i in range(len(group))}
    for edge in dag.get("edges") or []:
      a, b = ids.get(edge["from"]), ids.get(edge["to"])
      if a is None or b is None:
        continue
      if a not in preds[b]:
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

      # Reproduce _resolve_deps for the NV compute case.  sync_signals are
      # empty for a single NV device; all deps are (queue, producer_j + 1).
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
      if prev is not None and len(opt_deps) == 0 and \
          _match(str(prev["name"]), PDL_ENV["producers"]) and _match(str(node["name"]), PDL_ENV["consumers"]):
        pairs.append({
          "producer_id": prev["id"], "consumer_id": node["id"],
          "producer_name": str(prev["name"]), "consumer_name": str(node["name"]),
          "producer_kind": tg_kind(str(prev["name"])), "consumer_kind": tg_kind(str(node["name"])),
          "group": int(node["group_id"]), "queue": q,
        })

      active_qmd[q] = {"id": node["id"], "name": str(node["name"]), "kind": tg_kind(str(node["name"]))}
      last_j[q] = j
      loads[q] += 1

    breaks.append({"group": sorted(by_group).index(int(group[0]["group_id"])),
                   "queues": {str(q): loads[q] for q in range(num_queues)}})

  # A pair is an actual data dependency if the static DAG contains an edge
  # between the two nodes in either direction (the dependency graph is the
  # only edge authority available to the static census).
  edge_pairs: set[tuple[int, int]] = set()
  for edge in dag.get("edges") or []:
    edge_pairs.add((edge["from"], edge["to"]))
  for pair in pairs:
    pair["has_data_edge"] = (pair["producer_id"], pair["consumer_id"]) in edge_pairs

  # Same-queue consecutive producer->consumer candidates.  In the native path
  # this is identical to the armed set when no wait was encoded; keep it
  # separate so a future wait-semantic change is visible in the census.
  eligible: list[dict] = []
  armed_ids = {(p["producer_id"], p["consumer_id"]) for p in pairs}
  eligible = [dict(p) for p in pairs]

  return {
    "num_queues": num_queues,
    "assignments": [{"id": n["id"], "name": str(n["name"]), "queue": assignments[n["id"]],
                     "group": int(n["group_id"])} for n in nodes],
    "eligible_adjacent_pairs": eligible,
    "armed_pairs": pairs,
    "group_break_counts": breaks,
  }


def llama_chain_census(dag: dict) -> dict:
  nodes = {n["local_id"]: n for n in dag["nodes"]}
  rows = []
  bucket_counter: Counter = Counter()
  kind_counter: Counter = Counter()
  for edge in dag["raw_edges"]:
    producer, consumer = nodes[edge["from"]], nodes[edge["to"]]
    pk, ck = llama_kind(producer["role"]), llama_kind(consumer["role"])
    pb, cb = llama_bucket(pk), llama_bucket(ck)
    bucket_counter[(pb, cb)] += 1
    kind_counter[(pk, ck)] += 1
    rows.append({"from_local": edge["from"], "to_local": edge["to"],
                 "from_role": pk, "to_role": ck,
                 "from_bucket": pb, "to_bucket": cb})
  return {
    "programmatic_edge_count": len(rows),
    "by_bucket": {f"{a}->{b}": v for (a, b), v in sorted(bucket_counter.items())},
    "by_role": {f"{a}->{b}": v for (a, b), v in sorted(kind_counter.items())},
    "rows": rows,
  }


def pair_census(pairs: list[dict], pair_kind) -> dict:
  bucket_counter: Counter = Counter()
  role_counter: Counter = Counter()
  for pair in pairs:
    pk, ck = pair_kind(pair["producer_kind"], pair["consumer_kind"])
    bucket_counter[(pk, ck)] += 1
    role_counter[(pair["producer_kind"], pair["consumer_kind"])] += 1
  return {
    "count": len(pairs),
    "by_bucket": {f"{a}->{b}": v for (a, b), v in sorted(bucket_counter.items())},
    "by_role": {f"{a}->{b}": v for (a, b), v in sorted(role_counter.items())},
    "support_to_support": sum(v for (a, b), v in bucket_counter.items() if a == "support" and b == "support"),
    "support_to_gemv": sum(v for (a, b), v in bucket_counter.items() if a == "support" and b in ("anchor", "gemv")),
    "gemv_to_support": sum(v for (a, b), v in bucket_counter.items() if a in ("anchor", "gemv") and b == "support"),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--control", required=True, type=pathlib.Path)
  ap.add_argument("--llama", required=True, type=pathlib.Path)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  ap.add_argument("--queues", type=int, choices=(1, 2), default=2)
  args = ap.parse_args()

  tg = load_tinygrad(args.control)
  ll = load_llama(args.llama)
  schedule = tinygrad_schedule(tg, args.queues)
  llama_chain = llama_chain_census(ll)
  eligible_census = pair_census(schedule["eligible_adjacent_pairs"],
                                lambda pk, ck: (tg_bucket(pk), tg_bucket(ck)))
  armed_census = pair_census(schedule["armed_pairs"],
                             lambda pk, ck: (tg_bucket(pk), tg_bucket(ck)))

  doc = {
    "schema": "tinygrad.nv_pdl_phase_a_census.v1",
    "commit": "6570abc025514273faa100c66b979e531585a1e1",
    "pdl_env": PDL_ENV,
    "queues": args.queues,
    "tinygrad": {
      "node_count": len(tg["nodes"]),
      "edge_count": len(tg.get("edges") or []),
      "group_sizes": [sum(1 for n in tg["nodes"] if int(n["group_id"]) == gid)
                      for gid in sorted({int(n["group_id"]) for n in tg["nodes"]})],
      "eligible_adjacent": eligible_census,
      "armed": armed_census,
      "assignment_counter": dict(sorted(Counter(str(a["queue"]) for a in schedule["assignments"]).items())),
      "assignments": schedule["assignments"],
      "eligible_pairs": schedule["eligible_adjacent_pairs"],
      "armed_pairs": schedule["armed_pairs"],
      "group_break_counts": schedule["group_break_counts"],
      "note": "arm simulation includes only same-queue consecutive execs with no encoded wait; support-to-support coverage is zero by the tested env definition",
    },
    "llama": {
      "node_count": len(ll["nodes"]),
      "raw_edge_count": len(ll.get("raw_edges") or []),
      "programmatic_chain": llama_chain,
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "queues": args.queues,
    "tinygrad_eligible_adjacent": eligible_census,
    "tinygrad_armed": armed_census,
    "llama_programmatic_by_bucket": llama_chain["by_bucket"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
