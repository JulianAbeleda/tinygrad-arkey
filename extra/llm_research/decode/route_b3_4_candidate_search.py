#!/usr/bin/env python3
"""Route B3.4: offline selective-unalias candidate search (CPU-only).

Loads a B3.1 aligned capture report (tinygrad.route_b3.dag_attribution.v1,
as written by route_b3_dag_attribution.py) and predicts, purely offline from
the report, which exact logical buffers would remove which PLANNER_ALIAS
edges if held out of the shared planner arena, plus the added memory cost.
It emits a Pareto frontier of held-buffer candidate sets with predicted
edge-removal counts (by kind RAW/WAR/WAW), a logical-vs-physical
critical-path delta proxy, and a recommended set.

Removal rule (frozen-edge model, report-only):
  A PLANNER_ALIAS edge is predicted removed when its alias range
  [range[0], range[1]) on its physical arena is contained in the manifest
  placement (arena, offset, offset+aligned_nbytes) of at least one held
  buffer. SEMANTIC edges always stay. The model is deliberately coarse for
  large enclosing placements and is only a prediction; exact re-planning is
  deferred to B3.5 with the planner context.

Memory cost:
  sum over held buffers of round_up(logical_nbytes, 256), the same 256-byte
  block rounding the planner applies in tinygrad/schedule/memory.py.

Critical-path proxy:
  the anchored capture carries duration_us == 0 everywhere, so duration-
  weighted critical paths are not available offline. This tool uses an
  edge-count longest-path proxy: cp_delta_proxy = longest path length in
  edges of the remaining physical DAG minus the logical DAG's longest path.
  Duration-weighted values come later from CUPTI attach.

Search strategy is bounded (no powersets): all useful single-buffer
exclusions, top-256 by planner-edge bytes (the cap discipline from
route_b3_dag_attribution.py), critical-path-chain edge-cover candidates,
edge/byte-ratio greedy, a minimum-byte candidate reaching the 5% theoretical
recovery target, and the empty baseline.

Usage:
  PYTHONPATH=/home/ubuntu/tinygrad-arkey \
    .venv/bin/python extra/llm_research/decode/route_b3_4_candidate_search.py \
    --capture docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json \
    --out docs/task_workflow/output/nv-decode-overlap-b3-4-candidate-ledger-20260804.json
"""
from __future__ import annotations

import argparse, collections, json, sys
from typing import Any

import numpy as np

from extra.llm_research.decode.route_b3_dag_attribution import (
  build_attribution_fixture, build_partial_overlap_fixture, compute_attribution_report,
)

SCHEMA = "tinygrad.route_b3_4.candidate_search.v1"
ALIGNMENT = 256
KINDS = ("RAW", "WAR", "WAW")
SEMANTIC = "SEMANTIC"
PLANNER_ALIAS = "PLANNER_ALIAS"
UNKNOWN = "UNKNOWN"

DEFAULT_CONFIG = {
  "singles": True,
  "top_n_by_bytes": 256,
  "chain": True,
  "chain_full_cover": True,
  "greedy": True,
  "greedy_step_cap": 96,
  "min_five_pct": True,
  "baseline": True,
  "cp_mode": "exact",  # "exact" edge-count longest path, or "chain" proxy
  "top_candidates_n": 256,
}


class B3CandidateSearchError(ValueError):
  pass


def align_up(nbytes: int, alignment: int = ALIGNMENT) -> int:
  """Same 256-byte block rounding the planner uses (memory.py block_size)."""
  return (nbytes + alignment - 1) // alignment * alignment


def load_capture(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


# ---------------------------------------------------------------------------
# Indexing: planner edges grouped by alias range, manifest placements
# ---------------------------------------------------------------------------

def _index_manifest(manifest: dict) -> dict[str, list[tuple[int, int, str]]]:
  """Map arena label -> placements (offset, offset+aligned_nbytes, buf_id)."""
  by_arena: dict[str, list[tuple[int, int, str]]] = {}
  for bid, m in manifest.items():
    off = int(m["offset"])
    by_arena.setdefault(m["arena"], []).append((off, off + int(m["aligned_nbytes"]), bid))
  return by_arena


def _group_planner_edges(planner_edges: list[dict]) -> tuple[list[dict], list[int]]:
  """Group planner edges by (arena, alias range); return groups and per-edge group index."""
  groups: list[dict] = []
  index: dict[tuple[str, tuple[int, int]], int] = {}
  edge_group: list[int] = []
  for i, e in enumerate(planner_edges):
    rng = (e["range"][0], e["range"][1])
    key = (e["arena"], rng)
    gi = index.get(key)
    if gi is None:
      gi = len(groups)
      index[key] = gi
      groups.append({"arena": e["arena"], "range": list(rng), "edge_indices": [],
                     "by_kind": collections.Counter(), "cross_group": 0, "edge_bytes": 0})
    g = groups[gi]
    g["edge_indices"].append(i)
    g["by_kind"][e["kind"]] += 1
    g["cross_group"] += 1 if e.get("crosses_group") else 0
    g["edge_bytes"] += rng[1] - rng[0]
    edge_group.append(gi)
  return groups, edge_group


def _group_covering_buffers(group: dict, by_arena: dict[str, list[tuple[int, int, str]]]) -> list[str] | None:
  """Buffers whose placement contains the alias range; None when unattributable."""
  st, en = group["range"]
  placements = by_arena.get(group["arena"])
  if placements is None:
    if len(by_arena) == 1:
      placements = next(iter(by_arena.values()))
    else:
      return None
  return sorted(bid for off, end, bid in placements if off <= st and en <= end)


# ---------------------------------------------------------------------------
# Edge-count longest-path (topological order by node index; from < to verified)
# ---------------------------------------------------------------------------

def _incoming(n: int, edges: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
  froms: list[list[int]] = [[] for _ in range(n)]
  eidx: list[list[int]] = [[] for _ in range(n)]
  for i, e in enumerate(edges):
    f, t = int(e["from"]), int(e["to"])
    if 0 <= t < n and 0 <= f < n:
      froms[t].append(f)
      eidx[t].append(i)
  return ([np.array(x, dtype=np.int64) for x in froms],
          [np.array(x, dtype=np.int64) for x in eidx])


def _edge_count_cp(n: int, incoming_from: list[np.ndarray], incoming_eidx: list[np.ndarray],
                   removed: np.ndarray | None) -> int:
  """Longest path in edges; removed[i] marks edge i deleted from the walk."""
  dp = np.zeros(n, dtype=np.int64)
  if removed is None:
    for t in range(1, n):
      f = incoming_from[t]
      if f.size:
        dp[t] = int(dp[f].max()) + 1
  else:
    for t in range(1, n):
      f = incoming_from[t]
      if f.size == 0:
        continue
      m = removed[incoming_eidx[t]]
      if bool(m.all()):
        continue
      dp[t] = int(dp[f[~m]].max()) + 1
  return int(dp.max())


def _longest_chain(n: int, incoming_from: list[np.ndarray], incoming_eidx: list[np.ndarray]) -> list[int]:
  """One max-length edge-count path as an ordered node list (deterministic)."""
  dp = np.zeros(n, dtype=np.int64)
  pred = np.full(n, -1, dtype=np.int64)
  for t in range(1, n):
    f = incoming_from[t]
    if f.size == 0:
      continue
    vals = dp[f]
    j = int(vals.argmax())
    dp[t] = int(vals[j]) + 1
    pred[t] = int(f[j])
  chain: list[int] = []
  cur = int(dp.argmax())
  while cur != -1:
    chain.append(cur)
    cur = int(pred[cur])
  chain.reverse()
  return chain


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def _greedy_set_cover(buf_groups: dict[str, list[int]], buf_cost: dict[str, int],
                      group_edges: dict[int, int], cap: int) -> list[set[str]]:
  """Greedy by newly-covered-edge/byte ratio. Returns the candidate set after each step."""
  covered: set[int] = set()
  chosen: list[str] = []
  out: list[set[str]] = []
  for _ in range(cap):
    best_buf, best_gain, best_ratio = None, 0, 0.0
    for bid, gs in buf_groups.items():
      if bid in chosen:
        continue
      gain = sum(group_edges[g] for g in gs if g not in covered)
      if gain <= 0:
        continue
      cost = max(1, buf_cost.get(bid, 0))
      ratio = gain / cost
      if best_buf is None or ratio > best_ratio or (ratio == best_ratio and gain > best_gain):
        best_buf, best_gain, best_ratio = bid, gain, ratio
    if best_buf is None:
      break
    chosen.append(best_buf)
    for g in buf_groups[best_buf]:
      covered.add(g)
    out.append(set(chosen))
  return out


def _chain_cover_steps(chain_group_edges: dict[int, int], buf_groups: dict[str, list[int]],
                       buf_cost: dict[str, int], cap: int) -> tuple[list[set[str]], int]:
  """Greedy cover of chain planner edges by buffer; returns per-step sets and total covered."""
  covered: set[int] = set()
  chosen: list[str] = []
  out: list[set[str]] = []
  for _ in range(cap):
    best_buf, best_gain, best_ratio = None, 0, 0.0
    for bid, gs in buf_groups.items():
      if bid in chosen:
        continue
      gain = sum(chain_group_edges.get(g, 0) for g in gs if g not in covered)
      if gain <= 0:
        continue
      ratio = gain / max(1, buf_cost.get(bid, 0))
      if best_buf is None or ratio > best_ratio or (ratio == best_ratio and gain > best_gain):
        best_buf, best_gain, best_ratio = bid, gain, ratio
    if best_buf is None:
      break
    chosen.append(best_buf)
    for g in buf_groups[best_buf]:
      covered.add(g)
    out.append(set(chosen))
  return out, sum(chain_group_edges.get(g, 0) for g in covered)


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def search_candidates(report: dict, config: dict | None = None,
                      include_edge_pairs: bool = False) -> dict:
  cfg = dict(DEFAULT_CONFIG)
  if config:
    cfg.update(config)
  if cfg["cp_mode"] not in ("exact", "chain"):
    raise B3CandidateSearchError("cp_mode must be 'exact' or 'chain', got %r" % cfg["cp_mode"])

  manifest = report.get("manifest") or {}
  attributed = report.get("attributed_edges") or []
  planner_edges = [e for e in attributed if e.get("source") == PLANNER_ALIAS]
  semantic_edges = [e for e in attributed if e.get("source") == SEMANTIC]
  logical_edges = (report.get("arms") or {}).get("logical", {}).get("edges") or []
  nodes = (report.get("arms") or {}).get("physical", {}).get("nodes") \
    or (report.get("arms") or {}).get("logical", {}).get("nodes") or []
  n = max((int(nd["id"]) for nd in nodes), default=-1) + 1
  group_of = {int(nd["id"]): nd.get("group_id") for nd in nodes}

  # Index planner edges by alias range and manifest placements by arena.
  by_arena = _index_manifest(manifest)
  groups, edge_group = _group_planner_edges(planner_edges)
  covering: list[list[str] | None] = []
  unattributed = 0
  for g in groups:
    cov = _group_covering_buffers(g, by_arena)
    if cov is None:
      unattributed += 1
    covering.append(cov)

  buf_groups: dict[str, list[int]] = {}
  buf_stats: dict[str, dict[str, Any]] = {}
  for gi, cov in enumerate(covering):
    if not cov:
      continue
    for bid in cov:
      buf_groups.setdefault(bid, []).append(gi)
  for bid, gs in buf_groups.items():
    logical = int(manifest[bid]["logical_nbytes"]) if bid in manifest else 0
    aligned = int(manifest[bid]["aligned_nbytes"]) if bid in manifest else align_up(logical)
    buf_stats[bid] = {
      "groups": gs,
      "edges": sum(len(groups[g]["edge_indices"]) for g in gs),
      "edge_bytes": sum(groups[g]["edge_bytes"] for g in gs),
      "cost": aligned if aligned == align_up(logical) else align_up(logical),
    }
  buf_cost = {bid: s["cost"] for bid, s in buf_stats.items()}
  group_edges = {gi: len(groups[gi]["edge_indices"]) for gi in range(len(groups))}

  # Edge-count longest-path proxies for both arms.
  in_from_log, _ = _incoming(n, logical_edges)
  logical_cp = _edge_count_cp(n, in_from_log, [], None)
  full_edges = semantic_edges + planner_edges
  in_from_full, in_eidx_full = _incoming(n, full_edges)
  physical_cp = _edge_count_cp(n, in_from_full, in_eidx_full, None)
  chain = _longest_chain(n, in_from_full, in_eidx_full)
  planner_pair_to_idx = {(e["from"], e["to"]): i for i, e in enumerate(planner_edges)}
  chain_planner_idx = [planner_pair_to_idx[(a, b)] for a, b in zip(chain, chain[1:])
                       if (a, b) in planner_pair_to_idx]
  chain_by_group: dict[int, int] = {}
  for pi in chain_planner_idx:
    chain_by_group[edge_group[pi]] = chain_by_group.get(edge_group[pi], 0) + 1
  planner_delta_edges = max(0, physical_cp - logical_cp)
  five_pct_target = int(np.ceil(0.05 * planner_delta_edges))

  # Candidate sets (frozenset of held buffer ids -> strategy tags).
  candidates: dict[frozenset, list[str]] = {}

  def add(held: set[str], strategy: str) -> None:
    key = frozenset(held)
    if key not in candidates:
      candidates[key] = []
    if strategy not in candidates[key]:
      candidates[key].append(strategy)

  if cfg["baseline"]:
    add(set(), "baseline")
  if cfg["singles"]:
    for bid in sorted(buf_stats):
      if buf_stats[bid]["edges"] > 0:
        add({bid}, "singles")
  if cfg["top_n_by_bytes"]:
    ranked = sorted(buf_stats.items(), key=lambda kv: (kv[1]["edge_bytes"], kv[0]), reverse=True)
    for bid, _ in ranked[:int(cfg["top_n_by_bytes"])]:
      add({bid}, "top_bytes")
  if cfg["chain"]:
    for pi in chain_planner_idx:
      cov = covering[edge_group[pi]]
      if not cov:
        continue
      cheapest = min(cov, key=lambda bid: buf_cost.get(bid, 0))
      add({cheapest}, "chain")
  if cfg["greedy"]:
    for step_set in _greedy_set_cover(buf_groups, buf_cost, group_edges, int(cfg["greedy_step_cap"])):
      add(step_set, "greedy")
  if cfg["chain_full_cover"] or cfg["min_five_pct"]:
    chain_steps, chain_covered = _chain_cover_steps(chain_by_group, buf_groups, buf_cost,
                                                    int(cfg["greedy_step_cap"]))
    if cfg["chain_full_cover"] and chain_steps:
      add(chain_steps[-1], "chain_full_cover")
    if cfg["min_five_pct"]:
      reached = False
      picked = None
      for step_set in chain_steps:
        if not reached:
          picked = step_set
        covered_chain = sum(chain_by_group.get(g, 0)
                            for g in set().union(*(buf_groups[b] for b in step_set)))
        if covered_chain >= five_pct_target:
          reached = True
          break
      if picked is not None:
        add(picked, "min_five_pct")
      five_pct_reached = reached
  else:
    five_pct_reached = False

  # Compute a row per candidate set.
  s = len(semantic_edges)
  group_full_pos = [np.array([s + i for i in groups[gi]["edge_indices"]], dtype=np.int64)
                    for gi in range(len(groups))]
  rows: list[dict] = []
  for held_key in sorted(candidates, key=lambda k: (len(k), sorted(k))):
    held = sorted(held_key)
    covered_groups: set[int] = set()
    for bid in held:
      covered_groups.update(buf_groups.get(bid, []))
    total = 0
    by_kind: dict[str, int] = {k: 0 for k in KINDS}
    cross = 0
    chain_removed = 0
    for gi in covered_groups:
      g = groups[gi]
      total += len(g["edge_indices"])
      for k in KINDS:
        by_kind[k] += g["by_kind"].get(k, 0)
      cross += g["cross_group"]
      chain_removed += chain_by_group.get(gi, 0)
    added_bytes = sum(buf_stats[b]["cost"] for b in held)
    if cfg["cp_mode"] == "exact":
      removed = np.zeros(len(full_edges), dtype=bool)
      if covered_groups:
        removed[np.concatenate([group_full_pos[gi] for gi in sorted(covered_groups)])] = True
      remaining_cp = _edge_count_cp(n, in_from_full, in_eidx_full, removed)
      cp_delta = float(remaining_cp - logical_cp)
    else:
      cp_delta = float(max(0, planner_delta_edges - chain_removed))
    row: dict[str, Any] = {
      "held_buffer_ids": held,
      "added_memory_bytes": added_bytes,
      "removed_planner_edges_total": total,
      "removed_by_kind": by_kind,
      "removed_cross_group": cross,
      "removed_chain_planner_edges": chain_removed,
      "cp_delta_proxy": round(cp_delta, 3),
      "strategies": sorted(candidates[held_key]),
    }
    if include_edge_pairs:
      pairs = set()
      for gi in covered_groups:
        for i in groups[gi]["edge_indices"]:
          pairs.add((planner_edges[i]["from"], planner_edges[i]["to"]))
      row["removed_edge_pairs"] = [[f, t] for f, t in sorted(pairs)]
    rows.append(row)

  rows.sort(key=lambda r: (r["added_memory_bytes"], -r["removed_planner_edges_total"],
                           len(r["held_buffer_ids"]), r["held_buffer_ids"]))

  # Pareto frontier: minimize added bytes, maximize removed planner edges.
  frontier: list[int] = []
  best = -1
  for i, r in enumerate(rows):
    if r["removed_planner_edges_total"] > best:
      best = r["removed_planner_edges_total"]
      frontier.append(i)
  for i, r in enumerate(rows):
    r["pareto_rank"] = frontier.index(i) if i in frontier else None

  top_n = int(cfg["top_candidates_n"])
  top_candidates = sorted((r for r in rows if r["removed_planner_edges_total"] > 0),
                          key=lambda r: (-r["removed_planner_edges_total"],
                                         r["added_memory_bytes"], len(r["held_buffer_ids"])))
  top_candidates = [{k: v for k, v in r.items()} for r in top_candidates[:top_n]]

  # Recommended: min-byte 5% threshold candidate, else best density on the frontier.
  recommended = None
  if cfg["min_five_pct"]:
    five_rows = [r for r in rows if "min_five_pct" in r["strategies"]]
    if five_rows:
      r = five_rows[0]
      reached = sum(chain_by_group.get(g, 0) for g in set().union(
        *(buf_groups[b] for b in r["held_buffer_ids"]))) >= five_pct_target
      recommended = {k: r[k] for k in ("held_buffer_ids", "added_memory_bytes",
                                       "removed_planner_edges_total", "removed_by_kind",
                                       "removed_cross_group", "cp_delta_proxy")}
      recommended["rationale"] = ("minimum-bytes candidate reaching the %d%% theoretical "
                                  "recovery target of %d chain planner edges" % (5, five_pct_target)
                                  if reached else "chain cover exhausted before the 5%% target (%d chain planner edges total)" % len(chain_planner_idx))
      recommended["reached_five_pct"] = reached
  if recommended is None:
    dense = [rows[i] for i in frontier if rows[i]["removed_planner_edges_total"] > 0]
    if dense:
      r = max(dense, key=lambda x: (x["removed_planner_edges_total"] / max(1, x["added_memory_bytes"]),
                                    x["removed_planner_edges_total"]))
      recommended = {k: r[k] for k in ("held_buffer_ids", "added_memory_bytes",
                                       "removed_planner_edges_total", "removed_by_kind",
                                       "removed_cross_group", "cp_delta_proxy")}
      recommended["rationale"] = "best edge-removal density on the Pareto frontier"
      recommended["reached_five_pct"] = False

  per_group: dict[str, dict[str, int]] = {}
  for e in planner_edges:
    gid = group_of.get(e["to"])
    key = str(gid)
    row_g = per_group.setdefault(key, {"planner_edges": 0, "planner_cross_group_edges": 0})
    row_g["planner_edges"] += 1
    row_g["planner_cross_group_edges"] += 1 if e.get("crosses_group") else 0

  return {
    "schema": SCHEMA,
    "source": {
      "report_schema": report.get("schema"),
      "node_count": n,
      "semantic_edge_count": len(semantic_edges),
      "planner_edge_count": len(planner_edges),
      "manifest_buffer_count": len(manifest),
      "useful_buffer_count": len(buf_stats),
      "unattributed_alias_ranges": unattributed,
    },
    "model": {
      "removal_rule": ("PLANNER_ALIAS edge removed iff its alias range is contained in the "
                       "manifest placement of a held buffer; SEMANTIC edges stay"),
      "memory_rounding_bytes": ALIGNMENT,
      "memory_rounding_note": "round_up(logical_nbytes, 256), mirroring tinygrad/schedule/memory.py block_size",
      "cp_proxy_note": ("durations are 0 in the anchored capture; critical-path values are "
                        "edge-count longest-path proxies; duration-weighted values come later "
                        "from CUPTI attach"),
      "coarse_grain_note": ("containment attribution can over-predict removal for large "
                            "enclosing placements; exact re-planning is B3.5 work"),
    },
    "proxies": {
      "logical_cp_edges": logical_cp,
      "physical_cp_edges": physical_cp,
      "planner_delta_cp_edges": planner_delta_edges,
      "physical_chain_length": len(chain) - 1 if chain else 0,
      "chain_planner_edge_count": len(chain_planner_idx),
    },
    "search": {
      "strategies_enabled": sorted(k for k in ("singles", "top_n_by_bytes", "chain",
                                               "chain_full_cover", "greedy", "min_five_pct",
                                               "baseline") if cfg.get(k)),
      "top_n_by_bytes": int(cfg["top_n_by_bytes"]),
      "greedy_step_cap": int(cfg["greedy_step_cap"]),
      "five_pct_target_chain_edges": five_pct_target,
      "five_pct_reached": five_pct_reached,
      "cp_mode": cfg["cp_mode"],
    },
    "per_group": per_group,
    "rows": rows,
    "top_candidates": top_candidates,
    "pareto_frontier": frontier,
    "recommended": recommended,
    "stats": {
      "row_count": len(rows),
      "frontier_count": len(frontier),
      "edge_pairs_included": include_edge_pairs,
    },
  }


def _print_summary(ledger: dict) -> None:
  s = ledger["search"]
  p = ledger["proxies"]
  rec = ledger["recommended"]
  print("schema:", ledger["schema"])
  print("planner edges:", ledger["source"]["planner_edge_count"],
        "semantic:", ledger["source"]["semantic_edge_count"])
  print("cp proxy (edges): logical", p["logical_cp_edges"], "physical", p["physical_cp_edges"],
        "delta", p["planner_delta_cp_edges"])
  print("search:", s["strategies_enabled"], "cp_mode", s["cp_mode"])
  print("rows:", ledger["stats"]["row_count"], "frontier:", ledger["stats"]["frontier_count"])
  if rec:
    print("recommended held:", rec["held_buffer_ids"])
    print("recommended memory bytes:", rec["added_memory_bytes"],
          "removed planner edges:", rec["removed_planner_edges_total"],
          "by kind:", rec["removed_by_kind"],
          "cross group:", rec["removed_cross_group"],
          "cp delta proxy:", rec["cp_delta_proxy"])
    print("rationale:", rec["rationale"])
  else:
    print("recommended: None")


def main() -> int:
  ap = argparse.ArgumentParser(description="Route B3.4 offline selective-unalias candidate search")
  ap.add_argument("--capture", default=None, help="B3.1 aligned capture report JSON")
  ap.add_argument("--synthetic", action="store_true", help="run on the hermetic attribution fixture")
  ap.add_argument("--out", default=None, help="write the candidate ledger JSON here")
  ap.add_argument("--fast", action="store_true", help="use the chain cp proxy instead of exact edge-count longest path")
  ap.add_argument("--include-edge-pairs", action="store_true", help="embed removed edge pairs in rows (small reports only)")
  ap.add_argument("--top-n", type=int, default=DEFAULT_CONFIG["top_n_by_bytes"])
  args = ap.parse_args()

  if args.synthetic:
    calls, manifest = build_attribution_fixture()
    report = compute_attribution_report(calls, calls, manifest)
  elif args.capture:
    report = load_capture(args.capture)
  else:
    ap.error("no mode selected (--synthetic | --capture <json>)")
    return 2

  config = {"cp_mode": "chain" if args.fast else "exact",
            "top_n_by_bytes": args.top_n}
  ledger = search_candidates(report, config, include_edge_pairs=args.include_edge_pairs)
  if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
      json.dump(ledger, f, indent=2, sort_keys=True)
      f.write("\n")
    print("wrote ledger:", args.out)
  _print_summary(ledger)
  return 0


if __name__ == "__main__":
  sys.exit(main())
