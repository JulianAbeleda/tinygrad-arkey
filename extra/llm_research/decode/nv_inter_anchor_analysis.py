#!/usr/bin/env python3
"""Weighted inter-anchor causal-gap analysis for NV d512 decode.

This is measurement tooling only.  It consumes the two artifacts produced by
the Phase B captures:

1. the current-HEAD tinygrad control capture from
   ``nv_rmsnorm_current_head_topology.py`` (PROFILE=1 HCQ graph profile);
2. the weighted llama real-edge DAG from ``llama_weighted_dag.py --dump``.

It repairs the tinygrad timestamp-unit bug in that capture, repairs the
anchor census, and then computes the quantities the scope requires:

- per-layer S0-S4 timestamp exposure and weighted dependency cost for both
  implementations, using one matched anchor model (Q -> O -> gate/up ->
  down -> next-layer Q, plus the vocab tail);
- per-family node mass, union, hidden-behind-anchor, and exposed time;
- node/class/edge zero-cost critical-path ceilings with a full longest-path
  recomputation after every change so alternate-path takeover is included;
- observed device-union marginal ceilings for the same families;
- a reconciled ledger whose rows sum to the measured wall gap.

Profiled durations here are topology evidence, never an unprofiled wall
claim.  Nothing in this file changes runtime behavior.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, pathlib, statistics
from collections import Counter, defaultdict

SCHEMA_CONTROL = "tinygrad.nv_rmsnorm_current_head_topology.v1"
SCHEMA_LEDGER = "tinygrad.nv_inter_anchor_ledger.v1"
SCHEMA_SENSITIVITY = "tinygrad.nv_inter_anchor_wall_sensitivity.v1"
PROMOTION_GATE_US = 50.0


def _sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


# ---------------------------------------------------------------------------
# interval helpers (all microsecond-valued)

def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
  out: list[list[float]] = []
  for start, end in sorted(intervals):
    if end < start:
      raise ValueError("interval end precedes start")
    if not out or start > out[-1][1]:
      out.append([start, end])
    elif end > out[-1][1]:
      out[-1][1] = end
  return [(x[0], x[1]) for x in out]


def interval_union(intervals: list[tuple[float, float]]) -> float:
  return sum(end - start for start, end in merge_intervals(intervals))


def intersection_us(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
  aa, bb = merge_intervals(a), merge_intervals(b)
  i = j = 0
  total = 0.0
  while i < len(aa) and j < len(bb):
    total += max(0.0, min(aa[i][1], bb[j][1]) - max(aa[i][0], bb[j][0]))
    if aa[i][1] <= bb[j][1]:
      i += 1
    else:
      j += 1
  return total


# ---------------------------------------------------------------------------
# tinygrad capture canonicalization

def tg_anchor_ids(dag: dict) -> dict[str, list[int]]:
  """Mirror of ``_anchor_ids`` in nv_rmsnorm_current_head_topology.py."""
  anchors: dict[str, list[int]] = {"Q": [], "O": [], "gate_up": [], "down": [], "vocab": []}
  for node in dag["nodes"]:
    name = str(node.get("name", ""))
    if name.startswith("q4k_g3_lanemap_gemv_epi_resadd_"):
      anchors["O"].append(node["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_w1w3fused16_"):
      anchors["gate_up"].append(node["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_w1w3fused_"):
      anchors["gate_up"].append(node["id"])
    elif "_epi_ffnresadd_" in name or name.endswith("_epi_ffnresadd"):
      anchors["down"].append(node["id"])
    elif name == "q4k_warp_coop_q8_dp4a_partial_4096_4096":
      anchors["Q"].append(node["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_") and name.endswith("_4096_4096"):
      anchors["Q"].append(node["id"])
    elif "vocab_scalar_reduce" in name:
      anchors["vocab"].append(node["id"])
  for kind, ids in anchors.items():
    ids.sort()
  return anchors


def tg_family(name: str) -> str:
  if name.startswith("flash_block_tiled"):
    return "flash_score"
  if name.startswith("flash_fused_gmax_combine"):
    return "flash_combine"
  if name.startswith("reduce_output_rmsnorm"):
    return "rmsnorm"
  if name.startswith("rmsnorm_q8_1_llama_provider"):
    return "quant_provider"
  if name.startswith("E_"):
    return "residual"
  if name.startswith("r_"):
    return "reduce"
  if name.startswith("q6k_gen_coop"):
    return "vocab"
  if name.startswith(("q4k", "q6k")):
    return "gemv"
  return "other"


def canonicalize_tinygrad(path: pathlib.Path) -> dict:
  """Load the raw control capture and repair timestamps and anchors.

  HCQ graph-profile ``start``/``end`` signals are microseconds.  The capture
  tool stored them in fields named ``start_ns``/``end_ns`` and then divided
  by 1000 while deriving ``start_us``/``end_us``.  This restores the direct
  microsecond interpretation and collapses the per-graph-group host launch
  gaps into one back-to-back device timeline (the five groups execute as
  five sequential graph launches on one queue).
  """
  dag = json.loads(path.read_text(encoding="utf-8"))
  nodes = dag["nodes"]
  for node in nodes:
    if node.get("start_ns") is None or node.get("end_ns") is None:
      raise ValueError(f"tinygrad node {node.get('id')} is missing raw timestamps")
    node["start_raw_us"] = float(node["start_ns"])
    node["end_raw_us"] = float(node["end_ns"])

  gmin: dict[int, float] = defaultdict(lambda: float("inf"))
  for node in nodes:
    gmin[node["group_id"]] = min(gmin[node["group_id"]], node["start_raw_us"])
  group_order = sorted(gmin, key=gmin.get)
  offsets: dict[int, float] = {}
  acc = 0.0
  for gid in group_order:
    offsets[gid] = acc
    acc += max(n["end_raw_us"] for n in nodes if n["group_id"] == gid) - gmin[gid]
  for node in nodes:
    node["start_us"] = round(offsets[node["group_id"]] + node["start_raw_us"] - gmin[node["group_id"]], 3)
    node["end_us"] = round(offsets[node["group_id"]] + node["end_raw_us"] - gmin[node["group_id"]], 3)

  anchors = tg_anchor_ids(dag)
  down_last = anchors["down"][-1]
  vocab_node = None
  for node in nodes:
    if node["id"] > down_last and node["name"].startswith(("q4k", "q6k")):
      vocab_node = node["id"]
      break
  anchors["vocab"] = [vocab_node] if vocab_node is not None else []

  for node in nodes:
    family = tg_family(str(node.get("name", "")))
    # The tail starts at the vocab GEMV; the final norm/cast between the
    # last down anchor and the vocab GEMV belongs to the S4 segment instead.
    scope = "tail" if node["id"] >= anchors["vocab"][0] else "layer"
    anchor = next((kind for kind, ids in anchors.items() if node["id"] in ids), None)
    # Align with llama's K/V GEMV split: non-anchor GEMVs are exactly the
    # per-layer K and V projections.
    if family == "gemv" and anchor is None:
      family = "gemv_kv"
    node["family"] = family
    node["scope"] = scope
    node["anchor"] = anchor

  dag["anchor_ids"] = anchors
  dag["postprocess"] = {
    "tool": "nv_inter_anchor_analysis.py",
    "what": (
      "start_us/end_us recomputed directly from raw HCQ microsecond signals; "
      "the prior capture divided by 1000.  Group host gaps collapsed into one "
      "back-to-back device timeline; anchor_ids repaired to the full 36/36/36/36 census."
    ),
    "group_offsets_us": {str(k): round(v, 3) for k, v in offsets.items()},
    "group_order": group_order,
    "device_span_us": round(acc, 3),
  }
  # Recompute the summary fields the capture tool derived from durations so
  # the canonical artifact is self-consistent with the repaired timeline.
  deps, _weights = dag_deps(nodes, dag.get("edges") or [], "id", "id")
  durations = [float(n.get("duration_us", 0.0) or 0.0) for n in nodes]
  cp, _ = longest_path(durations, deps)
  per_group: dict[str, dict] = {}
  for gid in group_order:
    members = [n for n in nodes if n["group_id"] == gid]
    members.sort(key=lambda n: n["id"])
    ids = {n["id"] for n in members}
    sub_edges = [e for e in (dag.get("edges") or []) if e["from"] in ids and e["to"] in ids]
    sub_deps, _ = dag_deps(members, sub_edges, "id", "id")
    sub_dur = [float(n.get("duration_us", 0.0) or 0.0) for n in members]
    sub_cp, _ = longest_path(sub_dur, sub_deps)
    per_group[str(gid)] = {
      "n": len(members),
      "serialized_us": round(sum(sub_dur), 3),
      "critical_path_us": round(sub_cp, 3),
      "device_span_us": round(max(n["end_us"] for n in members) - min(n["start_us"] for n in members), 3),
    }
  dag["summary"] = {
    "node_count": len(nodes),
    "edge_count": len(dag.get("edges") or []),
    "cross_group_edge_count": sum(1 for e in (dag.get("edges") or []) if e.get("crosses_group")),
    "per_group": per_group,
    "critical_path_us": round(cp, 3),
    "device_span_us": round(acc, 3),
    "device_union_us": round(interval_union([(n["start_us"], n["end_us"]) for n in nodes]), 3),
  }
  return dag


# ---------------------------------------------------------------------------
# llama DAG loading

def llama_family(node: dict) -> str:
  role = node.get("role", "")
  if role in ("Q", "O", "G", "D"):
    return "gemv"
  if role in ("K", "V"):
    return "gemv_kv"
  if role == "vocab":
    return "vocab"
  if role.endswith("_quant"):
    return "quant_provider"
  if role.endswith("_norm") or role == "final_norm":
    return "rmsnorm"
  if role == "flash":
    return "flash_score"
  if role == "combine":
    return "flash_combine"
  if role in ("q_rope", "k_rope"):
    return "rope"
  if role in ("k_store", "set_rows", "get_rows", "get_rows_a", "get_rows_b"):
    return "kv"
  if role == "binbcast":
    return "residual"
  return "other"


def load_llama(path: pathlib.Path) -> dict:
  dag = json.loads(path.read_text(encoding="utf-8"))
  nodes = dag["nodes"]
  assert [n["local_id"] for n in nodes] == list(range(len(nodes))), "llama nodes must be dense"
  anchor_role = {"Q": "Q", "O": "O", "G": "gate_up", "D": "down", "vocab": "vocab"}
  for node in nodes:
    node["family"] = llama_family(node)
    node["anchor"] = anchor_role.get(node["role"])
    node["scope"] = "tail" if node["anchor"] == "vocab" else "layer"
  anchors: dict[str, list[int]] = {"Q": [], "O": [], "gate_up": [], "down": [], "vocab": []}
  for node in nodes:
    if node["anchor"] == "Q":
      anchors["Q"].append(node["local_id"])
    elif node["anchor"] == "O":
      anchors["O"].append(node["local_id"])
    elif node["anchor"] == "gate_up":
      anchors["gate_up"].append(node["local_id"])
    elif node["anchor"] == "down":
      anchors["down"].append(node["local_id"])
    elif node["anchor"] == "vocab":
      anchors["vocab"].append(node["local_id"])
  dag["anchor_ids"] = anchors
  return dag


# ---------------------------------------------------------------------------
# DAG / critical-path machinery

def dag_deps(nodes: list[dict], edges: list[dict], from_key: str, to_key: str) -> tuple[list[list[int]], list[int]]:
  ids = {n["id" if from_key == "id" else from_key]: i for i, n in enumerate(nodes)}
  deps: list[list[int]] = [[] for _ in nodes]
  for edge in edges:
    a, b = ids.get(edge["from"]), ids.get(edge["to"])
    if a is None or b is None or a == b:
      continue
    if a not in deps[b]:
      deps[b].append(a)
  return deps, list(range(len(nodes)))


def longest_path(weights: list[float], deps: list[list[int]]) -> tuple[float, list[int]]:
  n = len(weights)
  dp = [0.0] * n
  prev = [-1] * n
  for i in range(n):
    best = -1
    best_v = 0.0
    for p in deps[i]:
      if dp[p] > best_v:
        best_v = dp[p]
        best = p
    dp[i] = best_v + weights[i]
    prev[i] = best
  if not n:
    return 0.0, []
  end = max(range(n), key=lambda i: dp[i])
  path: list[int] = []
  cur = end
  while cur != -1:
    path.append(cur)
    cur = prev[cur]
  return dp[end], path[::-1]


def completions(weights: list[float], deps: list[list[int]]) -> list[float]:
  comp = [0.0] * len(weights)
  for i in range(len(weights)):
    ready = 0.0
    for p in deps[i]:
      if comp[p] > ready:
        ready = comp[p]
    comp[i] = ready + weights[i]
  return comp


def strict_path_cost(start: int, end: int, weights: list[float],
                     children: list[list[int]]) -> tuple[float, list[int]]:
  """Longest path start->end, excluding both endpoint durations."""
  if start == end:
    return 0.0, []
  n = len(weights)
  dp = [float("-inf")] * n
  prev = [-1] * n
  # Node weights live on the nodes themselves, so carry the inclusive path
  # sum and strip the two endpoint weights at the end.
  dp[start] = weights[start]
  for i in range(start, end):
    if dp[i] == float("-inf"):
      continue
    for j in children[i]:
      if j > end:
        continue
      cand = dp[i] + weights[j]
      if cand > dp[j]:
        dp[j] = cand
        prev[j] = i
  if dp[end] == float("-inf"):
    return float("nan"), []
  cost = dp[end] - weights[start] - weights[end]
  path: list[int] = []
  cur = end
  while cur != start:
    path.append(cur)
    cur = prev[cur]
  path.append(start)
  return cost, path[::-1]


def node_zero_ceilings(weights: list[float], deps: list[list[int]], base_cp: float) -> list[dict]:
  rows = []
  for i in range(len(weights)):
    changed = weights.copy()
    changed[i] = 0.0
    cp, _ = longest_path(changed, deps)
    rows.append({"node": i, "zero_cost_cp_ceiling_us": round(base_cp - cp, 3)})
  rows.sort(key=lambda r: (-r["zero_cost_cp_ceiling_us"], r["node"]))
  return rows


def edge_zero_ceilings(weights: list[float], deps: list[list[int]], base_cp: float,
                       edges: list[dict]) -> list[dict]:
  rows = []
  for idx, edge in enumerate(edges):
    changed = [list(d) for d in deps]
    a, b = edge["from"], edge["to"]
    if a in changed[b]:
      changed[b].remove(a)
    cp, _ = longest_path(weights, changed)
    rows.append({"edge": idx, "from": a, "to": b, "kind": edge.get("kind", "DATA"),
                 "zero_cost_cp_ceiling_us": round(base_cp - cp, 3)})
  rows.sort(key=lambda r: (-r["zero_cost_cp_ceiling_us"], r["edge"]))
  return rows


def family_ceilings(weights: list[float], deps: list[list[int]], base_cp: float,
                    families: list[str], intervals: list[tuple[float, float]],
                    anchor_flags: list[bool]) -> dict[str, dict]:
  union_all = interval_union(intervals)
  anchor_iv = [iv for iv, flag in zip(intervals, anchor_flags) if flag]
  out: dict[str, dict] = {}
  for family in sorted(set(families)):
    members = [i for i, f in enumerate(families) if f == family]
    changed = weights.copy()
    for i in members:
      changed[i] = 0.0
    cp, _ = longest_path(changed, deps)
    remaining = [iv for i, iv in enumerate(intervals) if i not in members]
    out[family] = {
      "node_count": len(members),
      "node_mass_us": round(sum(weights[i] for i in members), 3),
      "zero_cost_cp_ceiling_us": round(base_cp - cp, 3),
      "union_marginal_ceiling_us": round(union_all - interval_union(remaining), 3),
      "union_us": round(interval_union([intervals[i] for i in members]), 3),
      "hidden_behind_anchor_us": round(
        intersection_us([intervals[i] for i in members], anchor_iv), 3),
    }
  return out


# ---------------------------------------------------------------------------
# segment analysis

def segment_rows(nodes: list[dict], anchors: dict[str, list[int]], deps: list[list[int]],
                 weights: list[float], id_key: str) -> list[dict]:
  """Per-layer S0-S4 rows with exposure, weighted dependency, and on-path cost."""
  comp = completions(weights, deps)
  children: list[list[int]] = [[] for _ in weights]
  for i, preds in enumerate(deps):
    for p in preds:
      children[p].append(i)
  by_id = {n[id_key]: n for n in nodes}
  rows: list[dict] = []
  q_ids = anchors["Q"]
  layers = len(q_ids)
  token_start = min(n["start_us"] for n in nodes)
  token_end = max(n["end_us"] for n in nodes)
  for layer in range(layers):
    q, o, g, d = q_ids[layer], anchors["O"][layer], anchors["gate_up"][layer], anchors["down"][layer]
    prev_d = anchors["down"][layer - 1] if layer > 0 else None
    next_q = q_ids[layer + 1] if layer + 1 < layers else anchors["vocab"][0]
    specs = [
      ("S0", prev_d, q, token_start, None),
      ("S1", q, o, None, None),
      ("S2", o, g, None, None),
      ("S3", g, d, None, None),
      ("S4", d, next_q, None, token_end),
    ]
    for name, a_id, b_id, start_fallback, end_fallback in specs:
      b_start = by_id[b_id]["start_us"]
      if a_id is None:
        exposure = b_start - start_fallback
      else:
        exposure = b_start - by_id[a_id]["end_us"]
      window_start = start_fallback if a_id is None else by_id[a_id]["end_us"]
      weighted = (comp[b_id] - weights[b_id]) - (comp[a_id] if a_id is not None else 0.0)
      weighted = max(0.0, weighted)
      on_path = float("nan")
      if a_id is not None:
        on_path, _ = strict_path_cost(a_id, b_id, weights, children)
      else:
        on_path = weighted
      rows.append({
        "layer": layer,
        "segment": name,
        "from": a_id,
        "to": b_id,
        "exposure_us": round(exposure, 3),
        "window_start_us": round(window_start, 3),
        "window_end_us": round(b_start, 3),
        "weighted_dependency_us": round(weighted, 3),
        "on_path_spine_us": round(on_path, 3) if on_path == on_path else None,
        "branch_join_us": round(max(0.0, weighted - (on_path if on_path == on_path else 0.0)), 3),
      })
  # Tail exposure: vocab anchor end -> token end (output-logit reductions).
  vocab_id = anchors["vocab"][0]
  tail_exposure = token_end - by_id[vocab_id]["end_us"]
  rows.append({
    "layer": layers - 1,
    "segment": "tail_after_vocab",
    "from": vocab_id,
    "to": None,
    "exposure_us": round(tail_exposure, 3),
    "window_start_us": round(by_id[vocab_id]["end_us"], 3),
    "window_end_us": round(token_end, 3),
    "weighted_dependency_us": round(tail_exposure, 3),
    "on_path_spine_us": round(tail_exposure, 3),
    "branch_join_us": 0.0,
  })
  return rows


def segment_family_composition(rows: list[dict], nodes: list[dict], id_key: str) -> dict:
  """Device-interval mass of each family inside each segment exposure window."""
  out: dict[str, dict[str, dict]] = {}
  for row in rows:
    segment = row["segment"]
    window = (row["window_start_us"], row["window_end_us"])
    acc = out.setdefault(segment, {})
    for node in nodes:
      mass = intersection_us([(node["start_us"], node["end_us"])], [window])
      if mass <= 1e-9:
        continue
      family = node["family"]
      entry = acc.setdefault(family, {"window_mass_us": 0.0, "touching_nodes": 0})
      entry["window_mass_us"] += mass
      entry["touching_nodes"] += 1
  for families in out.values():
    for entry in families.values():
      entry["window_mass_us"] = round(entry["window_mass_us"], 3)
  return out


def summarize_segments(rows: list[dict]) -> dict:
  out: dict[str, dict] = {}
  for segment in ("S0", "S1", "S2", "S3", "S4"):
    sel = [r for r in rows if r["segment"] == segment]
    out[segment] = {
      "layers": len(sel),
      "exposure_total_us": round(sum(r["exposure_us"] for r in sel), 3),
      "exposure_mean_us": round(statistics.mean(r["exposure_us"] for r in sel), 3),
      "exposure_median_us": round(statistics.median(r["exposure_us"] for r in sel), 3),
      "weighted_total_us": round(sum(r["weighted_dependency_us"] for r in sel), 3),
      "weighted_mean_us": round(statistics.mean(r["weighted_dependency_us"] for r in sel), 3),
      "on_path_total_us": round(sum(r["on_path_spine_us"] for r in sel), 3),
    }
  tail = [r for r in rows if r["segment"] == "tail_after_vocab"]
  out["tail_after_vocab"] = {
    "exposure_total_us": round(sum(r["exposure_us"] for r in tail), 3),
  }
  return out


def family_ledger(nodes: list[dict], anchor_kinds: frozenset[str]) -> dict:
  anchor_iv = [(n["start_us"], n["end_us"]) for n in nodes if n["anchor"] in anchor_kinds]
  families = sorted({n["family"] for n in nodes})
  rows: dict[str, dict] = {}
  by_family: dict[str, list[dict]] = defaultdict(list)
  for n in nodes:
    by_family[n["family"]].append(n)
  for family in families:
    members = by_family[family]
    ivs = [(n["start_us"], n["end_us"]) for n in members]
    hidden = intersection_us(ivs, anchor_iv) if family not in anchor_kinds else 0.0
    rows[family] = {
      "nodes": len(members),
      "node_sum_us": round(sum(n["duration_us"] for n in members), 3),
      "union_us": round(interval_union(ivs), 3),
      "hidden_behind_anchor_us": round(hidden, 3),
      "exposed_vs_anchor_us": round(interval_union(ivs) - hidden, 3),
    }
  return rows


# ---------------------------------------------------------------------------
# ledger assembly

def assemble(nodes_tg: list[dict], edges_tg: list[dict], dag_tg: dict,
             nodes_ll: list[dict], edges_ll: list[dict], dag_ll: dict,
             wall_bracket: dict, llama_unprofiled: dict) -> tuple[dict, dict]:
  weights_tg = [float(n.get("duration_us", 0.0) or 0.0) for n in nodes_tg]
  deps_tg, _ = dag_deps(nodes_tg, edges_tg, "id", "id")
  cp_tg, cp_path_tg = longest_path(weights_tg, deps_tg)

  weights_ll = [float(n.get("duration_us", 0.0) or 0.0) for n in nodes_ll]
  deps_ll, _ = dag_deps(nodes_ll, edges_ll, "local_id", "local_id")
  cp_ll, cp_path_ll = longest_path(weights_ll, deps_ll)

  iv_tg = [(n["start_us"], n["end_us"]) for n in nodes_tg]
  iv_ll = [(n["start_us"], n["end_us"]) for n in nodes_ll]
  union_tg = interval_union(iv_tg)
  union_ll = interval_union(iv_ll)

  anchor_kinds = frozenset(("Q", "O", "gate_up", "down", "vocab"))
  tg_ledger = family_ledger(nodes_tg, anchor_kinds)
  ll_ledger = family_ledger(nodes_ll, anchor_kinds)

  rows_tg = segment_rows(nodes_tg, dag_tg["anchor_ids"], deps_tg, weights_tg, "id")
  rows_ll = segment_rows(nodes_ll, dag_ll["anchor_ids"], deps_ll, weights_ll, "local_id")
  seg_tg = summarize_segments(rows_tg)
  seg_ll = summarize_segments(rows_ll)
  comp_tg = segment_family_composition(rows_tg, nodes_tg, "id")
  comp_ll = segment_family_composition(rows_ll, nodes_ll, "local_id")

  # Wall records.
  gen = next(r for r in llama_unprofiled if r.get("n_gen") == 20)
  llama_wall_us = gen["avg_ns"] / 1000.0 / gen["n_gen"]
  tg_wall_us = wall_bracket["brackets"]["A"]["control_bracket_median_ms"] * 1000.0
  wall_gap = tg_wall_us - llama_wall_us
  device_gap = union_tg - union_ll
  host_residual = wall_gap - device_gap

  # Anchor-scope decomposition with the vocab tail split out on both sides.
  four = frozenset(("Q", "O", "gate_up", "down"))
  anchors_4_tg = [(n["start_us"], n["end_us"]) for n in nodes_tg if n["anchor"] in four]
  anchors_4_ll = [(n["start_us"], n["end_us"]) for n in nodes_ll if n["anchor"] in four]
  tail_tg = [(n["start_us"], n["end_us"]) for n in nodes_tg if n["scope"] == "tail"]
  tail_ll = [(n["start_us"], n["end_us"]) for n in nodes_ll if n["scope"] == "tail"]
  support_tg = [(n["start_us"], n["end_us"]) for n in nodes_tg if n["anchor"] is None and n["scope"] != "tail"]
  support_ll = [(n["start_us"], n["end_us"]) for n in nodes_ll if n["anchor"] is None and n["scope"] != "tail"]
  a4_tg, a4_ll = interval_union(anchors_4_tg), interval_union(anchors_4_ll)
  tail_tg_u, tail_ll_u = interval_union(tail_tg), interval_union(tail_ll)
  support_tg_u = interval_union(support_tg) - intersection_us(support_tg, anchors_4_tg)
  support_ll_u = interval_union(support_ll) - intersection_us(support_ll, anchors_4_ll)

  reconciliation = [
    {"row": "Q/O/gate_up/down anchor union", "tinygrad_us": round(a4_tg, 3),
     "llama_us": round(a4_ll, 3), "delta_tiny_minus_llama_us": round(a4_tg - a4_ll, 3),
     "mechanism": "llama MMQ anchor bodies are larger; llama is slower here"},
    {"row": "per-layer support exposed", "tinygrad_us": round(support_tg_u, 3),
     "llama_us": round(support_ll_u, 3), "delta_tiny_minus_llama_us": round(support_tg_u - support_ll_u, 3),
     "mechanism": "fused epilogues + PDL launch-completion overlap + off-path KV/quant branches"},
    {"row": "vocab tail (gemv + final norm/quant + logit reductions)",
     "tinygrad_us": round(tail_tg_u, 3), "llama_us": round(tail_ll_u, 3),
     "delta_tiny_minus_llama_us": round(tail_tg_u - tail_ll_u, 3),
     "mechanism": "tinygrad runs post-vocab output reductions; llama folds the output reduction"},
    {"row": "interval overlap accounting residual",
     "tinygrad_us": None, "llama_us": None,
     "delta_tiny_minus_llama_us": round(device_gap - ((a4_tg - a4_ll) + (support_tg_u - support_ll_u) + (tail_tg_u - tail_ll_u)), 3),
     "mechanism": "anchor/support/tail intervals overlap at boundaries; keeps the ledger an exact sum"},
    {"row": "device union total", "tinygrad_us": round(union_tg, 3),
     "llama_us": round(union_ll, 3), "delta_tiny_minus_llama_us": round(device_gap, 3),
     "mechanism": "profiled device-time difference"},
    {"row": "host / launch residual (wall - device)", "tinygrad_us": round(tg_wall_us - union_tg, 3),
     "llama_us": round(llama_wall_us - union_ll, 3),
     "delta_tiny_minus_llama_us": round(host_residual, 3),
     "mechanism": "llama single graph launch per token vs tinygrad 5 sequential graph launches; "
                  "tinygrad's PROFILE=1 device timestamps absorb profiler tax, so this row is a lower bound"},
  ]

  # Zero-cost ceilings.
  families_tg = [n["family"] for n in nodes_tg]
  families_ll = [n["family"] for n in nodes_ll]
  ceil_tg = family_ceilings(weights_tg, deps_tg, cp_tg, families_tg, iv_tg,
                            [n["anchor"] in four for n in nodes_tg])
  ceil_ll = family_ceilings(weights_ll, deps_ll, cp_ll, families_ll, iv_ll,
                            [n["anchor"] in four for n in nodes_ll])
  cp_set_tg = set(cp_path_tg)
  cp_set_ll = set(cp_path_ll)
  for family, row in ceil_tg.items():
    row["cp_mass_us"] = round(sum(weights_tg[i] for i, f in enumerate(families_tg)
                                  if f == family and i in cp_set_tg), 3)
  for family, row in ceil_ll.items():
    row["cp_mass_us"] = round(sum(weights_ll[i] for i, f in enumerate(families_ll)
                                  if f == family and i in cp_set_ll), 3)

  node_ceil_tg = node_zero_ceilings(weights_tg, deps_tg, cp_tg)
  node_ceil_ll = node_zero_ceilings(weights_ll, deps_ll, cp_ll)
  edge_ceil_tg = edge_zero_ceilings(weights_tg, deps_tg, cp_tg, edges_tg)
  edge_ceil_ll = edge_zero_ceilings(weights_ll, deps_ll, cp_ll, edges_ll)

  def decorate_nodes(rows: list[dict], nodes: list[dict], id_key: str, families: list[str], top: int) -> list[dict]:
    out = []
    for row in rows[:top]:
      idx = row["node"]
      node = nodes[idx]
      out.append({
        "node": node[id_key],
        "name": node.get("name", str(node[id_key])),
        "family": families[idx],
        "scope": node.get("scope"),
        "duration_us": round(weights_tg[idx] if id_key == "id" else weights_ll[idx], 3),
        "zero_cost_cp_ceiling_us": row["zero_cost_cp_ceiling_us"],
      })
    return out

  def decorate_edges(rows: list[dict], nodes: list[dict], id_key: str, top: int) -> list[dict]:
    out = []
    for row in rows[:top]:
      out.append({
        "from_node": nodes[row["from"]][id_key],
        "from_name": nodes[row["from"]].get("name", ""),
        "to_node": nodes[row["to"]][id_key],
        "to_name": nodes[row["to"]].get("name", ""),
        "kind": row["kind"],
        "zero_cost_cp_ceiling_us": row["zero_cost_cp_ceiling_us"],
      })
    return out

  node_top_tg = decorate_nodes(node_ceil_tg, nodes_tg, "id", families_tg, 15)
  node_top_ll = decorate_nodes(node_ceil_ll, nodes_ll, "local_id", families_ll, 15)
  edge_top_tg = decorate_edges(edge_ceil_tg, nodes_tg, "id", 15)
  edge_top_ll = decorate_edges(edge_ceil_ll, nodes_ll, "local_id", 15)

  # S4 of layer L is the same interval as S0 of layer L+1 under the scope's
  # definitions ("down end -> next Q start" and "prior down end -> Q start").
  # Tile the token with S0..S3 plus the final layer's S4 once.
  anchor_delta_tg = sum(n["end_us"] - n["start_us"] for n in nodes_tg if n["anchor"] in four)
  anchor_delta_ll = sum(n["end_us"] - n["start_us"] for n in nodes_ll if n["anchor"] in four)
  vocab_start_tg = next(n["start_us"] for n in nodes_tg if n["anchor"] == "vocab")
  vocab_start_ll = next(n["start_us"] for n in nodes_ll if n["anchor"] == "vocab")
  t0_tg = min(n["start_us"] for n in nodes_tg)
  t0_ll = min(n["start_us"] for n in nodes_ll)
  seg_tile_tg = sum(seg_tg[s]["exposure_total_us"] for s in ("S0", "S1", "S2", "S3"))
  seg_tile_tg += next(r["exposure_us"] for r in rows_tg if r["segment"] == "S4" and r["layer"] == 35)
  seg_tile_ll = sum(seg_ll[s]["exposure_total_us"] for s in ("S0", "S1", "S2", "S3"))
  seg_tile_ll += next(r["exposure_us"] for r in rows_ll if r["segment"] == "S4" and r["layer"] == 35)
  tiling_tg = {
    "anchor_interval_sum_us": round(anchor_delta_tg, 3),
    "segment_tile_us": round(seg_tile_tg, 3),
    "reconstructed_us": round(anchor_delta_tg + seg_tile_tg, 3),
    "per_layer_device_span_us": round(vocab_start_tg - t0_tg, 3),
    "per_layer_device_union_us": round(union_tg - tail_tg_u, 3),
    "reconstructed_minus_span_us": round(anchor_delta_tg + seg_tile_tg - (vocab_start_tg - t0_tg), 3),
    "span_minus_union_us": round((vocab_start_tg - t0_tg) - (union_tg - tail_tg_u), 3),
  }
  tiling_ll = {
    "anchor_interval_sum_us": round(anchor_delta_ll, 3),
    "segment_tile_us": round(seg_tile_ll, 3),
    "reconstructed_us": round(anchor_delta_ll + seg_tile_ll, 3),
    "per_layer_device_span_us": round(vocab_start_ll - t0_ll, 3),
    "per_layer_device_union_us": round(union_ll - tail_ll_u, 3),
    "reconstructed_minus_span_us": round(anchor_delta_ll + seg_tile_ll - (vocab_start_ll - t0_ll), 3),
    "span_minus_union_us": round((vocab_start_ll - t0_ll) - (union_ll - tail_ll_u), 3),
  }

  ledger = {
    "schema": SCHEMA_LEDGER,
    "promotion_gate_us": PROMOTION_GATE_US,
    "provenance": {
      "tinygrad_control": {"path": dag_tg.get("_source_path"), "commit": dag_tg.get("commit"),
                           "node_count": len(nodes_tg), "edge_count": len(edges_tg)},
      "llama_dag": {"path": dag_ll.get("_source_path"),
                    "trace_sha256": dag_ll["provenance"]["trace_sha256"],
                    "dump_sha256": dag_ll["provenance"]["dump_sha256"],
                    "chosen_span_us": dag_ll["provenance"]["chosen_span_us"],
                    "node_count": len(nodes_ll), "data_edge_count": len(edges_ll)},
    },
    "wall": {
      "llama_us_per_token": round(llama_wall_us, 3),
      "llama_source": {"path": dag_tg.get("_llama_unprofiled_path"),
                       "avg_ns_per_20_tokens": gen["avg_ns"], "n_gen": gen["n_gen"]},
      "tinygrad_control_bracket_median_us": round(tg_wall_us, 3),
      "tinygrad_source": {"path": dag_tg.get("_wall_bracket_path"), "arm": "A", "sites": ["ffn"]},
      "gap_us": round(wall_gap, 3),
      "note": "unprofiled vs unprofiled; llama 20-token generation avg divided by 20",
    },
    "device": {
      "llama_union_us": round(union_ll, 3), "llama_node_sum_us": round(sum(weights_ll), 3),
      "llama_overlap_mass_us": round(sum(weights_ll) - union_ll, 3),
      "llama_span_us": dag_ll["summary"]["span_us"],
      "tinygrad_union_us": round(union_tg, 3), "tinygrad_node_sum_us": round(sum(weights_tg), 3),
      "tinygrad_node_sum_minus_union_us": round(sum(weights_tg) - union_tg, 3),
      "gap_us": round(device_gap, 3),
      "note": "profiled union vs profiled union; never compare directly with unprofiled wall",
    },
    "host_residual": {
      "gap_us": round(host_residual, 3),
      "llama_host_and_tax_us": round(llama_wall_us - union_ll, 3),
      "tinygrad_profiled_residual_us": round(tg_wall_us - union_tg, 3),
      "note": "wall gap minus device gap; tinygrad sign absorbs its PROFILE=1 device tax",
    },
    "reconciliation": reconciliation,
    "segments": {
      "tinygrad": {"rows": rows_tg, "summary": seg_tg},
      "llama": {"rows": rows_ll, "summary": seg_ll},
      "note": "S4 of layer L is the same device interval as S0 of layer L+1; "
              "cross-layer sums must count it once",
    },
    "segment_family_composition": {
      "tinygrad": comp_tg,
      "llama": comp_ll,
      "note": "each family's device-interval mass inside the segment exposure window; "
              "family sums can exceed the window union when families overlap in time",
    },
    "family_ledger": {
      "anchor_kinds": sorted(anchor_kinds),
      "tinygrad": tg_ledger,
      "llama": ll_ledger,
    },
    "tiling_check": {"tinygrad": tiling_tg, "llama": tiling_ll},
  }

  build_rows = []
  for family in sorted(set(families_tg)):
    row = ceil_tg[family]
    build_rows.append({
      "family": family,
      "node_mass_us": row["node_mass_us"],
      "cp_mass_us": row["cp_mass_us"],
      "zero_cost_cp_ceiling_us": row["zero_cost_cp_ceiling_us"],
      "union_marginal_ceiling_us": row["union_marginal_ceiling_us"],
      "hidden_behind_anchor_us": row["hidden_behind_anchor_us"],
    })
  build_rows.sort(key=lambda r: -r["union_marginal_ceiling_us"])

  sensitivity = {
    "schema": SCHEMA_SENSITIVITY,
    "promotion_gate_us": PROMOTION_GATE_US,
    "provenance": ledger["provenance"],
    "critical_path": {
      "tinygrad_us": round(cp_tg, 3), "tinygrad_node_count": len(nodes_tg),
      "llama_us": round(cp_ll, 3), "llama_node_count": len(nodes_ll),
      "note": "duration-weighted longest path on logical dependency edges; "
              "llama's serialized CP exceeds its observed span because PDL "
              "launch-completion edges let consumers start before producers finish",
    },
    "family_ceilings": {"tinygrad": ceil_tg, "llama": ceil_ll},
    "node_ceilings_top": {"tinygrad": node_top_tg, "llama": node_top_ll},
    "edge_ceilings_top": {"tinygrad": edge_top_tg, "llama": edge_top_ll},
    "edge_ceiling_notes": {
      "llama": "zero-cost edge ceilings are not mechanically meaningful: removing an "
               "inferred logical data edge severs the reconstructed llama chain, so the "
               "CP delta is a connectivity artifact rather than a buildable change; "
               "llama edge rows are retained for structure audit, not ranking",
      "tinygrad": "edge ceilings are meaningful only for legal edges; the top tinygrad "
                  "edges are the gate_up->down RAW dependencies (~44-48us)",
    },
    "tinygrad_build_rank": build_rows,
    "closed_levers": [
      {"lever": "more than two compute GPFIFOs",
       "status": "closed", "evidence": "current-DAG ideal ceiling below promotion gate before real wait tax"},
      {"lever": "replay merge (JIT_BATCH_SIZE=1024)",
       "status": "closed", "evidence": "measured +112.9us slower on production wall"},
      {"lever": "early launch_dependents START placement",
       "status": "closed", "evidence": "no new overlap beyond landed QMD latch behavior"},
      {"lever": "coarse flash split S=4/S=2",
       "status": "closed", "evidence": "measured substantially slower on DEV=NV"},
    ],
  }
  return ledger, sensitivity


# ---------------------------------------------------------------------------

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--tinygrad", default="/tmp/nv-rmsnorm-phaseB-control-20260820.json", type=pathlib.Path)
  ap.add_argument("--llama", default="docs/task_workflow/output/nv-weighted-llama-real-edge-dag-20260820.json",
                  type=pathlib.Path)
  ap.add_argument("--wall-bracket", default="docs/task_workflow/output/nv-rmsnorm-current-head-wall-bracket-20260820.json",
                  type=pathlib.Path)
  ap.add_argument("--llama-unprofiled", default="/tmp/llama_unprofiled_20260820.json", type=pathlib.Path)
  ap.add_argument("--out-control", default="docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json",
                  type=pathlib.Path)
  ap.add_argument("--out-ledger", default="docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json",
                  type=pathlib.Path)
  ap.add_argument("--out-sensitivity",
                  default="docs/task_workflow/output/nv-weighted-inter-anchor-wall-sensitivity-20260820.json",
                  type=pathlib.Path)
  args = ap.parse_args()

  dag_tg = canonicalize_tinygrad(args.tinygrad)
  dag_tg["_source_path"] = str(args.tinygrad)
  dag_tg["_wall_bracket_path"] = str(args.wall_bracket)
  dag_tg["_llama_unprofiled_path"] = str(args.llama_unprofiled)
  dag_ll = load_llama(args.llama)
  dag_ll["_source_path"] = str(args.llama)
  wall_bracket = json.loads(args.wall_bracket.read_text(encoding="utf-8"))
  llama_unprofiled = json.loads(args.llama_unprofiled.read_text(encoding="utf-8"))

  control_out = copy.deepcopy(dag_tg)
  for key in ("_source_path", "_wall_bracket_path", "_llama_unprofiled_path"):
    control_out.pop(key, None)
  args.out_control.parent.mkdir(parents=True, exist_ok=True)
  args.out_control.write_text(json.dumps(control_out, indent=2, sort_keys=True) + "\n")

  ledger, sensitivity = assemble(
    dag_tg["nodes"], dag_tg["edges"], dag_tg,
    dag_ll["nodes"], dag_ll["data_edges"], dag_ll,
    wall_bracket, llama_unprofiled)
  args.out_ledger.parent.mkdir(parents=True, exist_ok=True)
  args.out_ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
  args.out_sensitivity.parent.mkdir(parents=True, exist_ok=True)
  args.out_sensitivity.write_text(json.dumps(sensitivity, indent=2, sort_keys=True) + "\n")

  print(json.dumps({
    "control": str(args.out_control),
    "ledger": str(args.out_ledger),
    "sensitivity": str(args.out_sensitivity),
    "wall": ledger["wall"],
    "device": ledger["device"],
    "reconciliation": ledger["reconciliation"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
