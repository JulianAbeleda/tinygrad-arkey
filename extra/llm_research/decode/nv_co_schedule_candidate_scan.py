#!/usr/bin/env python3
"""CPU-only in-graph co-schedule candidate scan for the NV decode overlap question.

The overlap attribution (445.954 us/token, llama's quantize_q8_1 mass hidden
behind MMVQ; see nv-decode-exposure-overlap-host-forward-scope-20260805.md)
can only be recovered without cross-queue waits by in-graph co-scheduling:
the CUDA driver overlapping dependency-independent kernels on ONE graph launch
(llama's mechanism).  Two-queue cuts are closed at the calibrated 3.1865
us/wait (nv-decode-p4-dependency-closed-cut-record-20260805.md), so this tool
scores every (support, quant/flash) pair that is dependency-independent in a
duration-bearing DAG:

- hideable_us = min(duration_support, duration_host): the duration that could
  sit inside the host kernel's shadow (llama-style interval containment);
- delta_cp_full_hide_us: the exact critical-path recovery if that support node
  vanished entirely (per-node ceiling, one longest-path recomputation);
- recovery_us: the exact critical-path recovery of THIS pair (host absorbs
  min(dS, dH)); exact for the top pairs, a proven upper bound otherwise;
- a greedy ranked selection that recomputes the critical path after every
  selection, so the ledger total never double-counts parallel branches.

Co-scheduling is a one-graph phenomenon: pairs are restricted to nodes within
the same graph launch (same group_id) unless --allow-cross-group is passed.
The emitted verdict is the +50 us promotion gate from the parent scope: any
co-schedule mechanism must clear +50 us of exact critical-path recovery on a
fresh DAG before a GPU arm.  This tool never books recovery and never changes
defaults.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib

SCHEMA = "tinygrad.nv_co_schedule_candidate_scan.v1"
PROMOTION_GATE_US = 50.0
SUPPORT_PREFIXES = ("E_", "r_")
HOST_PREFIXES = ("q4k", "q6k", "flash")
FAMILIES = ("q4k", "q6k", "flash")


def load(path: str) -> dict:
  with open(path, encoding="utf-8") as f: return json.load(f)


def classify(node: dict) -> str:
  """Support = E_/r_ with no metadata; host = quant/flash families; else fail."""
  name, metadata = node["name"], node.get("metadata")
  if name.startswith(SUPPORT_PREFIXES) and not metadata:
    return "support"
  if name.startswith(HOST_PREFIXES):
    return "host"
  raise ValueError(f"node {node['id']} ({name!r}) is neither support (E_/r_, no metadata) nor quant/flash host")


def family_of(name: str) -> str:
  for f in FAMILIES:
    if name.startswith(f): return f
  raise ValueError(f"node {name!r} is not a quant/flash host")


def _validate(dag: dict) -> tuple[list[dict], list[dict], list[list[int]]]:
  nodes, edges = dag.get("nodes"), dag.get("edges")
  if not isinstance(nodes, list) or not isinstance(edges, list):
    raise ValueError("DAG must be a dict with nodes/edges lists")
  if [n.get("id") for n in nodes] != list(range(len(nodes))):
    raise ValueError("node IDs must be dense program order")
  if not nodes:
    raise ValueError("DAG has no nodes")
  for n in nodes:
    if not isinstance(n.get("name"), str) or not n["name"]:
      raise ValueError(f"node {n.get('id')}: missing name")
    if not isinstance(n.get("duration_us"), (int, float)) or not n["duration_us"] > 0:
      raise ValueError(f"node {n['id']} needs a positive measured duration_us")
    if n.get("group_id") is None:
      raise ValueError(f"node {n['id']} needs a group_id")
  deps: list[list[int]] = [[] for _ in nodes]
  for e in edges:
    if "from" not in e or "to" not in e:
      raise ValueError(f"edge {e} needs from/to")
    a, b = int(e["from"]), int(e["to"])
    if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
      raise ValueError(f"edge {a}->{b} references a missing node")
    if a == b:
      raise ValueError(f"self-loop edge at node {a}")
    if a > b:
      raise ValueError(f"edge {a}->{b} violates program order; captures must be acyclic in program order")
    deps[b].append(a)
  return nodes, edges, deps


def _longest_path(deps: list[list[int]], durations: list[float]) -> float:
  """Longest path in the DAG (node durations, any source to any sink)."""
  longest = [0.0] * len(durations)
  for i in range(len(durations)):
    m = 0.0
    for p in deps[i]:
      if longest[p] > m: m = longest[p]
    longest[i] = m + durations[i]
  return max(longest)


def scan(dag: dict, *, same_group_only: bool = True, floor_us: float = 1.0,
         max_pairs: int = 2000, exact_pairs: int = 500) -> dict:
  nodes, edges, deps = _validate(dag)
  n = len(nodes)
  base_dur = [float(x["duration_us"]) for x in nodes]
  critical_path_us = _longest_path(deps, base_dur)

  # Transitive closure as integer bitsets: independence is no path either way.
  succ: list[list[int]] = [[] for _ in range(n)]
  for e in edges: succ[int(e["from"])].append(int(e["to"]))
  reach = [0] * n
  for i in range(n):
    seen: set[int] = set()
    stack = succ[i][:]
    while stack:
      j = stack.pop()
      if j in seen: continue
      seen.add(j)
      reach[i] |= 1 << j
      stack.extend(succ[j])

  classes = [classify(x) for x in nodes]
  supports = [i for i in range(n) if classes[i] == "support"]
  hosts = [i for i in range(n) if classes[i] == "host"]
  if not hosts:
    raise ValueError("DAG has no quant/flash host nodes; nothing can hide behind")

  def independent(a: int, b: int) -> bool:
    return not ((reach[a] >> b) & 1) and not ((reach[b] >> a) & 1)

  # Per-node full-hide critical-path recovery (one exact recomputation each).
  delta_cp: dict[int, float] = {}
  for s in supports:
    dur = base_dur.copy(); dur[s] = 0.0
    delta_cp[s] = critical_path_us - _longest_path(deps, dur)

  # Best partner per support node (largest hideable among independent hosts).
  best: dict[int, tuple[float, int]] = {}
  pairs: list[dict] = []
  for s in supports:
    d_s = base_dur[s]
    options = []
    for h in hosts:
      if not independent(s, h): continue
      if same_group_only and nodes[s]["group_id"] != nodes[h]["group_id"]: continue
      options.append((min(d_s, base_dur[h]), h))
    if not options: continue
    hide, h = max(options, key=lambda x: x[0])
    best[s] = (hide, h)
    dcp = delta_cp[s]
    for opt_hide, opt_h in sorted(options, key=lambda x: -x[0]):
      pairs.append({
        "support_id": s, "support_name": nodes[s]["name"], "support_group_id": nodes[s]["group_id"],
        "support_duration_us": round(d_s, 3),
        "host_id": opt_h, "host_name": nodes[opt_h]["name"], "host_group_id": nodes[opt_h]["group_id"],
        "host_family": family_of(nodes[opt_h]["name"]), "host_duration_us": round(base_dur[opt_h], 3),
        "same_group": nodes[s]["group_id"] == nodes[opt_h]["group_id"],
        "hideable_us": round(opt_hide, 3), "fully_hideable": bool(base_dur[opt_h] >= d_s),
        "delta_cp_full_hide_us": round(dcp, 3),
        "recovery_bound_us": round(min(dcp, opt_hide), 3),
      })
  pairs.sort(key=lambda r: (-r["hideable_us"], -r["support_duration_us"], r["support_id"]))

  # Exact pair recovery for the top rows: CP with this support reduced by its
  # hideable span.  The bound is proven for every row; exact rows recompute.
  for row in pairs[:exact_pairs]:
    s = row["support_id"]
    dur = base_dur.copy(); dur[s] = max(0.0, base_dur[s] - row["hideable_us"])
    row["recovery_us"] = round(critical_path_us - _longest_path(deps, dur), 3)

  # Greedy selection: repeatedly pick the pair with the largest exact recovery
  # on the CURRENT durations, until no candidate clears the floor.  Each support
  # node is claimed at most once (its kernel executes once).
  durs = base_dur[:]
  cp_now = critical_path_us
  claimed: set[int] = set()
  steps: list[dict] = []
  while True:
    cand = None
    for s in supports:
      if s in claimed or s not in best: continue
      if durs[s] <= 0: continue
      d_s = durs[s]
      partners = [h for h in hosts if independent(s, h) and
                  (not same_group_only or nodes[h]["group_id"] == nodes[s]["group_id"])]
      if not partners: continue
      max_dq = max(base_dur[h] for h in partners)
      hide = min(d_s, max_dq)
      # Recovery can never exceed the hidden span, so this bound is safe to
      # prune with.  Per-node delta_cp from the original graph is NOT safe
      # after earlier selections, so no other bound is used here.
      if hide < floor_us: continue
      dur = durs.copy(); dur[s] = max(0.0, d_s - hide)
      rec = cp_now - _longest_path(deps, dur)
      if rec >= floor_us and (cand is None or rec > cand[0]):
        cand = (rec, s, max(partners, key=lambda h: base_dur[h]))
    if cand is None: break
    rec, s, h = cand
    max_dq = base_dur[h]
    hide = min(durs[s], max_dq)
    durs[s] = max(0.0, durs[s] - hide)
    cp_now -= rec
    claimed.add(s)
    steps.append({"support_id": s, "support_name": nodes[s]["name"], "host_id": h,
                  "host_name": nodes[h]["name"], "host_family": family_of(nodes[h]["name"]),
                  "hideable_us": round(hide, 3),
                  "recovery_us": round(rec, 3), "remaining_critical_path_us": round(cp_now, 3)})

  # Per-population rows: host family buckets over the same-group pair set.
  population: dict[str, dict] = {}
  for fam in FAMILIES:
    rows = [r for r in pairs if r["host_family"] == fam]
    pop_s = {r["support_id"] for r in rows}
    if not rows:
      population[fam] = {"pair_count": 0, "support_count": 0, "containment_us": 0.0,
                         "ceiling_us": 0.0, "greedy_recovery_us": 0.0}
      continue
    dur = base_dur.copy()
    best_in_family: dict[int, float] = {}
    for r in rows:
      best_in_family[r["support_id"]] = max(best_in_family.get(r["support_id"], 0.0), r["hideable_us"])
    for s, hide in best_in_family.items():
      dur[s] = max(0.0, base_dur[s] - hide)
    greedy_sum = sum(st["recovery_us"] for st in steps if st["host_family"] == fam)
    population[fam] = {"pair_count": len(rows), "support_count": len(pop_s),
                       "containment_us": round(sum(r["hideable_us"] for r in rows), 3),
                       "ceiling_us": round(critical_path_us - _longest_path(deps, dur), 3),
                       "greedy_recovery_us": round(greedy_sum, 3)}

  # Ceilings: every independent support behind its best partner (co-schedulable),
  # and every support gone entirely (fusion-only bound, not a co-schedule claim).
  dur = base_dur.copy()
  for s, (hide, _) in best.items(): dur[s] = max(0.0, base_dur[s] - hide)
  co_schedule_ceiling_us = critical_path_us - _longest_path(deps, dur)
  dur = base_dur.copy()
  for s in supports: dur[s] = 0.0
  fusion_ceiling_us = critical_path_us - _longest_path(deps, dur)
  greedy_us = critical_path_us - cp_now
  verdict = "GPU_ELIGIBLE" if greedy_us >= PROMOTION_GATE_US else "CPU_NO_GO"

  return {
    "schema": SCHEMA,
    "capture_identity": {"node_count": n, "name_digest": hashlib.sha256(
      "\n".join(x["name"] for x in nodes).encode()).hexdigest()},
    "classification": {
      "support_nodes": len(supports),
      "support_duration_us": round(sum(base_dur[s] for s in supports), 3),
      "host_nodes": len(hosts),
      "host_q4k": sum(1 for h in hosts if nodes[h]["name"].startswith("q4k")),
      "host_q6k": sum(1 for h in hosts if nodes[h]["name"].startswith("q6k")),
      "host_flash": sum(1 for h in hosts if nodes[h]["name"].startswith("flash")),
    },
    "baseline": {"critical_path_us": round(critical_path_us, 3),
                 "serialized_us": round(sum(base_dur), 3),
                 "serialization_slack_us": round(sum(base_dur) - critical_path_us, 3)},
    "candidate_pairs": {"same_group_only": bool(same_group_only), "pair_count": len(pairs),
                        "support_count": len(best),
                        "containment_us": round(sum(r["hideable_us"] for r in pairs), 3),
                        "best_partner_containment_us": round(sum(hide for hide, _ in best.values()), 3),
                        "exact_recovery_rows": min(exact_pairs, len(pairs))},
    "pairs": pairs[:max_pairs],
    "per_node": [{"id": s, "name": nodes[s]["name"], "group_id": nodes[s]["group_id"],
                  "duration_us": round(base_dur[s], 3), "on_critical_path": bool(delta_cp[s] > 0),
                  "delta_cp_full_hide_us": round(delta_cp[s], 3),
                  "has_partner": s in best,
                  "best_hideable_us": round(best[s][0], 3) if s in best else 0.0,
                  "best_host_id": best[s][1] if s in best else None}
                 for s in supports],
    "per_population": population,
    "greedy": {"selection_floor_us": floor_us, "selected": steps,
               "selected_count": len(steps), "recovery_us": round(greedy_us, 3)},
    "ceiling": {"co_schedule_ceiling_us": round(co_schedule_ceiling_us, 3),
                "fusion_only_ceiling_us": round(fusion_ceiling_us, 3)},
    "gate_us": PROMOTION_GATE_US,
    "verdict": verdict,
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dag", required=True)
  ap.add_argument("--out")
  ap.add_argument("--allow-cross-group", action="store_true",
                  help="count pairs across graph launches (default: same-group only)")
  ap.add_argument("--floor-us", type=float, default=1.0)
  ap.add_argument("--max-pairs", type=int, default=2000)
  ap.add_argument("--exact-pairs", type=int, default=500)
  args = ap.parse_args()
  result = scan(load(args.dag), same_group_only=not args.allow_cross_group,
                floor_us=args.floor_us, max_pairs=args.max_pairs, exact_pairs=args.exact_pairs)
  text = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.out:
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
