#!/usr/bin/env python3
"""E2 decisive-experiment simulator: dependency-DAG critical path and list
schedules for HCQ graph profile captures (see
docs/task_workflow/input/nv-decode-parity-e1e3-measurement-scope-20260803.md,
section 2).

Input is the JSONL written by tinygrad/runtime/graph/hcq.py
`graph_profile_payload` when PROFILE=1 and HCQ_GRAPH_PROFILE_JSON are set.
Each line is one graph capture:

  {"schema": "tinygrad.hcq_graph_profile.v1",
   "entries": [{"device", "name", "metadata", "start", "end", "duration",
                "st_id", "en_id"}, ...],
   "deps": [[dep-entry-index, ...], ...]}

Timestamps and durations are microseconds; "deps" entries are 0-based indices
into "entries". Nodes with missing or absent dependency fields are treated as
independent. Nodes are grouped by explicit graph-group membership fields if
the records carry them ("graph_group_id", "graph_group", "group",
"graph_id"); otherwise each JSONL line (one capture) is one group. For each
group, and for the whole capture with all groups merged (no cross-line edges),
the simulator reports the serialized node-sum span, the unlimited-resource
critical path, deterministic list schedules on 2 and 3 queues (ready set,
static priority by longest remaining tail, earliest-available queue, no
preemption), overlap savings vs serialized, and which node classes overlap.

Usage:
  python3 extra/llm_research/decode/dag_critical_path_sim.py <input.jsonl> [--out X.json]
"""
from __future__ import annotations

import argparse, json, sys

GROUP_FIELDS = ("graph_group_id", "graph_group", "group", "graph_id")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def classify(name):
    low = name.lower()
    if "flash" in low:
        return "flash"
    if "q4k" in low or "q6k" in low:
        return "gemv"  # GEMV-class q4k/q6k
    if "1187" in name:
        return "scatter"  # five-lever scatter kernels carry the 1187 token count
    if "kv" in low:
        return "kv"
    if low.startswith("r_"):
        return "rmsnorm"
    if low.startswith("e_"):
        return "residual"
    return "other"


def build_nodes(record):
    entries = record.get("entries") or []
    rec_deps = record.get("deps")
    n = len(entries)
    nodes = []
    explicit = False
    for idx, e in enumerate(entries):
        dur = _f(e.get("duration"))
        if dur <= 0:
            s, en = e.get("start"), e.get("end")
            if s is not None and en is not None:
                dur = max(0.0, _f(en) - _f(s))
        deps = e.get("deps")
        if not isinstance(deps, list):
            deps = rec_deps[idx] if isinstance(rec_deps, list) and idx < len(rec_deps) else None
        deps = deps if isinstance(deps, list) else []
        valid = []
        for d in deps:
            if isinstance(d, (int, float)) and not isinstance(d, bool):
                di = int(d)
                if 0 <= di < n and di != idx and di not in valid:
                    valid.append(di)
        gfield = gkey = None
        for f in GROUP_FIELDS:
            if e.get(f) is not None:
                gfield, gkey = f, e[f]
                break
        if gfield is None:
            for f in GROUP_FIELDS:
                if record.get(f) is not None:
                    gfield, gkey = f, record[f]
                    break
        explicit |= gfield is not None
        nodes.append({"id": idx, "name": str(e.get("name", "node-%d" % idx)),
                      "duration": dur, "deps": valid, "metadata": e.get("metadata"),
                      "device": e.get("device"), "group_field": gfield,
                      "group_key": gkey})
    return nodes, explicit


def group_nodes(records):
    groups = {}
    order = []
    any_explicit = False
    field_used = None
    for rec in records:
        nodes, explicit = build_nodes(rec)
        any_explicit |= explicit
        lineno = rec["_lineno"]
        for node in nodes:
            if node["group_field"] is not None:
                key = str(node["group_key"])
                field_used = node["group_field"]
            else:
                key = "line-%04d" % lineno
            node["_lineno"] = lineno
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(node)
    dropped = 0
    for key in order:
        nodes = groups[key]
        pos = {(n["_lineno"], n["id"]): i for i, n in enumerate(nodes)}
        for i, node in enumerate(nodes):
            deps = []
            for d in node["deps"]:
                t = pos.get((node["_lineno"], d))
                if t is None:
                    dropped += 1  # dep target outside this group: independent
                elif t != i and t not in deps:
                    deps.append(t)
            node["deps"] = deps
    if any_explicit:
        grouping = "explicit field '%s'" % field_used
    else:
        grouping = "per-line capture (no explicit graph-group field)"
    return groups, order, grouping, dropped


def union_nodes(records):
    """All nodes in file order with deps remapped to global indices."""
    nodes = []
    pos = {}
    for rec in records:
        rec_nodes, _ = build_nodes(rec)
        lineno = rec["_lineno"]
        for node in rec_nodes:
            node["_lineno"] = lineno
            pos[(lineno, node["id"])] = len(nodes)
            nodes.append(node)
    dropped = 0
    for node in nodes:
        deps = []
        for d in node["deps"]:
            t = pos.get((node["_lineno"], d))
            if t is None:
                dropped += 1
            elif t not in deps:
                deps.append(t)
        node["deps"] = deps
    return nodes, dropped


def compute_est(nodes):
    n = len(nodes)
    durs = [node["duration"] for node in nodes]
    est = [0.0] * n
    state = [0] * n

    def dfs(i):
        if state[i] == 2:
            return est[i]
        if state[i] == 1:
            return 0.0  # cycle: treat edge as satisfied at 0
        state[i] = 1
        best = 0.0
        for d in nodes[i]["deps"]:
            v = dfs(d) + durs[d]
            if v > best:
                best = v
        est[i] = best
        state[i] = 2
        return best

    for i in range(n):
        dfs(i)
    return est


def compute_tails(nodes):
    n = len(nodes)
    durs = [node["duration"] for node in nodes]
    children = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        for d in node["deps"]:
            children[d].append(i)
    tail = [0.0] * n
    state = [0] * n

    def dfs(i):
        if state[i] == 2:
            return tail[i]
        if state[i] == 1:
            return durs[i]
        state[i] = 1
        best = 0.0
        for c in children[i]:
            v = dfs(c)
            if v > best:
                best = v
        tail[i] = durs[i] + best
        state[i] = 2
        return tail[i]

    for i in range(n):
        dfs(i)
    return tail


def list_schedule(nodes, tails, queues):
    """Ready set + static priority by longest remaining tail; earliest queue."""
    n = len(nodes)
    durs = [node["duration"] for node in nodes]
    children = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        for d in node["deps"]:
            children[d].append(i)
    need = [len(node["deps"]) for node in nodes]
    est = [0.0] * n
    start = [0.0] * n
    end = [0.0] * n
    q_free = [0.0] * queues
    ready = [i for i in range(n) if need[i] == 0]
    pending = set(range(n))
    while pending:
        if not ready:
            ready.append(min(pending))  # defensive: cycles/odd deps
        pick = max(ready, key=lambda i: (tails[i], -est[i], -i))
        q = min(range(queues), key=lambda j: (q_free[j], j))
        s = max(q_free[q], est[pick])
        start[pick] = s
        end[pick] = s + durs[pick]
        q_free[q] = end[pick]
        ready.remove(pick)
        pending.discard(pick)
        for c in children[pick]:
            if end[pick] > est[c]:
                est[c] = end[pick]
            need[c] -= 1
            if need[c] == 0:
                ready.append(c)
    return (max(end) if n else 0.0), start, end


def overlap_pairs(nodes, start, end):
    counts = {}
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if start[i] < end[j] and start[j] < end[i]:
                a, b = classify(nodes[i]["name"]), classify(nodes[j]["name"])
                if a > b:
                    a, b = b, a
                counts[(a, b)] = counts.get((a, b), 0) + 1
    return [[a, b, c] for (a, b), c in sorted(counts.items())]


def compute_metrics(nodes):
    n = len(nodes)
    durs = [node["duration"] for node in nodes]
    est = compute_est(nodes)
    cp = max((est[i] + durs[i] for i in range(n)), default=0.0)
    tails = compute_tails(nodes)
    serial = sum(durs)
    classes = {}
    for node in nodes:
        c = classify(node["name"])
        classes[c] = classes.get(c, 0) + 1
    cp_start, cp_end = est, [est[i] + durs[i] for i in range(n)]
    overlap = {"unlimited": overlap_pairs(nodes, cp_start, cp_end)}
    schedules = {}
    for q in (2, 3):
        span, start, end = list_schedule(nodes, tails, q)
        schedules["%dq" % q] = {"span_us": round(span, 3),
                                "start": [round(v, 3) for v in start],
                                "end": [round(v, 3) for v in end]}
        overlap["%dq" % q] = overlap_pairs(nodes, start, end)
    savings_us = {"critical_path": round(serial - cp, 3),
                  "2q": round(serial - schedules["2q"]["span_us"], 3),
                  "3q": round(serial - schedules["3q"]["span_us"], 3)}
    savings_pct = {k: (round(100.0 * v / serial, 2) if serial else 0.0)
                   for k, v in savings_us.items()}
    cp_nodes = sorted(range(n), key=lambda i: (est[i], i))
    return {"node_count": n,
            "serialized_span_us": round(serial, 3),
            "critical_path_us": round(cp, 3),
            "schedule_2q_us": schedules["2q"]["span_us"],
            "schedule_3q_us": schedules["3q"]["span_us"],
            "savings_us": savings_us,
            "savings_pct": savings_pct,
            "node_classes": dict(sorted(classes.items())),
            "overlapping_classes": {k: v for k, v in overlap.items()},
            "critical_path_node_ids": cp_nodes}


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            rec["_lineno"] = lineno
            records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="JSONL of HCQ graph profile captures")
    ap.add_argument("--out", help="write JSON result to this path")
    args = ap.parse_args()
    try:
        with open(args.input, encoding="utf-8"):
            pass
    except FileNotFoundError:
        print("input file not found: %s" % args.input, file=sys.stderr)
        sys.exit(1)
    records = load_records(args.input)
    if not records:
        print("no graph profile records found in %s" % args.input, file=sys.stderr)
        sys.exit(1)
    sys.setrecursionlimit(max(10000, 4 * sum(len(r.get("entries") or []) for r in records) + 100))

    groups, order, grouping, dropped = group_nodes(records)
    result = {"schema": "dag_critical_path_sim.v1", "input": args.input,
              "grouping": grouping, "groups": {}}
    for key in order:
        result["groups"][key] = compute_metrics(groups[key])

    union, union_dropped = union_nodes(records)
    total = compute_metrics(union)
    total["cross_group_edges"] = dropped + union_dropped
    total["cross_group_overlap_legal"] = total["cross_group_edges"] == 0
    result["total"] = total

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        for key in order:
            g = result["groups"][key]
            print("group %s: n=%d serial=%.3f cp=%.3f 2q=%.3f 3q=%.3f us"
                  % (key, g["node_count"], g["serialized_span_us"],
                     g["critical_path_us"], g["schedule_2q_us"], g["schedule_3q_us"]))
        t = result["total"]
        print("total: n=%d serial=%.3f cp=%.3f 2q=%.3f 3q=%.3f us, "
              "cross_group_edges=%d" % (t["node_count"], t["serialized_span_us"],
                                        t["critical_path_us"], t["schedule_2q_us"],
                                        t["schedule_3q_us"], t["cross_group_edges"]))
        print("wrote %s" % args.out)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
