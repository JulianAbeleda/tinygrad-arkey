"""Summarize HCQGraph timestamp exports without synchronizing kernels.

Names are intentionally classified conservatively: only generated names that are
already tied to an exact candidate role are labeled proven. Everything else falls
back to the explicit semantic role tag attached at the Tensor call site (see
tinygrad.tensor.role_metadata and its use in tinygrad/llm/model.py), and only
dispatches with neither a proven identity nor a role tag remain unknown.
"""
import argparse, collections, json

PROVEN_NAMES = {
  # These identities are the generated buffer2 kernels used by the four-role set.
  "E_4_8_32_4_2_2_2_2_4_2_127_2": "attn_kv",
  "E_4_32_32_4_2_2_2_2_4_2_127_2": "candidate_dense_ambiguous",
  "E_4_32_32_4_2_2_2_2_4_2_383_2": "ffn_gate_up",
}

def _busy_union(intervals):
  """Merged length of [start, end) tick intervals, without double-counting overlap.

  Naive per-dispatch duration sums over-count concurrent GPU work; this union is the
  honest non-overlapping busy total. Reported alongside the sum so the pair brackets
  device busy-time (sum = overlap-inflated upper, union = non-overlapping lower)."""
  spans = sorted((s, e) for s, e in intervals if e > s)
  if not spans: return 0.0
  total, cur_s, cur_e = 0.0, *spans[0]
  for s, e in spans[1:]:
    if s > cur_e: total += cur_e - cur_s; cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
  return total + (cur_e - cur_s)

def summarize(paths):
  buckets = collections.defaultdict(lambda: {"count": 0, "duration": 0.0, "names": collections.Counter()})
  device_spans = collections.defaultdict(list)
  for path in paths:
    for line in open(path, encoding="utf-8"):
      for row in json.loads(line)["entries"]:
        name = row["name"]
        metadata = row.get("metadata") or {}
        # resolution order: proven candidate identity (exact kernel-name join to a GEMM role) first, else
        # the semantic_op role tag attached at the Tensor call site (rms_norm/rope/attn_score/... -- see
        # tinygrad.tensor.role_metadata), else unknown.
        role = PROVEN_NAMES.get(name) or metadata.get("semantic_op") or "unknown"
        b = buckets[role]; b["count"] += 1; b["duration"] += float(row["duration"]); b["names"][name] += 1
        device_spans[row["device"]].append((float(row["start"]), float(row["end"])))
  roles = {k: {"count": v["count"], "duration": v["duration"], "names": dict(v["names"])} for k,v in sorted(buckets.items())}
  device_busy = {dev: {"sum_ticks": sum(e - s for s, e in spans if e > s), "union_ticks": _busy_union(spans)}
                 for dev, spans in sorted(device_spans.items())}
  return {"roles": roles, "device_busy": device_busy}

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--json", action="store_true")
  out = summarize(ap.parse_args().paths)
  if ap.parse_args().json: print(json.dumps(out, indent=2, sort_keys=True))
  else:
    print("\n".join(f"{k}: {v['count']} dispatches, {v['duration']:.3f} ticks" for k,v in out["roles"].items()))
    print("\n".join(f"{dev} busy: sum {d['sum_ticks']:.3f} ticks, union {d['union_ticks']:.3f} ticks (overlap "
                     f"{d['sum_ticks']-d['union_ticks']:.3f})" for dev,d in out["device_busy"].items()))
