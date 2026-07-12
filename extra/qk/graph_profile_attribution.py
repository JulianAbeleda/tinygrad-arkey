"""Summarize HCQGraph timestamp exports without synchronizing kernels.

Names are intentionally classified conservatively: only generated names that are
already tied to an exact candidate role are labeled proven; all other dispatches
remain unknown until a route census supplies a semantic join.
"""
import argparse, collections, json

PROVEN_NAMES = {
  # These identities are the generated buffer2 kernels used by the four-role set.
  "E_4_8_32_4_2_2_2_2_4_2_127_2": "attn_kv",
  "E_4_32_32_4_2_2_2_2_4_2_127_2": "candidate_dense_ambiguous",
  "E_4_32_32_4_2_2_2_2_4_2_383_2": "ffn_gate_up",
}

def summarize(paths):
  buckets = collections.defaultdict(lambda: {"count": 0, "duration": 0.0, "names": collections.Counter()})
  for path in paths:
    for line in open(path, encoding="utf-8"):
      for row in json.loads(line)["entries"]:
        name = row["name"]
        role = PROVEN_NAMES.get(name, "unknown")
        b = buckets[role]; b["count"] += 1; b["duration"] += float(row["duration"]); b["names"][name] += 1
  return {k: {"count": v["count"], "duration": v["duration"], "names": dict(v["names"])} for k,v in sorted(buckets.items())}

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--json", action="store_true")
  out = summarize(ap.parse_args().paths)
  print(json.dumps(out, indent=2, sort_keys=True) if ap.parse_args().json else "\n".join(f"{k}: {v['count']} dispatches, {v['duration']:.3f} ticks" for k,v in out.items()))
