#!/usr/bin/env python3
"""Generate the next untried candidate for the pure-search decode loop.

This is the GENERATE + PRUNE step the loop was missing (it was a refutation cycle, not a search). It reads the
declared search space + durable ledger (bench/qk-search-spaces/decode_attention_loop_search_space.json),
enumerates one-factor-at-a-time deltas from the baseline best-stack in priority order, prunes anything already in
the ledger (and, with --prune-predicted, anything the manifest predicts refuted), and emits the next untried
candidate as a ready-to-run env-flag string -- or SEARCH_SPACE_EXHAUSTED when the space is genuinely covered.

So the loop's NO_NEW_LEVER fires on real exhaustion, not when a human runs out of ideas.

Run: PYTHONPATH=. python3 extra/qk_pure_search_next_candidate.py [--prune-predicted] [--pairs]
"""
from __future__ import annotations
import json, pathlib, argparse, itertools

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE = ROOT / "bench/qk-search-spaces/decode_attention_loop_search_space.json"

def _key(delta: dict) -> str:
  return ",".join(f"{k}={v}" for k, v in sorted(delta.items()))

def _flags(baseline: dict, delta: dict) -> str:
  merged = {**baseline, **delta}
  return " ".join(f"{k}={v}" for k, v in merged.items())

def candidates(space: dict, pairs: bool):
  """One-factor-at-a-time deltas from baseline, priority-ordered; then (optionally) pairs of cheap knobs."""
  axes = sorted(space.get("axes", []), key=lambda a: a.get("priority", 99))
  singles = []
  for a in axes:
    for v in a["values"]:
      singles.append({"delta": {a["flag"]: v}, "axis": a, "kind": "single"})
  out = list(singles)
  if pairs:
    cheap = [a for a in axes if a.get("cost") == "cheap"]
    for a, b in itertools.combinations(cheap, 2):
      for va, vb in itertools.product(a["values"], b["values"]):
        out.append({"delta": {a["flag"]: va, b["flag"]: vb}, "axis": None, "kind": "pair"})
  return out

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--prune-predicted", action="store_true", help="also skip candidates the manifest predicts refuted")
  ap.add_argument("--pairs", action="store_true", help="after singles, enumerate pairs of cheap knobs")
  args = ap.parse_args()

  if not SPACE.exists():
    print(json.dumps({"verdict": "SEARCH_SPACE_MISSING", "path": str(SPACE.relative_to(ROOT))})); return 2
  space = json.loads(SPACE.read_text())
  baseline = space["baseline_stack"]
  tried = {row["candidate"] for row in space.get("ledger", [])}

  cands = candidates(space, args.pairs)
  total = len(cands)
  pruned_predicted = []
  for c in cands:
    k = _key(c["delta"])
    if k in tried: continue
    if args.prune_predicted and c.get("axis") and c["axis"].get("predicted"):
      pruned_predicted.append(k); continue
    out = {
      "verdict": "NEXT_CANDIDATE",
      "candidate": k,
      "kind": c["kind"],
      "delta": c["delta"],
      "env_flags": _flags(baseline, c["delta"]),
      "hypothesis": (c["axis"] or {}).get("hypothesis"),
      "predicted": (c["axis"] or {}).get("predicted"),
      "tried_count": len(tried),
      "remaining_after_this": sum(1 for cc in cands if _key(cc["delta"]) not in tried) - 1,
    }
    print(json.dumps(out, indent=2)); return 0

  print(json.dumps({
    "verdict": "SEARCH_SPACE_EXHAUSTED",
    "tried_count": len(tried), "candidate_total": total,
    "pruned_predicted": pruned_predicted,
    "note": "every priority-ordered candidate is in the ledger -- NO_NEW_LEVER is now genuine. Escalate to the next LAYER (more workgroups + cheaper combine) or expand the search space (add axes/values to the manifest).",
  }, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
