#!/usr/bin/env python3
"""Single source of truth for pure-search-loop candidate selection (GENERATE + PRUNE) and ledger I/O (REMEMBER).

The loop (`.claude/loop.md`) and `/pure-search-loop` MUST both pick candidates from here -- never from human hints
or the audit's prose `next_actions` (those are advisory only). This makes `SEARCH_SPACE_EXHAUSTED` mean the declared
active space is *actually* exhausted, not that a human ran out of ideas.

Inputs (split: immutable space vs mutable ledger):
  - space  : bench/qk-search-spaces/decode_attention_loop_search_space.json   (declared axes + baseline ONLY)
  - ledger : bench/qk-pure-search-loop/decode_attention_loop_ledger.jsonl      (append-only outcomes, one JSON/line)

Active space = one-factor singles (axis x value) + pairs of cheap knobs (pairs ON by default -- they ARE part of
Level-1 exhaustion). Candidates already in the ledger are pruned.

Usage:
  PYTHONPATH=. python3 extra/qk_pure_search_next_candidate.py            # emit next candidate / SEARCH_SPACE_EXHAUSTED
  PYTHONPATH=. python3 extra/qk_pure_search_next_candidate.py --no-pairs # singles only
  PYTHONPATH=. python3 extra/qk_pure_search_next_candidate.py --record '{"candidate":"DECODE_STAGE_COALESCE=2","outcome":"REFUTED_NO_SLOPE","gate":"isolated +x%","why":"..."}'
"""
from __future__ import annotations
import json, pathlib, argparse, itertools

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE = ROOT / "bench/qk-search-spaces/decode_attention_loop_search_space.json"
LEDGER = ROOT / "bench/qk-pure-search-loop/decode_attention_loop_ledger.jsonl"

# Outcomes the loop may record. PROMOTABLE is reserved for W==D + token-match (never local gates).
VALID_OUTCOMES = {"FAIL_CORRECTNESS", "REFUTED_OCCUPANCY", "REFUTED_NO_SLOPE", "LOCAL_PASS_WD_REQUIRED",
                  "REFUTED_WD", "PROMOTABLE"}

def _key(delta: dict) -> str:
  return ",".join(f"{k}={v}" for k, v in sorted(delta.items()))

def _flags(baseline: dict, delta: dict) -> str:
  return " ".join(f"{k}={v}" for k, v in {**baseline, **delta}.items())

def load_space() -> dict:
  return json.loads(SPACE.read_text())

def load_ledger_keys() -> tuple[set[str], int]:
  """Return (set of candidate keys seen, total ledger lines)."""
  if not LEDGER.exists(): return set(), 0
  keys, n = set(), 0
  for line in LEDGER.read_text().splitlines():
    line = line.strip()
    if not line: continue
    n += 1
    try: keys.add(json.loads(line)["candidate"])
    except Exception: pass
  return keys, n

def active_candidates(space: dict, include_pairs: bool) -> list[dict]:
  """Priority-ordered one-factor singles, then (default) pairs of cheap knobs. This IS the declared active space.

  An axis may carry `enable` (extra flags that must be set for it to take effect, e.g. a topology axis enabling
  DECODE_ATTN_BLOCK_TILE_FIXED_S=1) and `requires_wd` (its cost is in-model only -> the loop must gate it with W==D,
  not isolated timing). Topology axes are singles only (they don't pair with knobs)."""
  axes = sorted(space.get("axes", []), key=lambda a: a.get("priority", 99))
  out = [{"delta": {**a.get("enable", {}), a["flag"]: v}, "axis": a, "kind": a.get("kind", "single")}
         for a in axes for v in a["values"]]
  if include_pairs:
    cheap = [a for a in axes if a.get("cost") == "cheap"]   # only cheap KNOBS pair; topology axes stay single
    for a, b in itertools.combinations(cheap, 2):
      for va, vb in itertools.product(a["values"], b["values"]):
        out.append({"delta": {a["flag"]: va, b["flag"]: vb}, "axis": None, "kind": "pair"})
  return out

def record(line_json: str) -> int:
  obj = json.loads(line_json)
  if "candidate" not in obj or obj.get("outcome") not in VALID_OUTCOMES:
    print(json.dumps({"error": "record needs candidate + valid outcome", "valid_outcomes": sorted(VALID_OUTCOMES)}))
    return 2
  LEDGER.parent.mkdir(parents=True, exist_ok=True)
  with LEDGER.open("a") as f: f.write(json.dumps(obj) + "\n")
  print(json.dumps({"recorded": obj, "ledger": str(LEDGER.relative_to(ROOT))}))
  return 0

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--no-pairs", action="store_true", help="singles only (default: include pairs of cheap knobs)")
  ap.add_argument("--record", metavar="JSON", help="append an outcome line to the ledger and exit")
  ap.add_argument("--failed-rows", metavar="r1,r2", default=None,
                  help="parity-closure mode: only emit a candidate whose axis targets_delta is in this set of "
                       "FAILED parity rows (from qk_owned_oracle_parity_audit.py searchable_failed_rows). No "
                       "candidate may run unless it targets a failed row.")
  args = ap.parse_args()
  failed_rows = set(x.strip() for x in args.failed_rows.split(",") if x.strip()) if args.failed_rows else None

  if not SPACE.exists():
    print(json.dumps({"verdict": "SEARCH_SPACE_MISSING", "path": str(SPACE.relative_to(ROOT))})); return 2
  if args.record is not None:
    return record(args.record)

  space = load_space()
  baseline = space["baseline_stack"]
  include_pairs = not args.no_pairs
  cands = active_candidates(space, include_pairs)
  active_keys = {_key(c["delta"]) for c in cands}
  tried_keys, ledger_lines = load_ledger_keys()

  tried_in_space = active_keys & tried_keys
  historical = tried_keys - active_keys     # refutations recorded that are NOT in the current declared space
  counts = {
    "tried_in_space_count": len(tried_in_space),
    "remaining_in_space_count": len(active_keys - tried_keys),
    "active_space_total": len(active_keys),
    "historical_refutations_count": len(historical),
    "include_pairs": include_pairs,
    "ledger_lines": ledger_lines,
  }

  for c in cands:                            # priority order (singles before pairs)
    k = _key(c["delta"])
    if k in tried_keys: continue
    ax = c["axis"] or {}
    # parity-closure gate: in --failed-rows mode, ONLY a candidate that targets a FAILED parity row may run.
    if failed_rows is not None and ax.get("targets_delta") not in failed_rows: continue
    requires_wd = bool(ax.get("requires_wd"))
    print(json.dumps({
      "verdict": "NEXT_CANDIDATE", "candidate": k, "kind": c["kind"], "delta": c["delta"],
      "env_flags": _flags(baseline, c["delta"]),
      # owned-oracle reconstruction: every candidate must target a NAMED owned-vs-generated delta (taxonomy:
      # bench/qk-search-spaces/owned_delta_taxonomy.json). A candidate with no targets_delta is a SEARCH_SPACE_BUG
      # smell -- knobs may only be searched when the auditor predicts which delta they move.
      "targets_delta": ax.get("targets_delta"),
      "blocker_kind": ax.get("blocker_kind"),
      "requires_wd": requires_wd,
      "gate": "W==D (cost is in-model only; isolated timing MISLEADS)" if requires_wd else "isolated-then-W==D",
      "hypothesis": ax.get("hypothesis"), "predicted": ax.get("predicted"),
      **counts,
    }, indent=2))
    return 0

  verdict = ("NO_UNTRIED_CANDIDATE_TARGETS_A_FAILED_ROW" if failed_rows is not None else "SEARCH_SPACE_EXHAUSTED")
  note = ("no untried candidate targets a failed parity row " + str(sorted(failed_rows)) +
          " -> SEARCH_SPACE_BUG (the failed rows have no searchable axis, or their axes are exhausted): improve "
          "the search space or escalate to an instrumentation/primitive gap, do NOT loosen the parity gate."
          if failed_rows is not None else
          "every candidate in the declared active space (singles" + ("+pairs" if include_pairs else "") +
          ") is in the ledger -> exhaustion is genuine.")
  print(json.dumps({"verdict": verdict, "failed_rows_filter": sorted(failed_rows) if failed_rows else None,
                    "note": note, **counts}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
