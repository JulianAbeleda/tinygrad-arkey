#!/usr/bin/env python3
"""Canonical policy/headline consistency guardrail (docs hygiene; no GPU, no kernels).

Scans ONLY the canonical START-HERE doc set and fails (exit 1) if a current doc re-opens a CLOSED question:
  - a bare `87.6` quoted with no nearby context (ctx / ms / coincidence / never-quote ...);
  - an OPEN `PREFILL_V2=auto` global-default "owner call" (the decision is OFF, decided 2026-06-21);
  - an affirmative "flip global PREFILL_V2=auto" proposal (not the negation);
  - `87` presented AS the decode headline (the headline is the curve / ~67% llama, not the ctx0 peak);
  - bounded decode fusion presented as CURRENT implementation work (it is closed).

Precision: context is checked in a +/-1 line WINDOW (handles wrapped prose), with a broad explanatory allow-list;
lines that DESCRIBE this guardrail/audit (self-referential meta) are skipped. Scope/historical/per-phase docs are
intentionally NOT scanned. Run: PYTHONPATH=. python3 extra/qk_policy_consistency_check.py
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CANONICAL = [
  "docs/README.md", "bench/README.md",
  "docs/current-project-state-handoff-20260621.md",
  "docs/decode-prefill-headline-reconciliation-result-20260621.md",
  "docs/prefill-policy-integration-result-20260620.md",
]
# explanatory context that makes an `87.6` mention legitimate (it is being explained/closed, not quoted as headline)
_87_OK = ("ctx", "ms", "coincid", "ambig", "11.4", "contextual", "empty-kv", "empty kv", "never quote", "ms/token",
          "ms per", "numeric", "provenance", "non-headline", "two real", "opposite", "reconcil", "sources", "trap",
          "separate", "bare", "quote", "never", "85", "86", "87.0", "87.5", "87.62", "87.9", "us\b")
# context that makes a PREFILL_V2/flip/owner-call line a DECISION (off), not an open question
_DECIDED_OK = ("stays off", "stay off", "decided", "do not flip", "don't flip", "not flip", "off (decided", "opt-in",
               "no decode benefit", "stays **off", "default off", "remains off", "resident during decode")
# headline #4: context that makes an `87` line legitimate (it's saying the headline is NOT 87)
_HEAD_OK = ("curve", "~67", "67%", "non-headline", "ambig", "never", "contextual", "not the", "peak", "coincid")
# self-referential meta lines that DESCRIBE the guardrail/audit (allowed to quote the banned phrases)
_META = ("guardrail", "consistency check", "consistency: ", "re-open", "re-opens", "reopen", "stale reference",
         "stale ref", "banned", "qk_policy_consistency", "checker", "this file fails", "exit 1", "audit")


def scan(rel: str, lines: list[str]) -> list[str]:
  out = []
  low = [l.lower() for l in lines]
  for i, line in enumerate(lines):
    l = low[i]
    win = " ".join(low[max(0, i-1): i+2])  # +/-1 line context window
    if any(m in win for m in _META):  # skip self-referential meta/guardrail-description lines (windowed)
      continue
    if re.search(r"87\.6\b", line) and not any(w in win for w in _87_OK):
      out.append((i+1, "bare `87.6` without context", line))
    if "owner call" in l and ("prefill_v2" in l or "prefill default" in l or "global default" in l) \
       and not any(w in win for w in _DECIDED_OK):
      out.append((i+1, "open `PREFILL_V2` owner call (decision is OFF)", line))
    if ("flip" in l and "prefill_v2" in l and "auto" in l) and not any(w in win for w in _DECIDED_OK):
      out.append((i+1, "affirmative 'flip global PREFILL_V2=auto'", line))
    if "decode headline" in l and re.search(r"\b8[5-9]\b|87\.", line) and not any(w in win for w in _HEAD_OK):
      out.append((i+1, "`87` presented as the decode headline", line))
    if re.search(r"bounded.*fusion|micro-?fusion", l) \
       and any(w in l for w in ("current", "next work", "todo", "implement now", "tactical", "in progress")) \
       and not any(w in win for w in ("closed", "no-go", "no go", "exhausted", "refuted", "historical", "superseded")):
      out.append((i+1, "bounded decode fusion as current work (it is closed)", line))
  return out


def main() -> int:
  violations = []
  for rel in CANONICAL:
    p = ROOT / rel
    if not p.exists(): violations.append(f"{rel}:0  MISSING canonical doc"); continue
    for n, why, line in scan(rel, p.read_text(errors="ignore").splitlines()):
      violations.append(f"{rel}:{n}  {why}\n      > {line.strip()[:140]}")
  if violations:
    print(f"POLICY CONSISTENCY: FAIL ({len(violations)} issue(s)) — a canonical doc re-opens a closed question:\n")
    print("\n".join(violations))
    return 1
  print(f"POLICY CONSISTENCY: PASS — {len(CANONICAL)} canonical docs clean.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
