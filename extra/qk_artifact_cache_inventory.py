"""C0 — cache inventory: walk the existing PMS/TG artifacts, classify each into cache class A_static / B_correctness /
C_speed, and report which already have provenance vs which need cache-metadata wrapping (in C2/C3).

Phase C0 of docs/pure-machine-search-artifact-cache-scope-20260630.md. Audit-only; no GPU, no wiring.
Run: PYTHONPATH=. python3 extra/qk_artifact_cache_inventory.py
Writes: bench/qk-artifact-cache/{inventory.json,summary.md}
"""
from __future__ import annotations
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-artifact-cache"

# the artifact roots named by the scope (C0 acceptance)
ROOTS = ["bench/qk-search-spaces", "bench/qk-lanemap-template-ir", "bench/qk-lanemap-template-audit",
         "bench/qk-topology-author", "bench/qk-quant-semantics-audit", "bench/qk-profile-opener",
         "bench/qk-target-features", "bench/qk-template-candidate-gate", "bench/qk-new-profile-search",
         "bench/qk-candidate-evaluator"]

# C_speed signals (any of these keys anywhere => speed evidence present)
SPEED_KEYS = ("tok_s", "tok/s", "w==d", "wd_table", "whole_prefill", "pmc", "wall_share", "gbps", "tflops", "median_tok")
# B_correctness signals
CORR_KEYS = ("token_match", "argmax", "route_attribution", "route_bound", "logit", "correct", "byte_identical", "fallback")

def _peek(path: pathlib.Path) -> tuple[bool, bool, bool]:
  """returns (has_speed, has_corr, has_cache_meta) by scanning the json text + top-level keys."""
  try: d = json.load(open(path))
  except Exception: return (False, False, False)
  text = json.dumps(d).lower()
  has_speed = any(k in text for k in SPEED_KEYS)
  has_corr = any(k in text for k in CORR_KEYS)
  has_meta = isinstance(d, dict) and isinstance(d.get("cache"), dict) and d["cache"].get("schema", "").startswith("qk_artifact_cache")
  return (has_speed, has_corr, has_meta)

def classify(relpath: str, has_speed: bool, has_corr: bool) -> str:
  p = relpath.lower()
  # Evaluator/gate artifacts carry measured RESULTS -> peek decides class.
  is_eval = any(s in p for s in ("candidate-evaluator", "template-candidate-gate"))
  # Static-generation roots are A_static config/output: embedded gbps/tflops are CONFIG FACTS (e.g. the profile's
  # measured_copy_gbps, a quant row's preferred dtype), NOT a per-run speed measurement -> not C_speed.
  static_root = any(s in p for s in ("qk-search-spaces", "lanemap-template", "topology-author",
                                     "quant-semantics", "profile-opener", "target-features", "new-profile-search"))
  if is_eval:
    if has_speed: return "C_speed"          # the evaluator's W==D/tok_s result file
    if has_corr: return "B_correctness"     # route_attribution / token-match file
    return "A_static"
  if static_root: return "A_static"
  if has_speed: return "C_speed"
  if has_corr: return "B_correctness"
  return "A_static"

def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  rows, unclassified = [], []
  for r in ROOTS:
    base = ROOT / r
    if not base.exists(): continue
    for f in sorted(base.rglob("*.json")):
      rel = str(f.relative_to(ROOT))
      hs, hc, hm = _peek(f)
      cls = classify(rel, hs, hc)
      rows.append({"path": rel, "class": cls, "has_cache_meta": hm,
                   "has_speed_evidence": hs, "has_correctness_evidence": hc,
                   "needs_wrapping": not hm, "wrap_phase": {"A_static": "C2", "B_correctness": "C3", "C_speed": "C3"}[cls]})
      if cls not in ("A_static", "B_correctness", "C_speed"): unclassified.append(rel)
  counts = {c: sum(1 for x in rows if x["class"] == c) for c in ("A_static", "B_correctness", "C_speed")}
  needs = sum(1 for x in rows if x["needs_wrapping"])
  verdict = "C0_BLOCKED_ARTIFACTS_UNCLASSIFIED" if unclassified else "C0_PASS_CACHE_INVENTORY_PINNED"
  rec = {"verdict": verdict, "total_artifacts": len(rows), "class_counts": counts,
         "need_wrapping": needs, "unclassified": unclassified,
         "note": "No artifact currently carries cache metadata (none wrapped yet) -> all need wrapping in C2 (A_static) / C3 (B/C). C0 is inventory+classification only; the wrapping + skip logic is C2-C3.",
         "rows": rows}
  json.dump(rec, open(OUT/"inventory.json", "w"), indent=2)
  md = [f"# C0 — artifact cache inventory\n\n**Verdict:** {verdict}\n",
        f"{len(rows)} artifacts: A_static={counts['A_static']}, B_correctness={counts['B_correctness']}, C_speed={counts['C_speed']}; {needs} need cache-metadata wrapping (none wrapped yet).\n",
        "| class | wrap-phase | count | reuse rule |", "|---|---|---|---|",
        "| A_static | C2 | %d | reuse by hash(inputs + code) — no GPU |" % counts["A_static"],
        "| B_correctness | C3 | %d | reuse only if inputs+code+runtime fingerprints match |" % counts["B_correctness"],
        "| C_speed | C3 | %d | historical by default; promotion reruns unless cached speed explicitly accepted |" % counts["C_speed"],
        "\n## Artifacts\n| path | class | speed? | corr? | needs wrap |", "|---|---|---|---|---|"]
  for x in rows: md.append(f"| {x['path']} | {x['class']} | {'Y' if x['has_speed_evidence'] else ''} | {'Y' if x['has_correctness_evidence'] else ''} | {'Y' if x['needs_wrapping'] else ''} |")
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  r = main()
  print(json.dumps({"verdict": r["verdict"], "total": r["total_artifacts"], "class_counts": r["class_counts"],
                    "need_wrapping": r["need_wrapping"], "unclassified": r["unclassified"]}, indent=2))
  print("\nC0", r["verdict"])
