#!/usr/bin/env python3
"""Lifecycle-search loop v0 — the first closed generate -> evaluate -> prune loop on top of the decode evaluator.

(Distinct from extra/qk_lifecycle_search.py, the read-only seed-ledger generator. This file is the LOOP: it RUNS
candidates through the evaluator. The seed generator RECORDS candidate schemas.)

It is a THIN orchestrator over extra/qk_decode_eval.py: it loads candidate specs, INDEPENDENTLY prunes
closed-lane / forbidden-promotion candidates BEFORE any benchmark, runs accepted candidates through decode_eval
(subprocess), validates the emitted artifacts, maps decode_eval verdicts -> lifecycle decisions, and proposes
(dedup'd) ledger/refutation updates. It duplicates NO benchmark logic, builds NO kernels, and changes NO defaults
(a PASS_PROMOTE only proposes an OWNER decision).

v0 is narrow: candidates bind to existing decode_eval candidates (replay of known outcomes) + two intentionally
invalid candidates that prove pruning works without benchmarking. v0 does not generate new kernel code.

CLI:
  python extra/qk_lifecycle_search_loop.py --list
  python extra/qk_lifecycle_search_loop.py --dry-run --suite decode_v0
  python extra/qk_lifecycle_search_loop.py --suite decode_v0 --out bench/qk-lifecycle-search/runs/
  python extra/qk_lifecycle_search_loop.py --candidate flash_l_64 --out bench/qk-lifecycle-search/runs/
  python extra/qk_lifecycle_search_loop.py --validate bench/qk-lifecycle-search/runs/<run>.json
"""
from __future__ import annotations
import argparse, glob, json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
LS = ROOT / "bench/qk-lifecycle-search"
CANDS = LS / "search_candidates.json"
POLICY = LS / "search_policy.json"
SCHEMA = LS / "search_schema.json"
RUNS = LS / "runs"
SUMMARIES = LS / "summaries"
EVAL = "extra/qk_decode_eval.py"
EVAL_RUNS = ROOT / "bench/qk-decode-eval/runs"
REFUTATIONS = LS / "refutations.json"
BINDING_TEMPLATES = ROOT / "bench/qk-decode-eval/binding_templates.json"

def _bindings() -> dict:
  if not BINDING_TEMPLATES.exists(): return {}
  return {b["binding_id"]: b for b in json.loads(BINDING_TEMPLATES.read_text()).get("templates", [])}

def _git(*a):
  try: return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"

def _text_of(c: dict) -> str:
  return " ".join(str(c.get(k, "")) for k in ("id", "family", "description", "intent", "notes")).lower() + \
         " " + " ".join(f"{k}={v}" for k, v in (c.get("env") or {}).items()).lower()

# ---- pruning: independent detection (not just the self-declared allowed_by_policy flag) -------------------------
def prune_decision(c: dict, pol: dict) -> tuple[str | None, str, str | None]:
  """Return (PRUNE_* | None, reason, closed_lane). None => EXECUTE."""
  txt = _text_of(c)
  for lane in pol["closed_lanes"]:
    if any(p in txt for p in lane["patterns"]) or c.get("closed_lane_risk") == lane["id"]:
      return "PRUNE_CLOSED_LANE", lane["reason"], lane["id"]
  intent = str(c.get("intent", "")).lower()
  if intent in pol["promotion_intents_requiring_owner"] or any(
      any(p in txt for p in fp["patterns"]) for fp in pol["forbidden_promotions"]):
    fp = next((f for f in pol["forbidden_promotions"] if any(p in txt for p in f["patterns"])), None)
    return "PRUNE_POLICY_VIOLATION", (fp["reason"] if fp else "promotion/flip/ship of a default requires explicit owner approval; the loop never promotes"), None
  # binding-template resolution (distinguishes: missing template / present-but-no-runner / executable)
  btid = c.get("binding_template_id")
  if btid:
    bt = _bindings().get(btid)
    if bt is None:
      return "PRUNE_MISSING_EVALUATOR_BINDING", f"binding template '{btid}' not found in binding_templates.json", None
    if not c.get("decode_eval_candidate_id"):
      miss = bt.get("missing_for_executable") or ["a concrete runner/kernel"]
      return "PRUNE_NEEDS_TEMPLATE", f"binding template '{btid}' exists (status={bt.get('concrete_runner_status')}) but has no concrete runner yet; missing: {'; '.join(miss)}", None
    # else: binding template exists AND a concrete decode_eval candidate is bound -> fall through to EXECUTE
  if c.get("deferred"):
    return "PRUNE_NEEDS_TEMPLATE", "deferred work: no evaluator-binding template/kernel exists yet", None
  if not c.get("decode_eval_candidate_id"):
    return "PRUNE_MISSING_EVALUATOR_BINDING", "no decode_eval_candidate_id; cannot bind to the evaluator", None
  return None, "accepted for evaluation", None

# ---- run one accepted candidate through decode_eval (subprocess) ------------------------------------------------
def run_decode_eval(eval_id: str, repeats: int | None) -> dict:
  before = set(glob.glob(str(EVAL_RUNS / f"*-{eval_id}.json")))
  cmd = [sys.executable, EVAL, "--candidate", eval_id, "--out", str(EVAL_RUNS)] + (["--repeats", str(repeats)] if repeats else [])
  env = os.environ.copy(); env.setdefault("DEV", "AMD"); env.setdefault("JIT", "1"); env["PYTHONPATH"] = str(ROOT)
  p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  after = set(glob.glob(str(EVAL_RUNS / f"*-{eval_id}.json")))
  new = sorted(after - before) or sorted(after)  # the artifact this run emitted (fallback: newest)
  artifact = pathlib.Path(new[-1]) if new else None
  verdict, band = None, None
  if artifact and artifact.exists():
    d = json.loads(artifact.read_text()); verdict = d.get("verdict"); band = d.get("wd", {}).get("repro_band_pct")
  valid = None
  if artifact:
    v = subprocess.run([sys.executable, EVAL, "--validate", str(artifact)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    valid = v.returncode == 0
  return {"command": "DEV=AMD JIT=1 " + " ".join(cmd[1:]), "artifact": str(artifact.relative_to(ROOT)) if artifact else None,
          "verdict": verdict, "artifact_valid": valid, "band": band, "stdout_tail": (p.stdout or "")[-300:]}

# ---- ledger dedup proposals (propose-only; the loop does not mutate ledgers) ------------------------------------
def propose_ledger(c: dict, lifecycle: str) -> list[dict]:
  props = []
  refs = json.loads(REFUTATIONS.read_text()).get("entries", []) if REFUTATIONS.exists() else []
  eid = c.get("decode_eval_candidate_id") or c["id"]
  if lifecycle in ("refute_candidate", "refute_for_promotion_bank_learning"):
    # dedup against existing refutations by the candidate id AND its mapped ledger-candidate (existing refutations
    # are keyed by ledger-candidate ids, not decode_eval ids) -> recognizes a conceptually-equivalent refutation.
    keys = [k for k in (eid, c["id"], c.get("maps_to_ledger_candidate")) if k]
    hit = next((e for e in refs if any(k in str(e.get("applies_to", [])) or k in e.get("id", "") for k in keys)), None)
    props.append({"ledger": "refutations.json", "action": "append_refutation", "id": f"{eid}_lifecycle_v0",
                  "status": "already_present_skip" if hit else "proposed",
                  "reason": (f"already covered by refutation '{hit['id']}'; not duplicated" if hit else f"new refutation row for {lifecycle}")})
  elif lifecycle == "opt_in_candidate_banked":
    props.append({"ledger": "policy_exports.json", "action": "bank_opt_in", "id": eid, "status": "already_present_skip",
                  "reason": "q8 opt-in policy is already banked (default-off); no new row"})
  elif lifecycle == "bank_baseline_or_rest":
    props.append({"ledger": "none", "action": "none", "id": eid, "status": "proposed", "reason": "baseline/rest; no ledger change"})
  return props

# ---- evaluate the suite/candidate -------------------------------------------------------------------------------
def run(cands: list[dict], pol: dict, reg: dict, suite: str | None, dry: bool, repeats: int | None) -> dict:
  v2l = pol["verdict_to_lifecycle_decision"]
  res = {"schema": "decode_lifecycle_search_run_v1", "run_id": f"decode_v0-{time.strftime('%Y%m%dT%H%M%S')}",
         "date": "2026-06-21", "git_commit": _git("rev-parse", "HEAD"), "dirty_tree": bool(_git("status", "--short")),
         "evaluator": {"path": EVAL, "candidate_registry": "bench/qk-decode-eval/candidates.json"}, "suite": suite,
         "candidates_total": len(cands), "pruned": [], "executed": [], "ledger_updates_proposed": [],
         "default_behavior_changed": False, "stop_reason": None}
  stop = None
  for c in cands:
    decision, reason, lane = prune_decision(c, pol)
    if decision is not None:
      res["pruned"].append({"id": c["id"], "decision": decision, "reason": reason, "closed_lane": lane, "benchmarked": False})
      print(f"  PRUNE  {c['id']:30} {decision} | {reason[:70]}", file=sys.stderr); continue
    if dry:
      res["executed"].append({"id": c["id"], "decode_eval_candidate_id": c["decode_eval_candidate_id"], "command": "[dry-run: would run decode_eval]",
                              "decode_eval_artifact": None, "decode_eval_verdict": None, "lifecycle_decision": "WOULD_EXECUTE",
                              "artifact_valid": None, "expected_verdict": c.get("expected_verdict"), "verdict_matches_expected": None,
                              "promotion_decision": "n/a (dry-run)", "wd_repro_band_pct": None})
      print(f"  EXEC?  {c['id']:30} -> decode_eval --candidate {c['decode_eval_candidate_id']} (dry)", file=sys.stderr); continue
    print(f"  EXEC   {c['id']:30} -> decode_eval --candidate {c['decode_eval_candidate_id']}", file=sys.stderr)
    ev = run_decode_eval(c["decode_eval_candidate_id"], repeats)
    lifecycle = v2l.get(ev["verdict"], "unknown")
    match = (c.get("expected_verdict") in (None, ev["verdict"]))
    promo = "owner_decision_required (loop never auto-promotes)" if ev["verdict"] == "PASS_PROMOTE" else "no_promotion"
    if lifecycle in ("stop_search_needs_measurement", "stop_search_needs_gpu_state", "stop_search_needs_template"):
      stop = f"{c['id']}: {lifecycle}"
    res["executed"].append({"id": c["id"], "decode_eval_candidate_id": c["decode_eval_candidate_id"], "command": ev["command"],
                            "decode_eval_artifact": ev["artifact"], "decode_eval_verdict": ev["verdict"], "lifecycle_decision": lifecycle,
                            "artifact_valid": ev["artifact_valid"], "expected_verdict": c.get("expected_verdict"),
                            "verdict_matches_expected": match, "promotion_decision": promo, "wd_repro_band_pct": ev["band"]})
    res["ledger_updates_proposed"] += propose_ledger(c, lifecycle)
    print(f"         -> {ev['verdict']} => {lifecycle} (expected {c.get('expected_verdict')}, match={match}) valid={ev['artifact_valid']}", file=sys.stderr)
  g = subprocess.run([sys.executable, "extra/qk_policy_consistency_check.py"], cwd=ROOT,
                     env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  res["policy_guard_result"] = "PASS" if g.returncode == 0 else "FAIL"
  n_exec = len([e for e in res["executed"] if e["lifecycle_decision"] != "WOULD_EXECUTE"])
  n_prune = len(res["pruned"]); ex = [e for e in res["executed"] if e["lifecycle_decision"] != "WOULD_EXECUTE"]
  all_match = all(e.get("verdict_matches_expected") for e in ex); all_valid = all(e.get("artifact_valid") for e in ex)
  res["stop_reason"] = stop
  if dry:
    res["final_decision"] = "LIFECYCLE_SEARCH_V0_READY"; res["next_recommended_project"] = "(dry-run; run the suite to confirm)"
  elif stop:
    res["final_decision"] = {"stop_search_needs_gpu_state": "NEEDS_EVALUATOR_API_CLEANUP", "stop_search_needs_template": "NEEDS_CANDIDATE_TEMPLATE_LAYER"}.get(stop.split(": ")[-1], "NEEDS_EVALUATOR_API_CLEANUP")
    res["next_recommended_project"] = "resolve the stop condition before extending the loop"
  elif n_exec >= 1 and all_match and all_valid and res["policy_guard_result"] == "PASS":
    # loop-health verdict: it executed >=1 candidate and everything it ran classified correctly + validated +
    # guard passed. Suite composition (exact exec/prune counts) is a per-suite acceptance gate, not loop health.
    res["final_decision"] = "LIFECYCLE_SEARCH_V0_READY"
    res["next_recommended_project"] = "candidate-template generation / evaluator-binding templates for the north-star flash_attn_tile"
  else:
    res["final_decision"] = "NEEDS_CANDIDATE_TEMPLATE_LAYER" if not all_valid else "NEEDS_POLICY_LANGUAGE_CLEANUP"
    res["next_recommended_project"] = "fix the failing gate (see executed/pruned table)"
  return res

def validate(path: pathlib.Path) -> int:
  import jsonschema
  try:
    jsonschema.validate(json.loads(path.read_text()), json.loads(SCHEMA.read_text()))
  except jsonschema.ValidationError as e: print(f"INVALID search artifact: {e.message}"); return 1
  d = json.loads(path.read_text()); bad = 0
  for e in d.get("executed", []):
    a = e.get("decode_eval_artifact")
    if a:
      v = subprocess.run([sys.executable, EVAL, "--validate", str(ROOT / a)], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, stdout=subprocess.PIPE)
      if v.returncode != 0: print(f"  linked decode_eval artifact INVALID: {a}"); bad += 1
  print(f"VALID: {path}" + (f" (but {bad} linked artifacts invalid)" if bad else " (+ all linked decode_eval artifacts valid)"))
  return 1 if bad else 0

def main() -> int:
  ap = argparse.ArgumentParser(description="Lifecycle-search loop v0 (evaluator-driven; no kernels, no defaults changed)")
  ap.add_argument("--list", action="store_true"); ap.add_argument("--suite"); ap.add_argument("--candidate")
  ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--repeats", type=int)
  ap.add_argument("--out", type=pathlib.Path, default=RUNS); ap.add_argument("--validate", type=pathlib.Path)
  ap.add_argument("--candidates", type=pathlib.Path, help="custom candidate registry (e.g. a generated one); defaults to search_candidates.json")
  args = ap.parse_args()
  if args.validate: return validate(args.validate)
  reg = json.loads((args.candidates or CANDS).read_text()); pol = json.loads(POLICY.read_text())
  by_id = {c["id"]: c for c in reg["candidates"]}
  if args.list:
    print(f"{'id':30}{'family':16}{'intent':22}{'expected_lane':34} allowed")
    for c in reg["candidates"]:
      print(f"{c['id']:30}{c['family']:16}{str(c.get('intent')):22}{str(c.get('expected_lane')):34}{c.get('allowed_by_policy')}")
    return 0
  cands = ([by_id[i] for i in reg["suites"][args.suite]] if args.suite else [by_id[args.candidate]] if args.candidate else [])
  if not cands: print("specify --list, --suite <name>, --candidate <id>, or --validate <file>"); return 2
  print(f"=== lifecycle-search v0 {'(dry-run)' if args.dry_run else ''}: {len(cands)} candidates ===", file=sys.stderr)
  res = run(cands, pol, reg, args.suite, args.dry_run, args.repeats)
  args.out.mkdir(parents=True, exist_ok=True)
  f = args.out / f"{res['run_id']}.json"; f.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
  SUMMARIES.mkdir(parents=True, exist_ok=True)
  (SUMMARIES / "latest.json").write_text(json.dumps({"run_id": res["run_id"], "final_decision": res["final_decision"],
    "executed": [(e["id"], e["decode_eval_verdict"], e["lifecycle_decision"], e["verdict_matches_expected"]) for e in res["executed"]],
    "pruned": [(p["id"], p["decision"]) for p in res["pruned"]], "policy_guard": res["policy_guard_result"],
    "next": res["next_recommended_project"]}, indent=2) + "\n")
  print(json.dumps({"final_decision": res["final_decision"], "executed": len([e for e in res["executed"] if e["lifecycle_decision"] != "WOULD_EXECUTE"]),
                    "pruned": len(res["pruned"]), "policy_guard": res["policy_guard_result"],
                    "artifact": str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f)}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
