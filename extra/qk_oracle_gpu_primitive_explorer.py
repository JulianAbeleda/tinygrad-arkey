#!/usr/bin/env python3
"""Oracle-guided GPU primitive explorer — the GENERIC spec-driven runner.

The connective tissue the runner-design doc specified: it consumes ONE bounded search spec (the explorer spec JSON,
or a qk_search_spec.SearchRow via the adapter), resolves a per-lane backend, enumerates candidates from the spec's
declared knob ranges, VALIDATES every knob/value against the lane's allow-list (so a learned proposer can't inject a
hallucinated knob/value), and either:
  --dry-run : enumerate + structurally validate candidates, NO benchmark (cheap; the proposer/CI gate);
  (default) : drive the existing lane backend's cost-ordered gate stack (route/materialization/ISA/correctness/W==D),
              rank vs the frozen oracle, write per-run artifacts, and append one project-ledger entry per candidate.

It does NOT reimplement search or gates — for decode it wraps extra/qk_decode_search_runner.run_candidate (which spawns
the real gate). It adds: the spec->candidate adapter, knob-value validation, a per-lane gate/authority registry, and
spec-driven (not inline-literal) candidate generation. NO default flip, NO kernel/model change; W==D stays the only
decode promotion authority and a harness only RECOMMENDS.

  PYTHONPATH=. .venv/bin/python extra/qk_oracle_gpu_primitive_explorer.py \
    --spec bench/qk-oracle-gpu-primitive-explorer/spec_decode_policy_example.json \
    --out  bench/qk-oracle-gpu-primitive-explorer/runs/decode_policy_001 [--dry-run] [--max-candidates N]

  # adapter self-test (no GPU):  ... --selftest
See docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md.
"""
from __future__ import annotations
import argparse, itertools, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---- per-lane backend registry -------------------------------------------------------------------------------------
# Each lane declares: knob aliases (semantic/LoRA name -> real env var), allowed values per env var, a baseline env,
# the expected in-graph kernel symbol, the frozen-oracle file, and the authority class. Gated/non-promotion lanes
# declare why they cannot run a speed search, so the runner is honest about coverage instead of silently doing nothing.
DECODE_KNOB_ALIAS = {
  "DECODE_ATTN_AMDGCN_S": "DECODE_ATTN_AMDGCN_S", "S": "DECODE_ATTN_AMDGCN_S", "split_S": "DECODE_ATTN_AMDGCN_S",
  "DECODE_ATTN_AMDGCN_COMBINE": "DECODE_ATTN_AMDGCN_COMBINE", "combine": "DECODE_ATTN_AMDGCN_COMBINE",
  "combine_variant": "DECODE_ATTN_AMDGCN_COMBINE",
  "DECODE_ATTN_AMDGCN_MIN_CTX": "DECODE_ATTN_AMDGCN_MIN_CTX", "min_ctx": "DECODE_ATTN_AMDGCN_MIN_CTX",
}
DECODE_ALLOWED = {  # env var -> allowed values (validation allow-list; unknown value => structural reject, not benchmarked)
  "DECODE_ATTN_AMDGCN_S": [32, 48, 64, 96], "DECODE_ATTN_AMDGCN_COMBINE": ["base", "hd64"],
  "DECODE_ATTN_AMDGCN_MIN_CTX": [512, 1024],
}
LANES = {
  "decode_policy": {"kind": "runnable", "alias": DECODE_KNOB_ALIAS, "allowed": DECODE_ALLOWED,
    "baseline_env": {"DECODE_ATTN_KV_IDENTITY": 1}, "expected_kernel": "owned_flash_tile_gqa_whole",
    "oracle_file": "bench/qk-decode-search-readiness/baseline_oracle.json", "authority": "clean synced W==D"},
  "native_codegen_microprimitive": {"kind": "nonpromotion", "tool": "extra/qk_native_codegen_microsearch.py",
    "authority": "local correctness (rel_rmse<=1e-2) + ISA target; NEVER W==D",
    "note": "non-promotion lane; run its own tool directly. The generic runner does not benchmark it for speed."},
  "prefill_role_policy": {"kind": "gated", "verdict": "PREFILL_SEARCH_GATED_OFF_AT_REST",
    "note": "kernel at ~99.5% Tensile; lever is non-search in-model integration penalty. Reopen only role-specific."},
  "cross_shape": {"kind": "gated", "verdict": "CROSS_SHAPE_SEARCH_NEEDS_TARGETS",
    "note": "single gfx1100; no alt GPU/model; 14B/32B owner-gated."},
}
# alias the spec's lane strings (decode_policy / decode / native_codegen_microprimitive / ...) tolerantly
LANE_ALIASES = {"decode": "decode_policy", "decode_policy": "decode_policy",
  "native_codegen_microprimitive": "native_codegen_microprimitive", "native-codegen-microprimitive": "native_codegen_microprimitive",
  "prefill": "prefill_role_policy", "prefill_role_policy": "prefill_role_policy",
  "cross_shape": "cross_shape", "cross-shape": "cross_shape"}

# ---- SearchRow adapter ---------------------------------------------------------------------------------------------
def searchrow_to_spec(row: dict) -> dict:
  """Map a qk_search_spec.SearchRow dict -> an explorer spec. Only the decode attention-policy mapping is concrete
  today (op_scope=attention, search_space=primitive_policy -> S/combine/min_ctx). Other rows map to gated lanes."""
  ss = row.get("search_space"); op = row.get("op_scope")
  if op == "attention" and ss == "primitive_policy":
    return {"search_id": row.get("id", "searchrow"), "lane": "decode_policy", "oracle_id": "decode_whole_cache_owned_tile_8b_gfx1100",
            "candidate_generator": "extra/qk_decode_search_execute.py (Mode A)",
            "knobs_ranges": {"split_S": DECODE_ALLOWED["DECODE_ATTN_AMDGCN_S"],
                             "combine_variant": DECODE_ALLOWED["DECODE_ATTN_AMDGCN_COMBINE"],
                             "min_ctx": DECODE_ALLOWED["DECODE_ATTN_AMDGCN_MIN_CTX"]},
            "authority_benchmark": "clean synced W==D", "budget": "policy grid", "_from_searchrow": True}
  lane = {"prefill": "prefill_role_policy"}.get(row.get("phase"), "cross_shape")
  return {"search_id": row.get("id", "searchrow"), "lane": lane, "knobs_ranges": {}, "_from_searchrow": True,
          "_note": f"no concrete decode mapping for op_scope={op}/search_space={ss}; routed to gated lane"}

# ---- candidate enumeration + validation ----------------------------------------------------------------------------
def enumerate_candidates(lane_cfg: dict, knobs_ranges: dict, max_candidates: int | None):
  """Cartesian product of declared knob ranges -> candidates. Each knob name is resolved via the alias map to a real
  env var; each value is checked against the allow-list. Invalid knob/value => candidate marked structurally invalid
  (recorded, never benchmarked) -- this is the 'no hallucinated knob/value' guard for learned proposals."""
  alias, allowed = lane_cfg.get("alias", {}), lane_cfg.get("allowed", {})
  names = list(knobs_ranges.keys()); value_lists = [knobs_ranges[n] for n in names]
  cands = []
  for combo in itertools.product(*value_lists) if names else [()]:
    env, invalid = dict(lane_cfg.get("baseline_env", {})), []
    cid_parts = []
    for name, val in zip(names, combo):
      ev = alias.get(name)
      if ev is None: invalid.append(f"unknown_knob:{name}"); continue
      if ev in allowed and val not in allowed[ev]: invalid.append(f"value_out_of_range:{ev}={val}")
      env[ev] = val; cid_parts.append(f"{ev.split('_')[-1]}{val}")
    cid = "_".join(cid_parts) or "baseline"
    cands.append({"id": cid, "env": env, "structurally_valid": not invalid, "invalid_reasons": invalid,
                  "expected_kernel": lane_cfg.get("expected_kernel")})
  # dedup by id, keep order
  seen, uniq = set(), []
  for c in cands:
    if c["id"] in seen: continue
    seen.add(c["id"]); uniq.append(c)
  if max_candidates: uniq = uniq[:max_candidates]
  return uniq

# ---- run -----------------------------------------------------------------------------------------------------------
def load_spec(spec_arg: str) -> dict:
  d = json.loads(pathlib.Path(spec_arg).read_text())
  if "search_space" in d and "op_scope" in d:  # it's a SearchRow
    return searchrow_to_spec(d)
  return d

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--spec"); ap.add_argument("--out", default="bench/qk-oracle-gpu-primitive-explorer/runs/adhoc")
  ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--max-candidates", type=int, default=None)
  ap.add_argument("--selftest", action="store_true")
  ap.add_argument("--no-ledger", action="store_true", help="do not append to the durable project ledger (proof/CI runs)")
  args = ap.parse_args()

  if args.selftest:  # adapter/enumeration self-test, no GPU/tinygrad
    spec = load_spec("bench/qk-oracle-gpu-primitive-explorer/spec_decode_policy_example.json")
    lane = LANES[LANE_ALIASES[spec["lane"]]]
    cands = enumerate_candidates(lane, spec["knobs_ranges"], args.max_candidates)
    bad = [{"id": "hallucinated", "env": {}, "structurally_valid": False}]
    bad_enum = enumerate_candidates(lane, {"combine_variant": ["base", "hw128"], "split_S": [48]}, None)
    print(json.dumps({"selftest": "ok", "n_candidates": len(cands), "sample_ids": [c["id"] for c in cands[:6]],
      "hallucinated_value_rejected": [c["invalid_reasons"] for c in bad_enum if not c["structurally_valid"]],
      "searchrow_maps": searchrow_to_spec({"id": "r1", "op_scope": "attention", "search_space": "primitive_policy", "phase": "decode"})["lane"]}, indent=2))
    return

  assert args.spec, "--spec required (or --selftest)"
  spec = load_spec(args.spec)
  lane_key = LANE_ALIASES.get(spec.get("lane"), spec.get("lane"))
  lane_cfg = LANES.get(lane_key)
  out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
  base = {"date": "2026-06-23", "search_id": spec.get("search_id"), "lane": lane_key, "oracle_id": spec.get("oracle_id"),
          "authority": lane_cfg and lane_cfg.get("authority"), "dry_run": args.dry_run}

  if lane_cfg is None:
    res = {**base, "verdict": "EXPLORER_UNKNOWN_LANE_STOP", "stop_reason": f"unknown lane {spec.get('lane')}"}
    (out/"decision.json").write_text(json.dumps(res, indent=2)); print("EXPLORER " + json.dumps(res)); return
  if lane_cfg["kind"] in ("gated", "nonpromotion"):
    res = {**base, "verdict": ("EXPLORER_LANE_GATED" if lane_cfg["kind"] == "gated" else "EXPLORER_LANE_NONPROMOTION"),
           "lane_verdict": lane_cfg.get("verdict"), "note": lane_cfg.get("note"), "tool": lane_cfg.get("tool"),
           "stop_reason": "lane cannot run a speed search through the generic runner (by design)"}
    (out/"decision.json").write_text(json.dumps(res, indent=2)); print("EXPLORER " + json.dumps(res)); return

  cands = enumerate_candidates(lane_cfg, spec.get("knobs_ranges", {}), args.max_candidates)
  valid = [c for c in cands if c["structurally_valid"]]
  invalid = [c for c in cands if not c["structurally_valid"]]
  (out/"candidate_manifest.json").write_text(json.dumps({**base, "n_candidates": len(cands), "n_valid": len(valid),
    "n_invalid": len(invalid), "candidates": cands}, indent=2))

  if args.dry_run:
    res = {**base, "phase": "DRY_RUN", "n_candidates": len(cands), "n_structurally_valid": len(valid),
           "n_rejected_structural": len(invalid), "rejected": [{"id": c["id"], "why": c["invalid_reasons"]} for c in invalid],
           "verdict": "EXPLORER_DRY_RUN_OK" if valid else "EXPLORER_DRY_RUN_NO_VALID_CANDIDATES",
           "stop_reason": "dry-run: enumerated + validated, no benchmark"}
    (out/"decision.json").write_text(json.dumps(res, indent=2)); print("EXPLORER " + json.dumps(res)); return

  # real run: delegate each valid candidate to the decode backend gate (route/mat/ISA/correctness/W==D), rank vs oracle
  from extra.qk_decode_search_runner import run_candidate, ORACLE_FILE  # lazy (GPU)
  from extra import qk_project_search_ledger as LED
  oracle = json.loads(pathlib.Path(lane_cfg["oracle_file"]).read_text()); o_wd = oracle.get("wd", {})
  o1024 = o_wd.get("1024", {}).get("tok_s"); o512 = o_wd.get("512", {}).get("tok_s")
  results, ledger_lines = [], []
  with open(out/"results.jsonl", "w") as fh:
    for c in valid:
      res = run_candidate({"id": c["id"], "env": c["env"]}, oracle_tokens_file=str(ORACLE_FILE))
      if res.get("verdict") == "PASS" and o1024:
        w = res.get("wd", {})
        res["delta_vs_oracle_pct_1024"] = round(100*(w["1024"]["tok_s"]-o1024)/o1024, 1)
        res["delta_vs_oracle_pct_512"] = round(100*(w["512"]["tok_s"]-o512)/o512, 1) if o512 else None
        spread = max(w["512"].get("spread_pct", 1.0), o_wd.get("512", {}).get("spread_pct", 1.0))/100
        if o512 and w["512"]["tok_s"] < o512 * (1 - max(spread, 0.02)):
          res["verdict"] = "REJECT_WD_REGRESSION"; res["reject_reason"] = "ctx512_regression"
      fh.write(json.dumps(res)+"\n"); results.append(res)
      ledger_lines.append(LED.entry(candidate_id=f"decode/{spec.get('search_id','run')}/{c['id']}", lane="decode",
        primitive_class="attention", knobs=c["env"], oracle=spec.get("oracle_id", "decode_whole_cache_owned_tile_8b_gfx1100"),
        correctness=res.get("token_byte_identical"), route_identity=res.get("route"), materialization_abi=res.get("materialization"),
        isa=res.get("isa"), local_diagnostic=None, authority_benchmark=res.get("wd"), verdict=res.get("verdict"),
        stop_reason=res.get("reject_reason") or "passed all gates", artifact_links=[str(out/"results.jsonl")],
        learned_rule=None))
      print(f"[run] {c['id']:22} {res.get('verdict'):26} d1024={res.get('delta_vs_oracle_pct_1024')}", file=sys.stderr)
  passing = [r for r in results if r.get("verdict") == "PASS"]
  lb = sorted(passing, key=lambda r: -(r.get("delta_vs_oracle_pct_1024") or -999))
  best = lb[0] if lb else None
  beats = bool(best and (best.get("delta_vs_oracle_pct_1024") or 0) > max(o_wd.get("1024", {}).get("spread_pct", 1.0), 1.0))
  verdict = ("EXPLORER_DECODE_WINNER_FOUND_RECOMMEND_ONLY" if beats else
             ("EXPLORER_DECODE_ORACLE_REMAINS_BEST" if passing else "EXPLORER_DECODE_NO_PASSING_CANDIDATES"))
  decision = {**base, "n_candidates": len(cands), "n_valid": len(valid), "n_invalid": len(invalid), "n_passing": len(passing),
    "best_candidate": (best or {}).get("id"), "best_delta_1024": (best or {}).get("delta_vs_oracle_pct_1024"),
    "oracle_beaten_outside_spread": beats, "default_flipped": False, "verdict": verdict,
    "leaderboard": [{"id": r["id"], "d1024": r.get("delta_vs_oracle_pct_1024")} for r in lb],
    "recommendation": "oracle remains default; recommend-only" if not beats else "winner -> owner decision",
    "stop_reason": "spec grid exhausted"}
  # append ledger entries (one per candidate) unless suppressed for a proof/CI run
  if not args.no_ledger:
    with open(ROOT/"bench/qk-project-search-ledger/ledger.jsonl", "a") as lf:
      for e in ledger_lines: lf.write(json.dumps(e)+"\n")
  decision["ledger_appended"] = (not args.no_ledger) and len(ledger_lines)
  (out/"decision.json").write_text(json.dumps(decision, indent=2))
  print("EXPLORER " + json.dumps({k: decision[k] for k in ["verdict", "n_passing", "best_candidate", "best_delta_1024", "oracle_beaten_outside_spread"]}))

if __name__ == "__main__":
  main()
