#!/usr/bin/env python3
"""Candidate-template generation layer v0 — the 'generate' step of the lifecycle-search loop.

It expands route/fusion/layout TEMPLATES (bench/qk-lifecycle-search/templates.json) into legal decode candidate
SPECS in the search_candidates schema (decode_lifecycle_search_candidates_v1), each carrying full policy metadata.
It generates SPECS only — NO kernel code, NO new flags, NO benchmarks, NO subprocess eval. Executable variants bind
to EXISTING decode_eval candidates; invalid/deferred variants carry the metadata the loop prunes/defers on. The
emitted registry is consumed by `extra/qk_lifecycle_search_loop.py --candidates <file>` without hand editing.

CLI:
  python extra/qk_candidate_template_gen.py --list-templates
  python extra/qk_candidate_template_gen.py --template decode_flash_l_sweep --dry-run
  python extra/qk_candidate_template_gen.py --suite decode_template_v0 --out bench/qk-lifecycle-search/generated/
  python extra/qk_candidate_template_gen.py --validate bench/qk-lifecycle-search/generated/<file>.json
  python extra/qk_candidate_template_gen.py --suite decode_template_v0 \
         --emit-search-candidates bench/qk-lifecycle-search/search_candidates.generated.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LS = ROOT / "bench/qk-lifecycle-search"
TEMPLATES = LS / "templates.json"
TEMPLATE_SCHEMA = LS / "template_schema.json"
GEN = LS / "generated"
REFUTATIONS = LS / "refutations.json"
DATE = "2026-06-21"
# required keys every generated candidate must carry to be loop-consumable (decode_lifecycle_search_candidates_v1)
REQ = ("id", "family", "description", "source", "decode_eval_candidate_id", "intent", "env", "allowed_by_policy",
       "closed_lane_risk", "expected_verdict", "expected_lane")

from extra.qk_harness_contract import git_commit, dirty_tree  # provenance SSOT (was a local _git copy)

def mk(id, family, desc, template_id, deid, intent, env, allowed, closed_risk, first_gate, promo_gate,
       exp_verdict, exp_lane, maps=None, deferred=False, notes="", extra=None):
  c = {"id": id, "family": family, "description": desc, "source": f"generated:{template_id}", "template_id": template_id,
       "decode_eval_candidate_id": deid, "intent": intent, "env": env, "allowed_by_policy": allowed,
       "closed_lane_risk": closed_risk, "first_gate": first_gate, "promotion_gate": promo_gate,
       "stop_condition": "see template", "expected_verdict": exp_verdict, "expected_lane": exp_lane, "notes": notes}
  if maps: c["maps_to_ledger_candidate"] = maps
  if deferred: c["deferred"] = True
  if extra: c.update(extra)
  return c

BINDINGS = LS.parent / "qk-decode-eval/binding_templates.json"
def _binding(bid):
  if not BINDINGS.exists(): return None
  for b in json.loads(BINDINGS.read_text()).get("templates", []):
    if b["binding_id"] == bid: return b
  return None

# ---- per-template deterministic expanders --------------------------------------------------------------------
def _flash_l(t):
  out = []
  for fl in t["default_params"]["flash_l"]:
    if fl == 128:
      out.append(mk("gen_flash_l_128", "attention_split", "FLASH_L=128 default baseline (canonical curve).",
                    t["id"], "baseline_default", "measure_baseline", {}, True, None, "n/a (baseline)",
                    "n/a (not a promotion candidate)", "REST", "bank_baseline_or_rest",
                    notes="generated baseline; binds to decode_eval baseline_default"))
    elif fl == 64:
      out.append(mk("gen_flash_l_64", "attention_split", "FLASH_L=64 more KV-splits; known LOCAL-PASS/W==D-FAIL.",
                    t["id"], "flash_l_64", "measure_candidate", {"FLASH_L": "64"}, True, None,
                    "local >=1.05x vs gqa_coop_vec @ctx1024", "W==D >=5%@1024 or >=7%@4096",
                    "LOCAL_PASS_WD_FAIL", "refute_for_promotion_bank_learning",
                    maps="decode_vector_flash_tile_high_kvsplit",
                    notes="measuring allowed; promoting is not (see gen_promote_flash_l_64)"))
  out.append(mk("gen_promote_flash_l_64", "attention_split", "Attempt to PROMOTE FLASH_L=64 as the decode default.",
                t["id"], "flash_l_64", "promote_default", {"FLASH_L": "64"}, False, None, "n/a (pruned before eval)",
                "FLASH_L=64 failed W==D; not a default", None, "PRUNE_POLICY_VIOLATION",
                notes="invalid: promotion intent; pruned before any benchmark"))
  return out

def _q8(t):
  return [
    mk("gen_q8_opt_in", "q8_route", "q8 FFN handwritten route, opt-in only. Known PASS_OPT_IN.", t["id"],
       "q8_opt_in", "measure_opt_in", {"Q8_FFN_HANDWRITTEN": "1"}, True, None,
       "opt-in speed >=1.03x + dNLL <=0.01", "opt-in only; never a default without owner approval",
       "PASS_OPT_IN", "opt_in_candidate_banked", notes="banked opt-in; default stays off"),
    mk("gen_q8_default_attempt", "q8_route", "Attempt to make q8 the decode DEFAULT (q8 default).", t["id"],
       "q8_opt_in", "promote_default", {"Q8_FFN_HANDWRITTEN": "1"}, False, None, "n/a (pruned before eval)",
       "q8 is opt-in only; default forbidden", None, "PRUNE_POLICY_VIOLATION",
       notes="invalid: q8 default promotion; pruned before any benchmark"),
  ]

def _closed(t):
  spec = {"wmma_decode": ("Reopen WMMA/tensor-core flash decode as llama's path.", {"GGML_HIP_ROCWMMA_FATTN": "1"}),
          "mmvq": ("Reopen MMVQ as the decode gap.", {}),
          "bounded_fusion": ("Reopen bounded FFN/attention decode fusion.", {})}
  out = []
  for lane in t["default_params"]["lanes"]:
    desc, env = spec[lane]
    out.append(mk(f"gen_{lane}_reopen", "closed_lane_probe", desc, t["id"], None, "reopen_closed_lane", env, False,
                  lane, "n/a (pruned before eval)", "n/a", None, "PRUNE_CLOSED_LANE",
                  notes=f"invalid: closed lane {lane}; pruned before any benchmark"))
  return out

def _north_star(t):
  b = _binding("north_star_flash_attn_tile_v0") or {}
  req = list((b.get("required_candidate_params") or {}).keys())
  out = []
  # 1) the real north-star: kernel + local-A/B runner now EXIST (binding north_star_flash_attn_tile_v0) -> EXECUTE.
  #    MEASURED local A/B fail (0.58x@1024) -> FAIL_LOCAL_AB -> refute_candidate. No W==D route (gate missed).
  out.append(mk("gen_north_star_flash_attn_tile", "north_star_flash_attn_tile",
                "Warp-cooperative flash_attn_tile decode candidate (many KV-splits, GQA pack, register online softmax, many-wg combine) vs gqa_coop_vec, per binding north_star_flash_attn_tile_v0. Executable local A/B.",
                t["id"], "north_star_flash_attn_tile", "measure_candidate", {}, True, None,
                "local A/B >= 1.05x vs gqa_coop_vec @ctx1024 (MEASURED 0.58x -> miss)",
                "W==D >= 5%@1024 (not reached; local gate failed first)", "FAIL_LOCAL_AB", "refute_candidate",
                maps="decode_vector_flash_tile_high_kvsplit",
                notes="kernel + local-A/B runner implemented; local gate MISSED (warp partial alone >= coop; combine bandwidth-bound); refuted, not promoted; compares vs gqa_coop_vec",
                extra={"binding_template_id": "north_star_flash_attn_tile_v0", "maps_to_north_star": True,
                       "executable_status": "local_ab_implemented_failed_gate", "required_params": req,
                       "comparator": b.get("comparator"), "expected_first_real_gate": (b.get("gates") or {}).get("local_ab"),
                       "expected_stop_conditions": b.get("stop_conditions")}))
  # 2) executable plumbing selftest: binding EXISTS + concrete decode_eval runner -> EXECUTE -> SELFTEST_PASS (no perf)
  out.append(mk("gen_north_star_binding_selftest", "binding_selftest",
                "Binding-plumbing selftest (proves the binding->candidate->decode_eval->artifact path is executable). NOT a performance candidate.",
                t["id"], "north_star_binding_selftest", "binding_selftest", {}, True, None,
                "n/a (selftest)", "never promotable", "SELFTEST_PASS", "selftest_only_not_perf",
                notes="executable plumbing only; SELFTEST_PASS is not a perf pass; distinct from gen_north_star_flash_attn_tile",
                extra={"binding_template_id": "north_star_binding_selftest_v0", "executable_status": "executable_selftest"}))
  # 3) missing-binding demo: binding_template_id does NOT exist -> PRUNE_MISSING_EVALUATOR_BINDING
  out.append(mk("gen_north_star_missing_binding", "north_star_flash_attn_tile",
                "Demonstrates the missing-binding case: references a binding template id that does not exist.",
                t["id"], None, "deferred_north_star", {}, True, None, "n/a (missing binding)",
                "n/a", None, "PRUNE_MISSING_EVALUATOR_BINDING", deferred=True,
                notes="proves the loop distinguishes a missing binding template from a present-but-unimplemented one",
                extra={"binding_template_id": "north_star_flash_attn_tile_vX_does_not_exist", "executable_status": "missing_binding_template"}))
  return out

EXPANDERS = {"decode_flash_l_sweep": _flash_l, "q8_opt_in_policy": _q8,
             "closed_lane_reopen_attempts": _closed, "north_star_flash_attn_tile_placeholder": _north_star}

# ---- generation / provenance --------------------------------------------------------------------------------
def refutation_for(maps):
  if not maps or not REFUTATIONS.exists(): return None
  for e in json.loads(REFUTATIONS.read_text()).get("entries", []):
    if maps in str(e.get("applies_to", [])): return e["id"]
  return None

def check_candidate(c) -> list[str]:
  return [k for k in REQ if k not in c]

def expand_template(t) -> dict:
  cands = EXPANDERS[t["id"]](t)
  errs = {c["id"]: check_candidate(c) for c in cands}
  return {"template_id": t["id"], "template_family": t["family"], "input_params": t["default_params"],
          "generated_candidate_ids": [c["id"] for c in cands], "candidate_specs": cands,
          "rows": [{"id": c["id"], "policy_label": ("legal" if c["allowed_by_policy"] else "invalid"),
                    "expected_decision": c["expected_lane"], "decode_eval_binding": c["decode_eval_candidate_id"],
                    "binding_reason": (None if c["decode_eval_candidate_id"] else
                                       ("deferred: no evaluator binding yet" if c.get("deferred") else
                                        "no evaluator binding (closed/invalid lane)")),
                    "maps_to_existing_refutation": refutation_for(c.get("maps_to_ledger_candidate"))} for c in cands],
          "validation_result": "PASS" if not any(errs.values()) else f"MISSING_FIELDS {errs}"}

def generate(template_ids: list[str], templates: dict) -> dict:
  by = {t["id"]: t for t in templates["templates"]}
  exps = [expand_template(by[tid]) for tid in template_ids]
  return {"schema": "decode_candidate_generation_v1", "date": DATE, "generator": "extra/qk_candidate_template_gen.py",
          "git_commit": git_commit(), "dirty_tree": dirty_tree(),
          "templates": exps, "validation_result": "PASS" if all(e["validation_result"] == "PASS" for e in exps) else "FAIL"}

def to_search_registry(gen: dict) -> dict:
  cands, ids = [], []
  for e in gen["templates"]:
    for c in e["candidate_specs"]:
      c2 = {k: v for k, v in c.items()}; cands.append(c2); ids.append(c["id"])
  return {"schema": "decode_lifecycle_search_candidates_v1", "date": DATE,
          "comment": "GENERATED by extra/qk_candidate_template_gen.py from templates.json. Consumable by "
                     "qk_lifecycle_search_loop.py --candidates. Do not hand-edit; regenerate.",
          "candidates": cands, "suites": {"decode_template_v0": ids}}

def validate(path: pathlib.Path) -> int:
  d = json.loads(path.read_text()); bad = 0
  specs = d.get("candidates") if d.get("schema") == "decode_lifecycle_search_candidates_v1" else \
          [c for e in d.get("templates", []) for c in e.get("candidate_specs", [])]
  for c in specs:
    m = check_candidate(c)
    if m: print(f"  candidate {c.get('id')} MISSING {m}"); bad += 1
  print(f"VALID: {path} ({len(specs)} candidates)" if not bad else f"INVALID: {path} ({bad} bad)")
  return 1 if bad else 0

def main() -> int:
  ap = argparse.ArgumentParser(description="Candidate-template generator v0 (specs only; no kernels/benchmarks)")
  ap.add_argument("--list-templates", action="store_true"); ap.add_argument("--template"); ap.add_argument("--suite")
  ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--out", type=pathlib.Path, default=GEN)
  ap.add_argument("--validate", type=pathlib.Path); ap.add_argument("--emit-search-candidates", type=pathlib.Path)
  args = ap.parse_args()
  if args.validate: return validate(args.validate)
  templates = json.loads(TEMPLATES.read_text())
  if args.list_templates:
    print(f"{'template':40}{'family':20} description")
    for t in templates["templates"]: print(f"{t['id']:40}{t['family']:20}{t['description'][:60]}")
    return 0
  tids = (templates["suites"][args.suite] if args.suite else [args.template] if args.template else [])
  if not tids: print("specify --list-templates, --template <id>, --suite <name>, or --validate <file>"); return 2
  gen = generate(tids, templates)
  reg = to_search_registry(gen)
  if args.dry_run:
    print(json.dumps({"templates": tids, "generated": reg["suites"], "n_candidates": len(reg["candidates"]),
                      "decisions": {c["id"]: c["expected_lane"] for c in reg["candidates"]},
                      "validation": gen["validation_result"]}, indent=2))
    return 0
  def _rel(p): p = p.resolve(); return str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
  args.out.mkdir(parents=True, exist_ok=True)
  name = (args.suite or args.template) + ".json"
  (args.out / name).write_text(json.dumps(gen, indent=2, sort_keys=True) + "\n")
  (args.out / "latest.json").write_text(json.dumps(gen, indent=2, sort_keys=True) + "\n")
  if args.emit_search_candidates:
    args.emit_search_candidates.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"generated_artifact": _rel(args.out / name),
                    "search_candidates": (_rel(args.emit_search_candidates) if args.emit_search_candidates else None),
                    "n_candidates": len(reg["candidates"]), "validation": gen["validation_result"]}, indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
