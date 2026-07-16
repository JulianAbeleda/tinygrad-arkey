#!/usr/bin/env python3
"""PMS-R0: default-path kernel census.

Replaces the rough "about four hot kernels" framing with a machine-readable census of what actually runs on the default
decode/prefill path, who writes each kernel, who selects it, and where its authority lives. Every row is derived from
the ACTUAL model route guards (cited guard in tinygrad/llm/decode_routes.py + the route source files), NOT inferred from
filenames. The route identity is cross-checked against extra/qk/route_manifest.py (PMS-R1).

Run:  PYTHONPATH=. python3 extra/audit/pure_machine_search_default_path_census.py
Outputs (bench/pure-machine-search-default-path-census/):
  latest.json            -- full census (all rows + summary counts + the headline answer)
  summary.md             -- human table
  default_route_table.json  -- rows on the live default path
  fallback_table.json       -- fallback/reference/refuted/research rows (NOT on the default path)

This tool reads source files and writes JSON/MD only. It runs no kernels and changes no defaults.
"""
from __future__ import annotations
import json, pathlib
from extra.qk.route_manifest import (ROUTES, default_routes, default_purity_report, derive_purity_status,
                                      route_provenance, validate_manifest)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/pure-machine-search-default-path-census"

# writer taxonomy (legacy PMS-R0): generated | codegen_emitter | owned_asm | tinygrad_generated
# provenance taxonomy (strict purity gate): see extra/qk_route_manifest.ROUTE_PROVENANCE.
# selector taxonomy: BubbleBeam | manifest | env_guard | hardcoded_default | tinygrad_scheduler
#
# SINGLE SOURCE OF TRUTH: the route identity (workload, roles, quant, provenance, status, purity_status) lives in
# extra/qk/route_manifest.ROUTES. This census used to re-declare all of that as a second parallel copy; it now DERIVES
# those fields from the manifest and carries only census-specific COMMENTARY as an overlay keyed by route_id -- the
# quoted source guard, the writer/selector taxonomy, the authority artifact, and the next-action note. A new census row
# is a row of overlay data plus a manifest route; it cannot silently disagree with the manifest about a route's shape,
# quant, provenance, or purity_status. `_build_census_rows()` asserts every overlay route exists in the manifest.
#
# overlay fields (census commentary only): shape_guard | writer | selector | route_guard | kernel_source |
#   authority_artifact | rollback_flag | next_action.  purity_status is DERIVED from the manifest (route_manifest.
#   derive_purity_status); workload/role/quant are DERIVED from the manifest route.
CENSUS_OVERLAY = [
  # ----- decode Q4_K weight GEMV -----
  ("decode_q4k_g3_generated", {
    "shape_guard": "QK_ROUTE_POLICY decode_q4k_g3_generated per tensor OR g3_bubblebeam_shape OR DECODE_Q4K_G3_ANYSHAPE structural guard ((in//256)%4==0 and out%32==0)",
    "writer": "generated",
    "selector": "env_guard",
    "route_guard": "tinygrad/llm/decode_routes.py q4k_primitive_linear_call getenv('BUBBLEBEAM_FUTURESIGHT', 1)==1 (default-on) + _qk_route_policy_selects_q4k_g3 (BoltBeam QK_ROUTE_POLICY) + DECODE_Q4K_G3_ANYSHAPE default-on -> q4k_g3_lanemap_gemv_kernel fires FIRST for eligible shapes, short-circuiting the owned-warp guards; strict policy fails loud on hidden fallback",
    "kernel_source": "extra/qk/gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel (UOp program from extra/qk/gemv_g2_lanemap.py Q4KGateUpLaneMap)",
    "authority_artifact": "bench/amd-isa-backend-g3-weight-promotion/latest.json (AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT)",
    "rollback_flag": "BUBBLEBEAM_FUTURESIGHT=0 -> ordinary tinygrad graph; no manifest hand-kernel rollback remains",
    "next_action": "keep promoted; make BoltBeam-generated policy the selector authority; do NOT reopen Q4_K layout reshuffle while parity holds",
  }),
  # ----- decode Q6_K weight GEMV -----
  ("decode_q6k_coop_generated", {
    "shape_guard": "ffn_down 12288->4096 | long-K ffn_down in>=8192/out<100000 | lm_head out>=100000 | attn_v partial",
    "writer": "generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_guard": "tinygrad/llm/decode_routes.py q6k_primitive_linear_call generated branch: getenv('DECODE_Q6K_GENERATED', 1) or QK_ROUTE_POLICY decode_q6k_coop_generated -> emit_q6k_gemv_kernel(spec) fires the coop/partial route; shipped hand kernels short-circuited",
    "kernel_source": "extra/qk/q6k_route_spec.py emit_q6k_gemv_kernel (spec-driven lowering of Q6KGEMVRouteSpec -> q6k_gen_coop_* / q6k_gen_partial_*)",
    "authority_artifact": "bench/tg-p3-q6k-generated-coop/latest.json (TG_P3_PASS_Q6K_GENERATED_COOP: all_identical, worst gen/shipped 1.011)",
    "rollback_flag": "DECODE_Q6K_GENERATED=0 no longer selects a manifest hand-kernel rollback; generated Q6_K decode is the only manifest kernel route",
    "next_action": "keep promoted; BoltBeam owns Q6_K generated selection; TG-P4/P5 remain",
  }),
  # ----- decode attention -----
  ("decode_flash_live_split_g4_8b_kvboth", {
    "shape_guard": "B=1 Hq=32 Hkv=8 Hd=128 G=4 ctx>=512",
    "writer": "generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_guard": "tinygrad/llm/decode_routes.py attention live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0): default-on DECODE_LIVE_SPLIT=1 -> FlashDecodeAttentionSpec live-split block tile + fused combine",
    "kernel_source": "extra/qk/flash_decode_attention_spec.py FlashDecodeAttentionSpec -> existing live-split UOp tile/combine emitters",
    "authority_artifact": "bench/tg-p14-amd-recovery-and-pure-attention-landing/phase2_final_result.json (PASS_PROMOTION_CANDIDATE; practical roofline closeout)",
    "rollback_flag": "DECODE_LIVE_SPLIT=0 exits the live-split default; no manifest fallback route row remains",
    "next_action": "keep promoted; no handwritten attention kernel on the hot path",
  }),
  ("decode_flash_block_tile_g5_konly", {
    "shape_guard": "B=1 Hq=40 Hkv=8 Hd=128 ctx>=512",
    "writer": "generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_guard": "tinygrad/llm/decode_routes.py attention live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0; covers 14B Hq=40/G=5): QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_LIVE_SPLIT default 1; FlashDecodeAttentionSpec owns staging/geometry/combine",
    "kernel_source": "extra/qk/flash_decode_attention_spec.py FlashDecodeAttentionSpec -> live-split block tile path (staging='KV_BOTH', seqlen-bound per-split length)",
    "authority_artifact": "bench/gp-track/gp4_latest.json (GP4_PASS_TIER_A); W==D 8B/14B/32B token-identical to generic flash ref; 14B tok/s flat 69.24@MAXC1024 vs 69.04@MAXC8192 (no-collapse)",
    "rollback_flag": "DECODE_LIVE_SPLIT=0 exits the live-split default; no manifest fallback route row remains",
    "next_action": "keep promoted; 8B/14B/32B now share one modular structural-class route",
  }),
  # ----- prefill GEMM -----
  ("prefill_wmma_lds_dbuf_generated", {
    "shape_guard": "gfx1100 pp512 exact candidate set: attn_qo, attn_kv, ffn_down, ffn_gate_up",
    "writer": "tinygrad_generated",
    "selector": "promoted_candidate_set",
    "route_guard": "selected model inventory + scanned target + memory admission produce exact per-linear bindings",
    "kernel_source": "ordinary tinygrad Tensor matmul lowered with exact candidate-owned generated WMMA-LDS warmstart context",
    "authority_artifact": "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/whole-model-quality.json + whole-prefill-pinned.json",
    "rollback_flag": "none; absent exact binding selects the ordinary scheduler fallback",
    "next_action": "keep promoted while whole-model parity and pinned speed gates remain green; replace only with another qualified generated candidate set",
  }),
  ("prefill_v2_scheduler_matmul_default", {
    "shape_guard": "fp16 resident/chunked matmul outside promoted exact candidate applicability or memory admission",
    "writer": "tinygrad_generated",
    "selector": "planner_fallback",
    "route_guard": "tinygrad/llm/prefill_routes.py fallback path when no exact generated binding is attached",
    "kernel_source": "ordinary tinygrad graph lowering with the selected prefill warmstart table",
    "authority_artifact": "docs/pure-machine-search.md",
    "rollback_flag": "none",
    "next_action": "retain as a pure fallback for unsupported or memory-inadmissible shapes",
  }),
  ("prefill_q4k_direct_tile4x4_default", {
    "shape_guard": "direct-packed Q4_K prefill, memory-safe 14B/32B route",
    "writer": "generated",
    "selector": "env_default",
    "route_guard": "tinygrad/llm/prefill_routes.py Q4_K direct-packed default -> Q4KPrefillRouteSpec + emit_q4k_packed_prefill_kernel; _direct_packed_opts selects LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4",
    "kernel_source": "extra/qk/q4k_prefill_route_spec.py emit_q4k_packed_prefill_kernel (spec-driven lowering of Q4KPrefillRouteSpec -> q4k_gen_prefill_direct_out_* / q4k_gen_prefill_partials_*)",
    "authority_artifact": "docs/prefill-lessons-ledger.md",
    "rollback_flag": "PREFILL_Q4K_DIRECT_SCHEDULE=legacy",
    "next_action": "keep generated descriptor binding; Q4_K int8-WMMA remains a separate research substrate, not the shipped default",
  }),
  ("prefill_q6k_direct_generated", {
    "shape_guard": "direct-packed Q6_K prefill, memory-safe route; default when PREFILL_DIRECT_QUANTS includes Q6_K and no resident fp16 weight is available",
    "writer": "generated",
    "selector": "env_default",
    "route_guard": "tinygrad/llm/prefill_routes.py Q6_K direct-packed branch: PREFILL_Q6K_PACKED_LOAD default-on -> Q6KPrefillRouteSpec + emit_q6k_packed_prefill_kernel; direct_out for parts==1/PREFILL_DIRECT_OUT=1, otherwise partials",
    "kernel_source": "extra/qk/q6k_prefill_route_spec.py emit_q6k_packed_prefill_kernel (spec-driven lowering of Q6KPrefillRouteSpec -> q6k_gen_prefill_direct_out_* / q6k_gen_prefill_partials_*)",
    "authority_artifact": "test/unit/test_q6k_prefill_route_spec.py + test/unit/test_llm_prefill_routes.py",
    "rollback_flag": "PREFILL_Q6K_PACKED_LOAD=0 reaches the legacy non-packed debug path; no manifest default rollback remains",
    "next_action": "keep generated descriptor binding; Q6_K direct prefill is no longer unmanifested runtime handwritten debt",
  }),
]

def _census_row(route_id: str, overlay: dict) -> dict:
  """Merge a census overlay with the manifest route into one census row. Manifest is the source for
  workload/role/quant/purity_status; the overlay carries census-specific commentary. Key order is fixed so the
  emitted JSON/MD is byte-stable."""
  m = ROUTES[route_id]  # KeyError here means an overlay names a route the manifest does not define
  return {"route_id": route_id, "workload": m["workload"], "role": ",".join(m["roles"]),
          "quant": ",".join(m["quant"]),
          "shape_guard": overlay["shape_guard"], "writer": overlay["writer"], "selector": overlay["selector"],
          "route_guard": overlay["route_guard"], "kernel_source": overlay["kernel_source"],
          "authority_artifact": overlay["authority_artifact"], "rollback_flag": overlay["rollback_flag"],
          "purity_status": derive_purity_status(m["status"], m["provenance"]),
          "next_action": overlay["next_action"]}

CENSUS_ROWS = [_census_row(rid, overlay) for rid, overlay in CENSUS_OVERLAY]

# tinygrad-scheduler-generated coverage (the rest of the model): not enumerated as hot-kernel rows but recorded so the
# census answers "what is already tinygrad_generated".
TINYGRAD_GENERATED_COVERAGE = {
  "writer": "tinygrad_generated", "selector": "tinygrad_scheduler",
  "covers": ["rmsnorm", "rope/position", "q/k/v + o residual elementwise", "kv cache write path",
             "short-context attention (ctx<512)", "all graph ops not in the hot-kernel rows above"],
  "note": "generated by tinygrad/codegen via the scheduler; no hand-written kernel. Already 'generated enough' for this scope.",
  "purity_status": "tinygrad_generated"}

def build_census() -> dict:
  manifest_errors = validate_manifest()
  defaults = set(default_routes())
  # cross-check: every census route_id must exist in the manifest, and current_default must match manifest status.
  manifest_default = {rid for rid, r in ROUTES.items() if r["status"] in ("promoted_default", "default_shipped")}
  rows = []
  attribution_complete = not manifest_errors
  for r in CENSUS_ROWS:
    rid = r["route_id"]
    in_manifest = rid in ROUTES
    is_default = rid in defaults
    row = dict(r)
    row["current_default"] = is_default
    row["in_manifest"] = in_manifest
    row["provenance"] = route_provenance(rid) if in_manifest else "missing_manifest"
    row["replacement_scope"] = ROUTES.get(rid, {}).get("replacement_scope", "")
    row["final_default_allowed"] = row["provenance"] in ("machine_authored_generated", "tinygrad_scheduler_generated")
    # route attribution must be present (a model.py / route-file guard), else flag incomplete
    if not row.get("route_guard") or not in_manifest:
      attribution_complete = False
      row["attribution_missing"] = True
    rows.append(row)
  # also assert no manifest default route is missing from the census
  missing_from_census = sorted(manifest_default - {r["route_id"] for r in CENSUS_ROWS})
  if missing_from_census: attribution_complete = False

  default_rows = [r for r in rows if r["current_default"]]
  fallback_rows = [r for r in rows if not r["current_default"]]
  non_tinygrad_default = [r for r in default_rows if r["writer"] != "tinygrad_generated"]
  generated_default = [r for r in default_rows if r["provenance"] == "machine_authored_generated"]
  final_purity_debt = [r for r in default_rows if not r["final_default_allowed"]]
  transitional_default = [r for r in default_rows if r["provenance"] == "hand_authored_uop_template"]
  forbidden_default = [r for r in default_rows if r["provenance"] in ("external_handwritten_kernel", "rollback_oracle")]
  purity = default_purity_report()
  debt_clause = (
    "0 are final-default purity debt"
    if not final_purity_debt else
    f"{len(final_purity_debt)} {'is' if len(final_purity_debt) == 1 else 'are'} final-default purity debt "
    f"({', '.join(r['route_id'] for r in final_purity_debt)})"
  )

  verdict = "PMS_R0_PASS_CENSUS_PINNED" if attribution_complete else "PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING"
  headline = {
    "question": "Which non-tinygrad-generated kernels run on the DEFAULT path?",
    "count_non_tinygrad_generated_default_kernels": len(non_tinygrad_default),
    "non_tinygrad_generated_default_route_ids": [r["route_id"] for r in non_tinygrad_default],
    "of_which_search_generated": [r["route_id"] for r in generated_default],
    "of_which_final_purity_debt": [r["route_id"] for r in final_purity_debt],
    "of_which_transitional": [r["route_id"] for r in transitional_default],
    "of_which_forbidden_final_default": [r["route_id"] for r in forbidden_default],
    "interpretation": (
      f"{len(non_tinygrad_default)} kernels on the default path are non-tinygrad-generated. "
      f"{len(generated_default)} are machine-authored/generated ({', '.join(r['route_id'] for r in generated_default)}); "
      f"{debt_clause}. "
      "Everything else in the model is tinygrad_scheduler-generated."),
  }
  return {
    "scope": "PMS-R0 default-path kernel census",
    "profile_decode": "qwen3_8b_q4_k_m_gfx1100_decode", "profile_prefill": "qwen3_8b_q4_k_m_gfx1100_prefill",
    "verdict": verdict,
    "strict_default_purity_verdict": purity["verdict"],
    "manifest_errors": manifest_errors,
    "attribution_complete": attribution_complete,
    "missing_from_census": missing_from_census,
    "headline": headline,
    "purity_report": purity,
    "rows": rows,
    "default_route_table": default_rows,
    "fallback_table": fallback_rows,
    "tinygrad_generated_coverage": TINYGRAD_GENERATED_COVERAGE,
    "source": "derived from tinygrad/llm/decode_routes.py route guards + extra/qk/gemv_g3_codegen_lowering.py + extra/qk/prefill_graph_gemm_route.py + live-split attention route files; cross-checked vs extra/qk/route_manifest.py",
  }

def _md(c: dict) -> str:
  L = ["# PMS-R0 Default-Path Kernel Census", "",
       f"Verdict: **{c['verdict']}**", "",
       f"Strict default purity: **{c['strict_default_purity_verdict']}**", "",
       f"Headline: {c['headline']['interpretation']}", "",
       "## Default-path routes", "",
       "| route_id | workload | provenance | final default? | selector | quant | authority | rollback |",
       "|---|---|---|---|---|---|---|---|"]
  for r in c["default_route_table"]:
    final = "yes" if r["final_default_allowed"] else "no"
    L.append(f"| {r['route_id']} | {r['workload']} | {r['provenance']} | {final} | {r['selector']} | {r['quant']} | "
             f"{r['authority_artifact'].split(' (')[0]} | {r['rollback_flag']} |")
  L += ["", "## Fallback / reference / refuted / research routes (NOT default path)", "",
        "| route_id | provenance | purity_status | next_action |", "|---|---|---|---|"]
  for r in c["fallback_table"]:
    L.append(f"| {r['route_id']} | {r['provenance']} | {r['purity_status']} | {r['next_action']} |")
  L += ["", "## Strict-purity debt", ""]
  for r in c["purity_report"]["rows"]:
    if not r["final_default_allowed"]:
      scope = r.get("replacement_scope") or "missing"
      L.append(f"- **{r['route_id']}**: `{r['provenance']}`; replacement scope: {scope}")
  L += ["", "## Route attribution (cited guards)", ""]
  for r in c["rows"]:
    L.append(f"- **{r['route_id']}** ({'default' if r['current_default'] else 'fallback'}): {r['route_guard']}")
  if c["manifest_errors"]:
    L += ["", "## Manifest errors", ""]
    L += [f"- {x}" for x in c["manifest_errors"]]
  L += ["", "## tinygrad-scheduler coverage", "",
        f"Writer `tinygrad_generated` covers: {', '.join(c['tinygrad_generated_coverage']['covers'])}.", ""]
  return "\n".join(L)

def main(argv=None):
  import argparse
  ap = argparse.ArgumentParser(description="Audit tinygrad default-route provenance and final default purity.")
  ap.add_argument("--check", action="store_true", help="fail on manifest/census attribution drift")
  ap.add_argument("--strict-final-default", action="store_true",
                  help="also fail unless every selected default is final-pure generated/tinygrad-scheduler output")
  args = ap.parse_args(argv)
  OUT.mkdir(parents=True, exist_ok=True)
  c = build_census()
  json.dump(c, open(OUT / "latest.json", "w"), indent=2)
  json.dump(c["default_route_table"], open(OUT / "default_route_table.json", "w"), indent=2)
  json.dump(c["fallback_table"], open(OUT / "fallback_table.json", "w"), indent=2)
  open(OUT / "summary.md", "w").write(_md(c))
  print(f"wrote census to {OUT}")
  print("verdict:", c["verdict"])
  print("strict_default_purity:", c["strict_default_purity_verdict"])
  print(c["headline"]["interpretation"])
  if args.check and c["verdict"] != "PMS_R0_PASS_CENSUS_PINNED":
    raise SystemExit(1)
  if args.strict_final_default and c["strict_default_purity_verdict"] != "TINYGRAD_DEFAULT_PURITY_PASS":
    raise SystemExit(2)

if __name__ == "__main__":
  main()
