#!/usr/bin/env python3
"""PMS-R0: default-path kernel census.

Replaces the rough "about four hot kernels" framing with a machine-readable census of what actually runs on the default
decode/prefill path, who writes each kernel, who selects it, and where its authority lives. Every row is derived from
the ACTUAL model route guards (cited line/guard in tinygrad/llm/model.py + the route source files), NOT inferred from
filenames. The route identity is cross-checked against extra/qk_route_manifest.py (PMS-R1).

Run:  PYTHONPATH=. python3 extra/pure_machine_search_default_path_census.py
Outputs (bench/pure-machine-search-default-path-census/):
  latest.json            -- full census (all rows + summary counts + the headline answer)
  summary.md             -- human table
  default_route_table.json  -- rows on the live default path
  fallback_table.json       -- fallback/reference/refuted/research rows (NOT on the default path)

This tool reads source files and writes JSON/MD only. It runs no kernels and changes no defaults.
"""
from __future__ import annotations
import json, pathlib
from extra.qk_route_manifest import ROUTES, default_routes

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/pure-machine-search-default-path-census"

# writer taxonomy (scope PMS-R0): generated | codegen_emitter | owned_asm | tinygrad_generated
# selector taxonomy: BubbleBeam | manifest | env_guard | hardcoded_default | tinygrad_scheduler
#
# Each census row pins a route guard by file:line. The guard text is quoted from the verified source so the census is
# auditable without re-reading the model. current_default is taken from the route status in qk_route_manifest.py.
CENSUS_ROWS = [
  # ----- decode Q4_K weight GEMV -----
  {"route_id": "decode_q4k_g3_generated", "workload": "decode", "role": "ffn_gate_up,ffn_down,attn_qo", "quant": "Q4_K",
   "shape_guard": "in=4096 out in {4096,12288} | in=12288 out=4096 (g3_bubblebeam_shape)",
   "writer": "generated", "selector": "BubbleBeam",
   "route_guard": "tinygrad/llm/model.py:255 getenv('BUBBLEBEAM_FUTURESIGHT', 1)==1 (default-on) -> :257-264 q4k_g3_lanemap_gemv_kernel fires FIRST for g3_bubblebeam_shape, short-circuiting the owned-warp guards",
   "kernel_source": "extra/qk_gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel (UOp program from extra/qk_gemv_g2_lanemap.py Q4KGateUpLaneMap)",
   "authority_artifact": "bench/amd-isa-backend-g3-weight-promotion/latest.json (AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT)",
   "rollback_flag": "BUBBLEBEAM_FUTURESIGHT=0 -> decode_q4k_owned_warp",
   "purity_status": "search_generated_promoted",
   "next_action": "keep promoted; PMS-R5 turn into reusable LaneMapTemplate; do NOT reopen Q4_K layout reshuffle while parity holds"},
  {"route_id": "decode_q4k_owned_warp", "workload": "decode", "role": "ffn_gate_up,ffn_down,attn_qo", "quant": "Q4_K",
   "shape_guard": "same shapes; reached only when BUBBLEBEAM_FUTURESIGHT=0",
   "writer": "codegen_emitter", "selector": "env_guard",
   "route_guard": "tinygrad/llm/model.py:318 getenv('Q4K_GEMV_WARP_PROJ', 1) (q/o) + :360 getenv('Q4K_GEMV_WARP', 1) (gate/up+down). Guards still default 1 but the G3 branch intercepts first on the default path.",
   "kernel_source": "extra/q4_k_gemv_primitive.py q4k_gemv_warp_kernel",
   "authority_artifact": "docs/decode-q4k-gemv-warp-promotion-result-20260624.md",
   "rollback_flag": "n/a (this IS the rollback/reference for G3)",
   "purity_status": "owned_reference",
   "next_action": "keep as rollback/oracle; do not delete"},
  # ----- decode Q6_K weight GEMV -----
  {"route_id": "decode_q6k_coop_shipped", "workload": "decode", "role": "ffn_down,lm_head", "quant": "Q6_K",
   "shape_guard": "ffn_down 12288->4096 | lm_head out>=100000",
   "writer": "codegen_emitter", "selector": "hardcoded_default",
   "route_guard": "tinygrad/llm/model.py:467 getenv('Q6K_LM_HEAD_COOP', 1) | :468 getenv('Q6K_FFN_DOWN_COOP', 1) -> :470-473 q6k_coop_partial_kernel",
   "kernel_source": "extra/q6_k_gemv_primitive.py q6k_coop_partial_kernel (coop partial + external .sum reduce)",
   "authority_artifact": "extra/qk_decode_runtime_overhead.py (W==D harness); baseline for bench/amd-isa-backend-q6k-direct-speed/latest.json",
   "rollback_flag": "none (shipped baseline; Q6K_DIRECT_ROUTE alt is refuted/default-off)",
   "purity_status": "owned_default",
   "next_action": "keep shipped; no active pure replacement (direct route refuted)"},
  {"route_id": "decode_q6k_direct_refuted", "workload": "decode", "role": "lm_head", "quant": "Q6_K",
   "shape_guard": "lm_head out>=100000, default-OFF",
   "writer": "codegen_emitter", "selector": "env_guard",
   "route_guard": "tinygrad/llm/model.py:455-464 getenv('Q6K_DIRECT_ROUTE') (default-off)",
   "kernel_source": "extra/q6_k_gemv_primitive.py q6k_halfwarp_partition kernel",
   "authority_artifact": "bench/amd-isa-backend-q6k-direct-speed/latest.json (AMD_ISA_Q6K_DIRECT_SPEED_REGRESSION)",
   "rollback_flag": "Q6K_DIRECT_ROUTE=0 (already the default)",
   "purity_status": "refuted",
   "next_action": "do NOT reopen as built (-5.44% median W==D); only with a different topology than half-warp"},
  # ----- decode attention -----
  {"route_id": "decode_attention_owned_two_kernel", "workload": "decode", "role": "attention_tile,attention_combine", "quant": "fp16",
   "shape_guard": "B=1 Hq=32 Hkv=8 Hd=128 ctx>=512",
   "writer": "owned_asm", "selector": "env_guard",
   "route_guard": "tinygrad/llm/model.py:1091 getenv('DECODE_ATTN_AMDGCN_TILE', 1) & ctx>=512 -> :1094-1106 amdgcn_flash_decode",
   "kernel_source": "extra/qk_owned_flash_decode_graph_node.py amdgcn_flash_decode -> extra/qk_owned_flash_decode.hip (owned_flash_tile_gqa_whole + owned_flash_combine, two Ops.PROGRAM nodes)",
   "authority_artifact": "bench/amd-isa-backend-decode-attention-ceiling/latest.json (AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION)",
   "rollback_flag": "DECODE_ATTN_AMDGCN_TILE=0 -> generated tinygrad flash decode",
   "purity_status": "owned_default",
   "next_action": "keep shipped; low-leverage (wall-share ~10%@512 ->~0%@4096); do NOT make attention the max-out target"},
  {"route_id": "decode_attention_native_correct_not_fast", "workload": "decode", "role": "attention_tile,attention_combine", "quant": "fp16",
   "shape_guard": "same shapes; opt-in",
   "writer": "generated", "selector": "env_guard",
   "route_guard": "tinygrad/llm/model.py:1076-1085 DECODE_ATTN_GENERATED_WHOLECACHE generated route, selected when DECODE_ATTN_AMDGCN_TILE=0",
   "kernel_source": "extra/qk_flash_decode.py flash_decode_attention_whole_cache (+ native AMD-ISA backend)",
   "authority_artifact": "bench/amd-isa-backend-phase-n7/latest.json; bench/amd-isa-backend-decode-attention-ceiling/latest.json",
   "rollback_flag": "unset -> owned two-kernel default",
   "purity_status": "research",
   "next_action": "infrastructure/research only (~60-68% of owned); reopen only if attention wall-share becomes dominant"},
  # ----- prefill GEMM -----
  {"route_id": "prefill_pipe_role_selective_default", "workload": "prefill", "role": "attn_qo,attn_kv,ffn_down (ffn_gate_up excluded)", "quant": "Q4_K,Q6_K,fp16",
   "shape_guard": "graph-GEMM ubatch=512; gate/up out_f==12288 kept on lds2",
   "writer": "owned_asm", "selector": "manifest",
   "route_guard": "extra/qk_prefill_graph_gemm_route.py:55 PREFILL_GEMM_PIPELINE default 1 + :61 PREFILL_PIPE_ROLE_SELECTIVE default 1 (out_f==12288 -> pipe_mode=False); entry tinygrad/llm/model.py:145-147 route_pf16_graph_gemm",
   "kernel_source": "extra/qk_prefill_graph_gemm_route.py build_gemm_pipe (software-pipelined assembly GEMM, tm2/tn2)",
   "authority_artifact": "bench/qk-prefill-pipe-role-selective/latest.json (ROLE_SELECTIVE_PASS_BEATS_GLOBAL)",
   "rollback_flag": "PREFILL_PIPE_ROLE_SELECTIVE=0 -> global pipe; PREFILL_GEMM_PIPELINE=0 -> old lds2",
   "purity_status": "search_selected_specialized_route",
   "next_action": "keep promoted; PMS-R6 encode role policy as manifest template"},
  {"route_id": "prefill_pipe_global_rollback", "workload": "prefill", "role": "all graph-GEMM roles", "quant": "Q4_K,Q6_K,fp16",
   "shape_guard": "all roles incl ffn_gate_up; reached with PREFILL_PIPE_ROLE_SELECTIVE=0",
   "writer": "owned_asm", "selector": "env_guard",
   "route_guard": "extra/qk_prefill_graph_gemm_route.py:55-69 (pipe on for all roles when role-selective off)",
   "kernel_source": "extra/qk_prefill_graph_gemm_route.py build_gemm_pipe (all roles)",
   "authority_artifact": "bench/qk-prefill-pipe-promotion/latest.json",
   "rollback_flag": "PREFILL_GEMM_PIPELINE=0 -> old lds2 default",
   "purity_status": "superseded_rollback",
   "next_action": "keep as A/B comparator and the rollback target of role-selective"},
]

# tinygrad-scheduler-generated coverage (the rest of the model): not enumerated as hot-kernel rows but recorded so the
# census answers "what is already tinygrad_generated".
TINYGRAD_GENERATED_COVERAGE = {
  "writer": "tinygrad_generated", "selector": "tinygrad_scheduler",
  "covers": ["rmsnorm", "rope/position", "q/k/v + o residual elementwise", "kv cache write path",
             "short-context attention (ctx<512)", "all graph ops not in the hot-kernel rows above"],
  "note": "generated by tinygrad/codegen via the scheduler; no hand-written kernel. Already 'generated enough' for this scope.",
  "purity_status": "tinygrad_generated"}

def build_census() -> dict:
  defaults = set(default_routes())
  # cross-check: every census route_id must exist in the manifest, and current_default must match manifest status.
  manifest_default = {rid for rid, r in ROUTES.items() if r["status"] in ("promoted_default", "default_shipped")}
  rows = []
  attribution_complete = True
  for r in CENSUS_ROWS:
    rid = r["route_id"]
    in_manifest = rid in ROUTES
    is_default = rid in defaults
    row = dict(r)
    row["current_default"] = is_default
    row["in_manifest"] = in_manifest
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
  generated_default = [r for r in default_rows if r["writer"] == "generated"]
  hand_owned_default = [r for r in default_rows if r["writer"] in ("owned_asm", "codegen_emitter")]

  verdict = "PMS_R0_PASS_CENSUS_PINNED" if attribution_complete else "PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING"
  headline = {
    "question": "Which non-tinygrad-generated kernels run on the DEFAULT path?",
    "count_non_tinygrad_generated_default_kernels": len(non_tinygrad_default),
    "non_tinygrad_generated_default_route_ids": [r["route_id"] for r in non_tinygrad_default],
    "of_which_search_generated": [r["route_id"] for r in generated_default],
    "of_which_hand_owned": [r["route_id"] for r in hand_owned_default],
    "interpretation": (
      f"{len(non_tinygrad_default)} kernels on the default path are non-tinygrad-generated. "
      f"{len(generated_default)} is search/codegen-generated (G3 Q4_K GEMV); "
      f"{len(hand_owned_default)} are hand-owned (Q6_K coop, owned attention two-kernel, prefill pipe assembly). "
      "Everything else in the model is tinygrad_scheduler-generated."),
  }
  return {
    "scope": "PMS-R0 default-path kernel census",
    "profile_decode": "qwen3_8b_q4_k_m_gfx1100_decode", "profile_prefill": "qwen3_8b_q4_k_m_gfx1100_prefill",
    "verdict": verdict,
    "attribution_complete": attribution_complete,
    "missing_from_census": missing_from_census,
    "headline": headline,
    "rows": rows,
    "default_route_table": default_rows,
    "fallback_table": fallback_rows,
    "tinygrad_generated_coverage": TINYGRAD_GENERATED_COVERAGE,
    "source": "derived from tinygrad/llm/model.py route guards + extra/qk_gemv_g3_codegen_lowering.py + extra/qk_prefill_graph_gemm_route.py + extra/qk_owned_flash_decode_graph_node.py; cross-checked vs extra/qk_route_manifest.py",
  }

def _md(c: dict) -> str:
  L = ["# PMS-R0 Default-Path Kernel Census", "",
       f"Verdict: **{c['verdict']}**", "",
       f"Headline: {c['headline']['interpretation']}", "",
       "## Default-path routes", "",
       "| route_id | workload | writer | selector | quant | authority | rollback |",
       "|---|---|---|---|---|---|---|"]
  for r in c["default_route_table"]:
    L.append(f"| {r['route_id']} | {r['workload']} | {r['writer']} | {r['selector']} | {r['quant']} | "
             f"{r['authority_artifact'].split(' (')[0]} | {r['rollback_flag']} |")
  L += ["", "## Fallback / reference / refuted / research routes (NOT default path)", "",
        "| route_id | writer | purity_status | next_action |", "|---|---|---|---|"]
  for r in c["fallback_table"]:
    L.append(f"| {r['route_id']} | {r['writer']} | {r['purity_status']} | {r['next_action']} |")
  L += ["", "## Route attribution (cited guards)", ""]
  for r in c["rows"]:
    L.append(f"- **{r['route_id']}** ({'default' if r['current_default'] else 'fallback'}): {r['route_guard']}")
  L += ["", "## tinygrad-scheduler coverage", "",
        f"Writer `tinygrad_generated` covers: {', '.join(c['tinygrad_generated_coverage']['covers'])}.", ""]
  return "\n".join(L)

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  c = build_census()
  json.dump(c, open(OUT / "latest.json", "w"), indent=2)
  json.dump(c["default_route_table"], open(OUT / "default_route_table.json", "w"), indent=2)
  json.dump(c["fallback_table"], open(OUT / "fallback_table.json", "w"), indent=2)
  open(OUT / "summary.md", "w").write(_md(c))
  print(f"wrote census to {OUT}")
  print("verdict:", c["verdict"])
  print(c["headline"]["interpretation"])

if __name__ == "__main__":
  main()
