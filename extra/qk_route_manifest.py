"""Route manifest -- the single declarative source of truth for which decode/prefill routes exist on this fork, what
selects each one, what it rolls back to, and its current disposition. PMS-R1 of
docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md (supersedes the Phase-1 draft of
docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md).

This module is DATA + tiny helpers. It changes NO defaults and runs NO kernels; gates import it instead of copying
ad-hoc env maps. For each route, `env` is what you SET to force that route onto the active path; an empty `env` ({})
means the route is ALREADY the shipped default (no flag needed). `rollback` is the exact env to leave it.

CURRENT-STATE PIN (verified 2026-06-30 against tinygrad/llm/model.py + extra/qk_gemv_g3_codegen_lowering.py +
extra/qk_prefill_graph_gemm_route.py + extra/qk_owned_flash_decode_graph_node.py):

  * Decode Q4_K GEMV default = the GENERATED G3 LaneMap route. model.py:255 reads `getenv("BUBBLEBEAM_FUTURESIGHT", 1)`
    (DEFAULT-ON, flipped in commit 81370ae38). For the eligible Q4_K shapes (g3_bubblebeam_shape, model.py:256) the G3
    route at model.py:257-264 fires FIRST and short-circuits before the owned-warp guards (Q4K_GEMV_WARP_PROJ@318,
    Q4K_GEMV_WARP@360). So `decode_q4k_g3_generated` is the promoted default; the owned warp kernel is the
    rollback/reference one flag away (BUBBLEBEAM_FUTURESIGHT=0).
  * An earlier draft of this file had G3/owned default-status INVERTED (it predated the default flip). This version
    pins the real state: G3 = default, owned warp = rollback.

Token-identity / speed-equivalence proof: bench/amd-isa-backend-g3-weight-promotion/latest.json
(AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT; lag -0.13..+0.41% across ctx 512-4096; token_match + route_clean all ctx).
"""
from __future__ import annotations
import json, os, pathlib

PROFILE_DECODE = "qwen3_8b_q4_k_m_gfx1100_decode"
PROFILE_PREFILL = "qwen3_8b_q4_k_m_gfx1100_prefill"

# status vocabulary:
#   promoted_default        -> generated/search-selected route that is now the shipped default
#   default_shipped         -> shipped default whose writer is owned/hand asm (not yet replaced by a generated route)
#   rollback_reference      -> kept one flag away as the rollback/oracle for a promoted route; NOT on the default path
#   superseded_rollback     -> previously promoted, kept as the A/B rollback target for a newer default
#   refuted                 -> built, token-correct, route-bound, but speed-refuted; default-off, do not re-search as-is
#   correct_not_fast        -> generated/correct/route-bound but below promotion speed; infrastructure/research, not shipped
#
# purity_status vocabulary (docs/pure-machine-search.md definitions):
#   search_generated_promoted | owned_reference | owned_default | search_selected_specialized_route | refuted | research
ROUTES = {
  # ---------------- decode weight GEMV: Q4_K ----------------
  "decode_q4k_g3_generated": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "promoted_default",
    "roles": ["ffn_gate_up", "ffn_down", "attn_qo"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [
      {"role": "ffn_gate_up", "K": 4096, "N": 12288}, {"role": "ffn_down", "K": 12288, "N": 4096},
      {"role": "attn_qo", "K": 4096, "N": 4096}],
    "env": {},  # DEFAULT-ON: model.py:255 getenv("BUBBLEBEAM_FUTURESIGHT", 1). No flag needed.
    "rollback": {"BUBBLEBEAM_FUTURESIGHT": "0"},  # -> owned warp (decode_q4k_owned_warp)
    "strict_fallback": True,
    "expected_kernels": ["q4k_g3_lanemap_gemv_*"],
    "forbidden_kernels": ["q4k_gemv_warp_kernel (on the eligible roles)", "q4k_lane_partition_gemv_*", "fallback_graph"],
    "authority_gate": "extra/amd_isa_g3_weight_promotion_gate.py",
    "promotion_artifacts": ["bench/amd-isa-backend-g3-weight-promotion/latest.json",
                            "bench/amd-isa-backend-g3-weight-promotion/summary.md"],
    "purity_status": "search_generated_promoted",
    "selector": "BubbleBeam",
    "route_attribution": "tinygrad/llm/model.py:255-264 (g3 fires first for g3_bubblebeam_shape); writer extra/qk_gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel",
    "note": "generated wave32 UOp program lowered from the G2 Q4_K LaneMap (extra/qk_gemv_g2_lanemap.py). Speed-equivalent to owned warp (-0.13..+0.41% across ctx 512-4096), token-identical, route-clean. This is the closest thing to a pure-search default decode kernel."},
  "decode_q4k_owned_warp": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "rollback_reference",
    "roles": ["ffn_gate_up", "ffn_down", "attn_qo"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [
      {"role": "ffn_gate_up", "K": 4096, "N": 12288}, {"role": "ffn_down", "K": 12288, "N": 4096},
      {"role": "attn_qo", "K": 4096, "N": 4096}],
    "env": {"BUBBLEBEAM_FUTURESIGHT": "0"},  # SET this to force owned warp (the rollback for G3)
    "rollback": {},  # this IS the rollback target
    "strict_fallback": True,
    "expected_kernels": ["q4k_gemv_warp_4096_4096", "q4k_gemv_warp_kernel"],
    "authority_gate": "extra/amd_isa_g3_weight_promotion_gate.py",
    "promotion_artifacts": ["docs/decode-q4k-gemv-warp-promotion-result-20260624.md"],
    "purity_status": "owned_reference",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/model.py:318 (Q4K_GEMV_WARP_PROJ default 1, q/o) + :360 (Q4K_GEMV_WARP default 1, gate/up+down); reached only when BUBBLEBEAM_FUTURESIGHT=0 short-circuits the G3 branch. Writer extra/q4_k_gemv_primitive.py q4k_gemv_warp_kernel",
    "note": "hand-written owned warp GEMV. The Q4K_GEMV_WARP* guards still default to 1, but the G3 branch intercepts first for the eligible shapes when BUBBLEBEAM_FUTURESIGHT is on (the default). So owned warp is the rollback/reference, not the live default."},
  # ---------------- decode weight GEMV: Q6_K ----------------
  "decode_q6k_coop_shipped": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "default_shipped",
    "roles": ["ffn_down", "lm_head"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"role": "ffn_down", "K": 12288, "N": 4096}, {"role": "lm_head", "N": ">=100000"}],
    "env": {},  # Q6K_LM_HEAD_COOP / Q6K_FFN_DOWN_COOP both default to 1
    "rollback": {},  # no rollback flag: this is the shipped baseline; Q6_K direct (refuted) is the only alt route
    "strict_fallback": True,
    "expected_kernels": ["q6k_coop_partial_*"],
    "authority_gate": "extra/qk_decode_runtime_overhead.py",
    "promotion_artifacts": [],
    "purity_status": "owned_default",
    "selector": "hardcoded_default",
    "route_attribution": "tinygrad/llm/model.py:465-473 (Q6K_LM_HEAD_COOP@467, Q6K_FFN_DOWN_COOP@468 default 1); writer extra/q6_k_gemv_primitive.py q6k_coop_partial_kernel",
    "note": "shipped Q6_K route (coop partial + external .sum reduce) for FFN down / lm_head. Baseline the refuted direct route was measured against."},
  "decode_q6k_direct_refuted": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "refuted",
    "roles": ["lm_head"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"role": "lm_head", "N": ">=100000"}],
    "env": {"Q6K_DIRECT_ROUTE": "1"}, "rollback": {"Q6K_DIRECT_ROUTE": "0"},
    "strict_fallback": True,
    "expected_kernels": ["q6k_halfwarp_partition_151936_4096"],
    "authority_gate": "extra/qk_decode_runtime_overhead.py",
    "promotion_artifacts": ["bench/amd-isa-backend-q6k-direct-speed/latest.json",
                            "bench/amd-isa-backend-q6k-direct-speed/summary.md"],
    "purity_status": "refuted",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/model.py:455-464 (Q6K_DIRECT_ROUTE default-off); refuted vs decode_q6k_coop_shipped baseline.",
    "note": "half-warp direct Q6_K lm_head route: token-correct + route-bound, but W==D regressed -4.77..-6.06% (median -5.44%). Default-off. Do NOT re-chase as built (only reopen with a different topology than the half-warp partition)."},
  # ---------------- decode attention ----------------
  "decode_attention_owned_two_kernel": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "default_shipped",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "ctx": ">=512"}],
    "env": {},  # DECODE_ATTN_AMDGCN_TILE defaults to 1 at ctx>=DECODE_ATTN_AMDGCN_MIN_CTX (512)
    "rollback": {"DECODE_ATTN_AMDGCN_TILE": "0"},  # -> generated tinygrad flash decode path
    "strict_fallback": True,
    "expected_kernels": ["owned_flash_tile_gqa_whole", "owned_flash_combine"],
    "authority_gate": "extra/qk_decode_runtime_overhead.py",
    "promotion_artifacts": ["docs/decode-two-kernel-problem-audit-result-20260625.md",
                            "bench/amd-isa-backend-decode-attention-ceiling/latest.json"],
    "purity_status": "owned_default",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/model.py:1091-1106 (DECODE_ATTN_AMDGCN_TILE default 1, ctx>=512); writer extra/qk_owned_flash_decode_graph_node.py amdgcn_flash_decode (HIP .co split tile + separate combine, two Ops.PROGRAM graph nodes).",
    "note": "shipped decode attention: hand HIP split tile + separate combine. Combine/fused-lifecycle exhausted; ceiling audit (AMD_ISA_ATTENTION_CEILING_PASS_MOVE_TO_NON_ATTENTION) says attention wall-share is ~10%@ctx512 ->~0%@ctx4096; low-leverage."},
  "decode_attention_native_correct_not_fast": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "correct_not_fast",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "ctx": ">=512"}],
    "env": {"DECODE_ATTN_AMDGCN_TILE": "0", "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
            "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE": "1"},
    "rollback": {},  # -> owned two-kernel default
    "strict_fallback": True,
    "authority_gate": "extra/qk_decode_runtime_overhead.py",
    "promotion_artifacts": ["bench/amd-isa-backend-phase-n7/latest.json",
                            "bench/amd-isa-backend-decode-attention-ceiling/latest.json"],
    "purity_status": "research",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/model.py:1076-1085 (DECODE_ATTN_GENERATED_WHOLECACHE generated whole-cache flash decode) selected when DECODE_ATTN_AMDGCN_TILE=0.",
    "note": "native AMD-ISA / generated attention tile: correct + route-bound but ~60-68% of owned speed (native_vs_owned 68.3%@512, 60.1%@4096). Infrastructure/capability, not shipped. Low-leverage per ceiling audit."},
  # ---------------- prefill GEMM ----------------
  "prefill_pipe_role_selective_default": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["attn_qo", "attn_kv", "ffn_down"], "excluded_roles": ["ffn_gate_up"],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "graph-GEMM prefill ubatch=512; gate/up out_f==12288 kept on lds2"}],
    "env": {},  # PREFILL_GEMM_PIPELINE=1 and PREFILL_PIPE_ROLE_SELECTIVE=1 are BOTH the default now
    "rollback": {"PREFILL_PIPE_ROLE_SELECTIVE": "0"},  # -> global pipe (prefill_pipe_global_rollback)
    "strict_fallback": True,
    "expected_roles_pipe": ["attn_qo", "attn_kv", "ffn_down"], "excluded_roles_pipe": ["ffn_gate_up"],
    "authority_gate": "extra/qk_prefill_whole_synced.py",
    "promotion_artifacts": ["bench/qk-prefill-pipe-role-selective/latest.json",
                            "bench/qk-prefill-pipe-role-selective/summary.md"],
    "purity_status": "search_selected_specialized_route",
    "selector": "manifest",
    "route_attribution": "extra/qk_prefill_graph_gemm_route.py:55 (PREFILL_GEMM_PIPELINE default 1) + :61 (PREFILL_PIPE_ROLE_SELECTIVE default 1 -> gate/up out_f==12288 forced pipe_mode=False); entry tinygrad/llm/model.py:145-147 route_pf16_graph_gemm.",
    "note": "shipped prefill default: software-pipelined assembly GEMM (build_gemm_pipe tm2/tn2) applied role-selectively (gate/up kept on its faster lds2 path). ROLE_SELECTIVE_PASS_BEATS_GLOBAL: +2.9..3.7% over global pipe, +11.7..23.4% over old lds2 default, through ctx8192. Output-equivalent, spread <=0.3%."},
  "prefill_pipe_global_rollback": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "superseded_rollback",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*"}],
    "env": {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_PIPE_ROLE_SELECTIVE": "0"},
    "rollback": {"PREFILL_GEMM_PIPELINE": "0"},  # -> old lds2 default
    "strict_fallback": True,
    "authority_gate": "extra/qk_prefill_whole_synced.py",
    "promotion_artifacts": ["bench/qk-prefill-pipe-promotion/latest.json",
                            "bench/qk-prefill-pipe-promotion/summary.md"],
    "purity_status": "search_selected_specialized_route",
    "selector": "env_guard",
    "route_attribution": "extra/qk_prefill_graph_gemm_route.py:55 (pipe on for all roles when PREFILL_PIPE_ROLE_SELECTIVE=0).",
    "note": "global pipe (all roles): was TIER_A vs old lds2 default (+8.5..19.2%), superseded by role-selective (which excludes the saturated gate/up where pipe regressed ~17%). Kept as the A/B rollback comparator and the rollback target of role-selective."},
}

# Closed / refuted axes -- do not re-search without a NEW premise (PMS-R3 do_not_search carries these forward).
REFUTED = [
  {"axis": "q6k_direct_half_warp_route", "disposition": "refuted: W==D regression -4.77..-6.06% (median -5.44%)",
   "citation": "bench/amd-isa-backend-q6k-direct-speed/latest.json", "route_id": "decode_q6k_direct_refuted"},
  {"axis": "q4k_offline_layout_reshuffle", "disposition": "deprioritized: G3 matches owned, no layout gap to recover",
   "citation": "bench/amd-isa-backend-g3-weight-promotion/search_space_update.json"},
  {"axis": "attention_combine_fused_lifecycle", "disposition": "exhausted/low-leverage (combine overlaps in-graph; fused is codegen-walled)",
   "citation": "docs/decode-two-kernel-problem-audit-result-20260625.md"},
  {"axis": "native_attention_as_default", "disposition": "correct_not_fast (~60-68% of owned)",
   "citation": "bench/amd-isa-backend-phase-n7/latest.json", "route_id": "decode_attention_native_correct_not_fast"},
  {"axis": "n1b_scalar_address_path", "disposition": "refuted/dead", "citation": "bench/amd-isa-backend-phase-n1b/latest.json"},
  {"axis": "occupancy_lds_only_attention_tuning", "disposition": "refuted: no W==D movement", "citation": "bench/amd-isa-backend-phase-m/latest.json"},
  {"axis": "scheduler_only_attention_tuning", "disposition": "small/no movement", "citation": "bench/amd-isa-backend-phase-k/latest.json"},
]

# ---- tiny helpers (no side effects unless you call apply_route) ----
def route(route_id: str) -> dict:
  if route_id not in ROUTES: raise KeyError(f"unknown route_id {route_id!r}; known: {sorted(ROUTES)}")
  return ROUTES[route_id]

def route_env(route_id: str) -> dict:
  """The env vars to SET to force this route onto the active path ({} means it is the shipped default)."""
  return dict(route(route_id).get("env", {}))

def rollback_env(route_id: str) -> dict:
  """The env vars that leave this route for its rollback target ({} means it IS a rollback target)."""
  return dict(route(route_id).get("rollback", {}))

def apply_route(route_id: str, env: dict | None = None) -> dict:
  """Materialize a route's env onto a copy of `env` (default: a copy of os.environ). Returns the new env; does NOT
  mutate os.environ unless you pass it explicitly. strict_fallback routes set QK_STRICT_FALLBACK=1 (fail-loud)."""
  out = dict(os.environ if env is None else env)
  out.update({k: str(v) for k, v in route_env(route_id).items()})
  if route(route_id).get("strict_fallback"): out.setdefault("QK_STRICT_FALLBACK", "1")
  return out

def is_refuted(axis: str) -> bool:
  return any(r["axis"] == axis for r in REFUTED)

def default_routes() -> list[str]:
  """Routes on the live default path (promoted generated default OR owned shipped default)."""
  return [rid for rid, r in ROUTES.items() if r["status"] in ("promoted_default", "default_shipped")]

def routes_by_status(status: str) -> list[str]:
  return [rid for rid, r in ROUTES.items() if r["status"] == status]

def to_manifest_dict() -> dict:
  return {"_schema": "default route manifest (PMS-R1)", "generated_by": "extra/qk_route_manifest.py",
          "profiles": {"decode": PROFILE_DECODE, "prefill": PROFILE_PREFILL},
          "routes": ROUTES, "refuted_axes": REFUTED,
          "default_routes": default_routes(),
          "promoted_defaults": routes_by_status("promoted_default"),
          "owned_defaults": routes_by_status("default_shipped")}

def dump(out_path: str | None = None) -> str:
  """Write the canonical manifest json (bench/qk-search-spaces/default_route_manifest.json by default)."""
  root = pathlib.Path(__file__).resolve().parents[1]
  p = pathlib.Path(out_path) if out_path else (root / "bench/qk-search-spaces/default_route_manifest.json")
  p.parent.mkdir(parents=True, exist_ok=True)
  json.dump(to_manifest_dict(), open(p, "w"), indent=2)
  return str(p)

if __name__ == "__main__":
  path = dump()
  print(f"wrote default route manifest to {path}")
  print("default routes:", default_routes())
  print("promoted (generated/search-selected) defaults:", routes_by_status("promoted_default"))
  print(f"{len(ROUTES)} routes, {len(REFUTED)} refuted axes")
