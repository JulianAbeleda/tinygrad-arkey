"""Route manifest -- the single declarative source of truth for which decode/prefill routes exist on this fork, what
selects each one, what it rolls back to, and its current disposition. PMS-R1 of
docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md (supersedes the Phase-1 draft of
docs/claude-active-work-audit-and-agnostic-search-scope-20260630.md).

This module is DATA + tiny helpers. It changes NO defaults and runs NO kernels; gates import it instead of copying
ad-hoc env maps. For each route, `env` is what you SET to force that route onto the active path; an empty `env` ({})
means the route is ALREADY the shipped default (no flag needed). `rollback` is the exact env to leave it.

CURRENT-STATE PIN (verified 2026-07-03 against tinygrad/llm/decode_routes.py + extra/qk/gemv_g3_codegen_lowering.py +
extra/qk/prefill_graph_gemm_route.py + generated attention routes):

  * Decode Q4_K GEMV default = the GENERATED G3 LaneMap route. decode_routes.py:q4k_primitive_linear_call reads
    `getenv("BUBBLEBEAM_FUTURESIGHT", 1)` (DEFAULT-ON, flipped in commit 81370ae38). For eligible Q4_K shapes
    the G3 route fires FIRST and short-circuits before the owned-warp guards. So `decode_q4k_g3_generated` is the
    promoted default; the owned warp kernel is the rollback/reference one flag away (BUBBLEBEAM_FUTURESIGHT=0).
  * An earlier draft of this file had G3/owned default-status INVERTED (it predated the default flip). This version
    pins the real state: G3 = default, owned warp = rollback.

Token-identity / speed-equivalence proof: bench/amd-isa-backend-g3-weight-promotion/latest.json
(AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT; lag -0.13..+0.41% across ctx 512-4096; token_match + route_clean all ctx).
"""
from __future__ import annotations
import json, os, pathlib

PROFILE_DECODE = "qwen3_8b_q4_k_m_gfx1100_decode"
PROFILE_DECODE_LARGE = "qwen3_14b_32b_q4_k_m_gfx1100_decode"
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
#
# provenance vocabulary (strict default-purity audit):
#   machine_authored_generated  -> emitted from profile/grammar/search-owned lowering; allowed as final default
#   tinygrad_scheduler_generated -> ordinary tinygrad graph lowering; allowed as final default
#   hand_authored_uop_template  -> Python UOp custom_kernel body written by humans; transitional default only
#   external_handwritten_kernel -> HIP/ASM/C++/precompiled binary or explicit instruction emitter; not final default
#   rollback_oracle            -> handwritten/specialized route retained only as rollback/reference
ROUTE_PROVENANCE = (
  "machine_authored_generated", "tinygrad_scheduler_generated", "hand_authored_uop_template",
  "external_handwritten_kernel", "rollback_oracle",
)
FINAL_DEFAULT_PROVENANCE = {"machine_authored_generated", "tinygrad_scheduler_generated"}
TRANSITIONAL_DEFAULT_PROVENANCE = {"hand_authored_uop_template"}
FORBIDDEN_DEFAULT_PROVENANCE = {"external_handwritten_kernel", "rollback_oracle"}

ROUTES = {
  # ---------------- decode weight GEMV: Q4_K ----------------
  "decode_q4k_g3_generated": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "promoted_default",
    "roles": ["ffn_gate_up", "ffn_down", "attn_qo", "attn_k"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [
      {"role": "ffn_gate_up", "K": 4096, "N": 12288}, {"role": "ffn_down", "K": 12288, "N": 4096},
      {"role": "attn_qo", "K": 4096, "N": 4096},
      {"role": "anyshape", "condition": "DECODE_Q4K_G3_ANYSHAPE=1 and (K//256)%4==0 and N%32==0"}],
    "env": {},  # DEFAULT-ON: decode_routes.py q4k_primitive_linear_call getenv("BUBBLEBEAM_FUTURESIGHT", 1). No flag needed.
    "rollback": {"BUBBLEBEAM_FUTURESIGHT": "0"},  # -> owned warp (decode_q4k_owned_warp)
    "baseline_route_id": "decode_q4k_owned_warp",  # the oracle/baseline the evaluator measures against (== rollback target)
    "strict_fallback": True,
    "expected_kernels": ["q4k_g3_lanemap_gemv_*"],
    "forbidden_kernels": ["q4k_gemv_warp_kernel (on the eligible roles)", "q4k_lane_partition_gemv_*", "fallback_graph"],
    "authority_gate": "retired 2026-07-03; promotion banked in docs/qk-gate-series-conclusions.md (was extra/audit/amd_isa/g3_weight_promotion_gate.py)",
    "promotion_artifacts": ["bench/amd-isa-backend-g3-weight-promotion/latest.json",
                            "bench/amd-isa-backend-g3-weight-promotion/summary.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py q4k_primitive_linear_call (QK_ROUTE_POLICY selects decode_q4k_g3_generated per tensor when present, else g3 fires by default for g3_bubblebeam_shape or DECODE_Q4K_G3_ANYSHAPE structural eligibility; strict mode fails loud on hidden fallback); writer extra/qk/gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel",
    "note": "generated wave32 UOp program lowered from the G2 Q4_K LaneMap (extra/qk/gemv_g2_lanemap.py). Speed-equivalent to owned warp (-0.13..+0.41% across ctx 512-4096), token-identical, route-clean. DECODE_Q4K_G3_ANYSHAPE extends it structurally to larger dense Q4_K shapes (including attn_k when policy installs it). This is the positive-control pure-search default decode kernel."},
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
    "authority_gate": "retired 2026-07-03; promotion banked in docs/qk-gate-series-conclusions.md (was extra/audit/amd_isa/g3_weight_promotion_gate.py)",
    "promotion_artifacts": ["docs/decode-q4k-gemv-warp-promotion-result-20260624.md"],
    "purity_status": "owned_reference",
    "provenance": "rollback_oracle",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/decode_routes.py q4k_primitive_linear_call (Q4K_GEMV_WARP_PROJ default 1, q/o + Q4K_GEMV_WARP default 1, gate/up+down); reached only when BUBBLEBEAM_FUTURESIGHT=0 short-circuits the G3 branch. Writer extra/qk/quant/q4_k_gemv_primitive.py q4k_gemv_warp_kernel",
    "note": "hand-written owned warp GEMV. The Q4K_GEMV_WARP* guards still default to 1, but the G3 branch intercepts first for the eligible shapes when BUBBLEBEAM_FUTURESIGHT is on (the default). So owned warp is the rollback/reference, not the live default."},
  # ---------------- decode weight GEMV: Q6_K ----------------
  "decode_q6k_coop_generated": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "promoted_default",
    "roles": ["ffn_down", "lm_head", "attn_v"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"role": "ffn_down", "K": 12288, "N": 4096}, {"role": "ffn_down_longk", "K": ">=8192", "N": "<100000"},
                     {"role": "lm_head", "N": ">=100000"}, {"role": "attn_v", "enabled_by": "Q6K_COVER_MORE=1"}],
    "env": {},  # DEFAULT-ON: decode_routes.py q6k_primitive_linear_call getenv("DECODE_Q6K_GENERATED", 1). BoltBeam QK_ROUTE_POLICY can select per tensor.
    "rollback": {"DECODE_Q6K_GENERATED": "0"},  # -> shipped hand kernels (decode_q6k_coop_shipped)
    "baseline_route_id": "decode_q6k_coop_shipped",
    "strict_fallback": True,
    "expected_kernels": ["q6k_gen_coop_*", "q6k_gen_partial_*"],
    "forbidden_kernels": ["q6k_coop_partial_* (on the default path)", "q6k_gemv_partial_* (on the default path)"],
    "authority_gate": "extra/qk/q6k_generated_coop_gate.py",
    "promotion_artifacts": ["bench/tg-p3-q6k-generated-coop/latest.json", "bench/tg-p3-q6k-generated-coop/summary.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py q6k_primitive_linear_call generated branch (getenv('DECODE_Q6K_GENERATED', 1) or QK_ROUTE_POLICY decode_q6k_coop_generated); writer extra/qk/q6k_route_spec.py emit_q6k_gemv_kernel (spec-driven lowering of Q6KGEMVRouteSpec)",
    "note": "spec-driven generated Q6_K decode GEMV: emit_q6k_gemv_kernel lowers a Q6KGEMVRouteSpec (data) to the coop/partial UOp kernel. Byte-identical to the shipped hand templates (extra/qk/q6k_generated_coop_gate.py TG_P3_PASS: all_identical, worst gen/shipped time 1.011). Provenance conversion of the Q6_K default; shipped kernels retained as rollback/oracle (DECODE_Q6K_GENERATED=0)."},
  "decode_q6k_coop_shipped": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "rollback_reference",
    "roles": ["ffn_down", "lm_head", "attn_v"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"role": "ffn_down", "K": 12288, "N": 4096}, {"role": "ffn_down_longk", "K": ">=8192", "N": "<100000"},
                     {"role": "lm_head", "N": ">=100000"}, {"role": "attn_v", "enabled_by": "Q6K_COVER_MORE=1"}],
    "env": {"DECODE_Q6K_GENERATED": "0"},  # SET this to force the shipped hand kernels (the rollback for the generated route)
    "rollback": {},  # this IS the rollback target
    "strict_fallback": True,
    "expected_kernels": ["q6k_coop_partial_*", "q6k_gemv_partial_*"],
    "authority_gate": "extra/qk/q6k_generated_coop_gate.py",
    "promotion_artifacts": ["bench/tg-p3-q6k-generated-coop/latest.json"],
    "purity_status": "owned_reference",
    "provenance": "rollback_oracle",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/decode_routes.py q6k_primitive_linear_call shipped branch (reached only when DECODE_Q6K_GENERATED=0); writer extra/qk/quant/q6_k_gemv_primitive.py q6k_coop_partial_kernel / q6k_gemv_partial_kernel",
    "note": "hand-authored Q6_K coop/partial UOp templates. Byte-identical to the generated route (decode_q6k_coop_generated), retained as the rollback/oracle one flag away (DECODE_Q6K_GENERATED=0)."},
  "decode_q6k_direct_refuted": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "refuted",
    "roles": ["lm_head"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"role": "lm_head", "N": ">=100000"}],
    "env": {"Q6K_DIRECT_ROUTE": "1"}, "rollback": {"Q6K_DIRECT_ROUTE": "0"},
    "baseline_route_id": "decode_q6k_coop_shipped",  # the oracle/baseline the evaluator measures against
    "strict_fallback": True,
    "expected_kernels": ["q6k_halfwarp_partition_151936_4096"],
    "authority_gate": "extra/qk/decode_runtime_overhead.py",
    "promotion_artifacts": ["bench/amd-isa-backend-q6k-direct-speed/latest.json",
                            "bench/amd-isa-backend-q6k-direct-speed/summary.md"],
    "purity_status": "refuted",
    "provenance": "hand_authored_uop_template",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/decode_routes.py q6k_primitive_linear_call Q6K_DIRECT_ROUTE branch (default-off); refuted vs decode_q6k_coop_shipped baseline.",
    "note": "half-warp direct Q6_K lm_head route: token-correct + route-bound, but W==D regressed -4.77..-6.06% (median -5.44%). Default-off. Do NOT re-chase as built (only reopen with a different topology than the half-warp partition)."},
  # ---------------- decode attention ----------------
  "decode_attention_owned_two_kernel": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "removed",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "ctx": ">=512"}],
    "env": {},
    "rollback": {},
    "strict_fallback": True,
    "expected_kernels": ["owned_flash_tile_gqa_whole", "owned_flash_combine"],
    "authority_gate": "extra/qk/decode_runtime_overhead.py",
    "promotion_artifacts": ["docs/decode-two-kernel-problem-audit-result-20260625.md",
                            "bench/amd-isa-backend-decode-attention-ceiling/latest.json"],
    "purity_status": "removed",
    "provenance": "external_handwritten_kernel",
    "replacement_scope": "decode_flash_live_split_g4_8b_kvboth",
    "selector": "env_guard",
    "route_attribution": "removed from tinygrad/llm/model.py; retired handwritten HIP attention implementation pruned.",
    "note": "Retired owned HIP split tile + combine. Replaced as the 8B long-context default by decode_flash_live_split_g4_8b_kvboth so the hot path is generated machine-search code."},
  "decode_flash_live_split_g4_8b_kvboth": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "promoted_default",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "G": 4, "ctx": ">=512"}],
    "env": {},  # DEFAULT-ON for the validated 8B G=4 shape.
    "rollback": {"DECODE_LIVE_SPLIT": "0"},  # unified structural branch -> generic generated tinygrad flash decode, not owned HIP
    "baseline_route_id": "decode_attention_owned_two_kernel",
    "strict_fallback": True,
    "expected_kernels": ["flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", "flash_fused_gmax_combine"],
    "forbidden_kernels": ["owned_flash_tile_gqa_whole", "owned_flash_combine", "fallback_graph"],
    "authority_gate": "extra/qk/prefilled_route_parity.py",
    "promotion_artifacts": ["bench/tg-p14-amd-recovery-and-pure-attention-landing/phase1_kvboth_result.json",
                            "bench/tg-p14-amd-recovery-and-pure-attention-landing/phase2_final_result.json",
                            "bench/tg-p14-amd-recovery-and-pure-attention-landing/summary.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py flash_decode_attention_route 8B generated branch: B=1,Hq=32,Hkv=8,Hd=128 -> flash_decode_live_split_block_tile(..., staging='KV_BOTH', fused_combine=True). Writer extra/qk/live_split_geometry.py + extra/qk/flash_decode.py generated UOp kernels.",
    "note": "Promoted 8B long-context decode attention replacement. TG-P14 practical roofline closeout: worst-of-3 speed ctx512 98.5% / ctx4096 98.3% of owned, 48/48 deterministic prefilled token parity, route-bound, no hidden fallback. Default choice intentionally prefers generated machine-search code over the retired handwritten HIP route."},
  "decode_flash_block_tile_g5_konly": {
    "workload": "decode", "profile_id": PROFILE_DECODE_LARGE, "status": "promoted_default",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128, "ctx": ">=512"}],
    "env": {},  # DEFAULT-ON for the validated G=5 shape; BoltBeam QK_ROUTE_POLICY can select it by shape.
    "rollback": {"DECODE_LIVE_SPLIT": "0"},  # unified structural branch -> generic generated flash on rollback
    "baseline_route_id": "decode_attention_owned_two_kernel",
    "strict_fallback": True,
    "expected_kernels": ["flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel"],
    "forbidden_kernels": ["owned_flash_tile_gqa_whole", "fallback_graph"],
    "authority_gate": "extra/qk/decode_runtime_overhead.py",
    "promotion_artifacts": ["bench/gp-track/gp4_latest.json", "bench/gp-track/gp3_microgate.json",
                            "docs/gp5-final-report.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py flash_decode_attention_route UNIFIED live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0; covers 14B Hq=40/G=5). QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_LIVE_SPLIT default 1. Writer extra/qk/live_split_geometry.py flash_decode_live_split_block_tile(..., staging='KV_BOTH', fused_combine=True) -> generated UOp flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel.",
    "note": "Promoted 2026-07-03: 14B (Hq=40/Hkv=8/Hd=128) decode now shares the MODULAR live-split route with 8B/32B (no per-model Hq hardcode). Live per-split length ceildiv(Tc,S_occ) is SEQLEN-BOUND: 14B decode is FLAT across max_context (69.24 tok/s @MAXC=1024 vs 69.04 @MAXC=8192, live ctx ~550), vs the retired fixed-L g5 route which read the full max_context buffer every token and collapsed as MAXC rose. W==D token-identical to the generic generated flash reference (DECODE_LIVE_SPLIT=0) at 8B/14B/32B. Staging is KV_BOTH (self-contained LDS); K_ONLY is unsupported under live-split geometry (wrong global-V addressing -> garbage). S_occ fixed at 48 (=4*CU/Hkv occupancy default). Rollback DECODE_LIVE_SPLIT=0."},
  "decode_flash_block_tile_g5_8b_refuted": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "removed",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "G": 4, "ctx": ">=512"}],
    "env": {},
    "rollback": {},
    "baseline_route_id": "decode_attention_owned_two_kernel",
    "strict_fallback": True,
    "expected_kernels": ["flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", "flash_state_gmax_32_128", "flash_state_combine_32_128"],
    "authority_gate": "historical TG-P5/TG-P8 artifacts; route no longer selectable",
    "promotion_artifacts": ["bench/tg-p5-attention-generated-default/latest.json",
                            "bench/tg-p5-attention-generated-default/summary.md"],
    "purity_status": "removed",
    "provenance": "machine_authored_generated",
    "selector": "retired",
    "route_attribution": "removed from tinygrad/llm/model.py; superseded by decode_flash_live_split_g4_8b_kvboth.",
    "note": "Historical TG-P5/TG-P8 route. The generated G5 block-tile flash decode generalized correctly to the 8B geometry but was slower (87.6%/95.6% @ctx512/4096). It is no longer selectable because the validated live-split KV_BOTH route superseded it as the generated 8B default."},
  "decode_attention_generic_flash_generated": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "superseded_rollback",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "ctx": ">=512"}],
    "env": {"DECODE_LIVE_SPLIT": "0"},
    "rollback": {},  # -> promoted generated default route
    "strict_fallback": True,
    "authority_gate": "tinygrad generated flash attention fallback",
    "promotion_artifacts": ["bench/tg-p14-amd-recovery-and-pure-attention-landing/summary.md"],
    "purity_status": "research",
    "provenance": "tinygrad_scheduler_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/decode_routes.py flash_decode_attention_route generic generated flash_decode_attention fallback reached when DECODE_LIVE_SPLIT=0.",
    "note": "Generic generated tinygrad flash-decode fallback. It is retained only as a generated rollback/reference for the promoted live-split KV_BOTH route; the old native/HIP research route was pruned."},
  # ---------------- prefill GEMM ----------------
  "prefill_pipe_role_selective_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "graph-GEMM prefill ubatch=512; role_policy in the spec: pipe for attn_qo/attn_kv/ffn_down, lds for ffn_gate_up out_f==12288"}],
    "env": {},
    "rollback": {},
    "strict_fallback": True,
    "expected_kernels": ["prefill_gen_sched_gemm_*"],
    "forbidden_kernels": ["prefill_graph_gemm_* (on the default path)"],
    "authority_gate": "extra/qk/prefill_generated_schedule_gate.py",
    "promotion_artifacts": ["bench/tg-p4-prefill-generated-schedule/latest.json",
                            "bench/tg-p4-prefill-generated-schedule/summary.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm -> describe_prefill_schedule + emit_prefill_gemm_from_spec; writer extra/qk/prefill_schedule_spec.py (PrefillGEMMScheduleSpec lowered through the RDNA3 WMMA schedule generator ref.build_gemm_pipe / build_gemm_lds2).",
    "note": "spec-driven generated prefill GEMM schedule: PrefillGEMMScheduleSpec (data) captures the resolved tile/wave/pipeline/role-policy; emit_prefill_gemm_from_spec lowers it through the parameterized RDNA3 WMMA schedule generator. The RDNA3 WMMA instruction set is the target grammar; the schedule is machine-authored from the spec. The legacy fixed emit and PREFILL_GENERATED_SCHEDULE rollback were removed from runtime."},
  "prefill_q4k_direct_tile4x4_default": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "direct-packed Q4_K prefill, memory-safe 14B/32B route"}],
    "env": {},
    "rollback": {"PREFILL_Q4K_DIRECT_SCHEDULE": "legacy"},
    "baseline_route_id": "prefill_q4k_direct_packed_load_direct_out",
    "strict_fallback": True,
    "expected_kernels": ["prefill_q4k_direct_packed_load_direct_out_gemm_*"],
    "authority_gate": "extra/qk/prefill_boltbeam_trace.py",
    "promotion_artifacts": ["docs/prefill-packed-generated-tile-scope-20260704.md"],
    "purity_status": "search_selected_specialized_route",
    "provenance": "hand_authored_uop_template",
    "replacement_scope": "14B/32B Q4_K prefill needs a generated quantized MMQ substrate that fuses dequant and matmul; this direct-packed UOp template is only the memory-safe interim route.",
    "selector": "env_default",
    "route_attribution": "tinygrad/llm/prefill_routes.py _direct_packed_opts default Q4_K schedule: LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4; rollback PREFILL_Q4K_DIRECT_SCHEDULE=legacy.",
    "note": "Promoted Q4_K direct-packed schedule for 14B memory-safe prefill. Fable audit reframed the gap as dequant amortization across token/register tiles rather than a new kernel family. Clean pp512 improved 135.7 -> 173.6 tok/s on Qwen3-14B-Q4_K_M. Correctness matched the old lossless direct path for the checked ffn_gate row. This is a schedule default on the existing direct-output UOp route, not the final llama-class MMQ substrate."},
  "prefill_q4k_generated_tile_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": 17408, "K": 5120, "role": "ffn_gate_up", "note": "first 14B target from BoltBeam practical roofline"},
                     {"M": 512, "N": 5120, "K": 5120, "role": "attn_qo", "note": "second Q4 target after ffn_gate_up moves"},
                     {"M": 512, "N": 5120, "K": 17408, "role": "ffn_down", "note": "third Q4 target after topology proves"}],
    "env": {"PREFILL_QK_GENERATED_TILE": "1"},
    "rollback": {"PREFILL_QK_GENERATED_TILE": "0"},
    "baseline_route_id": "prefill_q4k_direct_packed_load_direct_out",
    "strict_fallback": True,
    "expected_kernels": ["prefill_q4_k_generated_tile_*"],
    "forbidden_kernels": ["prefill_q4k_direct_packed_load_direct_out_gemm_* (on selected roles)"],
    "authority_gate": "extra/qk/prefill_packed_roofline_scope.py + future generated-tile microgate",
    "promotion_artifacts": ["docs/prefill-packed-generated-tile-scope-20260704.md"],
    "purity_status": "research",
    "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "future tinygrad/llm/prefill_routes.py branch guarded by PREFILL_QK_GENERATED_TILE=1; intended writer is a PackedPrefillTileSpec lowered to generated UOps, not a source-string or external handwritten kernel.",
    "note": "14B/32B memory-safe parity research route. The first simple generated-UOp cooperative-lane probes are refuted: 256-thread lane_partials hit 0.99 GB/s, and one-wave direct_warp tops out at 1.29 GB/s on ffn_gate_up. After the Fable audit, the immediate shipped fix became prefill_q4k_direct_tile4x4_default. Keep this generated-tile route default-off; reopen only with a correct grouped/staged reduction or dequant-to-LDS/WMMA plan."},
  "prefill_q4k_int8_wmma_generated_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "Q4_K/Q8_1 generated MMQ substrate; int dot is Tensor.matmul(..., dtype=int) so iu8 WMMA must come from codegen TC matching"}],
    "env": {"PREFILL_Q4K_Q8": "wmma"},
    "rollback": {"PREFILL_Q4K_Q8": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default",
    "strict_fallback": True,
    "expected_kernels": ["prefill_q4k_q8_1_wmma_generated_gemm_* (route identity); generated matmul kernels must contain wmma_i32_16x16x16_iu8 on AMD"],
    "forbidden_kernels": ["new handwritten HIP/CUDA/source-string kernel", "prefill_q4k_direct_packed_load_direct_out_gemm_* (on selected roles)"],
    "authority_gate": "extra/qk/prefill_mmq_parity_gate.py + extra/qk/int8_wmma_codegen_gate.py",
    "promotion_artifacts": ["docs/14b-q4k-int8-wmma-substrate-scope-20260705.md"],
    "purity_status": "research",
    "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=wmma -> extra/qk/prefill_int8_wmma_spec.py Q4KInt8WMMAPrefillSpec + emit_q4k_int8_wmma_prefill_tensor; core RAW dot is ordinary Tensor.matmul(..., dtype=dtypes.int), relying on tinygrad/codegen/opt/tc.py and tinygrad/renderer/cstyle.py for iu8 WMMA.",
    "note": "Default-off first generated substrate for 14B Q4_K int8 MMQ. It reuses q8_1_quantize and Q4_K metadata helpers, validates algebra in the existing MMQ parity gate, and deliberately does not introduce a new handwritten kernel. Promotion requires an AMD gate proving wmma_i32_16x16x16_iu8 in generated code/ISA and 14B authority beating the direct-packed baseline."},
  "prefill_q4k_int8_wmma_tiled_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "bounded Q4_K/Q8_1 tiled generated WMMA route; RAW must be tile-local, not [groups,M,N]"}],
    "env": {"PREFILL_Q4K_Q8": "wmma_tiled"},
    "rollback": {"PREFILL_Q4K_Q8": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default",
    "strict_fallback": True,
    "expected_kernels": ["prefill_q4k_q8_1_wmma_tiled_generated_gemm_* or emitted Tensor kernels with wmma_i32_16x16x16_iu8"],
    "forbidden_kernels": ["new handwritten HIP/CUDA/source-string kernel", "inline asm", "route-local __builtin_amdgcn_wmma", "prefill_q4k_direct_packed_load_direct_out_gemm_* (on selected roles)"],
    "authority_gate": "extra/qk/q4k_wmma_tiled_lowering_feasibility.py + extra/qk/q4k_wmma_tiled_microgate.py + extra/qk/q4k_wmma_tiled_role_shape_gate.py",
    "promotion_artifacts": ["docs/q4k-wmma-fused-tiled-prefill-execution-scope-20260705.md"],
    "purity_status": "research",
    "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=wmma_tiled -> extra/qk/prefill_int8_wmma_spec.py Q4KInt8WMMATiledPrefillSpec + one-tile emit_q4k_int8_wmma_tiled_prefill_tensor. Full role shapes raise explicitly so this flag cannot silently fall through to the scalar Q4_K/Q8_1 GEMM route.",
    "note": "One-tile tiled WMMA substrate is correct and codegen-valid: lowering feasibility and Q4_K/Q8_1 microgate pass on AMD with wmma_i32_16x16x16_iu8. Full 14B role shapes are classified blocked.full_route_lowering_missing until a direct tiled scheduler/codegen lowering maps role shapes to bounded tiles. Promotion requires canonical 14B smoke beating the current direct-packed default."},
  "prefill_q4k_reduce_out_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "correct_not_fast",
    "roles": ["ffn_gate_up", "attn_qo", "attn_kv"], "excluded_roles": ["ffn_down"],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "K": 5120, "note": "tested with GROUP:0:10 on K=5120 roles"}],
    "env": {"PREFILL_Q4K_REDUCE_OUT": "1"},
    "rollback": {"PREFILL_Q4K_REDUCE_OUT": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default",
    "strict_fallback": True,
    "expected_kernels": ["prefill_q4k_direct_packed_load_reduce_out_gemm_*"],
    "authority_gate": "extra/qk/prefill_boltbeam_trace.py",
    "promotion_artifacts": ["docs/prefill-packed-generated-tile-scope-20260704.md"],
    "purity_status": "research",
    "provenance": "hand_authored_uop_template",
    "selector": "env_guard",
    "route_attribution": "extra/qk/quant/q4_k_gemv_primitive.py q4k_gemm_packed_load_reduce_out_kernel, selected by tinygrad/llm/prefill_routes.py when PREFILL_Q4K_REDUCE_OUT=1.",
    "note": "Default-off primitive correctness fix. It replaces the manual direct-output accumulator recurrence with a real Ops.REDUCE, making GROUP schedules numerically valid: GROUP:0:10 on real 14B ffn_gate rel_rmse ~=1.6e-6 vs the lossless direct path. It is not promoted because clean pp512 is 169.7 tok/s vs 173.6 for the current Q4 tile4x4 manual direct-output default. Use this as the correctness foundation for future grouped/staged combine work."},
  "prefill_q4k_mmq_direct_out_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "correct_not_fast",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "Q4_K/Q8_1 generated-UOp dot4 MMQ with in-kernel 8-lane reduce and direct output"}],
    "env": {"PREFILL_Q4K_Q8": "mmq_direct"},
    "rollback": {"PREFILL_Q4K_Q8": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default",
    "strict_fallback": True,
    "expected_kernels": ["prefill_q4k_q8_1_mmq_direct_out_gemm_*"],
    "forbidden_kernels": ["prefill_q4k_q8_1_mmq_direct_packed_gemm_* partial tensor path"],
    "authority_gate": "extra/qk/prefill_mmq_parity_gate.py + canonical 14B smoke",
    "promotion_artifacts": ["docs/14b-q4k-int8-wmma-substrate-scope-20260705.md"],
    "purity_status": "research",
    "provenance": "hand_authored_uop_template",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=mmq_direct -> extra/qk/quant/q4_k_gemv_primitive.py q4k_q8_1_sdot4_coop_direct_out_kernel. Reuses the existing Q4_K/Q8_1 dot4 algebra and the existing warp_reduce_sum direct-output pattern.",
    "note": "Correct and bounded 14B research route. It eliminates the [rows,tokens,8] partial tensor and avoids the WMMA Tensor graph-explosion guard, but canonical 14B pp512 smoke measured only 85 tok/s, slower than the current direct-packed default. Keep as topology evidence for in-kernel lane combine, not as the final route."},
  "prefill_pipe_global_rollback": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "superseded_rollback",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*"}],
    "env": {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_PIPE_ROLE_SELECTIVE": "0"},
    "rollback": {"PREFILL_GEMM_PIPELINE": "0"},  # -> old lds2 default
    "strict_fallback": True,
    "authority_gate": "extra/qk/prefill_whole_synced.py",
    "promotion_artifacts": ["bench/qk-prefill-pipe-promotion/latest.json",
                            "bench/qk-prefill-pipe-promotion/summary.md"],
    "purity_status": "search_selected_specialized_route",
    "provenance": "rollback_oracle",
    "selector": "env_guard",
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py:55 (pipe on for all roles when PREFILL_PIPE_ROLE_SELECTIVE=0).",
    "note": "global pipe (all roles): was TIER_A vs old lds2 default (+8.5..19.2%), superseded by role-selective (which excludes the saturated gate/up where pipe regressed ~17%). Kept as the A/B rollback comparator and the rollback target of role-selective."},
}

# Closed / refuted axes -- do not re-search without a NEW premise (PMS-R3 do_not_search carries these forward).
REFUTED = [
  {"axis": "q6k_direct_half_warp_route", "disposition": "refuted: W==D regression -4.77..-6.06% (median -5.44%)",
   "citation": "bench/amd-isa-backend-q6k-direct-speed/latest.json", "route_id": "decode_q6k_direct_refuted"},
  {"axis": "q4k_offline_layout_reshuffle", "disposition": "deprioritized: G3 matches owned, no layout gap to recover",
   "citation": "bench/amd-isa-backend-g3-weight-promotion/search_space_update.json"},
  {"axis": "prefill_q4k_simple_uop_cooperative_lane_tile", "domain": "prefill",
   "disposition": "refuted on 14B ffn_gate_up: lane_partials 0.99 GB/s; direct_warp sweep best 1.29 GB/s vs current direct-packed floor ~2.11 GB/s",
   "citation": "docs/prefill-packed-generated-tile-scope-20260704.md", "route_id": "prefill_q4k_generated_tile_research"},
  {"axis": "prefill_q4k_direct_group_reduce_current_uop", "domain": "prefill",
   "disposition": "refuted for the manual direct-output recurrence: GROUP:0:10 looked fast but is numerically wrong on real 14B ffn_gate (rel_rmse ~1.26); use PREFILL_Q4K_REDUCE_OUT=1 for correct-but-not-fast grouped semantics",
   "citation": "docs/prefill-packed-generated-tile-scope-20260704.md", "route_id": "prefill_q4k_direct_tile4x4_default"},
  {"axis": "attention_combine_fused_lifecycle", "domain": "attention", "disposition": "exhausted/low-leverage (combine overlaps in-graph; fused is codegen-walled)",
   "citation": "docs/decode-two-kernel-problem-audit-result-20260625.md"},
  {"axis": "g5_block_tile_8b_as_default", "domain": "attention", "disposition": "correct_not_fast: token-identical + route-bound but 87.6% of owned @ctx512 / 95.6% @ctx4096 (TG-P5)",
   "citation": "bench/tg-p5-attention-generated-default/latest.json", "route_id": "decode_flash_block_tile_g5_8b_refuted"},
  {"axis": "g5_block_tile_8b_L_geometry", "domain": "attention", "disposition": "refuted: L=128 is the geometry optimum (87.7%/95.9%); larger L monotonically worse (69%/75.6% at L=576, occupancy-starved) -- the generated route needs ~36 splits for parallelism so it over-launches at low ctx (TG-P8.2)",
   "citation": "bench/tg-p8-generated-8b-attention-parity/geometry_search.json"},
  {"axis": "g5_block_tile_8b_combine_lifecycle_cap", "domain": "attention", "disposition": "blocking: the generated 3-kernel gmax+combine lifecycle is 556us/fwd (83% of the ctx4096 attention delta) vs owned's fused 224us -> BINDING cap at ctx4096 (95.9%); a perfect tile saves only 112us. Combine COLLAPSE is refuted (guardrail #3); reopen only with a NEW non-collapse coordination primitive (TG-P8.1/P8.2)",
   "citation": "bench/tg-p8-generated-8b-attention-parity/latest.json"},
  {"axis": "live_split_geometry_8b_tile", "domain": "attention", "disposition": "SOLVED/PROMOTED: live-context split geometry (fixed S, per=ceildiv(Tc,S)) is expressible in generated UOp; the live-split route plus KV_BOTH staging and fused combine is now the 8B generated default. extra/qk/live_split_geometry.py",
   "citation": "bench/tg-p9-pure-attention-primitive-route/live_split_tile_microgate.json"},
  {"axis": "split_preserving_lse_combine_8b", "domain": "attention", "disposition": "EMITTER_BLOCKED (TG-P9.4): a split-preserving generated combine (de-dup the per-d fexp / fuse gmax without collapsing Hq*S or Hq*Hd) mis-vectorizes the reduction-accumulator REG to a non-assignable make_float4(...) store; REG_STORE_DEVEC=1 compiles but NaNs. The ctx4096 556us combine cap cannot be removed in current AMD codegen. Reopen: a codegen fix keeping the reduction-accumulator REG scalar for a multi-reduce/weight-sharing combine.",
   "citation": "bench/tg-p9-pure-attention-primitive-route/combine_microgate.json"},
]

# ---- DEFERRED capability frontier: blocked-but-OPEN, NOT refuted on merits. The pure-search north-star (replace the
#      two hand-written decode kernels with a fully searched native route) is gated on renderer lowering of these
#      primitives; reopen each when the capability lands. Distinct from REFUTED (lost on its merits) and from shipped. ----
DEFERRED_CAPABILITIES = [
  {"capability": "v_dot2_lowering", "status": "deferred", "domain": "codegen",
   "blocks": "native packed-fp16 dot in a searched decode-attention / GEMV kernel",
   "reopen_when": "the renderer lowers v_dot2 (packed fp16 dot) so the search space can emit it natively"},
  {"capability": "cross_lane_mixed_reduce", "status": "deferred", "domain": "codegen",
   "blocks": "native cross-lane reduction lowering for a searched tile (LDS path already native; ds_bpermute tree TODO)",
   "reopen_when": "the renderer lowers the cross-lane reduction so a searched kernel can own the reduction topology"},
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

def route_provenance(route_id: str) -> str:
  prov = str(route(route_id).get("provenance", ""))
  if prov not in ROUTE_PROVENANCE:
    raise ValueError(f"route {route_id!r} has invalid provenance {prov!r}; expected one of {ROUTE_PROVENANCE}")
  return prov

def default_purity_report() -> dict:
  """Strict final-default purity report. This is intentionally allowed to FAIL today.

  A generated/search route can be fast and correct but still fail final-default purity if its implementation is an
  external handwritten kernel. A hand-authored UOp route is reported as transitional debt: allowed to keep shipping
  only while its replacement scope is explicit.
  """
  defaults = default_routes()
  rows, forbidden, transitional = [], [], []
  for rid in defaults:
    r, prov = route(rid), route_provenance(rid)
    row = {"route_id": rid, "status": r["status"], "provenance": prov,
           "replacement_scope": r.get("replacement_scope", ""), "final_default_allowed": prov in FINAL_DEFAULT_PROVENANCE}
    rows.append(row)
    if prov in FORBIDDEN_DEFAULT_PROVENANCE: forbidden.append(rid)
    if prov in TRANSITIONAL_DEFAULT_PROVENANCE: transitional.append(rid)
  verdict = "TINYGRAD_DEFAULT_PURITY_PASS" if not forbidden and not transitional else "TINYGRAD_DEFAULT_PURITY_FAIL"
  return {"verdict": verdict, "default_routes": defaults, "rows": rows,
          "forbidden_default_routes": forbidden, "transitional_default_routes": transitional,
          "final_default_allowed_provenance": sorted(FINAL_DEFAULT_PROVENANCE)}

def validate_manifest() -> list[str]:
  errors: list[str] = []
  for rid, r in ROUTES.items():
    prov = r.get("provenance")
    if prov not in ROUTE_PROVENANCE:
      errors.append(f"{rid}: invalid or missing provenance {prov!r}")
    if r["status"] in ("promoted_default", "default_shipped"):
      if prov == "rollback_oracle":
        errors.append(f"{rid}: default route cannot be provenance=rollback_oracle")
      if prov in ("hand_authored_uop_template", "external_handwritten_kernel") and not r.get("replacement_scope"):
        errors.append(f"{rid}: non-pure default provenance={prov} requires replacement_scope")
  return errors

def to_manifest_dict() -> dict:
  return {"_schema": "default route manifest (PMS-R1)", "generated_by": "extra/qk/route_manifest.py",
          "profiles": {"decode": PROFILE_DECODE, "prefill": PROFILE_PREFILL},
          "provenance_vocabulary": list(ROUTE_PROVENANCE),
          "routes": ROUTES, "refuted_axes": REFUTED,
          "default_routes": default_routes(),
          "promoted_defaults": routes_by_status("promoted_default"),
          "owned_defaults": routes_by_status("default_shipped"),
          "default_purity": default_purity_report()}

def dump(out_path: str | None = None) -> str:
  """Write the canonical manifest json (bench/qk-search-spaces/default_route_manifest.json by default)."""
  root = pathlib.Path(__file__).resolve().parents[2]
  p = pathlib.Path(out_path) if out_path else (root / "bench/qk-search-spaces/default_route_manifest.json")
  p.parent.mkdir(parents=True, exist_ok=True)
  json.dump(to_manifest_dict(), open(p, "w"), indent=2)
  return str(p)

# ---- refuted axes: REFUTED is the SINGLE SOURCE; do_not_search (search_profiles.json) and the quant
#      known_refuted_route_families must agree with it (enforced by qk_search_space_manifest_check). ----
def disposition_class(disp: str) -> str:
  """Normalize a disposition to its class token (refuted / deprioritized / exhausted / correct_not_fast / small / ...)."""
  return (disp or "").split(":")[0].split("/")[0].split()[0].lower()

def refuted_index() -> dict[str, str]:
  """{key -> disposition_class} for every refuted axis, keyed by route_id when present else axis."""
  return {(r.get("route_id") or r["axis"]): disposition_class(r["disposition"]) for r in REFUTED}

def dump_refuted(out_path: str | None = None) -> str:
  """Write the canonical refuted-axes json FROM REFUTED (the single source for do_not_search / quant known_refuted)."""
  root = pathlib.Path(__file__).resolve().parents[2]
  p = pathlib.Path(out_path) if out_path else (root / "bench/qk-search-spaces/refuted_axes.json")
  p.parent.mkdir(parents=True, exist_ok=True)
  json.dump({"_schema": "canonical refuted axes (generated FROM qk_route_manifest.REFUTED)",
             "generated_by": "extra/qk/route_manifest.py:dump_refuted",
             "agreement_key": "route_id when present else axis; disposition compared by class token",
             "refuted_axes": REFUTED}, open(p, "w"), indent=2)
  return str(p)

if __name__ == "__main__":
  if (errs := validate_manifest()):
    raise SystemExit("manifest validation failed:\n- " + "\n- ".join(errs))
  path = dump()
  rpath = dump_refuted()
  print(f"wrote default route manifest to {path}")
  print(f"wrote canonical refuted axes to {rpath}")
  print("default routes:", default_routes())
  print("promoted (generated/search-selected) defaults:", routes_by_status("promoted_default"))
  print("default purity:", default_purity_report()["verdict"])
  print(f"{len(ROUTES)} routes, {len(REFUTED)} refuted axes")
