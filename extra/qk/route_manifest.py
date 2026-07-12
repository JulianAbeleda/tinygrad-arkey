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
#   compiler_primitive_spec_owned -> compiler/search spec owns route lifecycle; reusable backend atom may emit ASM
#   hand_authored_uop_template  -> Python UOp custom_kernel body written by humans; transitional default only
#   external_handwritten_kernel -> HIP/ASM/C++/precompiled binary or explicit instruction emitter; not final default
#   rollback_oracle            -> handwritten/specialized route retained only as rollback/reference
ROUTE_PROVENANCE = (
  "machine_authored_generated", "tinygrad_scheduler_generated", "hand_authored_uop_template",
  "compiler_primitive_spec_owned", "external_handwritten_kernel", "rollback_oracle",
)
FINAL_DEFAULT_PROVENANCE = {"machine_authored_generated", "tinygrad_scheduler_generated"}
TRANSITIONAL_DEFAULT_PROVENANCE = {"hand_authored_uop_template"}
FORBIDDEN_DEFAULT_PROVENANCE = {"external_handwritten_kernel", "rollback_oracle"}

# purity_status is a DERIVED human-facing label, not an independent axis. It is a pure function of (status, provenance)
# so it can never silently drift from them (F4: it used to be a hand-maintained third vocabulary that disagreed with
# the route's real status/provenance). derive_purity_status is the single source; validate_manifest() enforces that
# every stored purity_status equals the derived value, and the census sources its purity_status from here.
PURITY_STATUS_VOCAB = ("search_generated_promoted", "refuted", "research")

def derive_purity_status(status: str, provenance: str) -> str:
  if status in ("promoted_default", "default_shipped") and provenance in FINAL_DEFAULT_PROVENANCE:
    return "search_generated_promoted"
  if status == "refuted":
    return "refuted"
  return "research"

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
    "rollback": {},  # no backups: the handwritten owned-warp rollback was deleted 2026-07-06. BUBBLEBEAM_FUTURESIGHT=0 now falls to the ordinary tinygrad graph, not a hand kernel.
    "baseline_route_id": "ordinary_tinygrad_graph",  # bubblebeam-off baseline is the ordinary dequant+matmul graph
    "strict_fallback": True,
    "expected_kernels": ["q4k_g3_lanemap_gemv_*"],
    "forbidden_kernels": ["q4k_gemv_warp_kernel (on the eligible roles)", "q4k_lane_partition_gemv_*", "fallback_graph"],
    "authority_gate": "retired 2026-07-03; promotion banked in docs/prefill-lessons-ledger.md (was extra/audit/amd_isa/g3_weight_promotion_gate.py)",
    "promotion_artifacts": ["bench/amd-isa-backend-g3-weight-promotion/latest.json",
                            "bench/amd-isa-backend-g3-weight-promotion/summary.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py q4k_primitive_linear_call (QK_ROUTE_POLICY selects decode_q4k_g3_generated per tensor when present, else g3 fires by default for g3_bubblebeam_shape or DECODE_Q4K_G3_ANYSHAPE structural eligibility; strict mode fails loud on hidden fallback); writer extra/qk/gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel",
    "note": "generated wave32 UOp program lowered from the G2 Q4_K LaneMap (extra/qk/gemv_g2_lanemap.py). Speed-equivalent to owned warp (-0.13..+0.41% across ctx 512-4096), token-identical, route-clean. DECODE_Q4K_G3_ANYSHAPE extends it structurally to larger dense Q4_K shapes (including attn_k when policy installs it). This is the positive-control pure-search default decode kernel."},
  # decode_q4k_owned_warp REMOVED 2026-07-06 (no backups): the handwritten owned-warp/coop/direct/vdot rollback
  # path in decode_routes.py was deleted. BUBBLEBEAM_FUTURESIGHT=0 now falls to the ordinary tinygrad graph.
  # ---------------- decode weight GEMV: Q6_K ----------------
  "decode_q6k_coop_generated": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "promoted_default",
    "roles": ["ffn_down", "lm_head", "attn_v"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"role": "ffn_down", "K": 12288, "N": 4096}, {"role": "ffn_down_longk", "K": ">=8192", "N": "<100000"},
                     {"role": "lm_head", "N": ">=100000"}, {"role": "attn_v", "enabled_by": "Q6K_COVER_MORE=1"}],
    "env": {},  # DEFAULT-ON and unconditional: decode_routes.py q6k_primitive_linear_call. BoltBeam QK_ROUTE_POLICY can select per tensor.
    "rollback": {},  # no backups: the DECODE_Q6K_GENERATED=0 shipped-hand-kernel rollback was deleted. Generated Q6_K decode is the only kernel route.
    "baseline_route_id": "ordinary_tinygrad_graph",
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
  # Q6_K shipped/refuted hand-kernel rows REMOVED 2026-07-06 (no backups): their kernels
  # (q6k_coop_partial_kernel / q6k_gemv_partial_kernel / q6k_halfwarp_partition) were deleted in prior cuts, and the
  # DECODE_Q6K_GENERATED=0 / Q6K_DIRECT_ROUTE env rollbacks are gone. Generated Q6_K decode is the only kernel route.
  # ---------------- decode attention ----------------
  # Retired handwritten HIP owned split row REMOVED 2026-07-06 (no backups): tile + combine
  # tile + combine; its route_attribution target no longer exists in the code (model.py branch removed).
  "decode_flash_live_split_g4_8b_kvboth": {
    "workload": "decode", "profile_id": PROFILE_DECODE, "status": "promoted_default",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 32, "Hkv": 8, "Hd": 128, "G": 4, "ctx": ">=512"}],
    "env": {},  # DEFAULT-ON for the validated 8B G=4 shape.
    "rollback": {"DECODE_LIVE_SPLIT": "0"},  # exits the live-split default; no manifest fallback route row remains
    "baseline_route_id": "retired_owned_attention_ceiling",
    "strict_fallback": True,
    "expected_kernels": ["flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", "flash_fused_gmax_combine"],
    "forbidden_kernels": ["owned_flash_tile_gqa_whole", "owned_flash_combine", "fallback_graph"],
    "authority_gate": "extra/qk/prefilled_route_parity.py",
    "promotion_artifacts": ["bench/tg-p14-amd-recovery-and-pure-attention-landing/phase1_kvboth_result.json",
                            "bench/tg-p14-amd-recovery-and-pure-attention-landing/phase2_final_result.json",
                            "bench/tg-p14-amd-recovery-and-pure-attention-landing/summary.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "replacement_scope": "",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py attention live-split branch: B=1,Hq=32,Hkv=8,Hd=128 -> extra/qk/live_split_geometry.py flash_decode_live_split_block_tile -> extra/qk/flash_decode_attention_spec.py FlashDecodeAttentionSpec (FlashDecodeTileSpec + LiveSplitGeometrySpec + FlashCombineSpec).",
    "note": "Promoted 8B long-context decode attention replacement. TG-P14 practical roofline closeout: worst-of-3 speed ctx512 98.5% / ctx4096 98.3% of owned, 48/48 deterministic prefilled token parity, route-bound, no hidden fallback. Provenance conversion 2026-07-06: the default binding is now descriptor-owned through FlashDecodeAttentionSpec; no handwritten HIP attention route remains."},
  "decode_flash_block_tile_g5_konly": {
    "workload": "decode", "profile_id": PROFILE_DECODE_LARGE, "status": "promoted_default",
    "roles": ["attention_tile", "attention_combine"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [{"B": 1, "Hq": 40, "Hkv": 8, "Hd": 128, "ctx": ">=512"}],
    "env": {},  # DEFAULT-ON for the validated G=5 shape; BoltBeam QK_ROUTE_POLICY can select it by shape.
    "rollback": {"DECODE_LIVE_SPLIT": "0"},  # exits the live-split default; no manifest fallback route row remains
    "baseline_route_id": "retired_owned_attention_ceiling",
    "strict_fallback": True,
    "expected_kernels": ["flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128", "flash_fused_gmax_combine"],
    "forbidden_kernels": ["owned_flash_tile_gqa_whole", "fallback_graph"],
    "authority_gate": "extra/qk/decode_runtime_overhead.py",
    "promotion_artifacts": ["bench/gp-track/gp4_latest.json", "bench/gp-track/gp3_microgate.json",
                            "docs/gp5-final-report.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "replacement_scope": "",
    "selector": "BoltBeam_route_policy_or_env_default",
    "route_attribution": "tinygrad/llm/decode_routes.py flash_decode_attention_route UNIFIED live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0; covers 14B Hq=40/G=5). QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_LIVE_SPLIT default 1. Binding flows through extra/qk/live_split_geometry.py -> FlashDecodeAttentionSpec.",
    "note": "Promoted 2026-07-03: 14B (Hq=40/Hkv=8/Hd=128) decode now shares the modular live-split route with 8B/32B. Live per-split length ceildiv(Tc,S_occ) is seqlen-bound: 14B decode is flat across max_context (69.24 tok/s @MAXC=1024 vs 69.04 @MAXC=8192, live ctx ~550). W==D token-identical to the generic generated flash reference at 8B/14B/32B. Provenance conversion 2026-07-06: the default binding is descriptor-owned through FlashDecodeAttentionSpec."},
  # decode_flash_block_tile_g5_8b_refuted row REMOVED 2026-07-06 (no backups): historical TG-P5/TG-P8 route; its
  # kernels (flash_state_gmax/flash_state_combine) and route_attribution branch no longer exist.
  # Generic flash fallback row REMOVED 2026-07-06 (no backups): the DECODE_LIVE_SPLIT=0 fallback implementation was
  # deleted; no rollback target remains.
  # ---------------- prefill GEMM ----------------
  "prefill_v2_scheduler_matmul_default": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "ordinary PREFILL_V2 fp16 matmul under warmstart TC opts"}],
    "env": {},
    "rollback": {},
    "strict_fallback": True,
    "expected_kernels": [],
    "forbidden_kernels": ["prefill_gen_sched_gemm_* (on the default path)", "Ops.INS"],
    "authority_gate": "tinygrad/llm/model.py PREFILL_GRAPH_GEMM default",
    "promotion_artifacts": ["docs/pure-machine-search.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "tinygrad_scheduler_generated",
    "replacement_scope": "",
    "selector": "env_default",
    "route_attribution": "tinygrad/llm/prefill_routes.py route_prefill_linear default path: PREFILL_GRAPH_GEMM=0 -> x.cast(float16).linear(w.transpose(), bias) inside PREFILL_V2, with model.py installing warmstart TC opts around the prefill jit.",
    "note": "Strict pure-machine-search default for fp16 resident/chunked prefill: ordinary tinygrad graph lowering owns matmul scheduling. The raw RDNA3 graph-GEMM instruction-list route remains opt-in via PREFILL_GRAPH_GEMM=1."},
  "prefill_pipe_role_selective_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "graph-GEMM prefill ubatch=512; role_policy in the spec: pipe for attn_qo/attn_kv/ffn_down, lds for ffn_gate_up out_f==12288"}],
    "env": {"PREFILL_GRAPH_GEMM": "1"},
    "rollback": {"PREFILL_GRAPH_GEMM": "0"},
    "strict_fallback": True,
    "expected_kernels": ["prefill_gen_sched_gemm_*"],
    "forbidden_kernels": ["prefill_graph_gemm_* (on the default path)"],
    "authority_gate": "extra/qk/prefill_generated_schedule_gate.py",
    "promotion_artifacts": ["bench/tg-p4-prefill-generated-schedule/latest.json",
                            "bench/tg-p4-prefill-generated-schedule/summary.md"],
    "purity_status": "research",  # derived from (status=research, provenance=external_handwritten_kernel); was drifted to search_generated_promoted
    "provenance": "external_handwritten_kernel",
    "replacement_scope": "Route B: generated LDS+WMMA codegen substrate (PrefillWMMAScheduleSpec) replacing extra/qk/prefill/wmma.py raw Ops.INS. Schedule SELECTION is spec-generated, but the executing substrate wraps raw RDNA3 instruction lists -> external handwritten kernel under the strict rule.",
    "selector": "env_guard",
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm -> describe_prefill_schedule + emit_prefill_gemm_from_spec; writer extra/qk/prefill_schedule_spec.py (PrefillGEMMScheduleSpec lowered through the RDNA3 WMMA schedule generator ref.build_gemm_pipe / build_gemm_lds2).",
    "note": "The hybrid_machine_search path (pp512 ~4413 pinned): PrefillGEMMScheduleSpec (data) captures the resolved tile/wave/pipeline/role-policy and the schedule is machine-authored from the spec, while the executing substrate is the fast hand-coded RDNA3 WMMA backend atom (ref.build_gemm_pipe / build_gemm_lds2). Selected by PREFILL_GRAPH_GEMM=1 with NO primitive flags. Distinct from the spec-owned transport prefill_wmma_pipe_lds_dbuf_primitive_generated (~1332, primitive flags on), whose LDS2 lifecycle uses an ASM backend atom and is therefore not strictly pure. The legacy fixed emit and PREFILL_GENERATED_SCHEDULE rollback were removed from runtime."},
  "prefill_wmma_pipe_primitive_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["attn_qo", "attn_kv", "ffn_down"], "excluded_roles": ["ffn_gate_up"],
    "quant": ["fp16"],
    "shape_guards": [
      {"role": "attn_qo", "M": 512, "N": 4096, "K": 4096},
      {"role": "attn_kv", "M": 512, "N": 1024, "K": 4096},
      {"role": "ffn_down", "M": 512, "N": 4096, "K": 12288},
      {"note": "graph-GEMM selected, but transport is ordinary generated Tensor matmul with pipe warmstart opts"}],
    "env": {"PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_PIPE_PRIMITIVE": "1"},
    "rollback": {"PREFILL_WMMA_PIPE_PRIMITIVE": "0"},
    "strict_fallback": True,
    "expected_kernels": [],
    "forbidden_kernels": ["extra/qk/prefill/wmma.py instruction-list emitter", "Ops.INS", "prefill_gen_sched_gemm_* raw schedule substrate"],
    "authority_gate": "extra/qk/prefill_pipe_mvp_artifact.py --route-sample-correctness",
    "promotion_artifacts": ["bench/prefill-pipe-mvp/latest.json", "docs/prefill-lessons-ledger.md"],
    "purity_status": "research",
    "provenance": "tinygrad_scheduler_generated",
    "replacement_scope": "",
    "selector": "env_guard",
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm, gated by PREFILL_GRAPH_GEMM=1 + PREFILL_WMMA_PIPE_PRIMITIVE=1, lowers pipe roles through x.cast(float16).contiguous() @ w.cast(float16).contiguous().transpose() and installs WMMA pipe warmstart opts; no route-local raw instruction-list emitter.",
    "note": "Generated pipe primitive MVP route identity for attn_qo, attn_kv, and ffn_down. This deliberately does not reclassify prefill_pipe_role_selective_generated, which remains the raw Ops.INS oracle route. ffn_gate_up remains excluded on the existing LDS/raw route until a separate primitive exists. Whole-model AMD:ISA ownership is separately blocked on broader non-GEMM renderer coverage."},
  "prefill_wmma_pipe_lds_dbuf_primitive_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [
      {"role": "attn_qo", "M": 512, "N": 4096, "K": 4096, "primitive": "pipe"},
      {"role": "attn_kv", "M": 512, "N": 1024, "K": 4096, "primitive": "generated_pipe_no_local_stage"},
      {"role": "ffn_down", "M": 512, "N": 4096, "K": 12288, "primitive": "pipe"},
      {"role": "ffn_gate_up", "M": 512, "N": 12288, "K": 4096, "primitive": "lds_dbuf"},
      {"note": "graph-GEMM selected with generated pipe primitive roles plus generated LDS/DBUF primitive for ffn_gate_up; attn_kv is generated pipe transport with local staging disabled, retaining resource-gated raw fallback when that policy is disabled or unsafe"}],
    "env": {"PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1",
            "PREFILL_DBUF": "1"},
    "rollback": {"PREFILL_WMMA_PIPE_PRIMITIVE": "0", "PREFILL_WMMA_LDS_PRIMITIVE": "0", "PREFILL_DBUF": "0"},
    "strict_fallback": True,
    "expected_kernels": [],
    "forbidden_kernels": ["extra/qk/prefill/wmma.py instruction-list emitter outside explicit safety fallback",
                          "Ops.INS outside explicit safety fallback",
                          "prefill_gen_sched_gemm_* raw schedule substrate outside explicit safety fallback"],
    "authority_gate": "extra/qk/prefill_pipe_mvp_artifact.py --lds-primitive --lds-sample-correctness",
    "promotion_artifacts": ["bench/prefill-pipe-mvp/ffn-gate-up-lds-primitive.json",
                            "bench/prefill-whole-synced/lds-dbuf-promoted-smoke.json",
                            "docs/prefill-lessons-ledger.md"],
    "purity_status": "research",
    "provenance": "compiler_primitive_spec_owned",
    "replacement_scope": "S10 LDS2 route classification: WMMALDSSpec/LDS2 lifecycle ownership with ASM backend atom. This is not pure generated and not the full hand-kernel oracle.",
    "selector": "env_guard",
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm, gated by PREFILL_GRAPH_GEMM=1 + PREFILL_WMMA_PIPE_PRIMITIVE=1 + PREFILL_WMMA_LDS_PRIMITIVE=1 + PREFILL_DBUF=1; attn_qo/ffn_down use generated pipe primitive transport, attn_kv uses generated pipe transport with local staging disabled when generated local staging would exceed 64 KiB LDS, and ffn_gate_up uses the generated LDS/DBUF primitive.",
    "route_classification": "compiler_primitive_spec_owned__asm_backend_atom",
    "note": "Composed S10 route identity for lifecycle attribution: pipe primitive roles for attn_qo, attn_kv, and ffn_down; attn_kv disables local staging under LDS/DBUF because captured generated HIP with local staging declared 69632 bytes shared memory, and it retains resource-gated raw fallback if the no-local-stage policy is disabled or unsafe; WMMALDSSpec-owned LDS/DBUF primitive for ffn_gate_up. The S10 classification is compiler_primitive_spec_owned__asm_backend_atom: spec/compiler owned with a reusable ASM backend atom, distinct from both pure generated transport and the raw graph-GEMM full hand-kernel oracle."},
  "prefill_wmma_lds_single_buffer_candidate_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up"], "excluded_roles": ["attn_qo", "attn_kv", "ffn_down"], "quant": ["fp16"],
    "shape_guards": [{"role": "ffn_gate_up", "M": 512, "N": 12288, "K": 4096,
                      "primitive": "generated_lds_single_buffer"}],
    "env": {"PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1"},
    "rollback": {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": "", "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": ""},
    "strict_fallback": True, "expected_kernels": [],
    "forbidden_kernels": ["extra/qk/prefill/wmma.py instruction-list emitter", "Ops.INS",
                          "prefill_gen_sched_gemm_* raw schedule substrate"],
    "authority_gate": "canonical candidate admission + route binding + full-output correctness + kernel timing",
    "promotion_artifacts": [], "purity_status": "research", "provenance": "tinygrad_scheduler_generated",
    "replacement_scope": "Capability-admitted exact-workload candidates on the generated Tensor matmul/LDS single-buffer transport.",
    "selector": "canonical_candidate_env", "candidate_identity": None,
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm with a canonically admitted payload/hash pair; dynamic candidate identity is carried through KernelInfo and compiler cache identity.",
    "note": "Research route for frozen gfx1100 single-buffer capability admission. Each candidate remains exact-workload bound and requires emitted proof, correctness, and timing before promotion."},
  "prefill_wmma_lds_dbuf_primitive_mixed": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["fp16"],
    "shape_guards": [
      {"role": "attn_qo", "M": 512, "N": 4096, "K": 4096, "primitive": "raw_pipe_oracle"},
      {"role": "attn_kv", "M": 512, "N": 1024, "K": 4096, "primitive": "raw_pipe_oracle"},
      {"role": "ffn_down", "M": 512, "N": 4096, "K": 12288, "primitive": "raw_pipe_oracle"},
      {"role": "ffn_gate_up", "M": 512, "N": 12288, "K": 4096, "primitive": "lds_dbuf"},
      {"note": "decoupled S10 route: only ffn_gate_up exercises the WMMALDSSpec-owned LDS/DBUF primitive"}],
    "env": {"PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1", "PREFILL_DBUF": "1"},
    "rollback": {"PREFILL_WMMA_LDS_PRIMITIVE": "0", "PREFILL_DBUF": "0"},
    "strict_fallback": True,
    "expected_kernels": ["prefill_gen_sched_gemm_*"],
    "forbidden_kernels": [],
    "authority_gate": "extra/qk/prefill/s10_compile_capture.py --scenario lds-only",
    "promotion_artifacts": ["bench/prefill-pipe-mvp/ffn-gate-up-lds-primitive.json",
                            "bench/prefill-s10-lds2-ownership/compile-capture/report-lds-only.json"],
    "purity_status": "research",
    "provenance": "compiler_primitive_spec_owned",
    "replacement_scope": "S10 decoupled LDS2 route classification: WMMALDSSpec owns the ffn_gate_up LDS lifecycle while pipe roles remain on the existing raw graph-GEMM oracle. This isolates LDS migration from the generated pipe primitive.",
    "selector": "env_guard",
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm, gated by PREFILL_GRAPH_GEMM=1 + PREFILL_WMMA_LDS_PRIMITIVE=1 + PREFILL_DBUF=1 and PREFILL_WMMA_PIPE_PRIMITIVE unset/0; attn_qo/attn_kv/ffn_down use the existing raw pipe oracle, while ffn_gate_up uses the WMMALDSSpec-owned LDS/DBUF primitive.",
    "route_classification": "compiler_primitive_spec_owned__mixed_raw_pipe",
    "note": "This is the S10 decoupling route. It is not strict pure and not the composed generated pipe+LDS route; it exists to validate the primitive LDS ownership step without the attn_kv generated-pipe LDS overflow."},
  "prefill_q4k_direct_tile4x4_default": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "direct-packed Q4_K prefill, memory-safe 14B/32B route"}],
    "env": {},
    "rollback": {"PREFILL_Q4K_DIRECT_SCHEDULE": "legacy"},
    "baseline_route_id": "prefill_q4k_direct_packed_load_direct_out",
    "strict_fallback": True,
    "expected_kernels": ["q4k_gen_prefill_direct_out_*", "q4k_gen_prefill_partials_*", "q4k_gen_prefill_reduce_out_*"],
    "authority_gate": "extra/qk/prefill_boltbeam_trace.py",
    "promotion_artifacts": ["docs/prefill-lessons-ledger.md"],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "replacement_scope": "",
    "selector": "env_default",
    "route_attribution": "tinygrad/llm/prefill_routes.py Q4_K direct-packed default -> extra/qk/q4k_prefill_route_spec.py Q4KPrefillRouteSpec + emit_q4k_packed_prefill_kernel; _direct_packed_opts still owns the selected schedule LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4.",
    "note": "Q4_K direct-packed prefill provenance conversion. The default packed-load path no longer binds q4k_gemm_packed_load_direct_out_kernel/q4k_gemm_packed_load_kernel from prefill_routes.py; it emits q4k_gen_prefill_* kernels from Q4KPrefillRouteSpec. This preserves the memory-safe direct-packed schedule while moving the default route out of route-local hand-template ownership. PREFILL_Q4K_REDUCE_OUT=1 is also descriptor-owned as a correct-not-fast research selector."},
  "prefill_q6k_direct_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["ffn_gate_up", "attn_qo", "ffn_down", "attn_kv"], "excluded_roles": [],
    "quant": ["Q6_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "direct-packed Q6_K prefill, memory-safe route"}],
    "env": {},
    "rollback": {},
    "baseline_route_id": "ordinary_tinygrad_graph",
    "strict_fallback": True,
    "expected_kernels": ["q6k_gen_prefill_direct_out_*", "q6k_gen_prefill_partials_*"],
    "forbidden_kernels": ["q6k_gemm_packed_load_* (on the default path)", "q6k_gemm_* (on the default packed-load path)"],
    "authority_gate": "test/unit/test_q6k_prefill_route_spec.py + test/unit/test_llm_prefill_routes.py",
    "promotion_artifacts": [],
    "purity_status": "search_generated_promoted",
    "provenance": "machine_authored_generated",
    "selector": "env_default",
    "route_attribution": "tinygrad/llm/prefill_routes.py Q6_K direct-packed branch -> extra/qk/q6k_prefill_route_spec.py Q6KPrefillRouteSpec + emit_q6k_packed_prefill_kernel; reuses the Q6_K packed-load dequant grammar while descriptor data owns prefill output layout, token axis, parts, and opts.",
    "note": "Q6_K direct-packed prefill provenance conversion. The default packed-load path no longer binds q6k_gemm_packed_load_* hand UOp templates from prefill_routes.py; it emits q6k_gen_prefill_* kernels from Q6KPrefillRouteSpec. Non-packed-load Q6_K remains an explicit legacy debug path behind PREFILL_Q6K_PACKED_LOAD=0."},
  # prefill_q4k_generated_tile_research REMOVED 2026-07-06 (no backups): PREFILL_QK_GENERATED_TILE now fails loud.
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
    "promotion_artifacts": ["docs/prefill-lessons-ledger.md"],
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
    "authority_gate": "extra/qk/q4k_wmma_tiled_lowering_feasibility.py + extra/qk/q4k_wmma_tiled_microgate.py + extra/qk/q4k_wmma_tiled_surface_gate.py + extra/qk/q4k_wmma_tiled_lifecycle_gate.py + extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py + extra/qk/q4k_wmma_tiled_no_hand_kernel_gate.py",
    "promotion_artifacts": ["docs/prefill-lessons-ledger.md",
                            "docs/prefill-lessons-ledger.md"],
    "purity_status": "research",
    "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=wmma_tiled -> extra/qk/prefill_int8_wmma_spec.py Q4KInt8WMMATiledPrefillSpec + one-tile emit_q4k_int8_wmma_tiled_prefill_tensor for the direct runtime path (with bounded fallback to emit_q4k_int8_wmma_tiled_lifecycle_tensor); route authority additionally requires extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py to run a bounded synthetic generated tiled loop via emit_q4k_int8_wmma_tiled_exec_tensor.",
    "note": "One-tile tiled WMMA substrate is correct and codegen-valid: lowering feasibility and Q4_K/Q8_1 microgate pass on AMD with wmma_i32_16x16x16_iu8. Synthetic 14B role-shape execution now runs through a bounded generated tiled loop (emit_q4k_int8_wmma_tiled_exec_tensor) for route authority, while full runtime role-shape execution is still blocked on scheduler-owned tiled loop ownership. Promotion requires canonical 14B smoke beating the current direct-packed default."},
  "prefill_14b_q4k_q8_1_hybrid_mmq_atom": {
    "workload": "prefill", "profile_id": "qwen3_14b_q4k_m_gfx1100", "status": "research",
    "roles": ["ffn_gate_up"], "excluded_roles": ["attn_qo", "attn_kv", "ffn_down"],
    "quant": ["Q4_K"],
    "shape_guards": [{"role": "ffn_gate_up", "M": 512, "N": 17408, "K": 5120,
                      "activation": "Q8_1", "atom": "prefill_14b_q4k_q8_1_hybrid_mmq_atom",
                      "note": "M7 one-role transfer scaffold: only ffn_gate/up may bind the hybrid MMQ atom; every other prefill role remains direct-packed."}],
    "env": {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "1", "PREFILL_ROUTE_STRICT": "1"},
    "rollback": {"PREFILL_14B_Q4K_Q8_1_MMQ_ATOM": "0", "PREFILL_ROUTE": "direct_packed"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default",
    "strict_fallback": True,
    "expected_kernels": ["prefill_14b_q4k_q8_1_hybrid_mmq_atom_*"],
    "forbidden_kernels": ["q4k_gen_prefill_direct_out_* on ffn_gate_up when selected",
                          "hidden direct-packed fallback while route attribution claims hybrid MMQ",
                          "hybrid MMQ atom on attn_qo/attn_kv/ffn_down before explicit M8 rows exist"],
    "authority_gate": "M7 route policy load gate plus future whole-prefill authority artifact",
    "promotion_artifacts": [],
    "purity_status": "research",
    "provenance": "compiler_primitive_spec_owned",
    "replacement_scope": "14B Q4_K/Q8_1 MMQ hybrid route over a hand-written backend atom. This is opt-in research scaffolding only; the default remains direct-packed until whole-prefill authority proves promotion.",
    "selector": "BoltBeam_route_policy_explicit_atom_available",
    "route_attribution": "M7 scaffold only: QK_ROUTE_POLICY may select this route for 14B ffn_gate_up rows=17408 cols=5120 after atom_available=true is declared by the policy row. Runtime atom binding is intentionally fail-closed until the atom implementation lands.",
    "route_classification": "compiler_primitive_spec_owned__hand_mmq_backend_atom",
    "m8_expansion_order": [
      {"order": 1, "role": "ffn_gate_up", "quant": "Q4_K", "status": "m7_scaffold"},
      {"order": 2, "role": "attn_qo", "quant": "Q4_K", "status": "future_m8"},
      {"order": 3, "role": "attn_kv", "quant": "Q4_K", "status": "future_m8"},
      {"order": 4, "role": "ffn_down", "quant": "Q6_K", "status": "future_m8_if_profiled_hot"}],
    "note": "No default promotion. This route exists so one-role transfer artifacts can name the hybrid MMQ atom and fail loud when the atom is unavailable. The mixed policy is ffn_gate_up -> hybrid MMQ atom and all other 14B prefill roles -> direct-packed."},
  "prefill_q4k_reduce_out_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "correct_not_fast",
    "roles": ["ffn_gate_up", "attn_qo", "attn_kv"], "excluded_roles": ["ffn_down"],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "K": 5120, "note": "tested with GROUP:0:10 on K=5120 roles"}],
    "env": {"PREFILL_Q4K_REDUCE_OUT": "1"},
    "rollback": {"PREFILL_Q4K_REDUCE_OUT": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default",
    "strict_fallback": True,
    "expected_kernels": ["q4k_gen_prefill_reduce_out_*"],
    "authority_gate": "extra/qk/prefill_boltbeam_trace.py",
    "promotion_artifacts": ["docs/prefill-lessons-ledger.md"],
    "purity_status": "research",
    "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_REDUCE_OUT=1 -> extra/qk/q4k_prefill_route_spec.py Q4KPrefillRouteSpec(output_layout='reduce_out') + emit_q4k_packed_prefill_kernel.",
    "note": "Default-off primitive correctness fix. It replaces the manual direct-output accumulator recurrence with a real Ops.REDUCE, making GROUP schedules numerically valid: GROUP:0:10 on real 14B ffn_gate rel_rmse ~=1.6e-6 vs the lossless direct path. It is not promoted because clean pp512 is 169.7 tok/s vs 173.6 for the current Q4 tile4x4 manual direct-output default. Use this as the correctness foundation for future grouped/staged combine work."},
  # prefill_q4k_mmq_direct_out_research (+ the mmq/sdot4 PREFILL_Q4K_Q8 modes) REMOVED 2026-07-06 (no backups):
  # handwritten scalar-sdot4/Q8_1 MMQ prefill kernels deleted (confirmed ~85-237 tok/s dead end). Only the
  # generated int8-WMMA substrate (prefill_q4k_int8_wmma_generated_research) remains selectable via PREFILL_Q4K_Q8.
  # prefill_pipe_global_rollback REMOVED 2026-07-06 (no backups): PREFILL_PIPE_ROLE_SELECTIVE=0 now fails loud.
}

# Closed / refuted axes -- do not re-search without a NEW premise (PMS-R3 do_not_search carries these forward).
REFUTED = [
  {"axis": "q6k_direct_half_warp_route", "disposition": "refuted: W==D regression -4.77..-6.06% (median -5.44%)",
   "citation": "bench/amd-isa-backend-q6k-direct-speed/latest.json", "route_id": "decode_q6k_direct_refuted"},
  {"axis": "q4k_offline_layout_reshuffle", "disposition": "deprioritized: G3 matches owned, no layout gap to recover",
   "citation": "bench/amd-isa-backend-g3-weight-promotion/search_space_update.json"},
  {"axis": "prefill_q4k_simple_uop_cooperative_lane_tile", "domain": "prefill",
   "disposition": "refuted on 14B ffn_gate_up: lane_partials 0.99 GB/s; direct_warp sweep best 1.29 GB/s vs current direct-packed floor ~2.11 GB/s",
   "citation": "docs/prefill-lessons-ledger.md", "route_id": "prefill_q4k_generated_tile_research"},
  {"axis": "prefill_q4k_direct_group_reduce_current_uop", "domain": "prefill",
   "disposition": "refuted for the manual direct-output recurrence: GROUP:0:10 looked fast but is numerically wrong on real 14B ffn_gate (rel_rmse ~1.26); use PREFILL_Q4K_REDUCE_OUT=1 for correct-but-not-fast grouped semantics",
   "citation": "docs/prefill-lessons-ledger.md", "route_id": "prefill_q4k_direct_tile4x4_default"},
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
    # purity_status is derived, not declared: any stored value must equal derive_purity_status(status, provenance).
    if "purity_status" in r:
      expected = derive_purity_status(r["status"], str(prov))
      if r["purity_status"] != expected:
        errors.append(f"{rid}: purity_status={r['purity_status']!r} drifted from derived {expected!r} "
                      f"(status={r['status']}, provenance={prov})")
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
