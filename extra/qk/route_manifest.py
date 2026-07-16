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
import hashlib, json, pathlib
from collections.abc import Mapping
from types import MappingProxyType

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
  "decode_flash_live_split_g4_kvboth": {
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
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "superseded_rollback",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [],
    "quant": ["Q4_K", "Q6_K", "fp16"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "ordinary PREFILL_V2 fp16 matmul under warmstart TC opts"}],
    "env": {"PREFILL_GRAPH_GEMM": "0"},
    "rollback": {},
    "strict_fallback": True,
    "expected_kernels": [],
    "forbidden_kernels": ["prefill_gen_sched_gemm_* (on the default path)", "Ops.INS"],
    "authority_gate": "tinygrad/llm/model.py PREFILL_GRAPH_GEMM default",
    "promotion_artifacts": ["docs/pure-machine-search.md"],
    "purity_status": "research",
    "provenance": "tinygrad_scheduler_generated",
    "replacement_scope": "",
    "selector": "env_default",
    "route_attribution": "tinygrad/llm/prefill_routes.py route_prefill_linear default path: PREFILL_GRAPH_GEMM=0 -> x.cast(float16).linear(w.transpose(), bias) inside PREFILL_V2, with model.py installing warmstart TC opts around the prefill jit.",
    "note": "Pure scheduler rollback for unsupported shapes and explicit PREFILL_GRAPH_GEMM=0. It was superseded for the exact gfx1100 pp512 four-role workload by the correctness-gated generated WMMA-LDS candidate set."},
  "prefill_wmma_lds_dbuf_generated": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "promoted_default",
    "roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"], "excluded_roles": [], "quant": ["fp16"],
    "shape_guards": [
      {"role": "attn_qo", "M": 512, "N": 4096, "K": 4096, "primitive": "generated_lds_buffer2"},
      {"role": "attn_kv", "M": 512, "N": 1024, "K": 4096, "primitive": "generated_lds_buffer2"},
      {"role": "ffn_down", "M": 512, "N": 4096, "K": 12288, "primitive": "generated_lds_buffer2"},
      {"role": "ffn_gate_up", "M": 512, "N": 12288, "K": 4096, "primitive": "generated_lds_buffer2"}],
    "env": {},
    "rollback": {"PREFILL_GRAPH_GEMM": "0"},
    "strict_fallback": True, "expected_kernels": [],
    "forbidden_kernels": ["extra/qk/prefill/wmma.py instruction-list emitter", "Ops.INS",
                          "prefill_gen_sched_gemm_* raw schedule substrate"],
    "authority_gate": "canonical candidate admission + four-role route census + whole-model greedy parity + pinned whole-prefill timing",
    "promotion_artifacts": [
      "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/whole-model-quality.json",
      "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/whole-prefill-pinned.json"],
    "purity_status": "search_generated_promoted", "provenance": "tinygrad_scheduler_generated",
    "replacement_scope": "",
    "selector": "promoted_candidate_set", "candidate_identity": None,
    "candidate_set_path": "bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/candidate-set.json",
    "candidate_roles": ["attn_qo", "attn_kv", "ffn_down", "ffn_gate_up"],
    "route_attribution": "extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm with a canonically admitted payload/hash pair; dynamic candidate identity is carried through KernelInfo and compiler cache identity.",
    "note": "Promoted pure generated gfx1100 pp512 WMMA-LDS double-buffer route. The selected candidate payloads are the authoritative buffer-count source and specify buffer_count=2 for all four dense roles. Exact canonical identities are loaded from the promoted candidate-set artifact; research routes are not eligible for implicit selection."},
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
    "authority_gate": "extra/qk/q4k_wmma_tiled_lowering_feasibility.py + extra/qk/q4k_wmma_tiled_microgate.py + extra/qk/q4k_wmma_tiled_surface_gate.py + extra/qk/q4k_wmma_tiled_role_shape_exec_gate.py + extra/qk/q4k_wmma_full_role_contract_gate.py + extra/qk/q4k_wmma_tiled_no_hand_kernel_gate.py",
    "promotion_artifacts": ["docs/prefill-lessons-ledger.md"],
    "purity_status": "research",
    "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=wmma_tiled -> extra/qk/prefill_int8_wmma_spec.py Q4KInt8WMMATiledPrefillSpec; one tile uses emit_q4k_int8_wmma_tiled_prefill_tensor and larger shapes use emit_q4k_int8_wmma_tiled_scheduler_tensor. Typed ScheduleHints centralize partial-contiguous ownership and TC selection; the correction reduction is a bounded prerequisite while the full [groups,M,N] RAW tensor is never materialized.",
    "note": "Generated scheduler-owned tiled WMMA compiles all four exact Qwen3-14B role shapes to iu8 WMMA and passes bounded full-K numeric probes; full attn_kv (512,1024,5120) also passed sampled real-GPU correctness. Packed-Q4 decode is fused so the 14B model runs without persistent expanded-weight buffers. A route-bound replay measured 140 tok/s versus the recorded direct-packed authority baseline of 364.5 tok/s, so this schedule remains research-only and must not become the default."},
  "prefill_q4k_packed_ds4_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up", "attn_qo", "attn_kv"], "excluded_roles": ["ffn_down"],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "logical packed DS4 candidate; Q6_K remains on its own route"}],
    "env": {"PREFILL_Q4K_Q8": "packed_ds4"}, "rollback": {"PREFILL_Q4K_Q8": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default", "strict_fallback": True,
    "expected_kernels": ["q4k_q8_1_mmq_ds4_dot4x4_atom_*"],
    "forbidden_kernels": ["prefill_q4k_q8_1_wmma_generated_gemm_*"],
    "authority_gate": "docs/14b-mmq-logical-vocabulary-scope-20260715.md P1-P5",
    "promotion_artifacts": [], "purity_status": "research", "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=packed_ds4 -> shared logical candidate + DS4 pack/emitter",
    "note": "Opt-in logical packed DS4 research route. It includes GPU Q8 preparation, keeps direct-packed as rollback, and is not admitted to the default route or promotion ledger."},
  "prefill_q4k_packed_row_major_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up", "attn_qo", "attn_kv"], "excluded_roles": ["ffn_down"],
    "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": "*", "K": "*", "note": "logical row-major Q8 storage candidate; Q6_K remains on its own route"}],
    "env": {"PREFILL_Q4K_Q8": "packed_row_major"}, "rollback": {"PREFILL_Q4K_Q8": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default", "strict_fallback": True,
    "expected_kernels": ["q4k_q8_1_mmq_ds4_dot4x4_atom_*"],
    "forbidden_kernels": ["prefill_q4k_q8_1_wmma_generated_gemm_*"],
    "authority_gate": "docs/14b-mmq-logical-vocabulary-scope-20260715.md P1-P5",
    "promotion_artifacts": [], "purity_status": "research", "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=packed_row_major -> shared logical candidate + row-major Q8 pack/emitter",
    "note": "Opt-in row-major Q8 storage experiment. It removes DS4 transpose materialization but includes Q8 quantization and sums; direct-packed remains rollback and no promotion is implied."},
  "prefill_q4k_packed_fused_research": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up", "attn_qo", "attn_kv"], "excluded_roles": ["ffn_down"],
    "quant": ["Q4_K"], "shape_guards": [{"M": 512, "N": "*", "K": "*"}],
    "env": {"PREFILL_Q4K_Q8": "packed_fused"}, "rollback": {"PREFILL_Q4K_Q8": "0"},
    "baseline_route_id": "prefill_q4k_direct_tile4x4_default", "strict_fallback": True,
    "expected_kernels": ["q8_1_mmq_fused_row_major_*", "q4k_q8_1_mmq_ds4_dot4x4_atom_*"],
    "forbidden_kernels": ["prefill_q4k_q8_1_wmma_generated_gemm_*"],
    "authority_gate": "docs/14b-mmq-logical-vocabulary-scope-20260715.md P1-P5",
    "promotion_artifacts": [], "purity_status": "research", "provenance": "machine_authored_generated",
    "selector": "env_guard",
    "route_attribution": "tinygrad/llm/prefill_routes.py PREFILL_Q4K_Q8=packed_fused -> fused Q8 producer + shared logical DS4 atom",
    "note": "Opt-in fused Q8 producer experiment. It writes replicated scales/sums to keep the atom ABI explicit; direct-packed remains rollback and no promotion is implied."},
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
  "prefill_q4k_q8_1_hybrid_mmq_atom": {
    "workload": "prefill", "profile_id": PROFILE_PREFILL, "status": "research",
    "roles": ["ffn_gate_up"], "excluded_roles": ["attn_qo", "ffn_down", "attn_kv"], "quant": ["Q4_K"],
    "shape_guards": [{"M": 512, "N": 17408, "K": 5120, "note": "machine-generated MMQ candidate"}],
    "env": {}, "rollback": {"route": "direct_packed"}, "baseline_route_id": "direct_packed",
    "strict_fallback": True, "expected_kernels": ["q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0"],
    "forbidden_kernels": [], "authority_gate": "correctness evidence + resource evidence + timing evidence",
    "promotion_artifacts": [], "purity_status": "research", "provenance": "machine_authored_generated",
    "selector": "research_descriptor_only", "candidate_identity": "prefill_q4k_q8_1_hybrid_mmq_atom",
    "backend_strategy": "q4k_q8_1_mmq_amd_ds4_coop_tile_atom_v0", "rollback_route": "direct_packed",
    "research_only": True,
    "route_attribution": "extra/qk/mmq_machine_search.py candidate inventory; no tinygrad/llm/prefill_routes.py binding",
    "note": "Descriptor-only machine-generated MMQ candidate. Promotion is blocked until correctness, resource, and timing evidence are all present; direct_packed remains the rollback/default route.",
  },
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
  {"axis": "g5_block_tile_as_default", "compatibility_aliases": ("g5_block_tile_8b_as_default",), "domain": "attention", "disposition": "correct_not_fast: token-identical + route-bound but 87.6% of owned @ctx512 / 95.6% @ctx4096 (TG-P5)",
   "citation": "bench/tg-p5-attention-generated-default/latest.json", "legacy_route_id": "decode_flash_block_tile_g5_8b_refuted"},
  {"axis": "g5_block_tile_L_geometry", "compatibility_aliases": ("g5_block_tile_8b_L_geometry",), "domain": "attention", "disposition": "refuted: L=128 is the geometry optimum (87.7%/95.9%); larger L monotonically worse (69%/75.6% at L=576, occupancy-starved) -- the generated route needs ~36 splits for parallelism so it over-launches at low ctx (TG-P8.2)",
   "citation": "bench/tg-p8-generated-8b-attention-parity/geometry_search.json"},
  {"axis": "g5_block_tile_combine_lifecycle_cap", "compatibility_aliases": ("g5_block_tile_8b_combine_lifecycle_cap",), "domain": "attention", "disposition": "blocking: the generated 3-kernel gmax+combine lifecycle is 556us/fwd (83% of the ctx4096 attention delta) vs owned's fused 224us -> BINDING cap at ctx4096 (95.9%); a perfect tile saves only 112us. Combine COLLAPSE is refuted (guardrail #3); reopen only with a NEW non-collapse coordination primitive (TG-P8.1/P8.2)",
   "citation": "bench/tg-p8-generated-8b-attention-parity/latest.json"},
  {"axis": "live_split_geometry_tile", "compatibility_aliases": ("live_split_geometry_8b_tile",), "domain": "attention", "disposition": "SOLVED/PROMOTED: live-context split geometry (fixed S, per=ceildiv(Tc,S)) is expressible in generated UOp; the live-split route plus KV_BOTH staging and fused combine is now the 8B generated default. extra/qk/live_split_geometry.py",
   "citation": "bench/tg-p9-pure-attention-primitive-route/live_split_tile_microgate.json"},
  {"axis": "split_preserving_lse_combine", "compatibility_aliases": ("split_preserving_lse_combine_8b",), "domain": "attention", "disposition": "EMITTER_BLOCKED (TG-P9.4): a split-preserving generated combine (de-dup the per-d fexp / fuse gmax without collapsing Hq*S or Hq*Hd) mis-vectorizes the reduction-accumulator REG to a non-assignable make_float4(...) store; REG_STORE_DEVEC=1 compiles but NaNs. The ctx4096 556us combine cap cannot be removed in current AMD codegen. Reopen: a codegen fix keeping the reduction-accumulator REG scalar for a multi-reduce/weight-sharing combine.",
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
ROUTE_COMPATIBILITY_ALIASES = (
  {"canonical_route_id": "decode_flash_live_split_g4_kvboth",
   "compatibility_aliases": ("decode_flash_live_split_g4_8b_kvboth",)},
  {"canonical_route_id": "prefill_q4k_q8_1_hybrid_mmq_atom",
   "compatibility_aliases": ("prefill_14b_q4k_q8_1_hybrid_mmq_atom",)},
)

def _route_alias_map() -> dict[str, str]:
  return {alias: row["canonical_route_id"] for row in ROUTE_COMPATIBILITY_ALIASES
          for alias in row["compatibility_aliases"]}

class _CanonicalRouteTable(dict):
  """Canonical keys with exact legacy reads for old artifact/registry consumers."""
  def __contains__(self, route_id) -> bool:
    return dict.__contains__(self, route_id) or route_id in _route_alias_map()
  def __getitem__(self, route_id):
    return dict.__getitem__(self, _route_alias_map().get(route_id, route_id))
  def get(self, route_id, default=None):
    try: return self[route_id]
    except KeyError: return default

ROUTES = _CanonicalRouteTable(ROUTES)

def immutable_route_registry():
  """Return an immutable manifest snapshot for explicit selector/admission use."""
  def freeze(value):
    if isinstance(value, dict): return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)): return tuple(freeze(v) for v in value)
    return value
  return freeze(dict(ROUTES))

def canonical_route_id(route_id: str, registry: Mapping|None = None) -> str:
  """Resolve one complete legacy spelling. Prefix, case-folded, and partial matches are intentionally unsupported."""
  aliases = _route_alias_map()
  registry = immutable_route_registry() if registry is None else registry
  if route_id in registry: return route_id
  if route_id in aliases: return aliases[route_id]
  raise KeyError(f"unknown route_id {route_id!r}; known: {sorted((*registry, *aliases))}")

def route(route_id: str, registry: Mapping|None = None) -> dict:
  registry = immutable_route_registry() if registry is None else registry
  return registry[canonical_route_id(route_id, registry)]

def route_env(route_id: str) -> dict:
  """The env vars to SET to force this route onto the active path ({} means it is the shipped default)."""
  return dict(route(route_id).get("env", {}))

def rollback_env(route_id: str) -> dict:
  """The env vars that leave this route for its rollback target ({} means it IS a rollback target)."""
  return dict(route(route_id).get("rollback", {}))

# Semantic policy identity deliberately excludes benchmark/model labels.  Legacy artifacts may still carry those
# labels, but they are exposed only as provenance by promoted_prefill_candidate_policy().
_PROVENANCE_KEYS = frozenset(("profile", "profile_id", "profiles", "model", "model_id", "model_name",
                              "model_path", "filename", "size_label", "model_size"))

def _semantic_json(value) -> str:
  def clean(x):
    if isinstance(x, Mapping):
      return {str(k):clean(v) for k,v in sorted(x.items(), key=lambda item: str(item[0]))
              if str(k).lower().replace("-", "_") not in _PROVENANCE_KEYS}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if x is None or isinstance(x, (str, bool, int, float)): return x
    raise TypeError(f"semantic identity requires JSON values, got {type(x).__name__}")
  return json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _identity(namespace: str, value) -> str:
  return f"{namespace}:sha256:" + hashlib.sha256(_semantic_json(value).encode("ascii")).hexdigest()

def canonical_capability_identity(capability: Mapping) -> str:
  """Identity of the complete scanned target/route capability contract; labels never participate."""
  if not isinstance(capability, Mapping) or not capability: raise ValueError("capability must be a non-empty mapping")
  _target(capability.get("target"))
  return _identity("capability", capability)

def _target(value) -> dict:
  if not isinstance(value, Mapping): raise ValueError("exact scanned target facts are required")
  try: out = {"backend": str(value["backend"]), "arch": str(value["arch"]), "wave_size": int(value["wave_size"])}
  except (KeyError, TypeError, ValueError): raise ValueError("target requires backend/arch/wave_size scanner facts") from None
  if not out["backend"] or not out["arch"] or out["wave_size"] <= 0:
    raise ValueError("target requires backend/arch/wave_size scanner facts")
  # Preserve additional scanned geometry/resource facts as material identity inputs.
  return {**value, **out}

def _shape(row: Mapping) -> tuple[int, int, int]:
  shape = row.get("shape", row)
  try: out = tuple(int(shape.get(lo, shape.get(lo.upper()))) for lo in ("m", "n", "k"))
  except (AttributeError, TypeError, ValueError): raise ValueError("policy row requires exact positive M/N/K") from None
  if any(v <= 0 for v in out): raise ValueError("policy row requires exact positive M/N/K")
  return out  # type: ignore[return-value]

def _inventory_rows(inventory: Mapping) -> list[dict]:
  rows = inventory.get("rows")
  if not isinstance(rows, list) or not rows: raise ValueError("inventory requires non-empty rows")
  out = []
  for row in rows:
    if not isinstance(row, Mapping): raise ValueError("malformed inventory row")
    phase, role = row.get("phase", "prefill"), row.get("role")
    quant = row.get("quant_format", row.get("quant"))
    if not all(isinstance(x, str) and x for x in (phase, role, quant)): raise ValueError("inventory row lacks phase/role/quant")
    target = _target(row.get("target", inventory.get("target")))
    out.append({"phase": phase, "role": role, "quant": quant, "shape": dict(zip(("m", "n", "k"), _shape(row))),
                "target": target, "tensor_identities": sorted(row.get("tensor_identities", ())),
                "packed_abi": row.get("packed_abi", row.get("layout")), "call_count": row.get("call_count")})
  return out

def canonical_inventory_identity(inventory: Mapping) -> str:
  """Canonical identity of exact routed inventory content, independent of model/profile spelling."""
  rows = sorted(_inventory_rows(inventory), key=lambda r: (r["phase"], r["role"], r["quant"], *r["shape"].values()))
  derived = _identity("inventory", rows)
  recorded = inventory.get("inventory_identity")
  if recorded is not None:
    if not isinstance(recorded, str) or not recorded: raise ValueError("invalid inventory identity")
    if recorded.startswith("inventory:sha256:") and recorded != derived:
      raise ValueError("inventory identity mismatch")
    # Validate the un-namespaced identity emitted by prefill.workload_inventory without making its profile provenance
    # or model path semantic. The manifest identity additionally includes phase and scanned target facts.
    if inventory.get("schema") == "qk.packed_prefill_workload_inventory.v1":
      legacy_rows = []
      try:
        for row in inventory["rows"]:
          legacy_rows.append({"role": row["role"], "quant_format": row["quant_format"],
            "shape": {x:row["shape"][x] for x in ("m", "n", "k")},
            "layout": {x:row["layout"][x] for x in ("logical", "packed", "block_elems", "block_bytes")},
            "tensor_identities": sorted(row["tensor_identities"]), "call_count": row["call_count"],
            "source_bytes": row["source_bytes"]})
      except (KeyError, TypeError): raise ValueError("malformed workload inventory row") from None
      legacy_rows.sort(key=lambda x: (x["role"], x["quant_format"], x["shape"]["m"], x["shape"]["n"], x["shape"]["k"]))
      expected = hashlib.sha256(json.dumps(legacy_rows, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
      if recorded != expected: raise ValueError("inventory identity mismatch")
  return derived

def canonical_candidate_set_identity(candidate_set: Mapping) -> str:
  """Canonical candidate-set identity. Legacy profile fields are ignored, while payload and target facts remain exact."""
  entries = candidate_set.get("entries")
  if not isinstance(entries, list) or not entries: raise ValueError("candidate set requires non-empty entries")
  canonical = []
  for entry in entries:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("payload"), Mapping): raise ValueError("malformed candidate entry")
    canonical.append({"canonical_identity": entry.get("canonical_identity"), "payload": entry["payload"]})
  return _identity("candidate_set", sorted(canonical, key=_semantic_json))

def canonical_policy_rows(inventory: Mapping, capability: Mapping, candidate_set: Mapping, *,
                          route_id: str = "prefill_wmma_lds_dbuf_generated") -> tuple[dict, ...]:
  """Bind candidates to exact inventory/capability facts. Missing, duplicate, or mismatched coverage fails closed."""
  inv_rows, cap_id = _inventory_rows(inventory), canonical_capability_identity(capability)
  capability_target = _target(capability["target"])
  inv_id, set_id = canonical_inventory_identity(inventory), canonical_candidate_set_identity(candidate_set)
  candidates = {}
  for entry in candidate_set["entries"]:
    workload = entry["payload"].get("workload", {})
    key = (workload.get("phase", "prefill"), workload.get("role"),
           workload.get("quant_format", workload.get("quant", workload.get("dtypes", {}).get("b"))), _shape(workload),
           _semantic_json(_target(workload.get("target"))))
    if key in candidates: raise ValueError(f"duplicate structural candidate {key!r}")
    candidates[key] = entry
  rows = []
  for inv in inv_rows:
    if _semantic_json(inv["target"]) != _semantic_json(capability_target):
      raise ValueError("inventory target does not match scanned capability target")
    supported_phases = capability.get("phases", (capability.get("phase"),))
    supported_quants = capability.get("quant_formats", (capability.get("quant_format"), capability.get("quant")))
    if inv["phase"] not in supported_phases or inv["quant"] not in supported_quants:
      raise ValueError("inventory phase/quant is outside the scanned capability contract")
    key = (inv["phase"], inv["role"], inv["quant"], _shape(inv), _semantic_json(inv["target"]))
    entry = candidates.get(key)
    if entry is None: raise ValueError(f"candidate set does not cover exact inventory row {key[:4]!r}")
    rows.append({"phase": inv["phase"], "role": inv["role"], "quant": inv["quant"], "shape": inv["shape"],
      "target": inv["target"], "capability_identity": cap_id, "inventory_identity": inv_id,
      "candidate_set_identity": set_id, "candidate_identity": entry.get("canonical_identity"),
      "selected_route": route_id, "route_aliases": [route_id]})
  if len(candidates) != len(rows): raise ValueError("candidate set contains rows outside the exact inventory")
  return tuple(rows)

def lookup_policy_row(policy_rows, *, phase: str, role: str, quant: str, shape, target: Mapping,
                      capability_identity: str, inventory_identity: str, candidate_set_identity: str) -> dict | None:
  """Exact structural lookup. No wildcard, profile, status, or default-on fallback is permitted."""
  wanted = (phase, role, quant, _shape({"shape": shape}), _semantic_json(_target(target)), capability_identity,
            inventory_identity, candidate_set_identity)
  matches = [row for row in policy_rows if (row.get("phase"), row.get("role"), row.get("quant"), _shape(row),
    _semantic_json(row.get("target", {})), row.get("capability_identity"), row.get("inventory_identity"),
    row.get("candidate_set_identity")) == wanted]
  if len(matches) > 1: raise ValueError("ambiguous exact manifest policy lookup")
  return dict(matches[0]) if matches else None

def promoted_prefill_candidate_policy() -> dict:
  """Return the single promoted full-kernel candidate policy consumed by runtime and search selection.

  The manifest owns promotion; the candidate-set artifact owns exact payloads and canonical identities. Keeping those
  concerns separate lets machine search replace a candidate set without duplicating its schedules in runtime code.
  """
  route_id = "prefill_wmma_lds_dbuf_generated"
  row = route(route_id)
  if row.get("status") != "promoted_default" or row.get("provenance") != "tinygrad_scheduler_generated":
    raise RuntimeError(f"promoted prefill candidate policy is not eligible: route={route_id} "
                       f"status={row.get('status')!r} provenance={row.get('provenance')!r}")
  relpath = pathlib.Path(str(row["candidate_set_path"]))
  path = pathlib.Path(__file__).resolve().parents[2] / relpath
  if not path.is_file(): raise FileNotFoundError(f"promoted prefill candidate set is missing: {path}")
  artifact = json.loads(path.read_text())
  entries = artifact.get("entries", ())
  if artifact.get("schema") != "boltbeam.full_kernel_candidate_set.v1" or not isinstance(entries, list) or not entries:
    raise RuntimeError("promoted prefill candidate set has an invalid schema or no entries")
  profiles = tuple(sorted({str(entry["payload"]["workload"]["profile"]) for entry in entries}))
  roles = tuple(str(role) for role in row.get("candidate_roles", ()))
  if set(roles) != set(row.get("roles", ())):
    raise RuntimeError(f"promoted prefill candidate roles drifted from route roles: {roles!r} vs {row.get('roles')!r}")
  artifact_roles = {str(entry["payload"]["workload"]["role"]) for entry in entries}
  if artifact_roles != set(roles):
    raise RuntimeError(f"promoted prefill candidate artifact roles drifted from policy: {sorted(artifact_roles)!r} vs {roles!r}")
  return {
    "route_id": route_id, "route_aliases": (route_id,), "candidate_set_path": str(path),
    "candidate_set_identity": canonical_candidate_set_identity(artifact),
    # Compatibility artifact metadata only. It is intentionally not a semantic support predicate.
    "candidate_profiles": profiles, "provenance_profiles": profiles, "candidate_roles": roles,
    "semantic_policy_rows": (),
    "runtime_env": {
      "PREFILL_GRAPH_GEMM": "1",
      "BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH": str(path),
    },
  }

def promoted_prefill_candidate_supports_profile(profile_id: str) -> bool:
  """Deprecated provenance-only query. A profile can never establish runtime semantic support."""
  if not isinstance(profile_id, str): raise TypeError("profile_id must be a string")
  return False

def promoted_prefill_candidate_policy_rows(inventory: Mapping, capability: Mapping) -> tuple[dict, ...]:
  """Read the legacy promoted artifact and bind it to caller-supplied exact semantic facts."""
  policy = promoted_prefill_candidate_policy()
  artifact = json.loads(pathlib.Path(policy["candidate_set_path"]).read_text())
  return canonical_policy_rows(inventory, capability, artifact, route_id=policy["route_id"])

def apply_route(route_id: str, env: dict | None = None) -> dict:
  """Materialize a research route onto an explicit configuration copy.

  No ambient process configuration is consulted. ``strict_fallback`` routes
  set ``QK_STRICT_FALLBACK=1`` (fail-loud).
  """
  out = dict({} if env is None else env)
  out.update({k: str(v) for k, v in route_env(route_id).items()})
  if route(route_id).get("strict_fallback"): out.setdefault("QK_STRICT_FALLBACK", "1")
  return out

def is_refuted(axis: str) -> bool:
  return any(r["axis"] == axis or axis in r.get("compatibility_aliases", ()) for r in REFUTED)

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
    if r.get("research_only"):
      if r["status"] != "research":
        errors.append(f"{rid}: research_only route cannot claim final status {r['status']!r}")
      if not {"correctness evidence", "resource evidence", "timing evidence"}.issubset(
          {part.strip() for part in str(r.get("authority_gate", "")).split("+")}):
        errors.append(f"{rid}: research_only route must require correctness/resource/timing evidence")
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
