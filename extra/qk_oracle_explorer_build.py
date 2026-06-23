#!/usr/bin/env python3
"""Build the oracle-guided GPU primitive explorer registries/specs/gates (Phases 1,2,4-7). Pure JSON synthesis from
already-measured artifacts (docs + bench facts); NO measurement, NO search, NO behavior change. See
docs/oracle-guided-gpu-primitive-explorer-scope-20260623.md."""
import json, pathlib
OUT = pathlib.Path("bench/qk-oracle-gpu-primitive-explorer")
OUT.mkdir(parents=True, exist_ok=True)
def w(name, obj): (OUT/name).write_text(json.dumps(obj, indent=2)); print("wrote", name)

# ---------------- Phase 1: Oracle Registry ----------------
oracles = {
  "date": "2026-06-23", "phase": "ORACLE_REGISTRY", "hardware": "gfx1100 (RX 7900 GRE/XT/XTX), 24GB",
  "model": "Qwen3-8B-Q4_K_M", "principle": "an oracle is lifecycle-complete: source+ISA+route+ABI+correctness+authority+fallback+shapes+failure-modes",
  "oracles": [
    {"oracle_id": "decode_whole_cache_owned_tile_8b_gfx1100", "lane": "decode", "primitive_class": "attention/ABI",
     "purpose": "current default decode attention; at/above llama.cpp",
     "in_repo_baseline": "bench/qk-decode-search-readiness/baseline_oracle.json (ledger candidate decode/buffer_identity_whole_cache)",
     "source_artifacts": ["extra/qk_owned_flash_decode.hip", "extra/qk_owned_flash_decode_graph_node.py", "tinygrad/llm/model.py (DECODE_ATTN_KV_IDENTITY branch)"],
     "code_object": "b4_tile_whole_s48_*.co (symbol owned_flash_tile_gqa_whole + owned_flash_combine)",
     "knob": "DECODE_ATTN_KV_IDENTITY=1 (shipped default); policy knobs DECODE_ATTN_AMDGCN_{S=48,COMBINE=base,TILE=1,MIN_CTX=512}",
     "authority_benchmark": {"harness": "extra/qk_decode_search_gate.py::run_wd (clean synced W==D, fixed [[100]] token, PROFILE=0, 8 warmup + 30 rep median)",
       "frozen_oracle_no_warp_tok_s": {"512": 90.6, "1024": 89.3}, "spread_pct": "<=0.5",
       "canonical_warp_on_full_model_tok_s": {"512": 102.9, "1024": 101.3, "2048": 98.7, "4096": 94.2},
       "warp_flag_stack": "Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 Q4K_GEMV_WARP_PROJ=1",
       "vs_llama_pct": "102-105% (at/above parity)", "buffer_identity_win_vs_slice_pct": {"512": 18.7, "1024": 17.4, "2048": 16.3, "4096": 13.3}},
     "expected_correctness": {"greedy_tokens": [315, 24231, 6009, 979, 220, 576], "prompt": "The history of computing began when", "gate": "byte-identical"},
     "expected_route_signature": {"candidate_kernel_present": "owned_flash_tile_gqa_whole", "slice_route_absent": "owned_flash_tile_gqa", "program_node_count": 35},
     "expected_isa": {"vgpr": 60, "sgpr": 26, "scratch_bytes": 0, "spills": 0, "has_vector_dot": True, "has_lds": True,
       "has_cross_lane": True, "has_vector_global_load": True, "has_spill": False, "tag": "AMD_ISA_PRIMITIVE_CONFIRMED"},
     "expected_materialization_abi": {"E_49152_present": False, "full_maxc_copy_kernels": [], "buffer_identity_inputs": True, "principle": "#12 buffer-identity ABI"},
     "supported_shapes": {"model": "Qwen3-8B", "arch": "gfx1100", "head_dim": 128, "max_ctx": 4608, "fires_at_ctx>=": "DECODE_ATTN_AMDGCN_MIN_CTX (512)", "below": "falls to gqa_coop_vec"},
     "fallback": "DECODE_ATTN_KV_IDENTITY=0 -> slice route + E_49152 returns + buffer_identity_inputs=false (reject); DECODE_ATTN_AMDGCN_TILE=0 -> no owned route (route_not_firing)",
     "status": "ORACLE_FROZEN / WON_SHIPPED_DEFAULT_ON", "search_verdict": "DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST",
     "owner_policy": "default-on, owner-authorized 2026-06-23"},

    {"oracle_id": "q4k_gemv_warp_8b_gfx1100", "lane": "decode/codegen", "primitive_class": "GEMM/route_policy",
     "purpose": "lossless Q4_K FFN/proj weight-GEMV schedule (work-decomposition warp primitive)",
     "source_artifacts": ["extra/q4_k_gemv_primitive.py::q4k_gemv_warp_kernel (kernel q4k_gemv_warp_{rows}_{k})"],
     "knob": "Q4K_GEMV_WARP (gate/up), Q4K_GEMV_WARP_DOWN (down), Q4K_GEMV_WARP_PROJ (kv/qkv proj)",
     "default": "flags default-OFF but CANONICAL (several harnesses setdefault=1 as the decode baseline; reproduces 102.9/101.3/98.7/94.2)",
     "mechanism": "1 workgroup/row, 32 threads = 1 wave; 8 coalesced packed words/8 lanes; 4 block_groups K-parallel; per-lane REG FP-accumulate -> warp_reduce_sum (ds_bpermute) -> single store; LOSSLESS FP (no q8/int-dot)",
     "constraints": "lanes==32; k_blocks % 4 == 0; Q4_K only (k/v are Q6_K -> reuse ceiling 2)",
     "authority_benchmark": {"local_ab": "extra/qk_ffn_gemv_warp_ab.py", "in_model_wd": "extra/qk_ffn_gemv_warp_wd.py",
       "gate_rule": "(d1024>=+5% OR d4096>=+7%) AND no ctx512 regress AND tokens match",
       "verdicts": ["Q4K_GEMV_WARP_WD_PASS", "Q4K_GEMV_WARP_LOCAL_PASS_WD_FAIL", "Q4K_GEMV_WARP_BLOCKED_IMPLEMENTATION"]},
     "expected_isa": {"has_cross_lane": True, "ds_bpermute": True, "coalesced_global_load": True, "v_dot2": False, "note": "lossless FP, no int-dot"},
     "status": "shipped weight-GEMV lever; weight-GEMV at/below llama (-1.1 ms/tok @1024) -> 8B-speed lane CLOSED; framework flags it 'cross-shape only'"},

    {"oracle_id": "prefill_graph_gemm_8b_gfx1100", "lane": "prefill", "primitive_class": "GEMM",
     "purpose": "current prefill default after kv_proj BN64 fix (dependency-free hand-asm LDS GEMM)",
     "source_artifacts": ["extra/ build_gemm_lds2 (PREFILL_GRAPH_GEMM)", "tinygrad/llm/model.py (PREFILL_V2 path)"],
     "knob": "PREFILL_GRAPH_GEMM default-ON within PREFILL_V2; PREFILL_V2 itself default-OFF (VRAM-auto, owner-decided)",
     "authority_benchmark": {"harness": "extra/qk_prefill_whole_synced.py (whole multi-chunk SYNCED prefill = sum of synced concrete-chunk times; NOT nosync, NOT single chunk)",
       "post_kvproj_fix_tok_s": {"512": 3554, "1024": 3468, "2048": 3221, "4096": 2796}, "vs_tensile_pct": "~99.5%", "vs_llama_pct": "~91-116% (llama ~3020-3070)",
       "RETRACTED_stale": "the result-doc headline 1983 tok/s (~66% llama) is a stale/older/nosync number, superseded by the kv_proj BN64 fix"},
     "expected_correctness": "rel_err ~2.08e-4 vs reference (parity-class; no byte-identical greedy claim for whole-prefill)",
     "supported_shapes": {"primary_tuned": "M=512,N=12288,K=4096 (ffn gate/up)", "other_roles_tile_divisible": "ffn_down N4096/K12288, qo_proj 4096^2, kv_proj N1024/K4096, lm_head N>50000", "attention": "flash path, no GEMM kernel either route"},
     "status": "PREFILL_FRONTIER_AUDIT_COMPLETE / PREFILL_TENSILE_GAP_ATTRIBUTED; kernel SOLVED, GPU-level parity-to-+10% vs vendored Tensile; tuning knobs exhausted"},

    {"oracle_id": "owned_attention_isa_template", "lane": "codegen", "primitive_class": "codegen_microprimitive",
     "purpose": "ISA/resource REFERENCE for a healthy owned attention tile (v_dot2 + LDS + cross-lane + vector-load, no spill)",
     "producing_tool": "extra/qk_amdgpu_isa_primitive_audit.py (normalized via extra/qk_isa_primitive_audit.py -> bench/qk-isa-primitive-audit/owned_decode_attention.json)",
     "reference_isa": {"vgpr": "60 (range 56-64)", "scratch_bytes": 0, "spills": 0, "lds_bytes": 8192,
       "has_vector_dot": True, "has_lds": True, "has_cross_lane": True, "has_vector_global_load": True, "has_spill": False,
       "dtype": "fp16 cache/Q/K/V, fp32 accumulation (online-softmax m,l + fp32 partials)",
       "launch": "wave32, 4 warps, split-KV over S=48 workgroups, grid (Hkv,S,1) block (128,1,1), Hkv=8 baked, G=4, TK=16 LDS-staged"},
     "status": "ISA template / acceptance envelope for decode-attention candidates (drives the gate ISA-reject rules)"},

    {"oracle_id": "tensile_prefill_reference", "lane": "prefill/codegen", "primitive_class": "GEMM",
     "purpose": "external vendor GEMM reference for the prefill gap",
     "availability": "AVAILABLE_VIA_FLAG (PREFILL_TENSILE_GEMM=1, research-only route via extra/qk_tensile_inmodel.py route_pf16)",
     "in_repo_co_artifact": "UNVERIFIED — no tensile *.co found in tree; Tensile SOURCE dirs exist under ~/rocm-libraries-tensile-sparse, ~/rocm-tensile-legacy-sparse",
     "facts": {"macro_tile": "MT128x128x16, DepthU=16", "vgpr": 256, "lds_bytes": 25088, "tflops": "~66", "tok_s": "~2673 (~87% llama)", "correctness": "byte-identical rel_err 0"},
     "status": "research comparison baseline only; NOT a ship/default path; .co path must be verified before treating as a materialized in-repo oracle"},
  ],
  "verdict": "ORACLE_REGISTRY_READY",
}
w("oracles.json", oracles)

# ---------------- Phase 2: Search Spec Schema + 4 examples ----------------
schema = {
  "date": "2026-06-23", "phase": "SEARCH_SPEC_SCHEMA",
  "design": "REUSE the existing two-layer SSOT, do NOT invent a third. Spec layer = extra/qk_search_spec.py "
            "(SearchRow/Constraints/AcceptedPolicy); result/memory layer = bench/qk-project-search-ledger schema "
            "(15-field entry). This schema documents the unified spec a generic runner consumes.",
  "spec_layer_reused": {"module": "extra/qk_search_spec.py",
    "SearchRow": ["row_id(->id)", "phase", "model", "op_scope", "backend", "search_space", "objective", "constraints"],
    "enums": {"Phase": ["decode", "long_context_decode", "prefill"], "Model": ["qwen3_8b", "qwen3_14b", "qwen3_32b"],
      "OpScope": ["q4k_gemv", "q6k_gemv", "attention", "ffn_down", "lm_head", "scheduler"],
      "SearchSpace": ["primitive_policy", "demotion", "flash_threshold", "flash_variant", "storage", "schedule", "lds_blocking"],
      "Objective": ["tok_s", "hbm_pct", "serving_latency"], "BACKENDS": ["AMD"]},
    "Constraints": {"exact_required": True, "dnll_epsilon": 0.0, "max_storage_mb": None, "ctx_range": [1, 4096], "no_beam_remote": True},
    "AcceptedPolicy": ["model", "phase", "backend", "ctx_range", "objective", "baseline_tok_s", "accepted_tok_s", "quality_gate", "exactness", "commit", "memory_cap_mb", "hardware"]},
  "explorer_spec_fields": ["search_id", "lane", "oracle_id", "candidate_generator", "knobs_ranges", "structural_gates",
    "route_lifecycle_gates", "materialization_abi_gates", "isa_resource_gates", "correctness_gates", "authority_benchmark",
    "budget", "stop_rules", "result_schema_ref"],
  "lane_authority": {"decode": "clean synced W==D", "prefill": "clean synced whole-prefill (qk_prefill_whole_synced)",
    "native-codegen-microprimitive": "local correctness (rel_rmse<=1e-2) + ISA/resource target; NO W==D claim (non-promotion)",
    "cross-shape": "target-specific decode/prefill authority", "small-op-fusion": "W==D or whole-prefill after one manual fusion gate"},
  "gate_order": ["schema/structural", "route/lifecycle", "materialization/ABI", "ISA/resource", "correctness", "local diagnostic (optional)", "authority benchmark"],
  "result_memory_layer": {"ledger_file": "bench/qk-project-search-ledger/ledger.jsonl",
    "entry_fields_15": ["candidate_id", "lane", "primitive_class", "knobs", "oracle", "correctness", "route_identity",
      "materialization_abi", "isa", "local_diagnostic", "authority_benchmark", "verdict", "stop_reason", "artifact_links", "learned_rule"],
    "api": "extra/qk_project_search_ledger.py::entry(**kw)+validate(e); append json.dumps(e)+'\\n'"},
  "verdict": "SEARCH_SPEC_SCHEMA_READY",
}
w("search_spec_schema.json", schema)

w("spec_decode_policy_example.json", {
  "search_id": "decode/policy/owned_tile_S_combine_minctx", "lane": "decode",
  "oracle_id": "decode_whole_cache_owned_tile_8b_gfx1100", "candidate_generator": "extra/qk_decode_search_execute.py (Mode A)",
  "knobs_ranges": {"DECODE_ATTN_AMDGCN_S": [32, 48, 64, 96], "DECODE_ATTN_AMDGCN_COMBINE": ["base", "hd64"], "DECODE_ATTN_AMDGCN_MIN_CTX": [512, 1024]},
  "structural_gates": ["candidate={id,env}", "generated_code_objects=false"],
  "route_lifecycle_gates": ["candidate_kernel_present owned_flash_tile_gqa_whole", "slice_route_absent"],
  "materialization_abi_gates": ["E_49152_present=false", "buffer_identity_inputs=true"],
  "isa_resource_gates": ["VGPR<=96", "no spill/scratch", "keep v_dot2+lds+cross_lane"],
  "correctness_gates": ["byte-identical tokens [315,24231,6009,979,220,576]"],
  "authority_benchmark": "clean synced W==D @ctx 512/1024 (qk_decode_search_gate.run_wd)",
  "budget": "6 candidates", "stop_rules": ["first-fail short-circuit", "oracle recheck within 3% band else SEARCH_ORACLE_DRIFT_STOP", "winner needs delta@1024 > max(spread,1.0)%"],
  "result_schema_ref": "bench/qk-project-search-ledger ledger.jsonl (lane=decode)",
  "last_run_verdict": "DECODE_SEARCH_EXECUTED_ORACLE_REMAINS_BEST"})

w("spec_native_codegen_micro_example.json", {
  "search_id": "codegen/microprimitive/isa_expressibility", "lane": "native-codegen-microprimitive",
  "oracle_id": "owned_attention_isa_template", "candidate_generator": "extra/qk_native_codegen_microsearch.py",
  "knobs_ranges": {"expression": ["cross_lane_n32", "cross_lane_n64", "fp16_dot", "lds_reduce", "vector_load"]},
  "structural_gates": ["compile capture each emitted .co"],
  "route_lifecycle_gates": ["n/a (microprimitive, not in-model route)"],
  "materialization_abi_gates": ["n/a"],
  "isa_resource_gates": ["target flag present per AMDGCN audit: has_v_dot2 / has_cross_lane / has_lds / has_vector_global_load; no_spill"],
  "correctness_gates": ["rel_rmse vs numpy <= 1e-2"],
  "authority_benchmark": "LOCAL correctness + ISA target ONLY — NEVER W==D (non-promotion lane; cannot flip a default)",
  "budget": "5 expressions", "stop_rules": ["record target_present per candidate; no promotion"],
  "result_schema_ref": "bench/qk-project-search-ledger ledger.jsonl (lane=codegen)",
  "last_run_verdict": "NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND (2/4 native: LDS+vector-load; v_dot2 + cross-lane(ds_bpermute) are renderer GAPS)"})

w("spec_prefill_placeholder.json", {
  "search_id": "prefill/role_policy/PLACEHOLDER", "lane": "prefill", "oracle_id": "prefill_graph_gemm_8b_gfx1100",
  "candidate_generator": "NOT BUILT (gated)", "status": "PLACEHOLDER",
  "knobs_ranges": {"BN_waves_n_by_role": "TBD", "tile_sizes": "TBD", "BK": "TBD", "LDS_layout": "TBD", "prefetch_unroll": "TBD"},
  "authority_benchmark": "clean synced whole-prefill (qk_prefill_whole_synced)",
  "gate": "DISABLED — prefill search gated OFF at rest (PREFILL_SEARCH_GATED_OFF_AT_REST). Reopen only if per-role synced "
          "in-model attribution shows a material, kernel-searchable residual. Next step is the NON-search in-model "
          "integration-penalty fix + per-role attribution, not a GEMM search.",
  "result_schema_ref": "bench/qk-project-search-ledger ledger.jsonl (lane=prefill)"})

w("spec_cross_shape_placeholder.json", {
  "search_id": "cross-shape/route_policy/PLACEHOLDER", "lane": "cross-shape", "oracle_id": "TBD (target-specific)",
  "candidate_generator": "NOT BUILT (gated)", "status": "PLACEHOLDER",
  "knobs_ranges": {"model_shape_guards": "TBD", "route_thresholds": "TBD", "split_count": "TBD", "role_specific_policy": "TBD"},
  "authority_benchmark": "target-specific decode/prefill authority",
  "gate": "DISABLED — CROSS_SHAPE_SEARCH_NEEDS_TARGETS. Requires either an alternate GPU vendor backend with ISA tooling "
          "(NVIDIA cuobjdump/nvdisasm reading SASS; Intel ocloc/iga reading Xe ISA) or an alternate model shape. "
          "Single gfx1100 box; NVIDIA/Intel tooling absent; no 14B/32B until owner asks.",
  "result_schema_ref": "bench/qk-project-search-ledger ledger.jsonl (lane=cross-shape)"})

# ---------------- Phase 4: Decode backend integration ----------------
w("decode_backend_integration.json", {
  "date": "2026-06-23", "phase": "DECODE_SEARCH_BACKEND",
  "existing_backend": {"runner": "extra/qk_decode_search_runner.py (run_candidate, freeze_oracle, ORACLE_FILE)",
    "gate": "extra/qk_decode_search_gate.py::evaluate (cost-ordered: route-fire -> E_49152 -> buffer-identity -> tokens -> ISA -> W==D; +ctx512 regression)",
    "executors": {"mode_a_policy": "extra/qk_decode_search_execute.py", "mode_b_generated_tile": "extra/qk_decode_mode_b_execute.py (QK_CAND_KERNEL, generated_code_objects=true)"},
    "candidate_shape": "{'id': str, 'env': {knob: value}}", "ledger": "writes bench/qk-project-search-ledger + bench/qk-decode-machine-search/*"},
  "integration_status": "functional and ledger-wired for the DECODE lane; harness-contract CONFORMS 13/13",
  "adapter_gaps": ["runner hard-wired to decode attention-tile (fixed expected kernel symbol, fixed correctness prompt/tokens, fixed W==D ctxs, hardcoded ISA envelope VGPR<=96 + v_dot2/lds/cross-lane)",
    "does NOT consume qk_search_spec.SearchRow rows — grids are inline python literals",
    "needs a SearchRow->{env-knob dict, expected-kernel symbol, oracle file, reject envelope} adapter + a per-lane gate registry to drive from a generic spec"],
  "verdict": "DECODE_SEARCH_BACKEND_NEEDS_ADAPTER",
  "note": "INTEGRATED for decode use today; NEEDS_ADAPTER only to be driven by the generic spec-driven runner."})

# ---------------- Phase 5: Native-codegen backend integration ----------------
w("native_codegen_backend_integration.json", {
  "date": "2026-06-23", "phase": "NATIVE_CODEGEN_SEARCH_BACKEND",
  "tool": "extra/qk_native_codegen_microsearch.py (DEV=AMD PYTHONPATH=. .venv/bin/python ...)",
  "what_it_does": "grid of 5 bounded tinygrad expressions; compile-capture each .co; local numpy correctness; AMDGCN ISA-audit each kernel; OR target flag",
  "target_isa_facts": {"cross_lane_n32/n64": "has_cross_lane (ds_bpermute)", "fp16_dot": "has_v_dot2", "lds_reduce": "has_lds", "vector_load": "has_vector_global_load", "spill": "no_spill (max_scratch==0)"},
  "correctness_scorer": "rel_rmse = sqrt(mean((out-ref)^2))/(sqrt(mean(ref^2))+1e-9); pass <= 1e-2 (all passed at 1e-3)",
  "isa_scorer": "per-.co flags OR'd; target_present = flags[candidate.target]",
  "result": "NATIVE_CODEGEN_MICROSEARCH_EXECUTED_TARGET_FOUND — 2/4 native (LDS staging + vector global loads emittable); v_dot2 (fused fp16 dot) and cross-lane reduce (ds_bpermute) CONFIRMED RENDERER GAPS",
  "ledger_mapping": "5 entries lane=codegen in bench/qk-project-search-ledger/ledger.jsonl",
  "promotion_authority_rule": "ISA evidence + local correctness ONLY, NEVER W==D; this lane CANNOT promote a decode/prefill default",
  "verdict": "NATIVE_CODEGEN_SEARCH_BACKEND_INTEGRATED"})

# ---------------- Phase 6: Prefill search gate ----------------
w("prefill_search_gate.json", {
  "date": "2026-06-23", "phase": "PREFILL_SEARCH_GATE",
  "rule": "prefill search allowed only if a role-specific residual is material AND kernel-searchable",
  "evidence": {"kernel_vs_tensile": "~99.5% (parity); tuning knobs (BK/PAD/PLR/occupancy/WGM) EXHAUSTED",
    "transfer": "isolated kernel parity (63-78 TFLOPS) does NOT transfer; in-model gate/up ~22 TFLOPS; dominant lever = in-model integration penalty, NOT the kernel",
    "micro_residual": "+23% VALU address arithmetic vs Tensile (8.66M vs 7.04M, PMC-exact) = deterministic addressing fix, not a search knob",
    "criteria_failed": "4 of 6 (not kernel-bottlenecked; no local->whole transfer; knobs exhausted; expected search gain ~0)"},
  "caveat": "fresh synced per-role IN-MODEL breakdown was DEFERRED (prefill_v2 path intricate) -> per-role attribution is the first NON-search step before any in-model-penalty fix",
  "verdict": "PREFILL_SEARCH_GATED_OFF_AT_REST",
  "next_action": "NON-search: per-role synced in-model attribution, then the in-model integration-penalty/addressing fix (PREFILL_NEEDS_NONSEARCH_FIX_FIRST)"})

# ---------------- Phase 7: Cross-shape search gate ----------------
w("cross_shape_search_gate.json", {
  "date": "2026-06-23", "phase": "CROSS_SHAPE_SEARCH_GATE",
  "target_selection_requirements": ["model available", "baseline oracle available", "correctness harness", "route eligibility", "expected cost"],
  "availability_now": {"gpu": "single gfx1100 (AMD_ISA_AUDIT_READY)", "nvidia_tooling": "ABSENT (NVIDIA_ISA_AUDIT_BACKEND_SCOPED — cuobjdump/nvdisasm/SASS not buildable here)",
    "intel_tooling": "ABSENT (INTEL_ISA_AUDIT_BACKEND_SCOPED — ocloc/iga/Xe ISA not buildable here)",
    "alt_model": "none present; no pivot to 14B/32B until owner asks"},
  "verdict": "CROSS_SHAPE_SEARCH_NEEDS_TARGETS",
  "note": "ready_after_target_selection only once a vendor backend (NVIDIA/Intel ISA tooling) or an alternate model shape is provided"})

print("ALL_EXPLORER_ARTIFACTS_WRITTEN")
