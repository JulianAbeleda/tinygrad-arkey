# PMS-R0 Default-Path Kernel Census

Verdict: **PMS_R0_BLOCKED_ROUTE_ATTRIBUTION_MISSING**

Strict default purity: **TINYGRAD_DEFAULT_PURITY_PASS**

Headline: 6 kernels on the default path are non-tinygrad-generated. 6 are machine-authored/generated (decode_q4k_g3_generated, decode_q6k_coop_generated, decode_flash_block_tile_g5_konly, prefill_flash_attention_generated, prefill_q4k_direct_tile4x4_default, prefill_q6k_direct_generated); 0 are final-default purity debt. Everything else in the model is tinygrad_scheduler-generated.

## Default-path routes

| route_id | workload | provenance | final default? | selector | quant | authority | rollback |
|---|---|---|---|---|---|---|---|
| decode_q4k_g3_generated | decode | machine_authored_generated | yes | env_guard | Q4_K | bench/amd-isa-backend-g3-weight-promotion/latest.json | BUBBLEBEAM_FUTURESIGHT=0 -> ordinary tinygrad graph; no manifest hand-kernel rollback remains |
| decode_q6k_coop_generated | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | Q6_K | bench/tg-p3-q6k-generated-coop/latest.json | DECODE_Q6K_GENERATED=0 no longer selects a manifest hand-kernel rollback; generated Q6_K decode is the only manifest kernel route |
| decode_flash_block_tile_g5_konly | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | fp16 | bench/gp-track/gp4_latest.json | DECODE_LIVE_SPLIT=0 exits the live-split default; no manifest fallback route row remains |
| prefill_flash_attention_generated | prefill | machine_authored_generated | yes | shape_admitted_model_config_default | fp16 | extra/qk/prefill_flash_e2e_parity.py | none; automatic non-admission (shape outside ADMITTED_GRIDS, non-AMD backend, or non-gfx1100 arch) falls to ordinary SDPA -- no manifest hand-kernel rollback exists or is needed |
| prefill_wmma_lds_dbuf_generated | prefill | tinygrad_scheduler_generated | yes | promoted_candidate_set | fp16 | bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-v1/whole-model-quality.json + whole-prefill-pinned.json | none; absent exact binding selects the ordinary scheduler fallback |
| prefill_q4k_direct_tile4x4_default | prefill | machine_authored_generated | yes | env_default | Q4_K | docs/prefill-lessons-ledger.md | PREFILL_Q4K_DIRECT_SCHEDULE=legacy |
| prefill_q6k_direct_generated | prefill | machine_authored_generated | yes | env_default | Q6_K | test/unit/test_q6k_prefill_route_spec.py + test/unit/test_llm_prefill_routes.py | PREFILL_Q6K_PACKED_LOAD=0 reaches the legacy non-packed debug path; no manifest default rollback remains |

## Fallback / reference / refuted / research routes (NOT default path)

| route_id | provenance | purity_status | next_action |
|---|---|---|---|
| decode_flash_live_split_g4_8b_kvboth | machine_authored_generated | search_generated_promoted | keep promoted; no handwritten attention kernel on the hot path |
| prefill_v2_scheduler_matmul_default | tinygrad_scheduler_generated | research | retain as a pure fallback for unsupported or memory-inadmissible shapes |

## Strict-purity debt


## Route attribution (cited guards)

- **decode_q4k_g3_generated** (default): tinygrad/llm/decode_routes.py q4k_primitive_linear_call getenv('BUBBLEBEAM_FUTURESIGHT', 1)==1 (default-on) + _qk_route_policy_selects_q4k_g3 (BoltBeam QK_ROUTE_POLICY) + DECODE_Q4K_G3_ANYSHAPE default-on -> q4k_g3_lanemap_gemv_kernel fires FIRST for eligible shapes, short-circuiting the owned-warp guards; strict policy fails loud on hidden fallback
- **decode_q6k_coop_generated** (default): tinygrad/llm/decode_routes.py q6k_primitive_linear_call generated branch: getenv('DECODE_Q6K_GENERATED', 1) or QK_ROUTE_POLICY decode_q6k_coop_generated -> emit_q6k_gemv_kernel(spec) fires the coop/partial route; shipped hand kernels short-circuited
- **decode_flash_live_split_g4_8b_kvboth** (fallback): tinygrad/llm/decode_routes.py attention live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0): default-on DECODE_LIVE_SPLIT=1 -> FlashDecodeAttentionSpec live-split block tile + fused combine
- **decode_flash_block_tile_g5_konly** (default): tinygrad/llm/decode_routes.py attention live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0; covers 14B Hq=40/G=5): QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_LIVE_SPLIT default 1; FlashDecodeAttentionSpec owns staging/geometry/combine
- **prefill_flash_attention_generated** (default): tinygrad/llm/model.py Transformer._attention prefill_custom_kernel_attn branch (its own independent eligibility boundary, decoupled from the legacy prefill_tc_attn/shared_attention_proven_eligible proof): _should_use_custom_kernel_prefill_attn(n_heads, n_kv_heads, backend, arch) -> ADMITTED_GRIDS + AMD + gfx1100, threaded into TransformerConfig at model construction (P5b) -> route_prefill_attention(..., use_custom_kernel=True)
- **prefill_wmma_lds_dbuf_generated** (default): selected model inventory + scanned target + memory admission produce exact per-linear bindings
- **prefill_v2_scheduler_matmul_default** (fallback): tinygrad/llm/prefill_routes.py fallback path when no exact generated binding is attached
- **prefill_q4k_direct_tile4x4_default** (default): tinygrad/llm/prefill_routes.py Q4_K direct-packed default -> Q4KPrefillRouteSpec + emit_q4k_packed_prefill_kernel; _direct_packed_opts selects LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4
- **prefill_q6k_direct_generated** (default): tinygrad/llm/prefill_routes.py Q6_K direct-packed branch: PREFILL_Q6K_PACKED_LOAD default-on -> Q6KPrefillRouteSpec + emit_q6k_packed_prefill_kernel; direct_out for parts==1/PREFILL_DIRECT_OUT=1, otherwise partials

## tinygrad-scheduler coverage

Writer `tinygrad_generated` covers: rmsnorm, rope/position, q/k/v + o residual elementwise, kv cache write path, short-context attention (ctx<512), all graph ops not in the hot-kernel rows above.
