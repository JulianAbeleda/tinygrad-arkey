# PMS-R0 Default-Path Kernel Census

Verdict: **PMS_R0_PASS_CENSUS_PINNED**

Strict default purity: **TINYGRAD_DEFAULT_PURITY_FAIL**

Headline: 7 kernels on the default path are non-tinygrad-generated. 4 are machine-authored/generated (decode_q4k_g3_generated, decode_q6k_coop_generated, prefill_q4k_direct_tile4x4_default, prefill_q6k_direct_generated); 3 are final-default purity debt (decode_flash_live_split_g4_8b_kvboth, decode_flash_block_tile_g5_konly, prefill_pipe_role_selective_generated). Everything else in the model is tinygrad_scheduler-generated.

## Default-path routes

| route_id | workload | provenance | final default? | selector | quant | authority | rollback |
|---|---|---|---|---|---|---|---|
| decode_q4k_g3_generated | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | Q4_K | bench/amd-isa-backend-g3-weight-promotion/latest.json | BUBBLEBEAM_FUTURESIGHT=0 -> ordinary tinygrad graph; no manifest hand-kernel rollback remains |
| decode_q6k_coop_generated | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | Q6_K | bench/tg-p3-q6k-generated-coop/latest.json | DECODE_Q6K_GENERATED=0 no longer selects a manifest hand-kernel rollback; generated Q6_K decode is the only manifest kernel route |
| decode_flash_live_split_g4_8b_kvboth | decode | hand_authored_uop_template | no | BoltBeam_route_policy_or_env_default | fp16 | bench/tg-p14-amd-recovery-and-pure-attention-landing/phase2_final_result.json | DECODE_LIVE_SPLIT=0 exits the live-split default; no manifest fallback route row remains |
| decode_flash_block_tile_g5_konly | decode | hand_authored_uop_template | no | BoltBeam_route_policy_or_env_default | fp16 | bench/gp-track/gp4_latest.json | DECODE_LIVE_SPLIT=0 exits the live-split default; no manifest fallback route row remains |
| prefill_pipe_role_selective_generated | prefill | external_handwritten_kernel | no | BoltBeam_route_policy_or_env_default | Q4_K,Q6_K,fp16 | bench/tg-p4-prefill-generated-schedule/latest.json | none; legacy fixed emit removed |
| prefill_q4k_direct_tile4x4_default | prefill | machine_authored_generated | yes | env_default | Q4_K | docs/prefill-packed-generated-tile-scope-20260704.md | PREFILL_Q4K_DIRECT_SCHEDULE=legacy |
| prefill_q6k_direct_generated | prefill | machine_authored_generated | yes | env_default | Q6_K | test/unit/test_q6k_prefill_route_spec.py + test/unit/test_llm_prefill_routes.py | PREFILL_Q6K_PACKED_LOAD=0 reaches the legacy non-packed debug path; no manifest default rollback remains |

## Fallback / reference / refuted / research routes (NOT default path)

| route_id | provenance | purity_status | next_action |
|---|---|---|---|
| prefill_pipe_global_rollback | rollback_oracle | superseded_rollback | keep as A/B comparator and the rollback target of role-selective |

## Strict-purity debt

- **decode_flash_live_split_g4_8b_kvboth**: `hand_authored_uop_template`; replacement scope: Attention descriptor conversion: FlashDecodeTileSpec + LiveSplitGeometrySpec + FlashCombineSpec own topology, shared emitter lowers to codegen, generated-only binding gate. Until then the executing flash/live-split kernels are hand-authored Tensor.custom_kernel UOp templates, not ordinary scheduler output.
- **decode_flash_block_tile_g5_konly**: `hand_authored_uop_template`; replacement scope: Attention descriptor conversion (shared with 8B live-split): FlashDecodeTileSpec + LiveSplitGeometrySpec + FlashCombineSpec + generated-only binding gate. Executing block-tile/live-split kernels are hand-authored UOp templates until then.
- **prefill_pipe_role_selective_generated**: `external_handwritten_kernel`; replacement scope: Route B: generated LDS+WMMA codegen substrate (PrefillWMMAScheduleSpec) replacing extra/qk/prefill/wmma.py raw Ops.INS. Schedule SELECTION is spec-generated, but the executing substrate wraps raw RDNA3 instruction lists -> external handwritten kernel under the strict rule.

## Route attribution (cited guards)

- **decode_q4k_g3_generated** (default): tinygrad/llm/decode_routes.py q4k_primitive_linear_call getenv('BUBBLEBEAM_FUTURESIGHT', 1)==1 (default-on) + _qk_route_policy_selects_q4k_g3 (BoltBeam QK_ROUTE_POLICY) + DECODE_Q4K_G3_ANYSHAPE default-on -> q4k_g3_lanemap_gemv_kernel fires FIRST for eligible shapes, short-circuiting the owned-warp guards; strict policy fails loud on hidden fallback
- **decode_q6k_coop_generated** (default): tinygrad/llm/decode_routes.py q6k_primitive_linear_call generated branch: getenv('DECODE_Q6K_GENERATED', 1) or QK_ROUTE_POLICY decode_q6k_coop_generated -> emit_q6k_gemv_kernel(spec) fires the coop/partial route; shipped hand kernels short-circuited
- **decode_flash_live_split_g4_8b_kvboth** (default): tinygrad/llm/decode_routes.py attention live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0): default-on DECODE_LIVE_SPLIT=1 -> live-split block tile path (staging='KV_BOTH', fused_combine=True)
- **decode_flash_block_tile_g5_konly** (default): tinygrad/llm/decode_routes.py attention live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0; covers 14B Hq=40/G=5): QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_LIVE_SPLIT default 1; staging KV_BOTH (K_ONLY unsupported under live-split geometry)
- **prefill_pipe_role_selective_generated** (default): extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm -> describe_prefill_schedule + emit_prefill_gemm_from_spec
- **prefill_q4k_direct_tile4x4_default** (default): tinygrad/llm/prefill_routes.py Q4_K direct-packed default -> Q4KPrefillRouteSpec + emit_q4k_packed_prefill_kernel; _direct_packed_opts selects LOCAL:0:16, LOCAL:1:16, UPCAST:0:4, UPCAST:1:4
- **prefill_q6k_direct_generated** (default): tinygrad/llm/prefill_routes.py Q6_K direct-packed branch: PREFILL_Q6K_PACKED_LOAD default-on -> Q6KPrefillRouteSpec + emit_q6k_packed_prefill_kernel; direct_out for parts==1/PREFILL_DIRECT_OUT=1, otherwise partials
- **prefill_pipe_global_rollback** (fallback): extra/qk/prefill_graph_gemm_route.py:55-69 (pipe on for all roles when role-selective off)

## tinygrad-scheduler coverage

Writer `tinygrad_generated` covers: rmsnorm, rope/position, q/k/v + o residual elementwise, kv cache write path, short-context attention (ctx<512), all graph ops not in the hot-kernel rows above.
