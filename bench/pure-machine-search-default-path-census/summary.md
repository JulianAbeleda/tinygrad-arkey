# PMS-R0 Default-Path Kernel Census

Verdict: **PMS_R0_PASS_CENSUS_PINNED**

Strict default purity: **TINYGRAD_DEFAULT_PURITY_PASS**

Headline: 5 kernels on the default path are non-tinygrad-generated. 5 are machine-authored/generated (decode_q4k_g3_generated, decode_q6k_coop_generated, decode_flash_live_split_g4_8b_kvboth, decode_flash_block_tile_g5_konly, prefill_pipe_role_selective_generated); 0 are final-default purity debt (). Everything else in the model is tinygrad_scheduler-generated.

## Default-path routes

| route_id | workload | provenance | final default? | selector | quant | authority | rollback |
|---|---|---|---|---|---|---|---|
| decode_q4k_g3_generated | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | Q4_K | bench/amd-isa-backend-g3-weight-promotion/latest.json | BUBBLEBEAM_FUTURESIGHT=0 -> decode_q4k_owned_warp |
| decode_q6k_coop_generated | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | Q6_K | bench/tg-p3-q6k-generated-coop/latest.json | DECODE_Q6K_GENERATED=0 -> decode_q6k_coop_shipped (hand kernels) |
| decode_flash_live_split_g4_8b_kvboth | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | fp16 | bench/tg-p14-amd-recovery-and-pure-attention-landing/phase2_final_result.json | DECODE_LIVE_SPLIT=0 -> generic generated tinygrad flash decode |
| decode_flash_block_tile_g5_konly | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | fp16 | bench/gp-track/gp4_latest.json | DECODE_LIVE_SPLIT=0 |
| prefill_pipe_role_selective_generated | prefill | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | Q4_K,Q6_K,fp16 | bench/tg-p4-prefill-generated-schedule/latest.json | PREFILL_GENERATED_SCHEDULE=0 -> prefill_pipe_role_selective_default (legacy fixed emit) |

## Fallback / reference / refuted / research routes (NOT default path)

| route_id | provenance | purity_status | next_action |
|---|---|---|---|
| decode_q4k_owned_warp | rollback_oracle | owned_reference | keep as rollback/oracle; do not delete |
| decode_q6k_coop_shipped | rollback_oracle | owned_reference | keep as rollback/oracle; do not delete |
| decode_q6k_direct_refuted | hand_authored_uop_template | refuted | do NOT reopen as built (-5.44% median W==D); only with a different topology than half-warp |
| decode_attention_owned_two_kernel | external_handwritten_kernel | removed | retired; replaced by decode_flash_live_split_g4_8b_kvboth |
| decode_attention_generic_flash_generated | tinygrad_scheduler_generated | research | generated rollback/reference only; promoted live-split KV_BOTH remains the default |
| prefill_pipe_role_selective_default | rollback_oracle | search_selected_specialized_route | keep as rollback/oracle; do not delete |
| prefill_pipe_global_rollback | rollback_oracle | superseded_rollback | keep as A/B comparator and the rollback target of role-selective |

## Strict-purity debt


## Route attribution (cited guards)

- **decode_q4k_g3_generated** (default): tinygrad/llm/decode_routes.py q4k_primitive_linear_call getenv('BUBBLEBEAM_FUTURESIGHT', 1)==1 (default-on) + _qk_route_policy_selects_q4k_g3 (BoltBeam QK_ROUTE_POLICY) + DECODE_Q4K_G3_ANYSHAPE default-on -> q4k_g3_lanemap_gemv_kernel fires FIRST for eligible shapes, short-circuiting the owned-warp guards; strict policy fails loud on hidden fallback
- **decode_q4k_owned_warp** (fallback): tinygrad/llm/decode_routes.py q4k_primitive_linear_call getenv('Q4K_GEMV_WARP_PROJ', 1) (q/o) + getenv('Q4K_GEMV_WARP', 1) (gate/up+down). Guards still default 1 but the G3 branch intercepts first on the default path.
- **decode_q6k_coop_generated** (default): tinygrad/llm/decode_routes.py q6k_primitive_linear_call generated branch: getenv('DECODE_Q6K_GENERATED', 1) or QK_ROUTE_POLICY decode_q6k_coop_generated -> emit_q6k_gemv_kernel(spec) fires the coop/partial route; shipped hand kernels short-circuited
- **decode_q6k_coop_shipped** (fallback): tinygrad/llm/decode_routes.py q6k_primitive_linear_call shipped branch (DECODE_Q6K_GENERATED=0) -> q6k_coop_partial_kernel or q6k_gemv_partial_kernel
- **decode_q6k_direct_refuted** (fallback): tinygrad/llm/decode_routes.py q6k_primitive_linear_call getenv('Q6K_DIRECT_ROUTE') (default-off)
- **decode_attention_owned_two_kernel** (fallback): removed from tinygrad/llm/model.py; no env flag selects this route
- **decode_flash_live_split_g4_8b_kvboth** (default): tinygrad/llm/decode_routes.py flash_decode_attention_route UNIFIED live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0): default-on DECODE_LIVE_SPLIT=1 -> flash_decode_live_split_block_tile(..., staging='KV_BOTH', fused_combine=True)
- **decode_flash_block_tile_g5_konly** (default): tinygrad/llm/decode_routes.py flash_decode_attention_route UNIFIED live-split branch (structural class B=1,Hd=128,Hkv=8,Hq%Hkv==0; covers 14B Hq=40/G=5): QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_LIVE_SPLIT default 1; staging KV_BOTH (K_ONLY unsupported under live-split geometry)
- **decode_attention_generic_flash_generated** (fallback): tinygrad/llm/decode_routes.py flash_decode_attention_route generic flash_decode_attention fallback when DECODE_LIVE_SPLIT=0
- **prefill_pipe_role_selective_generated** (default): extra/qk/prefill_graph_gemm_route.py route_pf16_graph_gemm: getenv('PREFILL_GENERATED_SCHEDULE', 1) or QK_ROUTE_POLICY prefill_pipe_role_selective_generated -> describe_prefill_schedule + emit_prefill_gemm_from_spec
- **prefill_pipe_role_selective_default** (fallback): extra/qk/prefill_graph_gemm_route.py _kernel (legacy fixed emit; reached when PREFILL_GENERATED_SCHEDULE=0)
- **prefill_pipe_global_rollback** (fallback): extra/qk/prefill_graph_gemm_route.py:55-69 (pipe on for all roles when role-selective off)

## tinygrad-scheduler coverage

Writer `tinygrad_generated` covers: rmsnorm, rope/position, q/k/v + o residual elementwise, kv cache write path, short-context attention (ctx<512), all graph ops not in the hot-kernel rows above.
