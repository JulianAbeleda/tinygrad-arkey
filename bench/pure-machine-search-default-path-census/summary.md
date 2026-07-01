# PMS-R0 Default-Path Kernel Census

Verdict: **PMS_R0_PASS_CENSUS_PINNED**

Strict default purity: **TINYGRAD_DEFAULT_PURITY_FAIL**

Headline: 5 kernels on the default path are non-tinygrad-generated. 2 are machine-authored/generated (decode_q4k_g3_generated, decode_flash_block_tile_g5_konly); 3 are final-default purity debt (decode_q6k_coop_shipped, decode_attention_owned_two_kernel, prefill_pipe_role_selective_default). Everything else in the model is tinygrad_scheduler-generated.

## Default-path routes

| route_id | workload | provenance | final default? | selector | quant | authority | rollback |
|---|---|---|---|---|---|---|---|
| decode_q4k_g3_generated | decode | machine_authored_generated | yes | BubbleBeam | Q4_K | bench/amd-isa-backend-g3-weight-promotion/latest.json | BUBBLEBEAM_FUTURESIGHT=0 -> decode_q4k_owned_warp |
| decode_q6k_coop_shipped | decode | hand_authored_uop_template | no | hardcoded_default | Q6_K | extra/qk_decode_runtime_overhead.py | none (shipped baseline; Q6K_DIRECT_ROUTE alt is refuted/default-off) |
| decode_attention_owned_two_kernel | decode | external_handwritten_kernel | no | env_guard | fp16 | bench/amd-isa-backend-decode-attention-ceiling/latest.json | DECODE_ATTN_AMDGCN_TILE=0 -> generated tinygrad flash decode |
| decode_flash_block_tile_g5_konly | decode | machine_authored_generated | yes | BoltBeam_route_policy_or_env_default | fp16 | bench/gp-track/gp4_latest.json | DECODE_FLASH_BLOCK_TILE_G5=0 |
| prefill_pipe_role_selective_default | prefill | external_handwritten_kernel | no | manifest | Q4_K,Q6_K,fp16 | bench/qk-prefill-pipe-role-selective/latest.json | PREFILL_PIPE_ROLE_SELECTIVE=0 -> global pipe; PREFILL_GEMM_PIPELINE=0 -> old lds2 |

## Fallback / reference / refuted / research routes (NOT default path)

| route_id | provenance | purity_status | next_action |
|---|---|---|---|
| decode_q4k_owned_warp | rollback_oracle | owned_reference | keep as rollback/oracle; do not delete |
| decode_q6k_direct_refuted | hand_authored_uop_template | refuted | do NOT reopen as built (-5.44% median W==D); only with a different topology than half-warp |
| decode_attention_native_correct_not_fast | machine_authored_generated | research | infrastructure/research only (~60-68% of owned); reopen only if attention wall-share becomes dominant |
| prefill_pipe_global_rollback | rollback_oracle | superseded_rollback | keep as A/B comparator and the rollback target of role-selective |

## Strict-purity debt

- **decode_q6k_coop_shipped**: `hand_authored_uop_template`; replacement scope: docs/tinygrad-pure-search-codegen-audit-and-resolution-20260701.md#tg-p3-generate-q6_k-coop-from-a-route-spec
- **decode_attention_owned_two_kernel**: `external_handwritten_kernel`; replacement scope: docs/tinygrad-pure-search-codegen-audit-and-resolution-20260701.md#tg-p5-replace-owned-decode-attention-with-generated-route
- **prefill_pipe_role_selective_default**: `external_handwritten_kernel`; replacement scope: docs/tinygrad-pure-search-codegen-audit-and-resolution-20260701.md#tg-p4-generate-prefill-gemm-schedule

## Route attribution (cited guards)

- **decode_q4k_g3_generated** (default): tinygrad/llm/model.py:255 getenv('BUBBLEBEAM_FUTURESIGHT', 1)==1 (default-on) + :262 DECODE_Q4K_G3_ANYSHAPE default-on -> :264-299 q4k_g3_lanemap_gemv_kernel fires FIRST for eligible shapes, short-circuiting the owned-warp guards
- **decode_q4k_owned_warp** (fallback): tinygrad/llm/model.py:318 getenv('Q4K_GEMV_WARP_PROJ', 1) (q/o) + :360 getenv('Q4K_GEMV_WARP', 1) (gate/up+down). Guards still default 1 but the G3 branch intercepts first on the default path.
- **decode_q6k_coop_shipped** (default): tinygrad/llm/model.py:500 Q6K_COOP_RT + :501-508 Q6K_LM_HEAD_COOP/Q6K_FFN_DOWN_COOP/DECODE_Q6K_FFN_DOWN_LONGK default-on -> :510-518 q6k_coop_partial_kernel or q6k_gemv_partial_kernel
- **decode_q6k_direct_refuted** (fallback): tinygrad/llm/model.py:455-464 getenv('Q6K_DIRECT_ROUTE') (default-off)
- **decode_attention_owned_two_kernel** (default): tinygrad/llm/model.py:1091 getenv('DECODE_ATTN_AMDGCN_TILE', 1) & ctx>=512 -> :1094-1106 amdgcn_flash_decode
- **decode_flash_block_tile_g5_konly** (default): tinygrad/llm/model.py:1129-1140 QK_ROUTE_POLICY selected_route=decode_flash_block_tile_g5_konly if present, else DECODE_FLASH_BLOCK_TILE_G5 default 1; DECODE_FLASH_BLOCK_TILE_G5_KONLY default 1
- **decode_attention_native_correct_not_fast** (fallback): tinygrad/llm/model.py:1076-1085 DECODE_ATTN_GENERATED_WHOLECACHE generated route, selected when DECODE_ATTN_AMDGCN_TILE=0
- **prefill_pipe_role_selective_default** (default): extra/qk_prefill_graph_gemm_route.py:55 PREFILL_GEMM_PIPELINE default 1 + :61 PREFILL_PIPE_ROLE_SELECTIVE default 1 (out_f==12288 -> pipe_mode=False); entry tinygrad/llm/model.py:145-147 route_pf16_graph_gemm
- **prefill_pipe_global_rollback** (fallback): extra/qk_prefill_graph_gemm_route.py:55-69 (pipe on for all roles when role-selective off)

## tinygrad-scheduler coverage

Writer `tinygrad_generated` covers: rmsnorm, rope/position, q/k/v + o residual elementwise, kv cache write path, short-context attention (ctx<512), all graph ops not in the hot-kernel rows above.
