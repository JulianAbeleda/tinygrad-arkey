# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0`
- dataset: `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/prompts-holdout.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `38`
- generated tokens: `1037`
- quality: `fail` (0/38)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `semantic_schedule_v0:qwen3-14b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,direct_output,tie` | `fail` | 28 | 0.83 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,tie` | `fail` | 19 | 3.46 | <think>\n\n</think>\n\n{"label":"accept","reason":"accepted_runtime_path","retry":false} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.21 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 28 | 4.47 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:005-ffn-down-blk-5-ffn-down-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,reject` | `fail` | 28 | 5.57 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:006-ffn-down-blk-5-ffn-down-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.48 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:007-ffn-down-blk-5-ffn-down-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 28 | 4.46 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:008-attn-q-blk-0-attn-q-weight-direct-out` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,direct_output,tie` | `fail` | 28 | 5.23 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:009-attn-q-blk-0-attn-q-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,raw_accept_unconfirmed` | `fail` | 28 | 5.03 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.60 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 28 | 5.58 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.57 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.43 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-14b-q4-k-m:014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 19 | 3.87 | <think>\n\n</think>\n\n{"label":"accept","reason":"accepted_runtime_path","retry":false} |
| `semantic_schedule_v0_full_decode:qwen3-14b-q4-k-m:009-attn-q-blk-0-attn-q-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,reject` | `fail` | 28 | 7.54 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_ready",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-direct-out` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,direct_output,tie` | `fail` | 28 | 5.06 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:002-ffn-gate-blk-0-ffn-gate-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,reject` | `fail` | 28 | 4.96 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:003-ffn-gate-blk-0-ffn-gate-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 19 | 3.86 | <think>\n\n</think>\n\n{"label":"accept","reason":"accepted_runtime_path","retry":false} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:004-ffn-gate-blk-0-ffn-gate-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 28 | 5.47 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:005-ffn-down-blk-4-ffn-down-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,reject` | `fail` | 28 | 5.63 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:006-ffn-down-blk-4-ffn-down-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.59 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:007-ffn-down-blk-4-ffn-down-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,reject` | `fail` | 28 | 5.57 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:008-attn-q-blk-0-attn-q-weight-direct-out` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,direct_output,raw_accept_unconfirmed` | `fail` | 28 | 5.29 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:009-attn-q-blk-0-attn-q-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,raw_accept_unconfirmed` | `fail` | 28 | 5.07 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:010-attn-q-blk-0-attn-q-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.62 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:011-attn-q-blk-0-attn-q-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 28 | 5.63 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:012-ffn-down-blk-0-ffn-down-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.64 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:013-ffn-down-blk-0-ffn-down-weight-reduce-unroll4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,construction_blocked` | `fail` | 28 | 5.59 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0:qwen3-8b-q4-k-m:014-ffn-down-blk-0-ffn-down-weight-two-dim-local4` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,parts_local_policy,construction_blocked` | `fail` | 28 | 5.34 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_supported",\n  "retry": false\n} |
| `semantic_schedule_v0_full_decode:qwen3-8b-q4-k-m:009-attn-q-blk-0-attn-q-weight-row-upcast2` | `qk_flywheel,kernel_triage,holdout,semantic_schedule_v0,unknown,reject` | `fail` | 28 | 7.67 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "full_decode_ready",\n  "retry": false\n} |
| `semantic_codegen_v3:qwen3-14b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-packed-load-u32x4` | `qk_flywheel,kernel_triage,holdout,semantic_codegen_v3,packed_word_lane_unroll,tie` | `fail` | 28 | 9.06 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_codegen_v3:qwen3-8b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-packed-load-u32x4` | `qk_flywheel,kernel_triage,holdout,semantic_codegen_v3,packed_word_lane_unroll,tie` | `fail` | 28 | 11.81 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_codegen_v4:qwen3-14b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `qk_flywheel,kernel_triage,holdout,semantic_codegen_v4,vector_load,construction_blocked` | `fail` | 28 | 10.28 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `semantic_codegen_v4:qwen3-8b-q4-k-m:001-ffn-gate-blk-0-ffn-gate-weight-vector-load-u32x4` | `qk_flywheel,kernel_triage,holdout,semantic_codegen_v4,vector_load,construction_blocked` | `fail` | 28 | 9.75 | <think>\n\n</think>\n\n{\n  "label": "construction_blocked",\n  "reason": "construction_blocked",\n  "retry": false\n} |
| `qk_block_dot:compile_gate:8b-ffn-gate` | `qk_flywheel,kernel_triage,holdout,qk_block_dot,qk_block_dot,diagnostic_only` | `fail` | 28 | 10.34 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "diagnostic_only",\n  "retry": false\n} |
| `qk_block_dot:microbench:8b-ffn-gate` | `qk_flywheel,kernel_triage,holdout,qk_block_dot,qk_block_dot,reject` | `fail` | 28 | 10.45 | <think>\n\n</think>\n\n{\n  "label": "reject",\n  "reason": "static_gate_fail",\n  "retry": false\n} |
| `threeway_load:blk-0-ffn-gate-weight:tile_custom` | `qk_flywheel,kernel_triage,holdout,threeway_load,tile_custom,reject` | `fail` | 28 | 10.83 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "diagnostic_only",\n  "retry": false\n} |
| `threeway_load:blk-0-ffn-gate-weight:vector_load` | `qk_flywheel,kernel_triage,holdout,threeway_load,wide_load_only,reject` | `fail` | 28 | 12.82 | <think>\n\n</think>\n\n{\n  "label": "accept",\n  "reason": "diagnostic_only",\n  "retry": false\n} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `construction_blocked` | 0 | 19 | 0.00 |
| `diagnostic_only` | 0 | 1 | 0.00 |
| `direct_output` | 0 | 4 | 0.00 |
| `holdout` | 0 | 38 | 0.00 |
| `kernel_triage` | 0 | 38 | 0.00 |
| `packed_word_lane_unroll` | 0 | 2 | 0.00 |
| `parts_local_policy` | 0 | 8 | 0.00 |
| `qk_block_dot` | 0 | 4 | 0.00 |
| `qk_flywheel` | 0 | 38 | 0.00 |
| `raw_accept_unconfirmed` | 0 | 3 | 0.00 |
| `reject` | 0 | 9 | 0.00 |
| `semantic_codegen_v3` | 0 | 2 | 0.00 |
| `semantic_codegen_v4` | 0 | 2 | 0.00 |
| `semantic_schedule_v0` | 0 | 30 | 0.00 |
| `threeway_load` | 0 | 2 | 0.00 |
| `tie` | 0 | 6 | 0.00 |
| `tile_custom` | 0 | 1 | 0.00 |
| `unknown` | 0 | 18 | 0.00 |
| `vector_load` | 0 | 2 | 0.00 |
| `wide_load_only` | 0 | 1 | 0.00 |

## JSON Quality Axes

| axis | passed | scored | pass rate [95% CI] |
|---|---:|---:|---:|
| `parse_valid` | 0 | 38 | 0.00 [0.00, 0.09] |
| `no_extra_text` | 0 | 38 | 0.00 [0.00, 0.09] |
| `schema_ok` | 0 | 38 | 0.00 [0.00, 0.09] |
| `type_ok` | 0 | 38 | 0.00 [0.00, 0.09] |
| `value_correct` | 0 | 38 | 0.00 [0.00, 0.09] |
| `strict_pass` | 0 | 38 | 0.00 [0.00, 0.09] |
