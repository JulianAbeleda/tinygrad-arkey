# Decode MMVQ large project P2 kernarg capture result - 2026-06-19

Purpose: execute P2 from `decode-mmvq-large-project-scope-20260619.md`.

No tinygrad model route or default changed. HIP was used only in the separate llama.cpp capture process.

Artifacts:

- `extra/qk_decode_mmvq_kernarg_capture.cpp`
- `extra/qk_decode_mmvq_p2_contract.py`
- `bench/qk-decode-mmvq-large-project/p2_kernarg_capture.json`
- `bench/qk-decode-mmvq-large-project/p2_kernarg_capture.jsonl`
- `bench/qk-decode-mmvq-large-project/p2_kernarg_capture_summary.md`

## Verdict

`PASS`.

Captured `7` real llama.cpp decode MMVQ launches:

- Q4_K: `4`
- Q6_K: `3`

The first shim attempt missed them because llama.cpp uses direct `hipLaunchKernel`, not `hipModuleGetFunction` /
`hipExtModuleLaunchKernel`. The final shim exports versioned HIP symbols, intercepts `hipLaunchKernel@hip_4.2`, maps
local kernel handles by low address bits from `libggml-hip.so`, and reconstructs the `144` byte kernarg from the 19
direct launch arguments.

## Selected Templates

| template | role guess | num workgroups | local | ncols_x | stride_row_x | stride_col_dst | fusion |
|---|---|---:|---:|---:|---:|---:|---:|
| Q4_K no-fusion | attn_q_or_o | `[4096,1,1]` | `[32,1,1]` | `4096` | `16` | `4096` | false |
| Q6_K fusion-template | ffn_down | `[4096,1,1]` | `[32,2,1]` | `12288` | `48` | `4096` | true |
| Q6_K no-fusion | lm_head | `[151936,1,1]` | `[32,2,1]` | `4096` | `16` | `151936` | false |

Pointer offsets for rebinding:

- `vx`: `0`
- `vy`: `8`
- `ids`: `16`
- `dst`: `56`

## Decision

Proceed to P3 standalone correctness. Start with the Q4_K no-fusion template because it has no fusion pointer, no ids,
and a manageable `4096 x 4096` real tensor target.
