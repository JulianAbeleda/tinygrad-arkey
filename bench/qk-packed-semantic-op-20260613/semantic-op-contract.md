# Packed QK Semantic Op Contract

Decision: `semantic_op_contract_defined_no_runtime_lowering`

This artifact defines the next compiler-facing contract only. It does not add
a runtime lowering, benchmark result, or full-decode claim.

## Summary

- op: `QK_BLOCK_DOT`
- Q4 contract rows: `8`
- skipped rows: `6`
- runtime lowering exists: `False`
- next step: `minimal compile gate for QK_BLOCK_DOT renderer/core lowering`

## Contract Rows

| model | tensor | role | shape | load tile | lowering target | status |
|---|---|---|---:|---|---|---|
| `8B` | `blk.0.ffn_gate.weight` | `ffn_gate` | `12288x4096` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `8B` | `blk.4.ffn_down.weight` | `ffn_down` | `4096x12288` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `8B` | `blk.0.attn_q.weight` | `attn_q` | `4096x4096` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `8B` | `blk.0.attn_k.weight` | `attn_k` | `1024x4096` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `14B` | `blk.0.ffn_gate.weight` | `ffn_gate` | `17408x5120` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `14B` | `blk.5.ffn_down.weight` | `ffn_down` | `5120x17408` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `14B` | `blk.0.attn_q.weight` | `attn_q` | `5120x5120` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |
| `14B` | `blk.0.attn_k.weight` | `attn_k` | `1024x5120` | `u32x4_aligned` | `amd_renderer_pattern` | `design_only_no_runtime_lowering` |

## Hidden Boundary

The semantic op may hide Q4_K scale/min unpack, nibble extraction, and a
target load intrinsic spelling inside one block. It must not hide the row
loop, K-block loop, split-K output layout, partial reduction, full GEMV body,
or runtime policy selection.

## Promotion Gates

| gate | requirement |
|---|---|
| `reference_unpack` | bit-exact Q4_K unpack against extra.qk_layout.q4_k_reference |
| `amd_gemv_numeric` | random fp16 activation GEMV compare against current v1 primitive |
| `source_width` | generated source records intended packed load spelling |
| `target_width` | DEBUG=7 target block contains wide/coalesced load evidence |
| `scheduler_shape` | target workgroup shape preserves v1-like schedulable row/K parallelism; reject workgroup-size 1 |
| `target_body_size` | target instruction count must not exceed 2x comparable v1 kernel without a measured win |
| `microbench` | repeated dominant-shape median gain >= 10% before full decode |
| `full_decode` | 8B and 14B confirmation reruns accept; 32B optional only after promise |
| `greedy_ab` | end-to-end greedy output A/B passes |

## Skipped Rows

| model | tensor | format | reason |
|---|---|---|---|
| `8B` | `output.weight` | `Q6_K` | first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work |
| `8B` | `blk.0.attn_v.weight` | `Q6_K` | first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work |
| `8B` | `blk.0.ffn_down.weight` | `Q6_K` | first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work |
| `14B` | `output.weight` | `Q6_K` | first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work |
| `14B` | `blk.0.attn_v.weight` | `Q6_K` | first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work |
| `14B` | `blk.0.ffn_down.weight` | `Q6_K` | first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work |
