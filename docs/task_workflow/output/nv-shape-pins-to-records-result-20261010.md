# Shape pins move from code into promotion records, result 2026-10-10

Each pin below was a model's shape written as a constant. The kernels now validate only their real limits; the shapes
a route was measured at are data in its promotion record, read by one helper (`model_route_plan.shape_measured`,
`measured_entry`). Widening a route to a new shape is a new measurement recorded as data, never an edit to code.

| pin (before) | now | record field |
|---|---|---|
| `q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel` (12288,4096); its caller (1,4096,12288), norm (4096,) | `Q4KGateUpLaneMap.validate`; the caller checks the gate, up and norm agree | (harness lease only) |
| `Q4KGEMVEpilogue` ffn_down_fused / ffn_down_resadd rows == 4096 | removed | decode-ffn-down-resadd `measured_models` [{dim 4096, hidden_dim 12288}]; decode-q4k-epilogue-fusion `measured_gemv_shapes` [{rows 4096, k 12288}] |
| `Q6KGEMVRouteSpec` ffn_down_resadd rows == 4096 | removed | the same decode-ffn-down-resadd record |
| vocab four-warp N == 151936, K == 4096, device name starts with "NV" | `shape_measured(...)`; the device is the record's promoted target | decode-q6k-vocab-four-warp-fp16 `measured_gemv_shapes` [{rows 151936, k 4096}] |
| `q6k_v_mmvq` (1024,4096) and fixed K inside the emitter | `validate_four_warp(rows, k, warps_per_row)`: whole blocks split over warps x 2 sub-groups | decode-q6k-v-four-warp-fp16-geometry `measured_models` [{dim 4096, n_kv_heads 8, head_dim 128}] |
| flash load schedule num_blocks 36, n_heads 32, n_kv_heads 8, head_dim 128 | `shape_measured(..., field="measured_topologies")` | decode-flash-llama-vec-wide `measured_topologies` |
| `_SHARED_Q8_LEASE` blocks 1-12, 14-18, 25 | `measured_entry(..., field="measured_leases")` by (num_blocks, dim); no entry, no lease | decode-shared-q8-attention `measured_leases` |
| `shared_q8_provider.v1` k == 4096 | `validate_q8_provider(k)`: whole 32-element groups | none |
| `_emit_q6k_partial` `cdiv` (truncating) | `ceildiv`: the last part may be short, no block left out | (test_q6k_partial_ceiling.py) |

The model gate reads `measured_models`; the Q8_0 record's existing `measured_shapes` (its evidence list) is not a gate.

## Default decode unchanged (CPU)

`DecodeRouteShapeGate` decisions for every probed route, before and after: Qwen3-8B 15 admitted, Qwen3.8-27B 5,
Qwen3.5-4B 3, no route changed. The call-site records hold the shapes the constants held.

## New search family: q6_four_warp_gemv.v1 (RTX 5090, 27B, BoltBeam `kinds-27b-p4`)

warps_per_row is the dimension; BubbleBeam proposes 1 to 32 warps from the chip's thread limit, FutureSight's
divisibility rule drops the ones whose sweep does not divide k.

| role | rows x k | four-warp rows legal | BoltBeam's fastest | its us less floor | provider us | model's own us | verdict |
|---|---|---|---|---|---|---|---|
| attn_output Q6_K | 5120 x 6144 | 3 of 6 (1, 2, 4 warps) | four-warp, 128 threads | 18.4 | 19.9 | 21.1 | provider claim not reproduced: not promoted |
| lm_head Q6_K | 248320 x 5120 | 1 (2 warps) | coop, row_tile 2 | 674.4 | 673.6 | 655.9 | the model's own kernel is faster |

Decode in that run: 58.4 tok/s (17.1 ms per token), unchanged.

## Not generalized here

The shared-Q8 attention consumers (`shared_q8_attention.py` `_K, _Q_ROWS, _KV_ROWS = 4096, 4096, 1024`) and the other
four-warp modules' `ROWS, K` (q4k_gate_up_four_warp_mmvq, q4k_ffn_down_mmvq, q6k_ffn_down_mmvq, q6k_vocab_manyrow)
still carry the 8B's shapes inside their kernels; each needs its kernel rewritten over (rows, k), not a validate change.
