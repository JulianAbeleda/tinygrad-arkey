# NV sm_120: Q5_K in the block-quant decode GEMV family, result 2026-10-10 (not promoted)

Q5_K joins `BLOCK_QUANT_LAYOUTS` as data: 256-element blocks, 176 bytes (d, dmin, 12 packed scale/min bytes, 32 qh,
128 qs), 64 uint16 code units, so a 32-wide warp holds the block at two units a lane. One terms function decodes the
packed 6-bit scale/min (GGML get_scale_min_k4) and the fifth bit; the layout's scale is 1 because each term is the
dequantized weight. The record's formats stay `[Q8_0]`. Per-format rollback: `TINYGRAD_BLOCK_QUANT_FORMATS_DISABLE=Q5_K`.

## CPU (test/unit/test_block_quant_q5_k.py)

The serial spelling (the warp route's per-lane math) against tinygrad's GGUF dequantizer on random blocks: Q5_K at
5x768 (32 lanes), 3x512 (8 lanes), 4x256 (1 lane), and Q8_0/Q5_0/Q4_0 beside it; rtol 2e-4. BoltBeam's pure-Python
Q5_K reference matches ggml_data_to_tensor exactly (BoltBeam tests/test_block_quant_search.py).

## RTX 5090, the real Q5_K tensors of Qwen_Qwen3.5-4B-Q4_K_M (extra/llm_research/decode/block_quant_microgate.py)

| tensor | rows x k | lanes/block | warps/CTA | max rel err | kernel us (cold L2) | GB/s | % of 1,693.9 |
|---|---|---|---|---|---|---|---|
| blk.3.attn_k | 1024 x 2560 | 32 | 2 | 1.4e-7 | 10.5 | 172 | 10% |
| blk.3.attn_output | 2560 x 4096 | 32 | 2 | 1.9e-7 | 19.3 | 373 | 22% |

Correct on the GPU. Slow on these small tensors (1.8 and 7.2 MB; the Q8_0 1024x5120 shape ran at 34%): the per-unit
scale/min decode is more ALU than Q8_0's.

## Why not promoted

The fork cannot decode Qwen3.5-4B today, in either arm: prefill fails inside the graph rewrite
(`bad reshape: () -> (2560,)` under `model.generate`). Two earlier load failures on this model were fixed on the way:
the prefill inventory listed the next-n (MTP) block's tensors, and a partial Q6_K spec's row_tile was reported as drift.
With no in-model tok/s and no logits against control, Q5_K is not promoted. Q5_0 is not promoted either: the Nemotron
models that carry real Q5_0 load through nemotron_h.py, which dequantizes every weight to fp16 at load and keeps no
packed bytes for the route to bind.
