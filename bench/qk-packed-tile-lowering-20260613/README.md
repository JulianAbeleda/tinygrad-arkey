# QK Packed Tile Custom Lowering

Date: 2026-06-13

## Verdict

`semantic_custom_lowering_constructed_but_not_promoted`

The smallest real Q4_K `PackedQKTile` consumer now exists as
`q4k_gemv_tile_custom_partial_kernel`. It keeps fp16 activations, uses a custom
semantic kernel body, consumes Q4_K payload words with `tg_uint4`, unpacks
low/high nibbles, and writes the existing split-K partial shape.

Correctness passes on AMD:

- Q4_K unpack gate: exact for checked rows.
- Q4_K GEMV gate: random fp16 activation, `max_abs <= 0.0033` on full-shape
  microbench cases.
- `parts=1` and `parts=4` both pass on a two-row construction check.

Generated-source evidence passes:

- `load-width/report.md` classifies the kernel as `tile_custom_partial`.
- inferred load width is `vector_u32x4`.
- kernel name: `q4k_gemv_tile_custom_partial_2_4096_1`.

Performance is a weak microbench signal, not enough for full-decode promotion:

| tensor | mode | parts | ms | Q4 GB/s | max_abs | vs v1 |
|---|---|---:|---:|---:|---:|---:|
| `blk.0.ffn_gate.weight` | `tile_custom` | 1 | 0.131317 | 215.60 | 0.001793 | +7.20% |
| `blk.0.ffn_gate.weight` | `partial` v1 | 1 | 0.140775 | 201.11 | 0.001792 | baseline |
| `blk.0.attn_output.weight` | `tile_custom` | 1 | 0.137505 | 68.63 | 0.003206 | +5.83% |
| `blk.0.attn_output.weight` | `partial` v1 | 1 | 0.145520 | 64.85 | 0.003207 | baseline |

The pre-registered bar for full-decode promotion on this memory-access track is
a dominant-shape microbench gain strong enough to survive model-scope dilution,
expected around `>=10%`. This pass does not clear that bar. No full-decode run
was promoted.

## Commands

Construction gates:

```bash
DEV=AMD PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  ~/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 2 --iters 1 \
  --mode tile_custom --parts 1

DEV=AMD PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  ~/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 2 --iters 1 \
  --mode tile_custom --parts 4
```

Load-width report:

```bash
DEV=AMD DEBUG=4 PYTHONPATH=. .venv/bin/python extra/q4_k_gemv_primitive.py \
  ~/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --rows 2 --iters 1 \
  --mode tile_custom --parts 1 \
  > bench/qk-packed-tile-lowering-20260613/load-width/tile-custom-debug4.log 2>&1

PYTHONPATH=. .venv/bin/python extra/qk_load_width_report.py \
  bench/qk-packed-tile-lowering-20260613/load-width/tile-custom-debug4.log \
  --json bench/qk-packed-tile-lowering-20260613/load-width/report.json \
  --md bench/qk-packed-tile-lowering-20260613/load-width/report.md \
  --repo .
```

Microbench comparisons:

```bash
DEV=AMD PYTHONPATH=. .venv/bin/python extra/q4_k_bench.py \
  ~/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --iters 3 --primitive \
  --primitive-mode tile_custom --primitive-parts 1 --format json

DEV=AMD PYTHONPATH=. .venv/bin/python extra/q4_k_bench.py \
  ~/models/Qwen3-8B-Q4_K_M.gguf --device AMD \
  --tensor blk.0.ffn_gate.weight --iters 3 --primitive \
  --primitive-mode partial --primitive-parts 1 --primitive-opt LOCAL:0:32 \
  --format json
```

The same comparison was also run for `blk.0.attn_output.weight`.

## Interpretation

This closes the previous construction blocker: a semantic/custom Q4_K tile
consumer can be wired into a real GEMV partial kernel with fp16 activations and
wide source loads.

It does not yet prove a useful end-to-end optimization. The custom body is still
opaque to tinygrad's scheduler/search and only weakly beats the current v1
microbench on two Q4_K shapes. The next useful step is either source/counter
analysis of why the wide-load custom body only moves microbench by `5-7%`, or a
true renderer/core semantic op that exposes this lowering to the compiler
instead of leaving it as raw `Ops.CUSTOM`.
