# NV pp512 reusable Q8 activation producer lifecycle

## Verdict

The producer is reusable, but it is a secondary optimization rather than the
compressed-projection breakthrough.  On the retained llama pp512 trace, all
ABI-preserving sharing saves about **0.283 ms**: 89 of 249 conversion launches.
It moves the 36.608 ms llama wall model to about **36.325 ms** (14.09k tok/s),
and its 27.263 ms compressed dense lifecycle to about **26.980 ms**.

The high-value implementation fact is that llama does **not** currently share
these activations.  The trace has exactly 249 Q8 producers for 249 MMQs.  A
tinygrad compressed substrate should nevertheless make the packet an explicit
graph value so sibling projections can share it.

## Exact packet contract

`block_q8_1_mmq` covers 128 activation values and occupies 144 bytes:

- 128 signed int8 values;
- 16 bytes of metadata;
- quantization is per 32 values: `d = max(abs(x))/127`, then
  `q = round(x/d)`;
- Q4_K uses DS4 metadata: four `half2(d, sum(x))` values;
- Q6_K uses D4 metadata: four float32 `d` values;
- the producer treats each 128-value K block as a value and transposes those
  blocks across activation columns, allowing the MMQ to copy a contiguous
  shared-memory tile.

For 512x4096, one materialized packet is 2,359,296 bytes (2.25 MiB), versus
8 MiB for fp32 input.  Its launch geometry in the trace is grid `(512,8,1)`,
block `(128,1,1)`: 4,096 CTAs.  A 512x12288 down input uses grid `(512,24,1)`.

DS4 and D4 are equal-sized unions but are not ABI-compatible: D4 consumers
interpret metadata as float32, while DS4 stores two fp16 fields.  The int8
payload and numerical scale are conceptually common, but sharing across these
layouts requires a generalized consumer or a dual-metadata packet.

## Legal sharing at Qwen3-8B pp512

The retained model has 214 Q4/DS4 and 35 Q6/D4 MMQs.  The 35 Q6 calls comprise
18 V projections plus 17 down projections; the last FFN is pruned.

| scope | consumers | ABI-preserving producers | avoided |
| --- | ---: | ---: | ---: |
| Q/K/V | 108 | 54 | 54 |
| O | 36 | 36 | 0 |
| gate/up | 70 | 35 | 35 |
| down | 35 | 35 | 0 |
| total | 249 | 160 | 89 |

For 18 all-Q4 attention layers, Q/K/V share one DS4 packet.  For 18 mixed
Q4/Q4/Q6 layers, Q and K share DS4 while V still needs D4.  Gate and up always
share one DS4 packet.  O and down have no sibling consumer of the same input.

The shared object should live from completion of the norm/quant producer until
both or all sibling MMQs have consumed it, then be released.  It must not be
retained across transformer layers.  This bounds live extra storage to one
2.25 MiB DS4 packet, plus a second 2.25 MiB D4 packet during mixed QKV if both
are scheduled concurrently.

## Measured cost and traffic

The fresh Nsight kernel summary reports:

| layout | calls | total | average |
| --- | ---: | ---: | ---: |
| DS4 | 214 | 0.681 ms | 3.181 us |
| D4 | 35 | 0.178 ms | 5.084 us |
| total | 249 | 0.859 ms | 3.448 us weighted |

All 89 safely avoided launches are K=4096 DS4 producers, so the correct
estimate is `89 * 3.1806 us = 0.283 ms`, not 89 times the mixed average.  It
also avoids 364,544 producer CTAs, about 747 MiB of fp32 input reads and 200
MiB of Q8 packet writes per prompt (algorithmic bytes; not DRAM-counter bytes).

This optimization only raises the complete compressed dense service rate by
about one percent.  Therefore it belongs in the reusable substrate design,
but direct Q4/Q6 consumption and INT8 tensor-core MMA remain the roofline path.

## Gate and implementation boundary

No production route was edited.  The research analyzer is
`extra/llm_research/prefill/nv_q8_producer_lifecycle.py`; its pinned output is
`docs/task_workflow/evidence/nv-q8-prefill-producer-lifecycle-20260828.json`.
The unit gate checks packet size, trace call counts, layout split, and legal
reuse count.  It passes 3/3.

Before promotion, the compressed consumer must additionally pass a complete
chain gate: fp16/fp32 norm output -> one DS4 packet -> gate+up or Q+K(+Q4 V),
full-output comparison against independent producers, input immutability,
guard regions, finite outputs, and included producer+all-consumers timing.
The existing decode provider is a useful arithmetic oracle, but its M=1 row
layout and launch geometry do not constitute pp512 execution evidence.
