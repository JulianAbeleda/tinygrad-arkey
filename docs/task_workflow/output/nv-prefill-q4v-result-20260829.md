# NVIDIA pp512 Q4-V result (2026-08-29)

## Decision

**RETAIN, default-off.** The Q4-V route is qualified for the 18 GGML type-12
V projections. The 18 type-14 Q6 V projections remain on the FP16 fallback.

## Exact gates

- Producer symbol fixed to the exported `q8_compact_record_fp16` entry point.
- Producer-only realization is finite and nonzero.
- `blk.4.attn_v.weight` is the first real type-12 Q4 V weight; canonical packed
  words are used. The full output is finite and nonzero.
- The model graph contains exactly 18 Q4-V producers and 18 manifest-symbol V
  mains, with 18 unique record and output allocations.
- The prior non-vacuous deep20 authority reports exact V records/outputs, KV,
  logits, and token replay on all 20 cycles.
- Final short candidate census: 198 canonical weights, 198 producers/mains,
  zero copies/fixups, and 54 remaining FP16 overlays (18 Q6 V + 36 down).

## Matched wall result

Candidate: `nv-prefill-q4v-pass-final-20260829.json`.
Control: `nv-prefill-noq4v-control-20260829.json`.

| arm | min ms | median ms | tok/s |
|---|---:|---:|---:|
| Q4-V candidate | 72.116495 | 72.137084 | 7099.624 |
| no-Q4-V control | 74.002698 | 74.040490 | 6918.667 |

The candidate improves by **1.886203 ms minimum** and **1.903406 ms median**,
or **+180.957 tok/s / +2.62%** using minimum-wall throughput.

Full logits remain allclose: token `198` matches, maximum absolute difference
`0.10494757`, mean absolute difference `0.013181783`.
