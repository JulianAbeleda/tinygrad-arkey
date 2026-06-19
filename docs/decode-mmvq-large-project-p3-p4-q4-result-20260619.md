# Decode MMVQ large project P3/P4 Q4 result - 2026-06-19

Purpose: execute P3/P4 for the clean Q4_K no-fusion imported llama MMVQ template.

No model route or default changed.

Artifacts:

- `extra/qk_decode_mmvq_p3_q4_correctness.py`
- `extra/qk_decode_mmvq_p4_q4_perf.py`
- `bench/qk-decode-mmvq-large-project/p3_q4_correctness.json`
- `bench/qk-decode-mmvq-large-project/p4_q4_perf.json`

## P3 Verdict

`PASS`.

The imported llama Q4_K MMVQ descriptor runs through tinygrad HCQ on tinygrad-owned buffers.

Target:

- tensor: `blk.0.attn_output.weight`
- shape: `4096 x 4096`
- q4 bytes: `9437184`
- activation: locally packed llama `block_q8_1`
- launch: num workgroups `[4096,1,1]`, local `[32,1,1]`

Correctness vs CPU reference using the quantized q8 input:

| metric | value |
|---|---:|
| max_abs | `1.4305e-6` |
| mean_abs | `2.1342e-7` |
| max_rel | `0.0189` |

The max-relative row is high only where the reference is near zero; absolute error is the authority here.

## P4 Verdict

`PASS`.

The imported Q4_K consumer is fast when measured with the right queue protocol.

Naive `HCQProgram.__call__` timing was misleading because it submits every launch and measured host submit overhead.
The final P4 measurement enqueues `200` exec packets into one HCQ submit and uses device timestamps around the batch.

| metric | value |
|---|---:|
| device ms / launch | `0.01044` |
| wall ms / launch | `0.04748` |
| effective q4 GB/s | `903.91` |
| pct of 960 GB/s HBM peak | `94.16%` |
| gate | `>=60%` |

## Meaning

The source/object import consumer path is real:

- descriptor load works;
- raw kernarg rebinding works;
- standalone correctness works;
- standalone performance clears the gate.

This is no longer blocked by the imported kernel.

## Next Gate

P5 cannot be a pure consumer swap. The imported llama MMVQ kernels consume `block_q8_1` activations. In-model routing
therefore needs a q8_1 activation producer/reuse lifecycle:

- produce packed `block_q8_1` for the layer input;
- reuse it across all consumers that share that activation;
- route at least one high-share role through the imported consumer;
- include producer cost in the isolated role gate.

That is the next real project phase.
