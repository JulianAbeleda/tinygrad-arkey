# Pure residual attribution and optimization scope

## Objective

Explain and reduce the remaining gap between the four-role pure buffer2 result
and the 4.4k tok/s ctx512 line without guessing from synchronized debug shares.

Current pinned authority:

| Regime | ctx512 | Time |
|---|---:|---:|
| Original pure | 1,511 tok/s | 338.9 ms |
| Gate/up buffer2 | 2,431 tok/s | 210.6 ms |
| Four exact roles | 3,482 tok/s | 147.05 ms |
| 4.4k target | 4,400 tok/s | 116.36 ms |

Residual: approximately 30.7 ms at ctx512. The four-role census passes for
`ffn_gate_up`, `ffn_down`, `attn_qo`, and `attn_kv`; this scope must not redo
those candidates unless a measurement proves a regression.

## Evidence rules

Every claim must identify its timing regime, clock state, synchronization policy,
shape, route identity, and commit. DEBUG-synchronized kernel totals are an
inventory only, not an Amdahl percentage. Whole-model authority remains:
`K=8`, four warmups, three rounds, 512-token chunks, pinned clocks, clean tree,
passed candidate-set census, and strict-pure route binding.

Each lane has four gates: inventory, isolated correctness, pinned timing, and
whole-model delta. A lane cannot be promoted from a microbenchmark alone.

## Lane A: output projection / LM head (highest priority)

The prefill shape is **M=512, N=151936, K=4096**, not the M=1 decode shape in
the existing Q6_K coop artifact. The model computes logits for every context
token and slices the final token afterward. The existing ~7.8 ms result is
therefore not usable for ranking this residual.

1. Compile and execute the exact `q6k_gen_prefill_direct_out_151936_4096_512`
   route with the real packed output weight.
2. Join source, binary, resource, correctness, and clock-pinned timing evidence.
3. Compare shipped generated-coop, existing packed route, and candidate search
   populations under identical input/output contracts.
4. Run whole-model A/B with only LM-head route changed; retain parity evidence.

Success is a measured whole-prefill reduction, not a better isolated TFLOPS
number. Do not replace the existing route without binary and parity joins.

## Lane B: attention score/value GEMMs

The QK^T and P@V operations are dense GEMMs outside the four promoted linear
roles. Measure their concrete ctx512 shapes and call counts separately; do not
put them in the generic non-GEMM bucket.

## Lane C: non-GEMM attention and normalization

Inventory exact calls and shapes for RMSNorm, RoPE, QK score, masking, softmax,
AV, residual additions, and contiguous/reshape boundaries. Use an unsynchronized
device-event or prepared-call authority where possible. For each candidate:

1. isolate one operation or fused group;
2. prove numerical parity on nonconstant inputs;
3. measure pinned kernel and end-to-end deltas;
4. reject changes that merely move work into host synchronization or graph capture.

The first target is the largest reproducible sum of small operations, not an
arbitrary large kernel.

## Lane D: memory and quantized transport

Measure bytes, residency, dequantization, copies, and synchronization around
Q4_K/Q6_K weights and fp16 overlays. Separate bandwidth limitation from launch
latency. Verify VRAM and cache behavior. No new packed format is admissible until
its numerical contract, weight lifetime, and full-model memory budget are proven.

## Lane E: low-occupancy `attn_kv`

The exact buffer2 KV kernel is correct but reaches about 29.8 TFLOPS versus
58–61 TFLOPS for larger roles. Search only tile/wave/pipeline variants that fit
the exact 512x1024x4096 workload and preserve the candidate-set identity model.
Measure whether its whole-model delta is material before spending search budget.

## Lane F: residual dense routes and orchestration

Audit every prefill GEMM census entry against the four admitted roles. Identify
any dense route outside the set, paired call that falls back, graph recapture,
or route policy mismatch. Then measure kernel launch count, synchronization,
contiguous materialization, and TinyJit replay overhead. State restoration must
remain scoped per model/run.

## Execution sequence

1. Freeze and archive the current four-role authority as comparator.
2. Complete the inventory and attribution table for all lanes.
3. Run Lane A and Lane B isolated authorities in parallel.
4. Run Lane C and Lane D only where the inventory identifies measurable cost.
5. Add one passing lane at a time to the whole-model authority.
6. Re-run ctx512/1024/2048/4096 pinned sweep after every accepted lane.
7. If the residual remains, update the table with measured deltas and open the
   next smallest bounded lane; do not infer a winner from debug shares.

## Agent discipline

Agents must reuse existing route, candidate, timing, census, and parity helpers.
No duplicate kernel emitter, route selector, capability validator, or benchmark
authority is allowed. Host-only scope agents report first; implementation agents
start only after a lane has an exact measurement and a named existing extension
point. GPU agents run one lane at a time and commit evidence with the clean
source revision.

## Completion

This scope is complete when the 30.7 ms residual is partitioned into measured
lane deltas, every promoted lane has correctness/resource/timing/whole-model
evidence, and either the 4.4k line is reached or the remaining work is proven to
be outside the current pure prefill control surface (for example an LM-head or
non-GEMM ceiling requiring a separate generated primitive).
