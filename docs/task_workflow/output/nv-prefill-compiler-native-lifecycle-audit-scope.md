# NVIDIA prefill compiler-native lifecycle audit scope

## Objective

Explain the complete pp512 token-wall difference among:

1. the corrected tinygrad NVIDIA FP16 route;
2. the default-off tinygrad NVIDIA packed-Q4/Q8/IMMA v4 route;
3. llama.cpp's NVIDIA Q8_1 x packed-Q4/Q6 MMQ route; and
4. the promoted AMD packed-WMMA route as a compiler-ownership reference.

The end-state being evaluated is not an opaque faster cubin.  It is a normal
tinygrad-compiled NVIDIA projection in which the compiler owns the packed
weight layout, fragment decoding, tensor-core operation, buffers, and graph
dependencies.  On AMD this is packed decode plus FP16 WMMA.  The NVIDIA
analogue is packed decode plus signed-int8 IMMA, with any Q8 producer and
Stream-K fixup represented as explicit compiler/scheduler-owned graph values.

## Questions the audit must close

1. Where does every material part of the llama wall go?
2. Where does every material part of the tinygrad FP16 wall go?
3. Why does the qualified packed-v4 primitive improve in isolation but regress
   at the whole-model boundary?
4. Which projection roles remain on FP16, and what is their recoverable wall?
5. Which costs belong to arithmetic, global/shared-memory service, Stream-K,
   scratch footprint, graph materialization, submission, or requested-row
   liveness?
6. Which AMD compiler contracts transfer unchanged, which require an NVIDIA
   specialization, and which are hardware-specific and cannot transfer?
7. Is a compiler-native packed NVIDIA route feasible with the current IR and
   scheduler, or is a new typed fragment/buffer-ownership contract required?
8. What is the smallest test that distinguishes those two outcomes without
   prematurely building a production route?

## Non-goals

- Do not promote `NV_Q4_IMMA_PP512` or change its default.
- Do not tune another opaque CUDA kernel before the lifecycle ledger closes.
- Do not compare AMD and NVIDIA absolute throughput as if model, GPU, and shape
  were identical.
- Do not mix profiler-instrumented totals with unprofiled authority walls.
- Do not treat kernel counts, overlap, fusion, cp.async, or TMA as causes unless
  a matched measurement supports the claim.
- Do not claim generic dense-model support from one Qwen3-8B shape.

## Comparison arms

### A. Corrected tinygrad NVIDIA FP16 authority

- Exact model, prompt, pp512 contract, output token, and weight identity.
- Synchronized warm R9 wall, with an R7 fresh-process corroboration.
- Existing four-role bit-exact and whole-model SHA/argmax evidence remains a
  prerequisite.

### B. tinygrad NVIDIA packed-v4 research arm

- Same model/prompt as A.
- `NV_Q4_IMMA_PP512=1`, default-off admission, exact current provider identity.
- Confirm 72 Q8 producers, 72 IMMA mains, and 72 fixups.
- Preserve full-logit tolerance, token, finite, sentinel, read-only input, and
  complete-output gates.

### C. llama.cpp NVIDIA authority

- Same GGUF/model, pp512 prompt, requested-output contract, and stable clocks.
- Unprofiled same-session wall bracket is the comparison authority.
- Nsight/CUPTI is used only for call census and device-active accounting.
- Record all Q8_1 conversions, Q4/Q6 MMQs, fixups, attention kernels, support
  kernels, final-layer row gathering, and vocabulary MMVQ.

### D. AMD compiler-ownership reference

- Static and retained-evidence audit, not an absolute timing comparison.
- Trace packed weight from `prefill_packed_weight()` through
  `PackedWeightTransform`, postrange fragment production, `Ops.PROGRAM`,
  scheduler ownership, and graph replay.
- Enumerate the exact promoted shapes/roles and clearly label frozen-shape
  limits.

## Lifecycle ledger schema

Every row must contain:

- lifecycle region and projection role;
- invocation count;
- input/output dtype and logical shape;
- packed and expanded byte footprints;
- grid, block, CTAs, warps, and occupancy-relevant resources;
- global, L2, shared, local, and scratch bytes when measured;
- tensor instruction family and useful MACs;
- kernel-active duration;
- containing graph/device-active duration;
- synchronized wall contribution or a defensible bound;
- producer and consumer ownership;
- whether the value is compiler-owned, scheduler-owned, or opaque;
- correctness contract;
- evidence path;
- evidence class: `measured`, `derived`, `source-proven`, or `unknown`.

No derived number may be presented as a hardware counter or wall authority.

## Lifecycle stages to enumerate

1. Model-load representation and any one-time overlay construction.
2. Per-layer RMS normalization.
3. Activation dtype conversion/materialization.
4. Q8 quantization, scale production, and raw-sum metadata.
5. Packed-weight global loads.
6. Q4/Q6 metadata decode and nibble/bit extraction.
7. Shared-memory staging and fragment formation.
8. HMMA/WMMA/IMMA issue and accumulator correction.
9. Static-tile or Stream-K work ownership.
10. Partial-buffer production and fixup.
11. Projection output publication and dtype conversion.
12. RoPE, q/k norm, KV writes, SiLU/multiply, residuals, and layer handoffs.
13. Flash Attention main/reduction.
14. Final-layer requested-row pruning or lack thereof.
15. Vocabulary projection and token selection.
16. Graph capture/replay, host submission, synchronization, and unexplained
    wall residual.

## Work packages

### WP0: authority and hygiene

- Record branch, HEAD, device, driver, clocks/power state, model hash, command,
  environment, and dirty-tree status.
- Reuse retained evidence only when its binary/model/command identity is
  sufficient.  Mark stale or mismatched artifacts explicitly.
- Serialize GPU work with `/tmp/gpu-bench.lock`.

### WP1: static call and role census

- Produce a per-role table for Q, K, V, O, gate, up, down, final-layer vector
  work, and vocab.
- Reconcile 252 tinygrad FP16 projections, the 72-call packed research
  admission, and llama's 249 full-batch MMQs plus its pruned final-layer tail.
- For each role identify Q4_K/Q6_K, M/N/K, algorithmic bytes, and route owner.

### WP2: matched wall brackets

- Run A and B in alternating/reversed order under the same warmed clock state.
- Use synchronized R9 for canonical wall and fresh R7 for corroboration.
- Re-bracket C in the same session when possible; otherwise retain and label
  the exact authority boundary.
- Report min, median, range, token, and prompt tokens/s.  Never substitute a
  profiled wall for these numbers.

### WP3: device-active lifecycle traces

- Capture A, B, and C with their supported tracing mechanisms.
- Produce normalized categories: dense, attention, vocab, support, copies,
  graph/submission residual.
- For B separately total Q8 producer, IMMA main, fixup, remaining materialize
  copies, and non-native projection bodies.
- Reconcile summed active time to profiled wall and state the residual.

### WP4: projection triangle by role

For Q/O, K/V, gate/up, and down compare:

- tinygrad FP16 body;
- tinygrad packed-v4 where admitted;
- llama Q4/Q6 conversion + MMQ + fixup;
- analytic useful MACs and packed/expanded bytes.

Identify whether each gap is representation, service rate, occupancy geometry,
or lifecycle overhead.  Do not extrapolate gate/up results to other roles
without labeling them as estimates.

### WP5: hardware service accounting

- Retain or collect one matched counter set per representative role.
- Include DRAM/L2/shared traffic, L2 hit rate, tensor-pipe utilization, active
  warps, long/short scoreboard, barriers, registers, spills, and occupancy.
- For native HCQ paths where CUPTI cannot observe kernels, state the observation
  wall and use SASS plus analytic bytes rather than invented counters.
- Explain the isolated packed main versus 72-call in-model service-rate drop.

### WP6: graph, scratch, and ownership accounting

- Enumerate all packed-v4 live allocations and lifetimes.
- Reconcile Q8, scales, sums, outputs, partials, ids/maps, and total retained
  workspace bytes across 72 projections.
- Prove why raw shared scratch aliases watchdog and why dependency-carried reuse
  produces scheduler cycles.
- Account for remaining boundary materializations by bytes and active time.
- Identify the minimal scheduler representation required for ordered repeated
  native writes or a bounded non-aliasing alternative.

### WP7: AMD-to-NVIDIA compiler transfer map

For each AMD component classify it as:

- reusable unchanged;
- reusable with an NVIDIA implementation;
- hardware-specific/non-transferable; or
- missing from the NVIDIA compiler.

At minimum cover route admission, `PackedWeightTransform`, packed carrier,
postrange range ownership, typed fragment production, tensor-core selection,
shared-memory planning, compiler `Ops.PROGRAM` creation, buffer ownership,
scratch planning, callification, capture/replay, canaries, and shape guards.

The proposed NVIDIA compiler contract must prevent the already-observed generic
postrange bug where range permutation changes Q4 nibble parity.  It must expose
a typed logical `(N,K) -> int8 fragment` provider rather than treating packed
decode as an arbitrary ALU expression.

### WP8: correctness and generality boundaries

- State the numerical difference between FP16 activation/HMMA and Q8/IMMA.
- Preserve full-logit, selected-token, finite, sentinel, and read-only gates.
- List every currently supported shape and every untested shape.
- Separate a Qwen3-8B pp512 research admission from a generic dense-model
  compiler capability.

### WP9: recovery ledger and decision

Rank remaining levers by recoverable synchronized wall time, confidence,
implementation cost, and generality.  Include at least:

- compiler-native packed fragment provider;
- packed main service-rate improvement;
- graph-safe scratch reduction or no-partial ownership;
- expansion to Q/K/V/O/down;
- final-layer row pruning;
- quantized vocabulary head;
- remaining attention gap.

For each lever give a conservative, central, and physics-bound endpoint, with
the arithmetic that converts milliseconds into pp512 tokens/s.  Do not sum
overlapping recoveries.

## Completion gates

The audit is complete only when:

1. every lifecycle stage is measured, derived with explicit arithmetic, or
   marked unknown with a precise observation test;
2. every projection role has a route/count/dtype/byte/time entry;
3. A, B, and C wall authorities are not mixed with profiler totals;
4. the packed-v4 isolated-to-model loss is reconciled within a bounded residual;
5. AMD compiler ownership is traced from packed model buffer to scheduled
   program output;
6. the smallest compiler-native NVIDIA discriminator has exact inputs,
   outputs, failure modes, and promotion thresholds;
7. the report says either `INVEST` with a bounded implementation plan or
   `WALL` with the missing substrate and evidence needed to reopen it;
8. no production route is changed by the audit itself.

## Required deliverables

- `docs/task_workflow/output/nv-prefill-complete-lifecycle-ledger.md`
- a machine-readable JSON ledger beside the report;
- retained raw wall/profile/counter outputs under a new evidence directory;
- an AMD-to-NVIDIA transfer matrix;
- a ranked recovery table in milliseconds and pp512 tokens/s;
- the exact next discriminator, or a precise reason none is yet admissible.
