# BoltBeam-Generated Decode And Prefill Kernel Scope

Date: 2026-08-30

Status: implementation scope; production defaults unchanged

## Goal

Use BoltBeam to generate, measure, rank, and promote typed kernel plans for decode and prefill while tinygrad owns reusable semantic primitives, lowering, rendering, compilation, runtime buffers, and execution. The selected runtime must not depend on llama kernel binaries or route-local handwritten CUDA.

## Confirmed Existing BoltBeam Capability

BoltBeam already provides:

- canonical JSON and SHA-256 candidate identity;
- deterministic finite Cartesian and coupled-row candidate expansion;
- exact target and workload binding;
- an isolated JSONL provider protocol with describe, admit, compile, check, and measure stages;
- candidate-local invalid/unsupported/incorrect dispositions;
- generated source, plan, and binary identity joins;
- correctness and performance evidence hashes;
- deterministic ranking and selection;
- route-policy emission and closed-default promotion.

The existing control plane is sufficient. A replacement search engine is not required.

## Confirmed Gap

`boltbeam.full_kernel_candidate.v2` is matmul-specific. It requires M/N/K, operands A/B/C, and an OptOps schedule. It cannot truthfully describe activation quantization, Stream-K fixup, decode attention, or Flash attention.

The tinygrad CUDA provider currently admits a separate Flash descriptor but does not admit a shared typed primitive graph. The missing cross-repo contract is therefore `boltbeam.full_kernel_candidate.v3` plus a tinygrad primitive-plan adapter.

## Ownership Boundary

BoltBeam owns:

- exact workload and target facts;
- finite candidate topology and schedule parameters;
- candidate identity and provenance;
- campaign budgets and objectives;
- correctness/performance/resource evidence joins;
- ranking, rejection, and promotion.

Tinygrad owns:

- validation against live compiler/backend capability;
- typed primitive semantics;
- UOp construction and graph ownership;
- lowering, rendering, binary compilation, and cache identity;
- runtime buffers, graph capture, synchronization, and resource extraction;
- oracle execution and measured samples.

BoltBeam must not emit ISA, embed CUDA source, assign physical registers, or ship precompiled binaries.

## V3 Primitive Plan Contract

A candidate carries:

- generic named workload axes rather than forced M/N/K;
- named input/output operands with direction, dtype, layout, and quantization;
- exact resolved target identity;
- explicit dispatch and workgroup dimensions;
- dispatch/workgroup/subgroup/serial/reduce axes;
- a forward typed primitive DAG;
- named outputs;
- finite schedule parameters;
- static resource constraints;
- oracle and tolerances;
- generator provenance and exact applicability.

Initial primitive vocabulary:

- global/workgroup-memory load and store;
- workgroup barrier;
- scalar ALU, select, clamp, cast, and bitcast;
- precise division and ties-away rounding;
- subgroup shuffle and subgroup max/sum reductions;
- byte packing and bit unpacking;
- matrix MMA;
- online-softmax state update.

Every primitive must fail closed when the selected target has no lowering.

## Vertical Slice 1: DS4 Producer

Exact workload:

- phase: prefill;
- rows: 512;
- K: 4096;
- input: row-major FP16;
- output: segment-major Q8_1 DS4, 144 bytes per 128 values;
- correctness: byte-exact against CPU and llama oracle;
- performance control: llama producer median approximately 0.786 ms.

Initial population:

| Candidate | Grid | Block | Ownership |
| --- | --- | --- | --- |
| control | 16384 x 1 x 1 | 32 x 4 x 1 | one record per CTA, four warps per record |
| llama-shaped | 4096 x 1 x 1 | 32 x 4 x 1 | four records per CTA, one warp per record |

The grid arithmetic proves llama produces four records per CTA: 512 rows times 8 CTAs per row is 4096 CTAs, while the output contains 16384 records.

Search dimensions after the first two candidates execute:

- records per CTA: 1, 2, 4;
- warps per record: 4, 2, 1;
- values per lane: 1, 2, 4;
- store width: 1, 2, 4 bytes;
- shared versus register reduction where legal;
- flattened versus row/segment launch decomposition.

DS4 completion gates:

1. Candidate bytes and plan hash are reproducible.
2. Tinygrad admits the candidate using live NV facts.
3. Tinygrad emits source and binary identities without llama binaries.
4. One-record output is byte-exact against the CPU oracle.
5. M512/K4096 output is byte-exact against CPU and llama.
6. Timing and resources are returned through the provider protocol.
7. BoltBeam ranks the complete finite population.
8. A winner is promoted only if it improves the current native control and does not regress model-level timing.

## Decode Expansion

Candidate families:

- Q4_K and Q6_K GEMV;
- shared Q8 activation production and reuse;
- attention Q/K/V/O projections;
- KV-cache read/dequant layouts;
- decode Flash/online attention;
- RMSNorm, RoPE, residual, and output epilogues.

Required sweep axes:

- batch and token count;
- context length;
- layer depth and cache residency;
- Q4_K versus Q6_K;
- head count and grouped-query ratio;
- output dtype and epilogue composition.

## Prefill Expansion

Candidate families:

- packed Q4/Q6 tensor-core matmul;
- gate/up activation reuse and SiLU-multiply epilogue;
- FFN-down residual epilogue;
- Q/O and K/V role-specific layouts;
- Stream-K ownership, partial workspace, and deterministic fixup;
- cooperative Flash staging and online softmax/PV.

Required sweep axes:

- token count and context length;
- M/N/K role shapes;
- layer depth;
- owner count and serial tiles;
- stage and buffer count;
- CTA geometry and tensor-core tile;
- output dtype and fusion policy.

## Evidence And Promotion

Each measured candidate must carry:

- candidate, plan, generated source, and binary hashes;
- exact target and provider revision;
- compile disposition and resource facts;
- oracle identity and correctness result;
- synchronized raw timing samples;
- execution/graph-capture regime;
- route census proving no fallback;
- comparison against generic tinygrad, current promoted route, llama, and applicable roofline.

Isolated-kernel wins are necessary but insufficient. Promotion requires the matching phase/context/depth model ledger to improve.

## Slices

1. V3 schema, deterministic DS4 population, and CPU-only provider contract tests.
2. Live tinygrad DS4 plan lowering with compile-failure evidence.
3. Correct one-record and full-shape execution.
4. BoltBeam DS4 campaign, ranking, and selected-plan export.
5. Generated artifact loader and default-off runtime binding.
6. Decode candidate families and depth/context campaign.
7. Prefill matmul/Stream-K candidate families and campaign.
8. Decode and prefill attention candidate families and campaign.
9. Whole-model ledger, closed-default promotion, rollback, commit, and push.

## Non-Goals

- BoltBeam does not become the runtime executor.
- A local language model may propose candidate plans but cannot authorize them.
- Existing llama cubins remain development oracles only.
- Existing production defaults do not change during schema or admission work.
- Compile success without executed correctness and timing is not promotion evidence.
