# Llama Decode Primitive Difference Audit Scope

Date: 2026-06-21

Owner: next executor

Status: scope only

## Context

Current project state:

- prefill is solved and opt-in policy-shipped;
- global `PREFILL_V2` default stays off;
- decode headline is `~67%` llama at steady context;
- q8 remains default-off opt-in;
- bounded decode fusion is closed;
- raw fused flash, FFN activation fusion, and linearizer fused+coop scalar tile are all refuted;
- the only named remaining decode lever is deep codegen, likely WMMA/tensor-core flash decode or another
  llama-shaped primitive.

The latest failure is important:

| candidate | result | reason |
|---|---|---|
| raw fully fused flash | byte-exact, slower | fusion without coop dataflow loses |
| fused LDS+GQA scalar tile | byte-exact, much slower | decode `T=1` has no query-parallel axis; GQA consolidation reduces workgroups; q·k recompute remains |
| current `gqa_coop_vec` | still winner | preserves GPU-filling parallelism and GQA V reuse, despite split lifecycle |

The intuition is correct: **the blocker is tooling/primitive language, not one missing flag.** tinygrad can express
pieces of the winning primitive, but not yet the llama-class decode primitive as one schedulable/searchable object.

## Objective

Audit llama.cpp to identify the exact primitive and tooling differences that explain why llama still wins, and decide
whether the next project is:

1. `WMMA_FLASH_DECODE`: llama's attention primitive is the remaining decode gap.
2. `MMVQ_REOPEN`: llama's quantized matvec is again the dominant actionable gap under current HEAD.
3. `TOOLING_BLOCKED`: we cannot yet observe the difference at instruction/resource/body level.
4. `REST_DECODE`: no fundable bounded primitive remains without starting the north-star codegen project.

This is an audit first. Do not build kernels until the audit identifies a specific primitive delta and a gate.

## Why We Cannot Proceed Blindly

The project has repeatedly refuted proxy objectives:

- fewer kernels is not sufficient;
- fusion alone is not sufficient;
- LDS reuse alone is not sufficient;
- GQA V reuse alone is not sufficient;
- scalar fused decode tiles do not transfer from prefill because decode has `T=1`;
- raw custom kernels without graph/JIT/search integration are not enough;
- small-op fusion and launch removal do not move clean W==D decode.

The principle:

> A primitive is only useful if it preserves the parallelism/dataflow that fills the GPU and is expressible in the
> model lifecycle.

The current missing thing is not "a fused kernel"; it is an owned primitive language for llama-shaped decode:

- enough parallel work at `T=1`;
- cooperative GQA reuse without starving occupancy;
- q·k and P·V mapped to matrix/tensor hardware or equivalent high-throughput work decomposition;
- online softmax hidden under tile loads;
- graph/JIT integration;
- resource policy before timing.

## Existing Evidence To Reconcile

Read first:

- `docs/current-project-state-handoff-20260621.md`
- `docs/decode-latency-hiding-lifecycle-codegen-result-20260621.md`
- `docs/qk-llama-token-primitive-accounting-20260617.md`
- `docs/qk-8b-decode-block-primitive-map-20260617.md`
- `bench/qk-decode-fused-coop-primitive/path_diff.json`
- `bench/qk-decode-fused-coop-primitive/fused_lds_tile_ab.json`

Llama source checkout:

```text
/home/ubuntu/env/llama.cpp
```

Relevant llama source areas:

- `ggml/src/ggml-cuda/mmvq.cu`
- `ggml/src/ggml-cuda/vecdotq.cuh`
- CUDA/HIP flash attention implementation files under `ggml/src/ggml-cuda/`
- graph and flash routing in `src/llama-graph.cpp`

## Phase 1 — Source Primitive Inventory

Build a source-level table of llama's decode primitives versus tinygrad's.

Required table:

| role | llama primitive | llama source | tinygrad primitive | tinygrad source | status |
|---|---|---|---|---|---|

Must include:

- Q4_K/Q6_K MMVQ matvec;
- q8_1 activation producer/reuse;
- decode flash attention;
- stream-k/fixup/combine;
- RMSNorm/RoPE/residual;
- graph lifecycle / launch model.

Gate:

- identify whether llama's current AMD decode win at ctx512-4096 is more plausibly attention, MMVQ, or mixed.

## Phase 2 — Llama Runtime Trace Refresh

The older llama accounting is useful but stale relative to current tinygrad. Refresh the runtime trace if possible.

Run or reuse rocprof/trace tooling to capture one decode token at:

- ctx512;
- ctx1024;
- ctx4096.

Required outputs:

| ctx | llama tok/s | kernel count | top kernel families | flash share | MMVQ share | graph/other |
|---:|---:|---:|---|---:|---:|---:|

Also capture:

- kernel names;
- launch grid/workgroup;
- LDS/static shared memory;
- VGPR/SGPR if available;
- scratch/private segment if available;
- code object / symbol names.

Artifact target:

```text
bench/qk-llama-decode-primitive-audit/
```

Stop condition:

- if rocprof cannot produce body/resource details, record `TOOLING_BLOCKED` and identify the missing trace capability.

## Phase 3 — Attention Primitive Diff

Compare llama flash attention against tinygrad attention at ctx1024 and ctx4096.

Required table:

| feature | llama | tinygrad current | failed tinygrad candidates | implication |
|---|---|---|---|---|

Must answer:

- Does llama use a WMMA/tensor-core flash tile on AMD for this route, or a scalar/vector tile?
- How many workgroups are launched per layer/head/split?
- Does llama preserve query-head parallelism while reusing K/V?
- Does llama compute q·k once per score, or redundantly across output dimensions?
- Where does online softmax state live?
- Is P·V matrix-style, vector-style, or scalar accumulation?
- Does llama use LDS for K/V staging? If yes, how much and at what tile?
- Is the winning difference resource shape, instruction mix, or lifecycle integration?

Gate:

- `WMMA_FLASH_DECODE` only if llama evidence shows matrix/tensor-core or equivalent high-throughput attention body
  that tinygrad lacks and that plausibly explains the ctx-slope gap.

## Phase 4 — MMVQ Primitive Diff

Re-check whether MMVQ should be reopened.

Prior evidence says q6/MMVQ lanes were closed under the current role/tensor attribution. Older evidence says llama
MMVQ is a major primitive advantage. Reconcile those.

Required table:

| role | tinygrad current GB/s or ms | llama GB/s or ms | gap | current status | reopen? |
|---|---:|---:|---:|---|---|

Must cover:

- attn q/o;
- ffn gate/up;
- ffn down;
- lm_head;
- Q4_K vs Q6_K;
- q8 activation producer cost.

Gate:

- reopen MMVQ only if the refreshed llama-vs-tinygrad role table shows a clean, current, role-local gap that is
  larger than attention and not already refuted by q8/role attribution.

## Phase 5 — Tooling Gap Report

If the audit cannot explain llama's advantage from existing traces, produce a tooling gap report.

Required table:

| needed observation | why needed | current tool | status | next tool primitive |
|---|---|---|---|---|

Potential missing tooling:

- AMD code-object symbol-to-kernel-role join;
- disassembly/resource extraction for llama kernels;
- PC/stall/body timeline attribution;
- per-kernel effective bandwidth and instruction mix;
- graph replay lifecycle comparison;
- comparable clean W==D llama/tinygrad trace at same ctx.

Gate:

- if we cannot observe llama's attention body or MMVQ body well enough to name the difference, do not start a codegen
  project. Tooling is the next primitive.

## Phase 6 — Decision Doc

Write:

- `docs/llama-decode-primitive-difference-audit-result-20260621.md`

Minimum sections:

1. source inventory;
2. runtime trace table;
3. attention primitive diff;
4. MMVQ primitive diff;
5. tooling gaps;
6. decision: `WMMA_FLASH_DECODE`, `MMVQ_REOPEN`, `TOOLING_BLOCKED`, or `REST_DECODE`;
7. next implementation/tooling scope if applicable.

## Stop Conditions

Stop without implementation if:

- llama trace cannot be refreshed or trusted;
- kernel names cannot be mapped to roles;
- the only observed difference is launch count;
- the proposed primitive cannot preserve decode `T=1` parallelism;
- the proposed primitive has no clean W==D promotion path;
- the result would reopen a lane already refuted without new evidence.

## Expected Outcome

The likely result is one of:

- **Tooling first:** we need better llama body/resource attribution before touching codegen.
- **WMMA flash decode:** the remaining gap is specifically llama's high-throughput attention tile.
- **Rest decode:** bounded primitives are exhausted and only the broad north-star codegen project remains.

Do not treat "llama is faster" as sufficient. The audit must name the primitive, the dataflow/instruction/resource
difference, and the tinygrad language/tooling feature needed to express or search it.
