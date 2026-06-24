# SCOPE - Strong prefill primitive audit: Tensile and llama

Date: 2026-06-20

## Purpose

Build a strong audit of the fast prefill implementations, not just a decision-grade summary.

The goal is to understand the portable primitives behind:

- rocBLAS/Tensile fp16 GEMM prefill wins;
- llama.cpp pp512 prefill wins;
- tinygrad's current prefill gap to both.

This document scopes the work needed before machine search or native renderer work can target the right space.

## Current Starting Point

Clock is closed as a cause:

- `docs/prefill-clock-threeway-result-20260620.md`
- WMMA, Tensile, and llama all run at high SCLK/MCLK.
- `manual_peak` does not materially change throughput.

Known pp512 line:

| engine | current result | clock verdict |
|---|---:|---|
| tinygrad WMMA | ~1436-1438 tok/s | high clock |
| tinygrad + Tensile | ~2662-2664 tok/s | high clock |
| llama.cpp | ~3136-3139 tok/s avg | high clock |

Known anatomy:

- Tensile prefill win is dense fp16 GEMM dataflow: tiling, LDS/shared-memory staging, WMMA, K-loop scheduling, waits,
  and resource policy.
- llama pp512 prefill is mostly quantized MMQ/matmul plus rocBLAS GEMM; attention is not the pp512 bottleneck.
- tinygrad's current search space does not expose enough primitives to discover either fast implementation.

## Definition Of "Strong Audit"

A strong audit is complete only when each high-share primitive has:

1. provenance: exact binary/source/build/commit/model/hardware;
2. top-line timing: same clock policy and comparable pp512/pp1024 measurement;
3. kernel ledger: per-kernel share, role/family, call count, time, launch shape;
4. static kernel anatomy: instruction classes, VGPR/LDS/scratch, workgroup, waves, barriers/waits;
5. dataflow reconstruction: global -> LDS/shared -> fragment/register -> tensor op -> epilogue;
6. dynamic evidence: counters or trace evidence for memory traffic, LDS use, stalls, occupancy where available;
7. tinygrad contrast: same role/shape compared against the current tinygrad authority;
8. transfer row: portable primitive name, backend substrate needed, correctness gate, performance gate;
9. stop rule: what would close the row as non-transferable or already explained.

Decision-grade is not enough. "Tensile uses LDS" or "llama uses MMQ" is not a strong audit unless the dataflow and
transfer row are explicit enough to build a correctness harness or search space from it.

## Non-Goals

- No default route change.
- No new native scheduler/renderer implementation in this audit.
- No byte-for-byte clone requirement.
- No claim that Tensile primitives transfer to decode by analogy.
- No claim that llama decode MMVQ primitives automatically explain llama prefill MMQ.
- No blind machine search until the missing primitives are represented.

## Track A - Tensile Strong Audit

Target:

- selected rocBLAS/Tensile fp16 prefill kernels for `ffn_gate/up`, `ffn_down`, and `attn_q/o`;
- primarily the pp512 shapes used by `PREFILL_TENSILE_GEMM=1`;
- compare against tinygrad clean WMMA authority for the same dense GEMM roles.

Existing inputs:

- `bench/qk-tensile-extraction/*.json`
- `docs/prefill-tensile-DEFINITIVE-source-of-truth-20260619.md`
- `docs/prefill-tensile-tpe1-selection-result-20260619.md`
- `docs/prefill-tensile-tpe2-contract-result-20260619.md`
- `docs/prefill-tensile-tpe4-perf-result-20260619.md`
- `docs/prefill-tensile-tpe5-shape-matrix-result-20260619.md`
- `docs/tensile-variant-capture-result-20260619.md`
- `docs/amd-broad-backend-bb5a10-tensile-layout-audit-20260619.md`
- `docs/amd-broad-backend-bb5a9-causal-delta-package-20260619.md`
- `docs/prefill-primitive-pmc-result-20260619.md` with its validation caveat.

Required audit rows:

| Row | Question | Required Evidence | Strong Gate |
|---|---|---|---|
| A0 provenance | Which `.co`, solution, kernel name, ROCm/Tensile version, and shape? | source path, function name, kernarg, launch dims, build metadata | one selected solution per role is identified and reproducible |
| A1 solution parameters | What are MT, DepthU, WorkGroup, TT, vector widths, LDS bytes, PGR/PLR, LdsBlock? | Tensile metadata/source extraction plus disasm confirmation | parameter row maps to actual instructions |
| A2 static ISA ledger | What instructions implement the win? | disasm grouped into global load, LDS store/load, WMMA/FMA, waits/barriers, addr arithmetic | ledger exists for each role and selected kernel |
| A3 dataflow layout | How do A/B move from global to LDS to WMMA fragments? | address formulas, LDS offsets, bank/layout inference, role-specific diagrams | enough to write a correctness-only microkernel |
| A4 K-loop pipeline | Where are prefetch, local read, compute, wait, barrier stages? | ordered instruction trace/block annotations | stage order explains load/compute overlap |
| A5 resource policy | Why does the selected kernel avoid spill and keep occupancy? | VGPR/LDS/scratch/waves/workgroup metadata | scratch/private zero; resource row tied to timing |
| A6 dynamic counters | Does runtime evidence match the static story? | GL2C, LDS activity, wait/stall, waves/occupancy counters where available | counters directionally confirm less global traffic or fewer stalls |
| A7 variant contrast | Which Tensile variants lose and why? | solution sweep/variant capture tied to parameters | at least one loser/winner contrast per major primitive class |
| A8 tinygrad contrast | Which pieces tinygrad lacks today? | same-shape authority disasm and timing, PTM bridge, P0 baseline | rows name concrete tinygrad substrate gaps |
| A9 transfer row | What portable primitive should a native backend expose? | transfer matrix entry with correctness/perf gate | row is searchable/buildable only after substrate exists |

Tensile output artifact:

- `docs/prefill-tensile-strong-audit-result-20260620.md`
- `bench/prefill-strong-audit/tensile/*.json`

Tensile strong-audit pass condition:

- We can describe the selected kernel as a portable dataflow, not as "Tensile magic."
- We can name which primitive rows are missing in tinygrad: layout, staging, K-loop pipeline, waits, resource policy,
  or launch/timing.
- We can write the first correctness-only transfer harness without guessing the memory layout.

## Track B - llama Prefill Strong Audit

Target:

- llama.cpp pp512 and pp1024 prefill on Qwen3-8B-Q4_K_M;
- focus on the high-share `quantized MMQ/matmul` bucket, then rocBLAS GEMM, then attention and lifecycle kernels;
- compare against tinygrad WMMA and Tensile routes only after role/family mapping is explicit.

Existing inputs:

- `docs/llama-kernel-residual-primitive-audit-20260619.md`
- `bench/llama-kernel-residual-primitive-audit-20260619/*`
- `docs/prefill-clock-threeway-result-20260620.md`
- local llama.cpp: `/home/ubuntu/env/llama.cpp`, build `9592`, commit `ac4cddeb0`

Required audit rows:

| Row | Question | Required Evidence | Strong Gate |
|---|---|---|---|
| B0 provenance | Which llama build, flags, model, backend, graph mode, clock lane? | `llama-bench`, build flags, git commit, GPU/ROCm, clock telemetry | pp512/pp1024 top-line reproduced |
| B1 kernel ledger | Which kernels make up pp512/pp1024 prefill? | `rocprofv3` kernel trace grouped by family/role | >=95% of kernel time classified |
| B2 source map | Which source files/functions emit MMQ, rocBLAS GEMM, attention, q8 quant, dequant, convert, SwiGLU? | llama.cpp source references and build flags | high-share kernels mapped to source path/function |
| B3 MMQ anatomy | What is the quantized MMQ/matmul inner loop? | source + disasm: packed format, q8 producer, accumulation type, MFMA/MMQ path, scales/mins | no decode-MMVQ assumptions; prefill MMQ is mapped directly |
| B4 launch/resource policy | What launch shapes and resources do hot MMQ kernels use? | kernel metadata: block size, VGPR/LDS/scratch, calls, avg time | no unexplained high-share hot kernel remains |
| B5 memory/dataflow | Is llama pp512 limited by packed-weight reads, q8 activation, scales, LDS/shared memory, or tensor op issue? | counters or trace-derived memory/occupancy evidence | bottleneck class named per high-share family |
| B6 rocBLAS GEMM role | Which dense GEMMs remain and why are they not MMQ? | rocBLAS kernel names, shapes if recoverable, share/time | dense-GEMM bucket separated from MMQ bucket |
| B7 attention role | Why is attention small at pp512? | attention kernel family and time share, pp1024 comparison | pp512 attention not treated as primary bottleneck unless share changes |
| B8 tinygrad contrast | What does llama do that tinygrad does not? | side-by-side table against tinygrad/Tensile: packed quant, q8 lifecycle, library GEMM, graph/runtime | row names concrete transfer substrate, not vague "llama is faster" |
| B9 transfer row | Which llama prefill primitive can transfer? | row with correctness/perf gate and risk | row is either machine-searchable after substrate or closed |

llama output artifact:

- `docs/prefill-llama-strong-audit-result-20260620.md`
- `bench/prefill-strong-audit/llama/*.json`

llama strong-audit pass condition:

- The pp512 `quantized MMQ/matmul` bucket is split into source-visible primitive rows.
- The audit distinguishes decode MMVQ from prefill MMQ.
- We know whether llama's prefill advantage is primarily packed quantized matmul, library GEMM selection, graph/runtime
  packing, or some mix.
- We can say which primitive rows tinygrad would need before machine search is meaningful.

## Track C - Combined Transfer Matrix

The final combined document must produce one matrix across both fast implementations:

| Primitive | Tensile evidence | llama prefill evidence | tinygrad gap | Portable abstraction | Search-ready? |
|---|---|---|---|---|---|
| tensor-core dense GEMM | required | rocBLAS bucket | partial | WMMA/MFMA/MMA/DPAS | only after dataflow rows exist |
| shared/LDS staging | required for Tensile | audit required for MMQ/GEMM | missing/partial | shared-memory operand reuse | no |
| K-loop prefetch/pipeline | required for Tensile | audit required | missing | overlapped load/compute | no |
| packed quantized matmul | not the Tensile route | likely central | missing for prefill | packed weights + q8 activation MMQ | no |
| q8 producer lifecycle | not central | audit required | decode-only partial | activation quant contract | conditional |
| graph/runtime launch efficiency | HCQGraph route works | llama graph likely relevant | partial | low-overhead replay | conditional |
| attention prefill | not central at pp512 | small at pp512 | secondary | flash/SDPA tiles | not first |

Combined output artifact:

- `docs/prefill-tensile-llama-transfer-matrix-result-20260620.md`
- `bench/prefill-strong-audit/transfer_matrix.json`

Combined pass condition:

- Every high-share pp512 primitive maps to one of: transfer candidate, closed/low-EV, policy-only external artifact,
  or blocked by missing tooling.
- Machine-search readiness is explicit. A row can be searched only if its primitive representation, correctness
  oracle, resource constraints, and performance gate exist.

## Execution Phases

| Phase | Name | Output | Gate |
|---|---|---|---|
| PSA-0 | provenance freeze | commits, build flags, model, GPU, ROCm, clock baseline | no stale artifact ambiguity |
| PSA-1 | top-line reproduction | pp512/pp1024 WMMA/Tensile/llama with clock telemetry | numbers match current matrix within noise |
| PSA-2 | Tensile static audit | selected solution parameters + disasm ledger | A0-A5 complete |
| PSA-3 | Tensile dynamic audit | counters/variant contrasts + tinygrad authority contrast | A6-A9 complete |
| PSA-4 | llama prefill trace | pp512/pp1024 trace ledger and source map | B0-B2 complete |
| PSA-5 | llama MMQ anatomy | source/disasm/resource/counter audit of high-share MMQ | B3-B6 complete |
| PSA-6 | lifecycle/attention audit | q8/dequant/convert/SwiGLU/attention roles | B7-B9 complete |
| PSA-7 | combined transfer matrix | transfer rows + search readiness | C matrix complete |
| PSA-8 | closeout decision | native/search/artifact recommendations | no implementation starts without named row |

## Tooling To Reuse

Existing:

- `extra/qk_prefill_clock_threeway.py`
- `extra/qk_tensile_contract.py`
- `extra/qk_tensile_shape_matrix.py`
- `extra/qk_tensile_variant_ablation.py`
- `extra/qk_amd_bb5a10_tensile_layout_audit.py`
- `extra/qk_prefill_primitive_pmc.py`
- `extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py`
- `/home/ubuntu/env/llama.cpp/build/bin/llama-bench`
- `/opt/rocm-7.2.4/bin/rocprofv3`

Likely new tooling:

- `extra/qk_prefill_tensile_strong_audit.py`
- `extra/qk_llama_prefill_strong_audit.py`
- `extra/qk_prefill_transfer_matrix.py`

## Stop Rules

- Do not reopen clock as a cause unless a new three-way clock matrix contradicts the committed result.
- Do not run blind BEAM/search for prefill before the primitive rows exist.
- Do not call standalone LDS a transfer. LDS only matters with layout, K-loop overlap, waits, and resources.
- Do not use llama decode MMVQ as a substitute for llama prefill MMQ evidence.
- Do not compare raw isolated TFLOPS to e2e pp512 without an Amdahl and role-share table.
- Do not promote a native backend task unless it names the row it will implement and the correctness/performance gate.

## Final Decision This Scope Enables

The audit should end with one of four outcomes:

| Outcome | Meaning |
|---|---|
| `STRONG_AUDIT_COMPLETE_SEARCH_READY_ROWS` | at least one primitive row is represented well enough for machine search or native build |
| `STRONG_AUDIT_COMPLETE_ARTIFACT_ONLY` | fast route is understood, but native transfer is not worth the substrate cost |
| `TENSILE_STRONG_LLAMA_BLOCKED` | Tensile is implementation-grade, llama prefill remains source/trace blocked |
| `AUDIT_BLOCKED_BY_TOOLING` | required disasm/counter/source mapping cannot be obtained locally |

The desired result is not "build Tensile." The desired result is a portable primitive map that says exactly what a
different GPU backend must represent before search can be expected to find llama/Tensile-class prefill performance.
