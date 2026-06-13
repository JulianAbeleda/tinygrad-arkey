# AMD Decode Packed QK Semantic Op

Date: 2026-06-13

Status: phase-1 contract added; no runtime lowering or speed claim.

## Purpose

The raw packed-tile custom path answered one construction question and closed a
performance path:

- yes, AMD can emit target `global_load_b128` from the `tg_uint4` spelling;
- no, a full raw `Ops.CUSTOM` Q4_K GEMV body is not a useful integration
  surface because it hides the loop body from tinygrad and collapses the
  scheduled 32-lane shape into a workgroup-size-1 kernel.

The next step is therefore not another custom kernel. The next step is a
compiler-visible semantic operation that exposes packed QK load/decode/dot to
tinygrad while leaving row/K scheduling visible.

## Hypothesis

The remaining gap to llama.cpp is a packed-weight load-efficiency gap. The
missing compiler object is not "a faster full GEMV kernel"; it is a
schedulable packed block-dot operation:

```text
QK_BLOCK_DOT(packed Q4_K block words, fp16 activation block, row, k_block)
  -> float32 contribution
```

The semantic op may hide format-local details, but it must not hide the loops
that BEAM/tinygrad needs to schedule.

## Contract

The first contract is Q4_K only and is recorded by
`extra/qk_semantic_op.py` in
`bench/qk-packed-semantic-op-20260613/semantic-op-contract.md`.

Allowed inside the op:

- Q4_K scale/min unpack;
- Q4_K nibble extraction;
- lane mapping within one 256-element block;
- target load intrinsic or vector-load spelling.

Forbidden inside the op:

- row loop;
- K-block loop;
- split-K partial output layout;
- partial reduction kernel;
- full GEMV kernel body;
- runtime policy selection.

This is the key lesson from the failed `tile_custom` path: vector loads are not
enough if the lowering hides the scheduling surface.

## Initial Lowering Target

The first implementation target is an AMD renderer/core lowering, not
`Ops.CUSTOM` full-kernel source.

The expected shape is:

1. Keep the existing v1-style row/K/split loops in tinygrad UOps.
2. Replace only the per-block packed load/decode/dot fragment with a semantic
   QK operation.
3. Lower that fragment late in the HIP renderer or a PatternMatcher-style core
   pass.
4. Preserve v1-like scheduling evidence: target workgroup shape must not become
   workgroup-size 1.

TC/WMMA is the closest tinygrad precedent: a small hand-seeded semantic
capability becomes schedulable by the existing optimizer. The goal is not a
large handwritten kernel library.

## Gates

No runtime path is accepted until all gates pass:

- reference unpack is bit-exact against `extra.qk_layout.q4_k_reference`;
- AMD GEMV numeric compare passes with random fp16 activations;
- generated source records the intended packed-load spelling;
- DEBUG=7 target disassembly contains wide/coalesced load evidence;
- scheduler shape preserves v1-like row/K parallelism;
- target instruction count does not exceed 2x v1 without measured gain;
- repeated dominant-shape microbench median gain is at least `10%`;
- full-decode confirmation accepts on 8B and 14B;
- greedy output A/B passes.

32B remains optional and should run only after the 8B/14B gate shows promise.

## Compile Gate Result

`bench/qk-block-dot-compile-gate-20260613/` records the first minimal
`QK_BLOCK_DOT` compile gate.

Result: `qk_block_dot_compile_gate_passed_compile_shape`.

The experimental op:

- passes the local reference unpack and AMD GEMV numeric gate for the fixed 8B
  `blk.0.ffn_gate.weight` shape;
- preserves the v1 32-lane scheduled shape: workgroup `32`, `gidx0=2`,
  `lidx0=32`;
- emits the intended `tg_uint4` source load inside the block-local op;
- produces target wide-load evidence: `5` `global_load_b128` instructions
  versus `1` in the v1 partial kernel;
- stays within the pre-registered target-body gate: `333` parsed target
  instructions versus `296` for v1.

This is still compile-shape evidence only. It authorizes a repeated
dominant-shape microbench. It does not authorize runtime integration, full
decode, generated-policy promotion, or 32B work.

## Stop Rule

Stop this line if the first-class op cannot preserve both:

- target wide/coalesced packed loads; and
- schedulable row/K parallelism.

If preserving one destroys the other, this C-style renderer layer is not enough
to close the llama.cpp gap. The next choice would be deeper AMD assembly work or
a larger compiler project, not another schedule/codegen family.

## Current Artifact

`bench/qk-packed-semantic-op-20260613/` is design-only evidence:

- `8` Q4_K contract rows across 8B and 14B;
- `6` Q6_K rows skipped by rule;
- no runtime lowering;
- no benchmark;
- no full-decode run.

The next implementation commit should be a minimal compile gate for
`QK_BLOCK_DOT`, not runtime integration.

That minimal compile gate now exists in
`bench/qk-block-dot-compile-gate-20260613/`. The next implementation step is a
repeated microbench for the same fixed 8B dominant shape, gated by the artifact
summary. Full decode remains blocked until the repeated microbench clears the
pre-registered gain threshold.
