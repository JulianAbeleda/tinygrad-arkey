# AMD Decode Packed-Load Lowering

Date: 2026-06-13

Status: raw custom semantic Q4_K tile lowering closed; future work requires a
first-class packed QK semantic op or renderer lowering.

Update: Family C v0 has now been tested. It changed the expression shape to use
explicit packed-word lanes for Q4_K `ffn_gate`, but it tied on 8B/14B and
DEBUG=4 parsing still showed scalar `u32` loads. See
`bench/qk-ansor-transition-20260612/semantic-codegen-v3/verdict.md`. The
remaining packed-load work therefore needs hardware-counter profiling or a
deeper renderer/layout capability; do not broaden the v0 rewrite.

Update 2: the memory-access audit isolated and closed the next required source
capability. The normal UOp path now preserves a requested aligned
`uint32.vec(4)` global load/store on AMD, copies all lanes exactly, and DEBUG=4
source shows `unsigned_int4` vector pointer casts. See
`bench/qk-memory-access-20260613/audit.md`.

Update 3: Family C v1 attempted to use that source shape inside the real Q4_K
`ffn_gate` GEMV candidate. The 8B/14B harness rejected it as invalid before
timing: the kernel can request a `uint32x4` load, but tinygrad's current
tensor/UOp shape rules cannot yet scalar-extract or vector-arithmetically
consume that loaded value inside this GEMV without verifier/codegen failures.
See `bench/qk-ansor-transition-20260612/semantic-codegen-v4/verdict.md`.

Update 4: the next representation layer is now explicit as `PackedQKTile`.
`extra/qk_packed_tile.py` records Q4_K/Q6_K block layout, storage dtype, legal
load tiles, alignment, and search axes. Family C v4 candidate artifacts now
record the legal Q4_K `u32x4_aligned` load tile (`32` q-values per vector load)
instead of carrying the vector-load premise as prose. See
`docs/amd-decode-packed-qk-tile-design.md`. This is not a speed claim; it is the
descriptor layer needed before another vector-load GEMV construction attempt.

Update 5: the `PackedQKTile` consumption probe has run. Normal UOps still cannot
consume the Q4_K `uint32x4` load: scalar lane extraction through `GEP` fails the
verifier, and vector integer arithmetic fails shape validation. A custom
semantic kernel can consume the same tile shape: DEBUG=4 source parsing sees
`vector_u32x4`, and the kernel exactly loads `tg_uint4`, indexes lanes, unpacks
low/high Q4 nibbles, and accumulates a small dot. See
`bench/qk-packed-tile-consumption-20260613/README.md`. Verdict:
`semantic_custom_op_required`; do not run microbench or full decode until a
first-class packed QK load/decode/dot lowering exists.

Update 6: the first real custom semantic Q4_K tile consumer now exists as
`q4k_gemv_tile_custom_partial_kernel`. It keeps fp16 activations, emits
`vector_u32x4` source, and passes AMD Q4_K GEMV correctness for `parts=1` and
`parts=4`. The full-shape microbench signal is positive but weak: 8B
`ffn_gate` improves `201.11 -> 215.60 Q4-GB/s` (`+7.20%`) and
`attn_output` improves `64.85 -> 68.63 Q4-GB/s` (`+5.83%`). This is below the
pre-registered `>=10%` microbench bar for full-decode promotion, so no runtime
integration or full-decode run was promoted. See
`bench/qk-packed-tile-lowering-20260613/README.md`.

Update 7: repeated 8B analysis now compares v1 partial vs `tile_custom` across
five Q4_K tensors with five runs each. Source-shape evidence is real: v1 parses
as scalar `u32`, while `tile_custom` parses as `vector_u32x4`. Performance does
not generalize: gains range from `-2.04%` to `+7.51%`, with median gain
`-0.36%`; only `ffn_up` is materially positive. Verdict:
`diagnose_only_not_promoted`. Do not run full decode or runtime integration from
this raw custom path. See
`bench/qk-packed-tile-lowering-analysis-20260613/README.md`.

Update 8: DEBUG=7 target-disassembly close-out explains that negative result.
`tile_custom` does emit real target `global_load_b128` instructions (`32` in the
target block versus `1` for v1), but it also becomes a workgroup-size `1` raw
custom kernel with a much larger target body (`1293` parsed target instructions
versus `296` for v1). The current v1 path keeps the 32-lane scheduled shape
(`amdgpu_flat_work_group_size(1, 32)`) and already gets some target wide-load
combining from the AMD compiler. Verdict:
`raw_custom_tile_path_closed_not_promoted`. See
`bench/qk-packed-tile-research-closeout-20260613/README.md`.

Update 9: the next semantic boundary is now defined but not implemented.
`docs/amd-decode-packed-qk-semantic-op.md` and `extra/qk_semantic_op.py` define
`QK_BLOCK_DOT`: a Q4_K block-local load/decode/dot operation that may hide
nibble unpacking and target load spelling, but must leave row, K-block, split-K,
and reduction scheduling visible. Artifact:
`bench/qk-packed-semantic-op-20260613/README.md`. This is a design-only
contract, not a runtime lowering or speed claim.

Update 10: the cheap three-way load diagnostic has run and rejects the
wide-load-only branch under AMD device timing. The first run correctly exposed
a `vector_load` authoring bug: vector lanes were not reduced back to scalar
partials before the scalar combine. After fixing that with scalar inline lane
reduction, `vector_load` passes correctness and runs with the schedulable
`LOCAL:0:32` path. It still loses: `349.25` device Q4 GB/s on 8B `ffn_gate`,
versus v1 at `382.01` (`-8.58%`). `tile_custom` is an opaque no-LOCAL control
and reaches only `36.99`. Artifact:
`bench/qk-threeway-load-microbench-20260613/README.md`. This supersedes treating
the earlier weak wall-time `tile_custom` smoke signal as actionable. The next
work is instruction/source/counter diagnosis or lower-level renderer-quality
lowering, not another raw vector-load retry.

## Problem

The accepted Q4_K/Q6_K primitive path is correct and substantially faster than
the fused generic graph, but the model-scope roofline still shows a large
same-byte gap to llama.cpp. The current generated shared-storage rows reach
`51-62%` of llama.cpp and only `27-38%` of RX 7900 XTX peak bandwidth by the
full-file proxy.

The rejected surfaces mostly changed loop shape:

- `parts` / `LOCAL`;
- `direct_out`;
- `row_upcast` / reduction unroll;
- row grouping;
- isolated q8/vdot arithmetic.

Those are not the right default lever for a bandwidth-bound batch-1 GEMV unless
they also make memory transactions more efficient.

## Hypothesis

The remaining decode gap is a packed-weight load-efficiency gap:

```text
Q4_K/Q6_K bytes are present in compact form,
but tinygrad's current lowering does not issue memory transactions as efficiently
as llama.cpp's MMVQ path on RDNA.
```

The next useful compiler surface is therefore not another schedule knob. It is a
semantic packed-load lowering that represents the packed quant block as a memory
object the renderer/search loop can reason about.

## Candidate Surface

Family C should be a memory-access family, not a compute/reduction family.

Candidate descriptors should expose:

- quant format: `Q4_K` or `Q6_K`;
- block byte layout and block-stride;
- logical `N,K` shape;
- role/tensor family;
- packed storage dtype and alignment;
- target load width, initially `uint32` and then wider vector loads if the
  renderer can preserve them;
- lane-to-packed-word mapping;
- whether q8_1 activation staging is used;
- whether packed-dot emission is used;
- correctness boundary: reference unpack, AMD GEMV, full-decode A/B.

Candidate lowerings should try to change one memory-access mechanism at a time:

1. **Coalesced packed-word loading**
   Adjacent lanes read adjacent packed words. Avoid per-lane scalar byte gathers
   from unrelated addresses.

   Family C v0 tried the cheap expression-level version of this idea by making
   each reduce lane own one packed `uint32` and unroll four nibbles. It did not
   produce a speedup or vector-load evidence.

2. **Wider vector loads**
   Preserve `uint2`/`uint4`-style grouped loads in generated AMD C where
   alignment and layout permit it. A candidate must report generated source load
   width.

   Current gate: raw aligned `uint32x4` global buffer copy is available through
   normal UOps, but consuming the loaded vector inside the Q4_K GEMV is blocked
   by vector-lane extraction / vector-shape support. Family C v1 therefore
   rejected at construction, not at performance.

   Follow-up gate: the first `QK_BLOCK_DOT` semantic-op compile gate passes in
   `bench/qk-block-dot-compile-gate-20260613/`. Unlike the rejected raw custom
   tile path, it keeps the v1 32-lane row/K scheduled shape and still emits
   target `global_load_b128` evidence. This authorizes a repeated
   dominant-shape microbench, not runtime integration or full decode.

   Repeated microbench result:
   `bench/qk-block-dot-microbench-20260613/` rejects the first lowering. On the
   full 8B `ffn_gate` tensor, v1 reaches `407.99` median device Q4 GB/s while
   `QK_BLOCK_DOT` reaches `285.01`, a `-30.14%` regression. Wider target-load
   evidence plus preserved scheduling is therefore not sufficient with this
   C-style block body.

   Three-way diagnostic:
   `bench/qk-threeway-load-microbench-20260613/` compares the current v1
   partial kernel, schedulable `vector_load`, and opaque `tile_custom` on the
   same dominant 8B `ffn_gate` tensor. It rejects load width alone:
   corrected `vector_load` passes and is a `-8.58%` device-time regression
   versus v1; `tile_custom` is a `-90.32%` no-LOCAL control.

3. **Activation staging only when it supports load efficiency**
   q8_1 staging is useful if it aligns the compute with packed dot and keeps ALU
   hidden under memory. It is not a standalone speed claim.

4. **Packed dot as an accessory**
   `v_dot4`/`sudot4` may be needed to keep dequant arithmetic cheap. Prior local
   attempts show it is not the primary lever by itself.

## Gates

Do not install a runtime path from this family until all are true:

- reference unpack correctness passes;
- AMD GEMV numeric correctness passes;
- generated source confirms the intended load width/coalescing change;
- dominant-shape microbench gain is strong enough to survive full-decode
  dilution, expected `>=10%` before a full-decode run is worth starting;
- full-decode confirmation rerun accepts on 8B and 14B;
- greedy A/B passes;
- storage deltas are recorded.

32B is optional and should run only after 8B/14B show promise.

## Non-Goals

- No more `parts`/`LOCAL` sweeps over the current primitive family.
- No row-group broadening.
- No direct-output retry.
- No broadening of the Family C v0 packed-word-lane rewrite.
- No isolated `v_dot4` peephole as the next default task.
- No WMMA for batch-1 decode unless source inspection proves llama.cpp uses it
  in the decode path on gfx1100.
- No full-decode promotion for this raw custom lowering unless a repeated
  dominant-shape microbench clears the `>=10%` bar or a core lowering/search
  integration changes the premise.
- No broadening of raw `Ops.CUSTOM` Q4_K tile consumers as a performance path:
  repeated analysis shows vector-source loads alone do not produce a general
  Q4_K win.
- No further raw `Ops.CUSTOM` Q4_K tile variants. DEBUG=7 close-out shows the
  raw custom path trades vector source loads for a single-work-item, opaque,
  much larger target kernel body.
- No further Family C variants through normal UOps. Future variants should
  consume `PackedQKTile` or its successor through a first-class packed-load op,
  renderer PatternMatcher, or similarly explicit semantic lowering rather than
  repeating expression-level rewrites.
- No full decode for `QK_BLOCK_DOT` from compile-shape evidence alone. The
  passed compile gate only moves the next step to repeated dominant-shape
  microbenching.
- No runtime integration or full decode for the current `QK_BLOCK_DOT`
  lowering. Its repeated microbench regressed against v1.
- No continuation of the wide-load-only `vector_load` / raw `tile_custom`
  branch. The three-way device-timed diagnostic rejects it.

## Relationship To Ansor

This is still directionally Ansor-style, but one layer lower than the failed
schedule knobs. Ansor needs a meaningful search space. For packed GGUF formats,
that means first making the packed memory representation explicit enough for the
generator to emit load-layout choices. Once the memory-access lowering is
represented semantically, search can operate over a real axis instead of
reshuffling the current kernel.
