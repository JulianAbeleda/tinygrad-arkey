# AMD Decode Packed-Load Lowering

Date: 2026-06-13

Status: next design surface. No runtime implementation in this document.

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

2. **Wider vector loads**
   Preserve `uint2`/`uint4`-style grouped loads in generated AMD C where
   alignment and layout permit it. A candidate must report generated source load
   width.

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
- No isolated `v_dot4` peephole as the next default task.
- No WMMA for batch-1 decode unless source inspection proves llama.cpp uses it
  in the decode path on gfx1100.

## Relationship To Ansor

This is still directionally Ansor-style, but one layer lower than the failed
schedule knobs. Ansor needs a meaningful search space. For packed GGUF formats,
that means first making the packed memory representation explicit enough for the
generator to emit load-layout choices. Once the memory-access lowering is
represented semantically, search can operate over a real axis instead of
reshuffling the current kernel.
