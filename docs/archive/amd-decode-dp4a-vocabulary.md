# Phase D — teaching tinygrad the int8/DP4A vocabulary (so search can reach the decode GEMV)

Date opened: 2026-06-15
Goal: close the decode gap the *tinygrad-pure* way — expand the compiler's generative vocabulary so
the int8/DP4A Q4_K GEMV becomes something **search can find**, instead of something we hand-patch via
the `Ops.CUSTOMI` inline-asm escape hatch.

## The gap, confirmed in code

- The decode GEMV runs at ~58 tok/s (~32% of HBM peak) vs llama.cpp ~104 (~58%). The win llama.cpp
  has is a different *structure*: int8-quantized activations (q8_1) + **int4×int8 dot products
  (`v_dot4`/DP4A)** accumulated in int32, then scaled.
- **The instruction is known to tinygrad's assembler** (`V_DOT4_I32_I8` / `V_DOT4_U32_U8` in
  `runtime/autogen/amd/.../ins.py`) and we proved it runs on gfx1100.
- **But there is no codegen pattern that emits it.** Grep of `tinygrad/codegen/` + renderers: no
  lowering rule, no opt, no UOp turns a high-level int dot into `v_dot4`; the TC framework
  (`codegen/opt/tc.py`) is fp16/bf16 matrix-WMMA only (no `iu8`). The ONLY way our experiments used
  DP4A was hand-written inline asm via `Ops.CUSTOMI` (`extra/q4_k_gemv_primitive.py`:
  `asm volatile("v_dot4_u32_u8 ...")`). So search can't reach it — search tunes fp tiling knobs over
  the dequant->fp-multiply->reduce chain; the win is a different instruction + dtype it cannot
  introduce. **That vocabulary gap (not a search failure) is why decode lags and the GEMV opt space
  was flat (Phase M0).**

## Why DP4A is LIGHTER to add than WMMA (the key design point)

WMMA is warp-collaborative matrix multiply: the `TensorCore` dataclass carries M/N/K dims, threads,
elements-per-thread, and swizzles, and is plumbed through `_apply_tc_opt` -> `SHAPED_WMMA` ->
`lower_shaped_wmma` -> `Ops.WMMA` -> a renderer intrinsic. **DP4A is none of that** — it is a
single-lane vector op: `int32 acc += dot4(int8x4 a, int8x4 b)`. So it does NOT need a `TensorCore`
entry or swizzles. It needs only: a UOp, a fold-pattern, a one-line renderer rule, and a search
action. Much smaller surface than WMMA.

## Touch points (grounded in the WMMA precedent)

- `tinygrad/uop/ops.py` — add `Ops.DP4A` (srcs: packed_a:uint32, packed_b:uint32, acc:int32 -> int32).
  Spec rule in `uop/spec.py` (mirror the `Ops.WMMA` rule at `spec.py:113`).
- `tinygrad/codegen/` — a fold PatternMatcher (analogous to `rangeify.py::lower_shaped_wmma`,
  `rangeify.py:25`) that recognizes the idiom `REDUCE(ADD, MUL(a.cast(int32), b.cast(int32)))` over a
  contiguous 4-wide int8 axis (UNROLL=4), packs the 4 int8 lanes into a uint32, and emits `Ops.DP4A`.
- `tinygrad/renderer/cstyle.py` — one `string_rewrite` rule (mirror the WMMA emit at `cstyle.py:62`):
  `(UPat(Ops.DP4A, name="x"), lambda ctx,x: f"__builtin_amdgcn_sdot4({a},{b},{acc},false)")`
  (signed) / `udot4` (unsigned). No prefix function body needed — it is a compiler builtin (unlike
  WMMA which declares a `__WMMA_...` helper).
- `tinygrad/codegen/opt/search.py` — add a search action so BEAM/the cost-model loop can CHOOSE it
  (e.g. an `Opt(OptOps.DP4A, axis, ...)` analogous to the TC action at `search.py:22`), OR make the
  fold automatic when dtypes are int8 and the reduce axis is divisible by 4. The whole point: it must
  be *search-reachable*, not hand-applied.
- Model/graph: express activation -> q8_1 (int8 + per-block scale) quantization in the Q4_K matmul so
  the dot operands are int8. This is ordinary elementwise ops tinygrad can already emit; only the dot
  needs the new vocabulary.

## Phases (cheap, make-or-break first)

**D0 -- the ceiling probe (make-or-break, do FIRST, before any compiler change).** Before investing
in a core codegen feature, verify the *lever is real*: build/measure the best HAND-WRITTEN int8/DP4A
Q4_K decode GEMV (reuse the existing `CUSTOMI` DP4A plumbing) end-to-end and measure decode tok/s on
gfx1100. Pre-registered: if hand-DP4A approaches ~llama.cpp (~90-104 tok/s) -> DP4A is THE lever,
green-light the codegen work (D1-D4). If it plateaus well below (e.g. <75) -> DP4A alone is
insufficient (occupancy / access patterns / norm fusion also matter); rescope or stop. This is the
roofline discipline (M0 lesson): do not build a compiler feature for a lever that isn't the
bottleneck. (We are at ~58 now; first confirm whether that path even uses DP4A in the hot loop.)

**D0 -- RESULT (2026-06-15): gate NOT cleared. Phase D D1-D4 should NOT proceed.** `dp4a-d0/RESULT.md`.
Microbench (device Q4-GB/s): best int8 variant `intdot` (int8 MAC) = 242 on ffn_gate vs fp 173 (+40%)
but ~50% of llama.cpp's ~470-500; the EXPLICIT DP4A (`vdot`, the `v_dot4` asm Phase D would teach the
codegen) is the SLOWEST (35, asm volatile blocks scheduling). End-to-end intdot wired into decode =
28 tok/s, REGRESSED below fp (58) on unfused per-layer q8_1 quant. Optimistic fused ceiling ~81 tok/s
(~78% of llama.cpp) -- improvement, NOT parity. Diagnosis (consistent with M0): decode is
MEMORY/occupancy-bound; DP4A accelerates COMPUTE, the wrong axis -- which is why explicit DP4A is
slowest and int8 barely helps. llama.cpp's win is memory-side engineering + the int8 ACTIVATION (fewer
bytes), NOT the dot instruction. DECISION: do not build the DP4A codegen vocabulary; it optimizes the
wrong thing. The D0 gate caught the wrong lever before any compiler change (roofline discipline). The
phases below (D1-D4) are RETAINED for the record but are NOT to be built per this verdict.

**D1 -- the fold pattern.** Implement the int8-dot -> `Ops.DP4A` recognizer + int8x4->uint32 packing.
Unit-test on a tiny int8 dot. Risk: getting the compiler to *generate* the exact 4-wide int8 reduce
shape the pattern matches (needs UNROLL=4 + int8 dtypes + packing); the matcher may be fragile (the
SHAPED_WMMA drift lesson) -- gate on a minimal green example first.

**D2 -- the renderer emit.** Add the `Ops.DP4A` -> `__builtin_amdgcn_sdot4/udot4` string_rewrite;
confirm gfx1100 compiles + runs correct vs an fp reference.

**D3 -- make it search-reachable.** Add the search action / automatic trigger so BEAM + the N-loop
cost model can select DP4A; express the q8_1 activation path in the Q4_K matmul graph.

**D4 -- end-to-end + measure.** The decode GEMV now uses *search-found* (not hand-asm) DP4A. Measure
decode tok/s vs the D0 hand-written ceiling and vs llama.cpp; confirm the kernel came from codegen,
not `CUSTOMI`. Correctness-gated (q8_1 changes numerics slightly, as it does in llama.cpp).

## Pre-registered honesty + boundary

- D0 gates everything: if the hand-written DP4A ceiling isn't competitive, the codegen work is moot
  and we say so.
- This is a tinygrad-CORE change (`uop/`, `codegen/`, `renderer/`), not an `extra/` experiment -- the
  philosophically-pure "expand the Typesetter's vocabulary so the Judge can reach it" path, and
  upstreamable. It directly tests whether the decode gap is a *vocabulary* gap (closeable by codegen +
  search) or something deeper.
- Success = search produces a DP4A decode GEMV competitive with llama.cpp, with NO hand-written kernel
  in the hot path. That would be the first end-to-end vindication of the search philosophy on the
  decode kernel that currently out-runs it.
