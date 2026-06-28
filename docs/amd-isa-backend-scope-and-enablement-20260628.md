# Native AMD/rdna3 ISA backend — scope + how it enables pure machine search (2026-06-28)

The terminal of the scheduler investigation (`decode-codegen-scheduler-arm-{a,b}-result-20260628.md`): closing
decode attention to owned-quality via pure machine search requires a **native AMD ISA backend** — tinygrad doing its
own UOp→`Ops.INS` instruction selection + register allocation + scheduling + assembly, bypassing LLVM. This scopes
that build (staged, templated on `isa/x86.py`) and walks through why it is *the* enabling capability.

## How this enables pure machine search (the why)

**Today's ceiling.** The machine (codegen + BubbleBeam) can *generate* correct AMD kernels, but for the final mile —
instruction selection, register allocation, and **scheduling** — it hands source (HIP-C / LLVM-IR) to LLVM. LLVM owns
those decisions. Measured consequence (Arm A): every UOp reorder we feed LLVM stays in its **42–52 `s_waitcnt`
envelope**; owned hand-ASM is **21**. So the generated decode tile is capped at **33.7%/7.1% of owned**, and the
search cannot touch the last lever (the schedule) — it's LLVM's, not the machine's. The result: AMD pure-search
must route among hand-written kernels, because the machine can't schedule competitively.

**What the backend changes.** A native UOp→`Ops.INS`→`assemble_linear` backend gives tinygrad the **entire**
pipeline — isel, regalloc, scheduling, assembly — with **no LLVM in the loop**. Then:
1. **The schedule becomes a machine-owned, searchable decision.** The `Inst` stream is tinygrad's; the
   `qk_asm_scheduler` reg-DAG + `renderer/amd/schedule.py` latency metadata can reorder it, and BubbleBeam can
   search over scheduling / regalloc / pipeline-depth choices — the same levers the hand-ASM author uses.
2. **Generated kernels can reach hand-ASM quality** (owned's 21 waitcnt), because the machine now controls the same
   instruction stream owned does — not a source string LLVM re-optimizes.
3. **The hand kernels get retired.** The owned decode tile and prefill GEMM's residual become generated+scheduled
   assembly that matches them → the pure-default retirement gate (Phase 5) becomes reachable for AMD.
4. **It generalizes.** One backend serves decode attention, prefill GEMM, and every future op; the search's output
   quality becomes tinygrad's own, not comgr/LLVM's.

In one line: **the AMD ISA backend moves the final mile from LLVM into the machine's searchable control** — the
single capability that lets pure machine search produce competitive AMD kernels instead of routing among hand ones.

## Existing substrate (we are NOT starting from zero)
- **Shared framework**: `tinygrad/renderer/isa/__init__.py` — `ISARenderer` base + `IselContext` + regalloc plumbing
  (the regalloc *algorithm* lives in `codegen/__init__.py:210`, `pm_regalloc_rewrite`; the backend supplies the
  arch hooks). A backend provides: `pre_isel_matcher`, `isel_matcher`, `post_regalloc_matcher`, and
  `stack_pointer/copy/spill/fill/asm_str`.
- **Template**: `tinygrad/renderer/isa/x86.py` (905 lines) — the complete worked example of those matchers + hooks.
- **AMD register model**: `tinygrad/renderer/amd/dsl.py` — `Reg` (unified 0–511: SGPR 0–105, VGPR 256–511, slicing,
  neg/abs/hi).
- **AMD instruction set**: `tinygrad.runtime.autogen.amd.rdna3.ins` — **1357** encodable `Inst` types (v_*/s_*/ds_*/
  global_*).
- **Assembly + ELF**: `tinygrad/renderer/amd/elf.py:assemble_linear` — already encodes `Ops.INS` → runnable ELF +
  kernel descriptor (vgpr/sgpr/lds scan). The `do_assemble` codegen path (`codegen/__init__.py:239`) is wired.
- **Scheduler precedent**: `extra/qk_asm_scheduler.py` (reg def/use DAG over `list[Inst]`, fence-region reorder,
  identity + reorder-correctness proofs, inc0–3 tests) + `renderer/amd/schedule.py` (latency-class/wait-group/
  LDS-stage metadata).

## Staged build (each increment: gate before the next)

**Inc 0 — minimal end-to-end path (THIS increment).** New `tinygrad/renderer/isa/amd.py` with `AMDISARenderer(ISARenderer)`:
isel for the smallest viable op set (`DEFINE_GLOBAL`/index/`LOAD`/`STORE`, `ADD`/`MUL`, `SPECIAL` workitem), the
`copy/spill/fill/asm_str` hooks, wired into the AMD device renderer list (selected by `DEV=AMD:ISA`). **Gate**: a
trivial kernel (`out[i] = a[i] + b[i]`, or a single elementwise) compiles via `assemble_linear` and **runs
numerically correct** on gfx1100. This proves the framework + AMD register model + Inst emission + ELF all work
end-to-end. (Mirrors `qk_asm_scheduler` inc-0's "faithful + correct under identity" bar, at the renderer level.)

**Inc 1 — op coverage.** Casts, more ALU (sub/div/max/exp2/fma), compare/select, the `RANGE`/`END` reduce loop,
gated load/store. Gate: a small GEMV/reduction kernel runs correct; ISA diff sane.

**Inc 2 — the decode tile's special ops.** `fdot2`→`v_dot2acc_f32_f16`, the cross-lane reduce→`ds_bpermute`, LDS
staging→`ds_load/ds_store` + `DEFINE_LOCAL`, the online-softmax ALU. Gate: the block tile compiles via the ISA
backend + `BLOCK_TILE_MICROGATE_PASS` (token-correct).

**Inc 3 — the scheduler.** Mature `qk_asm_scheduler` + `schedule.py` into a latency/modulo scheduler on the `Inst`
stream (consumer-only `s_waitcnt`, load/reduce interleave, cross-iteration pipelining). Gate: block-tile `s_waitcnt`
drops toward owned's 21; route-bound W==D rises from 35.0/6.7.

**Inc 4 — search-bind.** Lift scheduling/regalloc/pipeline-depth into BubbleBeam as searched decisions; generality
proof on prefill GEMM. Gate: Phase-3 purity gate (generated route ≥ promotion threshold of owned, owned retired to
fallback).

## Constraints
Default-off / opt-in renderer (`DEV=AMD:ISA`); shipped default (HIPRenderer) + owned route byte-identical until a
gate promotes; do not edit `tinygrad/runtime/autogen/**`; correctness-first (every increment gated, abort only on
correctness); bracketed-prefix commits. This is a multi-increment capability — each increment is independently
useful and gated; inc-0 is the foothold that proves the path is real.
