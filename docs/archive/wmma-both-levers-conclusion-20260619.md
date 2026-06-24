# CONCLUSION — both WMMA levers explored; ~42 TFLOPS is the genuine pure-tinygrad RDNA3 ceiling

"Make it expressible" research, executed end-to-end (3 mapping agents + 2 worktree prototype/spike agents).

## Lever A (renderer addressing / base grouping) — REFUTED
Worktree prototype: grouped-base variant 41.1 vs 41.7 TFLOPS = **0.988x**, bitwise-identical, **ISA identical except
register renumbering**. clang ALREADY materializes base-VGPR pairs + uses the `offset:` immediate (incl negative
-4096); loop has 61 v_add not the ~128/160 CG-W claimed. **The addressing is already optimal; CG-W mis-read the
ISA.** Not the bottleneck.

## Lever B (software-pipelined K-loop) — collapse mechanism pinpointed; project-level; likely MOOT
- COLLAPSE ROOT CAUSE (proven on the live linearizer): a LOAD transitively depends on the RANGE UOp (INDEX/SHRINK
  contains RANGE), a HARD toposort edge (linearizer.py:44-46) forcing the load AFTER the RANGE. Even a synthetic
  k+1 "prefetch" load still carries the loop range (.ranges run_count 16, ops.py:410-427) -> structurally pinned
  inside the loop. This is exactly why the prior hand-UOp prefetch produced byte-identical ISA. NOT a CSE/renderer
  issue -- a core IR-structure invariant.
- TO MAKE EXPRESSIBLE: new `Ops.PREFETCH` (with .ranges NOT absorbing the loop RANGE) + deferred `Ops.WAIT` (wire
  the dead op) + a modulo-scheduling/loop-rotation pass (does NOT exist) + renderer async-load + s_waitcnt support.
  Three coupled subsystems touching the toposort invariant every kernel relies on. **Project-level: multi-day to
  multi-week, high-risk.**
- LIKELY MOOT even if built: (a) Route A A2 already HAND-implemented software pipelining (double-buffered, prefetch)
  in asm and got 24-32 TFLOPS -- BELOW the default LLVM's 42 (LLVM's instruction scheduler already hides latency
  well; the hand pipeline was single-wave-bound). (b) The real ceiling driver is OCCUPANCY (single-wave WMMA, no
  inter-wave latency hiding), and POWN's wave-count experiments (W4x2/W2x4/W4x4) ALL REGRESSED to 28-31 (VGPR/acc
  tradeoff). (c) LDS staging is IC-refuted on RDNA3 (~90% of 42 is global-direct WMMA scheduling). So pipelining
  specifically is not the missing piece -- occupancy-aware tiling/scheduling is, and that regressed when tried.

## Integrated honest verdict
**~42 TFLOPS (35% peak) is the genuine pure-tinygrad RDNA3 WMMA ceiling.** Neither dependency-free lever beats it:
A is refuted (addressing already optimal), B is project-level AND prior evidence (A2 hand-pipeline 24-32, POWN
occupancy regressions) says it won't close the gap. The 42->66 gap is a deeply-tuned, occupancy-balanced Tensile-
class kernel -- beyond tinygrad's codegen, and external Tensile DOESN'T transfer in-model (0.999x, the universal
integration-bottleneck pattern). 

**Prefill rests at PREFILL_V2 ~42 TFLOPS = ~47% of llama (auto-reproducible 1449 tok/s), with concrete-KV 1.24x
(byte-identical) as the shippable win.** The "make it expressible" path is closed for dependency-free pure-tinygrad;
the only frontier past 42 is a deep occupancy-aware codegen capability (the deepest wall, not a bounded arc) or an
external tuned kernel (doesn't transfer). Recommend NOT investing in Lever B.

## Files
wmma-make-expressible-scope, wmma-lever-a-grounding (refuted), this. Prior: POWN, route-a-a2-pipeline,
prefill-codegen-software-pipeline-result (CG-1).
