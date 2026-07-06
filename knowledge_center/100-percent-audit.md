# 100% Audit — Definition of Done + Gap Tracker

Snapshot date: 2026-07-06. Perf figures are from project notes/memory and may drift as the lowering refactor lands —
re-measure before trusting. This is the standing "definition of done" for a **quant-grade fast, minimal, AOT
transformer inference engine** on AMD gfx1100 (CUDA/NV/Metal future). Companion to
[minimization-principles.md](minimization-principles.md).

## What 100% means

A single-stream, latency-first inference engine where: the math runs at/near roofline, there is zero wasted motion
between kernels, the whole thing is compiled ahead-of-time to a data artifact and replayed by a tiny runner, the
kernels are machine-searched (not hand-tuned), and the authored surface contains only what's essential. Small and fast
are the same target — both come from "everything static, nothing dynamic at runtime."

## Scorecard

| Axis | Status | Have | Missing |
|---|---|---|---|
| **A. Kernel quality** | 🟡 PARTIAL | decode GEMV generated (G3); flash live-split; 8B gen_sched ~40 TFLOPS; 808 tok/s fused Q4_K WMMA (handwritten, blocked) | **14B prefill ~365 tok/s VALU-bound vs llama.cpp ~1849 (~5x)**; generated fused-dequant->fp16-LDS->WMMA substrate UNBUILT |
| **B. Overhead elimination** | 🟢 HAVE | TinyJit capture + HCQ graph replay; preallocated KV; GPU-side sync | (megakernel lives in axis A/D) |
| **C. AOT / compile-runtime split** | 🔴 MISSING | graph captured fresh at warmup | offline serialize of graph+kernels; standalone tiny runner (no compiler); serialize-and-replay probe |
| **D. Search** | 🟡 PARTIAL | BubbleBeam/FutureSight; gen_sched asm substrate | no e-graph/equality-saturation; nothing searches a megakernel |
| **E. Mechanical sympathy** | 🟡 PARTIAL | native PMC sampler; roofline/bound analysis | `clock_pin` deleted (jitter control gone); no speculative decode |
| **F. Minimization discipline** | 🟡 PARTIAL | authored/generated boundary (@generated + sz.py); decode generated; autograd removed | prefill defaults still handwritten (lowering refactor mid-flight); frontend op-prune; compile/runtime split |

## Ranked gaps (what to build, in order)

### [ ] 1. Close the 14B prefill kernel-quality gap  (axis A — highest impact)
The ~5x deficit vs llama.cpp is compute (VALU-bound), and nothing else on this list fixes it. Build the **generated
fused-dequant -> fp16-LDS -> WMMA** prefill substrate (the named solve in the project notes). Megakernel/AOT do NOT help
this — it's raw math throughput.
Done when: 14B prefill closes materially toward roofline / llama.cpp, via a *generated* (not handwritten) kernel.

### [ ] 2. Finish the lowering refactor: prefill -> generated  (axis A + F — in flight, Codex)
Until prefill kernels are generated (not the blocked handwritten `tile4x4`/packed-load defaults), A and F stay PARTIAL.
Done when: `pure_kernel_surface_audit` shows prefill defaults as generated/L3+; guard has no handwritten prefill default.

### [ ] 3. Megakernel: fuse the decode step into one persistent kernel  (axis B/D — latency frontier)
You have static replay but still launch N kernels/token. Fuse decode into ~1 persistent kernel (warp-specialized
producer/consumer, pipelined). Uniquely: **search** it with BubbleBeam rather than hand-write. Big batch-1 latency win;
does NOT fix prefill compute.
Done when: decode step replays as ~1 launch/token, latency measured below the multi-launch baseline.
**Scope:** `docs/generated-megakernel-decode-scope-20260706.md` (generated-only, reuses substrate, phased).
Refs: Hazy Research "No Bubbles"; Mirage (MPK); ETC dynamic megakernel; Ada-MK searched megakernel; AMD MI300X monokernel.

### [ ] 4. AOT serialize + tiny runner  (axis C — the smallest-shipped-thing unlock)
Capture exists (TinyJit); persist does not. First: the **serialize-a-captured-HCQ-graph-and-replay-it-in-a-fresh-process
-without-scheduler/codegen probe** — it measures whether the true runtime floor is ~5k or ~10k. Then a standalone runner.
Done when: a captured graph round-trips to disk and replays in a process that never imports codegen/schedule.

### [ ] 5. Restore mechanical-sympathy latency tools  (axis E)
`clock_pin` was deleted in the scorched-earth pass — it's a tail-latency/jitter tool, not research. Un-delete or
re-implement clock pinning for deterministic single-stream latency. (Speculative decode = optional throughput lever.)
Done when: GPU clocks can be pinned for a run; p99 decode jitter measured.

### [ ] 6. Technique upgrade: equality-saturation search  (axis D — later, non-blocking)
Replace ad-hoc BubbleBeam passes with rules-as-data + e-graph saturation (egg/egglog/TENSAT) — smaller optimizer, better
search space. Future research; not on the critical path.

## The critical path (one line)

**Generated prefill + fused-dequant WMMA (closes the 5x perf gap) -> serialize/replay probe (proves the AOT floor) ->
megakernel decode (latency frontier).** Everything else is supporting.
