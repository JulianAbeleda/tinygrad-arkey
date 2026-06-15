# Phase L — adding the memory-latency-hiding vocabulary to tinygrad (exhaustive scope)

Date opened: 2026-06-15
Goal: give tinygrad's codegen the *latency-hiding* language a memory-bound decode GEMV needs — so the
~56%-of-llama.cpp decode gap becomes something **search can reach**, instead of a vocabulary it lacks.
This is the real frontier D0 pointed at (decode is memory/occupancy-bound; DP4A was the wrong, compute
lever). It is a major codegen subsystem, so the scope is exhaustive but the BUILD is gated on cheap
hand-probes (the D0 discipline that just saved us a wasted feature).

## What we know (grounded)

- Decode = 58 tok/s / 278 GB/s (~32% of 859 peak); llama.cpp = 104 / ~470-500 (~58%). M0: the GEMV is
  latency/occupancy-bound and FLAT to the existing opts.
- tinygrad's FULL opt vocabulary is 10 words: `TC, UPCAST, UNROLL, LOCAL, THREAD, GROUP, GROUPTOP,
  NOLOCALS, PADTO, SWAP` + wide loads + SYNCHRONOUS LDS (`DEFINE_LOCAL`+`BARRIER`). It covers tiling,
  ILP, parallelism, occupancy-via-locals — but has NO async copy / prefetch / double-buffering /
  explicit wave-occupancy control. These ARE standard subsystems in CUTLASS / MLIR / JAX-Pallas.
- ARCHITECTURE (decisive for the mechanism): AMD's default renderer is `HIPRenderer` (cstyle) ->
  HIP C -> comgr/LLVM. tinygrad emits NO `s_waitcnt` and NO async-copy; **LLVM does the instruction
  scheduling and latency hiding.** (There is also `AMDLLVMRenderer` (LLVM-IR) and an asm path for
  fuller control.) Implication: LLVM already hides *some* latency, so the marginal value of explicit
  pipelining is UNCERTAIN -> probe before building. Codegen passes: `codegen/late/`
  (expander -> devectorizer -> gater -> linearizer -> regalloc); a pipelining pass slots here.

## The missing vocabulary, decomposed (exhaustive), cheapest+most-likely FIRST

**L1 — explicit occupancy control (waves-per-EU / `__launch_bounds__` / register budget).**
The smallest addition and a strong candidate for THE lever: a memory-bound kernel saturates bandwidth
by having enough waves in flight to cover load latency. If the GEMV's occupancy is capped (e.g. by
register pressure from the dequant), no amount of tiling helps. Mechanism: emit
`__attribute__((amdgpu_flat_work_group_size(...), amdgpu_waves_per_eu(N)))` / `__launch_bounds__` from
`HIPRenderer` (`renderer/cstyle.py`); expose `N` as an opt. Touch points: cstyle kernel-attribute
emission + `OptOps.WAVES` + `apply_opt` + a `search.py` action. SMALL.

**L2 — software prefetch (issue iteration i+1's loads ahead, in program order).**
Restructure the reduce loop so the next tile's global loads are issued before the current tile's
dequant/compute; rely on LLVM + occupancy to overlap them. No new intrinsic — it is a UOp-graph / loop
transform (hoist loads via `AFTER` ordering). Touch points: a pass in `codegen/late/` (near the
expander/linearizer) + `OptOps.PREFETCH(depth)` + search action. MEDIUM. Risk: LLVM may already do
this; P0 must show hand-prefetch beats the current kernel.

**L3 — double-buffered LDS (ping-pong `DEFINE_LOCAL`, software-pipelined stage).**
Two LDS buffers; while WMMA/compute reads buffer A (tile k), loads fill buffer B (tile k+1); swap.
This is the CUTLASS multistage pattern, the real overlap of memory with compute. tinygrad HAS the
pieces (`DEFINE_LOCAL`, `BARRIER`, `AFTER`) but only SYNCHRONOUS single-buffer; needs the ping-pong
loop transform + relaxed barriers. Touch points: `codegen/late/` pipelining pass +
`OptOps.DOUBLEBUF(stages)` + search action. MEDIUM-HEAVY. (Most relevant to the BATCHED/prefill GEMM,
overlaps with W2's double-buffering TODO.)

**L4 — explicit asynchronous copy (`__builtin_amdgcn_global_load_lds` async global->LDS DMA).**
True async (CUTLASS `cp.async` equivalent; RDNA3 has `global_load_lds`). Issues the DMA, computes,
then waits — fully decouples load from compute. Heaviest: needs the builtin emitted from cstyle (or
drop to `AMDLLVMRenderer`/asm for explicit `s_waitcnt` control), a wait/barrier op, and spec/UOp
support. Touch points: `renderer/cstyle.py` (or `renderer/amd/`), a new `Ops.ASYNC_LOAD`/wait in
`uop/ops.py`+`spec.py`, the pipelining pass, `OptOps`, search action. HEAVY — only if L1-L3 prove the
lever but cap below llama.cpp.

For ALL of L1-L4 the make-it-findable piece is the same: an `OptOps` entry + `apply_opt` handling
(`codegen/opt/postrange.py`) + a `search.py` action, so BEAM and the N-loop cost model can SELECT it.
That is the whole point — the language must be searchable, not hand-applied.

## Phases (probe-FIRST, build only what's proven)

**L0 — make-or-break hand-probes (do FIRST, before any codegen change; mirrors D0).**
For each lever, hand-build the minimal version on the decode GEMV and measure end-to-end tok/s vs 58
and llama.cpp's 104:
- L0a: occupancy — recompile the current GEMV kernel with `__launch_bounds__`/waves-per-eu variants
  (via a kernel-attribute injection or the asm path); does forcing higher occupancy move tok/s?
- L0b: prefetch — hand-write a GEMV that issues the next blocks' loads ahead; does it beat current?
- L0c: double-buffer — hand-write a ping-pong-LDS GEMV; does it beat current?
Pre-registered: if the BEST hand-probe approaches ~90-104 -> that lever is real; build ITS vocabulary
(L1-L4) and make it search-reachable. If NONE moves decode materially (e.g. all stay ~58-70) ->
memory-latency-hiding is NOT the binding constraint either (the gap is a fundamental occupancy ceiling
on this dequant-heavy GEMV, or LLVM already hides the latency), record honestly and STOP — do not
build the subsystem. Combine with the D0 int8 finding: even the int8 ceiling looked ~81 tok/s, so
pre-register that decode PARITY may be unreachable on this kernel and the honest outcome may be a
located ceiling, not a win.

**L0 -- RESULT (2026-06-15): STOP. Latency-hiding is NOT the binding constraint; do not build Phase L.**
`latency-L0/RESULT.md`. Baseline fp decode 58 tok/s / 278 GB/s; llama.cpp 104 / ~470-500.
- L0a occupancy via existing parts/LOCAL: FLAT (~82-86 GB/s across parts 1->16; re-confirms M0).
- L1 occupancy-FORCING (patched HIPRenderer `amdgpu_waves_per_eu(N)`, end-to-end): WAVES_PER_EU=2->30.0,
  4->29.7, 6->30.0, 8->21.3 tok/s -- forcing higher occupancy REGRESSES (~halves) decode. The
  compiler's default occupancy is already optimal; more waves -> spills/less ILP. Decode is NOT
  occupancy-starved (a starved kernel would SPEED UP with more waves).
- L2 prefetch-via-ILP: the UPCAST/UNROLL/parts knobs are FLAT (M0); LLVM already extracts ILP; kernels
  already overlap end-to-end (278 aggregate vs ~85-173 per-kernel).
Verdict: none of the accessible latency-hiding levers move decode; forcing occupancy makes it worse.
Per the pre-registered gate -> STOP. The decode GEMV is bound by what the compiler+LLVM already balance
(occupancy + scheduling) plus the Q4_K DEQUANT ALU cost (~3862 vector ops/kernel, M0), not by hideable
memory latency. Both scoped decode levers are now probed NEGATIVE (DP4A compute = D0; latency-hiding =
L0). The residual ~2x gap is a dequant-cost + kernel-structure (count/fusion) problem the scoped
vocabularies do not address; realistic ceiling ~81 tok/s (~78%), parity likely unreachable via codegen
vocabulary. The L0 probe (a cheap, reverted renderer patch) caught a non-binding constraint before
building a major subsystem. The phases below are RETAINED for the record but NOT to be built.

**L1-L4 — build the proven lever(s).** Only the lever(s) L0 proved real: the UOp/render touch point +
the `OptOps` action + the `search.py` wiring. Unit-test correctness; confirm gfx1100 codegen.

**L5 — search-reachable end-to-end.** The decode GEMV now uses *search-found* latency-hiding (not a
hand kernel). Measure decode tok/s vs the L0 hand ceiling and llama.cpp; confirm BEAM / the N-loop
cost model selects the new opt. Correctness-gated.

## Pre-registered honesty + boundary

- This is a tinygrad-CORE codegen subsystem (`codegen/late/`, `codegen/opt/`, `renderer/`, maybe
  `uop/`), far larger than DP4A. CUTLASS/MLIR devote whole pipelining passes to it.
- LLVM already schedules the HIP path, so the marginal value is uncertain -> L0 probes gate everything.
- Pre-registered ceiling realism: D0 put a well-implemented int8 GEMV at ~81 tok/s (~78%); if
  latency-hiding adds on top, parity is *possible* but not assumed. A located ceiling below llama.cpp
  is a real, acceptable result (the M0/roofline discipline), not a failure.
- Success = search produces a latency-hidden decode GEMV competitive with (or a clearly-located
  ceiling vs) llama.cpp, with NO hand-written kernel in the hot path — the philosophically-pure
  "expand the Typesetter's vocabulary so the Judge can reach it," on the decode kernel that currently
  out-runs tinygrad.

## Touch-point index (for the build, once L0 greenlights)

- `tinygrad/renderer/cstyle.py` — `HIPRenderer`: kernel-attribute emission (`__launch_bounds__` /
  `amdgpu_waves_per_eu`) for L1; `__builtin_amdgcn_global_load_lds` for L4.
- `tinygrad/renderer/amd/` + `AMDLLVMRenderer` — explicit `s_waitcnt` control if L4 needs it.
- `tinygrad/codegen/late/{expander,linearizer}.py` — the prefetch/double-buffer pipelining pass (L2/L3).
- `tinygrad/codegen/opt/__init__.py` — new `OptOps` (WAVES / PREFETCH / DOUBLEBUF / ASYNC).
- `tinygrad/codegen/opt/postrange.py` — `apply_opt` handling for the new opts.
- `tinygrad/codegen/opt/search.py` — search actions so the opts are BEAM/cost-model reachable.
- `tinygrad/uop/ops.py` + `tinygrad/uop/spec.py` — any new UOp (async-load/wait for L4) + spec rule.
