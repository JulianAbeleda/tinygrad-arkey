# Track B execution phases: generated WMMA GEMM -> competitive perf -> 8B/14B

Phased plan from the current state (any-K single-tile WMMA bit-exact on DEV=AMD:ISA) to a competitive generated
prefill GEMM, then 8B/14B, then fusion/megakernel. Each phase ends in a GATE; Phase 1's gate is the load-bearing
DECISION point (first real TFLOPS -> is Track B worth continuing). Structural facts cite `tinygrad/renderer/isa/amd.py`.

Reality: the core is a sequential chain in one file with serial AMD gates. Parallelize file-disjoint work; sequence
the coupled core; Fable-review on any real blocker (do not thrash).

## PHASE 1 — correctness at scale (unblocks the FIRST perf number)
Two sub-parts, then a joint gate.

### 1a. fp16 global access width (DONE this session)
Root cause (agent-verified, NOT an addressing bug): fp16 loads/stores emitted as `global_load_b32`/`global_store_b32`
-> the last fp16 element over-reads/writes 2 bytes past the buffer -> page-boundary MMU fault (single-tile worked only
via page slack). Fix: `global_load_u16`/`global_store_b16` when itemsize==2, in the 3 `lower_inst` global branches
(GLOBAL_LOAD ~683, GLOBAL_STORE ~733, GATED_STORE ~726). Addressing (gidx*tile) was already correct.
GATE: forced-TC-only 512x512x512 bit-exact + no MMU fault + FIRST (floor) TFLOPS.

### 1b. multi-output-tile register model
The `[FRAG_BASE=200, FRAG_TOP=238)` 38-VGPR window is a POC constraint; a real WMxWN output tile needs WM*WN*8
accumulators (4x4 = 128) + A/B frags -> ~192 VGPRs. KEY: the v>=238 garbage trap is a RAW-INS-only artifact; the ISA
renderer's ELF descriptor auto-sizes VGPR to the highest used reg, so the real ceiling is OCCUPANCY, not v238 -> we can
use most of the 256-file. Recommended (hybrid): pin accumulators LOW + contiguous, sized WM*WN*8, keyed PER-SUBTILE
`(id(dreg), idx.arg//8)` (current keying on `id(dreg)` alone aliases all subtiles -> the bug); A/B in a small window;
`_vpool` excludes both; no `regalloc.py` change; leave spill unimplemented (fit without spilling).
Sub-tasks: accumulator LOW region sized from WMMA out-vec width; split/parameterize `_frag_base` (C-low vs A/B); fix
C-base key to per-subtile (`amd.py:276-278, 480-497`); `_vpool` exclusion (`amd.py:82-88`); A/B residency choice.
GATE: DEV=PYTHON allocates WM*WN non-overlapping 8-aligned accumulators <256, no spill; DEV=AMD 64x64x64 + 512x512x512
+ 512x4096x4096 bit-exact + TFLOPS (with proper tiling). RISK: occupancy vs tile size (a search knob, not a blocker).

### PHASE 1 DECISION GATE (surface to user)
First real TFLOPS on a well-tiled generated GEMM vs (a) the HIP path, (b) the hand kernel ~58. Decides Phase 2+ vs
"bank the wins, keep the hand kernel." Even a low number is informative (headroom for LDS+pipelining).

## PHASE 2 — operand reuse (LDS staging)
Wire the built-but-unwired `operand_staging_policy` (codegen/opt/operand_staging.py) into `_tc_local_stage_wmma_sources`
(postrange.py) so each operand routes REGISTER vs LDS by (compute-cost x reuse). Build the LDS-staged operand path on
the ISA renderer (bufferize LOCAL -> ds_store/ds_load b128 cooperative load; b128 = Track-B L3). Cuts global bandwidth
via A-across-N / B-across-M reuse.
GATE: policy routes correctly; TFLOPS up on the LDS-routed operands; bit-exact.

## PHASE 3 — pipelining (B1; the ~40->58 lever)
Targeted `vmcnt(n)` waitcnt in `_insert_waitcnt` (R1 span-aware already done) + DBUF double-buffer (unroll-by-2 peel
`_prefill_dbuf_peel` + the existing list scheduler; no new modulo pass unless proven needed). Watch the s_delay_alu
backedge hazard.
GATE: TFLOPS approaches hand ~58; bit-exact.

## PHASE 4 — schedule search + role coverage
Wire BubbleBeam/FutureSight search over the substrate's schedule space (tile/wave/unroll/DBUF/LDS-pad/occupancy) per
role shape; regenerate the per-role schedule table. This is where the substrate can BEAT the hand kernel's fixed config.
GATE: whole-prefill 8B pp512 via the fixed harness vs hand 4413; per-role >= their ceilings.

## PHASE 5 — fusion + megakernel ("both, let search decide")
5a. Fuse epilogue/activation/residual/norm into the GEMM output stage (the whole-prefill "surpass" lever the opaque
hand kernel structurally can't do). 5b. Grid-barrier megakernel: decode megakernel (GEMV-compose, no WMMA) + prefill
megakernel (WMMA blocks + grid.sync + single-sink emission + vscnt). Make per-kernel vs fused expressible; let the
search decide per shape. (See memory megakernel-decode-grid-barrier.)
GATE: whole-model; search crowns per-kernel vs megakernel per shape.

## 14B (rides Phase 2/3)
The Q4_K->fp16 decode UOp already exists (reused: `w_f16` adapter landed); once Phase 2 (LDS) + Phase 3 exist, 14B is a
B-operand source-swap: stage the decode into the LDS B-tile (operand_staging -> LDS for the computed operand),
weights packed-resident (`prefill_packed_weight`, ~9GB, no OOM). Ceiling ~23 TFLOPS (decode-VALU-bound); win vs the
365 tok/s VALU fallback. Then delete extra/qk/prefill/wmma.py + confirm PURE_MACHINE_SEARCH_ONLY.

## Dependency graph (execution order)
1a (done) -> 1b -> [PHASE-1 GATE] -> 2 -> 3 -> {4, 14B} -> 5. Sequential core in isa/amd.py; parallelize only genuinely
file-disjoint pieces (e.g. per-role search shapes in Phase 4). Never force-kill a live DEV=AMD run (MES wedge).

Source: agent scopes (grid-addressing = width fix; register-model = hybrid low-accumulator) + docs/track-b-100pct-scope.md.
</content>
