# Regarding Pure Machine Search

Roadmap: `docs/pure-machine-search-roadmap.md`.

The goal of this fork is pure machine search: the scheduler generates every kernel, and a search picks the
config to ship. We do not use tinygrad's BEAM autotuner. We built our own candidate and lifecycle search
(`extra/qk_decode_eval.py`, `extra/qk_lifecycle_search_loop.py`) that decides which decode primitive and flag
config wins, gated by correctness and a per-token throughput bar.

We are not fully pure yet, but the current boundary is narrower than the old "two hand kernels" framing:

- Tracked Q4_K decode GEMV is generated under BubbleBeam G3 and speed-equivalent to the old owned warp route.
- Q6_K direct/lane-map routing was implemented and token-correct, then W==D-refuted; the shipped Q6_K coop route stays.
- Prefill defaults to a search-selected graph-GEMM route: the `pipe_tm2_tn2` pipeline applied role-selectively (pipeline on for the latency-bound projection and FFN-down roles, off for the already-saturated FFN gate/up). This role-selective default is promoted — it beats the plain global pipeline by ~3% and the prior default by ~12–23% through ctx 8192. Rollback: `PREFILL_PIPE_ROLE_SELECTIVE=0` (global pipeline), `PREFILL_GEMM_PIPELINE=0` (prior default).
- Decode attention still ships the owned two-kernel tile+combine route. A native AMD-ISA generated attention tile exists
  and is correct/route-bound, but it is not fast enough to promote and the ceiling audit says further attention work is
  low-leverage for whole-decode tok/s.

## Current status of the formerly hand-written decode kernels

These are the two historical hand-written decode kernels. GEMV has been replaced for tracked Q4_K decode roles;
attention remains active. The important update is that the attention problem is now a measured ceiling problem, not an
unbounded "keep tuning combine" problem.

### 1. Warp GEMV: `extra/q4_k_gemv_primitive.py`

- Status: superseded for tracked Q4_K decode GEMV by BubbleBeam G3 generated LaneMap routes. The current promotion
  verdict is `AMD_ISA_G3_PROMOTION_PASS_SPEED_EQUIVALENT`.
- Coverage: gate/up, FFN down, and Q4_K `4096x4096` projection route through generated G3 LaneMap programs under
  BubbleBeam/FutureSight, with no owned Q4_K GEMV or lane-partition bridge on that path.
- Fallback/reference: `Q4K_GEMV_WARP=1` (FFN gate/up), `Q4K_GEMV_WARP_DOWN=1` (FFN down). Revert with the flag set
  to 0.
- Why hand-written: the scheduler GEMV runs at about half of HBM peak (47 to 57%) because of the schedule:
  one thread per row, serial over K, uncoalesced. llama's MMVQ shape needs 128 threads per row with K-block
  parallelism and an in-kernel cross-lane (warp shuffle) reduce. The scheduler cannot emit the cross-lane
  reduce, so the generated GEMV leaves performance on the table.
- Result: generated G3 tracks owned within 0.5% at ctx512/1024/2048/4096 (`103.93 / 102.04 / 99.74 / 94.44 tok/s`
  in the promotion gate). This is a purity win, not a speedup over the owned route.
- Size: tiny, and tiny relative to llama. The kernel is emitted programmatically by `q4k_gemv_warp_kernel`, a few
  hundred lines (the rest of the 852-line file is Python dispatch, fallback, and tests). llama.cpp's quantized
  GEMV is 2,569 lines (`mmvq.cu` + `vecdotq.cuh`), roughly an order of magnitude larger, because it covers every
  quant type and shape. We cover Q4_K decode and let the scheduler generate the rest.
- We tried the instruction-level approach first (tinygrad's WMMA-style trick): a schedulable `udot4` builtin let
  the scheduler compose the GEMV (`Q4K_VDOT`). The kernel was correct at about 57% of peak, but in-model it lost
  at 0.96x, because every int-dot path pays a q8 activation-quant cost (about 7us per kernel) that eats the win.
  Only then did we hand-write the kernel.
  - [Schedulable udot4 GEMV built and refuted in-model](archive/qk-mmvq-int-dot-closeout-20260618.md)

### 2. Owned AMDGCN attention tile + combine: `extra/qk_owned_flash_decode.hip`

- Flag: `DECODE_ATTN_AMDGCN_TILE=1`, active at ctx >= 512 (`DECODE_ATTN_AMDGCN_MIN_CTX=512`).
- Why hand-written: the shipped route is two precompiled graph nodes: an owned split-KV tile plus a separate combine.
  The old scheduler path emitted scalar fp16 loads, no LDS, and no `v_dot2`; the later native AMD-ISA path closed those
  primitive gaps but still did not reach the owned route's per-token efficiency.
- Gain: about +12 to +22% decode, on top of the scheduler baseline.
- Size: tiny, and tiny relative to llama. 283 lines of HIP and AMDGCN in one file. llama.cpp's flash-attention is
  6,688 lines across `fattn-*.cu/cuh` (its `fattn-tile.cuh` alone is 1,357), because it is templated across head
  dims, dtypes, quant types, and GPU arches. We target one shape (Qwen3-8B Q4_K on gfx1100) and the scheduler
  generates the rest, so our tile is about 5x smaller than llama's tile kernel and over 20x smaller than its
  flash-attention family.
- Fallback: at ctx < 512 the model uses `FLASH_VARIANT=gqa_coop_vec`, which is scheduler-generated.
- We tried the instruction-level approach first (tinygrad's WMMA-style trick): the fused tile is expressible in
  the scheduler idiom, so we built it (Path A) rather than hand-writing. It lost. Later, the native AMD-ISA backend
  closed the major primitive gaps (`v_dot2`, cross-lane, LDS, barriers, waitcnt, grid parallelism, hardware exp,
  dynamic split count, register accumulators), but the generated route still lands around 60-68% of owned and attention
  contributes too little whole-decode wall time for more attention tuning to be the next max-out path.
  - [Path A fused softmax+V built and refuted at 0.725x](archive/fused-softmax-v-tail-candidate-result-20260621.md)
  - [Concrete fused-flash refuted at 0.965x (global loads, not LDS)](archive/fused-flash-concrete-gate-result-20260621.md)
  - [Two-kernel route audited and combine refuted](decode-two-kernel-problem-audit-result-20260625.md)
  - [Attention ceiling: move search to non-attention](amd-isa-decode-attention-ceiling-audit-scope-20260629.md)

## Evidence: we tried the scheduler path first

Neither kernel was hand-written by default. Each one followed measured scheduler attempts that fell short. We
measured the generated path, built scheduler-expressed alternatives, refuted them, and only then hand-wrote.

For the warp GEMV, the scheduler GEMV was measured at 47 to 57% of HBM peak vs llama MMVQ at about 70%, and a
schedulable int-dot variant was built and refuted in-model before the hand kernel won byte-identical.

- [Scheduler GEMV diagnosis, scope](archive/decode-ffn-gemv-scheduler-diagnostic-scope-20260622.md)
- [Scheduler GEMV diagnosis, result](archive/decode-ffn-gemv-scheduler-diagnostic-result-20260622.md)
- [Int-dot GEMV built and refuted in-model](archive/qk-mmvq-int-dot-closeout-20260618.md)
- [Hand warp GEMV wins, byte-identical](decode-q4k-gemv-warp-promotion-result-20260624.md)

For the attention tile, llama's tile measured 5 to 6 times faster standalone, ISA attribution showed the
scheduler emits scalar fp16 loads with no LDS and no `v_dot2`, and three scheduler-expressed builds were
refuted before the hand tile shipped.

- [llama tile 5 to 6x faster standalone](archive/llama-flash-attn-tile-oracle-result-20260621.md)
- [ISA attribution: scalar loads, 0 v_dot2, 0 LDS](archive/low-level-decode-attn-attribution-result-20260621.md)
- [Scheduler attempt: fused softmax+V, refuted at 0.725x](archive/fused-softmax-v-tail-candidate-result-20260621.md)
- [Scheduler attempt: tiled-matmul PV, blocked by layout](archive/matmul-pv-diagnostic-result-20260621.md)
- [Scheduler attempt: concrete fused-flash, refuted at 0.965x](archive/fused-flash-concrete-gate-result-20260621.md)
- [Native fused-flash linearizer scope](archive/native-fused-flash-linearizer-scope-20260621.md)
- [Hand AMDGCN tile adds +12 to +22%, capability gap confirmed](decode-campaign-final-synthesis-20260623.md)

## Everything else is scheduler-generated or generated through BubbleBeam

- The model bulk: norms, rope, projections, elementwise, the KV path.
- Attention below ctx 512: `gqa_coop_vec`, a tinygrad-expressed flash variant.
- Tracked Q4_K decode GEMV: BubbleBeam G3 generated LaneMap programs for gate/up, FFN down, and Q4_K projection.
- Several fallback decode GEMV variants that the scheduler can express: `MMVQ_COOP` (cooperative-K), `Q4K_VDOT`
  (schedulable builtin v_dot4).

So the active hand-written footprint in the default decode path is now the attention tile and combine lifecycle.
That footprint is retained because the generated/native route is correct but not fast enough, while the ceiling audit
shows the next whole-decode leverage is outside attention. The warp GEMV remains in the repo as fallback/reference and
is no longer required for the tracked Q4_K BubbleBeam default route.

## The rest of the hand-written source in the repo

The repo carries more hand-written `.hip` and `.cpp` than those two. None of it is default runtime. It is kept
as opt-in references, measurement tooling, or control experiments.

- Opt-in / research (off by default):
  - `extra/q8_ffn_*.py` (`Q8_FFN_HANDWRITTEN=0`): q8 FFN route, opt-in, dNLL-gated.
  - `extra/q4k_mmvq_handwritten.hip`, `extra/q4k_w4a16_handwritten.hip`: handwritten reference kernels.
  - `extra/q6_k_gemv_primitive.py`: Q6_K GEMV primitive (Q6_K down is coop-routed by default).
  - `Q4K_GEMV_WARP_PROJ`, `Q4K_VDOT`: research levers that did not transfer in-model.
- Measurement tooling (capture harnesses, not kernels the model runs):
  - `extra/qk_decode_mmvq_kernarg_capture.cpp`, `extra/qk_llama_fattn_kernarg_capture.cpp`,
    `extra/qk_tensile_kernarg_capture*.cpp`: kernarg capture for replaying vendor kernels.
- Control experiments (measured a ceiling, not routed into the model):
  - `extra/qk_prefill_blas_ceiling.cpp`, `extra/qk_prefill_blas_sequence.cpp`,
    `extra/qk_prefill_bridge_shim.cpp`: external BLAS ceiling (hipBLASLt / rocBLAS).
  - `extra/qk_tensile_solution_sweep.cpp`: Tensile solution sweep.
  - `extra/gemm/amd_seb/*.cpp`: step-by-step GEMM study kernels.
- Vendor / upstream: `extra/torch_backend/wrapped_tensor.cpp`.

## Upstream is not purely generated either

tinygrad positions itself as generating all kernels, with no hand-written kernels. That holds at the
whole-kernel level, but not at the primitive level. The renderers hand-code the hot instructions and splice
them into the generated kernels. The clearest is the tensor core: tinygrad's fast matmul depends on a
hand-coded WMMA or MFMA emission, not a search-discovered one. Citations are to upstream `tinygrad/tinygrad`
at commit `65dd099b6`.

- [Tensor core WMMA define, cstyle.py L564](https://github.com/tinygrad/tinygrad/blob/65dd099b635b8c2e34812cda0ee173b6aff343e2/tinygrad/renderer/cstyle.py#L564): `#define __WMMA_16_16_16_half_half __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12`.
- [Tensor core WMMA and MFMA emission, llvmir.py L44](https://github.com/tinygrad/tinygrad/blob/65dd099b635b8c2e34812cda0ee173b6aff343e2/tinygrad/renderer/llvmir.py#L44): `@llvm.amdgcn.mfma.*` and `@llvm.amdgcn.wmma.f32.16x16x16.f16` (the AMD_LLVM path this fork uses).
- [Workgroup barrier, cstyle.py L512](https://github.com/tinygrad/tinygrad/blob/65dd099b635b8c2e34812cda0ee173b6aff343e2/tinygrad/renderer/cstyle.py#L512): a fixed `__builtin_amdgcn_s_barrier` sequence (also [llvmir.py L196](https://github.com/tinygrad/tinygrad/blob/65dd099b635b8c2e34812cda0ee173b6aff343e2/tinygrad/renderer/llvmir.py#L196)).
- [fp8 convert, cstyle.py L500](https://github.com/tinygrad/tinygrad/blob/65dd099b635b8c2e34812cda0ee173b6aff343e2/tinygrad/renderer/cstyle.py#L500): the `__builtin_amdgcn_cvt_f32_fp8` builtin.

So the difference is degree, not kind. tinygrad hand-codes the hot instruction (WMMA, MFMA, the barrier); we
hand-code the hot kernel (the fused tile and the warp GEMV) where the scheduler cannot compose those
instructions into the shape we need. Neither is purely search-derived. Both hand-specify the primitives that
matter and generate the rest.

## The path to pure

The remaining active hand-written decode kernel is not blocked on the old primitive checklist anymore. Native AMD-ISA
attention proved that those primitives can be emitted and searched, but the result is `CORRECT_BUT_NOT_FAST` and the
whole-decode ceiling audit says attention parity would only move tok/s modestly. The practical path to pure is now:

1. keep generated G3 as the Q4_K default and owned warp as fallback/reference;
2. keep Q6_K direct default-off unless a future route beats coop in W==D;
3. keep owned attention as the default until a generated route clears promotion, but do not spend broad search there
   unless the ceiling changes;
4. push pure-search work where the wall still moves: prefill role-selective graph-GEMM is now promoted (done — it is
   the shipped prefill default); next is quant/shape-agnostic route generation.
