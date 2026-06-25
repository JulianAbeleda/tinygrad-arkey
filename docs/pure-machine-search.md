# Regarding Pure Machine Search

The goal of this fork is pure machine search: the scheduler generates every kernel, and a search picks the
config to ship. We do not use tinygrad's BEAM autotuner. We built our own candidate and lifecycle search
(`extra/qk_decode_eval.py`, `extra/qk_lifecycle_search_loop.py`) that decides which decode primitive and flag
config wins, gated by correctness and a per-token throughput bar.

We are not fully pure yet. Two kernels in the default runtime path are hand-written, because tinygrad's
scheduler cannot yet emit them. Everything else the model runs is scheduler-generated.

## The two hand-written kernels (default-on)

These are the only two hand-written kernels in the default decode path. Each is flag-gated and falls back to a
scheduler-generated path when disabled.

### 1. Warp GEMV: `extra/q4_k_gemv_primitive.py`

- Flag: `Q4K_GEMV_WARP=1` (FFN gate/up), `Q4K_GEMV_WARP_DOWN=1` (FFN down). Revert with the flag set to 0.
- Why hand-written: the scheduler GEMV runs at about half of HBM peak (47 to 57%) because of the schedule:
  one thread per row, serial over K, uncoalesced. llama's MMVQ shape needs 128 threads per row with K-block
  parallelism and an in-kernel cross-lane (warp shuffle) reduce. The scheduler cannot emit the cross-lane
  reduce, so the generated GEMV leaves performance on the table.
- Gain: about +9.6% at ctx 1024 and +8.5% at ctx 4096, byte-identical output.

### 2. Owned AMDGCN attention tile: `extra/qk_owned_flash_decode.hip`

- Flag: `DECODE_ATTN_AMDGCN_TILE=1`, active at ctx >= 512 (`DECODE_ATTN_AMDGCN_MIN_CTX=512`).
- Why hand-written: the fast attention kernel is a single fused tile that stages K and V in LDS and uses
  `v_dot2` with cross-lane reduction. The scheduler emits scalar fp16 loads, no LDS, and no `v_dot2`, which is
  about 5 to 6 times slower on this kernel standalone. The gap is a codegen capability, not tuning: native
  emission of `v_dot2` + cross-lane + LDS staging does not exist in the scheduler.
- Gain: about +12 to +22% decode, on top of the scheduler baseline.
- Fallback: at ctx < 512 the model uses `FLASH_VARIANT=gqa_coop_vec`, which is scheduler-generated.

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

## Everything else is scheduler-generated

- The model bulk: norms, rope, projections, elementwise, the KV path.
- Attention below ctx 512: `gqa_coop_vec`, a tinygrad-expressed flash variant.
- Several decode GEMV variants that the scheduler can express: `MMVQ_COOP` (cooperative-K), `Q4K_VDOT`
  (schedulable builtin v_dot4).

So the hand-written footprint in the default path is the minimum, and it is small. The attention tile is 283
lines of HIP and AMDGCN; the warp GEMV is a few hundred lines of HIP inside a Python dispatch wrapper (the rest
of its file is dispatch, fallback, and tests). That is the whole hand-written GPU surface, on the order of a
few hundred lines, against vendor hand-tuned kernel libraries like Tensile, Composable Kernel, and
FlashAttention that run to thousands or tens of thousands of lines. Two kernels, both necessary, neither
droppable without losing llama parity, and no third hand-written kernel in the default route.

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

## The path to pure

The two hand-written kernels are the exact shape of the remaining codegen gap: native `v_dot2`, cross-lane
reduction, and LDS staging. If the scheduler gains those, both hand-written kernels can be replaced by
generated ones and the fork becomes pure machine search. Until then, both stay flag-gated with a scheduler
fallback, so the generated path is always one env var away and serves as the portability and correctness baseline.
