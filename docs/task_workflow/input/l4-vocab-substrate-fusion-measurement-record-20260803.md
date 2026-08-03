# L4 vocab substrate fusion measurement record - Stage 1 values ceiling + Stage 2 fused shape

Date: 2026-08-03
Status: measurement record. Authorized by
`decode-gemv-efficiency-forward-scope-20260803.md` section 4 (Scope B) and section 9:
diagnostic probes and measurement only, no implementation code. It changes no code and
no promotion record; the landed values row (commit `ab3cb84c1`,
`Q6K_COOP_ROW_TILE_BY_TARGET = {("NV", "sm_120"): 2}`) stays untouched, as does every
existing emitter. Branch boundary: tinygrad `nvidia-bringup-20260731` at `44725ad41`
(ahead 2; tracked tree clean, only untracked files present).

## 1. Why this record exists

The vocab head is three pieces today: `q6k_gen_coop_151936_4096` at 330.1 us (row_tile=2,
86% of the 1792 GB/s ceiling), `q6k_vocab_scalar_reduce` at 72.5 us, and the scatter chain
(~54.5 us same-session prime trace: `E_1187_32_4` 3.46 + `r_32_4_1187` 38.37 +
`r_128_16_8_1187` 10.82 + `r_16_8` 1.89; the scope doc's ~0.07 ms estimate is the same
chain), for ~457 us total vs llama's single mmq vocab kernel at 303.75 us. Scope B asks
two questions, each with its own wmma_peak-style probe:

1. Is the installed 330.1 us values row already saturated at row_tile=2? (sweep the
   remaining occupancy/vector-width surface: accumulator staging, hoisted operand setup,
   block grouping; floor = llama 303.75 us)
2. Does the fused shape (`reduction="in_kernel"` on the 151936-row head, legal at
   row_tile=2 because `row_tile*lane_extent = 32` under the single-warp constraint) land
   near llama-class and remove the scalar-reduce + scatter kernels?

## 2. Protocol

- Probes: `extra/llm_research/microbench/q6k_vocab_coop_ceiling_cuda.cu` (Stage 1,
  faithful replica of the emitted coop inner loop at row_tile=2; grid = `ROWS/ROW_TILE`
  = 75968 blocks x 32 threads, one launch = one full vocab pass, host-looped timing
  under one CUDA event pair) and `extra/llm_research/microbench/q6k_vocab_coop_fused_probe.py`
  (Stage 2, renders both kernels through the real tinygrad emitter
  `Q6KGEMVRouteSpec(rows=151936, k=4096, row_tile=2, target="NV:sm_120")` + CUDARenderer,
  compiles the emitted sources verbatim with nvcc 13.2 `-arch=sm_120`, times each at its
  emitted geometry grid 75968 x (2,16), and validates the in_kernel (N,) output against
  the external_sum (N,16) partials reduced over pos on the host).
- Session: same RTX 5090 / sm_120 box as the wall authority, Qwen3-8B-Q4_K_M. All GPU
  runs serialized with `flock /tmp/nv_gpu.lock` and confirmed 0% util at lock
  acquisition. Fused prefill attention disabled
  (`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`) in every decode run.
- Evidence class: device-timed diagnostic microbench (Stage 1, Stage 2), plus the
  mandated controls: pg3 HIP render-equality (CPU-only, no lock), fixed-depth decode pin,
  decode sha256 + census row (all flocked).
- Config: NV sm_120, `Q6K_PRIMITIVE=1`, `Q6K_COOP_ROW_TILE_BY_TARGET` row untouched.
- Purity discipline: `-Xptxas -v` 0 spills at every config; PTX instruction mix inspected
  (see deviations for the nvdisasm note). Reference row must reproduce the installed
  kernel before any sweep number is believed.

## 3. Stage 1 - values ceiling at row_tile=2

The reference row (nacc=1, row_groups=1, xsh=0 - the installed configuration) measures
330.74-331.41 us / 1540-1544 GB/s / 86.0-86.1% vs the installed 330.1 us / 1.55 TB/s /
86%: replica validated to 0.2%. Full sweep (32 passes each, one pass = one vocab pass):

| config (nacc, row_groups, xsh) | us/pass | GB/s | % of 1792 |
| --- | ---: | ---: | ---: |
| 1, 1, 0 (installed) | 330.74-331.41 | 1540-1544 | 86.0-86.1 |
| 2, 1, 0 | 329.29 | 1550 | 86.5 |
| 4, 1, 0 | 327.26 | 1560 | 87.1 |
| 16, 1, 0 | 360.92 | 1414 | 78.9 |
| 8, 2, 0 | 349.06 | 1463 | 81.6 |
| 8, 4, 0 | 340.89 | 1498 | 83.6 |
| 1, 1, 1 (x hoisted to shared) | 504.00 | 1013 | 56.5 |
| 2, 1, 1 | 520.52 | 981 | 54.7 |
| 4, 1, 1 | 527.39 | 968 | 54.0 |
| 8, 1, 1 | 527.77 | 967 | 54.0 |
| 16, 1, 1 | 611.42 | 835 | 46.6 |
| 8, 2, 1 | 368.28 | 1386 | 77.4 |
| 8, 4, 1 | 358.38 | 1424 | 79.5 |

A/B confirmation (3 interleaved rounds, 64 passes each): nacc=1 -> 330.77 / 330.77 /
330.79; nacc=4 -> 327.02 / 327.10 / 327.05. The nacc=4 xsh=0 win is a consistent 1.1%
(87.1% of ceiling), repeatable, still ~7.6% above the llama floor.

**Stage 1 verdict: the installed row_tile=2 values row is saturated.** The full
remaining surface lands at or above the installed time except one 1.1% residual
(327.0 us at nacc=4, xsh=0); hoisted x is a 53% regression (shared round trip in the hot
loop), larger blocks lose, and nacc>4 loses. No occupancy/vector-width/hoisting knob at
row_tile=2 approaches llama's 303.75 us. The remaining gap to llama is not a values
knob on this surface.

## 4. Stage 2 - fused-shape probe (reduction="in_kernel")

Both arms rendered from the real emitter and timed at grid 75968 x (2,16), one warp per
block, host-looped timing at 32 and 64 passes. The in_kernel arm is legal: 4 x
`__shfl_xor_sync` steps
(16, 8, 4, 2) reduce the 16 pos lanes within the single warp; AMD row_tile=4 stays
`external_sum` (not legal under the single-warp constraint) and is not part of this
probe.

| arm | us/pass | GB/s | % of 1792 |
| --- | ---: | ---: | ---: |
| q6k_gen_coop_151936_4096 (external_sum, (N,16)) | 328.46-328.68 | 1553-1554 | 86.7 |
| q6k_gen_coop_151936_4096_inkernel (in_kernel, (N,)) | 315.86-316.22 | 1614-1616 | 90.1 |

Numerics: the in_kernel (N,) output is bit-identical to the external_sum (N,16) partials
reduced over pos on the host, max abs diff 0.000000 over all 151936 rows (max abs value
184023.7). The fused kernel is not only legal - it is 12.6 us faster than the external_sum
arm alone in the same session (less partial traffic, in-warp reduction free).

Stack arithmetic (same-session numbers):

| path | us |
| --- | ---: |
| current: coop 330.1 + scalar_reduce 72.5 + scatter chain ~54.5 | ~457.1 |
| fused: in_kernel coop 315.9 alone (scalar_reduce and scatter chain gone) | 315.9 |
| llama single mmq vocab kernel | 303.75 |

**Stage 2 verdict: GO.** The fused shape lands at 315.9 us vs llama 303.75 us (104% of
llama-class) on the same 151936x4096 shape and removes both separate kernels: the scalar
reduce (72.5 us) and the scatter chain (~54.5 us) are structurally gone, for a
same-session vocab-path reduction of ~141 us (457.1 -> 315.9). The fused kernel is
capability-gated and additive (new route admission; AMD and Metal legacy arms are not in
this probe - see controls for the byte-identical evidence).

## 5. Controls

- pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU, no lock): all 10
  legacy hashes byte-identical to the pinned table
  (312422c73a49 / 27857cb8ca03 / 851760e2053c / 39ddb717ddd4 / cc38fbb3db92 /
  5795e66a7292 / 344e1c388eeb / c708302aa2d2 / 66d4c4da3108 / c78e4651ad35), and the M2
  promoted fused row `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f` holds.
- Fixed-depth decode pin (nmeas=20, reps=3, d512, fused prefill disabled): token sha256
  `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3, first token
  `151936` 3/3, 1021 kernels/token, 171.98 tok/s median.
- Decode sha256 + census (`model_e2e_bench.py`, d512 prefill, 96 decode tokens):
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, first token ids
  start `50994, 82, 31109, 3508, ...` (matches section 8.2), census row
  `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`. The token stream is
  the strongest pin here because the vocab head selects the token; it holds.

## 6. Deviations

- `nvdisasm` is not installed on this box, so cuobjdump SASS dumps are unavailable;
  purity evidence is PTX-level (nvcc `-ptx` instruction mix): the reference hot loop is
  `ld.global.nc.b16` + `fma.rn.f32` with zero `ld.shared`/`st.shared` and one gated
  `st.global` sentinel; xsh=1 arms add one `bar.sync` + LDS reads for x; 0 spills at
  every config from `-Xptxas -v`.
- The ceiling probe's first build launched 1/16 of the work (grid = `ROWS/32` instead of
  `ROWS/2`, ~20.5 us = 330/16); the grid bug was found via the absurd 1387% GB/s read,
  fixed to `ROWS/ROW_TILE`, and the reference row then reproduced the installed kernel.
  The buggy numbers are excluded from this record; the fixed probe is the committed file.
- The scatter-chain row (54.54 us) is the same-session DEBUG=2 prime trace
  (`/tmp/l4_final_probe.log`), not re-measured by the Stage 2 probe; the probe times the
  two emitted kernels in isolation. The scope doc's ~0.07 ms scatter estimate is the same
  kernel set.
- The GPU became busy (100% util, another process) after the final control; every probe
  run in this record completed under `flock` with 0% util at acquisition. No further GPU
  work was attempted.
- Worktree at record time: tracked tree clean at `44725ad41`; untracked files from other
  agents (`dp4a_peak_cuda*`, `l2_q6k_partial_sweep*`, `scratchpad/t6_metal_admission_probe.py`)
  were not touched and are not committed here.

## 7. HARD STOP

Nothing beyond this scope. The fused shape is measured and worth landing as a
capability-gated SUBSTRATE variant, but the landing itself - the emitter change,
admission, settling command, and its own legacy-hash + fixed-depth wall gate - is a
separate implementation scope with its own review. No promotion to
`dev`/`exp`/`master`, no push; the parent pushes after review.

## 8. References

- `decode-gemv-efficiency-forward-scope-20260803.md` sections 4, 8, 9 (Scope B)
- `nv-decode-parity-final-20260802.md` (wall authority, pins)
- `decode-gap-per-target-lever-scope-20260802.md` (L4 verdict the scope builds on)
- `m2-decode_epilogue_fusion` record (kept OPEN for NV:sm_120 Q6K down-coop in-kernel merge)
- Probes: `extra/llm_research/microbench/q6k_vocab_coop_ceiling_cuda.cu`,
  `extra/llm_research/microbench/q6k_vocab_coop_fused_probe.py`
