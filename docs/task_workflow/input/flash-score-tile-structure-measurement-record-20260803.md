# Flash score tile structure measurement record - Scope C microbench

Date: 2026-08-03
Status: measurement record. Authorized by `decode-gemv-efficiency-forward-scope-20260803.md`
section 5 (Scope C): apply the wmma_peak-style diagnostic method to the whole-cache flash score
tile structure (shared UOp builder `flash_decode_attention.py:92`, installed row
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`): hoist the cache/score operand setup
out of the timed loop, use independent accumulators over the score computation, zero loads in the
hot loop where the structural variant allows, and sweep the tile geometry (LANES, WARPS/QG, TK,
the LDS staging shape SW = DECODE_STAGE_COALESCE surface) at max_context=4608 against the
llama-class ~3.2 us floor. It changes no implementation code: `flash_decode_attention.py` and
`decode_routes.py` are untouched; the deliverable is the microbench, the NV installed-row timing
probe, and this record. Branch boundary: tinygrad `nvidia-bringup-20260731` at `858effd46`.

## 1. Protocol

Probe: `extra/llm_research/microbench/flash_score_tile_peak_cuda.cu` (new file, wmma_peak/dp4a_peak
discipline: operand setup hoisted, NACC independent score chains, zero loads in the hot loop where
the structural variant allows, runtime trip count, never-taken keep-alive store, rendered source /
SASS inspected before believing a number). Installed-row in-loop rows and pins come from the new
`extra/llm_research/decode/flash_score_tile_nv_timing.py` (DEBUG=2 prime-token capture, same
harness convention as the campaign per-kernel tables).

- Session: this workspace, branch `nvidia-bringup-20260731`, HEAD `858effd46`.
- Config: NVIDIA GeForce RTX 5090 (sm_120, compute capability 12.0), driver 595.84, CUDA 13.2,
  `nvcc -O3 -arch=sm_120`.
- Evidence class: standalone synthetic-data timing (deterministic PRNG fp16 cache/query, full
  18.9 MB KV cache L2-resident, no model load) plus a CPU-only pg3 HIP render-equality control
  plus an NV in-loop per-kernel row capture on the real Qwen3-8B decode (pins, section 6).
- Timing: back-to-back kernel passes inside one launch (`cudaEventElapsedTime`), `--iters 3000`,
  reported as us per score kernel (measured ms / iters). Every GPU run was serialized with
  `flock /tmp/nv_gpu.lock -c "<cmd>"`; the lock file was not modified or deleted.
- Tile facts reproduced from the rendered CUDA (Tc=4608): grid (48 splits, 8 kvh, NG=G/WARPS),
  block (LANES, WARPS), THREADS = LANES*WARPS, head = kvh*G + qg*WARPS + warp, per-split window
  LPER = 96 aligned tokens (Tc=4608), NB = ceil(LPER/TK) runtime tiles, RP = Hd/(2*LANES) fdot2
  dots per lane per token, log2(LANES)-step `__shfl_xor_sync` reduce, scale 1/sqrt(128), online
  softmax (max, exp2, correction/probability), PV merge over R = Hd/LANES, den/mx update.
- Modes: MODE=0 REG (one tile's K/V register-resident per lane, timed loop has zero loads and zero
  LDS; the tile score is replayed for all NB tiles - the pure structural ceiling), MODE=1 LDS
  (whole split window staged once into 48 KB dynamic shared before timing; timed loop reads LDS
  only), MODE=2 TILE (per-tile LDG->STS + 2 barriers per tile, reproduces the installed-row
  structure). SW = DECODE_STAGE_COALESCE surface {0,1,2,4,8} affects the MODE=1/2 staging index.
- Geometry swept: LANES {16,32,64} x WARPS {1,2,4} x TK {8,16,32} x SW {0,1,2,4,8} x MODE {0,1,2}
  (SW only reaches the timed loop in MODE=2), 189 rows, NACC=8 default; anchor geometry
  (LANES=32, WARPS=4, TK=16, SW=1, MODE=2) is the installed row's shape.

## 2. Launch-rejection bug found and fixed (this session)

The first build had a silent launch-rejection bug for MODE=1 with LANES=64 at LPER=96: all nine
rows reported `us_kernel=0.001` (constant ~5 us total regardless of iters). nsys showed API-success
launches but NO GPU kernel dispatched (`/tmp/nsys_mode1_64.nsys-rep`). Root cause: `run_cfg`
guarded `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` with
`if (shmem > 48*1024)`; for MODE=1 LPER=96, shmem = 49152 = exactly 48*1024, so the attribute was
NOT set, but the kernel needs 49152 dynamic + 16 static > 48 KB default, and the driver silently
dropped the launch (no error surfaced; the warmup error check did not fire).

Fix (in the committed .cu): the attribute is now set unconditionally with the exact requested
dynamic shared size, `cudaGetLastError` checked after the set, and the timed-launch error check
captures the sync error directly. Verified: LANES=64 MODE=1 LPER=96 now reports 58.6 us and scales
with iters; the 32-lane anchor rows are bit-close to the pre-fix sweep (26.467 vs 26.458 at
LANES=32/WARPS=1/TK=16/MODE=1; 13.934 vs 13.977 MODE=2 anchor). A `--verbose` flag was added for
the diagnostics used to find this; it prints nothing unless requested.

## 3. Purity verification (SASS, sm_120, cuobjdump)

MODE=0 anchor (LANES=32, WARPS=4, TK=16, NACC=8, LPER=96): 0 spills (REG 159, STACK 0, LOCAL 0),
all 136 LDG.E.U16 in the preload prologue (offsets <= 0x4c20), zero LDS/STS/BAR in the whole
function, exactly one never-taken gated STG keep-alive sentinel, zero LDL/STL. The timed loop
contains only FFMA/FADD/FMUL/HADD2/SHFL/MUFU/FSETP plus the gated sentinel - the zero-load claim
holds. MODE=1/2 carry their LDS reads (and MODE=2 its per-tile LDG+STS+BAR) by construction.

## 4. Results (RTX 5090, sm_120, standalone, iters=3000, max_context=4608 shape LPER=96)

| mode | best config (LANES, WARPS, TK, SW) | us/kernel | vs 3.2 us floor | note |
| --- | --- | ---: | ---: | --- |
| MODE=2 (installed structure) | 16, 4, 32, 4 | 11.709 | 3.7x | best installed-structure row |
| MODE=2 (installed structure) | 16, 4, 32, 2 | 12.172 | 3.8x | |
| MODE=2 (installed structure) | 32, 4, 32, 1 | 12.401 | 3.9x | |
| MODE=2 anchor (32, 4, 16, 1) | 32, 4, 16, 1 | 14.150 | 4.4x | installed-row shape |
| MODE=2 (32, 4, 16, SW sweep) | 32, 4, 16, 4 | 13.188 | 4.1x | SW surface flat 13.2-14.8 |
| MODE=1 (LDS window) | 32, 4, 8, 0 | 11.143 | 3.5x | best LDS row |
| MODE=0 (zero loads) | 16, 4, 8, 0 | 5.311 | 1.7x | pure structural ceiling |
| MODE=0 (zero loads) | 32, 1, 8, 0 | 5.742 | 1.8x | |
| MODE=0 (zero loads) | 32, 4, 16, 0 | 6.358 | 2.0x | anchor geometry, zero loads |
| MODE=0 worst | 16, 1, 32, 0 | 29.76 | 9.3x | |
| MODE=2 worst | 16, 1, 16, 4 | 62.278 | 19x | single-warp + 4-wide stage |

Run-to-run stability: a second full sweep (this session, `--iters 3000`) agrees with the
pre-session sweep within 0.31 us on every row (typical <0.2 us). The three best rows per mode are
stable across the two sessions (MODE=2: 11.709/11.710, MODE=1: 11.143/11.098, MODE=0: 5.311/5.313).

## 5. d512 calibration (installed row shape) and the values-vs-structure question

d512 shape (LPER=16, nvalid=11, LANES=32, WARPS=4, TK=16, SW=1, MODE=2, iters=3000):

| NACC | qhoist | us/kernel |
| --- | --- | ---: |
| 1 (serial chain, installed render) | 1 | 2.334 |
| 1 (serial chain, installed render) | 0 (per-token q loads) | 2.506 |
| 2 | 1 | 2.050 |
| 4 | 1 | 1.956 |
| 8 | 1 | 1.945 |

The installed in-loop row at d512 measured 7.52 us/kernel median (36x) in this session (section 6),
matching the campaign 7.6 us. The microbench is a PEAK measure: it runs back-to-back passes in one
launch with the full cache L2-resident and operand setup hoisted, so it reads below the in-loop
per-launch row. The earlier "7.6 us at Tc=513" calibration claim in the pre-NACC build was a
cold-L2/single-launch reading: at iters=1 (cold L2, single launch) the same d512 shape reads
6.752 us, converging to 2.33 us at iters=3000. The discrepancy is therefore methodology, not
structure: the microbench isolates the warm steady-state structural rate (wmma_peak discipline),
while the installed row's in-loop time includes cold L2 per launch plus surrounding kernel traffic.

Values question: the 36-row values sweep (decode-gap scope section 10, best 7.41 us vs 7.59
baseline, tok/s 149.1-163.4) is the control that this sweep is STRUCTURAL. No geometry at
max_context=4608 clears the ~3.2 us floor even with zero loads in the hot loop (best MODE=0
5.311 us = 1.7x floor; best installed-structure MODE=2 11.709 us = 3.7x floor). The structure's
own floor is the 5.3 us zero-load ceiling - still 1.7x above llama-class. The shared tile
structure is confirmed as the substrate: no tile geometry, staging shape, or independent-accumulator
count closes the gap; the score kernel's recoverable mass is not in the tile decomposition.

## 6. Installed-row in-loop rows and controls (NV, this session)

`extra/llm_research/decode/flash_score_tile_nv_timing.py` on Qwen3-8B-Q4_K_M, d512, DEBUG=2
prime token:

| kernel | count | median us | min us | max us |
| --- | ---: | ---: | ---: | ---: |
| flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128 | 36 | 7.52 | 6.85 | 8.58 |
| flash_fused_gmax_combine_32_128 | 36 | 3.39 | 3.33 | 4.64 |

The combine is at llama parity (campaign 3.6 us) and is not part of the recoverable mass. Delta to
chase at d512: 7.52 - 3.17 (llama flash_attn_ext_vec, campaign scope 14.1) = 4.35 us x 36 =
0.157 ms node-sum, consistent with the scope's ~0.16 ms bound. Same-session llama-bench wall on the
same model: tg10 @ d512 248.01 +/- 7.77 tok/s (matches campaign 248.20 +/- 7.37).

Pins (all matched this session): token sha256
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9` 3/3, first token `151936` 3/3,
decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, bench census
`prefill_overlay_promotion` = `candidate_set:sha256:
1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`. Census: 1021 kernels/token,
6183 us/token. Fused prefill attention disabled via
`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`.

## 7. Go/no-go

NO-GO for the tile-structure-as-lever hypothesis. No geometry at max_context=4608 reaches the
llama-class ~3.2 us floor, even with zero loads in the hot loop (best 5.311 us = 1.7x the floor;
the installed-structure MODE=2 rows sit 3.5-4.4x above it). The values question closes permanently:
the 36-row values sweep (7.41 vs 7.59 us) and this structural sweep both fail the bar, so neither
route values nor tile geometry is the lever. The whole-cache tile structure is confirmed SUBSTRATE:
any future fix must move the shared emitter's thread decomposition / staging shape itself (a
structural change), must render identically across the pg3 AMD/Metal arms (section 8.1/8.3 of the
scope), and must be re-gated on the fixed-depth wall + token sha before any promotion. The combine
(3.39 us in-loop, llama parity) is untouched.

## 8. Controls

- pg3 decode render-equality (HIP arm, render-only, CPU-only, `--renderer hip`, this session): all
  10 legacy rows byte-identical to the section 8.1 pin table (flash score `66d4c4da3108`, combine
  `c78e4651ad35`, q4k `312422c73a49`/`27857cb8ca03`/`851760e2053c`/`39ddb717ddd4`, q6k
  `cc38fbb3db92`/`5795e66a7292`/`344e1c388eeb`/`c708302aa2d2`), plus the M2 promoted fused row
  `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f`. The Metal arm was not run (macOS-only).
- NV pins (section 6): token sha, first token, decode sha, bench census all matched this session.
- Model / harness: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`; fused prefill attention disabled by
  the house convention; the microbench itself is standalone (no model load).

## 9. Deviations

1. The d512 calibration is a PEAK measure, not an in-loop replica: warm iters=3000 reads
   2.334 us (NACC=1) vs the installed in-loop 7.52 us; cold single-launch (iters=1) reads
   6.752 us. The earlier pre-NACC-build "7.6 us at Tc=513" calibration was a cold/launch-bound
   reading, documented in section 5; the structural verdict uses the warm steady-state peak per
   the wmma_peak discipline and the max_context=4608 sweep, where the floor comparison is made.
2. The LANES=64 MODE=1 launch-rejection bug (section 2) invalidated nine rows of the first sweep;
   those rows are only counted from the fixed binary, and the fix is verified (58.6 us, iters-
   scaling, anchor bit-close).
3. The `--verbose` diagnostics and the unconditional shared-memory attribute set were added to the
   microbench to find the bug; both are kept in the committed source (they change nothing unless
   `--verbose` is passed).

## 10. HARD STOP

This record is measurement evidence only. No implementation work follows in this session: no
emitter change, no route row, no geometry change, and `flash_decode_attention.py` /
`decode_routes.py` stay untouched. A structural (SUBSTRATE) change, if ever attempted, requires a
separate variant-specific scope with its own settling command, legacy hash controls, correctness
pins, and fixed-depth wall gate, reviewed before any code. Nothing here authorizes promotion or a
push; the parent pushes after review.
