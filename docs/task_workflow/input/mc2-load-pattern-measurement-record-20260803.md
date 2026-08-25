# MC2 load-pattern measurement record - vec width / lanes / alignment / residency sweep on partial, coop-down, and q4k gate/up

Date: 2026-08-03
Status: measurement record. Authorized by
`decode-gemv-instruction-bandwidth-scope-20260803.md` section 4.2 (MC2 probe: load-pattern
sweep on the partial and down shapes, extended to the q4k gate/up shape) and the section 10
amendment (per-item HARD STOP lifted; stop only on a blocker with no named next step, or on
the coordination handoff below). Diagnostic probe and measurement only: no route admission,
no emitter change in the runtime. Branch boundary: tinygrad `nvidia-bringup-20260731` at
`af4ad91d8` (MC3 commits landed on top of `e71f2ef17`; the decode tree
`decode_kernels.py` / `decode_routes.py` is byte-identical to the pins' last run).

## 1. Why this record exists

MC2 asks whether the L2-recorded binding constraint of the decode GEMV class - the load
pattern, not the math (L2 record section 6: ALU mix 30-70x below memory times) - can be
moved by sweeping the load shape on the three deficit shapes: the Q6_K partial route
(11% of the bandwidth ceiling, 0.20 TB/s vs llama's 1.04 TB/s), the Q6_K coop-down route
(46% of ceiling, 0.82 TB/s vs llama's 1.4 TB/s), and the q4k gate/up shape (per-role at
parity, but the floor is llama's fused pair). The sweep surface (scope section 4.2): vector
width of packed-storage loads (LDG.128 / LDG.U32 / LDG.U16; the installed partial route is
scalar U16 halfword window loads), lanes per thread and threads per block (recorded best
row is split-reduce-4 with 4 threads per part; llama's mmq blocks are 128 threads),
alignment of the per-part window start and per-thread stride (the serial 4-block x 16-pos
chain is the structural suspect from L2 record section 6), prefetch depth and L2 residency
for the 3.44 MB Q6_K weight set, and occupancy at the emitted geometry (register cap,
block grouping). All under the wmma_peak discipline: hoisted operands, independent
accumulators, zero loads in the hot loop where the knob allows, SASS-inspected purity
before believing a number.

## 2. Protocol

Probe: `extra/llm_research/microbench/l2_q6k_partial_sweep.cu` (extended in place; the
original L2 CFGS table, kernels, CLI flags, and output format are byte-behaviorally intact,
verified by reproducing the L2 rows: legacy_32 12.89-12.92 us standalone vs the L2 record's
12.92). New `--shape partial|coop|q4k`, `--rows`, `--bw`, `--only` knobs select the CFGS2
table (46 rows) over three kernel families: the partial family (`k_legacy_v` vec-width
variants of the installed 32-thr structure, `k_merge_v` split-reduce vec variants), the
coop family (`k_coop_legacy` installed control replica, `k_coop_v` vec variants,
`k_coop_s2/s4` split-reduce, `k_coop_group` group-lane layout), and the q4k family
(`k_q4k_legacy` installed control replica, `k_q4k_v` vec/xsmem/grouping variants,
`k_q4k_qv` quad-lane u128). All 71 kernels in the binary compile with 0 spills, 0 stack
frames (`--ptxas-options=-v`); no spills anywhere in the SASS.

- Session: this workspace, branch `nvidia-bringup-20260731`, HEAD `af4ad91d8`.
- Config: NVIDIA GeForce RTX 5090 (sm_120), CUDA 13.2,
  `nvcc -O3 -arch=sm_120 -std=c++17 --ptxas-options=-v`. Deterministic synthetic packed
  weights/x (finite fp16 d slots, bounded scales, random nibbles), no model load.
- Evidence classes: OBSERVED = measured this session under the lock; INFERRED = arithmetic
  (flagged inline). Lifecycle vocabulary only; no composed forecasts, no wall claims.
- Timing: best-of-N back-to-back passes inside one kernel launch (`cudaEventElapsedTime`),
  `--iters 2000 --reps 5` for all go/no-go rows. Per-pass time = measured ms / iters. Every
  GPU run was serialized with `flock /tmp/nv_gpu.lock -c "<cmd>"` with 0% GPU utilization
  confirmed at lock acquisition; the lock file was never modified or deleted.
- Numerics: every non-control CFGS2 row is spot-checked against the installed control
  replica's output (row totals for q4k/coop; per-part or row totals for partial), one pass.
  Max relative error 0 to 5.4e-3 across all rows (fp32 reassociation noise on magnitudes
  ~1e6-1e8; fp16-exponent-masked fills keep the comparison NaN-free).
- BW ceiling: `--bw` streams each shape's exact weight set (Q6_K 210 B/block, Q4_K 144
  B/block) with `k_bw_read` (5x LDG.128 per loop body, 512 blocks x 256 thr) as the
  L2-resident achievable-bandwidth control.

## 3. Purity - ptxas and SASS table

`--ptxas-options=-v` on the full binary: all 71 entry points report 0 bytes stack frame,
0 bytes spill stores, 0 bytes spill loads. `cuobjdump --dump-sass` (nvdisasm from the
triton package, same note as MC3 and mma_peak): 0 LDL/STL in every kernel. The xsmem
variants intentionally contain LDS (staged x) and STS (one-time per-launch staging);
staging loads/stores sit outside the timed loop by construction (hoisted before
`for (t...)`). Representative census (whole-kernel static SASS counts):

| kernel family | regs | weight LDG (width) | x access | FFMA | FMUL | SHFL | STG | LDL/STL |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `k_legacy` (partial installed replica) | 75 | 37 U16 | L2 | 16 | 32 | 0 | 1 | 0 |
| `k_merge<8,4>` (partial best row) | 40 | 37 U16 | L2 | 16 | 32 | 4 | 1 | 0 |
| `k_merge_v<8,4,32>` (partial u32) | 40 | 16 U16 + 41 U32 | L2 | 16 | 32 | 4 | 1 | 0 |
| `k_merge_v<8,4,128>` (partial u128) | 40 | 16 U16 + 162 U32 + 8 u128 | L2 | 16 | 32 | 4 | 1 | 0 |
| `k_merge_v<8,4,16,xsmem>` (partial u16 smem) | 39 | 3 U16 + 16 S8 + 5 LDG.64 | LDS.128 | 8 | 16 | 4 | 1 | 0 |
| `k_legacy_v<128>` (partial u128 legacy structure) | 73 | 32 U16 + 324 U32 + 16 u128 | L2 | 32 | 64 | 0 | 1 | 0 |
| `k_coop_legacy` (coop installed replica) | 72 | 37 U16 | L2 | 16 | 32 | 4 | 1 | 0 |
| `k_coop_v<2,32>` (coop u32) | 51 | 4 U16 + 11 U32 | L2 | 4 | 8 | 4 | 1 | 0 |
| `k_coop_v<2,128>` (coop u128) | 76 | 4 U16 + 44 U32 + 2 u128 | L2 | 4 | 8 | 4 | 1 | 0 |
| `k_coop_v<2,16,xsmem>` (coop u16 smem) | 42 | 9 U16 + 5 LDG.64 | LDS.128 | 4 | 11 | 4 | 1 | 0 |
| `k_coop_s4<8,xsmem>` (coop split-4) | 40 | 44 U32 + 2 u128 | LDS.128 | 4 | 11 | 5 | 1 | 0 |
| `k_coop_group<2,xsmem>` (group lanes) | 126 | 1 U16 + 1 S8 + 5 LDG.64 | LDS.128 | 16 | 35 | 64 | 1 | 0 |
| `k_q4k_legacy` (q4k installed replica) | 70 | 8 U32 + 32 U16 | L2 | 64 | 16 | 5 | 1 | 0 |
| `k_q4k_qv<16,xsmem>` (q4k winner) | 40 | 5 u128/block (1 hdr + 4 qpack) | LDS.128 | 256+2 HFMA2 | 16 | 3 | 1 | 0 |
| `k_q4k_v<4,128,xsmem>` (q4k grouped u128) | 36 | 16 U32 + 1 u128 + 5 LDG.64 | LDS.128 | 64 | 16 | 5 | 1 | 0 |
| `k_bw_read` (ceiling control) | 28 | 5 u128 | - | 0 | 0 | 0 | 1 | 0 |

SASS facts that explain the results:

1. Q6_K u128 cannot be pure: the 210-byte Q6_K block is 2 mod 16, so the u128 variants
   lower to a heavily predicated LDG.128/LDG.32 mix (e.g. `k_legacy_v<128>`: 16 predicated
   LDG.128 plus 324 LDG.32 plus 32 LDG.U16). The Q6_K rows never get a clean 128-bit load;
   their u128 variants are uniformly slower than the installed u16 loads.
2. Q4_K u128 can be pure: Q4_K blocks are 144 B, 0 mod 16, and the quad-lane mapping
   (`lane>>1` group-of-2 bg, `(lane&1)*4` wc0) makes the 4 qpack windows of a
   (bg, wc-quad) one 16B-aligned uint4 each. `k_q4k_qv` emits exactly 5 LDG.128 per block
   body (1 hdr + 4 qpack) with zero predication; x is staged once per launch (8 KB, LDG.64
   prologue) and read in-loop as LDS.128. This is the only pure-LDG.128 row in the sweep.
3. The xsmem knob removes the installed per-row-block L2 re-read of x (the coop route's
   row_tile=2 layout reads x once per row block) at the cost of one 8 KB smem stage + sync
   per block launch; LDS reads in the loop replace the installed LDG x reads.

## 4. Results (RTX 5090, sm_120, standalone, iters=2000 reps=5 best-of)

### 4.1 Partial 1024x4096 Q6_K (3.44 MB set)

| config | blocks | thr | us/pass | TB/s |
| --- | ---: | ---: | ---: | ---: |
| legacy_32 (installed, external_sum) | 128 | 32 | 12.89-12.92 | 0.27 |
| r8_split4_128thr (L2 best row) | 128 | 128 | 7.36-7.39 | 0.47 |
| legacy_32_u32 | 32 | 32 | 30.71-30.72 | 0.11 |
| legacy_32_u128 | 32 | 32 | 59.66-59.69 | 0.06 |
| legacy_32_u128_pf2 | 32 | 32 | 59.26-59.28 | 0.06 |
| legacy_32_u128_al (interleaved part map) | 32 | 32 | 59.36-59.38 | 0.06 |
| legacy_32_u128_xsmem | 32 | 32 | 97.74-97.77 | 0.04 |
| r8_split2_64thr_u32 | 128 | 64 | 18.90 | 0.18 |
| r8_split2_64thr_u128 | 128 | 64 | 35.35-35.36 | 0.10 |
| r8_split2_64thr_u128_pf2 | 128 | 64 | 33.61-33.62 | 0.10 |
| r8_split4_128thr_u32 | 128 | 128 | 12.97-12.98 | 0.27 |
| r8_split4_128thr_u128 | 128 | 128 | 25.01-25.02 | 0.14 |
| r8_split4_128thr_xsmem (u16) | 128 | 128 | 32.77-32.88 | 0.10 |
| r8_split4_128thr_u128_xsmem | 128 | 128 | 88.54-88.61 | 0.04 |
| r16_split4_256thr_u128_xsmem | 64 | 256 | 111.84-111.96 | 0.03 |
| r32_split4_512thr_u128_xsmem | 32 | 512 | 119.84-120.30 | 0.03 |
| bw read (ceiling) | 512 | 256 | 1.97-2.02 | 1.70-1.75 |

Every new load shape is worse than the installed split-4 row (7.38 us). Wider loads, smem
x staging, part-map interleaving, and prefetch all lose on the 210-byte Q6_K blocks; the
best new row (r8_split4_128thr_u32, 12.98 us) is 1.76x slower than the installed u16
split-4. The binding constraint on this shape is not the per-thread serial chain structure:
splitting the pos chain and widening the loads does not move the number.

### 4.2 Coop-down 4096x12288 Q6_K (41.29 MB set)

| config | blocks | thr | us/pass | TB/s |
| --- | ---: | ---: | ---: | ---: |
| coop_legacy (installed control) | 2048 | 32 | 26.48-26.59 | 1.55-1.56 |
| coop_legacy_u32 | 2048 | 32 | 65.71 | 0.63 |
| coop_legacy_u128 | 2048 | 32 | 108.04 | 0.38 |
| coop_legacy_u128_pf2 | 2048 | 32 | 108.92 | 0.38 |
| coop_legacy_xsmem | 2048 | 32 | 126.42 | 0.33 |
| coop_legacy_u128_xsmem | 2048 | 32 | 219.28 | 0.19 |
| coop_128thr_u128_xsmem | 512 | 128 | 114.01 | 0.36 |
| coop_256thr_u128_xsmem | 256 | 256 | 110.68 | 0.37 |
| coop_512thr_u128_xsmem | 128 | 512 | 109.59 | 0.38 |
| coop_s2_u128 | 4096 | 32 | 99.13 | 0.42 |
| coop_s2_128thr_u128_xsmem | 1024 | 128 | 99.69 | 0.41 |
| coop_s2_256thr_u128_xsmem | 512 | 256 | 109.52 | 0.38 |
| coop_s2_512thr_u128_xsmem | 256 | 512 | 108.96 | 0.38 |
| coop_s4_512thr_u128_xsmem | 512 | 512 | 118.77 | 0.35 |
| coop_group_32thr_u128_xsmem | 2048 | 32 | 219.51 | 0.19 |
| coop_group_128thr_u128_xsmem | 512 | 128 | 113.88 | 0.36 |
| coop_group_256thr_u128_xsmem | 256 | 256 | 109.71 | 0.38 |
| coop_group_512thr_u128_xsmem | 128 | 512 | 109.20 | 0.38 |
| bw read (ceiling) | 512 | 256 | 6.68 | 6.18 |

All 17 new shapes are worse than the installed control (26.5 us / 1.55 TB/s). The coop
route's row_tile=2 layout already reads full 210-byte Q6_K windows per thread; every
repartition (u32/u128, split-reduce, group lanes, smem x) multiplies the load count or the
sync cost without helping. The installed coop row is the local optimum of this surface.

### 4.3 Q4K gate/up 12288x4096 (28.31 MB set)

| config | blocks | thr | us/pass | TB/s |
| --- | ---: | ---: | ---: | ---: |
| q4k_legacy (installed control) | 12288 | 32 | 16.08-16.48 | 1.72-1.76 |
| q4k_legacy_u16 | 12288 | 32 | 18.77 | 1.51 |
| q4k_legacy_u128 | 12288 | 32 | 19.50 | 1.45 |
| q4k_legacy_u128_pf2 | 12288 | 32 | 19.55 | 1.45 |
| q4k_legacy_xsmem | 12288 | 32 | 11.68-11.72 | 2.41-2.42 |
| q4k_4row_128thr_u32 | 3072 | 128 | 16.70 | 1.70 |
| q4k_4row_128thr_u32_xsmem | 3072 | 128 | 10.95-10.98 | 2.58-2.59 |
| q4k_4row_128thr_u128_xsmem | 3072 | 128 | 11.60 | 2.44 |
| q4k_4row_128thr_u128_xsmem_pf2 | 3072 | 128 | 11.59 | 2.44 |
| q4k_8row_256thr_u128_xsmem | 1536 | 256 | 11.98 | 2.36 |
| q4k_16row_512thr_u128_xsmem | 768 | 512 | 11.45 | 2.47 |
| q4k_32row_1024thr_u128_xsmem | 384 | 1024 | 12.99 | 2.18 |
| q4k_16row_128thr_u128_quad_xsmem | 768 | 128 | 10.00-10.15 | 2.79-2.83 |
| bw read (ceiling) | 512 | 256 | 5.09 | 5.57-5.58 |

The q4k shape is the one shape where the load pattern moves the number. The installed
control (16.1-16.5 us / 1.74 TB/s) is beaten by every smem-staged grouping, and the
quad-lane u128 row is the winner: 10.00-10.15 us / 2.79-2.83 TB/s, 1.60-1.65x faster than
the installed control and ~51% of the set's streaming ceiling (5.57 TB/s). The winning
geometry is the only pure-LDG.128 row in the sweep: 768 blocks x 128 threads (16 rows per
block, 8 lanes per row), each thread owns 4 consecutive wc in one group-of-2, loads 5
LDG.128 per block body (1 hdr + 4 strided qpack windows), reads x from smem (LDS.128), and
reduces with a 3-step XOR ladder (xor 4/2/1) over the 8 lanes; 40 registers, 0 spills.

### 4.4 L2 residency and occupancy (rows subset)

| row | rows | us/pass | TB/s | note |
| --- | ---: | ---: | ---: | ---: |
| r8_split4_128thr_u32 | 1024 | 12.97-12.98 | 0.27 | full set |
| r8_split4_128thr_u32 | 256 | 12.93 | 0.07 | flat: latency/occupancy-bound, not BW-bound |
| r8_split4_128thr_u32 | 64 | 12.87 | 0.02 | flat |
| coop_legacy | 4096 | 26.48-26.59 | 1.55 | full set |
| coop_legacy | 1024 | 14.21-14.25 | 0.73 | ~3.6-3.9 us fixed overhead + linear |
| q4k_16row_128thr_u128_quad_xsmem | 12288 | 10.00-10.15 | 2.79-2.83 | full set (L2-resident) |
| q4k_16row_128thr_u128_quad_xsmem | 4096 | 3.79 | 2.49 | 256 blocks |
| q4k_16row_128thr_u128_quad_xsmem | 2048 | 2.99 | 1.58 | 128 blocks: under-occupied grid |

The partial route is flat across a 16x row range (12.87-12.98 us): shrinking the L2-resident
set changes nothing, confirming the partial binding constraint is latency/occupancy, not
bandwidth or set size. The q4k quad row degrades to 1.58 TB/s at 128 blocks (below 1
block/SM at 170 SMs) but runs at 2.79-2.83 TB/s at the full 768-block geometry; the real
route launches at the full grid, so the full-shape number is the go/no-go evidence.

## 5. Mandatory controls

| control | recorded | standalone (this probe) | verdict |
| --- | --- | ---: | --- |
| `q6k_gen_partial_1024_4096_4` (LOCAL:0:32) | 12.92 us standalone / 17.15 in-loop | 12.89-12.92 us | reproduced (offset ratio 1.33, within the documented 1.3-1.6x) |
| split-reduce-4 best row | 7.38 us (L2 record) | 7.36-7.39 us | reproduced |
| `q6k_gen_coop_4096_12288_inkernel` | 34.90 us in-loop | 26.48-26.59 us | reproduced (offset ratio 1.31-1.32, within 1.3-1.6x) |
| `q4k_g3_lanemap_gemv_12288_4096` | 20.69 us in-loop | 16.08-16.48 us | reproduced (offset ratio 1.26-1.29) |
| 466.6 us anomaly (L2 record section 5) | explained, not re-litigated | - | respected |

Methodology offset (same note as the L2 and MC3 records): recorded numbers are per-token
kernel medians inside the real decode loop; this microbench times back-to-back passes
inside one launch with the weight set L2-resident and sustained clocks. Standalone is
systematically ~1.3-1.6x faster, and the offset is uniform across the working controls, so
within-microbench comparisons and the go/no-go ranking are unaffected.

## 6. Go/no-go per decomposition

| decomposition | best legal row | llama-class floor | verdict |
| --- | --- | --- | --- |
| partial 1024x4096 Q6_K k/v | 7.36-7.39 us / 0.47 TB/s (installed split-4; all 14 new shapes worse) | 3.3 us / 1.04 TB/s | NO-GO |
| coop-down 4096x12288 Q6_K | 26.48-26.59 us standalone / 34.90 in-loop / 1.55 TB/s (installed control; all 17 new shapes worse) | 29.3 us / 1.4 TB/s | NO-GO |
| gate/up pair 12288x4096 Q4_K | quad u128 smem row: 10.00-10.15 us per kernel standalone; pair 20.0-20.3 us standalone; in-loop pair estimate 25.2-34.5 us (INFERRED: x1.26-1.7 offset) | 37.9 us (llama fused pair) | CLEARS (INFERRED for in-loop; OBSERVED standalone) |

Partial and coop-down close NO-GO exactly as the L3/L5 closures: no legal load shape clears
the floor, and the installed decompositions are the local optima of the swept surface. The
q4k gate/up pair clears the floor on the quad u128 shape under every measured
standalone-to-in-loop offset convention: the measured q4k offset is 1.26-1.29x (20.69 /
16.1-16.5), the L2 record documents up to 1.6x, and even 1.7x keeps the pair estimate
(34.5 us) under the 37.9 us floor. The verdict is therefore robust to the methodology
offset; the isolated same-session d512 wall (scope 5.3) remains the landing-scope ranking
step and is not run here (see section 8, deviations).

## 7. Winning row and exact additive-route design (coordination handoff)

The q4k gate/up shape clears the llama-class floor, so per the MC2 coordination handoff the
additive route family is NOT implemented: MC1/MC3 run in parallel and own the same files
(`decode_routes.py` / `decode_kernels.py`); concurrent edits are not allowed. This record
carries the winning row and the exact design:

1. Winning row: `k_q4k_qv<16>` geometry as probed - 768 blocks x 128 threads at
   12288x4096 (16 rows/block, 8 lanes/row), thread owns wc-quad `(lane&1)*4` in
   group-of-2 `lane>>1`, 4 qpack windows + 1 hdr as pure LDG.128 per block body, x staged
   to smem once per launch (8 KB, outside the timed loop), LDS.128 x reads in-loop,
   3-step XOR ladder (4/2/1), 40 regs, 0 spills, 10.00-10.15 us / 2.79-2.83 TB/s
   standalone = ~51% of the 5.57 TB/s streaming ceiling.
2. Additive route design: new kernel builder in `decode_kernels.py`, e.g.
   `q4k_g3_lanemap_gemv_qv_kernel(rows, k)` emitting the quad-lane u128 shape with the
   per-thread wc-quad mapping above; new kernel name following the house suffix convention
   (e.g. `q4k_g3_lanemap_gemv_qv_12288_4096`). Legacy `q4k_g3_lanemap_gemv_12288_4096`
   rows and hashes untouched. Per-target admission: NV sm_120 only initially (CUDA-only
   gate, matching the L4 vocab pattern); AMD/Metal admitted routes keep the 10 legacy pg3
   hashes + `add50a7aa43f` unchanged and would carry their own pg3 hash for the new
   kernel. The kernel-count question is MC3's, not MC2's: the quad route replaces each of
   the two gate/up kernels' internals, not their count.
3. Real-route cost notes: each launch pays one 8 KB smem x stage + sync per block
   (amortized to noise in the probe's 2000-pass loop; INFERRED ~0.3-1 us per launch in the
   real loop, covered by the floor margin). The estimate above uses the probe's
   amortized standalone numbers.
4. Landing gates: pg3 re-derive, NV pins 3/3, isolated same-session d512 wall vs llama,
   like-for-like cap. Node-sum upper bound for the gate/up roles only:
   72 x (pair 41.4 - 25.2/34.5 estimate) us = 0.5-1.2 ms upper bound (INFERRED arithmetic,
   for parent sizing; no composed forecast, no wall claim). Per like-for-like cap
   discipline the isolated same-session measurement replaces the bound.

## 8. Controls

- pg3 decode render-equality (HIPRenderer gfx1100, render-only, CPU-only, no lock): all
  10 legacy hashes byte-identical to the pinned table (312422c73a49 / 27857cb8ca03 /
  851760e2053c / 39ddb717ddd4 / cc38fbb3db92 / 5795e66a7292 / 344e1c388eeb /
  c708302aa2d2 / 66d4c4da3108 / c78e4651ad35), and the M2 promoted fused row
  `q6k_gen_coop_4096_12288_inkernel` = `add50a7aa43f` holds. Script exits 0.
- NV pins: no decode code changed (the probe is a standalone .cu; `git diff e71f2ef17..HEAD`
  on `decode_kernels.py` / `decode_routes.py` is empty, so the decode tree is identical to
  the MC3 record's pin runs). Pins state unchanged: first token 151936, fixed-depth token
  sha256 `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, decode sha256
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, census candidate_set
  sha256 `1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`. No harness
  rerun was performed (CPU-only control requirement; no decode-code delta exists to pin).
- GPU discipline: every run under `flock /tmp/nv_gpu.lock` with 0% util at acquisition;
  lock file untouched; fused prefill attention disabled in any decode context (none used
  here).

## 9. Deviations

- No new probe file: the sweep extends `l2_q6k_partial_sweep.cu` in place (the task's
  preferred option). The original flags, CFGS table, kernels, and output format are
  byte-behaviorally intact; the mandatory controls reproduce the L2 rows.
- Probe development notes (all fixed before any number was believed; the committed file and
  the final binary are clean): (1) `stage_x` originally copied half the vector (XHALVES/8
  instead of /4), making legacy parts 2/3 read zeros - the root cause of the earlier
  xsmem anomaly, fixed and verified by debug dumps; (2) coop row_tile writes were gated on
  `threadIdx.x == 0`/`lane == 0` instead of `pos == 0`/`grp == 0`, writing only even rows;
  (3) the coop split-4 ladder was missing the `xor 2` step; (4) the q4k u128 qpack windows
  were contiguous instead of strided per group-of-2; (5) the q4k quad x base was missing
  the x4 scaling; (6) the partial interleaved-map rows compare row totals, not per-part;
  (7) legacy-family partial rows needed a separate part-buffer sized ROWS*PARTS; (8) all
  smem x buffers are 16B-aligned; (9) the power-of-2 `--rows` check applies only when
  `--rows` is explicit. Final rebuild: 71/71 kernels, 0 spills, all spot checks pass.
- `nvdisasm` ships in the triton package, not the CUDA toolkit; it was added to PATH for
  the cuobjdump SASS dump (same note as MC3 and mma_peak).
- Isolated same-session d512 wall: not run. The cleared shape is probe-only; the wall
  requires the additive route wired into the emitters, which is exactly the concurrent-edit
  ban while MC1/MC3 run. The wall is recorded as the landing-scope ranking step.
- Residency rows use powers of two (the q4k 3072-row intermediate was skipped by the
  probe's power-of-2 check; the installed shapes are powers of two).
- H4 is an 8B-aligned 4-halfword struct: CUDA 13.2 has no `__half4`; vector x loads use it.
- Worktree at record time: tracked tree clean except the probe file (this record's
  [test] commit) and user-owned doc modifications (`docs/README.md`,
  `docs/beating-llama-first-principles-20260731.md`, `docs/what-makes-inference-fast.md`);
  untracked user-owned artifacts (`dp4a_peak_cuda*`, `flash_score_tile_peak_cuda`,
  `l2_q6k_partial_sweep`, `q6k_vocab_coop_ceiling_cuda`,
  `scratchpad/t6_metal_admission_probe.py`) untouched.

## 10. Closeout

MC2 diagnostic complete: partial and coop-down close NO-GO with the ceiling/floor tables
as the permanent record; the q4k gate/up pair clears the llama-class floor on the quad
u128 smem-staged load shape (10.00-10.15 us per kernel standalone, 2.79-2.83 TB/s).
Verdict: **floor cleared - implementation pending parent coordination**. No implementation
code, no promotion, no push; parent review happens on the commits.

## 11. References

- `decode-gemv-instruction-bandwidth-scope-20260803.md` sections 2, 4.2, 5, 10
- `l2-q6k-partial-singlepass-measurement-record-20260803.md` (floors, offset, anomaly)
- `mc3-w1w3-fusion-measurement-record-20260803.md` (sibling record, pin state, style)
- `like-for-like-cap-settling-record-20260803.md` (class census, 0.597 ms delta, cap)
- Probe: `extra/llm_research/microbench/l2_q6k_partial_sweep.cu`
