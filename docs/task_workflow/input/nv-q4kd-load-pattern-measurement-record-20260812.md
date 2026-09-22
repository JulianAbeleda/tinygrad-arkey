# Q4 FFN-down load-pattern measurement record - 4096x12288 Q4_K, sweep extension (rank-1 closure)

Date: 2026-08-12
Status: measurement record. Authorized by `nv-quant-gemv-llama-audit-20260812.md`
section 7 (rank 1: the Q4 FFN-down 4096x12288 shape was never load-pattern swept; run the
MC2-style surface, gate on the llama-class floor, keep the MC2 lesson: a standalone winner
is NOT evidence until it survives the in-loop census). Diagnostic probe and measurement
only: no route admission, no emitter change in the runtime. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `7f1fa850d` (this record's [test] + [docs] commits land on
top of it; the decode tree `decode_kernels.py` / `decode_routes.py` is byte-identical to
the audit's control census).

## 1. Why this record exists

The audit's per-shape deficit table left exactly one untried surface on the quant row:
the Q4 FFN-down shape (4096x12288), our largest per-shape ratio deficit (2.27x llama:
26.75 us in-loop median vs llama's 11.776 us/node floor) and the largest untouched kernel
family by mass (18 kernels, 481.50 us/token). MC2 swept the partial Q6, coop-down Q6, and
Q4 gate/up shapes in 08-03 but never the Q4 down shape. This record sweeps the same
surface family (vec width / rows-per-block / x smem staging / prefetch / quad-lane u128)
on the down geometry and gates it against the llama floor.

## 2. Protocol

Probe: `extra/llm_research/microbench/l2_q6k_partial_sweep.cu` extended in place. The
q4k gate/up family (`k_q4k_legacy`, `k_q4k_v`, `k_q4k_qv`) is templated on (NKB, BPG,
XHALVES) so the identical bodies serve the down shape with `NKB=48, BPG=12, XHALVES=12288`
(Q4KD rows 4096, K 12288, 48 blocks/row, 12 blocks/group); the gate/up shapes keep
`NKB=16, BPG=4, XHALVES=4096`. New `--shape q4kd` selects the 13-row CFGS2 table (shape=3):
control replica, vec u16/u128, prefetch-2, x smem, 4/8/16/32-row grouping, and the
quad-lane u128 row. All 84 kernels in the binary compile with 0 spills, 0 stack frames
(`--ptxas-options=-v`).

- Session: this workspace, branch `nvidia-bringup-20260731`, HEAD `7f1fa850d` + this
  record's commits.
- Config: NVIDIA GeForce RTX 5090 (sm_120), CUDA 13.2,
  `nvcc -O3 -arch=sm_120 -std=c++17 --ptxas-options=-v`. Deterministic synthetic packed
  weights/x (finite fp16 d slots, bounded scales, random nibbles), no model load.
- Evidence classes: OBSERVED = measured this session under the lock; INFERRED = arithmetic
  (flagged inline). Lifecycle vocabulary only; no composed forecasts, no wall claims.
- Timing: best-of-N back-to-back passes inside one kernel launch (`cudaEventElapsedTime`),
  `--iters 2000 --reps 5` for all rows, `--reps 10` re-confirmation on the control and the
  two best rows. Per-pass time = measured ms / iters. Every GPU run was serialized with
  `flock /tmp/gpu-bench.lock -c "<cmd>"` with 0% GPU utilization confirmed at lock
  acquisition; the lock file was never modified or deleted.
- Numerics: every non-control row is spot-checked against the installed control replica's
  output (row totals), one pass. Max relative error 2.856e-3 to 3.427e-3 across all rows
  (fp32 reassociation noise on magnitudes ~1e6-1e8).
- BW ceiling: `--bw` streams the exact Q4_K weight set (144 B/block, 4096 x 48 = 28.31 MB)
  with `k_bw_read` as the L2-resident achievable-bandwidth control.

## 3. Purity - ptxas

All 84 entry points report 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads.
The q4kd instantiations of the four templated kernels are present (`k_q4k_legacy<48,12>`,
`k_q4k_v<...48,12,12288>`, `k_q4k_qv<16,true,48,12,12288>`) alongside the gate/up
instantiations (`<16,4,4096>`), so both shapes share the exact same SASS-quality gate as
the MC2 record: no spills anywhere in the binary.

## 4. Results (RTX 5090, sm_120, standalone, iters=2000 reps=5 best-of)

Q4 FFN-down 4096x12288 (28.31 MB set, 48 blocks/row):

| config | blocks | thr | us/pass | TB/s |
| --- | ---: | ---: | ---: | ---: |
| q4kd_legacy (installed control) | 4096 | 32 | 18.47-18.78 | 1.51-1.53 |
| q4kd_legacy_u16 | 4096 | 32 | 21.12 | 1.34 |
| q4kd_legacy_u128 | 4096 | 32 | 22.17 | 1.28 |
| q4kd_legacy_u128_pf2 | 4096 | 32 | 22.03 | 1.29 |
| q4kd_legacy_xsmem | 4096 | 32 | 23.65 | 1.20 |
| q4kd_4row_128thr_u32 | 1024 | 128 | 19.41 | 1.46 |
| q4kd_4row_128thr_u32_xsmem | 1024 | 128 | 11.81-11.88 | 2.38-2.40 |
| q4kd_4row_128thr_u128_xsmem | 1024 | 128 | 13.20 | 2.15 |
| q4kd_4row_128thr_u128_xsmem_pf2 | 1024 | 128 | 12.84 | 2.21 |
| q4kd_8row_256thr_u128_xsmem | 512 | 256 | 13.95 | 2.03 |
| q4kd_16row_512thr_u128_xsmem | 256 | 512 | 13.24 | 2.14 |
| q4kd_32row_1024thr_u128_xsmem | 128 | 1024 | 12.45-12.47 | 2.27 |
| q4kd_16row_128thr_u128_quad_xsmem | 256 | 128 | 11.43-11.45 | 2.47-2.48 |
| bw read (ceiling) | 512 | 256 | 5.07 | 5.58 |

All spot checks pass (max|rel| 2.856e-3 to 3.427e-3). The two best rows are stable across
reps 5 and reps 10 (control 18.47/18.78, 4row u32 xsmem 11.81/11.88, quad 11.43/11.45).

## 5. Mandatory controls and offset

| control | in-loop (census) | standalone (this probe) | verdict |
| --- | ---: | ---: | --- |
| `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` (M2b) | 26.75 us (median, x18) | 18.47-18.78 us | reproduced (offset ratio 1.42-1.45, within the documented 1.26-1.7x) |
| `q4k_g3_lanemap_gemv_12288_4096` (gate/up control, regression check) | 20.69 us (MC2 record) | 16.17 us | reproduced (offset ratio 1.28; MC2 record: 16.08-16.48 us, offset 1.26-1.29) |

Methodology offset (same note as the L2/MC2/MC3 records): recorded numbers are per-token
kernel medians inside the real decode loop; this microbench times back-to-back passes
inside one launch with the weight set L2-resident and sustained clocks. Standalone is
systematically ~1.26-1.7x faster, and the offset is uniform across the working controls,
so within-microbench comparisons and the go/no-go ranking are unaffected.

## 6. Go/no-go - the Q4 down shape does NOT clear the floor

Llama floor: 11.776 us/node in-loop. At the measured offset conventions (1.26-1.7x), the
standalone equivalent floor is ~6.9-9.3 us. The best row is the quad u128 smem at
11.43-11.45 us standalone (2.47-2.48 TB/s, 44% of the 5.58 TB/s ceiling) - 1.23-1.65x
above the standalone floor, and 1.58-1.64x faster than the installed control.

In-loop estimate for the best row (INFERRED): 11.43-11.45 x 1.26-1.7 = 14.4-19.5 us,
i.e. 1.22-1.65x ABOVE llama's 11.776 us floor even at the most optimistic 1.26x offset
(14.4 us > 11.776 us). At the measured control offset for this exact shape (1.42-1.45x),
the estimate is 16.2-16.6 us = 1.38-1.41x above the floor.

Verdict: NO-GO. No load-pattern row on the Q4 FFN-down shape clears the llama-class floor
under ANY offset convention; the gap is not movable by the load-pattern surface. The Q4
down row closes with this floor table, matching the audit's gate: "if not, the Q4 down row
closes NO-GO with a floor table and the quant row is exhausted except for closed
mechanisms." Per the MC2 lesson no in-loop census is warranted: a standalone winner must
clear the standalone floor first (MC2's gate/up quad cleared standalone but regressed
in-loop); here no row clears even the standalone floor, so there is no winner to census.

## 7. Ledger impact

- Q4 FFN-down (rank 1 of the audit): closed NO-GO. Remaining mass on this row stays with
  the installed `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` (481.50 us/token, 18
  kernels) until a non-load-pattern mechanism is named. The shared-Q8 FFN-down consumer
  was already WALL NO-GO (+151.192 us/token, audit section 4).
- Quant GEMV class: with rank 1 closed, every per-shape surface from the audit has now
  been swept or is closed (rank 2 partial Q6 NO-GO, rank 3 shared-Q8 lease landed/closed,
  rank 4 vocab near-parity). No new additive-route candidate is admitted by this record.
- The gate/up and down shapes share identical kernel bodies; the down shape's extra
  blocks/row (48 vs 16) and reduced row count (4096 vs 12288) are the only differences,
  and the sweep shows the down geometry does not transfer the gate/up win: the same quad
  u128 geometry is 11.43 us on down vs 10.00-10.15 us on gate/up for the same 28.31 MB
  set (256 vs 768 blocks).

## 8. References

- `nv-quant-gemv-llama-audit-20260812.md` (rank 1 gate, per-shape deficit table)
- `mc2-load-pattern-measurement-record-20260803.md` (sweep surface, controls, offset)
- `nv-decode-llama-live-gemv-route-audit-20260805.md` (llama path, per-shape medians)
- Control census: `/tmp/m1_costgate_ab_fixed.json` (control arm, q4k down 26.75 us x18)
- Commit `7f1fa850d` (audit), this record's [test] commit (probe extension) and [docs]
  commit (this file)

## 9. Deviations

- No new probe file: the sweep extends `l2_q6k_partial_sweep.cu` in place (MC2
  precedent). The gate/up rows, CFGS table, CLI flags, and output format are
  byte-behaviorally intact; the mandatory controls reproduce the MC2 gate/up rows
  (control 16.17 vs 16.08-16.48).
- The q4k template parameters were renamed `KB -> NKB` to dodge the file's `#define KB 16`
  (the preprocessor was mangling the template declarations); the rename is internal to the
  templated kernels and the emitted SASS is unchanged for the gate/up rows (control
  reproduced exactly).
- CUDA 13.2 rejects default template arguments on `__global__` templates, so all call
  sites pass explicit template args (gate/up rows unchanged: `<Q4K_KB,4,Q4K_K>`; q4kd rows:
  `<Q4KD_KB,Q4KD_BPG,Q4KD_K>`).
- Worktree at record time: tracked tree clean except the probe file (this record's [test]
  commit) and this document.
