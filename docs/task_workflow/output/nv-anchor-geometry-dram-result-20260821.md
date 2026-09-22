# NV anchor DRAM/occupancy bracket: K/V are the low-bandwidth anchors, not gate_up

Date: 2026-08-21

Commit: `6570abc025514273faa100c66b979e531585a1e1`

Evidence: `docs/task_workflow/evidence/nv-anchor-geometry-dram-20260821/`

Tool: `extra/llm_research/decode/nv_anchor_geometry_dram.py`

## 1. Decision

**The attention-side low-bandwidth anchors are K and V, not gate_up.**  A
prior-session estimate labelled `gate_up` at ~742 GB/s by counting only one of
its two weight tensors.  Correcting that byte count moves gate_up to ~1492
GB/s (near peak), while K sits at ~508 GB/s and V at ~788 GB/s.  This is
measured with ncu DRAM/occupancy counters on the exact production CUDA source
compiled standalone, the same method as the 2026-08-14 FFN-down occupancy
proof.

No production change and no performance promotion follow from this record.

## 2. Per-anchor effective bandwidth (from the canonical capture)

Weight bytes are arithmetic from the Q4_K/Q6_K layouts; durations are the
median HCQ profile durations in
`docs/task_workflow/output/nv-rmsnorm-phaseB-control-20260820.json`.

| kernel | n | median us | weight bytes | effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| gate_up `w1w3fused16_12288_4096` | 36 | 37.95 | 56,623,104 | 1492 |
| down `q6k...epi_ffnresadd_4096_12288` | 18 | 30.37 | 41,287,680 | 1360 |
| Q `q4k_g3_lanemap_gemv_4096_4096` | 19 | 8.83 | 9,437,184 | 1069 |
| O `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` | 36 | 9.09 | 9,437,184 | 1038 |
| **V `q6k_v_four_warp_fp16_direct_1024_4096`** | 10 | 4.37 | 3,440,640 | **788** |
| **K `q4k_g3_lanemap_gemv_1024_4096`** | 28 | 4.64 | 2,359,296 | **508** |
| total | 147 | 2602.02 | 3,401,121,792 | 1307 |

The prior-session "gate_up ~742" was exactly half the correct 1492: it counted
one Q4_K projection instead of the two (gate + up) the fused kernel streams.
The two real low-bandwidth anchors are K and V.

## 3. ncu DRAM and occupancy bracket

Rendered production source, compiled with nvcc `-arch=sm_120a -O3`, profiled
with ncu 2026.2.1 isolated cold replay (L2 flushed).  Each row is one kernel
launch.

| kernel | grid x block | regs/thread | warps_active | dram throughput |
| --- | ---: | ---: | ---: | ---: |
| K | 1024 x 32 (1 warp/row) | 61 | 11.9% | 21.2% |
| V | 1024 x 128 (4 warps/row) | 39 | 47.2% | 30.9% |
| gate_up | 12288 x 32 (1 warp/row) | 74 | 44.8% | 79.2% |

`dram__throughput.avg.pct_of_peak_sustained_elapsed` and
`sm__warps_active.avg.pct_of_peak_sustained_active`.  The raw
`dram__bytes_read.sum` counter returned `n/a` in `--metrics` mode, so the byte
counts in section 2 remain the arithmetic source; the throughput percentage is
the hardware-normalized counter.

## 4. First-principles read (inferred)

Occupancy alone does not explain the spread.  V and gate_up have nearly
identical occupancy (47.2% vs 44.8%) but gate_up achieves 2.6x the DRAM
throughput (79.2% vs 30.9%).  The controlling difference is kernel duration:

- gate_up streams 56.6 MB for ~38-41 us, long enough to hide the ~500 ns DRAM
  latency ramp and reach steady-state bandwidth.
- V streams 3.44 MB for ~4.4-6.4 us, and K streams 2.36 MB for ~4.6-6.4 us;
  both are too short to amortize the latency ramp.  K is additionally
  row-starved: 1024 blocks of 1 warp each lands at ~12% occupancy.

So the lever is not "make gate_up faster"; it is the K/V support: two short,
low-parallelism streams on the Q->O critical path.  The arithmetic ceiling is
instructive: K+V is 5.80 MB/token-layer, whose DRAM floor at 1700 GB/s is
~3.4 us, against the measured ~9.0 us of the two serial kernels.

## 5. Relationship to prior records

- The 2026-08-14 FFN-down proof (1 warp/row -> 38.8% occupancy -> 54.5% DRAM)
  was about a long 4096x12288 kernel.  That occupancy->DRAM chain does not
  transfer verbatim to the short KV kernels; duration, not occupancy, is the
  dominant term there.
- The 2026-08-05 P4 two-queue cut record rejected queue-level co-scheduling of
  K/V/QKV ("all work is MMQ").  That is a different mechanism from fusing K+V
  into one longer kernel stream; this record does not reopen or contradict it.

## 6. Labels

- `observed`: ncu DRAM throughput and occupancy counters, register counts,
  production kernel identities and grid/block geometry.
- `inferred`: the per-anchor effective GB/s (weight bytes divided by median
  HCQ duration), the duration-vs-occupancy attribution, and the K+V DRAM floor.
- `unmeasured`: the end-to-end wall effect of a K+V fusion or split-K arm; no
  candidate kernel was built or timed here.  `dram__bytes_read.sum` is also
  unmeasured for these kernels under `--metrics` mode.

## 7. Evidence

- `docs/task_workflow/evidence/nv-anchor-geometry-dram-20260821/result.json`
  (schema `tinygrad.nv_anchor_geometry_dram.v1`).
- `docs/task_workflow/evidence/nv-anchor-geometry-dram-20260821/sha256.txt`.
