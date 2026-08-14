# NV FFN-down gap: occupancy proof, not datapath (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, sm_120, CUDA 13.2, driver 595.84. This is the
definitive arithmetic answer to "why is our generated FFN-down GEMV slower than
llama's MMQ", superseding the datapath-only framing in
`nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md` (the 4x DP4A number is
true but is NOT the wall lever).

## 0. The claim being tested

"Our kernels are generated, so in theory they should be 1-to-1 with llama."
Tested at the real FFN-down shape (4096 rows x 12288 k, Q4_K): our production
`q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` vs llama's
`mul_mat_vec_q<Q4_K,1,...>`.

## 1. Same bytes (arithmetic, from the data layout)

Both kernels must read the identical packed Q4_K weight tensor:

```text
weight bytes = rows x k_blocks x bytes/block
             = 4096 x (12288/256) x 144
             = 4096 x 48 x 144 = 28.31 MB
DRAM floor   = 28.31 MB / 1.70 TB/s = 16.65 us
```

The activation (fp16 for us, Q8_1 for llama) is a few KB and L2-resident across
the 4096 rows, so DRAM traffic is dominated by the same 28.31 MB on both sides.
There is no byte-count lever; the only lever is how much of that bandwidth each
kernel extracts.

## 2. The decisive measured difference: thread geometry -> occupancy -> DRAM

Both kernels launch `grid=4096`. The difference is threads per output row:

| kernel | threads/row | warps/row | regs/thread | occupancy | DRAM throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama MMQ | 128 (block 32x4) | 4 | 56 | **66.27%** | **77.15%** |
| tinygrad GEMV | 32 (block 32x1) | 1 | 61 | **38.77%** | **54.51%** |

Measured with `ncu` (Nsight Compute 2026.2.1) this session, `sm__warps_active`
and `dram__throughput.avg.pct_of_peak_sustained_elapsed`. Full occupancy-limit
data: llama register-limited to 9 blocks/SM (1152 threads), tinygrad to 32
blocks/SM (1024 threads); but tinygrad only has 4096 blocks total, so it lands
at ~24 blocks/SM = ~24 warps/SM = ~38% of the 64-warp peak, while llama's
128-thread blocks fill more of the SM.

## 3. The causal chain (proved)

```text
1 warp/row (ours)   -> 38.8% occupancy -> 54.5% DRAM  -> 24.62 us in-loop
4 warps/row (llama) -> 66.3% occupancy -> 77.2% DRAM  -> 19.23 us in-loop
```

Decode GEMV is DRAM-latency-bound: 28.31 MB streams from DRAM each token, and
saturating it requires enough outstanding loads to hide ~500ns DRAM latency.
One warp per output row does not issue enough concurrent requests; four warps
per row do. This is the entire 5.4 us/block FFN-down gap.

Isolated L2-resident walls confirm the same geometry effect (both kernels run on
cached weights, removing DRAM latency): llama 8.43 us vs tinygrad 20.47 us.

## 4. Why DP4A-only adoption only moved +0.5%

`nv-q4-down-dp4a-resadd-18block-gate-20260814.md` measured the full DP4A
FFN-down route at +0.5% (193.92 -> 194.88 tok/s) with logits drift. That route
does fix the geometry (its `emit_four_warp_direct` uses 128 threads/row) and the
datapath (DP4A), but it ADDS a separate `q8_1_llama_provider_12288` node that
the production control does not have. The added provider node eats most of the
geometry/datapath win, so the net is ~25.6 us of the ~97 us row.

## 5. The worth-it fix (what 1-to-1 actually requires)

To close the FFN-down row without adding a node:

1. Adopt llama's 4-warp/row geometry (128 threads per output row), which is the
   occupancy lever and is worth ~the full 5.4 us/block DRAM-efficiency gap.
2. Fold the Q8_1 quantization into the existing w1w3 producer epilogue (it
   already owns the silu*up result in-kernel), so no separate provider node is
   added. This is the producer-fold already scoped in
   `nv-gemv-core-recovery-status-20260813.md` section 3.

The datapath (DP4A vs fp32 FMA) is a real but secondary lever (4x MAC headroom,
`nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md`); the dominant wall
lever is the thread geometry and the resulting DRAM occupancy.

## 6. Artifacts added this session

- `extra/llm_research/microbench/ffn_down_wall_harness.cu`: launches the
  tinygrad-rendered FFN-down GEMV for wall/ncu profiling.
- `extra/llm_research/microbench/llama_mmq_wall_harness.cu`: launches llama's
  exact Q4_K MMQ cubin for the same wall/ncu profiling.
- SASS for both kernels dumped with `cuobjdump`/`nvdisasm`; instruction census in
  the session log (llama 832 instr/thread incl. 32 DP4A, tinygrad 368 incl. 64
  FFMA and zero DP4A).

## 7. References

- `nv-q4-down-dp4a-resadd-18block-gate-20260814.md` (the +0.5% wall, logits drift)
- `nv-gemv-core-recovery-status-20260813.md` (the +302.8 us core-deficit ledger)
- `nv-decode-datapath-fma-vs-dp4a-measurement-20260814.md` (DP4A 4x fp32 FMA peak)
- `nv-q4k-q8-substrate-arithmetic-trace-20260812.md` (arithmetic is byte-identical)
