# Coalesced packed-word Q4_K scheduler GEMV — result (2026-06-25)

## Verdict: **`Q4K_PACKED_SCHEDULER_GEMV_TRAILS_OWNED`** — packed-word coalescing is NOT reachable in pure Tensor ops; it needs a `CUSTOM`/codegen capability

Follows M6 (`docs/scheduler-gemv-vs-owned-result-20260625.md`), which found a scheduler GEMV ~2× off the owned warp
kernel with the gap = the Q4_K dequant lifecycle + **coalesced packed-word loads + block-group-K** (not the reduce).
This step attacked the packed-word coalescing directly: a word-structured tinygrad-ops dequant whose load unit is
the uint32 word (`extra/qk_q4k_scheduler_gemv.py`, `Q4K_GEMV_SCHEDULER=2`).

## What was built (and verified correct)
- `q4k_dequant_words` (Tensor ops over `[…,n_blocks,36]` uint32) — layout validated **byte-exact (max_abs_diff 0.0)**
  against the gguf reference (`ggml_data_to_tensor` type 12).
- `q4k_scheduler_matvec` — fused dequant→matvec; matches a dense reference matvec on all finite entries.
- Routed in-model for FFN gate/up (`model.py` `Q4K_GEMV_SCHEDULER==2`); **greedy tokens match the owned route** in
  the 3-arm W==D.

## Measurement (clock-pinned interleaved W==D, FFN gate/up, `extra/qk_q4k_packed_gemv_wd.py`)

| ctx | owned | sched_fp16 (M6) | sched_packed (this) | packed vs fp16 | packed vs owned |
|----:|----:|----:|----:|----:|----:|
| 512  | 103.1 | 50.6 | **22.4** | −125% | −359% |
| 1024 | 101.3 | 50.1 | **22.4** | −124% | −353% |
| 2048 | 98.7  | 49.5 | **22.2** | −122% | −344% |
| 4096 | 94.1  | 48.3 | **22.0** | −119% | −328% |

`tokens_match` True all arms/ctx; route verified (owned: 72 owned gate/up kernels; sched arms: 0). The
word-structured packed GEMV is **worse** than the fp16-logical scheduler GEMV — restructuring the dequant to read
words made the access pattern *worse*.

## Root cause (DEBUG=4 of the routed kernel `r_128_32_3_16_8_8_4`)
For the large gate/up output (12288), the matvec heuristic chooses **output-parallel** (`gidx0*32 + lidx0` over
output rows, 3-row upcast) + **serial-K** (`Ridx0<16` over the 16 k-blocks). The packed-word loads are therefore
**strided per output row** — `alu0 = gidx0*55296 + lidx0*1728`, so adjacent lanes (`lidx0`, `lidx0+1`) read words
**1728 apart** (= 3 rows × 576 words/row) — i.e. **uncoalesced**, and K is never grouped onto lanes (so the
cross-lane lowering doesn't even apply here). The owned kernel does the opposite: **one workgroup per row**, lanes =
within-block word-col (`lane4=pos//4`) + 4-way block-group-K → 8 adjacent lanes read 8 adjacent words (coalesced).

The thread-map is the **heuristic's** choice, not the dequant expression's — so restructuring the Tensor-ops dequant
cannot impose the owned kernel's per-row-workgroup + word-col-lane mapping, and the lowerer does not synthesize a
coalesced packed-word load from the gather. My explicit per-block word indexing just produced a worse (strided)
pattern.

## Conclusion
A **pure-scheduler** Q4_K GEMV cannot match the owned kernel: fp16-logical caps ~2× off, word-structured packed is
~4.6× off (worse). The owned kernel's advantage — **coalesced packed-word load + per-row-workgroup K-split
thread-map** — is a **`CUSTOM`/codegen capability**, NOT reachable by restructuring the dequant expression or by the
MV opts. This is the precise missing search-space primitive (recorded in `decode_ffn_gemv_gfx1100_v1.json`):
> a codegen capability to emit a coalesced packed-word load with a custom per-row-workgroup + word-col-lane
> thread-map (what the owned kernel hand-codes via `CUSTOM`).

This closes the "can a pure scheduler GEMV match the owned kernel" question: **no**, with the gap precisely
attributed. The M5 cross-lane lowering remains the only in-model-proven scheduler primitive for the GEMV, but it is
low-leverage; the GEMV's owned-kernel edge is a thread-map/coalescing codegen gap, not the reduce and not byte count.

## Status
Research only; **no default change**. `Q4K_GEMV_SCHEDULER` (1=fp16-logical, 2=word-packed) and `WARP_REDUCE_LOWERING`
stay default-off; the owned warp GEMV remains the shipped default (owned arm reproduces ~94–103 tok/s). Next: either a
`CUSTOM`/codegen capability for coalesced packed-word loads + custom thread-map (deep), or pivot to the `v_dot2`
renderer lowering (the attention tile's gap). Artifacts: `bench/qk-scheduler-gemv-vs-owned/packed_wd.json`,
`extra/qk_q4k_scheduler_gemv.py`, `extra/qk_q4k_packed_gemv_wd.py`.
