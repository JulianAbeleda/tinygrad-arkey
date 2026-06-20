# Prefill Graph-Route Attribution Result (Step 1 localizing audit) — BRANCH B

Date: 2026-06-20
Repo: `/home/ubuntu/tinygrad-arkey`, branch `qk-prefill-flag-leak-resolution`
GPU: gfx1100 (RX 7900 GRE). Model: Qwen3-8B-Q4_K_M.
Tools: `extra/qk_prefill_inmodel_attribution.py` (PREFILL_GRAPH_GEMM=1, ctx768) +
`extra/qk_prefill_pmu_atlas.py` (PMC=1, per-kernel VALU%/L2-hit classification).

## Why this audit

Per `docs/prefill-next-gap-scope-20260620.md` Step 1: the prior attribution (47% gate/up,
28% attn, 23% other-mm, 1.6% norm) was on the **baseline** forward. The graph route
(`PREFILL_GRAPH_GEMM`, now promoted default-on) sped up the FFN matmuls, so the time mix
has shifted. Re-localize on the 256 ms graph forward to pick the bigger lever.

## Result — the mix shifted to attention

The attribution tool's coarse bucketer mislabels the new `prefill_graph_gemm_*` kernels as
"other" (they don't start with `r_`/`wmma`). Corrected from the per-kernel list:

| group | % of GPU time | what |
|---|---:|---|
| **attention** (`r_*start_pos*` reduces) | **~47%** | dominated by ONE kernel `r_2_512_*start_pos*_8_4_4_16` = **30%** |
| FFN + projection GEMMs (`prefill_graph_gemm_*`) | ~48% | gate/up 21.2% + down 12.6% + 4096² (q/o) 8.6% + kv 5.6% |
| elementwise/norm/cast | ~3% | |
| other small matmul | ~2% | |

vs the baseline split (gate/up 47% / attn 28% / other-mm 23%): the graph route did its job —
FFN matmuls dropped, and **attention rose to co-dominant (~47%), concentrated in a single
naive reduce kernel at 30% of the whole forward.** That kernel is the biggest single lever.

## PMC bottleneck class (per dominant kernel)

`qk_prefill_pmu_atlas.py` (eager forward; attention kernels are identical regardless of the
graph route, so the class transfers). Caveat carried from memory: `v_wmma` is a multi-cycle
instruction with **no VALU counter**, so low VALU% on a GEMM means WMMA, not idle — use L2-hit
+ kernel identity, not VALU%, to classify the matmuls.

| kernel | gpu% | VALU% | L2-hit% | class |
|---|---:|---:|---:|---|
| `prefill_graph_gemm_512_12288_4096` (gate/up) | 26.3 | 1.5 | 57.1 | **WMMA-compute-bound** (cache-served) |
| `prefill_graph_gemm_512_4096_12288` (down) | 11.9 | 1.5 | 70.9 | WMMA-compute-bound |
| `prefill_graph_gemm_512_4096_4096` (q/o) | 9.6 | 1.4 | 69.0 | WMMA-compute-bound |
| attention reduce `r_16_32_2_8_16_4_4_*` | 14.2 | **13.5** | 67.6 | **ALU-compute-bound, NO tensor cores** |
| attention reduce `r_32_16_8_8_16_4_4_*` | 3.9 | **21.0** | 96.8 | ALU-compute-bound, NO tensor cores |
| small softmax/score reduces `r_128_32_4_*`, `r_512_16_256` | 1–2 ea | <1 | 0–3 | bandwidth-bound (tiny) |

Two facts that decide the branch:

1. **The matmuls are already WMMA/cache-compute-bound, NOT bandwidth-bound** (L2-hit 57–80%,
   weights reused across the 512-token tile). So **Branch A (int8-MMQ) premise is FALSE** — int8
   reduces HBM traffic, but these GEMMs aren't HBM-bound; the residual matmul gap is WMMA
   *scheduling* (the POWN-walled SW-pipelined-K-loop codegen problem), which int8 doesn't fix.
2. **Attention is a naive `r_` reduce with real ALU work and no tensor cores.** This is exactly
   the workload flash/TC attention replaces (TC `Q@Kᵀ` + online softmax + TC `P@V`).

## Cross-check vs llama (the gap lives in attention)

llama prefill reference (rocprofv3, memory): 74% int8-MMQ + 9.2% Tensile fp16 + **4.4% flash
attention**. tinygrad spends **~47%** on attention. In absolute terms attention is ~120–131 ms
of the ~256–280 ms forward — tinygrad's attention *alone* is ~75% of llama's entire pp512
(170 ms). llama proves causal 512-token attention can be ~5% with flash. **This is where the
34%-to-llama gap lives**, and it is non-TC naive attention, not the matmuls.

## DECISION: Branch B — flash/TC attention on concrete KV

- **Lever:** replace the naive `r_2_512_*start_pos*` softmax-attention reduce with tensor-core
  flash attention on concrete KV.
- **Why B over A:** B targets the single largest kernel (30%) AND the larger addressable share
  (~47% vs matmuls already at the WMMA-scheduling wall); B is the cheaper build (scaffolding
  exists); A's bandwidth premise is measured-false.
- **Have (scaffolding):** `PREFILL_CONCRETE_KV` (concrete start_pos → KV concrete → attention TC
  can fire; prior ~1.24× e2e), `PREFILL_TC_ATTN` (explicit TC `Q@Kᵀ` + softmax + `P@V`). Prior
  TC probe was 0.79× on **symbolic** KV — the concrete-KV regime is the one where it can win.
- **Gate (iron law):** rel RMSE < 1e-2 + sampled/chunked NLL dNLL ≤ 0.01 + greedy-exact + SYNCED
  arbiter (K forwards / one sync) vs llama + fallback + OOM. Default-off unless owner-approved;
  gfx1100-restricted. No BEAM.

## Next step

Wire/measure `PREFILL_CONCRETE_KV` + `PREFILL_TC_ATTN` together on the graph-route forward and
take a synced before/after on the dominant attention kernel + whole-forward tok/s. Expected
ceiling if attention approaches llama's flash share: forward → roughly the ~135 ms matmul half,
i.e. toward parity with / below llama pp512 — but the WMMA-scheduling wall on the matmul half
remains the eventual floor.
