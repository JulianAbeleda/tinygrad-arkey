# Prefill Next-Gap Scope: from 66% → toward llama (localizing audit first, then int8/attention)

Date: 2026-06-20

## Where we are (verified, synced)

Dependency-free graph route PROMOTED default-on (gfx1100-guarded). Synced ladder on Qwen3-8B / gfx1100:

| path | synced ms/512 | tok/s | % of llama |
|---|---:|---:|---:|
| baseline WMMA (`PREFILL_V2`) | 414 | 1236 | 41% |
| Tensile `.co` (dependency, research) | 276 | 1855 | 61% |
| **graph route (promoted, dep-free)** | **256** | **1998** | **66%** |
| llama pp512 | 170 | 3020 | 100% |

The "86%/93% of llama" history was the **nosync mirage** (settled: `docs/prefill-tensile-86-settled-nosync`,
`docs/prefill-TRUE-throughput-and-matmul-penalty`). 66% is the real ceiling reached. Remaining gap to llama
(~34%) is **real GPU-compute reduction**, not a measurement fix and not Tensile.

## Step 1 (DO FIRST — cheap, uses tools we have): localizing audit on the GRAPH-ROUTE forward

The existing attribution (47% gate/up, 28% attention, 23% other-matmul, 1.6% norm) was on the **baseline**
forward. The graph route sped up the FFN matmuls, so the remaining-time mix has SHIFTED — attention's share
rises, gate/up's drops. Re-localize on the 256 ms graph forward to pick the bigger lever.

- Tool: `extra/qk_prefill_inmodel_attribution.py` run with `PREFILL_GRAPH_GEMM=1` (per-kernel GPU-time buckets
  from `ProfileGraphEvent`; relative % is trustworthy on the real warm forward).
- Tool: `extra/qk_prefill_pmu_atlas.py` / `qk_pmc_capture.py` — classify the now-dominant kernels
  (bandwidth-bound vs compute-bound, L2 hit, VALU).
- Caveat already known: profiling can hit the tinygrad AMD allocator OOM at ctx2048 — run attribution at the
  smaller context (ctx768) where it succeeded; relative % holds.
- Acceptance: a synced per-kernel breakdown of the 256 ms forward + a bandwidth/compute label per dominant
  kernel. Output: `docs/prefill-graph-route-attribution-result-2026MMDD.md`.

This decides the branch below.

## Step 2 — the lever (branch on Step 1)

### Branch A: matmul-bandwidth still dominant → int8-quantized GEMM (llama's trick)
llama's prefill bulk uses int8 MMQ — weights stay ~4.5-bit, far less HBM traffic than our dequant→fp16 WMMA.
This is the biggest structural lever but the **largest build**:
- **Need (do NOT have):** (1) an int8/MMQ GEMM kernel (we have fp16 `build_gemm_lds2`, not int8); (2) on-the-fly
  activation→int8 (Q8_1-style) quantization for prefill.
- **Have:** the build/wire/gate infra — `assemble_linear`→ELF, `Tensor.custom_kernel`, the `_pf16` graph-route
  hook (`extra/qk_prefill_graph_gemm_route.py`), and all gates.
- Risk: lossy (int8) → must pass the quality gate; new kernel correctness; a real multi-day kernel effort.

### Branch B: attention now dominant → flash/TC attention on concrete KV (cheaper, scaffolding exists)
- **Have (scaffolding):** `PREFILL_CONCRETE_KV` (concrete start_pos → KV concrete → attention TC can fire; prior
  ~1.24× e2e), `PREFILL_TC_ATTN` (explicit TC Q@Kᵀ + softmax + P@V). Prior probe was 0.79× on SYMBOLIC KV; the
  concrete-KV regime is the one where it can win.
- Less new-kernel than int8; mostly wiring + the concrete-KV interaction, gated.

## Tooling readiness summary

| capability | status |
|---|---|
| synced throughput (arbiter) | ✅ proven (`*_default_perf`, `*_settle`) |
| per-kernel attribution | ✅ (relative, real forward) |
| PMC bottleneck class (bw vs compute) | ✅ |
| correctness gate (rel RMSE) | ✅ |
| quality gate (sampled/chunked NLL + greedy) | ✅ (VRAM-safe; full 512×vocab NLL OOMs — do not use) |
| fallback + OOM audits | ✅ |
| build/wire/gate infra for a new route | ✅ |
| **int8 GEMM kernel + activation quant** | ❌ NEW BUILD (Branch A) |

**Verdict: audit-complete (we can find and quantify the next gap with current tools); build-partial (the int8
kernel is the one missing piece, and only if Step 1 says bandwidth).**

## Iron law (do not regress)
- **SYNCED measurement only.** Arbiter = K forwards / one `dev.synchronize()` / total/K. Never trust a nosync
  `realize()` loop (host dispatch, inflated ~2.3× and up to absurd). Compare synced-vs-synced to llama.
- Every route change gated: correctness (rel RMSE < 1e-2) + quality (sampled NLL dNLL ≤ 0.01 + greedy-exact) +
  synced perf + fallback + OOM. Default-off unless owner-approved; gfx1100-restricted.
- No BEAM (hangs gfx1100).
