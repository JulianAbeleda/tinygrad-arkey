# Arc 1 — mmvq_q6k full work-decomposition (lm_head): opt-search REFUTED; cooperative-k needs a kernel rewrite

The first build attempt at the only un-refuted decode lever (the full MMVQ work-decomposition). Started with
lm_head Q6_K (151936×4096, the worst-efficiency role at ~10% peak). **Verdict: the bounded schedule/opt-search
finds no correct win; the cooperative-k coalescing lever is real but requires a kernel rewrite, and the only
positive signal so far is a correctness-broken artifact.** No default changed. RX 7900 XTX, Qwen3-8B.

## Diagnosis (measured) — the current kernel is uncoalesced

`q6k_gemv_partial_kernel` rendered: opts `LOCAL:0:64`, **one row per thread** (64 rows/workgroup, 2374
workgroups), each thread fully reduces its row over k. Adjacent threads read **different rows (a full row
apart) → uncoalesced** weight loads. Achieved **92 GB/s ≈ 10% of HBM peak** (llama MMVQ ~70%). The uncoalesced
one-row-per-thread access — not the dot or raw bandwidth — is why it's slow.

## Schedule sweep (real lm_head weights, isolated, correctness-gated)

| schedule | µs | GB/s | speedup | err vs base | correct |
|---|---|---|---|---|---|
| `LOCAL:0:64` (base) | 5482 | 92 | 1.00× | 0 | ✓ |
| `LOCAL:0:32` | 5504 | 92 | 1.00× | 0 | ✓ |
| `LOCAL:0:128` | 5550 | 91 | 0.99× | 0 | ✓ |
| `GROUP:0:16` (cooperative-k) | 1451 | 348 | 3.78× | **0.95** | ✗ |
| `LOCAL:0:16+GROUP:0:16` | 481 | **1050** | 11.4× | **0.95** | ✗ |

- **Correct schedules (LOCAL variants) give NO improvement** (92 GB/s) — the row-per-thread decomposition is
  the binding inefficiency, and resizing LOCAL doesn't coalesce it.
- **GROUP (cooperative-k) is the lever direction** — but it **breaks correctness** (err 0.95) on the kernel's
  hand-rolled `.set(...end=pos)` reduction, and **1050 GB/s > HBM peak (900) for a 506 MB read is physically
  impossible → the GROUP kernel is doing *less work*, not coalescing faster.** A less-work artifact (the same
  warm-cache/less-work class that inflated the earlier dp4a 1.77× and READRAW 730). **NOT a validated win.**

## Verdict

**Bounded opt-search REFUTED** (no correct ≥1.3×): `OptOps.GROUP` cannot legally cooperative-reduce this
kernel's hand-rolled accumulator. The cooperative-k coalescing requires a **kernel rewrite** — a Q6_K GEMV whose
k-reduction is split across LOCAL threads with a *correct* cross-thread reduction (LDS/barrier or warp-reduce
WR1–3, or a GROUP-compatible standard reduce). That is the real Phase F MMVQ build, now **precisely diagnosed**
(uncoalesced → coalesced-k) but **unproven** (the only speedup signal is the broken artifact).

## Scoped next build (cooperative-k Q6_K GEMV) — high-value, high-risk, unproven

- **Design:** workgroup = a tile of rows; LOCAL threads split the k-reduction (adjacent threads read adjacent
  k of the same rows → coalesced fp16/packed loads); cross-thread reduce via LDS+barrier (or WR1–3 warp-reduce);
  Q6_K 6-bit unpack in-kernel; epilogue affine. Standard-reduce form so the optimizer can't drop work.
- **Isolated gate:** correct (bit/tol vs current) AND ≥1.3× on lm_head, ≥1.5× on the worst role — measured at a
  *real* bandwidth (reject any >peak/less-work result).
- **In-model gate:** W==D, byte-identical greedy, ≥5% decode @ctx512 and ctx1024.
- **Risk/uncertainty:** (a) the 6-bit unpack ALU may cap the real coalesced BW well below llama's 70% (the 92→?
  is unknown once correctness is enforced); (b) cross-thread reduction overhead; (c) prior LDS-cooperative builds
  (attention v3) lost to cache-served baselines. The gqa_coop_vec precedent (a coalescing rewrite that DID win
  +6.5..+48.8%) is the encouraging analogy, but that had no cross-thread reduction.

## Status
Arc-1 opt-search settled (refuted). The cooperative-k Q6_K **kernel rewrite** is the concrete remaining
decode lever — diagnosed, scoped, gated — but it is a substantial new-kernel build with an unproven real payoff
(only a less-work artifact suggests it). Recommend a go/no-go before committing the build effort. Sweep artifact:
`bench/qk-mmvq-q6k/lm_head_opt_sweep.json`.
