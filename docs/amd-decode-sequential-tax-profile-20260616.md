# Decode sequential-tax profile — 2026-06-16

Phase 2 / Track 1 baseline (machine-search-decode-context-plan). **Measurement-first:**
before building any overlap/fusion lever, quantify the non-GEMV sequential tax on the
*current* (post-prune, default-on) build. Reuses the existing profiler
(`extra/q4_k_profile_report.py`) — no new tooling.

## Method (reproducible)

```
DEBUG=2 JIT_BATCH_SIZE=1 DEV=AMD .venv/bin/python -m tinygrad.llm.cli \
  -m models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 8 > tg-8b-q4q6-primitive-named.log 2>&1
sed 's/\x1b\[[0-9;]*m//g' tg-8b-q4q6-primitive-named.log > plain.log   # strip ANSI (profiler regex needs plain)
.venv/bin/python -m extra.q4_k_profile_report plain.log
```

`JIT_BATCH_SIZE=1` ("named" mode) disables graph batching so every kernel is a
separate AMD dispatch and DEBUG=2 prints per-kernel `tm`. Per the profiler's own note,
**named wall-time is attribution-only** (residual 83% = disabled-batching launch
overhead — *not* a real cost); read the **% AMD-kernel** column, not wall time. Hardware:
RX 7900 XTX gfx1100. Steady-state over 7 tokens (first dropped).

## Result — the token's in-graph kernel time

| bucket | % AMD kernel | note |
|---|---:|---|
| q4k_primitive_gemv | 34.7 | weight read (Q4_K tensors) |
| q6k_primitive_gemv | 37.6 | weight read (Q6_K tensors) |
| **GEMV subtotal** | **72.3** | **at batch-1 occupancy ceiling (B1 = decisive negative)** |
| norm_sampling_misc | 11.3 | RMSNorm + lm_head reductions (`r_32_4_1187*`) + sampling |
| other_amd | 7.4 | rope / elementwise / sdpa-ish (`r_2_8_128*`, `E_*`) |
| attention_misc | 4.5 | KV reduces (`r_*_(start_pos+1)_*`) |
| q4k_primitive_reduction | 2.6 | split-K partial-reduction tails |
| fallback_quant_fused | 1.7 | residual generic dense quant |
| copy | 0.2 | |
| **non-GEMV subtotal** | **27.7** | **the sequential tax** |

## Interpretation

- **Confirms the reframe** (roadmap + capstone "~71% GEMV / ~29% non-GEMV") on the current
  post-refactor build: GEMV ≈ 72%, non-GEMV ≈ 28%. The parity gap to llama is the non-GEMV
  **tax running sequentially** (`token = GEMV + non-GEMV`), not a slow GEMV (GEMV is already
  ~52% of peak ≈ llama's per-kernel rate; B1 proved a faster per-kernel GEMV is not the path).
- **27.7% is the upper bound on what B2 (overlap) can reclaim** — pipelining non-GEMV behind
  the next layer's weight stream moves `token → max(GEMV, non-GEMV)` instead of sum.
- **Target ranking for Track 1** (largest fusable/overlappable first):
  1. `norm_sampling_misc` (11.3%) — RMSNorm + the lm_head `r_*_1187` reductions + sampling.
     Norms are fusable into the adjacent GEMV epilogue/prologue; lm_head + sampling run once
     per token and are pure overlap candidates.
  2. `other_amd` (7.4%) — rope + elementwise; small, many, classic fusion candidates.
  3. `attention_misc` (4.5%) — already addressed at long context by shipped flash
     (`FLASH_DECODE`); at short context it's small. Lower priority than 1–2.

## Caveat / next measurement

This is the **named** (attribution) profile — it tells us the *relative* kernel cost. It does
NOT directly measure how much non-GEMV already overlaps the weight stream in the real
**batched** runtime. The next measurement step (before building B2) is to confirm the batched
token is `GEMV + non-GEMV ≈ sum` (little existing overlap) — e.g. compare the batched
wall-time/token against the GEMV-only lower bound — so the ~28% is genuinely on the table and
not already hidden. Only then build the overlap lever (as a separate `[nn]`/`[codegen]`,
AMD-validated commit, gated, exact-output).

Anchors: `docs/amd-decode-beyond-llama-roadmap.md` (B2 = lead parity lever),
`docs/amd-decode-arc-synthesis.md` §6, `docs/amd-decode-capstone.md`,
`structure/Development/machine-search-decode-context-plan-2026-06-16.md` (Track 1).
