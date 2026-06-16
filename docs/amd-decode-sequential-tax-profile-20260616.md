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

## Gate closed: the token IS a single-queue sum (no existing overlap)

The named profile gives *relative* kernel cost; the open question was whether the batched
runtime already overlaps non-GEMV behind the weight stream (which would make the 28% not
reclaimable). **It does not — confirmed architecturally, not by a noisy timing:**

- `tinygrad/runtime/graph/hcq.py:51` allocates **one** `hw_compute_queue_t` per device
  (`comp_queues`); copy queues (SDMA) are separate and default to 1.
- No multi-stream / concurrent-compute / overlap primitive exists in the JIT or HCQ graph path
  (grep for `stream|overlap|concurrent|parallel.*queue` in `engine/jit.py` + `graph/hcq.py` is
  empty).

So every decode kernel — GEMV and non-GEMV — executes **back-to-back on a single compute
queue**: `token wall ≈ sum(all kernel exec) + small host`. This matches the capstone's prior
empirical decomposition (20.7 ms GPU-busy/token = 10.4 GEMV + 10.3 non-GEMV, 97% GPU-busy).
**Conclusion: the ~28% non-GEMV is additive and genuinely reclaimable** — there is no existing
overlap to compete with.

### What this means for the B2 design (next, in plan mode)

Overlap requires running non-GEMV concurrently with the GEMV weight stream, but within a layer
non-GEMV *depends* on the GEMV output — so the opportunity is **cross-layer**: layer N's
non-GEMV (norm/attention/sampling-tail) overlapping layer N+1's weight-GEMV stream. Two
candidate mechanisms: (a) a second compute queue with explicit cross-layer signalling, or
(b) fusing the small non-GEMV ops into the GEMV's memory-bound slack (the norm/elementwise
buckets are the fusable ones). Both are `[nn]`/`[codegen]` hot-path changes requiring
exact-output gating + AMD validation — scoped separately before any edit.

Anchors: `docs/amd-decode-beyond-llama-roadmap.md` (B2 = lead parity lever),
`docs/amd-decode-arc-synthesis.md` §6, `docs/amd-decode-capstone.md`,
`structure/Development/machine-search-decode-context-plan-2026-06-16.md` (Track 1).
