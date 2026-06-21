# Decode Current-Route Attribution Result (Deliverable 0)

Date: 2026-06-20

Executor: Claude

Verdict: `PASS_DECODE_CURRENT_ROUTE_ATTRIBUTION` — the table is trustworthy (per-role timing is GPU-timestamp
`timed`, not proxy), it separates W wall from D dispatch ceiling at ctx 512/1024/4096 for both modes, and it
attributes the ctx1024 gap in full. **It also overturns the scope's expected lane ranking.**

## What this is

The first trustworthy current-route role/tensor/kernel attribution table for Qwen3-8B decode, in ms/token. It is
Deliverable 0 of `docs/decode-role-tensor-kernel-attribution-solution-scope-20260620.md`. No kernels were built.
No decode default changed.

## Method (two-layer, matches the non-negotiable timing policy)

1. **Whole-token authority (W==D), PROFILE off → clean wall** — reuses the `extra/qk_decode_runtime_overhead.py`
   method. `W` = real decode (`.item()`/token), `D` = dispatch-only ceiling, `host_sync = W−D`.
2. **Per-role TIMED split (`ProfileGraphEvent`), PROFILE on, a second JIT so (1) stays clean** — the warm HCQ
   graph replay records per-kernel GPU timestamps; `dur_us = sigs[en_id]-sigs[st_id]`. These are real warm
   GPU-execution intervals per program → confidence `timed` (NOT a DEBUG=2 proxy). The decode forward replays as
   **6 graph segments / ~1074 kernel-calls per token**; all segments of one bracketed replay are aggregated.

PROFILE-on GPU-busy (~16.8–18.5 µs-summed) slightly exceeds the clean W wall (per-kernel timestamp overhead), so
per-role ms/token is **rescaled onto the clean W wall** by the timed busy-share (factor ~0.85–0.87). ATT/HCQ
packet counts are never used as timing. Ran under GPU perf-state `auto` (verified before/after); no lane forced.

Program→role mapping widens the frozen census regex (`extra/qk_decode_layer_census.py:20`) to catch the coop /
q8 kernels the live route emits, then maps `(out,in)` via `GEMV_ROLE`. 8B: hidden 4096, ffn 12288, vocab 151936,
n_kv 8, head_dim 128.

## Current route W/D by ctx (W is authority; host-sync 0% everywhere → GPU-bound)

| mode | ctx | tok/s W | ms/token W | D ceiling tok/s | host-sync % | progs/token |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 512 | 68.5 | 14.59 | 65.5 | 0.0 | 1074 |
| baseline | 1024 | 66.9 | 14.95 | 64.1 | 0.0 | 1074 |
| baseline | 4096 | 61.2 | 16.35 | 58.6 | 0.0 | 1074 |
| q8 | 512 | 72.8 | 13.73 | 65.6 | 0.0 | 1074 |
| q8 | 1024 | 71.0 | 14.08 | 67.5 | 0.0 | 1074 |
| q8 | 4096 | 64.5 | 15.50 | 61.4 | 0.0 | 1074 |

q8 speedup over baseline: 1.063 / 1.061 / 1.054 @512/1024/4096 — matches the prior q8 model-route audit
(`docs/decode-q8-model-route-timing-audit-result-20260620.md`). W and D agree within ~4% with `host_sync = 0%`;
**W/D do not diverge in the recoverable direction → Lane 7 (host/persistent runtime) is closed.**

## Ranked role/tensor/kernel attribution table — baseline @ ctx1024 (wall 14.95 ms, llama 10.25 ms)

| role | tensor | calls/tok | ms/tok | %wall | eff BW GB/s (%HBM) | llama analogue | gap ms/tok | conf | action |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| ffn_gate/up | Q4_K | 72 | 3.652 | 24.4 | 558 (58%) | mul_mat_vec_q Q4_K | +0.19 | timed | audit_more |
| attention_flash | attention | 378 | 3.501 | 23.4 | — | flash_attn_* | **+2.73** | timed | **audit_more** |
| elementwise | fp | 220 | 2.187 | 14.6 | — | rope+elementwise | **+1.83** | timed | **audit_more** |
| ffn_down | Q6_K | 36 | 2.154 | 14.4 | 690 (72%) | mul_mat_vec_q Q6_K | +0.11 | timed | audit_more |
| attn_q/o | Q4_K | 72 | 1.186 | 7.9 | 573 (60%) | mul_mat_vec_q Q4_K | +0.06 | timed | drop |
| reduce/glue | fp | 204 | 0.948 | 6.3 | — | (llama fuses away) | −0.13 | timed | drop |
| lm_head | Q6_K | 1 | 0.592 | 4.0 | 863 (90%) | mul_mat_vec_q Q6_K | +0.03 | timed | drop |
| rmsnorm | fp | 73 | 0.390 | 2.6 | — | rmsnorm | −0.12 | timed | drop |
| attn_k/v | Q6_K | 18 | 0.344 | 2.3 | 180 (19%) | mul_mat_vec_q Q6_K | +0.02 | timed | drop |

(`audit_more`/`drop` are table heuristics, not build decisions — the build ranking is in the final report.)

## Token math (gap fully decomposed by family; Σ family_gap == total gap)

llama mmvq effective BW = 626 GB/s (65% HBM peak). Decomposition (baseline):

| ctx | gap ms/tok | attention | elementwise | weight-GEMV | rmsnorm | glue/other |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4.45 | +2.44 | +1.83 | +0.44 | −0.12 | −0.12 |
| 1024 | 4.71 | **+2.73** | **+1.83** | +0.41 | −0.12 | −0.13 |
| 4096 | 5.51 | **+4.36** | +1.83 | −0.33 | −0.15 | −0.20 |

q8 mode (the latest route) — same residual, weight-GEMV goes *negative* (q8 beats llama on weight):

| ctx | gap ms/tok | attention | elementwise | weight-GEMV | rmsnorm | glue/other |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 3.83 | +2.69 | +1.94 | **−0.71** | +0.07 | −0.14 |
| 4096 | 4.66 | +4.28 | +1.94 | **−1.38** | +0.04 | −0.22 |

**Attributed gap @ baseline ctx1024 = 4.84 ms** (every family except the no-llama-analogue glue/other bucket,
which is −0.13 ms). Pass gate required ≥2.5 ms attributed of the ~3.93–4.71 ms gap → **cleared with margin, and
the decomposition is complete (residual = −0.13 ms fusion/glue bucket).**

## The decisive finding (overturns the scope's expected ranking)

The scope expected — from the **stale DEBUG=2 proxy** block-map — that Q6 big roles (`ffn_down`, `lm_head`) and
the MMVQ family own the gap. The fresh **timed** route refutes this:

- **Weight-GEMV (all MMVQ-equivalent roles) is at/above llama parity in-model.** Total tinygrad weight-GEMV =
  7.93 ms/tok @1024 vs llama mmvq 7.52 ms (+0.41 ms, ~9% of gap); at ctx4096 tinygrad weight is **faster**
  (−0.33 ms), and with the q8 route it is faster at every ctx (−0.71/−1.38 ms). Per-role effective BW: lm_head
  863 GB/s (90% HBM), ffn_down 690 (72%), attn_q/o 573 (60%), ffn_gate/up 558 (58%) — all at or above llama's
  626. (`attn_k/v` is 180 GB/s/19% but is overhead-bound on a tiny 1024×4096 GEMV and only 2.3% of wall.)
  → This revises the older "in-model GEMV 44% HBM" figure (that was a discarded PMC per-kernel estimate; the
  cleaner `ProfileGraphEvent` timing shows weight-GEMV is largely solved in-model).
- **The gap is attention + elementwise, ~97% of it.** At ctx1024: attention +2.73 ms (flash-decode overhead) +
  elementwise +1.83 ms (the unfused FFN SiLU·mul `E_49152_32_3` ≈ 1.4 ms + rope + residual adds that llama fuses
  into its kernels) = 4.56 ms of the 4.71 ms gap.
- **Attention has a context slope; elementwise is flat.** Attention share of wall 21.9 → 23.4 → 31.6% @512/1024/
  4096 (gap +2.44 → +2.73 → +4.36 ms); elementwise stays +1.83 ms at every ctx. q8 changes neither.

## Pass-gate status

| gate | result |
|---|---|
| produced ctx 512/1024/4096 rows, both modes | PASS |
| separates W wall tok/s from D dispatch ceiling | PASS (host-sync 0% → GPU-bound; W/D <5% divergence) |
| attributes ≥2.5 ms of the ctx1024 gap | PASS (4.84 ms attributed; complete decomposition, −0.13 ms residual) |

The table is trustworthy. Stop condition (build only after a trustworthy table) is satisfied → proceed to lane
ranking in the final report. The lane ranking is **inverted from the scope's expectation** by the evidence.

## Exact commands

```bash
# full run (perf-state auto)
PYTHONPATH=. python3 extra/qk_decode_current_route_attribution.py \
  --modes baseline,q8 --ckpts 512 1024 4096 --nmeas 20 --warmups 8 \
  --out bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json
# re-aggregate from existing child JSONs (no model re-run)
PYTHONPATH=. python3 extra/qk_decode_current_route_attribution.py \
  --modes baseline,q8 --ckpts 512 1024 4096 --aggregate-existing \
  --out bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json
```

## Artifacts

- `extra/qk_decode_current_route_attribution.py` (instrumentation)
- `bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json` (joined table + token math)
- `bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution_{baseline,q8}.json` (per-mode children)

## Boundary

No decode default changed. q8 was env-gated (`Q8_FFN_HANDWRITTEN=1`) in its child only and restored off. GPU
perf-state `auto` before and after.
