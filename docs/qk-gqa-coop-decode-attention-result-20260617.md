# Target A — GQA-cooperative decode-attention: RESULT = SHIPPED (default `gqa_coop`) — 2026-06-17

> **SUPERSEDED AS DEFAULT (still valid).** `gqa_coop` was default until 2026-06-17, when **`gqa_coop_vec`**
> (gqa_coop + coalesced LOCAL-d loads, +6.5…+48.8% over gqa_coop, slope → −8%) replaced it. Current authority:
> `qk-gqa-coop-vector-load-result-20260617.md`. `gqa_coop` remains a valid `FLASH_VARIANT` override; the win
> below (vs hoisted) is still correct.

The llama-derived lever (`docs/llama-rocm-decode-attention-audit-20260617.md`): llama's `flash_attn_tile`
reuses one K/V tile across the GQA group; tinygrad's hoisted `flash_partial_v2` makes the query head a GLOBAL
axis, re-reading V[kv] **4×** across the group. Target A builds the tinygrad-native version of that reuse and
ships it. RX 7900 XTX, Qwen3-8B-Q4_K_M. Byte-identical greedy. No llama code copied; `custom_kernel`/UOps only.

## Verdict: SHIPPED — `FLASH_VARIANT` default flipped `hoisted` → `gqa_coop`

In-model W==D (the trusted method), default path, byte-identical greedy at every ctx:

| ctx | hoisted (was default) | **gqa_coop (new default)** | speedup | % of llama |
|---|---|---|---|---|
| 512  | 43.1 | **44.8** | +3.9% | 44% → **45%** |
| 1024 | 38.7 | **41.3** | +6.7% | 40% → **42%** |
| 2048 | 32.5 | **36.3** | +11.7% | 34% → **38%** |
| 4096 | 24.7 | **29.6** | +19.8% | 27% → **32%** |

Gates (all PASS): ≥5% @ctx1024 (+6.7–7.0%), ≥10% @ctx4096 (+19.4–19.8%), no ctx512 regression (+3.9%),
byte-identical greedy. **Flattens the long-ctx slope: −43% → −34% decay** (toward llama's −7%), exactly the
attention gap the audit identified. ctx<512 stays SDPA (unaffected).

## What the kernel is

`flash_partial_coop_kernel` (`extra/qk_flash_decode.py`): drop-in for `flash_partial_v2` (same `prob` input +
`pout[(h*S+s)*W+d]` layout, so `flash_{max,prob,gmax,den,combine}` are unchanged). Difference: the **kv-head is
the GLOBAL axis** (not the query head), so `V[kv,t,d]` is read **once** per (kv,split,d) thread and reused across
the G=4 query heads via **G register accumulators** (no LDS — the multi-reg-reduce pattern proven in
`extra/lds_attention_tile.py`). Cuts actual V traffic 4×. Bit-identical to v1/hoisted (self-test, max|diff|=0).

## Phase results + the measurement lesson (important)

| phase | measurement | result |
|---|---|---|
| 1 — isolated partial (DEBUG2, warm) | coop vs v2 | **2.1–3.0×** — but a **warm-Infinity-Cache artifact** |
| diag — in-pipeline partial (DEBUG2) | coop vs v2 @KV4096 | 700 vs 881µs = **1.26×** (the real partial gain) |
| 3 — isolated full attention (DEBUG2) | coop vs hoisted | 0.94 / 1.09 / 1.11 / 1.19× (below the 1.3× proxy gate; ctx512 "regression") |
| 5 — **in-model W==D (authoritative)** | coop vs hoisted | **+3.9 / +7.0 / +12.0 / +19.4%**, all gates PASS |

**Lesson (reinforced):** isolated DEBUG2 proxies misled in BOTH directions — the warm-cache isolated partial
*over*-stated (3× vs real 1.26×), and the isolated full-attention DEBUG2 *under*-stated (0.94× @ctx512 vs the
real +3.9% in-model). The in-model W==D byte-identical path is the only trustworthy gate (cf.
`amd-decode-measurement-confounds`). I followed it past the isolated proxy and it cleared every ship gate.

Why it works in-model when the isolated 3× was fake: in the real decode path V is not cache-resident, so cutting
V traffic 4× (cooperative reuse) is a real win — but bounded (the partial is ~1.26× faster in-pipeline, and it's
~one of several attention kernels), which is why in-model is +4–20%, not 3×. It grows with ctx because the
partial's share of decode grows (13%→47%).

## Not done (future, optional — each gated)

- **Vectorized coalesced fp16 K/V loads** (Phase 2, skipped — Phase 1 already cleared the gate). Could push
  effective BW further.
- **Stream-K adaptive KV split** (Phase 4) — would raise occupancy at short ctx and long-ctx; the audit's other
  llama ingredient. Build only if a further ≥1.3× is wanted.
- Per-band L; fold gqa_coop into the variant search grid (`qk_flash_variant_search`).

## Files / commits

`extra/qk_flash_decode.py` (`flash_partial_coop_kernel` + `gqa_coop` variant + self-test),
`tinygrad/llm/model.py` (default flip), `extra/qk_gqa_coop_decode_attention.py` (Phase 0/1 harness).
Commits: `[test] Phase 1`, `[codegen] add gqa_coop`, `[nn] default -> gqa_coop`, `[docs] this`.
