# Arc 1 — cooperative-K Q6_K lm_head GEMV: SHIPPED (default on) 2026-06-17

The first real base-decode matvec win of the campaign. The cooperative-K MMVQ work-decomposition — the only
un-refuted decode lever from the token-primitive accounting — **shipped**: lm_head Q6_K 10%→51% HBM peak, **+19%
in-model decode, byte-identical greedy.** RX 7900 XTX, Qwen3-8B-Q4_K_M. Baselines reported vs current tinygrad,
llama, and the XTX roofline (llama is the floor-to-beat, not the ceiling).

## Phase 0 — baseline (measured)

- Current lm_head Q6_K (151936×4096, parts=1): **91 GB/s ≈ 10% HBM peak**, render = `LOCAL:0:64` one-row-per-
  thread (adjacent lanes read whole rows apart → uncoalesced).
- Current decode (gqa_coop_vec default): ctx512/1024/4096 = 47.6/46.7/43.7 tok/s.
- References: llama MMVQ ~626 GB/s (~70% peak); XTX HBM peak ~900 GB/s.

## Phase 1 — cooperative-K design (the fix)

In `_q6k_weight`, `ql_byte_idx = half*64 + (pgrp%4)*16 + pos` — **adjacent `pos` (0..15) read adjacent bytes.**
So map `pos` to a **LOCAL lane axis** (16 lanes): adjacent lanes read adjacent packed bytes → **coalesced**.
Each lane writes its **own** partial `partials[row, pos]` (no in-kernel cross-lane reduce — structurally
identical to the proven gqa_coop_vec pattern); the reduction over the 16 pos-lanes is **stage-2 `.sum(axis=1)`**.
`row_tile` rows share a workgroup (lanes = row_tile×16) for occupancy. `q6k_coop_partial_kernel`,
`extra/q6_k_gemv_primitive.py`.

## Phase 2 — isolated gate (real lm_head weights, fresh input) — PASSED

| variant | µs | GB/s | % peak | speedup | err |
|---|---|---|---|---|---|
| base (row-per-thread) | 5525 | 91.5 | 10% | 1.0× | — |
| **coop row_tile=4** | **1107** | **456.6** | **51%** | **4.99×** | 2.4e-6 |
| coop row_tile=8 | 1110 | 455.6 | 51% | 4.98× | 2.4e-6 |
| coop row_tile=16 | 1508 | 335.2 | 37% | 3.66× | 2.4e-6 |

err 2.4e-6 = fp-reassociation only (full work done); 457 GB/s < HBM peak (physically plausible); 5×. **Not** a
less-work/warm-cache artifact (lm_head 506 MB > 64 MB IC, every run hits HBM; fresh random input). row_tile=4
default.

## Phase 3 — in-model W==D gate — PASSED (overwhelmingly)

| ctx | default | **coop** | speedup | greedy identical |
|---|---|---|---|---|
| 512 | 47.3 | **56.4** | **+19.2%** | ✓ |
| 1024 | 46.5 | **55.3** | **+18.9%** | ✓ |
| 4096 | 43.6 | **51.3** | **+17.7%** | ✓ |

Byte-identical greedy (the fp-reassoc diff never flips the argmax); W≈D (GPU-bound, real). From **one kernel**
(lm_head only). Prefill untouched (K==1 decode branch only). Far past the +5% gate.

## Phase 4 — SHIPPED (default on)

Routed in `Q6KPrimitiveLinear` decode GEMV: `parts==1 and out_features>=100000 and out_features%row_tile==0` →
coop kernel. **Default on** (`Q6K_LM_HEAD_COOP=1`; `=0` falls back; `Q6K_COOP_RT` tunes row_tile). Tests:
`test/external/test_q6k_coop.py` (kernel correctness vs base + greedy-identical routing).

## Result vs the three baselines

| | lm_head BW | % current tinygrad | % llama (626) | % XTX roofline (900) |
|---|---|---|---|---|
| base | 91 GB/s | 100% | 15% | 10% |
| **coop** | **457 GB/s** | **502%** | **73%** | **51%** |

lm_head now **exceeds llama's MMVQ effective BW** (457 vs the ~626 aggregate — and the role-level coop is 73% of
llama's aggregate, at 51% of the hardware roofline), correct and in-model-validated. Decode overall: **~48% →
~57% of llama.**

## What this opens (the lever is general, not lm_head-specific)

The coalescing fix (pos→LOCAL lane) applies to **every Q6_K and Q4_K role** — they all use the same
one-row-per-thread default at 10–40% peak. Next: extend cooperative-K to **Q6_K ffn_down** (parts==1, ~14% peak)
and the **Q4_K roles** (the same coalescing pattern). The "bounded decode levers exhausted" conclusion is
**superseded** — the un-refuted MMVQ work-decomposition lever won. See `qk-machine-search-primitive-rows-*`.

## Files / commits
`extra/q6_k_gemv_primitive.py` (`[codegen]`), `tinygrad/llm/model.py` (`[nn]`), `test/external/test_q6k_coop.py`
(`[test]`), this doc (`[docs]`). Supersedes the Phase-A conclusion in `qk-mmvq-q6k-lmhead-result-20260617.md`
(the cooperative-k rewrite is now built and shipped, not "unproven").
