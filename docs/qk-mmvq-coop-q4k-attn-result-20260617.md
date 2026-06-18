# MMVQ_COOP → Q4_K attn_q/o: SHIPPED (default on) — 2026-06-17

The Q4_K member of the cooperative-K family, on the one Q4_K role with a real coalescing pathology. **Shipped:
+5-6% in-model decode, byte-identical, decode ~64% → ~68% of llama.** RX 7900 XTX, Qwen3-8B-Q4_K_M. The
`q4k_coop_partial_kernel` already existed (built in the ffn_gate/up arc); this arc was routing + the in-model
gate only.

## Phase 0 — inventory

Q4_K attn_q/o: **72 linears** (query + output proj, 2/layer × 36), shape **4096×4096, parts=1** — uniquely
identified by out==in==4096. (ffn_gate/up is 12288×4096; ffn_down Q4_K is 4096×12288 parts=4.)

## Isolated (from the ffn_gate/up arc, real weights)

| | GB/s | % HBM peak | speedup | err |
|---|---|---|---|---|
| base (row-per-thread) | 169 | **19%** | 1.0× | — |
| coop row_tile=16 | 258 | **29%** | **1.52×** | 2e-6 |

Poorly-coalesced default (unlike ffn_gate/up at 41%) → coop is worth it.

## Phase 2 — in-model W==D gate — PASSED

| ctx | default (Q6_K coop on) | **+attn_q/o** | speedup | greedy identical |
|---|---|---|---|---|
| 512 | 64.2 | **68.3** | **+6.4%** | ✓ |
| 1024 | 62.7 | **66.3** | **+5.7%** | ✓ |
| 4096 | 57.7 | **60.9** | **+5.5%** | ✓ |

Beat the Amdahl estimate (~+3%) because q+o is 72 linears. W≈D, byte-identical. Cleared the +3% "default on"
threshold → **default on** (`Q4K_ATTN_QO_COOP=1`; `=0` falls back; `Q4K_COOP_RT=16`). Test:
`test/external/test_q6k_coop.py`.

## Cumulative — all MMVQ_COOP roles vs the original pre-coop default

| ctx | original | +lm_head | +ffn_down | +attn_q/o | total | % of llama |
|---|---|---|---|---|---|---|
| 512 | 47.3 | 56.4 | 64.3 | **68.3** | **+44%** | 48% → **69%** |
| 1024 | 46.5 | 55.3 | 62.9 | **66.3** | **+43%** | 48% → **68%** |
| 4096 | 43.6 | 51.3 | 57.8 | **60.9** | **+40%** | 48% → **66%** |

## MMVQ_COOP family — final status

| role | quant | base % peak | coop | status |
|---|---|---|---|---|
| lm_head | Q6_K | 10% | 5.0× | SHIPPED (default on) |
| ffn_down | Q6_K | 14% | 2.77× | SHIPPED (default on) |
| **attn_q/o** | Q4_K | 19% | 1.52× | **SHIPPED (default on)** |
| ffn_gate/up | Q4_K | 41% | 1.18× | REFUTED (already coalesced) |
| attn_k/v | Q6_K | — | — | small share, parts=4 (not pursued) |

## Phase 3 — what remains

**The low-risk role-by-role MMVQ_COOP expansion is now done.** Every role with a poorly-coalesced default
(<~30% peak) is shipped; the remaining roles (Q4_K ffn_gate/up, Q4_K ffn_down) are already ~40% peak — coop
doesn't clear the gate. The MMVQ_COOP rule: **apply coop only where baseline coalescing is bad; do not
generalize by quant type.**

Decode is now **~66-69% of llama** (from ~48%). The remaining ~31-34% gap is **deeper full-MMVQ** (pushing the
already-decent roles from ~40-51% toward llama's ~70% — uncertain, bigger build) or a different phase (prefill
WMMA, 14B). No more low-risk coop roles.

## Files / commits
`tinygrad/llm/model.py` (`[nn]`, routing + default), `test/external/test_q6k_coop.py` (`[test]`), this doc
(`[docs]`). Kernel `q4k_coop_partial_kernel` already committed. Flag `Q4K_ATTN_QO_COOP` (on), `Q4K_COOP_RT`
(row_tile, default 16).
