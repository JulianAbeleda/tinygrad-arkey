# MMVQ_COOP extension — Q6_K ffn_down: SHIPPED (default on) 2026-06-17

Generalizing the proven cooperative-K MMVQ primitive (`qk-mmvq-q6k-lm-head-arc-20260617.md`) to the next
highest-value sibling role. **ffn_down Q6_K shares the exact same pathology and the exact same kernel — it
shipped with zero kernel changes.** RX 7900 XTX, Qwen3-8B-Q4_K_M.

## Phase 0 — role inventory (measured)

Q6_K ffn_down: **18 linears** (half the layers; the other 18 ffn_downs are Q4_K parts=4), shape **4096×12288,
parts=1**, opts `LOCAL:0:64` — **identical one-row-per-thread pathology** as lm_head, same `_q6k_weight` byte
indexing. `bench/qk-mmvq-coop-ffn-down/role_inventory.json`.

## Phase 1–3 — isolated baseline + coop (real weights, fresh input)

| variant | µs | GB/s | % HBM peak | speedup | err |
|---|---|---|---|---|---|
| base (row-per-thread) | 327 | 125 | 14% | 1.0× | — |
| **coop row_tile=4** | **118** | **347** | **39%** | **2.77×** | 2.8e-6 |
| coop row_tile=8 | 123 | 334 | 37% | 2.67× | 2.8e-6 |

Correct (fp-reassoc tol), <peak (real, not less-work), 2.77× ≥ the 1.3× gate. row_tile=4. The **same**
`q6k_coop_partial_kernel` (shape-parameterized) — no new code.

## Phase 4 — in-model W==D gate — PASSED (stacks on lm_head)

| ctx | lm_head-coop | **+ffn_down** | speedup | greedy identical |
|---|---|---|---|---|
| 512 | 56.8 | **64.3** | **+13.2%** | ✓ |
| 1024 | 55.9 | **62.9** | **+12.5%** | ✓ |
| 4096 | 51.2 | **57.8** | **+12.9%** | ✓ |

Far past the +3% gate. **Default on** (`Q6K_FFN_DOWN_COOP=1`; `=0` falls back). Tests:
`test/external/test_q6k_coop.py`.

## Cumulative (both cooperative-K Q6_K roles vs the original default)

| ctx | original | lm_head | +ffn_down | total | % of llama |
|---|---|---|---|---|---|
| 512 | 47.3 | 56.4 | **64.3** | **+36%** | 48% → **65%** |
| 1024 | 46.5 | 55.3 | **62.9** | **+35%** | 48% → **64%** |
| 4096 | 43.6 | 51.3 | **57.8** | **+33%** | 48% → **63%** |

## Phase 5 — next generalization

ffn_down **shares the same pathology** (confirmed). The MMVQ_COOP family is compounding exactly as intended.
Remaining roles, all the same one-row-per-thread default:
- **Q4_K ffn_gate/up** (12288×4096, parts=1, ~40% peak, large decode share) — **next** (needs a Q4_K coop kernel;
  Q4_K unpack differs from Q6_K so it's a sibling kernel, not a parameter change). Expected +5–10%.
- **Q4_K attn_q/o** (4096×4096) — after. Expected +2–5%.
- Q6_K attn_k/v (1024×4096, parts=4) — small; split-K interaction, lower priority.

Recommend **Q4_K ffn_gate/up** next (biggest remaining share). Each role keeps its own isolated + in-model gate.

## Files / commits
`tinygrad/llm/model.py` (`[nn]`, routing + default), `test/external/test_q6k_coop.py` (`[test]`), this doc
(`[docs]`). Kernel unchanged (`q6k_coop_partial_kernel`, already shipped). Family `MMVQ_COOP`:
`Q6K_LM_HEAD_COOP` (on), `Q6K_FFN_DOWN_COOP` (on), `Q6K_COOP_RT` (row_tile, default 4).
