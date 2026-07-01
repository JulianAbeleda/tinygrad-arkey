# L3 Q6_K ffn_down Long-K Route — Result (14B + 32B, real GPU) — PROMOTED

Lever: speed up the biggest single decode weight wall — Q6_K `ffn_down`
(14B 17408->5120, 32B 25600->5120), ~253 GB/s / ~11.5 ms/tok — with a GENERIC
structural route generalization, no handwritten kernel, no model/shape hardcode.

## Root cause (found via BoltBeam profile + code)

BoltBeam's reader surfaced that Q4_K_M stores ~half of `ffn_down` as **Q6_K**. The
shipped Q6_K coop-partial route (which reaches ~51% HBM peak) is gated on the **8B**
ffn_down dims: `out==4096 and in==12288`. So 14B/32B Q6_K ffn_down did not match and
fell through to the slower generic `q6k_gemv_partial` path (~253 GB/s). This is the
same class of bug as the earlier attn_k route-miss: a hardcoded-shape gate.

## Fix (generic)

`tinygrad/llm/model.py`: generalize the coop gate structurally — route Q6_K ffn_down
through coop when `in_features >= 8192 and out_features < 100000` (long-in, moderate-out,
not lm_head), behind `DECODE_Q6K_FFN_DOWN_LONGK` (default-on, rollback = set 0). Not a
model-dim hardcode; a structural class. 8B ffn_down (already coop) and lm_head / attn_v
(out or in outside the class) are unaffected.

## Results — PROMOTED

Route-bound: `q6k_coop_partial_5120_17408` now fires 20x/step for 14B ffn_down
(previously the generic partial). Token-identical at 8B/14B/32B.

Authority W==D (`qk_decode_runtime_overhead.py`, synced, NMEAS=40):

| model | ctx | baseline tok/s | L3 tok/s | delta |
|-------|-----|----------------|----------|-------|
| 14B   | 128 | 44.50 | 52.20 | **+17.3%** |
| 14B   | 512 | 42.90 (rerun 42.90) | 50.20 | **+17.0%** |
| 32B   | 128 | 22.40 | 26.90 | **+20.1%** |
| 8B    | 512 | 107.40 | 107.40 | 0.0% (unaffected, still > llama) |

BoltBeam verdict: **promote, tier A** (+17.2%), required guardrails
(correctness/route-bound/speed/rollback) pass. Candidate `decode_q6k_ffn_down_longk`,
rollback `DECODE_Q6K_FFN_DOWN_LONGK=0`.

## Parity impact

14B decode ~67% -> ~76% of llama; 32B similar. This is the largest clean parity move
of the L1-L3 sweep — and it generalizes to 32B unchanged (the "use both to iterate
both" test passes: one structural rule, both shape families win).
