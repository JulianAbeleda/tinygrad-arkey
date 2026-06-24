# MMVQ_COOP — remaining-role census (2026-06-17)

After shipping cooperative-K on three roles (decode ~48% → ~68% of llama), this census audits every remaining
matvec role to decide whether more low-risk coop wins exist. **No new kernels built.** Verdict: **the low-risk
role-by-role MMVQ_COOP expansion is DONE.** RX 7900 XTX, Qwen3-8B-Q4_K_M, HBM peak ~900 GB/s, ~4.68 GB
weights/token.

## Shipped state

Decode (in-model W==D, byte-identical greedy): ctx512/1024/4096 = **68.3 / 66.3 / 60.9 tok/s = 69% / 68% / 66%
of llama** (98.6 / 97.6 / 92.2). Up from the pre-coop 47.3 / 46.5 / 43.6 (~48%). Default-on flags:
`Q6K_LM_HEAD_COOP`, `Q6K_FFN_DOWN_COOP`, `Q4K_ATTN_QO_COOP` (each with `=0` fallback; `Q6K_COOP_RT`=4,
`Q4K_COOP_RT`=16).

## Census table (every matvec role)

| role | quant | shape | parts | weight-traffic share | base % peak | coop % peak | isolated speedup | expected e2e | verdict |
|---|---|---|---|---|---|---|---|---|---|
| lm_head | Q6_K | 151936×4096 | 1 | 10.8% | 10% | 51% | 5.0× | done (+19%) | **A shipped** |
| ffn_down | Q6_K | 4096×12288 | 1 | 15.7% | 14% | 39% | 2.77× | done (+13%) | **A shipped** |
| attn_q/o | Q4_K | 4096×4096 | 1 | 14.5% | 19% | 29% | 1.52× | done (+6%) | **A shipped** |
| **ffn_gate/up** | Q4_K | 12288×4096 | 1 | **44.0%** | **41%** | 47% | 1.18× | <gate via coop | **B refuted (coop)** / **D** deeper-MMVQ |
| ffn_down | Q4_K | 4096×12288 | 4 | 10.9% | 35.5% | 40% | 1.13× | <gate | **B refuted (already coalesced)** |
| attn_k/v | Q6_K | 1024×4096 | 4 | 1.3% | **8.9%** | 14% | 1.57× | ~+0.5% | **B too small by Amdahl** |

(Shares are weight-traffic fraction of the ~4.68 GB/token; they sum to ~97%. The 54 missing k/v projections are
not Q4K/Q6K-primitive-backed.) Coop measured with the existing `q6k_coop_partial_kernel` / `q4k_coop_partial_kernel`
on real weights, fresh input, fp-reassoc-exact (err 6e-7…1.4e-6).

## Phase 4 — classification

- **A. Shipped:** lm_head, ffn_down (Q6_K), attn_q/o (Q4_K) — 41% of weight traffic, now coalesced.
- **B. Refuted / closed:**
  - Q4_K ffn_gate/up (coop 1.18×) and Q4_K ffn_down parts=4 (1.13×) — **already coalesced (35-41% peak)**, coop
    below the 1.3× gate. Do not route.
  - Q6_K attn_k/v (1.57× isolated, base 8.9% peak) — poorly coalesced but **only 1.3% of weight traffic →
    Amdahl ~+0.5% e2e**, below the 2-3% candidate floor. Too small.
- **C. Candidate (base<30% peak AND ≥1.3× AND e2e≥2-3% AND no prior refutation):** **none.** Every role is
  either shipped, already-coalesced, or too small.
- **D. Deep / new primitive:** Q4_K ffn_gate/up is the **largest single role (44% of weight traffic) but already
  at 41% peak**; the simple lane4-coop only reaches 47%. Closing to llama's ~70% needs a **deeper full-MMVQ
  kernel** (better tiling / dp4a-in-coop / q8_1 activations) — a different kernel family, high-risk, uncertain.

## Phase 5 — ranked next actions

| rank | candidate | expected e2e | confidence | risk | reason | recommendation |
|---|---|---|---|---|---|---|
| 1 | **Q4_K ffn_gate/up deeper full-MMVQ** (41%→~70% peak) | +5-12% if it reaches llama-class | low-med | **high** | largest role (44% traffic); coop insufficient (47%); llama proves 70% exists | only with a real new-kernel-family search + hard in-model gate |
| 2 | **prefill WMMA** (different phase) | prefill +10-30% | medium | med-high | revived WMMA's regime; prefill 81%→more; doesn't touch decode | strong next if switching phase |
| 3 | **14B/32B** (different target) | n/a | medium | med | more GPU-bound; coop primitives amortize better | strategic |
| 4 | Q6_K attn_k/v coop | ~+0.5% | high | low | passes isolated but tiny share | skip (not worth the route) unless trivially free |

## Closed / refuted (do not reopen as coop routes)

- Q4_K ffn_gate/up coop, Q4_K ffn_down parts=4 coop — already coalesced (35-41% peak).
- Q6_K attn_k/v coop — too small (1.3% share, ~+0.5%).
- (Prior, separate) dp4a both quants, Q4K_FUSE, stream-K, decode_attention_v3, schedule-knob-only, sub4, naive
  spec, ring2.

## The sharpened MMVQ_COOP rule

**Apply cooperative-K coalescing only where (a) the role's default bandwidth is poor (<~30% of HBM peak) AND
(b) the role is a non-trivial share of weight traffic.** Do NOT generalize by quant type — Q4_K is
role-dependent (attn_q/o poorly coalesced → win; ffn_gate/up already coalesced → no win). The remaining decode
gap to llama is no longer addressable by local coop routes; it is deeper full-MMVQ (push the already-decent
35-47%-peak roles toward 70%) or a different phase.

## Files
`bench/qk-mmvq-coop-remaining-census/result.json` (+ `measure.json`). Census only — no kernels built, no routes
added.
