# Base-decode GEMV structural plan — 2026-06-17

Audit of tinygrad's decode GEMV vs llama's MMVQ (`llama-rocm-gemv-primitive-audit-20260617.md`), to find the
next *searched* GEMV target. **Verdict: base GEMV structurally mapped; NO bounded target earned** (the obvious
dp4a difference does not cash out e2e). No kernel built. No defaults changed. No Q4K_FUSE.

## 1. tinygrad GEMV state map [measured]

Defaults active: `FLASH_VARIANT=gqa_coop`, primitives decode_enabled, **no dense fallback** (all 199 q4k linears
use `Q4KPrimitiveLinear`; Q6_K roles use the Q6 primitive). Storage = shared (typed views over GGUF). Demotion
+ generated-policy default-off. Dot strategy (default) = **fp dequant + fp dot** (`q4_k_gemv_primitive.py`).

| role | quant | share @ctx512¹ | parts | dp4a-eligible (parts==1)? |
|---|---|---|---|---|
| ffn_down | Q4_K | 18% | >1 (split-K) | no |
| gate/up | Q4_K | 14% | 1 | yes |
| lm_head | Q6_K | 13% (tail) | >1 | no |
| attn_q/o | Q4_K | 9.5% | 1 | yes |
| attn_k/v | Q4_K | 1.7% | 1 | yes |

199 q4k linears: **163 parts==1 (vdot-eligible), 36 parts>1** (the big split-K roles ffn_down/lm_head). ¹shares
from `bench/qk-decode-block-map`. There is a gated dp4a path (`Q4K_VDOT=1`, parts==1 only).

## 2. llama vs tinygrad (the one structural difference)

| | llama MMVQ | tinygrad default | tinygrad `Q4K_VDOT` |
|---|---|---|---|
| dot | **int8 dp4a** (`__builtin_amdgcn_sdot4`, ~1.35 VALU/wt) | fp dequant+dot (~4.06 VALU/wt) | dp4a (`udot4`), parts==1 only |
| activation | q8_1 **once**, shared across same-input linears | fp16 | q8_1 **per-linear** |
| coverage | all roles | all (fp) | only the 163 parts==1 roles |

The concrete difference: **llama does the dot in hardware int8 dp4a; tinygrad's default does fp dequant.**
(NOT a fusion difference — gate/up and q/k/v are separate linears in both.)

## 3. Why no bounded target is earned [measured, the decisive part]

`Q4K_VDOT=1` (the dp4a path) **in-model W==D**: ctx128 49.3→49.8 (+1.0%), ctx512 37.0→37.3 (+0.8%) — **null.**
Diagnosis (eager breakdown @ctx128, Q4K_VDOT): q8_1 quant **2.2%** (NOT the blocker), vdot covers 24% (parts==1),
the big roles (ffn_down/lm_head) are 34.7% fp (parts>1, dp4a-ineligible).

**Amdahl reconciliation:** if dp4a delivered its standalone 1.77× on the 24% it covers, e2e would be **+11.7%**.
Measured **+1%** ⇒ the real in-pipeline dp4a speedup is **~1.04×, not 1.77×**. The standalone 1.77× (prior
`dp4a-d0`, and the READRAW 730 vs fp 365 GB/s) was a **warm-cache / standalone artifact** — the *same* lesson as
the gqa_coop partial (isolated 3× → in-pipeline 1.26×). **In-pipeline the decode GEMV is bandwidth/issue-bound,
not dot-ALU-bound**: the dequant ALU overlaps the (memory-bound) weight read, so removing it (dp4a) barely moves
e2e. This is why fp "4.06 VALU/wt" doesn't cost what the instruction count suggests.

So: the obvious structural difference (dp4a) is **refuted e2e**. Quant amortization is moot (quant is 2.2%).
Other candidates: Q6_K dp4a (same null mechanism); parts/tile search (exhausted — memory-bound, schedule knobs
don't move it, prior arcs); layout/repack (shared storage already optimal); reduction/partials (the parts>1
split-K is doing real work, removing it regressed before). **None clears ≥5%.**

## 4. The one untested branch (low expected value)

dp4a on the **parts>1 roles (ffn_down in=12288, lm_head)** has never been e2e-tested (Q4K_VDOT is parts==1 only).
ffn_down's larger reduction dim means more dot work, so dp4a *could* matter more there. Amdahl: 1.77× on those
31% → +15.6%; but at the realized ~1.04× (parts==1 evidence) → ~+1%. **Probability of clearing ≥5% is low** given
the parts==1 roles realized only ~1.04× e2e. Settling it needs a **split-K dp4a kernel** (not a tiny repro), so
it is not earned here.

## 5. Verdict + recommendation

**Base GEMV path structurally mapped; no bounded GEMV target earned.** The decode GEMV is bandwidth/issue-bound
e2e, where llama's dp4a advantage (real standalone) does not translate (+1% measured). The ~45% short-ctx gap to
llama is **distributed/structural** (bandwidth efficiency + ~780 progs/token granularity), not a single dp4a fix.

- **Do NOT** fund a dp4a GEMV build (refuted e2e). **Do NOT** reopen Q4K_FUSE. **Do NOT** chase program count.
- **Do NOT** trust standalone/DEBUG2 GEMV speedups (warm-cache-inflated; the 1.77× and READRAW were artifacts) —
  only in-model W==D.
- Highest-EV next work is NOT base-GEMV: it's the bounded **attention** follow-ons (vectorized loads / stream-K,
  `qk-gqa-coop-next-attention-levers-20260617.md`) or accepting the current state. If GEMV is pursued despite
  this, the only branch left is a split-K dp4a ffn_down/lm_head probe (low EV, must pass in-model ≥5% W==D).

## Kill gates (if anyone funds a GEMV build later)
- Isolated/DEBUG2 speedup is NOT a gate (artifact-prone). Gate = **in-model W==D ≥5% @ctx512 byte-identical**.
- If a split-K dp4a ffn_down probe is <5% in-model → stop; the GEMV is bandwidth-bound, confirmed.
