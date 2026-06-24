# Next decode-attention levers after gqa_coop (2026-06-17) — design/ranking only

`gqa_coop` shipped (cooperative GQA V-reuse, default). Decode now 44.8/41.3/36.3/29.6 tok/s @ctx
512/1024/2048/4096 = **45/42/38/32% of llama**, slope −34%. This ranks the remaining bounded attention levers.
No code built here.

**Amdahl frame:** `flash_partial_coop` is ~the dominant attention kernel; attention is ~13%@ctx512 → ~47%@ctx4096
of decode GPU. So further attention wins are bounded: even a 1.5× on the partial ≈ +5%@ctx1024, +12%@ctx4096.
Diminishing — the larger remaining gap is **base-decode** (GEMV + ~780 progs/token, the short-ctx ~55% gap).

## Candidates

| cand | lever | expected (decode) | risk | gate | verdict |
|---|---|---|---|---|---|
| **A. vectorized fp16 K/V loads in gqa_coop** | half2/uint4 coalesced V load (gqa_coop already reads V once; widen the load) | +3–5% @ctx≥1024 | med (renderer may not emit vector loads from the index pattern; partial may be issue- not width-bound) | ≥3% @ctx1024 or ≥5% @ctx4096, byte-identical | **PICK (smallest next attention step)** |
| B. stream-K / adaptive KV split | gqa_coop uses the kv-head axis → **4× fewer workgroups** than hoisted (8 vs 32) → occupancy-starved at short/mid ctx (why its win is +3.9%@512 vs +19.8%@4096). Stream-K splits KV to refill CUs + fixup/combine | +5–10% @ctx512–2048 (closes the short-ctx occupancy deficit) | high (extra fixup/combine kernels, partial-reduction correctness) | ≥5% @ctx2048 and ≥8% @ctx4096, no ctx512 regression | defer (bigger build; do if A small) |
| C. context-band variant policy | choose hoisted vs gqa_coop per ctx | ~0 | med | one variant wins a band by ≥3% | **REJECT** — gqa_coop dominates hoisted at *every* ctx (+3.9…+19.8%); no band-split earns its multi-graph cost |
| D. base-decode GEMV / program granularity | the 2.3× base-decode gap (not attention) | **large** (the short-ctx ~55% gap) | high, separate arc | own arc | **bigger prize — separate, not attention** |

## Recommendation

**If staying in attention: pick A (vectorized fp16 loads).** Smallest bounded step, builds directly on the
shipped gqa_coop kernel, low blast radius, clear kill gate. Escalate to **B (stream-K)** only if A's gain is
small — B specifically fixes gqa_coop's known occupancy deficit (4× fewer workgroups) and is the audit's other
llama ingredient, but it's a larger, higher-risk build.

**Honest steer:** attention is now Amdahl-bounded (A ≈ +5%, B ≈ +5–10% short/mid). The **larger remaining
campaign prize is base-decode (D)** — the short-ctx 55% gap is mostly GEMV/program-granularity, not attention.
If the goal is the headline tok/s, D (a separate arc) outweighs further attention tuning. If the goal is
long-context serving specifically, A then B continue to flatten the slope toward llama.

## Kill gates (carried into whichever is funded)
- A: isolated partial must show ≥1.2× *and* the win must survive **in-model W==D byte-identical** (isolated
  DEBUG2 is untrustworthy — the gqa_coop 3× was a warm-cache artifact; only in-model counts).
- B: must not regress ctx512; the fixup/combine overhead must not eat the occupancy gain.
- Any: no default flip without in-model ≥ gate at ctx1024+ and byte-identical greedy.
