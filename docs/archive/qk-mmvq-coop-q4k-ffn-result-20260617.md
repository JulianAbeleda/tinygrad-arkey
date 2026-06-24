# MMVQ_COOP → Q4_K — ffn_gate/up REFUTED (isolated); attn_q/o is the validated next target — 2026-06-17

Extending the cooperative-K MMVQ family from Q6_K to Q4_K. Assigned role: **Q4_K ffn_gate/up**. Verdict:
**refuted at the isolated gate** — but the probe revealed the Q4_K story is **role-dependent**, and **attn_q/o**
is the real Q4_K opportunity. RX 7900 XTX, Qwen3-8B. Built `q4k_coop_partial_kernel` (correct, kept); no
defaults changed; ffn_gate/up not wired.

## The Q4_K cooperative-K kernel

`q4k_coop_partial_kernel` (sibling of `q6k_coop_partial_kernel`): the Q4_K quant word index is
`4 + (grp//2)*8 + pos//4`, so the within-block **word index `lane4` = pos//4 (0..7)** becomes a LOCAL lane axis
→ adjacent lanes read adjacent packed words → coalesced. Each lane reads one qword + its 4 nibbles across the 8
groups (`_q4k_block_dot_packed_load`), writes its own `partials[row, lane4]`; stage-2 `.sum(axis=1)` reduces the
8 lanes (no in-kernel reduce). Correct (fp-reassoc tol, err ~2e-6) on all roles tested.

## Phase 0–3 — isolated (real weights, fresh input)

| Q4_K role | shape | count | base GB/s (% peak) | coop GB/s (% peak) | speedup | gate ≥1.3× |
|---|---|---|---|---|---|---|
| **ffn_gate/up** | 12288×4096 | 72 | 365.6 (**41%**) | 423 (47%) | **1.16–1.18×** | **❌ FAIL** |
| **attn_q/o** | 4096×4096 | 72 | 169 (**19%**) | 258 (**29%**) | **1.47–1.52×** | **✅ PASS** |

err 2e-6 (correct), all < HBM peak (real, not less-work).

## Verdict — ffn_gate/up REFUTED

**The Q4_K default for ffn_gate/up is already well-coalesced (41% of HBM peak)** — nothing like the Q6_K
disaster (10–14%). So cooperative-K adds only +16–18% (47% peak), below the 1.3× isolated gate → **refuted, not
wired** (per "stop at isolated <1.3×"). The Q6_K wins were large precisely because Q6_K's default was uniquely
bad; **Q4_K is role-dependent.**

## Phase 5 — next target (data-driven)

**Q4_K attn_q/o** is the validated next target: its default is poorly coalesced (**19% peak**), coop gives
**1.52×** (29% peak), cleared the isolated gate. It was out of scope for this task ("do not touch attn_q/o yet"),
so it is **not wired** — but the kernel is built and probe-validated; the only remaining step is its in-model
W==D gate (expected borderline-to-good +3% e2e; q/o is a smaller share than gate/up). **Recommend greenlighting
Q4_K attn_q/o next.** Q4_K ffn_gate/up should NOT be revisited (already coalesced).

## Status of the MMVQ_COOP family

| role | quant | base % peak | coop | status |
|---|---|---|---|---|
| lm_head | Q6_K | 10% | 5.0× | SHIPPED (default on) |
| ffn_down | Q6_K | 14% | 2.77× | SHIPPED (default on) |
| ffn_gate/up | Q4_K | 41% | 1.18× | REFUTED (already coalesced) |
| attn_q/o | Q4_K | 19% | 1.52× | validated isolated — next (not wired) |

Decode currently ~64% of llama (the two Q6_K roles). ffn_gate/up adds nothing; attn_q/o is the remaining bounded
coop lever.

## Files / commits
`extra/q4_k_gemv_primitive.py` (`q4k_coop_partial_kernel`, `[codegen]` — correct durable primitive, validated for
attn_q/o), this doc (`[docs]`). Artifact `bench/qk-mmvq-coop-q4k-ffn/role_inventory.json`. No `[nn]` (nothing
wired — ffn_gate/up refuted, attn_q/o deferred).
