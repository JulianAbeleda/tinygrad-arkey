# Q6_K split-K dp4a (ffn_down / lm_head) — REFUTED at the Phase-0 Amdahl gate (not built) 2026-06-17

The last bounded decode-GEMV stone. Verdict: **refuted without building** — the realized in-pipeline dp4a
speedup (Q4_K precedent ~1.04×) makes ≥5% e2e implausible despite a large Q6_K share. RX 7900 XTX, Qwen3-8B.

## Phase 0 — role map + share + Amdahl [measured]

Enumerated primitive linears (199). Q6_K roles (37) — and the premise correction: these are **parts==1**, not
parts>1:

| role | shape (out×in) | quant | parts | count |
|---|---|---|---|---|
| lm_head | 151936×4096 | Q6_K | **1** | 1 |
| ffn_down (half the layers) | 4096×12288 | Q6_K | **1** | 18 |
| ffn_down (other half) | 4096×12288 | Q4_K | 4 | 18 |
| attn_k/v | 1024×4096 | Q6_K | 4 | 18 |

So the untested branch is really **Q6_K dp4a (parts==1 lm_head + Q6_K-ffn_down)**, not "split-K parts>1".

Decode share @ctx512 (gqa_coop_vec default, eager proxy): q4k_gemv 31.4%, other 24.0%, **q6k ffn_down/kv 17.4%
+ q6k lm_head 14.0% = Q6_K 31.5%**, attention 13.1%.

**Amdahl ceiling** on the 31.5% Q6_K share:
- optimistic standalone **1.77×** → **+15.9%** e2e (could clear 5%)
- **realized in-pipeline ~1.04×** (the Q4_K dp4a precedent) → **+1.2%** e2e (far below 5%)

## Why the realized speedup is ~1.04×, not 1.77× [the decisive argument]

The Q4_K dp4a path (`Q4K_VDOT`) measured **+1% e2e** in-model W==D
(`qk-base-decode-gemv-structural-plan-20260617.md`), even though the dp4a kernel is ~1.77× standalone. Reason:
the decode GEMV is **in-pipeline bandwidth/issue-bound** — each weight is read once (arithmetic intensity ~0.3-0.5
FLOP/byte, memory-bound regardless of quant/shape), so removing the dequant ALU (dp4a) barely moves e2e; the
standalone 1.77× (and READRAW 730 vs 365 GB/s) were **warm-cache artifacts**. Q6_K is the **same memory-bound
GEMV class** (lm_head alone reads ~500 MB Q6_K/token; ffn_down ~40 MB) with the **same q8_1+dp4a mechanism**, so
it will realize the same ~1.04×. No reason large-K (ffn_down) or large-out (lm_head) roles would be more
dot-bound — they are *more* bandwidth-bound (bigger reads).

## Verdict — REFUTED (gate not earnable), not built

≥5% e2e is implausible (realistic +1.2%); a Q6_K 6-bit-pack dp4a split-K kernel is substantial work
(`q6_k_gemv_primitive.py` is fp dequant; dp4a needs int 6-bit unpack + q8_1 + parts accumulation) for an
expected +1%. Per "build only if the gate earns it" / "no broad GEMV rewrite" → do not build.

**This was the last bounded decode-GEMV branch.** See `qk-decode-bounded-levers-exhausted-20260617.md`.
