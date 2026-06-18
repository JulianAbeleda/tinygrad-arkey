# PREFILL WMMA arc — Phase 0/1 audit (2026-06-18)

A prefill arc (not decode). Goal: use WMMA where it's structurally right — prefill large-M kernels. **Phase 1
proved the target: prefill attention QK/PV (no WMMA today, ~24% of the forward). lm_head last-token is REFUTED
(the jit already fuses it); FFN/projections are already WMMA via the PREFILL_V2 warmstart-TC.** RX 7900 XTX,
Qwen3-8B-Q4_K_M. No kernels built yet; no defaults changed.

## Phase 0 — prefill ledger

- llama pp512: **3069 tok/s**.
- tinygrad PREFILL_V2 (concrete-512 jit, fp16 realized weights, warmstart-TC): **2459 tok/s, 208 ms/512 = ~81%
  of llama** (13× over the symbolic baseline).
- Forward GPU breakdown (first-call per-kernel DEBUG2, warmstart applied):
  - **matmul (FFN gate/up/down + attn q/k/v/o projections): ~67.5%** — **already WMMA** (10 wmma kernels via
    warmstart-TC).
  - **attention QK/PV/softmax: ~24%** — **NO WMMA** (10 `start_pos` attention kernels, zero wmma).
  - lm_head: fused/cheap (see below); norm/rope/elementwise: ~3%.
- `bench/qk-prefill-wmma/baseline.json`.

## Phase 1 — rank WMMA targets

| target | share | current impl | WMMA applies? | verdict |
|---|---|---|---|---|
| A. FFN + projection matmuls | ~67.5% | warmstart-TC (**already WMMA**, 10 kernels) | yes, done | skip — hard to beat the existing TC |
| **B. attention QK/PV** | **~24%** | **plain reduce, NO WMMA** (10 start_pos kernels) | **yes — T=512 large-M** | **EARNED — the target** |
| C. lm_head over prefill | fused | jit fuses `[:, -1, :]` into lm_head | n/a | **REFUTED** |
| D. norm/RoPE/residual | ~3% | elementwise | no | too small |

### C (lm_head last-token) — REFUTED [measured]
`forward` does `logits()[:, -1, :]`. Eager, lm_head over all 512 tokens is 183 ms (40% of a 462 ms eager
forward) vs 25 ms last-token — a 7.2× waste. **But in the PREFILL_V2 jit the slice is already fused** (full-graph
schedule computes only the last token): adding an explicit `last_only` slice moved pp512 only **2467 → 2485
(+0.7%)**, below gate. The eager 40% was a no-fusion artifact. Change reverted (no dead flag). lm_head is not a
prefill lever here.

### B (attention QK/PV) — EARNED
Prefill attention at T=512 is large-M (the regime where the decode-attention WMMA refutation does NOT apply —
that failed only because decode is low-M/cache-bound). The QK score matmul ([T,Hd]×[Hd,T] = [512,512]/head) and
PV ([512,512]×[512,128]) are natural WMMA targets, and they currently use **no tensor cores**. Amdahl: attention
~24%; if WMMA gives QK/PV ~1.5-2× → **~+5-12% e2e** (the softmax part isn't WMMA-able, so realistic is the lower
end). Clears the +5% expected-upside bar to proceed to Phase 2.

## Phase 2 plan (next — the WMMA build)

Build an isolated prefill-attention WMMA tile via the revived SHAPED_WMMA custom_kernel path:
- shapes: T=512, Hq=32, Hkv=8, Hd=128, GQA=4, **causal**, fp16.
- QK^T tile (WMMA) → scale + causal mask → softmax → PV tile (WMMA).
- gate: isolated ≥1.3× vs the current PREFILL_V2 attention path, correct (dNLL-class tol), sane compile/memory,
  no cache artifact.
- then Phase 3 in-model behind `PREFILL_WMMA=1` (only when `PREFILL_V2=1`, concrete T=512, validated shape, AMD):
  pp512 ≥+10%, dNLL ≤0.01, no decode regression.

**Risk:** the SHAPED_WMMA path is revived (smoke + Q@K tile passed) but a full causal-GQA softmax-fused prefill
attention is a substantial new-kernel build; WMMA fragment layout + causal masking + the symbolic start_pos may
need care. Recommend explicit go before the build.

## Status
Phase 0/1 done. Target proven: **prefill attention QK/PV WMMA** (~24%, no WMMA today). lm_head refuted, FFN
already WMMA. Phase 2 (the WMMA attention tile) is the next build — deferred for go/no-go.

## Files
`bench/qk-prefill-wmma/baseline.json`, this doc. No model changes (last-token probe reverted).
