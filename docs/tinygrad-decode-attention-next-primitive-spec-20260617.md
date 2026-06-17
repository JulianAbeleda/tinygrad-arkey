# tinygrad decode-attention — next primitive target spec (2026-06-17)

Derived from the llama ROCm audit (`llama-rocm-decode-attention-audit-20260617.md`). Design only — no kernel
built here. The audit pins exactly what makes llama's decode attention context-flat, so this target is grounded
in the real primitive, not a guess.

## What llama wins with (audited, not guessed)

llama decode attention = the **TILE** kernel (`fattn-tile`) + stream-K split + combine. Its advantage over
tinygrad's `flash_partial_v2` is a **combination**, not one trick:
1. **GQA group batched as the tile width** (`ncols = gqa_ratio = 4`): K/V tile staged once, reused across the 4
   query heads → **no 4× V re-read** (tinygrad re-reads V per head).
2. **Stream-K adaptive KV split**: KV split across enough blocks to fill all CUs at any context (occupancy=8/CU)
   + a fixup/combine — vs tinygrad's fixed-`L` split (`S=cdiv(KV,128)`), which under/over-fills by context.
3. **Vectorized coalesced fp16 LDS staging** (half2 16-byte copies) → high effective BW; tinygrad's per-thread
   strided reads sit at ~33 GB/s.

**Why this matters for the "don't pick WMMA/LDS-v3 again" rule:** v3 refuted *naive single-query LDS/WMMA*.
llama uses LDS tiling and wins — but only *with* (1)+(2). The decisive ingredients are GQA-batched tile width +
stream-K occupancy + vectorized loads, which the v3 probe lacked. So the target below is NOT a repeat of v3.

## Candidate scoring

| candidate | effect ctx512 / 1024 / 4096 | impl risk | correctness risk | search knobs | kill gate |
|---|---|---|---|---|---|
| **A. GQA-batched cooperative tile + vectorized fp16 LDS load** (read K/V once per group, half2/uint4) | small / med / **large** | med-high (UOp LDS + vectorized) | low (exact vs SDPA) | tile width=ncols, nbatch_fa, vec width | if effective BW ≤ ~1.3× of 33 GB/s isolated → stop |
| B. Stream-K adaptive KV split + combine | small / med / large | high (fixup/combine, partial dedup) | med (reduction order) | nblocks, split policy | if no occupancy gain over fixed-L at small KV |
| C. Alternative KV layout | ~0 | low | low | layout | already fp16/contiguous → low value |
| D. KV quantization (Q8/Q4 cache) | BW ↓ but quality risk | med | **high (dNLL)** | quant type | dNLL gate |

A and B are the audited levers; C/D are secondary. **A subsumes the core inefficiency (4× V + low BW); B adds
occupancy.** They compose into "llama's tile," but A is the smaller, more decisive *first* test.

## Selected first target: **A — GQA-batched cooperative decode-attention tile (vectorized fp16 LDS), isolated**

Replace `flash_partial_v2` (the 47%@ctx4096 kernel) with a tile that processes the **GQA group (G=4 query
heads) together**, staging the K/V tile into LDS **once** with **vectorized coalesced fp16 loads**, reused across
the 4 heads. Keep the existing `flash_max/prob/gmax/den/combine` scaffold initially (swap only the partial), or
fuse scores+softmax+PV into the tile as a stretch.

- **Signature (isolated):** `partial_gqa_tile(prob[Hq,KV], V[Hkv,MAXC,Hd], KV, L, nbatch_fa) -> pout[Hq,S,Hd]`,
  workgroup = (kv-head, KV-split), tile width = G, LDS V tile [nbatch_fa, Hd] vectorized-loaded once.
- **Microbench (isolated, the gate that v3's naive-LDS failed but with the missing ingredients):** vs current
  `flash_partial_v2` at KV 512/1024/4096, DEBUG=2 device tm + computed effective GB/s.
  - **Pass:** ≥1.3× on the partial kernel **and** effective BW materially above 33 GB/s (toward IC/HBM).
  - **Kill:** if GQA-batched + vectorized + LDS is still ≤ current (the IC-served baseline wins *again*, now
    even with the audited ingredients) → the gap is not closable in tinygrad's codegen for this regime →
    document and stop; do not escalate to full stream-K.
- **In-model gate (only if isolated passes):** wire behind `FLASH_VARIANT=gqa_tile`, ctx 512/1024/2048/4096,
  **byte-identical greedy**, W==D; need **≥5% decode @ctx1024**, **≥10% @ctx4096**, no >2% ctx512 regression.
- **Then (only if A pays):** add **B (stream-K split)** for further occupancy.

## Honest priority caveat (read before funding)

Attention is the **long-context** lever, not the headline gap. Perfect (flat) attention would make tinygrad
~flat at its base-decode rate (~43 tok/s) — i.e. ctx4096 ~24.8 → ~40 (**+60% at long ctx**), but still **~44%
of llama**, because the **base-decode 2.3× gap (GEMV + ~780 progs/token vs llama's fused kernels)** is the larger
structural limiter at common contexts. If the goal is the headline number, base-decode is the bigger prize; if
the goal is long-context serving, target A is correct.

## Expected upside (Amdahl, from measured shares)

`flash_partial_v2` ≈ 13%@ctx512 / 21%@ctx1024 / 47%@ctx4096 of decode. If A reaches ~2× on it (plausible if BW
lifts from 33 GB/s toward ~70+):
- ctx512 ~+7%, ctx1024 ~+12%, ctx4096 ~**+31%** decode. (Clears the in-model gate at ctx≥1024.)
- Lower bound (1.4× on the kernel): ctx1024 ~+6%, ctx4096 ~+18%.

## Canonical matched baseline

`docs/qk-llama-baseline-xtx-20260617.md` + `bench/qk-llama-baseline-xtx/result.json` (Phase 5: measured llama
vs tinygrad % at ctx 512/1024/2048/4096, XTX provenance correction). This audit's provenance:
`bench/llama-rocm-attention-audit/provenance.json`.
