# PREFILL WMMA Phase 2 — isolated WMMA attention PASSES; in-model salvage validated (2026-06-18)

Phase 2 of the prefill WMMA arc. **Isolated gate PASSES (TC attention 2.5× over SDPA, correct).** The prior
in-model refutation was purely the symbolic-KV blocker; this phase **validates the deferred salvage (concrete
start_pos is achievable in-model)** and pinpoints exactly what Phase 3 needs. No model.py change, no defaults
touched. RX 7900 XTX, Qwen3-8B.

## Reuse of prior work

`PREFILL_TC_ATTENTION` (Option B: explicit Q@Kᵀ TC → fp16 scores → softmax → P@V TC, GQA via broadcast) was
already built/probed (`docs/amd-prefill-tc-attention-probe-20260617.md`, `extra/qk_prefill_tc_wr_softmax_probe.py`).
Rather than rebuild a SHAPED_WMMA tile, Phase 2 re-verified it and tested the one thing the prior probe could not.

## Phase 2C/2D — isolated gate (re-verified)

`qk_prefill_tc_wr_softmax_probe.py`, real prefill attention shapes (Hq=32, Hkv=8, Hd=128, GQA=4, causal):

| KV | SDPA | explicit TC | speedup | rel err | tc_fired |
|---|---|---|---|---|---|
| 512 | 1.05 ms | 0.45 ms | **2.36×** | 0.037 | ✓ |
| 1024 | 2.18 ms | 1.00 ms | **2.19×** | 0.020 | ✓ |
| 3584 | 7.30 ms | 2.86 ms | **2.55×** | 0.016 | ✓ |

**PASS** (≥1.3×, correct to fp16 reassoc/q-tol). The WMMA prefill-attention win is real.

## In-model status — the symbolic-KV blocker and its salvage

Prior probe: in-model TC attention was **0.79-0.92× (slower)** — root cause: prefill `start_pos` is **symbolic**
(`v_start_pos.bind`, for one jit across chunks), so `KV=start_pos+T` is symbolic and the concrete-shape TC does
**not** fire. Confirmed here: symbolic start_pos → **attention-wmma = 0**.

**New this phase — the deferred salvage is feasible:**
- Passing a **concrete** int `start_pos=0` to a **fresh** model runs cleanly in-model (1353 tok/s, no error).
  The prior "blocked by jit arg plumbing" was only that the shared `prefill_v2_jit`, once captured symbolic,
  rejects a concrete arg (`JitError: args mismatch`). A fresh/per-start_pos jit captures concrete fine.
- **But concrete-KV SDPA attention still does NOT auto-TC** (attention-wmma = 0 even with concrete start_pos):
  the default gfx1100 optimizer only applies TC via the warmstart-opts table, which covers the FFN/projection
  linears, **not** attention. So concrete KV alone is insufficient.

**Conclusion: the win requires BOTH (a) the explicit TC-attention branch (Option B) AND (b) concrete KV
(concrete start_pos per chunk).** With both, the 2.5× isolated win fires in-model (concrete KV is the only
structural difference between the 2.5×-standalone and 0.8×-in-model, now shown achievable).

## Phase 2E verdict: isolated PASS → Phase 3 earned (with a compile-cost caveat)

## Phase 3 plan (deferred for go/no-go — model.py change)

1. **Re-add the gated `PREFILL_WMMA` TC-attention branch** in `_attention` (Option B from the probe), active only
   when `PREFILL_V2=1` AND `start_pos` is a concrete int AND validated Qwen3-8B shape AND AMD. Decode (flash) and
   symbolic prefill untouched; falls back to SDPA otherwise.
2. **Concrete start_pos per chunk:** the prefill loop binds `start_pos` to a concrete int per 512-chunk so KV is
   concrete and the TC fires. Cost: **one jit/compile per distinct start_pos** (e.g. 0/512/1024/1536 for a 2048
   prompt) instead of one symbolic jit — extra first-use compile latency (warm pp512 unaffected; long-prompt
   first-token latency rises). For pp512 (single 512 chunk) it's exactly 1 jit.
3. **Gates:** warm pp512 ≥+10% (expected ~+10-14%: attention ~24%, TC ~2.5× on the matmul part, softmax
   unchanged); dNLL ≤0.01 (fp16 scores — the probe was greedy-byte-identical on the smoke); no decode regression;
   VRAM within budget (Option B materializes [Hq,T,KV] fp16 scores: 32×512×512×2 = 16 MB/layer transient — sane).
4. Default decision after gates: keep behind `PREFILL_V2` boundary; default `PREFILL_WMMA=1` only if pp512 ≥+10%
   AND dNLL accepted AND the per-chunk compile cost is acceptable; else opt-in.

## Final report
1. baseline attention: SDPA, no WMMA, ~24% of forward.
2. candidate: explicit Q@Kᵀ TC + softmax + P@V TC (Option B, exists).
3. correctness: rel err 0.016-0.037 (fp16/q reassoc) — acceptable.
4. speed: **isolated 2.36-2.55× PASS**; in-model needs concrete KV (now shown feasible) + the explicit branch.
5. memory: ~16 MB/layer transient fp16 scores — sane.
6. **Phase 3 earned** (isolated passed; salvage feasible) — pending go/no-go (model.py change + per-chunk jit).
7. files: this doc; `bench/qk-prefill-wmma/baseline.json`; reused `qk_prefill_tc_wr_softmax_probe.py`,
   `qk_prefill_tc_attention_measure.py`. No model/default changes.

## Risk / honesty
The 2.5× is on attention matmuls only; softmax (not TC) caps the realized attention speedup nearer ~1.5-2×, so
e2e is more likely +8-12% than +14%. The per-chunk concrete-jit recompile is a real cost for multi-chunk prompts
(not for pp512). dNLL must be run (fp16 scores are lossy vs SDPA's higher-precision path).
