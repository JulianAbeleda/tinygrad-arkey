# Decode Fused+Coop Primitive — Build Result (first gate)

Date: 2026-06-21

Scope: `docs/decode-fused-coop-primitive-implementation-scope-20260621.md` (LINEARIZER_FIRST)

Verdict: **FIRST GATE FAILED — STOP the fused-tile route, return to bridge-fallback or rest.** A fused flash tile
that adds the two missing coop features (LDS K/V reuse + GQA V-reuse) to the raw fused tile is **byte-exact but
0.21× @ctx1024 / 0.12× @ctx4096** vs the current winner `gqa_coop_vec` — *worse* than the raw fused tile's 0.40×.
Classified blocker: **decode-regime occupancy + q·k compute-mapping**, not LDS/barrier or V-reuse. Per the owner
directive ("do not tune blindly; record the blocker"), the fused-scalar-tile lever is **refuted at decode shape**.
Default decode behavior NOT changed.

## First-gate measurement (`extra/qk_decode_fused_lds_tile_ab.py`, clock-pinned, warm, byte-exact)

| ctx | fused-LDS+GQA tile | gqa_coop_vec (winner) | fused/coop | gate ≥1.05× |
|---:|---:|---:|---:|---|
| 1024 | 511.2 µs | 106.6 µs | **0.21×** | MISS |
| 4096 | 1344.9 µs | 164.4 µs | **0.12×** | MISS |

Both byte-exact (err 0.000). For reference, the *raw* fused tile (no LDS, no GQA) was 0.40× / 0.30×
(`docs/decode-latency-hiding-lifecycle-codegen-result-20260621.md`). **Adding LDS K/V reuse + GQA made it
WORSE, not better.**

## Why it failed (classification — into the owner's named buckets)

1. **Occupancy (dominant).** GQA consolidation moves the workgroup from per-query-head (`Hq·S` = 256 workgroups
   @ctx1024) to per-kv-head (`Hkv·S` = 64 workgroups), a **4× drop in workgroup count**, and serializes the G=4
   query heads in an in-kernel loop. The "coop" GQA reuse *reduced* the parallelism that was filling the GPU. This
   is why LDS+GQA (0.21×) is worse than the raw per-query-head tile (0.40×).
2. **q·k compute-mapping (structural).** The fused tile computes the q·k score with 128 d-threads each recomputing
   the full 128-mul dot (per-thread redundancy). LDS staging fixes the K/V *memory* redundancy (and correctness is
   exact) but **cannot fix the ALU redundancy** — the score is still computed ~128× over. `gqa_coop_vec` computes
   the score **once via a matmul** (efficient, no redundancy), which a scalar fused tile cannot match.
3. **NOT LDS/barrier overhead** (the barrier is one per workgroup, negligible) and **NOT V-reuse failure**
   (V-reuse worked; output byte-exact). The cooperative-q·k alternative (reduce the dot across d-lanes) was not
   built because it (a) re-implements the matmul as L×G LDS reductions = more barriers, and (b) does not address
   the occupancy loss — both predicted to remain below 1.05× for the same regime reason (#4).
4. **Regime mismatch (root cause).** The proven toy `extra/lds_attention_tile.py` wins at **prefill** because it
   has a T (multi-query) axis (16–32 query positions) that fills the workgroup with independent threads, each
   reusing LDS K/V. **Decode has T=1** (a single token), so there is no query-parallelism to fill the workgroup;
   the parallelism must come from the Hd output lanes, which forces either the per-thread q·k redundancy or
   cross-lane reductions. The toy's *capability* (a fused single-kernel tile is linearizer-expressible) is real,
   but its *speedup* is prefill-specific and does not transfer to the decode regime.

## Consequence for the decision

`LINEARIZER_FIRST` was the right call on *feasibility* (the fused tile compiles, runs, is byte-exact, needs no
compiler surgery), but the **performance premise does not hold at decode shape**: a fused *scalar* tile is
structurally bounded below the matmul-based split `gqa_coop_vec`, because (a) decode lacks the multi-query
parallelism fused tiles exploit, and (b) the q·k wants a matmul/tensor-core, not hand-rolled scalar reduction.

**This also bounds the BRIDGE fallback as originally scoped:** wrapping the *existing scalar* raw tile would ship
the 0.21–0.40× loss. A bridge could only win by wrapping a **WMMA/tensor-core flash tile** (llama's
`flash_attn_tile` approach) — i.e. the q·k and P·V on tensor cores, hiding softmax under the tile loads. That is a
genuinely deeper kernel (RDNA3 WMMA flash, the prefill-class codegen), not a wrap of what exists.

## Recommendation: **REST decode at the current route** (or fund a WMMA flash-decode kernel as a separate project)

- The fused-scalar-tile lever (linearizer port AND a simple raw bridge) is **refuted at decode shape** — measured,
  not inferred (0.21× / 0.12×).
- The only remaining path to llama-class decode attention is a **WMMA/tensor-core flash-decode tile** (q·k + P·V
  on WMMA, online softmax hidden under tile loads). That is the deepest codegen lane (RDNA3 WMMA, the same class
  as the prefill POWN/software-pipelined wall) and is a distinct, multi-week project — fund only as an explicit
  north-star decision.
- Otherwise **rest decode**: ~86 tok/s @ctx0, 68/66/61 @ctx512/1024/4096 (~67% llama), q8 opt-in +~7%. Bounded
  fusion, micro-fusion, launch-removal, and now fused-scalar-tile are all closed/refuted.

## Gates status

| gate | result |
|---|---|
| correctness (byte-exact vs ref) | PASS (err 0.000) |
| local ≥1.05× vs gqa_coop_vec @ctx1024 | **FAIL (0.21×)** — STOP per scope |
| one-layer in-model / W==D | not run (local gate failed) |

## Artifacts

- `extra/qk_decode_fused_lds_tile_ab.py`, `bench/qk-decode-fused-coop-primitive/fused_lds_tile_ab.json`
- decision package: `bench/qk-decode-fused-coop-primitive/{path_diff,bridge_feasibility,linearizer_feasibility,decision_matrix}.json`
- lifecycle ledger: `bench/qk-lifecycle-search/{generated_candidates,refutations}.json`

## Boundary

No decode default changed (`tinygrad/llm/model.py` untouched; no `FLASH_VARIANT` wired). The fused raw tile is a
research harness in `extra/`. Clock pinned for the local diagnostic; `auto` restored after (verified).
