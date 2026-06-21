# Decode Latency-Hiding Lifecycle / Codegen Result

Date: 2026-06-21

Scope: `docs/decode-latency-hiding-lifecycle-codegen-scope-20260621.md`

Verdict: **`ROADMAP`** — the latency-hiding *design* is valid (the fully-fused flash tile is byte-exact), but no
**bounded** prototype clears the gate: the naive fused tile is **2.5–3.3× slower** than the optimized split path,
because winning requires fusion **and** the per-kernel coop optimizations in one kernel — a deep codegen
capability (linearizer coupled-multi-reduce, or a raw-kernel↔JIT bridge for a hand-optimized tile). **Decode is
marked: bounded fusion exhausted; remaining path is broad lifecycle/codegen research.** Default decode behavior
NOT changed.

## Conclusions (preserved — bounded decode fusion is CLOSED)

The single load-bearing conclusion: **fusion alone is not the win.** The raw fully-fused flash tile was
byte-exact but *slower* because it lost the current UOp `gqa_coop_vec` path's GQA V-reuse, coalescing, and
dataflow advantages. The next valid decode project is therefore **not** another micro-fusion or launch-removal
pass; it is **"fused + coop-optimized in one primitive,"** which means linearizer/codegen work or a raw-kernel↔JIT
bridge. Specifically:

1. **Bounded FFN activation fusion (`E_49152` silu·up) — REFUTED as work-conserved.** Built byte-exact, eliminated
   the launch, **0% faster** (`docs/decode-ffn-activation-producer-fusion-result-20260620.md`). The down-prologue
   latency-hiding variant is also closed (4096× recompute blowup).
2. **Bounded attention reduce/stat micro-fusion — REFUTED / no-go.** The dominant `reduce_fixup` is the intrinsic
   O(KV) Q@Kᵀ score reduce, `softmax_stats`' slope is the O(KV) per-key `exp`; only ~0.5 ms of fixed helpers are
   fusible, and those are work-conserved (`docs/decode-attention-fusion-analysis-result-20260620.md`).
3. **Raw fully-fused flash tile — CORRECT but SLOWER.** Byte-exact (max_err 0.0) yet 2.5–3.3× slower than the
   optimized split coop path (266 vs 106 µs @ctx1024; 553 vs 165 µs @ctx4096) — it lacks GQA V-reuse/coalescing.
4. **Decode remains ~67% llama at steady context** (~86% only at ctx≈0 empty KV). The realistic-ctx headline is
   unchanged.
5. **Next work is ROADMAP / codegen only — not tactical patching.** No more bounded fusion, launch-removal, or
   micro-fusion passes (all refuted). The only live lever is an optimized fused flash primitive, gated by deep
   codegen capability (see §9).

## 1. Baseline and authority (Phase 0 reconciled)

`docs/decode-prefill-headline-reconciliation-result-20260621.md`: "87.6" is a numeric coincidence — the reported
"87.6 tok/s" was the genuine **ctx≈0 empty-KV rate** (reproduced ~85–86 today), which collides with a separate
ctx4096 `decode_ms=87.6` (=11.4 tok/s); either reading lands at the same decision. **Clean-wall decode (PROFILE=0,
auto clock, HEAD):**

| ctx | 0 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 85.7 | 70.8 | 68.0 | 66.3 | 63.5 | 60.6 |
| % llama | ~86% | — | 69% | 68% | 67% | 66% |

Host-sync 0% (GPU-bound). Stable baseline confirmed — the ~67% llama gap at realistic ctx stands.

## 2. Latency-hiding opportunity atlas (Phase 1)

| family | tinygrad visible cost | what llama hides it under | why llama hides it | tinygrad blocker | prototype |
|---|---:|---|---|---|---|
| attention online-softmax + score | reduce_fixup ~1.8ms + softmax_stats ~0.9ms @1024 (grows w/ ctx) | the fused `flash_attn_tile` | softmax/score ALU overlaps KV-tile HBM loads; no materialized score/stat tensors | linearizer can't couple q·k reduce with the softmax reduce → 6 separate kernels (`qk_flash_decode.py:73-80`) | **A: fully-fused flash tile** |
| FFN activation `E_49152` (silu·up) | ~1.24ms flat | inside the HBM-bound `mul_mat_vec_q` | activation ALU hides under weight-load latency | custom GEMV can't reuse act-loaded-once across all rows (coop loads per-workgroup only → 4096× recompute) **and** Phase B1 proved naive fusion conserves work | **B: closed** |
| weight GEMV (Q4/Q6 mmvq) | at llama parity (Del 0) | — | — | — | none (solved) |

Gate: the attention candidate's expected recovered movement is ≥5% @ctx1024 (attention is 23% of wall) and ≥7%
@ctx4096 (32% of wall) — **identified**, and it is latency-hiding (overlap), not launch-count reduction.

## 3. Selected prototype and why (Phase 2 → Candidate A)

**Candidate A — fully-fused flash decode tile.** Candidate B (FFN) is closed: Phase B1 (built, byte-exact,
0% — work conserved, `docs/decode-ffn-activation-producer-fusion-result-20260620.md`) plus the down-prologue's
4096× recompute blowup. So A is the only live latency-hiding lever, and its math already exists as the **raw C
kernels** at `qk_flash_decode.py:26-65` (`flash_partial_src` does Q·K + online-softmax + V accumulation in **one**
kernel; `flash_reduce_src` combines splits — 2 kernels total vs the UOp path's 6).

## 4. Correctness

Both paths **byte-exact** vs the numpy reference (max_err = 0.000) at ctx1024 and ctx4096. The fused tile is
numerically correct; the issue is purely performance.

## 5. Local diagnostic timing (clock-pinned, warm; `extra/qk_decode_fused_flash_tile_ab.py`)

Real decode shapes (Hq=32, Hkv=8, Hd=128), per-layer attention call, **both warm-JIT/precompiled** (fair):

| ctx | raw fused (2 kernels) | UOp coop (6 kernels) | fused speedup | correctness |
|---:|---:|---:|---:|---|
| 1024 | 266.1 µs | 105.7 µs | **0.40×** | both exact |
| 4096 | 553.3 µs | 165.0 µs | **0.30×** | both exact |

**The fully-fused tile is 2.5–3.3× slower.** The fusion (fewer launches, qk/softmax/V interleaved) is real, but
the raw kernel is the naive v1 style — scalar per-thread q·k, no GQA V-reuse, no coalesced V loads — and those
losses dwarf the fusion benefit. The optimized-but-split `gqa_coop_vec` path (V read once per G=4 heads, coalesced
fp16 loads, the shipped default) wins decisively. (Earlier a 41× "fused win" was a measurement artifact: the UOp
path was rebuilding its graph every call; warm-JITting it corrected the comparison.)

## 6. Full W==D timing

Not run: the local prototype did not clear its local gate (0.30–0.40× vs the ≥1.15× needed), so no candidate was
promoted to full-route W==D. Per the scope's stop condition ("local timing matches/loses → stop"), the bounded
prototype is refuted as-is.

## 7. Lifecycle-search encoding (Phase 4) — done

`bench/qk-lifecycle-search/`:
- candidate `decode_fully_fused_flash_tile` (state `roadmap_codegen_blocked`).
- refutations: `fused_flash_naive_loses_to_optimized_split`, `ffn_down_prologue_recompute_blowup`.

## 8. Default behavior changed: **NO**

`tinygrad/llm/model.py` unmodified. Prototype + harness are research-only under `extra/`. Clock pinned only for
local diagnostics; `auto` restored (verified). All headline numbers are clean-wall (PROFILE=0, auto).

## 9. Recommendation: **ROADMAP — the win requires the north-star codegen capability, not a bounded patch**

The bounded fusion lane is **exhausted and the latency-hiding prototype is refuted as a bounded patch**, but the
*design* is validated: the fused flash math is correct, and the loss is purely the naive kernel's missing
per-kernel optimizations. A **winning** decode lever therefore requires an **optimized fused flash tile** —
fusion (qk+softmax+V in one kernel) combined with the coop optimizations (GQA V-reuse, coalescing, occupancy),
i.e. llama's `flash_attn_tile`. That needs one of:

1. **Linearizer: coupled multi-accumulator reduce** — express a q·k reduce nested with the softmax reduce in one
   UOp kernel (currently trips range-ordering, `qk_flash_decode.py:73-80`; same class as the prefill
   POWN/software-pipelined-K wall). This would let the optimized fused tile be written in the UOp path and
   JIT-integrated.
2. **Raw-kernel ↔ JIT bridge** — let `custom_kernel` (or an equivalent) embed a hand-written/asm optimized fused
   flash kernel into the decode graph (the raw C path "can't bridge into the JITted attention graph",
   `qk_flash_decode.py:67-69`). Then port `gqa_coop_vec`'s V-reuse/coalescing into the single raw fused kernel.

Both are **multi-week codegen capability projects**, not tactical decode optimizations. Per the scope's expected
decision: **decode is bounded-fusion-exhausted; the remaining path is broad lifecycle/codegen research, and
pursuing it is the explicit north-star machine-search/codegen decision** — recommend the owner take it on only as
that funded effort, or **rest decode at the current route** (~86 tok/s @ctx0, ~61–68 @ctx512-4096, ~67% llama;
q8 opt-in +~6%). Do not spend more effort on bounded micro-fusion or launch-removal (refuted).

## Commands

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_fused_flash_tile_ab.py
DEV=AMD JIT=1 python3 -m tinygrad.llm.cli -m <Qwen3-8B-Q4_K_M.gguf> --benchmark 30   # ctx0 baseline
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_runtime_overhead.py              # ctx128-4096 baseline
```

## Artifacts

- `extra/qk_decode_fused_flash_tile_ab.py`, `bench/qk-decode-latency-hiding-lifecycle/fused_flash_tile_ab.json`
- `bench/qk-headline-reconciliation/result.json`, `docs/decode-prefill-headline-reconciliation-result-20260621.md`
- `bench/qk-lifecycle-search/{generated_candidates,refutations}.json`
