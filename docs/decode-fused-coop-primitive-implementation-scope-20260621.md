# Decode Fused+Coop Primitive — Implementation Scope (LINEARIZER_FIRST)

Date: 2026-06-21

Owner: next executor

Status: implementation scope (decision package upstream)

## Decision (roadmap package result)

**`LINEARIZER_FIRST`.** The only live decode lever — *fused + coop-optimized in one primitive* — is buildable in
the **UOp path with no compiler surgery**, because the fused tile is **already proven**:
`extra/lds_attention_tile.py` expresses a one-kernel flash tile (q·k + softmax + V) with LDS-cooperative K/V reuse
+ barrier, correctness-tested and faster than global-reread. The coupled-*online* softmax that trips the
linearizer is **never needed** — the two-pass single-accumulator formulation (pass1 max, pass2 weighted-V with a
1s-augmented denominator) is exact and fused.

Decision package artifacts: `bench/qk-decode-fused-coop-primitive/{path_diff,bridge_feasibility,
linearizer_feasibility,decision_matrix}.json`.

- **Why not BRIDGE_FIRST:** the bridge chain is proven (Tensile arc) but needs a still-unwritten optimized raw
  kernel + 2 chained PROGRAM nodes + the un-landed ProgramInfo grid-dim emission + a raw kernarg runner, and it is
  not north-star reusable. The bridge agent called it "lower-risk than fighting the linearizer" — but that fight
  is already won, so that premise is void. **BRIDGE stays the fallback** if the linearizer tile cannot beat
  `gqa_coop_vec` at decode shape.
- **The one bounded risk** (gated below): `lds_attention_tile.py` beats *global-reread*, not yet the optimized
  6-kernel `gqa_coop_vec` at decode shape. The first gate measures exactly that; a miss falls back cheaply.

## What to build

A new `FLASH_VARIANT=gqa_lds_fused` (default off) in `extra/qk_flash_decode.py`: a **single** UOp `custom_kernel`
that does, per (kv-head, KV-split) workgroup, the whole flash tile for that split's G=4 query heads — replacing
the current 6-kernel `gqa_coop_vec` lifecycle (score matmul + flash_max/prob/partial/gmax/den/combine).

Port the proven idiom from `extra/lds_attention_tile.py`, re-shaped for decode GQA:
- **workgroup = (kv-head `kvh`, split `s`)**; lanes cover the G=4 query heads × output-dim `d` (W=Hd+1, the 1s
  denom column).
- **LDS-stage K[kvh, split] and V[kvh, split] once** (cooperative load + `UOp.barrier`), reused across the G query
  heads and all d-lanes → kills both the raw tile's 128× q·k redundancy (q·k reads K from LDS) and its 4× V
  redundancy (GQA V-reuse). LDS budget: K+V fp16 = 2·L·Hd·2 ≤ 64KB → L≤128 at Hd=128 (matches `FLASH_L=128`).
- **two-pass single-accumulator** (the linearizer-legal idiom): pass1 running max over the split's keys; pass2
  weighted-V accumulate with the 1s-augmented denom; then the existing cross-split LSE combine (keep
  `flash_gmax/den/combine` OR fold the split reduction in — start by keeping it; it is small and already coop).
- symbolic start_pos / KV length via the **existing UOp path** (`Tc_b` bound for the score slice, `Tc_u` unbound
  twin for ranges) — already handled in `flash_decode_attention`; no new var-binding work.

## Exact files to edit

- `extra/qk_flash_decode.py` — add `gqa_lds_fused` to `FLASH_DECODE_VARIANTS`; add the fused-tile kernel
  generator(s); wire selection in `flash_decode_attention` (the `_partial` dispatch + a fused branch). The model
  already routes `FLASH_VARIANT` here (`tinygrad/llm/model.py:886`), so **no model.py edit** is needed.
- (reference only, do not edit) `extra/lds_attention_tile.py` (idiom source),
  `tinygrad/uop/ops.py` (range/set/after/end/placeholder/group/barrier), `tinygrad/codegen/late/devectorizer.py`
  (reduce_to_acc / merge_reduce_ends — same-range constraint).

## Feature flags

- `FLASH_VARIANT=gqa_lds_fused` — selects the new primitive (default stays `gqa_coop_vec`).
- `FLASH_L` — LDS tile length, default 128 (≤128 by LDS budget).
- No other env changes. **No default change**; the new variant is opt-in until the W==D gate passes and the owner
  approves a default flip.

## Gates (each must pass before the next)

1. **Toy / standalone gate** (`extra/qk_decode_fused_lds_tile_ab.py`, clock-pinned local diagnostic):
   - correctness: byte-exact vs SDPA / current flash decode at decode shape (Hq=32, Hkv=8, Hd=128), ctx 1024/4096
     (tolerance = existing flash policy, fp-reassoc).
   - **the decision-gating measurement**: fused LDS tile vs `gqa_coop_vec` (warm-JIT both, fair) must be
     **≥1.05× at ctx1024** (heading toward the ≥5% full-route target). If it merely matches or loses → **STOP the
     linearizer route, fall back to BRIDGE_FIRST or rest** (record as `fused_lds_tile_cannot_beat_coop`).
2. **One-layer in-model gate**: a single decoder layer runs the new variant end-to-end (real KV cache, symbolic
   start_pos), output matches the `gqa_coop_vec` layer within policy; no GPU hang/MMU fault.
3. **Full W==D gate** (`extra/qk_decode_runtime_overhead.py` with `FLASH_VARIANT=gqa_lds_fused`, clean wall
   PROFILE=0 auto clock, median-of-5): **≥5% @ctx1024 or ≥7% @ctx4096; no ctx512 regression >1%**; host-sync
   stays non-target.
4. **Correctness / dNLL / tok0 gate**: greedy tok0 byte-identical to default decode, or dNLL within the existing
   decode policy across ctx 512/1024/4096; VRAM reported.

## Rollback plan

The variant is a default-off `FLASH_VARIANT` value. Rollback = do not set the flag (the default `gqa_coop_vec`
path is untouched). If a gate fails, leave the kernel in `extra/` behind the flag (research) and record the
refutation; `tinygrad/llm/model.py` is never modified, so there is nothing to revert in the default route.

## Promotion (separate owner approval)

Only after the full W==D gate passes: consider flipping `FLASH_DECODE_DEFAULT_VARIANT` to `gqa_lds_fused`. That
default change requires explicit owner approval and a fresh clean-wall headline rerun.

## Artifact paths

- `extra/qk_decode_fused_lds_tile_ab.py`, `bench/qk-decode-fused-coop-primitive/fused_lds_tile_ab.json`
- `docs/decode-fused-coop-primitive-result-20260621.md` (build result, when executed)
- decision package: `bench/qk-decode-fused-coop-primitive/{path_diff,bridge_feasibility,linearizer_feasibility,decision_matrix}.json`
- lifecycle ledger: `bench/qk-lifecycle-search/generated_candidates.json`
  (`decode_fused_coop_lds_tile_LINEARIZER_FIRST`, state `fundable_linearizer_first`).

## Durable project state (unchanged by this scope)

Prefill solved + opt-in policy-shipped; global `PREFILL_V2` default off; bounded decode fusion closed;
steady-context decode ~67% llama; the only live decode lever is fused+coop in one primitive — now with a
**fundable LINEARIZER_FIRST route and a BRIDGE_FIRST fallback**. No model-route default changed by this package.
