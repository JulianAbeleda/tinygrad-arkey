# Parity-closure loop run — route-binding finding (2026-06-27)

First run of the parity-driven loop (`.claude/loop.md` + `qk_owned_oracle_parity_audit.py`). It worked as designed
and surfaced a high-value finding the raw metric would have hidden.

## The run
1. **Parity audit** → `PARITY_OPEN__FAILED_ROWS_TARGETABLE`; `searchable_failed_rows = [waitcnt, latency_shadow_fill,
   split_kv_combine]`.
2. **Generator (`--failed-rows`)** → the only untried candidate targeting a failed row:
   `DECODE_ATTN_BLOCK_TILE_FIXED_S=1, DECODE_ATTN_FUSED_XLANE_SCORE_PV_S=64` (targets `split_kv_combine`,
   `requires_wd`). No knob/hand-picking.
3. **W==D gate** (the only promotion authority) → **96.9 / 89.0 tok/s @ ctx512/4096 ≈ 94% of owned (103/94)**.

## Why this did NOT become PROMOTABLE — route attribution caught a false positive
94% of owned from the block-tile route would be a 14× jump over its baseline (32.8/6.2) from a split-count change —
implausible. `DEBUG=2` attribution showed the decode-attention kernel that fired was **`owned_flash_tile_gqa_whole`**
(×36), NOT `flash_block_tiled`. The candidate's flags **fell back to owned** (`DECODE_ATTN_AMDGCN_TILE` defaults to 1
and takes precedence). The 94% was *owned's* W==D, not the candidate's.

Forcing `DECODE_ATTN_AMDGCN_TILE=0` did **not** bind the block tile either — it fell back to **`gqa_coop_vec`**
(`flash_partial_coop_vec` + `flash_max/prob/den/gmax/combine`).

## Finding: the generated block-tile route is NOT route-bound in the model W==D path
With the flags as the search space declares them, the block tile **never fires in-model** — it always falls back
(owned with `AMDGCN_TILE=1`, gqa_coop_vec with `AMDGCN_TILE=0`). Consequences:
- The candidate **cannot be W==D-evaluated** as wired → outcome `TOOLING_BUG` (the move is unobservable; underlying
  `PRIMITIVE_PLACEMENT_BUG`: the primitive exists but is not route-bound).
- The transfer snapshot's `block_tile_route_full_stack` W==D (32.8/6.2) is **unverified** — it is the
  session-reported number the audit already flagged (`wd_authority=session_reported_not_harness_measured`); the
  block tile may never have been truly W==D-measured in-model.

## The loop worked
The parity-closure discipline did exactly its job: it **refused to record PROMOTABLE from raw tok/s** and forced
route attribution + token-match, catching a false 94%-of-owned win and exposing a deeper tooling/route-binding gap.
Without it, the loop would have falsely claimed promotion.

## Next step (named, not chased)
**Route-bind the generated block tile into the model decode path** so the W==D harness actually exercises it (a
`PRIMITIVE_PLACEMENT_BUG`/route-binding fix in `tinygrad/llm/model.py` decode-attention selection), and regenerate a
**harness-measured** transfer snapshot. Only then can `split_kv_combine` / `wd_tok_s` parity rows be honestly
evaluated. The parity matrix should add a `route_bound` check (does `flash_block_tiled` actually fire in-model?)
before any W==D row is trusted.

Ledger after this run: 16 REFUTED_NO_SLOPE, 2 REFUTED_OCCUPANCY, **1 TOOLING_BUG**. Not pushed.
