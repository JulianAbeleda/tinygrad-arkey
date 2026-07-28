# PREFILL_CAUSAL_TILE_SKIP promotion readiness (prepared, NOT flipped)

Status as of 2026-07-24: **default stays OFF.** This document assembles the evidence and mechanism for a human to
flip the default; it does not flip it.

## What this is

Commit `c44905a18` added `PREFILL_CAUSAL_TILE_SKIP` (default OFF) to
`amd_gfx1100_q16_grid_hd128_loop_attention`. It replaces the fixed KV-loop extent with a dynamic per-wave bound
`min(full_kv_tiles, ceildiv(query_start + q_tile*16 + 16, 16))`, so a wave stops once its own last query row is
covered. Exact: a fully-masked tile is a mathematical no-op (weight 0 everywhere, `old_m == new_m` so `alpha == 1`).

## Promotion mechanism

This is **not** a new route. It is a flag inside the already-promoted `prefill_flash_attention_generated` row in
`extra/llm_research/route_manifest.py`: same route id, same kernel emitter (`amd_gfx1100_q16_grid_hd128_loop_attention`), same
`route_attribution` chain, same `provenance` (`machine_authored_generated`) whether the flag is 0 or 1. The dynamic
loop bound is derived entirely from descriptor fields (`query_start`, `q_tile`, `grid.q_tiles`) already threaded
through `FlashPrefillAttentionSpec` -- it does not introduce a new handwritten kernel, ASM, or a second emitter
variant, so it stays inside `FINAL_DEFAULT_PROVENANCE` (`machine_authored_generated` / `tinygrad_scheduler_generated`)
and nowhere near the forbidden set (`external_handwritten_kernel`, `hand_authored_uop_template`, `rollback_oracle`).

Because the route identity does not change, promoting the flag is an **amendment to the existing row**
(`extra/llm_research/route_manifest.py`'s `prefill_flash_attention_generated` `promotion_artifacts`/`note`), not a new `ROUTES`
entry. `extra/llm_research/pure_search_guard.py`'s `HOT_FAMILIES["prefill_attention"]` model resolves this route unconditionally
pure regardless of the flag's value (`_prefill_attention_rolled_back` always returns `False`, since there is no env
key that de-selects the base route) -- so the guard would not, on its own, notice a change to the flag's default.
The authority gate below is the thing that must gate the flip, not the guard.

## Authority gate

`extra/llm_research/prefill/prefill_causal_tile_skip_promotion_gate.py` (run: `PYTHONPATH=. python3
extra/llm_research/prefill/prefill_causal_tile_skip_promotion_gate.py`):

- Derives the **required shapes** directly from `ROUTES["prefill_flash_attention_generated"]["shape_guards"]`
  (currently 8B `Hq=32` and 14B `Hq=40`, both `Hd=128`) -- not hardcoded, so a future shape_guard addition is
  automatically covered.
- Reads evidence **only** from `docs/prefill-causal-tile-skip-evidence-20260724.json`, a structured transcription of
  the measured numbers already collected (numerics, real-model token parity, three same-session paired A/B whole-model
  runs, attention-local PMC). It never invents a number.
- For each required shape, requires: numerics PASS, token-parity PASS, whole-model paired A/B with >= 3 pairs, mean
  delta >= 1.0% at both pp512 and pp4096, signal >= 2x the recorded same-config noise floor, and attention-local PMC
  corroboration.
- **Fails closed**: a shape with no evidence entry (or a missing/malformed evidence file) is a gate FAILURE, not a
  pass-by-omission.
- Confirmed by running it: 8B is fully covered and passing; 14B has no evidence entries at all, so the gate reports
  `AUTHORITY_GATE: FAIL` today, correctly.

## Evidence summary (already collected, not re-run here)

- Numerics: `max_abs_err=6.104e-05 PASS` at Hd=64 AND Hd=128 (identical to baseline).
- Real-model 8B token parity: `SDPA=198 FUSED=198 MATCH PASS`.
- Whole-model, three independent same-session paired A/B runs (tok/s):

| pair | pp512 | pp4096 |
|---|---|---|
| 1 | 3607 -> 3680 (+2.02%) | 3003 -> 3054 (+1.70%) |
| 2 | 3424 -> 3481 (+1.66%) | 2880 -> 2921 (+1.42%) |
| 3 | 3404 -> 3459 (+1.62%) | 2858 -> 2911 (+1.85%) |
| **mean** | **+1.77%** | **+1.66%** |

Back-to-back noise on the same config is 0.59%, so the signal is ~3x noise, and it reproduces across a 5% drift in
the absolute baseline (docs/prefill-needle-theories-20260724.md THEORY 3).

- Attention-local, PMC at ctx4096: aggregate busy `800.1M -> 706.3M = 1.133x`, matching the `1.121x` predicted from
  causal-waste arithmetic. Per chunk: kv512 1.830x, kv2048 1.146x, kv4096 1.088x.

## What is still MISSING before a human should flip the default

1. **14B is UNTESTED and untestable here** -- the single biggest gap. `prefill_flash_e2e_parity.py`'s 14B arm fails on
   in-process VRAM (8B still resident in the same process), and there is a separate known packed-WMMA canary
   HW-fault on this box. `prefill_flash_attention_generated`'s own `shape_guards` admit BOTH the 8B (`Hq=32`) and 14B
   (`Hq=40`) grids -- the flag is currently only half-proven against this row's admitted surface. This needs its own
   process (not sharing VRAM with an 8B run) and needs the packed-WMMA canary fault resolved or worked around
   (`TINYGRAD_PREFILL_PACKED_WMMA=0`) before it can be measured safely.
2. No pinned/attributed whole-model run through the same harness that stamps `mode:authority` with
   `comparator_id`/`candidate_id`/`primitive_class`/`threshold`/`ledger`/`quality_gate` (the `prefill_whole_synced.py`
   `authority_completeness_gate` field set) exists yet for this flag specifically; the evidence above was collected
   ad hoc during THEORY 3 investigation, not through that harness. Worth doing before flip, even for 8B, so the
   promotion has a first-class artifact rather than only a doc table.
3. No route-census / candidate-set identity check exists for this flag (the `prefill_wmma_lds_dbuf_generated` row's
   pattern) -- not obviously required since this isn't a candidate-set route, but worth a second opinion before flip.

## What I would push back on if asked to flip now

Nothing in the task framing was wrong. The one nuance worth being explicit about: this promotion is gated on a flag
inside an existing row, not a new row, and the existing `pure_search_guard.py` HOT_FAMILIES model does not track this
flag's rollback state at all (it hardcodes `rolled_back_to_oracle=False` unconditionally for `prefill_attention`, and
the flag does not have its own env-driven rollback distinct from the base route). That means flipping this default
does **not** need a guard change to stay pure -- but it also means the guard gives zero signal about which arm
(flag on/off) is actually active. If the default is ever flipped, whoever does it should also decide whether the
guard's report should start naming the flag's state explicitly (it currently cannot distinguish
`PREFILL_CAUSAL_TILE_SKIP=0` from `=1`, both resolve to the same `effective_route`/`provenance`).
