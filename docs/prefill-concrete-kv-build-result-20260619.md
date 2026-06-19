# BUILD RESULT — concrete-KV prefill: 1.24x, byte-identical, validated (P0-P2 done; Option-B refuted)

Executed the build scope (`prefill-concrete-kv-build-scope-20260619.md`). Gated `PREFILL_CONCRETE_KV` (default off);
decode + symbolic prefill untouched.

## What was built
- `PREFILL_CONCRETE_KV`: generate passes a CONCRETE int start_pos per prefill chunk (KV=start_pos+T concrete ->
  SDPA's attention reduce tiles / TC fires; symbolic KV blocked it). `__call__` uses a per-start_pos concrete jit
  dict (`prefill_v2_jits`); symbolic path keeps the single `prefill_v2_jit`.
- `PREFILL_TC_ATTN`: gated explicit Option-B TC attention on the concrete branch (Q@KT TC + fp16 scores + softmax
  + P@V TC, GQA broadcast).

## Results
- **P0 — single-chunk (prompt<=512, start_pos=0): VALIDATED.** Greedy generation byte-identical (symbolic vs
  concrete); prefill 1.24x (clock-controlled A/B, same forward).
- **P1 — multi-chunk (prompt>512): correctness VALIDATED.** 1024-tok prompt (2 chunks @ start_pos 0,512, two
  concrete jits) -> byte-identical generation. Per-chunk replay = 1.24x.
- **P2 — Option-B TC attention: REFUTED (no stack).** concrete+SDPA vs concrete+Option-B = **1.001x**, rel_err 0,
  byte-identical. The 1.24x already comes from SDPA going TC-able on concrete shapes; explicit Option-B adds nothing
  (its 2.56x-standalone was vs NON-concrete SDPA). **Option-B NOT wired (P3 skipped).**

## The simplified, validated prefill win
**concrete-KV alone -> 1.24x prefill, byte-identical.** SDPA on concrete KV captures the full attention win; no
explicit TC attention, no Tensile, no matmul work needed (all exhausted/refuted earlier).

## Deployment caveat (multi-chunk compile cost)
Concrete start_pos => one concrete jit per distinct start_pos (K = ceil(prompt/512)). The 1.24x is per-chunk
REPLAY; each jit is a one-time compile (~seconds incl. warmstart). So:
- single-chunk (prompt<=512) OR server (cached jits, repeated requests): clean 1.24x win.
- cold one-shot multi-chunk: K compiles may exceed the ~80ms/chunk saving -> only pays off when cached.
Default stays OFF (PREFILL_CONCRETE_KV=0); flip per deployment (default-on safe for prompt<=512 / server).

## Files
`tinygrad/llm/model.py` (PREFILL_CONCRETE_KV, PREFILL_TC_ATTN, prefill_v2_jits, _attention TC branch). Scope:
`prefill-concrete-kv-build-scope-20260619.md`. Commits 026c4c9ff (P0), 4a145e7b8 (P2).
