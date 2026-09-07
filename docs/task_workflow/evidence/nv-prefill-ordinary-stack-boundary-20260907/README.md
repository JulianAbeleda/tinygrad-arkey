# NV ordinary generated pp512 boundary checkpoint (2026-09-07)

The exact NV/Qwen3-8B 512-token generated stack is now the ordinary selector;
`NV_COMPILER_FULL_PACKED_PP512=0` rolls the complete stack back, while the
existing component rollback variables remain available.  Explicit llama and
compiler diagnostic overrides retain precedence.  The packed Q4 publication
subfeature remains explicit opt-in because fresh composed deep replay was not
deterministic.

The packed-publication ownership audit found one writer for every aligned
word: lane `l` writes the disjoint base
`y*2560 + z*640 + (l>>2)*80 + (l&3)*16`, with word offsets
`0,4,8,12,5120,5124,5128,5132`.  CUDA source places a full CTA
`__syncthreads()` immediately after these stores and SASS places `BAR.SYNC`
after the paired `STS.128` instructions.  Despite this, an enabled fresh
deep-five run corrupted cycle 2, while explicit rollback passed two fresh
deep-five runs.  With publication disabled by default, one fresh run passed
all five cycles and a second differed only in its first instrumented cycle;
cycles 1-4 and every structural invariant were exact.  This points at the
manual GKQO capture/baseline instrumentation rather than a demonstrated word
writer collision.

The maintained observer entrypoint exposed and crossed a separate automatic-K
typed boundary.  An opaque Q8 producer returns `AFTER(buffer, call)`; the
tile-major carrier now views the original uint buffer while the `AFTER`
dependency wraps that logical carrier.  This keeps the `(512,4096)` shape and
the producer ordering edge without a cast, copy, or materialization.  The
focused carrier and selector suite passes 21 tests.  The maintained K1 run now
executes through scheduling, but its legacy dense candidate-set gate still
reports three missing registry entries (`attn_qo`, `ffn_down`, and
`ffn_gate_up`) because those roles are served by model-owned generated
captures rather than the old dense registry.  That census integration remains
the next correctness-only milestone; the executed 16904.7 ms smoke is not a
timing authority.

Artifacts from the bounded runs were written to:

- `/tmp/ordinary_stack_clean_deep5_a.json` (PASS, exact 252/234/162)
- `/tmp/ordinary_stack_clean_deep5_b.json` (cycle 0 differs; cycles 1-4 exact)
- `/tmp/prefill_whole_ordinary_k1_fixed3.log` (executed, stale route-census gate)
