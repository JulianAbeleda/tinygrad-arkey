# P1/P2 BLOCKED — non-authoritative driver (do not trust these tok/s)

The P0-turn P1/P2 tools use a STANDALONE eager driver (load_model_and_tokenizer + per-chunk m.forward,
_prefill_v2=True, no JIT graph). Measured current_default = ~217 tok/s vs the AUTHORITY ~3597 (P0 / aggressive-proof) —
~16x too slow, and route attribution came back EMPTY ([]) (Context(PROFILE=1) captured no kernels in this eager path;
the decode attribution tools also `import tinygrad.runtime.ops_amd`). Arm deltas are masked by Python/launch overhead
(current 217 ~= eightwave_off 219 ~= pipe_tm2_tn2 218), so the harness cannot validate arms either.

VERDICT: PREFILL_P1_BLOCKED_NOISY_OR_STALE (non-authoritative). P2 NOT RUN (same flawed driver -> would be garbage).
FIX (next turn): rebuild P1/P2 on the AUTHORITATIVE prefill harness that produced
bench/qk-prefill-aggressive-target-proof-20260624/{whole_prefill_baseline,authority,whole_prefill_chunk_series}.json
(cfg/route/chunk_series via the real graph-gemm JIT path), not a standalone eager m.forward loop.
P0's analytical ceiling (PREFILL_P0_PASS_CEILING_PINNED) is unaffected and stands.
