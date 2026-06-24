# Decode MMVQ large project P7d one-role timing scope - 2026-06-19

Purpose: determine whether the P7c model-integrated imported Q4_K route produces a real local timing win after graph
and side-buffer integration overhead.

## Target

Measure exactly one role:

- `blk.0.attn_output`;
- real Qwen3-8B Q4_K_M weights;
- real pre-`attn_output` activation captured from `TransformerBlock._attention`;
- baseline: current tinygrad `attn_output(out_in)`;
- candidate: `route_imported_q4_mmvq(attn_output, out_in, q8_side, out_side)`;
- both timed as TinyJit graph calls, interleaved in one process.

## Method

1. Load the model with defaults.
2. Temporarily wrap `block.attn_output` to capture the true `out_in` passed by `_attention`.
3. Restore the original linear.
4. Preinstall the imported Q4 programs outside the timed functions.
5. Build two `TinyJit` functions: baseline and imported route.
6. Warm both functions, then run interleaved A/B timing with device synchronization.
7. Run one `_attention` call with `DECODE_MMVQ_IMPORT_Q4=True` to confirm the model branch still routes.

## Gates

P7d passes only if:

- both TinyJit functions run;
- candidate output is stable across graph replays;
- model branch route still allocates the P7c side buffers;
- median candidate wall time is at least `1.10x` faster than baseline for the role.

If timing does not pass, stop before FFN expansion and diagnose the integration overhead.
