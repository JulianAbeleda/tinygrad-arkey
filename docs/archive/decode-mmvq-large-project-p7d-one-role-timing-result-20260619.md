# Decode MMVQ large project P7d one-role timing result - 2026-06-19

Purpose: execute `decode-mmvq-large-project-p7d-one-role-timing-scope-20260619.md`.

Artifacts:

- `extra/qk_decode_mmvq_p7d_one_role_timing.py`
- `bench/qk-decode-mmvq-large-project/p7d_one_role_timing.json`

## Result

Verdict: **NO_LOCAL_TIMING_WIN**.

The imported Q4_K route is runnable, stable, and reachable through the P7c model branch, but it is slower than the
current tinygrad role path for `blk.0.attn_output`.

| metric | value |
|---|---:|
| baseline median | `0.1064 ms` |
| imported median | `0.1396 ms` |
| speedup | `0.763x` |
| replay max diff | `0.0` |
| baseline vs q8 path max_abs | `0.00241` |

Gates:

| gate | result |
|---|---|
| baseline runs | PASS |
| imported runs | PASS |
| imported replay stable | PASS |
| model branch routed | PASS |
| speedup `>=1.10x` | FAIL |
| default unchanged | PASS |

## Method Correction

P7d captures the true pre-`attn_output` activation by temporarily wrapping `block.attn_output` while running
`TransformerBlock._attention`. This avoids the earlier probe pattern that used `_attention(...)` output as if it were
the input to `attn_output`.

The timing authority is same-process interleaved TinyJit wall time with `Device.synchronize()` after each call:

```text
baseline: original_linear(out_in)
candidate: route_imported_q4_mmvq(original_linear, out_in, q8_side, out_side)
```

## Interpretation

For the 4096-row attention output role, importing llama's Q4 consumer plus a separate q8 producer does not pay after
integration overhead. The standalone imported consumer is fast, but the one-role lifecycle has two graph launches and
q8 side-buffer work, while the existing tinygrad path is already cheap for this shape.

This does not prove the larger `ffn_gate/up` lifecycle is impossible; those roles have 12288 rows and share one input.
But P7d fails the agreed one-role timing gate, so expanding blindly would violate the measure-first rule. The next valid
step is a new scoped diagnostic that asks whether `gate+up` q8 amortization changes the result, not a direct route
expansion.
