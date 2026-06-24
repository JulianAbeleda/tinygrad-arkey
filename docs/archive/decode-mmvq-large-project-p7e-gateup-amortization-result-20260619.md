# Decode MMVQ large project P7e gate/up amortization result - 2026-06-19

Purpose: execute `decode-mmvq-large-project-p7e-gateup-amortization-scope-20260619.md`.

Artifacts:

- `extra/qk_decode_mmvq_p7e_gateup_amortization.py`
- `bench/qk-decode-mmvq-large-project/p7e_gateup_amortization.json`

## Result

Verdict: **NO_GATEUP_TIMING_WIN**.

The imported Q4_K route is correct and replay-stable for the `ffn_gate/up` shared-input pair, but it is slower than the
current tinygrad path even when q8 production is amortized across both consumers.

| metric | value |
|---|---:|
| rows per role | `12288` |
| baseline median, gate+up | `0.1685 ms` |
| imported median, q8+gate+up | `0.2264 ms` |
| speedup | `0.744x` |
| replay max diff | `0.0` |
| gate q8-path max_abs vs baseline | `0.00401` |
| up q8-path max_abs vs baseline | `0.00674` |

Gates:

| gate | result |
|---|---|
| baseline runs | PASS |
| imported runs | PASS |
| imported replay stable | PASS |
| speedup `>=1.10x` | FAIL |
| default unchanged | PASS |

## Interpretation

P7e tested the favorable case that P7d intentionally left open:

```text
one 4096-wide activation
one q8 producer
two imported Q4_K consumers
12288 rows each
```

That still loses locally. The imported llama Q4 consumer is fast in isolation, but the graph-level lifecycle cost of q8
production plus separate imported launches does not beat tinygrad's current in-model Q4 linears for the pair.

Combined with P7d:

- `attn_output`: imported route loses (`0.763x`);
- `ffn_gate/up`: imported route loses (`0.744x`).

So the imported Q4 decode route is now closed as a local timing win. The large MMVQ project still taught us the exact
llama contract, raw-kernarg rebind, and graph-safe artifact launch mechanics, but it should not proceed to model-wide
routing without a different primitive, such as a fused producer+multi-consumer kernel or native renderer transfer that
removes the separate-launch lifecycle cost.
