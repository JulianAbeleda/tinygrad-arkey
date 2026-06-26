# Decode Attention Online-State+PV Tile P9 Scalar Numeric Result

## Verdict

`ONLINE_STATE_PV_P9_FAIL__NAN`

P9 compared the scalar online-state tile path against a NumPy reference across required deterministic cases.

Artifact:

- `bench/qk-decode-attention-online-state-pv-p9-scalar-numeric/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p9_scalar_numeric.py
```

## Cases

| `Tc` | `L` | `Sval` | Verdict |
|---:|---:|---:|---|
| 128 | 64 | 2 | `FAIL__NAN` |
| 130 | 64 | 3 | `FAIL__NAN` |
| 32 | 64 | 1 | `FAIL__NAN` |
| 256 | 64 | 4 | `FAIL__NAN` |

## Important Localization

Score and final output are numerically correct, but direct state-column reads contain NaNs.

| Case | score err | output err | state issue |
|---:|---:|---:|---|
| 128 | `5.2e-08` | `2.2e-08` | `m/l/PV` direct reads contain NaN |
| 130 | `7.5e-08` | `2.6e-08` | `m/l/PV` direct reads contain NaN |
| 32 | `5.2e-08` | `3.0e-08` | `m/PV` direct reads contain NaN, `l` huge |
| 256 | `6.7e-08` | `1.1e-08` | `m/l/PV` direct reads contain NaN |

## Interpretation

The scalar online-state route computes the final attention output correctly against NumPy for all required cases.

The P9 failure is state observability, not final scalar correctness:

- generated score is correct;
- final `flash_state_combine` output is correct;
- direct reads of the intermediate tile state buffer show NaNs/unstable values.

That means the scalar path is good enough as a final-output reference, but not good enough as a direct state-column oracle without a dedicated state-dump kernel or stricter store/read contract.

This changes the P8 interpretation: scalar P5 final output was not the blocker. The P7 token mismatch remains an x-lane merge/math problem, but debugging it should compare final output and/or use a dedicated debug state dump rather than trusting raw intermediate state columns from the normal route buffer.

## Decision

Do not fix scalar recurrence based on P9; scalar final output is correct.

Next step:

```text
P10: isolated x-lane final-output numeric gate against NumPy and scalar final output
```

P10 should avoid direct raw state-column assertions unless it adds a dedicated debug state-dump path. It should compare:

- scalar final output vs NumPy;
- x-lane final output vs NumPy;
- x-lane final output vs scalar final output;
- optional debug-only state columns if explicitly emitted with a stable store contract.
