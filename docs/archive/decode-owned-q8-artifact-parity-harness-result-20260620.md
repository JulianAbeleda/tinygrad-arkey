# Decode Owned q8 Artifact Parity Harness Result - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_ARTIFACT_PARITY_HARNESS_READY`

This creates the comparison harness for the owned q8 lifecycle successor. It does not implement the successor and does
not change defaults.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_artifact_parity_harness.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_artifact_parity_harness_result.json
```

## Parity Matrix

| route | status | policy |
|---|---|---|
| baseline default decode | authority baseline | keep default |
| q8 FFN handwritten artifact | measured hardened opt-in | default off, `Q8_FFN_HANDWRITTEN=1` |
| owned q8 lifecycle successor | target row, unimplemented | default off until measured |

## Successor Required Evidence

The owned successor must provide:

- owned producer/cache implementation row;
- owned `ffn_gate`/`ffn_up` packed q4/q8 consumer row;
- lifecycle `<= 115.24us`;
- W==D speedup at least matching the q8 artifact at ctx `512/1024/4096`;
- multi-window dNLL `<= 0.01`;
- fallback and coverage policy equal or stricter than the q8 artifact.

## Decision

The artifact remains a measured hardened opt-in. The successor now has a parity harness and target row, but no owned
implementation evidence yet.

Next: build either an owned producer/cache candidate or an owned consumer candidate. Search stays blocked until one is
lowerable and measurable.
