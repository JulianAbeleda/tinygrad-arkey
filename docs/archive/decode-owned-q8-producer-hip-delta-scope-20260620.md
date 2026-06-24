# Decode Owned q8 Producer HIP Delta Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_HIP_DELTA_SCOPE_READY`

After the HCQ parity closeout, the remaining producer question is narrower:

```text
owned COMGR HCQ producer: 15.70us
HIP-runtime producer oracle: 7.501us
```

This delta should not block route-level HCQ parity. It is a separate optimization/codegen question.

## Scope

| row | question | status |
|---|---|---|
| HD-1 runtime boundary | is the gap HIP runtime vs HCQ dispatch/measurement? | blocked by process/runtime boundary |
| HD-2 ISA delta | what static codegen/resource difference exists? | do now |
| HD-3 optimized producer | can owned lowering reach `<=7.5us`? | blocked on HD-2 or hand/codegen work |

Next executable probe:

```text
extra/qk_decode_owned_q8_producer_codegen_delta_probe.py
```
