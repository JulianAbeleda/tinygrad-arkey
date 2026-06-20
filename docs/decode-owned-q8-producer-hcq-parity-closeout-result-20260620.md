# Decode Owned q8 Producer HCQ Parity Closeout - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_HCQ_PARITY_ROW`

The owned COMGR producer is accepted as the HCQ-parity producer/cache row.

| row | runtime | producer us |
|---|---|---:|
| owned COMGR producer | tinygrad HCQ / COMGR | `15.70` |
| HCQ hipcc/LLD artifact producer | tinygrad HCQ / AMDProgram | `21.70` |
| HIP-runtime producer oracle | HIP runtime events | `7.501` |

Decision:

- use the owned COMGR producer for the owned q8 lifecycle successor HCQ-parity row;
- keep the HIP-runtime `7.501us` producer as a separate upper-oracle target;
- do not block HCQ artifact parity on the HIP-runtime target.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_hcq_parity_closeout.py
```
