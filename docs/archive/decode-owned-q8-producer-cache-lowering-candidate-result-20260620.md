# Decode Owned q8 Producer/Cache Lowering Candidate Result - 2026-06-20

Verdict: `BLOCKED_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_TOO_SLOW`

This probe measures the first owned producer/cache candidate:

```text
owned_hcq_comgr_q8_rmsnorm_side
source: extra.q8_ffn_hcq_artifact.NORM_SOURCE
runtime: tinygrad AMD HCQ / COMGR
```

It checks:

- fp RMSNorm correctness;
- q8 byte size and dequant error;
- no in-process HIP runtime;
- producer median latency against the `<=7.501us` artifact target.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_cache_lowering_candidate.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_producer_cache_lowering_candidate_result.json
```

## Result

| metric | value |
|---|---:|
| producer median | `15.70us` |
| artifact target | `7.501us` |
| fp max abs | `4.77e-7` |
| q8 dequant max abs | `0.01165` |
| q8 bytes | `4608` |

The candidate is semantically correct and uses no in-process HIP runtime, but it is about `2.09x` slower than the
artifact producer target.
