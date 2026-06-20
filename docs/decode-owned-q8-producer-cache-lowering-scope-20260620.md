# Decode Owned q8 Producer/Cache Lowering Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_PRODUCER_CACHE_LOWERING_SCOPE_READY`

This scopes the first owned producer/cache lowering candidate.

Candidate:

```text
owned_hcq_comgr_q8_rmsnorm_side
source: extra.q8_ffn_hcq_artifact.NORM_SOURCE
runtime: tinygrad AMD HCQ via Device.compiler COMGR path
```

This is owned enough for the producer/cache track because it is launched by tinygrad HCQ and compiled through the
tinygrad AMD compiler path. It is not the external hipcc/LLD artifact.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_cache_lowering_scope.py
```

Next executable probe:

```text
extra/qk_decode_owned_q8_producer_cache_lowering_candidate.py
```

The candidate must match reference semantics and measure against the `<=7.501us` producer lifecycle target before it
can be used as route-level evidence.
