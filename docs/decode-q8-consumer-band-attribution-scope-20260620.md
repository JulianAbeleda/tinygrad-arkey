# Decode q8 Consumer Band Attribution Scope

Date: 2026-06-20

## Goal

Close the next bounded decode question: whether the fused hipcc/LLD `q8_mmvq_gateup` consumer's `~101us` repeated band is
a stable consumer issue/resource problem or a session/order artifact.

## Scope

Run one attribution pass that:

1. captures static metadata for the `q8_mmvq_gateup` artifact;
2. measures first-N, repeated same-dispatch, dummy/after-dummy, and mixed-order consumer rows across fresh sessions;
3. classifies the outcome as stable slow, bimodal session state, fast, or mixed protocol.

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_consumer_band_attribution.py --sessions 5
```

## Boundary

This pass changes no decode defaults and promotes no route. Existing PMC artifacts do not carry decoded counter values,
so counter-grade PMC/SQTT rates are out of scope unless the profiling stack is extended. Static issue analysis is allowed
as the fallback evidence for this pass.

## Gate

Correctness must pass before timing is interpreted. If all protocols reconcile to the slow band, the consumer band is
real enough to justify consumer issue/resource work. If the band is bimodal or protocol-dependent, timing policy/session
state must be resolved before rewriting the consumer.
