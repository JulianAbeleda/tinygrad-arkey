# Decode q8 Model Route Timing Audit Scope

Date: 2026-06-20

## Goal

Audit the actual in-model `Q8_FFN_HANDWRITTEN=1` graph route under `auto` and `manual_peak`, rather than the isolated
`wait=True` producer/consumer micro-harness.

## Command

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_model_route_timing_audit.py
```

## Method

Reuse the accepted W/D decode method:

- `W`: real decode replay with one `.item()` sync per token;
- `D`: dispatch-only graph replay, no per-token `.item()`, one final sync;
- compare baseline and q8 under each clock lane.

## Gate

Pass if the q8 model route under `manual_peak` has positive speedup and host-sync residual remains below `10%`.

If this passes, primitive fusion is not justified by host-wait evidence.
