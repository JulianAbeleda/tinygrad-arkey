# Decode q8 Session-Band Authority Scope

Date: 2026-06-20

## Goal

Find whether a bounded warm-state policy can force the q8 lifecycle fast band, or whether q8 promotion remains blocked by
uncontrolled session state.

## Scope

Run a fresh-process matrix over warm-state protocols:

- `cold`
- `producer_warm`
- `consumer_warm`
- `lifecycle_warm`
- `producer_then_consumer_warm`

Each child records boundary `rocm-smi --showgpuclocks` samples, then measures prebuilt-q8 consumer, producer-only, and
producer->consumer lifecycle rows.

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_session_band_authority.py
```

Default run is `3` fresh sessions per protocol to keep the authority search bounded. A protocol that clears the
`115.24us` lifecycle target should be rerun at higher session count before any promotion discussion.

## Boundary

This pass changes no decode defaults. `rocm-smi` boundary samples are coarse telemetry, not counter-grade PMC authority.
If no warm policy clears target, the q8 route remains blocked on session-band authority.
