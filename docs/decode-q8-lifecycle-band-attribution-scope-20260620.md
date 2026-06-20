# Decode q8 Lifecycle Band Attribution Scope

Date: 2026-06-20

## Goal

Resolve the contradiction between:

- `decode_q8_nt1024_reconciliation`: full lifecycle repeated median `~122us`, with consumer `~101us`;
- `decode_q8_consumer_band_attribution`: steady prebuilt-q8 consumer `~89-90us`.

## Scope

Run a fresh-session lifecycle attribution with the owned `NT=1024` COMGR q8 producer and hipcc/LLD fused gate/up
consumer. Split:

1. prebuilt-q8 consumer repeat;
2. producer->consumer lifecycle first-N and steady rows;
3. producer-only repeat;
4. producer->dummy-consumer->consumer rows.

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_lifecycle_band_attribution.py --sessions 5
```

## Decision Gate

- If prebuilt consumer is fast but consumer-after-producer is slow, the blocker is producer->consumer adjacency or
  launch/cache/session composition.
- If all steady components are fast but total still misses `115.24us`, this is a target/policy or launch accounting
  decision, not a kernel-body rewrite.
- If steady lifecycle clears `115.24us`, q8 promotion policy can reopen.

No defaults change in this pass.
