# DBUF Epoch Lifecycle Checker Scope

Date: 2026-07-09.

## Goal

Create a separate tool for reasoning about DBUF epoch ownership without depending on the S10 fast path.

The tool is:

```text
extra/qk/prefill/dbuf_epoch_lifecycle_checker.py
```

It validates an event trace:

```text
produce(role, epoch, slot, window)
barrier()
consume(role, epoch, slot, window)
```

## Why Separate

The S10 blocker is not instruction encoding. It is the epoch lifecycle:

```text
prologue produce epoch0
body consume epoch i / produce epoch i+1
tail consume final epoch
```

The checker is intentionally independent from tinygrad lowering so it can later be decoupled into a standalone lifecycle
analysis tool. S10 can export traces into it, but S10 should not be required to run it.

## Checks

| Rule | Meaning |
|---|---|
| one producer per consumer | every `consume` has exactly one matching prior `produce` |
| epoch/slot agreement | producer and consumer agree on role, epoch, slot, and window |
| barrier separation | at least one barrier lies between producer and consumer |
| no early overwrite | a slot cannot be reused before the previous epoch in that slot is consumed |
| no duplicate consume | a producer cannot feed multiple consumers unless the trace explicitly models fanout later |

## Current Status

Implemented as a standalone metadata/event checker. It does not alter S9/S10 execution.

The first exporter is also implemented:

```text
S10 hybrid role trace -> hand_coded_epoch_primitive -> DBUF checker events
```

Command:

```bash
PYTHONPATH=. python3 extra/qk/prefill/dbuf_epoch_lifecycle_checker.py \
  --s10-role-trace bench/prefill-s10-lds2-ownership/hybrid-s9-s10-role-trace.json \
  --k-tiles 4 --json
```

This validates the current `ffn_gate_up` `DBUFEpochPrimitive` contract without changing the fast path.

Unit gate:

```bash
PYTHONPATH=. pytest -q test/unit/test_dbuf_epoch_lifecycle_checker.py
```

The built-in canonical DBUF plan passes; intentionally broken epoch, slot-overwrite, missing-barrier, and duplicate
consume traces fail.

## Decoupling Path

1. Keep the checker pure metadata/event based.
2. Add exporters from S10 role/spec traces into checker events. Done for the hybrid role trace.
3. Add exporters from generated DBUF attempts into checker events.
4. Move the checker out of `extra/qk/prefill` once it has a stable schema and multiple producers.
