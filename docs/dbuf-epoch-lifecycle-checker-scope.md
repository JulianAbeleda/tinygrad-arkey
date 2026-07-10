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
wait(kind, count, phase)
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
| optional layout proof | when `layout_key` is exported, producer and consumer must agree on A/B WMMA operand layout |
| optional wait proof | with `--require-p5`, VM/LGKM wait events must bracket LDS production, barriers, and consume/WMMA phases |

## Exhaustive S10 Target

The current checker is phase 1 only. To become the central proof object for pure S10, the checker must carry enough
information to prove that a generated DBUF lifecycle can replace the hand-coded `DBUFEpochPrimitive`.

Target event schema:

```python
produce(
  role="A" | "B",
  epoch=int,
  slot=int,
  window="logical tile/window id",
  lds_window={"base": int, "bytes": int, "stride": int},
  global_window={"tensor": str, "m": [lo, hi], "n": [lo, hi], "k": [lo, hi]},
  layout="a_row_major" | "b_transposed",
  value_key={"tensor": str, "tile": tuple, "epoch": int},
  wait_source="vmem",
)

sync(kind="barrier")
wait(kind="vmcnt" | "lgkmcnt", count=int)

consume(
  role="A" | "B",
  epoch=int,
  slot=int,
  window="same logical tile/window id",
  lds_window={"base": int, "bytes": int, "stride": int},
  fragment={"shape": "wmma_f16_16x16x16", "operand": "A" | "B"},
  lane_map="rdna3_a_row" | "rdna3_b_col",
  consumer="wmma_i",
)
```

Required proof layers:

| Layer | Proof | Why S10 needs it | Status |
|---|---|---|---|
| P1 epoch lifecycle | producer/consumer/barrier/overwrite correctness | proves the DBUF ring is logically safe | done |
| P2 byte-window proof | producer and consumer agree on exact LDS byte interval | prevents same-slot wrong-window bugs | checker/exporter ready for S10 LDS spec |
| P3 value-key proof | global tile loaded by producer equals tile consumed by WMMA | prevents wrong epoch/tensor/tile values | not started |
| P4 layout proof | A/B row/BT layout matches the WMMA operand contract | prevents correct bytes in wrong lane order | checker/exporter ready for S10 LDS spec static layout |
| P5 wait/sync proof | VMEM waits, LGKM waits, and barriers are present in the right phase | prevents memory visibility hazards | checker schema and strict mode implemented; exporters pending |
| P6 lifetime/pressure proof | address/fragment live ranges are bounded by the lifecycle | prevents the generated route from recreating VGPR pressure failures | not started |
| P7 lowered-stream proof | generated graph/stream exports actual stores/loads/waits/WMMA into this schema | proves S10 generation, not only the hand-coded primitive metadata | not started |

S10 is ready to reopen generated DBUF replacement only when P1-P7 pass on:

1. the hand-coded S9/S10 hybrid trace,
2. the generated candidate trace,
3. and a diff proving both traces have equivalent role/epoch/window/value coverage.

## Exporters Needed

| Exporter | Source | Output | Status |
|---|---|---|---|
| E1 hybrid primitive exporter | `hybrid-s9-s10-role-trace.json` | P1 events from `DBUFEpochPrimitive` | done |
| E2 S10 LDS spec exporter | `WMMALDSSpec` / slot identity proof | P2 LDS windows | done |
| E3 hand lifecycle exporter | `kernel_lifecycle_trace.py` / `wmma.py` lifecycle template | P1/P2/P5 hand oracle events from the LDS2 template | done for template oracle |
| E4 generated postrange exporter | pre-lowering `Ops.STAGE` / owner metadata | P1-P4 generated candidate events | pending |
| E5 lowered stream exporter | generated AMD ISA or UOp stream | P5-P7 actual instruction events | pending |

## Done Definition For "All S10 Info"

The checker is complete enough for S10 when it can answer:

```text
For every WMMA operand consumer in generated ffn_gate_up:
  which global tile produced it?
  which LDS byte window carried it?
  which DBUF slot and epoch owned it?
  which barrier/wait made it visible?
  which fragment/lane map feeds WMMA?
  was the slot overwritten only after the last consumer?
  were live ranges bounded enough to avoid recreating the known pressure wall?
```

Current answer: epoch/slot/barrier is proven, the checker validates LDS byte-window equality, and S10 LDS spec exports
static A/B layout keys. The checker also has an opt-in P5 strict mode for VM/LGKM wait events, but no production exporter
feeds wait events yet.

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

The second exporter is implemented:

```text
WMMALDSSpec / wmma_lds_slot_identity_proof -> P2 LDS byte-window events + P4 static layout keys
```

Tool:

```text
extra/qk/prefill/dbuf_s10_lds_spec_exporter.py
```

Unit gate:

```bash
PYTHONPATH=. pytest -q test/unit/test_dbuf_epoch_lifecycle_checker.py test/unit/test_dbuf_s10_lds_spec_exporter.py
```

The built-in canonical DBUF plan passes; intentionally broken epoch, slot-overwrite, missing-barrier, and duplicate
consume traces fail. P5 strict mode is available through:

```bash
PYTHONPATH=. python3 extra/qk/prefill/dbuf_epoch_lifecycle_checker.py --canonical --require-p5 --json
```

Legacy traces without explicit wait events still pass P1/P2/P4 checks, but fail when P5 is required.

The third exporter is implemented for the hand LDS2 template oracle:

```text
default_lds2_lifecycle_template + default_lds2_memory_layout + default_lds2_wait_policy -> P1/P2/P5 checker events
```

Command:

```bash
PYTHONPATH=. python3 extra/qk/prefill/dbuf_epoch_lifecycle_checker.py \
  --hand-lds2 --k-tiles 4 --require-p5 --json
```

This proves the hand oracle's logical DBUF ring, exact LDS slot byte windows, and VM/LGKM wait phases. It is not the P7
lowered-stream proof; final instruction rows still need a fail-closed exporter once role/epoch/slot metadata survives
lowering.

## Decoupling Path

1. Keep the checker pure metadata/event based.
2. Add exporters from S10 role/spec traces into checker events. Done for the hybrid role trace and S10 LDS spec byte windows.
3. Add exporters from generated DBUF attempts into checker events.
4. Move the checker out of `extra/qk/prefill` once it has a stable schema and multiple producers.
