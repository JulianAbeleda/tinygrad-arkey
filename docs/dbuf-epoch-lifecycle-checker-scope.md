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
| P3 value-key proof | global tile loaded by producer equals tile consumed by WMMA | prevents wrong epoch/tensor/tile values | checker schema ready; exporters pending |
| P4 layout proof | A/B row/BT layout matches the WMMA operand contract | prevents correct bytes in wrong lane order | checker/exporter ready for S10 LDS spec static layout |
| P5 wait/sync proof | VMEM waits, LGKM waits, and barriers are present in the right phase | prevents memory visibility hazards | checker strict mode, side-channel reconciliation, live wait anchors, and byte-window bridge P5 implemented |
| P6 lifetime/pressure proof | address/fragment live ranges are bounded by the lifecycle | prevents the generated route from recreating VGPR pressure failures | advisory summary checker implemented |
| P7 lowered-stream proof | generated graph/stream exports actual stores/loads/waits/WMMA into this schema | proves S10 generation, not only the hand-coded primitive metadata | active packed-LDS trace exports through normalized byte-window producer ownership |
| P8 phase-cluster quality | waits stay clustered around phase boundaries instead of appearing per-fragment/per-WMMA | catches performance-shape regressions after correctness proofs pass | checker implemented; trace exporter emits `p8_phase_cluster_quality` |

S10 is ready to reopen generated DBUF replacement only when P1-P7 pass on:

1. the hand-coded S9/S10 hybrid trace,
2. the generated candidate trace,
3. and a diff proving both traces have equivalent role/epoch/window/value coverage.

P8 is not part of this correctness reopening condition. It is a performance-shape gate layered after P1-P7: the stream may
be logically correct and still fail P8 if the waits are distributed in a way that destroys the intended DBUF schedule.

P8 `phase_cluster_quality` scope:

- Input: lowered-stream phase summaries with counted wait instructions and WMMA instructions.
- Output: advisory pass/fail plus density, burst, and precondition metrics under `p8_phase_cluster_quality`.
- Correctness boundary: P8 must not claim epoch, byte-window, value, layout, sync visibility, lifetime, or lowered-stream
  correctness. Those remain P1-P7.
- Fail signature for the generated S10 shape: `157 waits / 32 WMMA`, `waits_per_wmma=4.9`.
- Pass signature for the hand/hybrid LDS2 shape: `18 waits / 64 WMMA`, `waits_per_wmma=0.28`.
- Intended decision: fail generated shapes that technically satisfy proof obligations but emit fine-grained wait traffic
  instead of clustered phase-boundary waits.
- Current implementation:
  - `extra/qk/prefill/dbuf_epoch_lifecycle_checker.py::check_phase_cluster_quality`
  - `extra/qk/prefill/kernel_lifecycle_trace.py` emits `p8_phase_cluster_quality` for every final-stream report.
  - Real hand LDS2 `ffn_gate_up` trace passes as `hand_lds2_quality`: `18 waits / 64 WMMA`, `max_wmma_burst=8`.
  - Current generated S10 DBUF `2x4` fails as `baseline_like:correctness_may_pass_but_wait_amortization_fails`:
    `157 waits / 32 WMMA`, `max_wmma_burst=1`, `ds_load_b128_per_wmma=4.0`.

## Exporters Needed

| Exporter | Source | Output | Status |
|---|---|---|---|
| E1 hybrid primitive exporter | `hybrid-s9-s10-role-trace.json` | P1 events from `DBUFEpochPrimitive` | done |
| E2 S10 LDS spec exporter | `WMMALDSSpec` / slot identity proof | P2 LDS windows | done |
| E3 hand lifecycle exporter | `kernel_lifecycle_trace.py` / `wmma.py` lifecycle template | P1/P2/P5 hand oracle events from the LDS2 template | done for template oracle |
| E4 generated postrange exporter | pre-lowering `Ops.STAGE` / owner metadata | P1 generated owner events from postrange owner records | done for owner records |
| E5 lowered stream exporter | generated AMD ISA or UOp stream | fail-closed P7 status plus checker export when role/epoch/slot metadata or side-channel anchors exist | fail-closed plus side-channel anchor reconciliation implemented |

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
static A/B layout keys. The checker has a P3 `value_key` field and rejects partial or mismatched value proofs, but no
exporter currently emits semantic global tile identity. The checker also has an opt-in P5 strict mode for VM/LGKM wait
events. P8 now scores whether the final stream is hand-quality enough to promote; it does not change P1-P7 correctness.

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

The fourth exporter is implemented for generated postrange owner records:

```text
prefill_stage_owner_audit.owner_records -> conservative P1 checker events
```

It requires exactly A+B owner records with `nbuf=2` and `lds_buffer_id`, then exports role/epoch/slot ownership windows
like `A:owner990:slot0`. It deliberately does not claim P2 byte windows, P3 value keys, P5 waits, or P7 final-stream
facts.

The P6 pressure checker is implemented as an advisory summary gate:

```text
check_pressure_summary(summary)
```

It fails known unsafe summaries such as VGPR peak overflow, `64 V_OFFSET + 64 V_IADD` live to reduce END, DBUF address
values live to reduce END, address values feeding non-address consumers, and rematerializable values feeding WMMA/data or
control uses. It remains advisory until backed by allocator live intervals or exported UOp def/use traces.

The E5/P7 lowered-stream bridge is implemented in `kernel_lifecycle_trace.py`:

```text
_p7_lowered_stream_export(ops, reaching, side_channel=None)
```

It exports actual lowered `ds_store_b128` / `s_barrier` / `ds_load_b128` rows into checker events only when every store
and load carries explicit logical `role`, `epoch`, and `slot` metadata. If that metadata is absent, it returns
`status=fail_closed` with physical counts and the reason instead of guessing from registers or byte windows. Current real
lowered streams are expected to fail closed unless either complete metadata is present or side-channel rows can be anchored
to actual lowered rows.

The tracer now keeps generated final UOps through row extraction, normalizes complete `dbuf_lifecycle` tags into `dbuf`
rows, and reports existing `wmma_frag_proof` / `wmma_frag_buffer_proof` / `tc_local_stage_store` tags as `dbuf_partial`.
The attempted direct tag-survival route is intentionally not used: AMD regalloc treats tuple tags as register definitions,
so proof metadata cannot simply survive through regalloc in `UOp.tag`. The remaining P7 hook must be a side-channel export
before metadata tags are stripped, or a metadata carrier that regalloc explicitly ignores.

The side-channel route is now formalized through `DBUF_D3A_AUDIT_LOG` rows:

```python
{
  "kind": "dbuf_lifecycle_event",
  "op": "produce" | "consume" | "barrier" | "wait",
  "role": "A" | "B",
  "epoch": ...,
  "slot": ...,
  "window": ...,
  "uop_id": ...,
  "wait_kind": "vm" | "lgkm" | "full",  # wait rows only
  "count": 0,                           # wait rows only
}
```

`kernel_lifecycle_trace.py::_side_channel_lifecycle_events` normalizes those rows into checker events, including optional
P3 `value_key` and P5 wait events. `_reconcile_side_channel_to_rows` then matches side-channel rows to actual lowered
`ds_store_b128`, `ds_load_b128`, `s_barrier`, and `s_waitcnt` rows by stable `uop_id` or final `idx` anchors before running
the checker. If reconciliation passes, `_p7_lowered_stream_export` exports the checked final stream; if any anchor is
missing or maps to the wrong physical op, it remains fail-closed.

Current live coverage:

- final-row extraction exports `uop_id` for UOp-backed rows,
- D3A produce events emit `uop_id`,
- D3A explicit barrier events emit `uop_id`,
- LDS fragment consumes emit live `ds_load_b128` anchors,
- waitcnt insertion emits live `s_waitcnt` side-channel anchors,
- UOp rewrite/lowering emits transitive anchor aliases so side-channel ids can follow final lowered rows,
- synthetic side-channel tests prove P1/P3/P5/P7 reconciliation behavior.

Current generated-trace result:

- Active generated packed-LDS shapes `2x2`, `4x2`, and `2x4` now export through
  `proof_source=normalized_lds_byte_window_store_cover`.
- `2x2`: `ds_store_b128=32`, `ds_load_b128=32`, `s_barrier=2`, checked event count `94`, `check.ok=true`.
- `4x2` and `2x4`: `ds_store_b128=48`, `ds_load_b128=48`, `s_barrier=2`, checked event count `142`,
  `check.ok=true`.
- Direct side-channel producer rows are still absent for this route. The producer proof comes from normalized physical
  `ds_store_b128` byte-window coverage of each logical 32-byte consume.
- Strict bridge P5 passes for all three active shapes. The bridge validates VM waits before LDS produces, LGKM drains before
  barriers, and targeted LGKM waits before the WMMA that consumes each loaded fragment.
- `p7_hand_oracle_diff` now passes for `2x2`, `4x2`, and `2x4` with
  `equivalence=contract_level_not_byte_identical`. The generated final-stream proof has A/B role coverage, balanced
  produce/consume counts per role, LDS-window-backed consumes, VM/LGKM wait coverage, and strict bridge P5; the hand oracle
  remains the logical lifecycle template, not a byte-identical stream target.

Remaining MVP work:

1. Keep `4x4` parked under the hardware/register-budget decision; do not reopen it as part of this P7 MVP.

## Decoupling Path

1. Keep the checker pure metadata/event based.
2. Add exporters from S10 role/spec traces into checker events. Done for the hybrid role trace and S10 LDS spec byte windows.
3. Add exporters from generated DBUF attempts into checker events.
4. Move the checker out of `extra/qk/prefill` once it has a stable schema and multiple producers.
