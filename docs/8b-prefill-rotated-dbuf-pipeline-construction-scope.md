# 8B Prefill Rotated DBUF Pipeline Construction Scope

## Big Picture

The generated K-major + DBUF path is correct but too dense. It emits extra global/LDS staging work per WMMA, so useful
WMMA math is surrounded by too much bookkeeping.

The failed late-suppression probes proved the performance lever is real but the layer is wrong:

| probe | structural movement | correctness |
| --- | --- | --- |
| StageOwner B phase suppression | `global_load_b128` and `ds_store_b128` drop by 4 | WRONG `rr=1.4e+00` |
| Producer map after suppression | every load still has a reaching store | wrong epoch/value reaches 4 loads |

So the primitive fix is not "delete duplicate stores after lowering." The primitive fix is:

```text
construct the DBUF slot/epoch lifecycle before lowering, so each consumer load has exactly one producer owner.
```

## Current Proof

`extra/qk/prefill/prefill_stage_owner_audit.py` now proves the key boundary:

| boundary | stage owners | A/B roles | nbuf | reduce range | verdict |
| --- | ---: | --- | ---: | ---: | --- |
| `postrange` | 2 | A+B | 2 | yes | ownership-ready |
| `full` | 0 | none | none | no | ownership lost |

Command:

```bash
PYTHONPATH=. python3 extra/qk/prefill/prefill_stage_owner_audit.py \
  --shape 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --boundary postrange --json
```

Key result:

```json
{
  "stage_tagged_count": 2,
  "stage_roles": ["A", "B"],
  "stage_count_by_role": {"A": 1, "B": 1},
  "stage_nbufs": [2],
  "stage_has_reduce_range_count": 2,
  "pre_lowering_ownership_ready": true
}
```

This means the data needed to build the correct lifecycle exists before full lowering:

```python
RotatedStageOwner(
  role,              # A or B
  lds_buffer_id,     # local tile buffer
  nbuf,              # 2 for DBUF
  reduce_epoch,      # K epoch carrier
  dbuf_slot,         # reduce_epoch % nbuf
  producer_phase,    # prologue/body/tail
  consumer_phase,    # compute slot
)
```

## Done Definition

100% for this scope means:

| Layer | Done when |
| --- | --- |
| L0 ownership audit | `postrange` reports `pre_lowering_ownership_ready=true`; `full` reports ownership loss. |
| L1 owner object | postrange can enumerate A/B `RotatedStageOwner` records for slot 0/1 without final addr-family guessing. |
| L2 lifecycle construction | prologue emits only warmup owner(s); body emits future owner(s); tail consumes remaining owner(s). |
| L3 structural trace | generated `2x2` has fewer global/store ops per WMMA without late suppression flags. |
| L4 correctness | generated `2x2`, `512x5120x5120`, `loc=2`, `unr=2` returns `status=ok`. |
| L5 performance | generated route beats the current correct K-major/D3 band and moves toward hand LDS2 density. |
| L6 promotion | whole-prefill route can enable the primitive without claiming a raw hand kernel. |

## Implementation Phases

### P0. Freeze Evidence

Keep these as reference facts:

| path | status |
| --- | --- |
| no suppression | correct, about `9.39 TFLOPS`, `global/store=2.75 per WMMA` |
| B StageOwner suppression | smaller stream, wrong output |
| producer map | suppression changes 4 producer epochs |

No further work should use `PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS=1` as a fix.

### P1. Ownership Object Probe

Extend `prefill_stage_owner_audit.py` to emit owner records:

```python
OwnerKey = (
  role,
  lds_buffer_id,
  nbuf,
  reduce_range_id,
  dbuf_slot_expr,
  tile_count,
  tile_elems,
)
```

Pass:

```text
one A owner and one B owner exist at postrange
both have nbuf=2
both have reduce_range
no owner exists only at full-lowering time
```

Status: complete as audit-only.

Observed owner records:

| role | lds_buffer_id | nbuf | reduce_epoch | dbuf_slot_expr | global range |
| --- | ---: | ---: | --- | --- | --- |
| A | 990 | 2 | `(0, AxisType.REDUCE)` | `((0, AxisType.REDUCE)) % 2` | `(1, AxisType.GLOBAL)` |
| B | 991 | 2 | `(0, AxisType.REDUCE)` | `((0, AxisType.REDUCE)) % 2` | `(2, AxisType.GLOBAL)` |

### P2. Non-Destructive Lifecycle Planner

Add an audit-only planner:

```text
PREFILL_DBUF_ROTATED_PIPELINE_AUDIT=1
```

It should output the intended hand-style schedule:

```python
prologue:
  produce slot0 epoch0
  barrier

body:
  consume slot0 epoch0
  produce slot1 epoch1
  barrier
  consume slot1 epoch1
  produce slot0 epoch2
  barrier

tail:
  consume final slot
```

Pass:

```text
planner predicts exactly one producer for each consumer slot/epoch
planner never decides from final LDS address registers
```

Status: complete as audit-only inside `prefill_stage_owner_audit.py`.

Observed planner result:

```json
{
  "ok": true,
  "source": "audit_only_hand_lds2_style_rotation",
  "producer_count": 6,
  "consumer_count": 6,
  "late_suppression_allowed": false
}
```

### P3. Construct One Role, Audit Only

Start with B because B is where late suppression moved the stream and corrupted values.

Rule:

```text
B owner is constructed once per slot/epoch before lowering.
All B consumers reference that owner.
No original B store is deleted after lowering.
```

Pass:

```text
final stream unchanged or only metadata changes
producer map remains 32/32 covered
GPU correctness remains ok
```

### P4. Destructive One-Role Rewrite

Only after P3:

```text
emit B producers from the planner
do not emit legacy duplicate B producers for the same owner
```

Pass:

```text
status=ok
B producer assignments do not switch to wrong epochs
global/store per WMMA decreases
```

### P5. Add A

Repeat P3/P4 for A after B is correct.

Pass:

```text
A+B route correct
global/store per WMMA moves below 2.75
no late suppression flags enabled
```

### P6. Scheduler/Waitcnt Tuning

Only after construction is correct:

```text
target waitcnt and scheduler placement for the new lifecycle
```

Pass:

```text
correctness unchanged
TFLOPS improves beyond the current correct band
```

## Stop Conditions

Stop and do not tune performance if:

| condition | meaning |
| --- | --- |
| owner records do not exist at postrange | wrong layer; need earlier graph metadata |
| a consumer has two producers | lifecycle construction is ambiguous |
| a consumer has no producer | rewrite is incomplete |
| final producer map changes epoch unexpectedly | same bug as late suppression |
| correctness fails before structural density improves | construction semantics are wrong |

## Why This Raises TFLOPS

Useful FLOPs are fixed. TFLOPS rises when the same WMMA math takes less time.

The rotated pipeline reduces:

```text
global loads per WMMA
LDS stores per WMMA
waits/barriers per WMMA
address bookkeeping per WMMA
```

and enables:

```text
current-slot compute overlapped with next-slot staging
```

That is the hand LDS2 lesson. The generated path must construct that lifecycle up front instead of trying to recover it
by deleting stores after the renderer has lost epoch ownership.
