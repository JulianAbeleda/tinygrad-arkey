# Fast Prefill Active-Shapes Scheduler Blocker

Date: 2026-07-08.

## Summary

Scheduler tuning is currently blocked because the final active-shape stream does not expose future-slot work inside the
WMMA body.

The active generated LDS/DBUF shapes are structurally correct enough to stage through LDS:

```text
global_load_b128 -> ds_store_b128 -> s_barrier -> ds_load_b128 -> v_wmma
```

but the generated stream is still shaped as:

```text
prologue:
  global_load_b128 all staged fragments
  ds_store_b128 all staged fragments
  s_barrier
  first ds_load_b128 group

body:
  ds_load_b128 current fragment
  v_wmma
  ds_load_b128 current fragment
  v_wmma
  ...

tail:
  epilogue stores
```

What we need for scheduler/waitcnt to matter is:

```text
prologue:
  stage slot 0

body:
  prefetch/store future slot
  load current slot from LDS
  v_wmma current slot

tail:
  consume final slot
```

## Evidence

Command family:

```bash
DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --active-generated --shapes '2,2;4,2;2,4' --m 512 --n 5120 --k 5120 --loc 2 --unr 2
```

Observed active-shape lifecycle:

| shape | packed LDS | scalar LDS fallback | future-slot global/store before WMMA | waits/WMMA |
| --- | --- | ---: | --- | ---: |
| `2x2` | yes | 0 | no | 3.312 |
| `4x2` | yes | 0 | no | 2.656 |
| `2x4` | yes | 0 | no | 2.656 |

Probe correction on 2026-07-08: `native_isa_l4_stream_probe.py` no longer counts `ds_load_b128` as future staging
work. `ds_load_b128` between WMMAs is current-slot consumption; scheduler-readiness now requires body
`global_load_b128` or `ds_store_*` work. With that corrected definition, `D3_cadence.body_has_next_slot_work=false`
and `D7_scheduler_readiness.ok=false` on the active route.

`AMD_ISA_SCHED=0` and `AMD_ISA_SCHED=1` produce the same cadence for `2x2`.

Existing flags did not expose the missing body prefetch:

| flag set | result |
| --- | --- |
| baseline DBUF bundle | no future-slot work before current compute |
| `PREFILL_DBUF_GLOBAL_ADDR_INLOOP=1` | unchanged |
| `PREFILL_DBUF_LDS_LOAD_SERIAL=1` | unchanged |
| `PREFILL_DBUF_LDS_RELOAD_ANCHOR=1 PREFILL_DBUF_LDS_LOAD_SERIAL=1 PREFILL_DBUF_LDS_BASE_REMAT=1 PREFILL_DBUF_LDS_BASE_REMAT_DEEP=1` | unchanged |

## Diagnosis

The list scheduler is not the primitive blocker.

Reasons:

1. The scheduler only reorders within basic blocks.
2. `s_barrier` is a hard scheduling barrier.
3. Current generated code emits all global loads and LDS stores before the barrier.
4. The WMMA body contains only current-slot `ds_load_b128` work.
5. No legal scheduler can move future-slot stores across the barrier into the body without changing memory semantics.

So the missing primitive is a pre-scheduler codegen shape:

```text
peeled prologue/body/tail DBUF lifecycle
```

not a waitcnt tweak.

## Current Blockers

| Blocker | Meaning |
| --- | --- |
| Proof metadata propagation | UOp `tag` metadata from postrange does not survive to AMD WMMA operand lowering. This blocks proof-keyed resident reuse. |
| DBUF body shape | `_prefill_dbuf_peel` creates extra K work and D2 slot proof is now visible, but staged global/LDS stores are still grouped before the barrier instead of emitted as next-slot body work. |
| Scheduler visibility | The scheduler sees no future global/store work between WMMAs, so waitcnt tuning has no useful overlap to preserve. |

## Reuse-Based Trial Result

Using only existing surfaces:

```text
kernel_lifecycle_trace.py --active-generated --shapes '2,2'
native_isa_l4_stream_probe.py --m-up 1
wmma_frag_key_audit.py --shapes '2,2'
a_fragment_alias_probe.py --cases 2,2
```

Current `2x2` finding:

| Gate | Result |
| --- | --- |
| D2 two-slot identity | PASS in `native_isa_l4_stream_probe.py`: normalized LDS byte-window proof is covered for both operands. |
| Current-slot LDS consumption | PRESENT: body regions contain `ds_load_b128` before WMMAs. |
| Future-slot staging | ABSENT: body regions contain no `global_load_b128` or `ds_store_*`. |
| Proof-key reuse | FAIL-CLOSED: address-only reuse groups exist, but no proof metadata reaches AMD operand lowering. |

The next codegen change should reuse these same surfaces as gates. Do not add another DBUF tracer.

## Next Primitive Fix

Implement an explicit generated DBUF lifecycle before AMD scheduling:

```text
stage(slot 0)
barrier
for k-phase:
  stage(next slot)        # global_load_b128 + ds_store_b128, if next exists
  load(current slot)      # ds_load_b128
  compute(current slot)   # v_wmma
  barrier/slot edge
tail compute
```

This must be done before waitcnt/scheduler tuning. After this shape exists, targeted waitcnt can be judged on whether it
leaves future VMEM/LDS work outstanding while draining only the exact current-slot consumers.

## Stop Condition

Do not keep tuning `AMD_ISA_SCHED`, `AMD_ISA_WAITCNT_TARGETED`, or LDS remat flags until the lifecycle tracer shows:

```text
future_slot_before_compute = True
```

for at least `2x2`.
