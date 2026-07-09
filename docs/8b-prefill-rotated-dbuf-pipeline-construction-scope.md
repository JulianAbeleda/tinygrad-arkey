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

Status: blocked.

The P4 readiness gate now reports:

```json
{
  "ready": false,
  "blocked_at": "P4",
  "reason": "no implemented owner-aware STAGE lowering hook; current lowering materializes generic local stores before renderer",
  "required_hook": "lower Ops.STAGE with RotatedStageOwner so legacy duplicate producers are never emitted",
  "forbidden_fallback": "PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS late deletion"
}
```

This is the clear blocker for P4-P9. Existing alternatives were checked:

| route | result |
| --- | --- |
| renderer late deletion | structurally smaller but wrong epoch/value |
| StageOwner late suppression | wrong output |
| existing `PREFILL_TC_LOCAL_STAGE_COOP_POST=1` shortcut | not viable as a quick P4 path; small trace compile did not finish within the bounded poll |
| generic `Ops.STAGE` lowering | currently emits local stores through `rangeify.bufferize_to_store`, with no rotated owner hook |

Therefore P4 cannot be completed by tuning flags or renderer suppression. It needs a new owner-aware lowering path at the
`Ops.STAGE` materialization boundary.

#### P4A. Owner-Aware STAGE Lowering Scope

The concrete hook is:

```text
tinygrad/schedule/rangeify.py::bufferize_to_store
```

That is where a `LOCAL` `Ops.STAGE` currently becomes:

```python
buf = LOCAL placeholder
store_idx = buf.index(idx)
do_store = store_idx.store(stage_src)
return buf.after(do_store.barrier())
```

The current lowering is value-correct but owner-blind. It sees a staged tensor and materializes generic local stores. It
does not know the rotated DBUF lifecycle:

```python
owner = RotatedStageOwner(
  role, lds_buffer_id, nbuf,
  reduce_epoch, dbuf_slot,
  producer_phase, consumer_phase,
)
```

P4 must add a default-off owner-aware path at this boundary. The first implementation should be deliberately narrow:

| Step | Change | Pass condition |
| --- | --- | --- |
| P4A.1 parse owner metadata | Add a small parser for `wmma_frag_buffer_proof` / rotated owner tags at `Ops.STAGE` lowering time. | Unit test proves A/B role, LDS id, `nbuf=2`, tile size, and reduce carrier are recognized without importing prefill audit code into generic rangeify. |
| P4A.2 non-destructive trace | Under `PREFILL_DBUF_ROTATED_STAGE_LOWERING_AUDIT=1`, emit/collect the exact owners seen by `bufferize_to_store` while keeping the produced graph byte-for-byte behavior-equivalent. | Existing generated route remains correct; audit shows the same A/B owners as `prefill_stage_owner_audit.py --boundary postrange`. |
| P4A.3 B-only construction | Under `PREFILL_DBUF_ROTATED_STAGE_LOWERING=1 PREFILL_DBUF_ROTATED_STAGE_ROLE=B`, construct the B producer lifecycle from the owner plan and do not emit the legacy duplicate B producer for the same owner. | B stream shrinks, producer map remains 32/32 covered, and no load switches to a different epoch/value producer. |
| P4A.4 B correctness/timing | Run bounded `2x2`, `512x5120x5120`, `loc=2`, `unr=2`. | `status=ok`; global/store per WMMA decreases versus current D3; no late suppression flags. |
| P4A.5 A+B construction | Enable A after B passes. | A+B remains correct and reduces global/store density below the current `2.75/WMMA` band. |

The lowering contract is:

```python
def lower_stage(stage):
  owner = parse_rotated_owner(stage.tag)
  if owner is None or not rotated_stage_enabled(owner.role):
    return legacy_bufferize_to_store(stage)

  plan = rotated_plan_for(owner)
  assert plan.has_exactly_one_prior_producer_for_each_consumer()
  assert plan.has_barrier_between_producer_and_consumer()
  assert plan.epoch_key_includes_reduce_epoch_and_slot()
  return materialize_owner_plan_without_legacy_duplicate(stage, plan)
```

The safety rules are stricter than value equality:

| Rule | Why |
| --- | --- |
| Match by owner tuple, not final LDS address. | The failed suppression probe proved slot/address coverage can still pick the wrong epoch. |
| Producer key includes role, LDS id, DBUF slot, reduce epoch, tile index, byte window. | Static expression equality is not enough in a rotated loop. |
| Every consumer must have exactly one prior producer. | A covered load can still be covered by the wrong store. |
| A barrier must separate producer and consumer on every emitted path. | LDS visibility is synchronization-dependent, not just dominance-dependent. |
| No renderer late deletion fallback. | `PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS` already produced smaller wrong streams. |

The first useful small test is not a performance run. It is:

```bash
PYTHONPATH=. PREFILL_DBUF_ROTATED_STAGE_LOWERING_AUDIT=1 \
  python3 extra/qk/prefill/prefill_stage_owner_audit.py \
  --shape 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --boundary postrange --json
```

Done for P4A.2 means the lowering hook sees the same owner set that the postrange audit sees. Only then should the
B-only destructive flag exist.

#### P4B. What Not To Build

Do not build these as fixes:

| Non-fix | Reason |
| --- | --- |
| More `PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS` variants | Late deletion already changed producer epochs and returned wrong values. |
| Renderer moved-store memoization | It can reduce counts but collapses non-equivalent producers. |
| Scheduler/waitcnt tuning before P4 | It tunes the current over-staged lifecycle, not the target lifecycle. |
| Full A+B destructive rewrite first | If B-only fails, A+B doubles the ambiguity and hides the first bad ownership transition. |

The next implementation checkpoint is therefore P4A.1/P4A.2: prove the actual `Ops.STAGE` lowering hook can see the
same owner identity that exists at postrange, without changing the stream.

Status update:

| Check | Result |
| --- | --- |
| `PREFILL_DBUF_ROTATED_STAGE_LOWERING_AUDIT=1`, `--boundary full` | PASS. The real `bufferize_to_store` hook sees exactly two owners: A/B, `nbuf=2`, with reduce/global ranges still present. |
| Existing B tile-key helper on active `512x5120x5120` | BLOCKED. It silently falls back because B global `tile_count=160`; staging all N tiles as one LDS object is too large. |
| `PREFILL_TC_LOCAL_STAGE_B_TILEKEY_DROP_GLOBAL=1` active-shape probe | REFUTED. It engages the helper but returns `WRONG rr=1.2e+00` and worsens density to about `94.75 inst/WMMA`, `4.0 ds_store_b128/WMMA`, `4.0 ds_load_b128/WMMA`. |
| Existing B tile-key helper on small-N `512x512x5120` where `tile_count<=64` | REFUTED. It is still `WRONG rr=1.2e+00`, so the issue is not only active-shape LDS size; the helper's B layout/consumer mapping is incorrect. |
| `PREFILL_TC_LOCAL_STAGE_B_TILEKEY_GENERIC_LAYOUT=1` | REFUTED. It changes B to the generic `(warp*2 + local)*128 + fragment` address shape but still returns `WRONG rr=1.2e+00`; density remains worse at about `96.1 inst/WMMA`. |
| `PREFILL_TC_LOCAL_STAGE_B_TILEKEY_GENERIC_NO_SLOT=1` | REFUTED. Removing the explicit `kr%nbuf` slot from the generic-layout probe still returns `WRONG rr=1.3e+00`. |

Conclusion: P4A is complete, but destructive P4 is still blocked. The current `rangeify` hook can see ownership, but it
does not own the paired read index. The only existing co-located B helper that owns store+load is not semantically
correct, and layout constants are not the missing piece. P4 must therefore proceed by either:

1. replacing B tile-key with a new postrange owned-stage group that constructs B producer and B consumer from the same
   `RotatedStageOwner`.
2. only if that is too invasive, first add a graph-level diff/audit that maps generic STAGE B's exact scalar producer
   order to WMMA B fragment lanes, then rebuild B tile-key from that proof rather than by address-shape guessing.

Do not attempt A+B destructive lowering until B-only is correct on both small-N and active-N.

#### P4C. Generic B Contract For The Replacement

The known-correct generic B path is now audited by:

```bash
PYTHONPATH=. python3 extra/qk/prefill/prefill_stage_owner_audit.py \
  --shape 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --boundary postrange --json
```

`generic_b_stage_contract.ok=true` means the B operand is still represented before lowering as:

```text
STAGE dtype:        dtypes.half.vec(16)
STAGE shape:        [32, 2, 16]
STAGE index ranges: WARP x LOCAL
consumer:           INDEX(STAGE_B, WARP, LOCAL)
contract arg:       ((5,2), (4,2), (3,2), (9,2))
```

This is the replacement contract. A correct owned B stage must preserve this vector payload and consumer identity. The
refuted tile-key helper used scalar lane-oriented packing, which is why address-formula tuning did not fix correctness.

Implementation order for the next push:

| Step | Work | Gate |
| --- | --- | --- |
| P4C.1 | Keep `generic_b_stage_contract` green on the known-correct path. | `ok=true`, one B stage, one direct B consumer. |
| P4C.2 | Add a default-off `OwnedBStage` builder that emits the same vector `STAGE_B` contract first, with no rotation. | Correctness and density identical to baseline. |
| P4C.3 | Move only the owner/lifecycle metadata into `OwnedBStage`; no store count change yet. | Producer map unchanged; metadata says each consumer has one owner. |
| P4C.4 | Only then add rotation/future ownership. | B-only correctness first; density must improve without late suppression. |

Status:

| Check | Result |
| --- | --- |
| `generic_b_stage_contract` on baseline | PASS: one B `STAGE`, one direct B consumer, vector payload over `WARP x LOCAL`. |
| `PREFILL_DBUF_OWNED_B_STAGE_IDENTITY=1` | PASS: correct and structurally identical to baseline (`39.062 inst/WMMA`, `3.312 wait/WMMA`, `2.0 global_b128/WMMA`, `2.0 ds_store_b128/WMMA`, `4.0 ds_load_b128/WMMA`, `0.125 barrier/WMMA`). |
| `PREFILL_DBUF_OWNED_B_STAGE_IDENTITY=1 PREFILL_DBUF_OWNED_B_STAGE_META=1` | PASS: B stage and B consumer both carry `owned_stage=B_IDENTITY`, `producer_epoch=same_reduce`, `consumer_epoch=same_reduce`, `rotation=none`; stream remains correct and structurally unchanged. |

This completes P4C.1/P4C.2/P4C.3. The next non-speculative step is P4C.4: introduce a rotated B producer epoch only
after the metadata can prove that every B consumer still has one owner. The first rotated probe must remain B-only and
must fail closed before changing A or scheduler placement.

P4C.4 readiness currently blocks:

```json
{
  "ready": false,
  "blocked_at": "P4C.4",
  "reason": "rotated B needs a prologue/body/tail split before behavior changes; current contract still binds producer and consumer to the same reduce epoch",
  "required_next_object": "OwnedBStage(prologue produce k0, body consume k and produce k+1, tail consume final)",
  "forbidden_shortcut": "substitute reduce_epoch -> reduce_epoch+1 inside the existing STAGE without first/tail guards"
}
```

This is the next real layer. The identity owned B stage is safe, but rotation is not a local expression rewrite. It needs
a lifecycle object that can represent warmup and drain explicitly.

#### P4D. OwnedBStage Lifecycle Object

P4D is the bridge from safe identity metadata to a behavior-changing rotated B stage. It is intentionally split into
audit object first, emitter second.

The audit object now exists as `owned_b_stage_lifecycle` in:

```text
extra/qk/prefill/prefill_stage_owner_audit.py
```

Real command:

```bash
PYTHONPATH=. PREFILL_DBUF_OWNED_B_STAGE_IDENTITY=1 PREFILL_DBUF_OWNED_B_STAGE_META=1 \
  python3 extra/qk/prefill/prefill_stage_owner_audit.py \
  --shape 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --boundary postrange --json
```

Current result:

```json
{
  "owned_b_stage_lifecycle": {
    "ok": true,
    "source": "audit_only_owned_b_stage_lifecycle",
    "reduce_size": 80,
    "prologue": ["produce B slot0 epoch k0", "barrier"],
    "body": ["consume B slot k%2 epoch k", "produce B slot (k+1)%2 epoch k+1", "barrier"],
    "tail": ["consume B slot last%2 epoch last"]
  },
  "p4c_rotation_readiness": {
    "ready": false,
    "reason": "audit lifecycle exists, but no postrange/codegen emitter can yet materialize separate B prologue/body/tail producers",
    "next_implementation_hook": "postrange owned-stage rewrite before generic Ops.STAGE lowering"
  }
}
```

Implementation contract for the emitter:

| Phase | Emits | Must prove |
| --- | --- | --- |
| Prologue | `produce(B, slot=0, epoch=k0)` then barrier | First consumer has a completed producer. |
| Body | `consume(B, slot=k%2, epoch=k)` and `produce(B, slot=(k+1)%2, epoch=k+1)` then barrier | Current consumer never reads the future producer; future producer is visible before next iteration. |
| Tail | `consume(B, slot=last%2, epoch=last)` | Last produced value is consumed once and not overwritten first. |

Forbidden shortcut:

```text
rewrite existing same-epoch STAGE index to k+1 without first/tail guards
```

Next code layer:

```text
postrange owned-stage rewrite before generic Ops.STAGE lowering
```

This is where P4D stops right now. The lifecycle is scoped and machine-audited; the behavior-changing emitter is still
missing.

#### P4E. OwnedBStage Emitter Boundary

The postrange hook is now explicit:

```text
tinygrad/codegen/opt/postrange.py::_tc_local_stage_b_src
```

Emitter modes:

| Flag | Result |
| --- | --- |
| `PREFILL_DBUF_OWNED_B_STAGE_EMIT=identity` | Emits the known-correct generic vector `STAGE_B` contract. This is behavior-neutral and remains correct. |
| `PREFILL_DBUF_OWNED_B_STAGE_EMIT=audit` | Alias for identity emission while collecting proof metadata. |
| `PREFILL_DBUF_OWNED_B_STAGE_EMIT=rotate` | Fails fast with `KernelOptError` until a prologue/body/tail materializer exists. |

Measured identity gate:

```text
status=ok
inst/WMMA=39.062
wait/WMMA=3.312
global_b128/WMMA=2.0
ds_store_b128/WMMA=2.0
ds_load_b128/WMMA=4.0
barrier/WMMA=0.125
```

Measured rotate gate:

```text
status=KernelOptError
message=PREFILL_DBUF_OWNED_B_STAGE_EMIT=rotate requires a prologue/body/tail OwnedBStage emitter; refusing same-epoch STAGE mutation
```

This is intentional. The emitter boundary is installed, but only identity emission is implemented. The next code change
must implement the materializer, not bypass the gate.

#### P4F. Hand ASM Lifecycle Oracle

The hand LDS2 ASM path is now used as an oracle for lifecycle construction, not as code to copy. The reusable trace is:

```bash
PYTHONPATH=. DEV=AMD:ISA \
  python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --kind hand-lds2 --m 512 --n 5120 --k 5120 \
  --wm 2 --wn 2 --waves-m 1 --waves-n 1 --bk 32 --dbuf 1 \
  --json
```

It emits `owned_b_emitter_oracle`:

```json
{
  "prologue_store_slots": [0],
  "body_compute_slots": [0, 1],
  "body_store_slots": [1, 0],
  "tail_compute_slots": [0, 1],
  "asm_stream_facts": {
    "prologue_store_count": 8,
    "body_store_count": 24,
    "body_loads_before_first_body_store_count": 8,
    "pipeline_epoch_candidate": true,
    "prologue_body_physical_window_overlap_count": 8
  }
}
```

Compiler contract derived from ASM:

```text
prologue:
  produce slot 0

body:
  consume slot 0
  consume slot 1
  produce slot 1
  produce slot 0

tail:
  consume slot 0
  consume slot 1
```

The generated identity path proves the gap: with the same DBUF-safe env, generated has correct LDS operands but all
stores are in the prologue:

```text
generated identity:
  ds_store_b128: 32
  ds_load_b128: 64
  s_barrier: 2
  prologue stores: 32
  body stores: 0
  pipeline_epoch_candidate: false
```

So the materializer's job is not merely "emit LDS." It must move from prologue-only staging to hand-style body staging:

```text
current generated: produce all before compute
target generated:  warmup once, then produce future slot inside body
```

This is the ASM-derived lifecycle construction target for `OwnedBStageEmitter`.

### P4G. Owned-B Emitter Object Boundary

Status: implemented for identity only.

`PREFILL_DBUF_OWNED_B_STAGE_EMIT=object_identity` now routes B staging through an explicit
`OwnedBStageEmitter` at `postrange._tc_local_stage_b_src`, while deliberately emitting the same generic vector
`STAGE_B` graph as the known-correct identity path.

This proves the boundary is usable without changing behavior:

```text
shape=2x2
status=ok
TFLOPS=7.84
WMMA=16
global_load_b128=32
ds_store_b128=32
ds_load_b128=64
s_barrier=2
inst/WMMA=39.062
wait/WMMA=3.312
global/WMMA=2.0
store/WMMA=2.0
load/WMMA=4.0
```

The postrange owner audit still reports:

```text
owned_b_stage_lifecycle.ok=true
p4c_rotation_readiness.ready=false
blocked_at=P4C.4
required_next_object=OwnedBStage(prologue produce k0, body consume k + produce k+1, tail consume final)
```

Pass:

```text
object_identity preserves correctness
object_identity preserves density counters
rotate still fails closed
```

Next:

```text
replace identity emission inside OwnedBStageEmitter with a real prologue/body/tail materializer
do not implement rotate as same-epoch STAGE mutation or late store suppression
```

### P4H. Rangeify Rotate Hook

Status: implemented as a fail-closed hook only.

`PREFILL_DBUF_OWNED_B_STAGE_EMIT=rotate PREFILL_DBUF_OWNED_B_STAGE_META=1` now creates an explicit B rotate owner
tag:

```text
owned_stage=B_ROTATE
lifecycle=prologue_body_tail
rotation=kr_mod_nbuf
```

`rangeify.bufferize_to_store` has a narrow tagged branch for that owner before generic local `STAGE` lowering emits
the prologue-only store:

```text
if role=B and owned_stage=B_ROTATE:
  validate nbuf/tile_count/tile_elems/reduce carrier
  fail closed until the prologue/body/tail materializer exists
```

Verified:

```text
full-boundary audit reaches tinygrad/schedule/rangeify.py::_prefill_dbuf_owned_b_stage_lowering
error=PREFILL_DBUF owned B rotate lowering reached rangeify hook, but prologue/body/tail materializer is not implemented
```

This proves the tag survives to the only safe destructive boundary. It also prevents the bad outcome where `rotate`
silently falls back to identity or to late store suppression.

Remaining blocker:

```text
implement _prefill_dbuf_owned_b_stage_lowering as a real materializer:
  prologue produce k0
  body consume k and produce k+1
  tail consume final
```

### P5. Add A

Repeat P3/P4 for A after B is correct.

Pass:

```text
A+B route correct
global/store per WMMA moves below 2.75
no late suppression flags enabled
```

Status: blocked on P4. A cannot be added until one-role destructive owner lowering is correct.

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

Status: blocked on P4/P5. Scheduler tuning before correct owner-aware construction would tune the wrong lifecycle.

### P7. Whole-Prefill Integration

Status: blocked on P4/P5. Whole-prefill integration requires the generated primitive to be correct and smaller in the
bounded GEMM first.

### P8. Promotion Gate

Status: blocked on P4/P7. There is no route to promote until owner-aware construction passes correctness and performance.

### P9. Cleanup

Status: partially actionable only for documentation. The late-suppression probes remain as negative evidence; they must
not be enabled as fixes. Code cleanup should wait until the owner-aware `Ops.STAGE` lowering exists, otherwise we lose
the repros that explain the blocker.

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
