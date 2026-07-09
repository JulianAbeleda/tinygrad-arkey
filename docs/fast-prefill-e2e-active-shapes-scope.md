# Fast Prefill E2E Active-Shapes Scope

Date: 2026-07-08.

## Decision

The E2E path to fast generated prefill on gfx1100 is:

```text
active shapes only: 2x2, 4x2, 2x4
packed LDS staging
two-slot DBUF cadence
proof-safe A/B fragment reuse
then wait/scheduler tuning
then model-level promotion
```

Generated `4x4` is parked by `docs/gfx1100-4x4-path-parked-scope.md` and is not an E2E blocker.

## Current State

| Layer | Current state | Evidence / note |
| --- | --- | --- |
| Shape policy | `4x4` parked; active shapes are `2x2`, `4x2`, `2x4`. | `PREFILL_ALLOW_PARKED_4X4=1` is required to reopen. |
| Baseline correctness | Active shapes have passing generated variants. | Existing shape matrix runs pass for no-LDS DBUF and/or DBUF-safe variants. |
| Hand target | Hand LDS2 is much denser. | Hand has fewer LDS loads, waits, and instructions per WMMA. |
| Generated gap | Generated reloads A/B too often and waits too often. | `ds_load/WMMA=4.0` generated vs `2.0` or `1.5` hand. |
| Reuse lever | Unsafe resident reuse improves density but NaNs. | `ds_load/WMMA 4.0 -> 2.0`, correctness fails. |
| Primitive blocker | Missing semantic proof for A/B fragment reuse. | Current key lacks slot, K phase, row/col identity, producer/overwrite epochs. |
| Scheduler tuning | Not ready. | Wait tuning should follow correct reuse/cadence. |
| E2E promotion | Not proven. | Need same-clock prefill/model measurements after primitive lands. |

## 100% Definition

This E2E work is complete only when every gate below passes.

| Gate | Requirement | Evidence |
| --- | --- | --- |
| E0. Policy fence | `4x4` remains excluded from default gfx1100 generated prefill. | Active tools default to `2x2;4x2;2x4`; table/warmstart sanitizes `4x4`. |
| E1. Active-shape correctness | Generated `2x2`, `4x2`, and `2x4` pass GPU correctness under the selected route. | `hand_vs_generated_shape_matrix.py --shapes '2,2;4,2;2,4'` all `status=ok`. |
| E2. Packed LDS | Promoted path uses packed global/LDS traffic. | `global_load_b128`, `ds_store_b128`, `ds_load_b128`; scalar fragment LDS stores are absent on promoted path. |
| E3. DBUF cadence | Two-slot identity and prefetch/consume cadence are visible. | Blocked: current stream has current-slot `ds_load_b128` between WMMAs, but no body `global_load_b128`/`ds_store_*` next-slot work. |
| E4. Proof-safe reuse | A/B resident reuse is enabled only when the full proof key is available. | Blocked: proof tags from postrange do not survive to AMD operand lowering; address-only promotions fail closed. |
| E5. Density win | Generated staging density improves toward hand. | `2x2 ds_load/WMMA <= 2.0`; `4x2/2x4 ds_load/WMMA <= 1.5` where feasible. |
| E6. Wait/bookkeeping win | Waits and instructions per WMMA improve over current generated DBUF-safe baseline. | Same harness structural counters. |
| E7. Timing win | Active generated shapes improve same-clock TFLOPS over current generated DBUF-safe baseline. | Repeated pinned-clock runs, same env. |
| E8. Model-level win | End-to-end prefill route improves real model prefill throughput without correctness regressions. | Existing model/prefill bench reports improved prefill time/tok/s. |
| E9. Default safety | Fast path remains opt-in until E1-E8 pass, with rollback flags documented. | Unit tests pass; route gate artifact records chosen params and policy. |

## Phase Plan

### Phase 0: Freeze Baseline And Policy

Purpose: make sure every later delta is compared against the right target.

Commands:

```bash
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --shapes '2,2;4,2;2,4' --loc 2 --unr 2 --pin-clock --json

PYTHONPATH=. python3 extra/qk/prefill_v2_schedule_table_gate.py \
  --shapes 4096x4096 --no-artifact --compact
```

Done when:

- `4x4` is absent from default active shape matrix.
- Table gate reports sanitized params when a frozen table row contains `u0=4,u1=4`.
- Baseline JSON is saved or referenced in a scope update.

Parallelizable: yes. This can run while Phase 1 code reading happens.

### Phase 1: Proof Metadata Contract

Purpose: define the metadata that must survive from postrange staging to AMD WMMA lowering.

Required fields:

```text
role: A | B
lds_buffer_id
dbuf_slot
k_phase / reduce step
logical_row_or_col
byte_start
byte_len = 32
producer_epoch
overwrite_epoch
```

Implementation target:

```text
postrange staging site
  tags LDS fragment carrier with proof metadata

devectorizer / late lowering
  preserves metadata through expansion

AMD isel
  sees proof metadata on the WMMA operand carrier
```

Done when:

- `PREFILL_WMMA_FRAG_KEY_DUMP=1` prints every field or an explicit unprovable reason.
- Default behavior is unchanged when dump/reuse flags are off.
- Unit tests cover metadata preservation for at least one A and one B fragment.

Parallelizable: partially. Code-path audit and dump-tool formatting can run in parallel; final metadata schema must be agreed first.

### Phase 2: Offline Grouping Audit

Purpose: prove which fragments may be reused before changing behavior.

Audit algorithm:

```text
for each active shape:
  for each WMMA operand:
    collect proof key or unprovable reason

  group by:
    current carrier identity
    address-only key
    proof key

  report:
    consumers per group
    rejected address-only merges
    expected hand-like reuse groups
```

Done when:

- `2x2`, `4x2`, and `2x4` audits produce proof-safe groups for at least the known hand-like B-column reuse.
- Every unsafe address-only merge is either accepted by proof key or rejected with a named missing/different field.
- No promotion happens in this phase.

Parallelizable: yes. Shape-specific audits can run independently.

### Phase 3: Fail-Closed Resident Reuse

Purpose: replace unsafe reuse with proof-keyed reuse.

Behavior:

```text
proof_key = build_frag_key(operand)

if proof_key is None:
  emit current per-WMMA reload path
else:
  use resident fragment keyed by proof_key
```

Invariants:

- Never merge A and B.
- Never merge different DBUF slots.
- Never merge different K phases.
- Never merge different row/column identities.
- Never reuse across producer/overwrite epochs.
- Never let address-only equality promote a fragment.

Done when:

- `PREFILL_WMMA_CHAIN_AB_RESIDENT=1` is correct on `2x2`.
- `ds_load/WMMA` improves on `2x2` without scalar LDS fallback.
- With deliberately missing metadata, the path falls back instead of NaNing.

Sequential: yes. This depends on Phase 1 and Phase 2.

### Phase 4: Expand To 4x2 And 2x4

Purpose: prove the primitive generalizes to both active rectangular shapes.

Commands:

```bash
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes '4,2;2,4' --loc 2 --unr 2 --pin-clock --json
```

Done when:

- `4x2` and `2x4` are correct with proof-keyed reuse enabled.
- Both shapes improve `ds_load/WMMA`, `wait/WMMA`, and/or `inst/WMMA` over current generated DBUF-safe baseline.
- Any asymmetry between A-heavy `4x2` and B-heavy `2x4` is documented.

Parallelizable: yes. `4x2` and `2x4` investigations can split.

### Phase 5: Restore/Validate Packed LDS DBUF Cadence

Purpose: make the generated lifecycle match the hand LDS2 class rather than just reducing load count.

Expected lifecycle:

```text
prologue:
  global_load_b128 A/B slot 0
  ds_store_b128 A/B slot 0
  barrier

body:
  global_load_b128 A/B next slot
  ds_load_b128 A/B current slot
  WMMA current slot
  ds_store_b128 A/B next slot
  barrier / slot-safety edge

tail:
  consume final slot
  epilogue stores
```

Done when:

- Lifecycle tracer marks two-slot identity.
- Future-slot memory work exists before current-slot compute finishes.
- Promoted path has no scalar `ds_store_b16` / fragment scalar LDS staging.

Sequential: after Phase 3 for promoted path, but lifecycle tracing can be prepared earlier.

### Phase 6: Waitcnt And Scheduler Tuning

Purpose: reduce waits only after the stream has correct reusable work to overlap.

Allowed work:

- targeted `vmcnt`/`lgkmcnt` refinement,
- local scheduling around DS/global load clusters,
- coalescing redundant waits,
- preserving barriers and exact producer-consumer waits.

Disallowed work:

- scheduling around missing proof metadata,
- hiding correctness failures with waits,
- tuning parked `4x4`.

Done when:

- `wait/WMMA` improves on active shapes.
- GPU correctness still passes.
- Same-clock TFLOPS improves beyond measurement noise.

Sequential: after Phase 5.

### Phase 7: E2E Model Gate

Purpose: prove the kernel-level win matters in the actual prefill route.

Measurements:

```text
same model
same prompt length / PREFILL_UBATCH
same clock policy
same route policy
baseline generated vs new generated
```

Done when:

- Model-level prefill throughput improves.
- No route falls back unexpectedly.
- Route artifact records active shapes and parked `4x4` policy.
- Rollback flags are documented.

Sequential: after Phase 6.

## Parallel Work Breakdown

| Workstream | Can run now? | Output |
| --- | --- | --- |
| W0. Baseline runner | Yes | Current active-shape timing/structure JSON. |
| W1. Metadata path audit | Yes | Exact carrier nodes/files where proof metadata must attach and survive. |
| W2. Grouping audit tool | Yes | Shape-by-shape proof-key grouping report. |
| W3. Lifecycle tracer cleanup | Yes | Active-shape DBUF cadence report. |
| W4. Proof-keyed reuse implementation | After W1/W2 | Fail-closed resident reuse. |
| W5. Rectangular-shape expansion | After W4 | `4x2`/`2x4` correctness and density report. |
| W6. Wait/scheduler tuning | After E3 stream shape exists | Reduced waits/instructions and timing win. |
| W7. E2E route gate | After W6 | Model-level promotion evidence. |

## Reuse Policy

Do not add a new harness for this path unless one of these existing surfaces cannot answer the question:

| Need | Existing surface to use | Do not duplicate with |
| --- | --- | --- |
| Active-shape timing and hand/generated counters | `extra/qk/prefill/hand_vs_generated_shape_matrix.py` | another benchmark wrapper |
| Generated kernel lifecycle and active DBUF cadence | `extra/qk/prefill/kernel_lifecycle_trace.py --active-generated` | another lifecycle tracer |
| Lower-level native-ISA DBUF gate details | `extra/qk/prefill/native_isa_l4_stream_probe.py` | another stream scanner |
| WMMA A/B proof-key grouping | `extra/qk/prefill/wmma_frag_key_audit.py` | another reuse audit script |
| Byte-window LDS alias/proof checks | `extra/qk/prefill/a_fragment_alias_probe.py` | another LDS address analyzer |
| Schedule-table/default route policy | `extra/qk/prefill_v2_schedule_table_gate.py` | another policy gate |
| Fast-prefill generated search runs | `extra/qk/prefill_v2_schedule_search.py` | another runner |

Any new code should be one of:

- a small extension to the existing surface above,
- a shared helper used by at least two existing probes,
- or the actual codegen/runtime change under test.

Anything else is scope sprawl.

## Risk Register

| Risk | Symptom | Mitigation |
| --- | --- | --- |
| Metadata is lost in late lowering | Dump shows `unprovable` for all operands. | Attach metadata at a carrier that survives devectorization, or add a side table keyed by stable UOp identity. |
| Proof key is too strict | Correct but no density improvement. | Audit rejected groups; relax only with a named proof field, never address-only equality. |
| Proof key is too weak | NaN / wrong output under reuse. | Fail closed; add negative tests for slot, K phase, and row/column mismatch. |
| LDS lifecycle still reloads too often | Correctness passes but `ds_load/WMMA` remains 4.0. | Fix reuse before wait tuning. |
| Scheduler masks a bug | Timing improves but correctness is unstable. | Scheduler phase is blocked until proof-safe reuse is correct. |
| Active shapes cannot reach target | `2x2/4x2/2x4` plateau below goal. | Only then reopen resource proof for larger/future shapes; do not reopen `4x4` by default. |

## Immediate Next Step

Run Phase 0 baseline, then implement Phase 1 metadata dump over active shapes only.

First implementation target:

```text
PREFILL_WMMA_FRAG_KEY_DUMP=1
shape: 2x2
output: one JSON row per WMMA operand with proof fields or explicit unprovable reasons
behavior: no codegen change
```

After that, use the audit to make proof-keyed resident reuse correct on `2x2` before expanding to `4x2` and `2x4`.
