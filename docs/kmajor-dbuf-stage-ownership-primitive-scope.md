# K-Major DBUF Stage Ownership Primitive Scope

Date: 2026-07-08

## Big Picture

Goal: make generated native-ISA fp16 prefill match the hand LDS2 primitive on the active shapes (`2x2`, `4x2`, `2x4`) without relying on handwritten kernels.

Current generated K-major lowering now matches hand on A/B LDS fragment reuse:

| Shape | Generated `ds_load_b128/WMMA` | Hand `ds_load_b128/WMMA` |
| --- | ---: | ---: |
| `2x2` | `2.0` | `2.0` |
| `4x2` | `1.5` | `1.5` |
| `2x4` | `1.5` | `1.5` |

The remaining gap is not fragment reuse. It is stage ownership and cadence.

Generated still presents the renderer with a graph shaped like:

```text
stage all K/DBUF producers
barrier
load LDS fragments
run WMMAs
epilogue
```

Hand LDS2 is shaped like:

```text
stage initial slot
barrier
for phase:
  load current LDS fragments
  run WMMA cluster
  stage future slot around/after current compute
  barrier for next slot
```

The primitive fix is therefore:

```text
own DBUF stage placement as a move, not as an additive marker
```

## Current Evidence

### E1. OptOps Do Not Produce Hand Cadence

Schedule-grid trace over `u0/u1`, `loc`, and `unr`:

| Finding | Meaning |
| --- | --- |
| Every valid shape had `between_global_regions=0`. | No schedule knob tested creates hand-style interleaved staging. |
| Increasing `UNR` reduces static `inst/WMMA` but doubles pre-stage work and grows LDS. | Amortization helps counters but does not fix producer cadence. |
| `LOC=0` is best among current knobs. | Lower LDS/thread footprint, but still front-loads staging. |

Representative `2x2 loc=0 unr=2`:

```text
wmma_count=16
pre_first_wmma:
  global_load_b128=32
  ds_store_b128=32
  ds_load_b128=8
  s_waitcnt=36
between_global_regions=0
```

Representative hand LDS2 `2x2`:

```text
pre_first_wmma:
  global_load_b128=16
  ds_store_b128=8
  ds_load_b128=8
  s_waitcnt=4
between_global_regions>0
```

Conclusion: this is not solved by the current schedule table.

### E2. Waitcnt Is Secondary

Targeted/default wait variants changed static wait counts slightly, but did not move TFLOPS enough.

| Variant | Result |
| --- | --- |
| Targeted waitcnt | Correct, but high wait density remains. |
| Non-targeted/default | Slightly fewer waits on K-major, not faster enough. |
| Conservative waitcnt | Correct but much worse. |

Conclusion: waitcnt tuning comes after stage ownership. It cannot create missing overlap.

### E3. Address Folding Helps, But Is Not Sufficient

Region-relative LDS address split plus memo reduces dynamic address math:

```text
PREFILL_DBUF_LDS_REGION_BASE_SPLIT=1
PREFILL_DBUF_LDS_REGION_BASE_MEMO=1
```

Measured direction:

| Shape | Before | After |
| --- | ---: | ---: |
| `2x2 inst/WMMA` | about `34.6` | about `31.9` |
| `4x2 inst/WMMA` | about `29.2` | about `27.0` |
| `2x4 inst/WMMA` | about `29.2` | about `27.5` |

Conclusion: keep this as a supporting density fix, but it is not the primitive.

### E4. Additive D3A Is Refuted

K-major D3A marker:

```text
PREFILL_WMMA_KMAJOR_D3A_MARKER=1
PREFILL_DBUF_D3A_POST=1
```

Result:

| Route | `2x2 TFLOPS` | `inst/WMMA` | `global_b128/WMMA` | `ds_store_b128/WMMA` |
| --- | ---: | ---: | ---: | ---: |
| K-major base | about `12.0` | about `31.9` | `2.0` | `2.0` |
| Additive D3A | about `9.1` | about `38.6` | `3.125` | `3.125` |

Conclusion: adding future stage work without suppressing original pre-stage work is strictly wrong.

### E5. Renderer-Only Steal Probe Is Too Late

The first destructive probe attempted to:

1. Find the existing producer from the WMMA operand.
2. Emit it later.
3. Mark the original store as stolen.
4. Suppress the original lowering.

It still regressed:

```text
steal probe: global_b128/WMMA=3.5, ds_store_b128/WMMA=3.5, TFLOPS about 8.45
```

The audit showed why:

```text
postrange stage identity is lost by the time isel_store sees lowered address carriers
candidate selection also repeatedly stole the first slot/window
```

Conclusion: a pure renderer-side heuristic is not robust enough. Stage ownership metadata must originate at postrange, where stage producers are created.

## Root Cause

The current producer graph is born in:

```text
tinygrad/codegen/opt/postrange.py::_tc_local_stage_coop_operand
```

It does:

```python
stores = [...]
stage = UOp.group(*stores)
bar = UOp.barrier(stage)
ordered_local = bsh.after(bar)
scalar = ordered_local.index(slot + row*16 + frag_idx).load()
```

That makes every stage producer a mandatory dependency of the operand load. The AMD renderer receives a graph where the safe order is all producers before compute.

Hand does not use that ownership model. Hand owns stage placement explicitly and only materializes the producer when that DBUF slot is needed.

## 100% Definition

This work is complete only when all of the following pass.

| Gate | Requirement |
| --- | --- |
| G1. Correctness | `2x2`, `4x2`, `2x4` generated routes are `status=ok`, finite RMSE, no MMU fault. |
| G2. No over-stage | Generated `global_load_b128/WMMA` and `ds_store_b128/WMMA` do not exceed base route. |
| G3. Hand cadence appears | Generated trace has nonzero between-WMMA global stage regions. |
| G4. Fragment reuse preserved | `ds_load_b128/WMMA` remains `2.0` for `2x2`, `1.5` for `4x2`/`2x4`. |
| G5. Static density improves | `inst/WMMA` drops toward hand without increasing memory work. |
| G6. TFLOPS moves | Generated active-shape TFLOPS beats current K-major baseline at same clock policy. |
| G7. Default-off safety | New primitive is flag-gated until all active-shape gates pass. |
| G8. Tests | `python3 -m unittest test.unit.test_amd_isa_wmma` passes. |

Promotion target:

```text
generated active shapes structurally match hand cadence enough that scheduler/waitcnt tuning becomes the next bottleneck
```

Not required for this scope:

```text
4x4
full hand TFLOPS parity
default-on promotion
```

## Design Options

### Option A: Renderer-Only Stealing

Mechanism:

```text
discover original stage producer from WMMA operand
emit moved producer in K-major phase loop
suppress original producer in isel_store
```

Status: refuted as currently designed.

Why:

```text
original postrange identity is not stable at isel_store
candidate selection can grab wrong/partial stage windows
suppression becomes heuristic and still duplicates producers
```

Use only as a diagnostic, not as the production fix.

### Option B: Postrange Stage Metadata + Renderer Move

Mechanism:

1. Postrange tags each cooperative stage producer with a stable stage key.
2. The operand load carries a reference to that stage key.
3. K-major lowering decides which stage keys are prologue-owned and which are phase-owned.
4. Phase-owned producers are emitted at phase boundaries.
5. Original pre-stage lowering suppresses phase-owned producers by exact key.

This is the primitive route.

Stage key shape:

```python
StageKey = (
  "tc_local_stage",
  role,              # A or B
  lds_buffer_id,
  dbuf_slot_expr,    # preferably symbolic/stable, not id-only
  tile_index_expr,
  row_or_col,
  byte_start,
  byte_len,
)
```

Hard rule:

```text
for every stage key, exactly one owner emits it
```

### Option C: Postrange Emits Stage Pseudos

Mechanism:

Replace raw `STORE` groups with an explicit stage pseudo:

```python
TC_LDS_STAGE(key, src_fragment, dst_lds_window)
TC_LDS_WAIT(key)
TC_LDS_LOAD(key, frag_idx)
```

AMD lowering lowers pseudos into:

```text
global_load_b128
ds_store_b128
s_barrier / wait
ds_load_b128
```

This is cleaner long-term, but larger.

Recommendation:

```text
Do B first. If metadata threading becomes brittle, graduate to C.
```

## Implementation Plan

### Phase 0: Clean Probe State

Objective: keep only useful default-off findings.

Tasks:

| Task | File | Output |
| --- | --- | --- |
| Keep region-base split/memo if tests pass. | `tinygrad/renderer/isa/amd.py` | Supporting address-density flag. |
| Remove or keep default-off failed steal probes as diagnostic only. | `tinygrad/renderer/isa/amd.py` | No accidental promotion. |
| Ensure no flag changes default behavior. | all touched files | Existing tests pass. |

Gate:

```bash
python3 -m unittest test.unit.test_amd_isa_wmma
```

### Phase 1: Stage-Key Authority In Postrange

Objective: assign stable ownership keys when stage producers are created.

Tasks:

| Task | File | Detail |
| --- | --- | --- |
| Define a stable stage key helper. | `tinygrad/codegen/opt/postrange.py` | Role, LDS buffer, DBUF slot, tile index, row/col, byte window. |
| Tag each store index/value pair. | `postrange.py::_tc_local_stage_coop_operand` | Two b128 stores per 16-half fragment must have distinct byte windows. |
| Tag the corresponding operand load/proof. | `postrange.py::_wmma_frag_proof_tag` path | Load can name the producers it depends on. |
| Add trace dump. | postrange | Dump `StageKey` rows under a flag. |

Gate:

```text
For 2x2, there are exactly 32 original stage producer keys:
  A: 16 b128 stores
  B: 16 b128 stores
and WMMA phase operands reference the matching key pairs.
```

### Phase 2: Ownership Plan Builder

Objective: decide which stage keys stay in prologue and which move into phase loop.

Tasks:

| Task | File | Detail |
| --- | --- | --- |
| Build K-major stage plan. | `tinygrad/renderer/isa/amd.py::_try_wmma_kmajor_phase` | Group by phase and fragment. |
| Phase 0 owner = original prologue. | amd.py | Preserve correctness baseline. |
| Phase >0 owner = K-major loop. | amd.py | Emit before consuming phase fragments. |
| Record `moved_stage_keys`. | `IselContext` field | Used by suppression. |

Pseudocode:

```python
for phase_i in phases:
  if phase_i == 0:
    dep = original_barrier
  else:
    moved = []
    for key in stage_keys_needed_by_phase(phase_i):
      moved.append(emit_stage_for_key(key, dep=prev_phase_last))
      moved_stage_keys.add(key)
    dep = barrier_or_order(moved)

  load phase fragments with dep
  emit WMMA cluster
```

Gate:

```text
Moved-stage count is nonzero for phase >0.
Each moved key has exactly one original key match.
No moved key belongs to phase 0.
```

### Phase 3: Exact Suppression

Objective: suppress only original producers that have been moved.

Tasks:

| Task | File | Detail |
| --- | --- | --- |
| Preserve stage key through lowering. | `isel_index` / address carrier | Avoid relying on Python object identity. |
| Suppress matching original LDS stores. | `isel_store`, `isel_gated_store` | Return order-preserving `NOOP`. |
| Do not suppress unmatched producers. | amd.py | Fail closed. |
| Barrier policy. | amd.py | Keep phase-0 barrier; suppress or neutralize all-moved barriers only when safe. |

Suppression invariant:

```python
if stage_key in moved_stage_keys:
  original store lowers to NOOP(order)
else:
  original store lowers normally
```

Gate:

```text
Additive D3A failure signature disappears:
  global_b128/WMMA <= base
  ds_store_b128/WMMA <= base
```

### Phase 4: Structural Cadence Gate

Objective: prove generated stream now has hand-style interleaving.

Use existing tools:

```bash
extra/qk/prefill/hand_vs_generated_shape_matrix.py
extra/qk/prefill/kernel_lifecycle_trace.py
extra/qk/prefill/native_isa_l4_stream_probe.py
```

Required output:

| Metric | Required |
| --- | --- |
| `between_global_regions` | `>0` |
| `global_b128/WMMA` | no increase vs base |
| `ds_store_b128/WMMA` | no increase vs base |
| `ds_load_b128/WMMA` | unchanged |
| `inst/WMMA` | improves or stays neutral before timing |

### Phase 5: Correctness Gate

Command:

```bash
env DEV=AMD:ISA ... \
  python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand \
  --shapes '2,2;4,2;2,4' --loc 0 --unr 2 --pin-clock --json
```

Required:

```text
2x2 ok
4x2 ok
2x4 ok
```

No:

```text
NaN
MMU fault
group_segment_size fault
regalloc no-spill fault
```

### Phase 6: Performance Gate

Compare these rows:

| Route | Purpose |
| --- | --- |
| Current K-major base | Baseline. |
| K-major + region-base split/memo | Address-density baseline. |
| K-major + stage ownership | Candidate primitive. |
| Hand LDS2 | Structural target. |

Promotion condition:

```text
candidate TFLOPS > current K-major base TFLOPS on at least 2/3 active shapes
and no active shape regresses structurally
```

### Phase 7: Scheduler/Waitcnt Follow-Up

Only after Phase 4-6 pass:

| Work | Why Later |
| --- | --- |
| Targeted waitcnt tuning | Needs real between-WMMA producer/consumer overlap. |
| List scheduler tweaks | Current scheduler cannot create missing producers. |
| WMMA adjacency tuning | Only useful once stage cadence is correct. |

## Parallelization Plan

### Can Run In Parallel

| Workstream | Owner | Output |
| --- | --- | --- |
| W1. Stage-key design audit | Explorer/agent | Confirm key is stable and complete. |
| W2. Trace extension | Worker | Add/report per-stage-key lifecycle table. |
| W3. Address-density cleanup | Worker | Keep region split/memo clean, default-off, tested. |
| W4. Hand cadence reference table | Worker | Fixed target counters per active shape. |

### Must Run In Sequence

| Step | Depends On | Why |
| --- | --- | --- |
| S1. Stage keys in postrange | none | Foundation. |
| S2. K-major ownership plan | S1 | Needs keys. |
| S3. Suppression | S2 | Needs moved-key set. |
| S4. Correctness | S3 | Moved producer order can break correctness. |
| S5. Performance | S4 | Only meaningful after correctness. |
| S6. Scheduler/waitcnt | S5 | Needs real overlap to tune. |

## Escape Hatches

Stop and reassess if any of these happen:

| Symptom | Meaning |
| --- | --- |
| Cannot create stable stage keys before lowering. | Move to explicit stage pseudo design. |
| Suppression removes producers but correctness fails. | Ownership plan is missing barrier/slot overwrite constraints. |
| Structural overlap appears but TFLOPS falls. | Scheduler/waitcnt becomes next bottleneck. |
| Memory op density rises. | Still additive, not destructive. |
| LDS exceeds 64 KiB. | Unroll/slot plan is not viable for gfx1100. |

## Current Recommendation

Proceed with Option B:

```text
postrange stage keys
K-major ownership plan
exact original-stage suppression
structural cadence gate
then correctness/perf
```

Do not spend more time on:

```text
OptOps knob search
waitcnt-only tuning
additive D3A
renderer-only heuristic stealing without postrange keys
```

## 2026-07-08 Probe Result: Renderer Suppression Is Not The Primitive

Small 2x2 probe, `LOC=0`, `UNR=2`, K-major base with region-base split/memo:

| Variant | Correct | TFLOPS | inst/WMMA | wait/WMMA | global_b128/WMMA | ds_store_b128/WMMA | ds_load_b128/WMMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | yes | ~12.0 | 31.875 | 2.562 | 2.0 | 2.0 | 2.0 |
| Moved stage stores, no suppression | yes | ~9.0 | 39.562 | 4.062 | 3.125 | 3.125 | 2.0 |
| Moved stage stores, slot suppression | no, `rr=nan` | 0.0 | 30.938 | 2.562 | 1.625 | 1.625 | 2.0 |

Interpretation:

```text
moved stores themselves are semantically valid
destructive suppression is invalid at renderer-matcher scope
```

The failure mode is graph-rewrite side effects. `_dbuf_d3a_probe_marker` mutates renderer context while the graph rewriter is still free to visit nodes that do not survive into the final linear stream. Those dead moved-store candidates add slot keys to the stolen set. Later, live original stores with the same LDS slot can be suppressed even though the matching moved store is not reachable in the final stream.

Evidence:

```text
audit stolen candidates: 24 slots
original stage stores:   32 slots
final stream with suppression: 26 global_load_b128 / ds_store_b128
expected if replacement were exact: 32
```

The missing six moved stores are enough to corrupt the LDS tile. Adding an `s_barrier` after moved store groups fixed the wait shape but not correctness, proving the remaining issue is ownership/reachability, not only LDS visibility.

### Revised Primitive Boundary

Renderer-side stage stealing can remain as an additive probe only. It must not own destructive suppression.

The primitive fix must move one level earlier:

```text
postrange creates explicit stage ownership:
  original stage store either remains
  or is structurally replaced by a moved stage producer in the same graph

renderer lowers that already-owned graph:
  no global stolen-slot side effects
  no suppress-by-slot from speculative matcher visits
```

Required next implementation layer:

| Work | Success Condition |
| --- | --- |
| Replace side-effect stolen set with graph-local ownership metadata. | No context mutation determines whether a live store is deleted. |
| Suppress only the exact original store structurally paired to a reachable moved producer. | Final stream keeps the same producer count as base unless intentionally reduced. |
| Keep the moved-stage barrier in the owned graph. | `s_waitcnt lgkmcnt(0)` before `s_barrier`, then `ds_load_b128`. |
| Re-run 2x2 before active shapes. | Correct first, then density/perf. |

## 2026-07-08 Follow-Up: Full Moved Emission Is Correct But Not Useful

After adding a moved-store barrier and making moved-store memoization separately gated:

```text
PREFILL_WMMA_KMAJOR_STAGE_STEAL_MEMO=0
PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS=1
```

the destructive renderer probe becomes correct on all active shapes, but it is slower than the K-major baseline because
it still over-stages. This confirms that ordering can be made correct, but the renderer probe is not the primitive.

Active-shape comparison, `LOC=0`, `UNR=2`, region-base split/memo enabled:

| Shape | Route | Correct | TFLOPS | inst/WMMA | wait/WMMA | global_b128/WMMA | ds_store_b128/WMMA | ds_load_b128/WMMA |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x2 | K-major base | yes | 12.05 | 31.875 | 2.562 | 2.0 | 2.0 | 2.0 |
| 2x2 | renderer moved A+B, no memo | yes | 9.01 | 38.312 | 4.250 | 2.75 | 2.75 | 2.0 |
| 4x2 | K-major base | yes | 10.35 | 27.000 | 1.781 | 1.5 | 1.5 | 1.5 |
| 4x2 | renderer moved A+B, no memo | yes | 7.44 | 33.406 | 3.375 | 2.25 | 2.25 | 1.5 |
| 2x4 | K-major base | yes | 10.18 | 27.500 | 1.781 | 1.5 | 1.5 | 1.5 |
| 2x4 | renderer moved A+B, no memo | yes | 7.30 | 33.594 | 3.375 | 2.25 | 2.25 | 1.5 |

2x2 one-operand probes:

| Route | Correct | TFLOPS | inst/WMMA | wait/WMMA | global_b128/WMMA | ds_store_b128/WMMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| moved A only | yes | 10.15 | 34.938 | 3.312 | 2.375 | 2.375 |
| moved B only | yes | 9.46 | 35.188 | 3.312 | 2.375 | 2.375 |

Memo remains refuted:

| Variant | Result | Meaning |
| --- | --- | --- |
| `STAGE_STEAL_MEMO=1` | `WRONG rr=nan`, `global/store=1.625 per WMMA` | The memo key collapses non-equivalent producers or hides a required producer. |
| memo hit with fresh barrier | still wrong | The failure is not only missing `s_barrier`; it is ownership/key equivalence. |

Audit of the correct no-memo 2x2 probe:

```text
stolen candidates: A=24, B=24
each moved slot appears twice
original stage stores: 32
final stream: 44 global_load_b128 / 44 ds_store_b128
```

So the probe is now a correct over-stage, not a candidate performance fix.

### Updated Decision

Park renderer-stage stealing as a diagnostic only. It should remain behind explicit flags and must not be promoted.

The next primitive must be a real graph-level stage representation:

```text
Option C: postrange emits explicit stage pseudos / owned stage groups
```

Reason:

```text
plain post-stage stores do not preserve exact producer identity into isel_store
slot-based suppression can be made correct only by emitting extra producers
memo-based dedup needs a stronger key than the renderer can infer after lowering
```

The smallest next implementation should not try another renderer heuristic. It should introduce a graph-owned stage
object or equivalent metadata before packed-store rewriting, then lower exactly one owner per stage key.

## 2026-07-08 Phase-Key Probe: Correctness Is Key-Geometry Sensitive

`K-major 2x2, LOC=0, UNR=8` is the useful stress case:

| Phase Key | Result | Notes |
| --- | --- | --- |
| default `PREFILL_WMMA_PHASE_TILE_BYTES_{A,B}=128` | `WRONG rr=nan` | 64 WMMAs, producer/consumer windows line up, but resident key aliases an unsafe phase. |
| exact 32-byte window key | compile fails | Too many resident A/B fragments for the current `[40,200)` window. |
| 256-byte key | correct, ~4.5 TFLOPS | Fixes correctness for 2x2 `UNR=8`, but regresses/breaks `UNR=2` by over-coalescing loads. |
| 512-byte key | correct, ~4.5 TFLOPS | Same performance class as 256-byte key. |

This proves the K-major correctness problem is not simple LDS bounds or producer/consumer identity:

```text
full pre-isel audit for UNR=8:
  stores: 256
  WMMA operands: 128
  producer windows: 64
  consumer windows: 64
  intersection: 64
```

The problem is the resident fragment equivalence class. The key must be derived from the actual StageTile/DBUF geometry;
a fixed 128-byte or 256-byte heuristic is shape/unroll sensitive.

Decision:

```text
do not promote the 256-byte key as a global fix
keep K-major as a diagnostic branch
bank direct/register-resident schedule-table route for current E2E progress
```
