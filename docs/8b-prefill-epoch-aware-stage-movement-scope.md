# 8B Prefill Epoch-Aware Stage Movement Scope

Date: 2026-07-09.

## Goal

Make the generated LDS/DBUF prefill primitive perform the same lifecycle compression that the hand LDS2 oracle gets:

```text
body DBUF staging + K-major fragment reuse + WMMA clustering
without duplicate staging traffic
without deleting required phase producers
```

This is the primitive needed after the lifecycle compression audit:

```text
generated K-major:              12.51 TFLOPS, D3=false, ds_load/WMMA=2.0
generated K-major + D3 steal:   10.33 TFLOPS, D3=true,  ds_load/WMMA=2.0, but duplicate stores/barriers
hand LDS2 oracle:               inst/WMMA=9.547, wait/WMMA=0.406, global/store/load=1/1/2
```

The next fix is not "add D3." D3 exists. The fix is to make D3 movement lean and safe.

## Current Failure

The current broad suppression rule is effectively:

```python
if lds_slot in stolen_stage_stores:
  suppress_original_store()
```

That is unsafe because DBUF reuses LDS slots across K phases:

```text
slot 16, epoch 0: producer needed before first WMMA
slot 16, epoch 1: producer moved into body
slot 16, epoch 2: later producer
```

Suppressing by slot alone can delete epoch 0. The bounded run then returns `WRONG rr=nan`.

Observed broad suppression result:

```text
global/WMMA: 3.125 -> 1.625
store/WMMA:  3.125 -> 1.625
status:      WRONG rr=nan
```

Audit reason:

```text
stage stealing records stolen windows,
but late original stores mostly have key=None and only an absolute LDS slot.
```

## Existing Code Landmarks

| Layer | File/function | Current role |
|---|---|---|
| Stage/proof tags | `tinygrad/codegen/opt/postrange.py::_wmma_frag_proof_tag` | Builds proof tags with `producer_epoch` and `overwrite_epoch`. |
| Cooperative LDS store tags | `postrange.py::_tc_local_stage_coop_operand` | Tags stage store indices as `("tc_local_stage_store", operand_idx, lds_buffer_id, stage_store_i)`. |
| K-major reuse key | `tinygrad/renderer/isa/amd.py::_wmma_frag_phase_reuse_key` | Groups LDS fragments by role/buffer/phase row or col. |
| D3 movement | `amd.py::_dbuf_d3a_probe_marker` | Emits moved stage stores before later WMMAs. |
| Stage candidate lookup | `amd.py::_dbuf_stage_candidates` | Finds stores feeding a WMMA carrier. |
| Current stolen set | `amd.py::_dbuf_d3a_probe_marker` | Adds `id(cand)`, `skey`, and `("lds_slot", abs_slot)`. |
| Current unsafe suppress | `amd.py::isel_store`, `amd.py::isel_gated_store` | Suppresses by `id`, stage key, address key, or absolute slot. |

## Required Primitive

Replace slot-only suppression with epoch-aware suppression:

```python
stage_epoch_key = (
  lds_buffer_id,
  lds_slot_or_window,
  producer_epoch,
  overwrite_epoch,
)

if stage_epoch_key in moved_stage_epochs:
  suppress_original_store()
else:
  keep_original_store()
```

The key point:

```text
same LDS slot + different producer epoch != duplicate
same LDS slot + same producer epoch == duplicate
```

## Definition Of Done

The primitive is complete only when this table improves:

| Gate | Required |
|---|---|
| Correctness | bounded generated matrix status `ok`; no NaN. |
| D3 cadence | `D3=true`, `body_has_next_slot_work=true`. |
| Fragment reuse | `ds_load_b128/WMMA <= 2.0`. |
| Clustering | max WMMA cluster `>= 3`, target `4`. |
| Store density | `global_b128/WMMA <= 2.25` and `ds_store_b128/WMMA <= 2.25`. |
| Barrier density | close to baseline total `2`, not stage-steal total `17`. |
| Timing | bounded TFLOPS exceeds generated K-major `12.51`, not merely K-major+D3 steal `10.33`. |

Only then run whole-prefill.

## Phase Plan

### P0. Bank Baselines

Use the existing audit rows:

| variant | TFLOPS | inst/WMMA | wait/WMMA | global/WMMA | store/WMMA | load/WMMA | D3 | cluster |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| generated K-major | 12.51 | 34.625 | 2.875 | 2.0 | 2.0 | 2.0 | false | 3 |
| K-major + D3 stage steal | 10.33 | 42.500 | 4.562 | 3.125 | 3.125 | 2.0 | true | 3 |
| hand LDS2 oracle | structural | 9.547 | 0.406 | 1.0 | 1.0 | 2.0 | true | 4 |

Done when:

- these rows are recorded in `docs/8b-prefill-lifecycle-compression-audit-20260709.md`;
- no new e2e run is used as a substitute for this bounded gate.

### P1. Preserve Epoch Metadata To Suppression Point

Problem:

```text
postrange stage/proof metadata exists early,
but late original stores often arrive at suppression with key=None.
```

Implementation candidates, in order:

1. Extend the stage-store tag so it includes a stable epoch token:

```python
("tc_local_stage_store_epoch",
  operand_idx,
  lds_buffer_id,
  stage_store_i,
  producer_epoch,
  overwrite_epoch)
```

2. When lowering packed stores, propagate this tag through:

```text
INDEX -> STORE -> packed GROUP -> DS_STORE_B128 carrier
```

3. If the original store cannot retain a tag, compute an epoch key at candidate selection time and store it in a side map
   keyed by `id(cand)` and by the emitted moved store.

Done when:

- `DBUF_D3A_AUDIT_LOG` shows non-`None` epoch keys for both:
  - stolen moved stores,
  - original stores considered for suppression.

### P2. Replace Slot Suppression With Epoch Suppression

Add a new flag; do not mutate the old broad suppress behavior silently:

```text
PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS_EPOCH=1
```

Rules:

```python
if exact_epoch_key(original_store) in moved_epoch_keys:
  suppress
else:
  keep
```

Forbidden:

```python
if ("lds_slot", slot) in moved:
  suppress
```

Done when:

- broad suppress can still reproduce the old wrong result under its old flag;
- epoch suppress is separately opt-in and produces correct output.

### P3. Bounded Structural Gate

Run:

```bash
DEV=AMD:ISA ... \
PREFILL_WMMA_KMAJOR_PHASE=1 \
PREFILL_WMMA_AB_PROOF_KEY=1 \
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1 \
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1 \
PREFILL_WMMA_KMAJOR_D3A_MARKER=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL_MEMO=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS_EPOCH=1 \
PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --active-generated --kind generated --shapes 2,2 \
  --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --target AMD:ISA:gfx1100 --json
```

Pass condition:

```text
D3=true
ds_load_b128/WMMA <= 2.0
global_b128/WMMA <= 2.25
ds_store_b128/WMMA <= 2.25
barriers <= 3 total
max_cluster >= 3
```

### P4. Bounded Timing/Correctness Gate

Run:

```bash
DEV=AMD:ISA ... PYTHONPATH=. python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --skip-hand --hand-reps 1 --hand-iters 1 --json
```

Pass condition:

```text
status == ok
TFLOPS > 12.51
```

If TFLOPS is between `10.33` and `12.51`, the primitive is correct but not useful enough.

### P5. Route Transfer

Only after P3/P4:

1. enable the epoch suppress path only for the composed generated route;
2. run per-role correctness/timing via `prefill_pipe_mvp_artifact.py`;
3. run `prefill_whole_synced.py --require-route prefill_wmma_pipe_lds_dbuf_primitive_generated`.

Pass condition:

```text
whole prefill beats Path1 and records the composed route binding.
```

## Parallel Work

| Lane | Parallel? | Output |
|---|---:|---|
| A. Metadata propagation audit | yes | Where epoch tag disappears: postrange, devectorizer, pre-isel pack, or isel. |
| B. Suppression key design | yes | Exact key tuple and matching rules. |
| C. Trace gate test | yes after key design | Unit/trace assertion for D3 + density thresholds. |
| D. Implementation | after A/B | `SUPPRESS_EPOCH` opt-in path. |
| E. Timing/e2e transfer | after D | bounded timing, then whole-prefill. |

## Stop Condition

Call this path blocked only if:

- the epoch tag is proven unavailable before isel,
- a side map keyed at `_dbuf_stage_candidates` cannot identify original stores at suppression,
- and every suppress candidate either corrupts output or fails to reduce store/barrier density.

Until then, the path is clear: make stage suppression producer-epoch aware.

## P1/P2 Candidate Result - 2026-07-09

First candidate tested:

```text
PREFILL_WMMA_KMAJOR_STAGE_STEAL_SUPPRESS_EPOCH=1
```

Implementation shape:

```python
stage_epoch_key = ("stage_epoch", absolute_lds_slot, value_source_key)
```

Where `value_source_key` is the source identity of the packed `global_b128` stage value. The audit showed:

```text
stolen candidates: 24 with non-None value keys
late original stores: 32 with non-None value keys
intersection: 24 exact (slot, value_source_key) pairs
```

So this was a real test of a stronger key than slot-only suppression.

Result:

| variant | inst | wait/WMMA | global/WMMA | store/WMMA | load/WMMA | barriers | D3 | cluster | bounded status |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| K-major | 554 | 2.875 | 2.000 | 2.000 | 2.000 | 2 | false | 3 | ok, `12.51 TFLOPS` |
| K-major + D3 stage steal | 680 | 4.562 | 3.125 | 3.125 | 2.000 | 17 | true | 3 | ok, `10.33 TFLOPS` |
| `SUPPRESS_EPOCH=(slot,value)` | 542 | 3.062 | 1.625 | 1.625 | 2.000 | 17 | true | 3 | wrong, `rr=nan` |

Conclusion:

```text
(slot, value_source_key) is still not a sufficient producer epoch.
```

This means the moved D3 stage stores are not yet a safe replacement for the original prologue stores. The issue is not
just that suppression matched too broadly by slot. Even suppressing the exact moved value/window corrupts output.

Updated diagnosis:

```text
Before suppressing originals, prove the moved body store is self-sufficient:
  same value,
  same LDS slot,
  same required timing point,
  correct barrier/wait ordering before the consuming ds_load,
  not merely "a duplicate-looking store exists later."
```

Next candidate should therefore add a moved-store self-sufficiency proof before suppression:

```text
moved_store_epoch_key = (
  lds_slot,
  value_source_key,
  first_consuming_wmma_or_phase,
  dependency_anchor_before_consumer,
)
```

The suppress rule must require that the moved store dominates the consuming `ds_load_b128` in the final stream. If the
only proof is the source value and slot, keep the original store.
