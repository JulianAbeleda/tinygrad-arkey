# 8B Prefill Epoch-Aware DBUF Pipeline Construction Scope

Date: 2026-07-09.

## Big Picture

S10 lost the S9 hand-LDS2 win when `ffn_gate_up` moved from the hand-shaped LDS2 lifecycle to generated LDS/DBUF
ownership. The generated path can show either K-major fragment reuse or D3/body staging, but the current combined
paths either duplicate too much work or corrupt output when they suppress originals.

The next primitive is therefore not generic D3, and not a stronger after-the-fact suppress predicate. It is:

```text
construct the DBUF pipeline with explicit warmup/body epochs
so prologue stores feed only warmup consumers
and body stores feed steady-state rotated consumers
```

## Current Facts

| Route | TFLOPS / tok/s evidence | inst/WMMA | wait/WMMA | global/WMMA | store/WMMA | load/WMMA | barriers | D3 | cluster | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| S9 hand LDS2 whole-prefill | pp512 `4413`, pp4096 `3237` | hand-like | low | amortized | amortized | amortized | low | true | high | ok |
| S10 pure generated baseline | pp512 `1629.74`, pp4096 `1420.14` | high | moderate | high | n/a | n/a | low | partial cadence | low | ok |
| S10 composed generated primitive | pp512 `1493.4`, pp4096 `1325.35` | high | high | partial | partial | partial | high-ish | incomplete | low/partial | ok |
| generated K-major `2x2` | `12.51` bounded | `34.625` | `2.875` | `2.0` | `2.0` | `2.0` | `2` | false | `3` | ok |
| K-major + D3 stage steal | `10.33` bounded | `42.500` | `4.562` | `3.125` | `3.125` | `2.0` | `17` | true | `3` | ok, too heavy |
| `(slot,value)` suppress | n/a | `542 inst total` | `3.062` | `1.625` | `1.625` | `2.0` | `17` | true | `3` | `WRONG rr=nan` |

The failed `(slot,value)` suppress is decisive: matching the LDS slot and value source is still not enough. Store
deletion is a reaching-definitions problem, not a value-equality problem. A moved body store must reach every consuming
`ds_load_b128` first, on every path, with the same runtime epoch value and an intervening barrier.

## Failure Model

Current unsafe shape:

```python
moved = emit_stage_store(original_store, dep=phase_dep)
record(("stage_epoch", lds_slot, value_source_key))

if original_store_key in moved_stage_epochs:
  suppress_original_store()
```

Why this can corrupt:

```text
same slot + same value-looking source != same required timing point
```

A moved store can be correct for a later consumer while the original prologue store is still required for an earlier
consumer. Suppressing the original by value/window deletes a required producer and produces NaN.

## Primitive Fix

Build epochs correctly first; suppression is only an optional cleanup:

```python
distance = 1

# warmup: only the first distance epochs
for k in range(distance):
  store LDS[slot(k)] = value(k)
barrier()

for k in range(0, K):
  frag = load LDS[slot(k)]
  wmma(frag)

  # steady state: produce future epochs only
  if k + distance < K:
    store LDS[slot(k + distance)] = value(k + distance)
    barrier()
```

If we keep a cleanup pass, its invariant is:

```text
A store may be deleted only if every load it reaches is also reached first, on all paths, by an equivalent store of the
same runtime epoch value, with a synchronization point between replacement store and load.
```

In practice, most true prologue stores should remain: they exist to feed warmup consumers before the body store exists.

## Code Landmarks

| Area | File/function | Reason |
| --- | --- | --- |
| Stage tags | `tinygrad/codegen/opt/postrange.py::_tc_local_stage_coop_operand` | Emits `tc_local_stage_store` tags on LDS stage stores. |
| Stage candidate lookup | `tinygrad/renderer/isa/amd.py::_dbuf_stage_candidates` | Finds original stores feeding a WMMA carrier. |
| D3 movement | `amd.py::_dbuf_d3a_probe_marker` | Emits moved body stores and records stolen keys. |
| Current suppress sites | `amd.py::isel_store`, `amd.py::isel_gated_store` | Suppresses originals under broad or `(slot,value)` flags. |
| Gate extraction | `extra/qk/prefill/native_isa_l4_stream_probe.py::_dbuf_gate_summary` | D3/D7 structural truth source. |
| Lifecycle tracer | `extra/qk/prefill/kernel_lifecycle_trace.py` | Existing no-GPU final-stream audit. |
| Pipeline audit | `kernel_lifecycle_trace.py::dbuf_pipeline_construction_audit` | Classifies prologue/body physical LDS store overlap as pipeline rotation, not redundancy proof. |

## Done Definition

The path is complete only if one opt-in generated `2x2` route satisfies all of:

| Gate | Required |
| --- | --- |
| G0 correctness | bounded generated matrix `status == ok`, finite output, no NaN. |
| G1 D3 | `D3_cadence.ok=true` and `body_has_next_slot_work=true`. |
| G2 reuse | `ds_load_b128/WMMA <= 2.0`. |
| G3 no duplicate transport | `global_b128/WMMA <= 2.25` and `ds_store_b128/WMMA <= 2.25`. |
| G4 barriers | total barriers `<= 3`; explicitly not the old stage-steal `17`. |
| G5 clustering | max WMMA cluster `>= 3`, target `4`. |
| G6 no density regression | `inst/WMMA` and `wait/WMMA` no worse than K-major+D3 steal; target better than K-major alone. |
| G7 timing | bounded TFLOPS beats K-major `12.51`; otherwise it is structurally interesting but not useful enough. |

Only after G0-G7 pass should this transfer to composed whole-prefill.

## Phase Plan

### P0. Bank The Baseline Rows

Re-run or reference existing artifacts for:

```text
K-major
K-major + D3 stage steal
(slot,value) suppress wrong-output row
hand LDS2 structural oracle
```

Stop/go:

```text
go if the rows reproduce the known pattern
stop if the current tree no longer reproduces the wrong suppress row, because the suppression premise changed
```

### P1. Pipeline Construction Audit

Use the lifecycle trace's `dbuf_pipeline_construction_audit` first:

```text
classify DS stores by prologue/body/tail
record physical LDS windows present in both prologue and body
record body loads before the first body store
```

Stop/go:

```text
go if the audit shows prologue/body physical-window overlap and body loads before body stores
stop if there is no overlap; then duplicate traffic is coming from another source
```

### P2. Epoch-Aware Pipeline Construction

Add a new opt-in construction flag; leave old suppress flags reproducible:

```text
PREFILL_WMMA_KMAJOR_PIPELINE_EPOCHS=1
```

Rules:

```text
prologue emits warmup epochs only
body emits future/steady-state epochs only
slot/value equality never authorizes deletion
cleanup suppression, if any, must be MemorySSA/reaching-def safe
```

### P3. Structural Gate

Use the existing lifecycle tracer, not a new harness:

```bash
DEV=AMD:ISA PYTHONPATH=. \
PREFILL_WMMA_KMAJOR_PHASE=1 \
PREFILL_WMMA_AB_PROOF_KEY=1 \
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1 \
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1 \
PREFILL_WMMA_KMAJOR_D3A_MARKER=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL_MEMO=1 \
PREFILL_WMMA_KMAJOR_PIPELINE_EPOCHS=1 \
python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --active-generated --kind generated --shapes 2,2 \
  --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --target AMD:ISA:gfx1100 --json
```

Pass:

```text
D3 true, load/WMMA <= 2.0, global/store <= 2.25, barriers <= 3, cluster >= 3
```

### P4. Bounded Timing

Use the existing matrix harness:

```bash
DEV=AMD:ISA PYTHONPATH=. \
PREFILL_WMMA_KMAJOR_PHASE=1 \
PREFILL_WMMA_AB_PROOF_KEY=1 \
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1 \
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1 \
PREFILL_WMMA_KMAJOR_D3A_MARKER=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL=1 \
PREFILL_WMMA_KMAJOR_STAGE_STEAL_MEMO=1 \
PREFILL_WMMA_KMAJOR_PIPELINE_EPOCHS=1 \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 \
  --skip-hand --hand-reps 1 --hand-iters 1 --json
```

Pass:

```text
status == ok
TFLOPS > 12.51
```

### P5. Whole-Prefill Transfer

Only after P3/P4:

```text
enable the dominated suppress path only in the composed generated route
run per-role correctness/timing
run whole-prefill with --require-route prefill_wmma_pipe_lds_dbuf_primitive_generated
```

Pass:

```text
whole-prefill beats Path1 or records a named residual bottleneck
```

## Stop Conditions

This path is blocked only if all are true:

- warmup/body epochs cannot be represented in the current D3 stage construction;
- the pipeline construction audit shows no way to distinguish warmup producers from steady-state producers;
- every construction attempt either corrupts output or keeps barriers/stores above the gate.

Until then, the next action is pipeline construction, not broad scheduler tuning and not equality-key suppression.
