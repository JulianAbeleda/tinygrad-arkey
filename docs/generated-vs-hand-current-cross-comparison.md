# Generated vs Hand Current Cross-Comparison

Date: 2026-07-08.

## Direction Change

The generated `4x4` path is now parked on gfx1100. The current comparison matrix remains useful as evidence, but the
active development target is `2x2`, `4x2`, and `2x4` with LDS/DBUF reuse and cadence. See
`docs/gfx1100-4x4-path-parked-scope.md`.

## Current Correctness Boundary

The generated prefill schedule-search `4x4` case means `u0=4,u1=4` in
`extra/qk/prefill/hand_vs_generated_shape_matrix.py`. This is not the same thing as the small
`native_isa_l4_stream_probe.py --m-up 2` structural route.

Confirmed generated boundary:

| env | shape | result | key structure |
| --- | --- | --- | --- |
| no local stage, `PREFILL_DBUF=0` | `2x2` | ok | 8 WMMAs, direct global operands |
| no local stage, `PREFILL_DBUF=0` | `4x2` | ok | 16 WMMAs, direct global operands |
| no local stage, `PREFILL_DBUF=0` | `2x4` | ok | 16 WMMAs, direct global operands |
| no local stage, `PREFILL_DBUF=0` | `4x4` | compile failure | full 4x4 cannot compile without newer machinery |
| no local stage, `PREFILL_DBUF=1` | `2x2` | ok | 16 WMMAs, direct global operands |
| no local stage, `PREFILL_DBUF=1` | `4x2` | ok | 32 WMMAs, direct global operands |
| no local stage, `PREFILL_DBUF=1` | `2x4` | ok | 32 WMMAs, direct global operands |
| no local stage, `PREFILL_DBUF=1` | `4x4` | `WRONG rr=nan` | 64 WMMAs, direct global operands |
| DBUF-safe LDS staging | `4x4` | `WRONG rr=nan` | 64 WMMAs, 64 global loads, 64 LDS stores, 256 LDS loads |

Toggles that do not fix DBUF-safe `4x4`:

| toggle | result |
| --- | --- |
| `AMD_ISA_SCHED=0` | still NaN |
| `AMD_ISA_WAITCNT_TARGETED=0` | still NaN |
| `PREFILL_DBUF_LDS_CONST_IMM=0` | still NaN |
| `PREFILL_DBUF_LDS_INDEX_SPLIT=0` | still NaN |
| `PREFILL_DBUF_LDS_STORE_BASE_SPLIT=0` | still NaN |
| `PREFILL_DBUF_DIRECT_B128_CHAIN=0` | still NaN |

Interpretation:

The immediate generated `4x4` correctness bug is not caused by LDS slot addressing, DS immediate folding, scheduler motion,
or targeted waitcnt. The failure already exists in the no-LDS `PREFILL_DBUF=1` direct-global `u0=4,u1=4` route. The bug is
therefore tied to the DBUF-expanded 64-WMMA chain plus full 4x4 accumulator/output shape.

## Why Hand LDS2 Is Fast

Hand LDS2 has three structural advantages.

### 1. More useful WMMAs per staging bundle

For the same shape family:

| shape | path | WMMAs | global b128 | LDS stores | LDS loads | LDS loads / WMMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2x2` | generated DBUF-safe | 16 | 32 | 32 | 64 | 4.0 |
| `2x2` | hand LDS2 | 32 | 32 | 32 | 64 | 2.0 |
| `4x2` | generated DBUF-safe | 32 | 48 | 48 | 128 | 4.0 |
| `4x2` | hand LDS2 | 64 | 48 | 48 | 96 | 1.5 |
| `2x4` | generated DBUF-safe | 32 | 48 | 48 | 128 | 4.0 |
| `2x4` | hand LDS2 | 64 | 48 | 48 | 96 | 1.5 |

Hand is not merely better encoded. It does less staging work per WMMA.

### 2. Explicit DBUF cadence

Hand LDS2 trace shows:

```text
global_load_b128 -> ds_store_b128 -> barrier
ds_load_b128 -> WMMA clusters
next-slot global/LDS staging between WMMA groups
```

The lifecycle tracer marks hand LDS2 as:

| gate | hand LDS2 |
| --- | --- |
| two-slot identity | ok |
| body next-slot work | ok |
| WMMA operands from LDS | ok |
| scalar LDS stores | 0 |

Generated can show structural DBUF in the LDS route, but still reloads A/B per subtile and lacks proof-safe resident
fragment reuse.

### 3. Far less bookkeeping per WMMA

Representative per-WMMA counters:

| shape | path | inst / WMMA | wait / WMMA | memops / WMMA |
| --- | --- | ---: | ---: | ---: |
| `2x2` | generated DBUF-safe | 39.1 | 3.31 | 10.0 |
| `2x2` | hand LDS2 | 12.8 | 0.56 | 5.0 |
| `4x2` | generated DBUF-safe | 35.4 | 2.66 | 9.0 |
| `4x2` | hand LDS2 | 10.5 | 0.28 | 4.0 |
| `2x4` | generated DBUF-safe | 34.4 | 2.66 | 9.0 |
| `2x4` | hand LDS2 | 10.5 | 0.28 | 4.0 |

This explains the TFLOPS gap: hand keeps WMMA issue density high by amortizing staging, reducing waits, and avoiding
generated address/bookkeeping work.

## Archived Diagnostic Target

This was the previous next target. It is now parked for gfx1100 unless `4x4` is explicitly reopened:

```bash
PREFILL_TC_LOCAL_STAGE=0 PREFILL_DBUF=1 \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes 4,4 --loc 2 --unr 2 --pin-clock --json
```

That route has no LDS stores or loads, so any failure there is in the DBUF-expanded WMMA chain, accumulator mapping,
global operand mapping, or epilogue/output indexing.
