# Scope: Generated LDS Reuse/Cadence New Focus

Date: 2026-07-08.

## Status Override: 4x4 Parked On gfx1100

This scope has been narrowed. The generated `4x4` path is parked for gfx1100 because the register budget is the binding
resource: 128 VGPRs are consumed by C accumulators before A/B fragments, DBUF phase state, addresses, scratch, and the
epilogue. The active target is now `2x2`, `4x2`, and `2x4`.

Authoritative policy: `docs/gfx1100-4x4-path-parked-scope.md`.

Do not continue no-LDS `4x4` NaN isolation or generated LDS/DBUF `4x4` spill work unless the parked path is explicitly
reopened. The old `4x4` material below is retained as archive/debug context.

## Thesis

The register-resident gate answered the fork: pure register-resident native ISA is not enough to reach hand-class prefill
TFLOPS. Hand LDS2 is fast because it combines:

```text
LDS staging + explicit DBUF cadence + A/B fragment reuse + low wait/bookkeeping density
```

Therefore the generated route remains LDS-style reuse/cadence, but the work order changes:

```text
restore LDS staging on fitting shapes
then add proof-safe resident A/B reuse
then tune waits/scheduler
```

Do not optimize or diagnose the parked no-LDS `PREFILL_DBUF=1 u0=4,u1=4` route as part of the active path.

## Current Measurements

attn_qo shape:

```text
M=512, N=5120, K=5120
```

### Register-Resident Gate

Measured with:

```bash
DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=0 PREFILL_DBUF=0 \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes '2,2;4,2;2,4;4,4' --loc 2 --unr 2 --pin-clock --json
```

| shape | status | TFLOPS | structure |
| --- | --- | ---: | --- |
| `2x2` | ok | 21.5 | 8 WMMAs, direct global operands |
| `4x2` | ok | 15.7 | 16 WMMAs, direct global operands |
| `2x4` | ok | 10.0 | 16 WMMAs, direct global operands |
| `4x4` | no-spill fail | 0.0 | cannot compile full shape without newer machinery |

Measured with no LDS but `PREFILL_DBUF=1`:

| shape | status | TFLOPS | structure |
| --- | --- | ---: | --- |
| `2x2` | ok | 31.5 | 16 WMMAs, direct global operands |
| `4x2` | ok | 19.1 | 32 WMMAs, direct global operands |
| `2x4` | ok | 13.6 | 32 WMMAs, direct global operands |
| `4x4` | `WRONG rr=nan` | 0.0 | 64 WMMAs, direct global operands |

Conclusion:

```text
register-resident is a useful control, not the final path
```

### Hand LDS2 Advantage

| shape | path | TFLOPS | WMMAs | global b128 / WMMA | LDS load / WMMA | inst / WMMA | wait / WMMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2x2` | generated LDS/DBUF-safe | 8.0 | 16 | 2.0 | 4.0 | 39.1 | 3.31 |
| `2x2` | hand LDS2 | 36.3 | 32 | 1.0 | 2.0 | 12.8 | 0.56 |
| `4x2` | generated LDS/DBUF-safe | 14.7 | 32 | 1.5 | 4.0 | 35.4 | 2.66 |
| `4x2` | hand LDS2 | 45.6 | 64 | 0.75 | 1.5 | 10.5 | 0.28 |
| `2x4` | generated LDS/DBUF-safe | 14.9 | 32 | 1.5 | 4.0 | 34.4 | 2.66 |
| `2x4` | hand LDS2 | 37.5 | 64 | 0.75 | 1.5 | 10.5 | 0.28 |

Hand is faster because it amortizes staged A/B fragments over more WMMAs and keeps WMMA issue density high.

## Active 100% Definition

This focus is complete only when all gates pass:

| Gate | Requirement | Evidence |
| --- | --- | --- |
| G0. Parked path fenced | `4x4` is not part of default active generated development on gfx1100. | Active matrices use `--shapes '2,2;4,2;2,4'`; `4x4` requires explicit override/reopen. |
| G1. LDS restore | Generated LDS/DBUF-safe `2x2`, `4x2`, and `2x4` are all correct. | Same harness, all active-shape statuses `ok`. |
| G2. No scalar LDS fallback | Promoted LDS route uses packed `global_load_b128`, `ds_store_b128`, `ds_load_b128`; no scalar fragment LDS stores. | Structural counts show `ds_store_b16=0`, `ds_store_b32=0` for promoted staging. |
| G3. Proof-safe resident reuse | Generated A/B reuse is enabled only with slot/phase/epoch proof. | Fragment audit shows promotion-safe groups and no address-only reuse. |
| G4. Density improvement | `ds_load/WMMA` moves toward hand: `2x2 <= 2.0`, `4x2/2x4 <= 1.5` where feasible. | Structural matrix counters. |
| G5. Wait density improvement | Waits drop after reuse/cadence; scheduler tuning never precedes correctness/reuse. | `wait/WMMA` improves over DBUF-safe baseline. |
| G6. TFLOPS improvement | Generated beats current generated DBUF-safe and approaches hand in the same pinned-clock harness. | Repeated timing runs, same env. |
| G7. Default safety | Default behavior remains unchanged until correctness and perf are both proven. | Flags remain opt-in; tests pass; parked `4x4` is not selected. |

## Archived Phase 1: No-LDS `4x4` NaN Isolation

Status: parked. Do not run this phase for the active gfx1100 path.

### Baseline command

```bash
PYTHONPATH=. DEV=AMD:ISA \
AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=0 PREFILL_DBUF=1 REGALLOC_ADDR_REMAT=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes 4,4 --loc 2 --unr 2 --pin-clock --json
```

Current result:

```text
status = WRONG rr=nan
wmma_count = 64
global_load_b128 = 256
ds_store_b128 = 0
ds_load_b128 = 0
```

Because there is no LDS traffic, the candidate causes are limited.

### Candidate H1: accumulator mapping collision

Theory:

```text
64-WMMA DBUF chain maps two logical C subtiles to the same accumulator VGPR run,
or reads/writes the wrong accumulator slice during chain construction.
```

Probe:

```text
dump every v_wmma:
  wmma_idx
  logical subtile id if derivable
  src2/vdst register span
  upstream output store region
```

Expected failure signal:

```text
same accumulator span used for different logical output subtiles without intentional K-accumulation
or output stores read from a different span than the WMMA wrote
```

Acceptance:

```text
every logical output subtile has one stable accumulator span across K tiles,
and no two distinct output subtiles share the same span unless they are the same C element group
```

### Candidate H2: A/B operand mapping error

Theory:

```text
the 64-WMMA expanded chain uses the wrong A or B fragment for at least one subtile.
```

Probe:

```text
dump each WMMA operand:
  wmma_idx
  operand role A/B
  global pointer param
  dynamic expression key
  const byte/lane window
  logical output subtile coordinates if derivable
```

Expected failure signal:

```text
repeated A/B fragment where logical row/column should differ,
or missing/duplicated K phase fragments across the 64 WMMA chain
```

Acceptance:

```text
for each output subtile and K phase, A row and B column fragments match the expected cartesian product
```

### Candidate H3: DBUF chain expansion / reduction ordering

Theory:

```text
PREFILL_DBUF=1 doubles/peels/unrolls the chain such that WMMA accumulation order or C init is wrong
only at full 4x4.
```

Probe matrix:

| probe | expected |
| --- | --- |
| `2x2`, `4x2`, `2x4` with `PREFILL_DBUF=1` | ok |
| `4x4` with `PREFILL_DBUF=1 --unr 2` | NaN |
| `4x4` with `PREFILL_DBUF=1 --unr 4` | NaN |
| compare WMMA count and accumulator spans across above | first divergence identifies chain expansion boundary |

Acceptance:

```text
identify the exact expansion at which C init, prior-C source, or K-phase ordering diverges
```

### Candidate H4: output epilogue indexing

Theory:

```text
WMMA math may be correct but stores write to the wrong output offsets or overlap.
```

Probe:

```text
dump global_store_b16:
  store_idx
  source register span
  output byte address expression/range
  logical output element if derivable
```

Expected failure signal:

```text
duplicate output store addresses for distinct C lanes,
missing output windows,
or source accumulator span mismatch
```

Acceptance:

```text
output stores cover the expected tile exactly once and read from the matching accumulator span
```

## Phase 2: Restore LDS/DBUF On Active Shapes

Run on `2x2`, `4x2`, and `2x4`. Do not block this phase on parked `4x4`.

Re-enable:

```text
PREFILL_TC_LOCAL_STAGE=both
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1
PREFILL_TC_LOCAL_STAGE_POST=1
PREFILL_LDS_PACK_WITHLOCAL_B128=1
PREFILL_DBUF_LDS_CONST_IMM=1
PREFILL_DBUF_LDS_INDEX_SPLIT=1
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1
PREFILL_DBUF_DIRECT_B128_CHAIN=1
PREFILL_DBUF_LDS_ADDR_USE_DEP=1
REGALLOC_ADDR_REMAT=1
```

Run:

```bash
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env dbuf-safe --skip-hand --shapes '2,2;4,2;2,4' --pin-clock --json
```

If correctness fails only after LDS is restored, return to LDS-specific slot/store/load proof on the active shape that
failed.

## Phase 3: Proof-Safe Resident A/B Reuse

Current unsafe reuse result:

```text
PREFILL_WMMA_AB_ADDR_KEY=1
PREFILL_WMMA_CHAIN_AB_RESIDENT=1
```

Structural win:

```text
ds_load/WMMA: 4.0 -> 2.0
```

Correctness:

```text
WRONG rr=nan
```

Audit result:

```text
promotion_safe_group_count = 0
```

Required proof key:

```text
FragKey(
  role,
  lds_buffer_id,
  dbuf_slot,
  k_phase,
  logical_row_or_col,
  byte_start,
  byte_len,
  producer_epoch,
  overwrite_epoch,
)
```

Implementation order:

1. Carry metadata from postrange staging:

```text
role, nbuf, tile_count, tile_idx, tile_elems, kr, barrier/producer token
```

2. Preserve metadata to the WMMA operand carrier.
3. Make `_wmma_frag_reuse_key` return `None` unless all proof fields exist.
4. If proof is absent, fall back to current per-WMMA reload.
5. Re-run `wmma_frag_key_audit.py`.

## Phase 4: Waitcnt/Scheduler Tuning

Only after:

```text
G0 parked 4x4 fenced
G1 LDS restored correct on active shapes
G3 proof-safe reuse correct
```

Targets:

| metric | current generated | hand target |
| --- | ---: | ---: |
| `wait/WMMA` `2x2` | 3.31 | 0.56 |
| `wait/WMMA` `4x2/2x4` | 2.66 | 0.28 |
| `inst/WMMA` `2x2` | 39.1 | 12.8 |
| `inst/WMMA` `4x2/2x4` | 34-35 | 10.5 |

Scheduler tuning is a final density step, not a correctness mechanism.

## Work That Is Explicitly Out Of Scope For This Focus

- New int8/MMQ architecture.
- HIP schedule-side warmstart tuning.
- Handwritten kernel promotion.
- Scheduler/waitcnt tuning before active-shape correctness.
- Address-only resident reuse.
- No-LDS `4x4` NaN isolation on gfx1100.

## Immediate Next Task

Build/finalize active-shape proof-safe fragment metadata:

```text
WMMA operand -> role, slot, K phase, row/column identity, byte window, producer epoch, overwrite epoch
```

Then use it to enable resident A/B reuse only when the proof key is complete.
