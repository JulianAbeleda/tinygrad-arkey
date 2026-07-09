# Scope: generated WMMA fragment reuse to hand LDS2 density

Goal: make generated native-ISA LDS/DBUF reuse staged A/B fragments across subtiles the way `build_gemm_lds2` does.
The target is lower per-WMMA overhead, not another DS offset or waitcnt tweak.

## gfx1100 4x4 Policy

`4x4` is parked on gfx1100. This scope targets `2x2`, `4x2`, and `2x4`; do not make `4x4` correctness or spill-freedom a
gate for fragment reuse. See `docs/gfx1100-4x4-path-parked-scope.md`.

## Current measured blocker

`extra/qk/prefill/hand_vs_generated_shape_matrix.py` shows generated does too much staging work per useful WMMA.

For `M=512,N=5120,K=5120`, generated `2x2` DBUF currently emits:

| path | WMMAs | global_b128 | ds_store_b128 | ds_load_b128 | ds_load/WMMA | wait/WMMA | inst/WMMA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generated `2x2` | 16 | 32 | 32 | 64 | 4.0 | 3.31 | 39.1 |
| hand LDS2 `2x2` | 32 | 32 | 32 | 64 | 2.0 | 0.56 | 12.8 |

The important comparison is not just TFLOPS. Hand amortizes the same 32/32/64 staging bundle over twice the WMMA work.
Generated reloads A/B for every subtile.

## Negative controls already tested

| experiment | result | conclusion |
| --- | --- | --- |
| `LOC=0` vs `LOC=2` | same `ds_load/WMMA=4.0` | local shape knob does not create reuse |
| `UNR=4` | WMMAs increase, but staging increases proportionally | plain unroll duplicates work; no reuse |
| `UNR=8` | over 64 KiB LDS | not a viable path |
| materialized LDS offsets | more instructions, same memory density | DS offset folding is not the primitive perf fix |
| no DBUF | worse density | DBUF removal is not the fix |

## Experimental implementation status

Two default-off flags were added in `tinygrad/renderer/isa/amd.py`:

| flag | purpose | result |
| --- | --- | --- |
| `PREFILL_WMMA_AB_ADDR_KEY=1` | key resident A/B fragments by source address structure instead of carrier identity | no effect alone; current path is unrolled-chain, not rolled-resident |
| `PREFILL_WMMA_CHAIN_AB_RESIDENT=1` | make the unrolled WMMA-chain path allocate resident A/B fragments by reuse key | structural win but numerically wrong |

With both flags on:

| metric | before | experiment |
| --- | ---: | ---: |
| `ds_load_b128_count` | 64 | 32 |
| `ds_load/WMMA` | 4.0 | 2.0 |
| `wait/WMMA` | 3.31 | 2.50 |
| `inst/WMMA` | 39.1 | 34.25 |
| GPU correctness | ok | `WRONG rr=nan` |

This proves the lever is real: resident fragment reuse can move generated toward hand density. It is not yet safe.

## Current diagnosis

The unsafe experiment likely over-merges fragments whose contiguous address structure looks equivalent but whose logical
role/phase is not equivalent. The current key is too weak for promotion.

The primitive fix is a proof-based row/column grouping key:

```text
fragment_key =
  operand_role          # A or B
  dbuf_slot / K phase
  local tile identity
  row-or-column identity
  16-lane contiguous byte window
  barrier/order epoch
```

Only fragments with identical proof keys may share a resident A/B VGPR run.

## 100% Definition

1. Generated `2x2` with reuse is GPU-correct.
2. Generated `2x2` reaches `ds_load/WMMA <= 2.0` and keeps `ds_store_b128`/`global_load_b128` packed.
3. Generated `4x2` and `2x4` are GPU-correct and improve density versus current `ds_load/WMMA=4.0`.
4. Generated `4x4` remains parked and is not selected by default on gfx1100.
5. The reuse key is fail-closed: if role/slot/phase/epoch cannot be proven, it falls back to current per-WMMA reloads.
6. The default path remains unchanged until correctness and TFLOPS both improve.

## Next Work

1. Add a fragment-key dump for every WMMA operand before reuse.
2. Compare keys for current generated `2x2` against hand LDS2 row/column expectations.
3. Strengthen `PREFILL_WMMA_AB_ADDR_KEY` to include operand role, DBUF slot, K phase, and barrier/order epoch.
4. Re-run the resident-chain experiment.
5. Promote only if correctness passes and density remains near hand.

## Proof-safe key design

The failed experiment is valuable because it lowered `ds_load/WMMA` from `4.0` to `2.0`, exactly the intended density
move, but produced `WRONG rr=nan`. That means the physical reuse mechanism works, while the equivalence key is too weak.

The safe key must answer one question:

```text
May these two WMMA operands legally read the same already-loaded 8-VGPR fragment?
```

The answer is yes only if every field below matches.

| Field | Why it is required | If missing |
| --- | --- | --- |
| `operand_role` = A or B | A row fragments and B column fragments can have similar byte windows but different WMMA operand interpretation. | A/B cross-merge corrupts multiply inputs. |
| `buffer_slot` | DBUF has at least two LDS slots; same offset shape in slot 0 vs slot 1 is not the same live data. | Reads stale/future K-block data. |
| `k_phase` / reduce step | Same row/column in different K tile is different data. | Reuses previous K fragment. |
| `logical_row_or_col` | Hand reuse is row-wise for A and col-wise for B, not arbitrary address equality. | Reuses across output subtiles that need different row/column fragments. |
| `contiguous_byte_window` | The 16 half lanes must be exactly the same 32-byte logical fragment. | Partial fragment alias. |
| `producer_epoch` | The LDS stores feeding the fragment must dominate all reusing WMMAs and be separated by the right barrier. | Reuse before producer or after overwrite. |
| `consumer_epoch` | A resident fragment cannot be overwritten before all consumers have read it. | WAR corruption between WMMAs. |

Minimal key shape:

```text
FragKey(
  role: "A" | "B",
  lds_buffer_id,
  dbuf_slot,
  k_phase,
  row_or_col_id,
  byte_start,
  byte_len = 32,
  producer_barrier_epoch,
  overwrite_epoch,
)
```

The current experimental key is closer to:

```text
(pointer_identity, dynamic_address_expr, first_const_lane)
```

That explains the failure: it proves contiguity, but not role, slot, K phase, logical row/col identity, or lifetime epoch.

## Implementation phases

### Phase 1: key dump, no behavior change

Add `PREFILL_WMMA_FRAG_KEY_DUMP=1` and print one row per WMMA operand:

```text
wmma_idx, operand_role, carrier_id, pointer_id, dynamic_expr_key,
first_const_lane, byte_window, inferred_slot, inferred_k_phase,
barrier_epoch, key_or_reason_unprovable
```

100% gate:

- Default codegen unchanged.
- Dump shows generated `2x2` currently has 16 WMMAs but more than the hand-expected A/B groups.
- Any unprovable field is explicit, not silently omitted.

### Phase 2: offline grouping audit

Build grouping summaries from the dump:

```text
for each role:
  groups_by_current_id_key
  groups_by_address_key
  groups_by_proof_key
  consumers_per_group
```

Expected target for generated `2x2`:

```text
current:
  ds_load_b128 = 64
  WMMAs = 16
  ds_load/WMMA = 4.0

proof-safe reuse target:
  ds_load_b128 = 32
  WMMAs = 16
  ds_load/WMMA = 2.0
```

The audit must explain exactly which pairs the unsafe address key merged that the proof key rejects.

### Phase 3: proof-keyed resident reuse

Replace `PREFILL_WMMA_CHAIN_AB_RESIDENT`'s current reuse key with the proof key. Fail closed:

```text
if proof_key is None:
  use current per-WMMA reload path
else:
  use resident fragment keyed by proof_key
```

100% gate:

- Generated `2x2` correctness `ok`.
- `ds_load/WMMA <= 2.0`.
- `wait/WMMA` and `inst/WMMA` improve versus baseline.

### Phase 4: shape expansion

Run:

```bash
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py --generated-env current --skip-hand --shapes 2,2 --pin-clock --json
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py --generated-env current --skip-hand --shapes 4,2 --pin-clock --json
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py --generated-env current --skip-hand --shapes 2,4 --pin-clock --json
```

100% gate:

- `2x2`, `4x2`, `2x4` all correct.
- All three improve `ds_load/WMMA` versus current.
- No shape regresses to scalar LDS stores.

### Phase 5: promotion decision

Only after correctness and density pass:

- compare TFLOPS over repeated pinned runs,
- keep default off until speed improves beyond noise,
- document rollback flags.

## Current best next step

Do **not** keep tuning `UNR`, `LOC`, or DS offset folding. The next productive action is Phase 1: add a fragment-key dump
and use it to identify exactly which unsafe merge caused the NaN.

## Phase 1/2 Result: fragment-key audit

Added:

```bash
python3 extra/qk/prefill/wmma_frag_key_audit.py --m-up 1
```

The tool runs the existing centralized `native_isa_l4_stream_probe.py` with:

```text
PREFILL_WMMA_AB_ADDR_KEY=1
PREFILL_WMMA_CHAIN_AB_RESIDENT=1
PREFILL_WMMA_FRAG_KEY_DUMP=1
```

and groups the emitted `WMMA_FRAG_KEY_JSON` rows.

Observed audit:

| signal | value |
| --- | ---: |
| fragment rows | 16 |
| roles | A=8, B=8 |
| reuse groups | 10 |
| reused groups | 2 |
| promotion-safe groups | 0 |

The only reused groups are B fragments:

| role | consumers | carriers | tiles | const byte starts | safe |
| --- | ---: | ---: | --- | --- | --- |
| B | 4 | 1 | `[1]` | `[8192]` | false |
| B | 4 | 1 | `[0]` | `[10240]` or `[8704]` depending probe shape | false |

Interpretation:

- The unsafe experiment is mostly proving B-column resident reuse, which is directionally the hand-LDS2 behavior.
- The current key proves contiguous LDS byte windows, but it cannot prove `dbuf_slot`, `k_phase`, producer barrier epoch,
  or overwrite epoch.
- A/B role is present at the `_ab_base(("A"|"B", key))` allocation call site, so direct A/B cross-merge is unlikely to be
  the observed corruption. The reusable primitive itself is still role-less and should not be promoted.

Negative ordering test:

```bash
PREFILL_DBUF_LDS_LOAD_SERIAL=1 \
PREFILL_WMMA_AB_ADDR_KEY=1 \
PREFILL_WMMA_CHAIN_AB_RESIDENT=1 \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes 2,2 --pin-clock --json
```

Result:

```text
status = WRONG rr=nan
ds_load/WMMA = 2.0
```

So the NaN is not fixed by simply serializing the LDS fragment loads. The blocker is the missing semantic proof for
when a B fragment may remain resident across consumers.

## Next Primitive Step

Carry proof metadata from the staging site into the WMMA operand carrier. `postrange.py` already computes the semantic
pieces:

```text
nbuf = PREFILL_DBUF_NBUF()
kr = prefill_dbuf_reduce_range(src.ranges)
tile_count = prod(tile_ranges)
tile_idx = ...
slot = ((kr % nbuf) * tile_count + tile_idx) * tile_elems
```

but AMD isel currently receives only the arithmetic expression. The fail-closed fix is:

1. Tag the cooperative LDS load/index carrier with `role`, `nbuf`, `tile_count`, `tile_idx`, `tile_elems`, `kr`, and
   a barrier/producer epoch token at the postrange staging site.
2. Preserve that metadata through expansion/devectorization for the scalar load feeding the WMMA `CONTRACT`.
3. Replace `_wmma_frag_addr_key` with a proof-key helper that returns non-`None` only when the metadata is present:

```text
FragKey(role, lds_buffer_id, dbuf_slot, k_phase, logical_row_or_col, byte_start, byte_len, producer_epoch)
```

4. If the metadata is absent, fall back to the current per-WMMA reload path; do not use address-only reuse.
