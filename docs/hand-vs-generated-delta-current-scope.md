# Hand vs Generated Delta Current Scope

Date: 2026-07-08.

## Goal

Use hand LDS2 as a measurement reference for the active generated `2x2` prefill route, not as a permanent handwritten
kernel. The question is:

```text
What exact lifecycle / instruction / wait / staging differences explain why hand LDS2 is fast while generated DBUF/D3A is not?
```

The answer must narrow the next implementation lever. It is not enough to say "hand asm is better"; the output must name
which machine-code property generated code should copy or avoid.

## Shared Floor

Both routes ultimately become AMD RDNA3 instruction lists:

| Path | Tail-off before shared floor |
|---|---|
| generated | graph/UOps -> isel -> regalloc -> scheduler/waitcnt -> final `Inst` list |
| hand LDS2 | Python fixed instruction builder -> final `Inst` list |

The shared floor is:

```text
Inst list -> assemble_linear -> ELF -> HSA launch -> GPU
```

So the comparison must happen at final instruction stream level, not at high-level graph names.

## Current Known Facts

From current runs and docs:

| Fact | Current evidence |
|---|---|
| Generated baseline DBUF beats generated D3A on the bounded worker. | `loc=2,unr=2`: baseline about `7.6-7.9 TFLOPS`, D3A about `7.0 TFLOPS`. |
| D3A wait serialization is real but incomplete. | `AMD_ISA_WAITCNT_D3A_SKIP_STORE_LOAD=1` improves D3A to about `7.26-7.28`, still below baseline. |
| A-only and B-only D3A behave similarly. | Operand choice is not the obvious primary trigger. |
| A+B D3A is worse. | More staging is not automatically better. |
| Hand LDS2 is structurally much denser. | Existing comparison shows much lower inst/wait/memops per WMMA. |
| Hand LDS2 active trace works. | `hand-lds2 2x2` produced `64 WMMA`, `64 global_load_b128`, `64 ds_store_b128`, `128 ds_load_b128`, `26 waits`. |
| Generated active lifecycle trace works after correction. | `_generated_active_insts` now uses `AMDISARenderer(Target.parse(args.target))` for final stream generation. |

## Required Comparison Matrix

For each route:

1. Generated baseline DBUF, active `u0=2,u1=2,loc=2,unr=2`.
2. Generated D3A, same shape.
3. Generated D3A plus diagnostic wait skip, same shape.
4. Hand LDS2 `wm=2,wn=2,waves_m=1,waves_n=1,bk=64,dbuf=1`.

Collect:

| Metric | Why it matters |
|---|---|
| TFLOPS / correctness status | Performance ground truth. |
| `v_wmma` count | Denominator for all ratios. |
| total instructions / WMMA | General bookkeeping density. |
| bytes / WMMA | Encoding/code-size pressure. |
| `s_waitcnt` / WMMA | Wait density. |
| `global_load_b128` / WMMA | Global staging amortization. |
| `ds_store_b128` / WMMA | LDS write cost. |
| `ds_load_b128` / WMMA | LDS read reuse/amortization. |
| `v_pack_b32_f16` / WMMA | Generated-only pack overhead. |
| barriers / WMMA | Pipeline segmentation. |
| body future staging distance | Whether staging is early enough to hide. |
| wait before each WMMA cluster | Whether WMMAs are issued in dense groups. |
| WMMA operand origins | Direct global vs LDS vs mixed. |

## Decision Questions

The trace must answer these yes/no questions:

| Question | If yes | If no |
|---|---|---|
| Does hand use fewer waits per WMMA by clustering multiple WMMAs after one LDS wait? | Implement WMMA cluster scheduling / LDS-load grouping. | Focus elsewhere. |
| Does hand avoid `v_pack_b32_f16` entirely? | Implement direct b128 fragment path or packed carrier reuse. | Pack overhead less likely primary. |
| Does hand stage future slot earlier than generated D3A? | Move D3A placement earlier / increase lookahead. | Placement not the main issue. |
| Does hand do fewer LDS loads per WMMA? | Generated reloads fragments too often; implement fragment reuse/residency. | Look at wait/pack cost. |
| Does hand have fewer memory ops per WMMA? | Reduce staging breadth or shape selection. | Wait scheduling likely dominates. |
| Does generated D3A add work but not reduce any later work? | D3A is structurally correct but not a profitable primitive yet. | Continue D3A tuning. |

## Existing Tools To Use

Primary structural tracer:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py ...
```

Primary perf/shape matrix:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py ...
```

Primary generated bounded worker:

```bash
DEV=AMD:ISA WORKER=1 MM=512 OUTF=5120 INF=5120 U0=2 U1=2 LOC=2 UNR=2 \
PREFILL_DBUF=1 AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 REGALLOC_ADDR_REMAT=1 \
AMD_ISA_WAITCNT_TARGETED=1 PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill_v2_schedule_search.py
```

## Work Packages

### H1. Generated Active Trace Fix

Owner writes only if needed in `extra/qk/prefill/kernel_lifecycle_trace.py`.

Tasks:

- Make generated active trace force/use `AMDISARenderer` / `DEV=AMD:ISA`, not HIP.
- Produce JSON rows for baseline DBUF, D3A, and D3A+wait-skip.
- Confirm ratios and body regions are visible.

100% gate:

```text
generated active trace returns final instruction counts for all three generated routes.
```

### H2. Hand LDS2 Reference Refresh

Read-only unless a bug in reporting is found.

Tasks:

- Run hand LDS2 `2x2` trace and, if feasible, hand timing with small reps.
- Extract ratios per WMMA and WMMA cluster cadence.
- Identify whether hand clusters 4 WMMAs per wait/staging group and how far staging is ahead.

100% gate:

```text
hand LDS2 metrics table is refreshed from current code, not copied from old docs.
```

### H3. Delta Classification

Read-only analysis against H1/H2 outputs and current docs.

Tasks:

- Rank the observed deltas by likely TFLOPS impact.
- Map each delta to one compiler primitive:
  - alias-aware waitcnt,
  - earlier D3A placement,
  - fragment reuse/residency,
  - direct b128/no-pack lowering,
  - shape selection / less staging.

100% gate:

```text
produce a ranked list where every item has: evidence, expected win, implementation layer, and smallest next test.
```

## Completion Bar

This scope is complete when we have a single table:

```text
route x {TFLOPS, WMMA, inst/WMMA, waits/WMMA, packs/WMMA, DS loads/WMMA, DS stores/WMMA, body staging distance}
```

and a ranked recommendation for the next small implementation test. If the table shows generated D3A adds work without
reducing later work, the right decision is to stop optimizing D3A until a placement/reuse primitive changes that equation.

## Agent Results

Agents spawned:

| Agent | Work package | Result |
|---|---|---|
| Peirce | H1 generated active trace fix | PASS. Updated `extra/qk/prefill/kernel_lifecycle_trace.py`; generated active trace now returns final AMD:ISA instruction metrics. |
| Beauvoir | H2 hand LDS2 refresh | PASS. Refreshed current hand LDS2 structure and low-rep timing. |
| Euler | H3 delta classification | PASS. Ranked move-the-needle deltas and recommended next small test. |

### Current Generated vs Hand Table

Active shape: `m=512,n=5120,k=5120`, generated `u0=2,u1=2,loc=2,unr=2`, hand `wm=2,wn=2,waves_m=1,waves_n=1,bk=64,dbuf=1`.

| route | TFLOPS note | insts | bytes | WMMA | inst/WMMA | wait/WMMA | global/WMMA | ds_store/WMMA | ds_load/WMMA | pack/WMMA | barrier/WMMA | future slot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| generated baseline DBUF | ~7.6-7.9 | 625 | 3904 | 16 | 39.062 | 3.312 | 2.0 | 2.0 | 4.0 | 0.0 | 0.125 | false |
| generated D3A | ~7.0 | 709 | 4384 | 16 | 44.312 | 4.812 | 2.75 | 2.75 | 4.0 | 0.0 | 0.125 | true |
| generated D3A + wait-skip diagnostic | ~7.26-7.28 | 698 | 4340 | 16 | 43.625 | 4.125 | 2.75 | 2.75 | 4.0 | 0.0 | 0.125 | true |
| hand LDS2 | 14.77 low-rep | 611 | 4316 | 64 | 9.547 | 0.406 | 1.0 | 1.0 | 2.0 | 0.0 | 0.062 | true |

The table answers the immediate question: current D3A creates a visible future slot, but it does not reduce the later
work. It adds instructions, waits, global loads, and LDS stores per WMMA while leaving generated `ds_load_b128/WMMA` at
`4.0`, twice the hand LDS2 rate.

### Hand Cadence Target

Current hand LDS2 clusters WMMAs in groups of four:

```text
8x ds_load_b128 -> s_waitcnt lgkmcnt(0) -> 4x v_wmma
```

Representative clusters:

```text
153-156, 166-169, 179-182, 192-195,
256-259, 269-272, 282-285, 295-298,
362-365, 375-378, 388-391, 401-404,
433-436, 446-449, 459-462, 472-475
```

There are no waits between the four WMMAs inside a hand cluster. Future-slot staging appears in bulk rollover gaps:

```text
between_wmma_195_256
between_wmma_298_362
between_wmma_404_433
```

This is different from generated D3A, where the future store is too local and does not create a broad reuse/cluster
window.

### Ranked Deltas

| Rank | Delta likely moving TFLOPS | Evidence | Smallest next test |
|---:|---|---|---|
| 1 | D3A future staging is too close to current-slot LDS loads. | Wait-skip recovers ~0.25 TFLOPS, but D3A still trails baseline. | Move D3A stage anchor one WMMA cluster earlier and measure waits/TFLOPS. |
| 2 | Generated lacks hand-style WMMA clusters. | Hand has `0.406` waits/WMMA and 4-WMMA clusters; generated baseline has `3.312` waits/WMMA. | Add/report max consecutive WMMAs without waits for generated routes, then cluster LDS loads/WMMAs. |
| 3 | Generated reloads too many LDS fragments per WMMA. | Generated has `4.0 ds_load_b128/WMMA`; hand has `2.0`. | Prototype fragment reuse/residency for one operand and require ds_load/WMMA to drop. |
| 4 | D3A adds work without reducing later work. | D3A raises inst/WMMA and waits/WMMA while ds_load/WMMA stays `4.0`. | Park D3A as perf route until placement/reuse changes this equation. |
| 5 | Hand is much denser overall. | Hand `9.547 inst/WMMA`; generated baseline `39.062`; D3A `44.312`. | Use density ratios as promotion gates for every new primitive. |
| 6 | Direct b128/no-pack path is secondary for this specific delta. | Current active generated table has `0.0 pack/WMMA`, yet still trails badly. | Do not lead with pack work for this active 2x2 delta. |

## Next Implementation Test

The recommended next small implementation test is:

```text
flag-gated D3A placement/lookahead variant:
  emit the existing future ds_store_b128 one WMMA cluster earlier,
  keep D2/D3/D7 true,
  avoid scalar LDS stores and spills,
  see whether normal waits approach the wait-skip diagnostic without AMD_ISA_WAITCNT_D3A_SKIP_STORE_LOAD=1,
  and require bounded 2x2 loc=2 unr=2 TFLOPS to recover at least the ~0.25 TFLOPS wait-skip gap.
```

If this does not move TFLOPS, the next primitive should shift from D3A placement to generated fragment reuse, because the
largest structural delta versus hand is still `ds_load_b128/WMMA` and WMMA clustering, not packs.

## Match-Hand Execution Plan

The table changes the priority order. Matching hand does not mean copying the handwritten builder; it means making the
generated final instruction stream converge on the same density class.

### Target Ratios

| Metric | Current generated baseline | Current generated D3A | Hand LDS2 target | Meaning |
|---|---:|---:|---:|---|
| `ds_load_b128/WMMA` | `4.0` | `4.0` | `2.0` for `2x2`; `1.5` on rectangular shapes | Primary reuse gap. |
| `wait/WMMA` | `3.312` | `4.812` | `0.406` current low-rep trace | Primary scheduler/cadence gap. |
| `inst/WMMA` | `39.062` | `44.312` | `9.547` current low-rep trace | Overall bookkeeping gap. |
| `global_load_b128/WMMA` | `2.0` | `2.75` | `1.0` | Staging amortization gap. |
| `ds_store_b128/WMMA` | `2.0` | `2.75` | `1.0` | D3A currently over-stages. |

### Design Thesis

The primitive fix is not broad D3A tuning. Current D3A proves the opposite: it adds future-slot work but does not reduce
current-slot work. The first promoted primitive must therefore be proof-safe A/B fragment reuse. D3A placement and
waitcnt tuning become profitable only after reuse creates a larger WMMA cluster to schedule around.

Desired generated lifecycle:

```text
load/store packed LDS fragments once per reusable A/B proof group
ds_load_b128 A/B fragments into resident VGPRs
issue multiple WMMAs from those resident fragments
only then refill/reload the next proof group
```

This is the generated analogue of the hand cadence:

```text
8x ds_load_b128 -> s_waitcnt lgkmcnt(0) -> 4x v_wmma
```

### Phase M0. Freeze Baselines

Purpose: every later change must prove it moves at least one target ratio.

Run:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --kind generated-active --m 512 --n 5120 --k 5120 --u0 2 --u1 2 --loc 2 --unr 2 --json

DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --kind generated-active --m 512 --n 5120 --k 5120 --u0 2 --u1 2 --loc 2 --unr 2 \
  --extra-env PREFILL_DBUF_D3A_POST=1 --json

DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py \
  --kind hand-lds2 --m 512 --n 5120 --k 5120 --wm 2 --wn 2 --waves-m 1 --waves-n 1 --bk 64 --dbuf 1 --json
```

100% gate:

- baseline, D3A, and hand rows are present in one JSON/table artifact;
- ratios are computed from final AMD:ISA instructions;
- `4x4` remains excluded from the active gate on gfx1100.

### Phase M1. Proof-Key Audit Before Behavior Change

Purpose: explain why unsafe resident reuse lowered `ds_load/WMMA` but produced wrong output.

Use/extend:

```bash
PYTHONPATH=. python3 extra/qk/prefill/wmma_frag_key_audit.py --m-up 1
```

The audit must emit, per WMMA operand:

```text
role, lds_buffer_id, dbuf_slot, k_phase, logical_row_or_col,
byte_start, byte_len, producer_epoch, overwrite_epoch,
current_carrier_id, address_only_key, proof_key_or_reason
```

100% gate:

- address-only merge groups that produced the previous wrong result are named;
- each rejected merge has a specific mismatched/missing field;
- no generated instruction stream changes in this phase.

### Phase M2. Fail-Closed Resident Fragment Reuse

Purpose: cut `ds_load_b128/WMMA` first, because this is the largest structural delta.

Behavior:

```text
for each WMMA operand:
  key = proof_key(operand)
  if key is None:
    emit current per-WMMA ds_load_b128 path
  else:
    load fragment once into a resident VGPR span for that key
    reuse resident span for all WMMAs in the proven consumer epoch
```

Invariants:

- never merge A with B;
- never merge different DBUF slots;
- never merge different K phases;
- never merge different row/column identities;
- never reuse past an overwrite epoch;
- address equality alone never promotes reuse.

100% gate:

- generated `2x2` is GPU-correct;
- `ds_load_b128/WMMA` improves from `4.0` toward `2.0`;
- scalar LDS fallback stays absent;
- if proof metadata is missing, the route falls back instead of producing NaN.

### Phase M3. WMMA Cluster Scheduling

Purpose: after reuse, reduce wait density by issuing multiple WMMAs after one LDS wait.

Target pattern:

```text
cluster:
  ds_load_b128 fragments for one proof group
  s_waitcnt lgkmcnt(0)
  v_wmma
  v_wmma
  v_wmma
  v_wmma
```

100% gate:

- lifecycle trace reports max consecutive WMMAs without an intervening wait;
- generated `wait/WMMA` improves versus baseline;
- correctness remains passing;
- no broad wait-skip diagnostic is required.

### Phase M4. Reintroduce D3A Only As Paid Future Work

Purpose: make future staging profitable instead of merely visible.

D3A is allowed to promote only if it reduces a later metric:

```text
D3A may increase body future-slot work
only if it reduces later global_load/WMMA, ds_store/WMMA, ds_load/WMMA, wait/WMMA, or TFLOPS time.
```

100% gate:

- D3A no longer raises `inst/WMMA` and `wait/WMMA` without reducing another target ratio;
- future-slot stores are placed early enough that normal waitcnt does not serialize them immediately before current-slot loads;
- bounded TFLOPS beats the non-D3A generated baseline, not just the D3A baseline.

### Phase M5. Active Shape Expansion

Purpose: prove the primitive is not a `2x2` special case.

Run:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env current --skip-hand --shapes '2,2;4,2;2,4' --loc 2 --unr 2 --pin-clock --json
```

100% gate:

- `2x2`, `4x2`, and `2x4` are correct;
- `ds_load/WMMA`, `wait/WMMA`, and `inst/WMMA` improve over current generated DBUF-safe baselines;
- `4x4` remains parked unless explicitly reopened.

### Phase M6. E2E Promotion

Purpose: prove the kernel-level density improvement matters for prefill.

100% gate:

- repeated pinned-clock kernel runs improve beyond noise;
- model-level prefill improves on the existing route harness;
- route artifact records selected active shape, flags, and rollback path;
- default remains unchanged until correctness and performance both pass.

## Work Order

| Order | Work | Parallel? | Why this order |
|---:|---|---|---|
| 1 | M0 baseline freeze | Yes | Prevents another build-before-measure branch. |
| 2 | M1 proof-key audit | Yes | Names the exact unsafe reuse before codegen changes. |
| 3 | M2 fail-closed reuse | No | Primary structural fix for `ds_load/WMMA`. |
| 4 | M3 cluster scheduling | No | Wait reduction only becomes meaningful after reuse creates clusters. |
| 5 | M4 paid D3A | Partially | D3A should follow reuse/cluster, not lead. |
| 6 | M5 active shapes | Yes after M2/M3 | Validates generality across fitting shapes. |
| 7 | M6 E2E promotion | No | Final route decision. |

## Stop Conditions

Stop and rescope if any of these occur:

- proof-key audit cannot identify row/column, slot, or epoch for generated operands;
- fail-closed reuse cannot improve `ds_load/WMMA` while preserving correctness;
- cluster scheduling reduces waits but TFLOPS does not move after repeated same-clock runs;
- D3A still only adds work after reuse/cluster is in place.

## M1 Small-Test Result

Before implementing fail-closed reuse, we reran the smallest `2x2` test on the actual `dbuf-safe` LDS path.

Command class:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env dbuf-safe --skip-hand --shapes '2,2' --loc 2 --unr 2 --pin-clock --json

DEV=AMD:ISA PREFILL_WMMA_AB_ADDR_KEY=1 PREFILL_WMMA_CHAIN_AB_RESIDENT=1 PYTHONPATH=. \
  python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --generated-env dbuf-safe --skip-hand --shapes '2,2' --loc 2 --unr 2 --pin-clock --json
```

Result:

| Route | Status | TFLOPS | inst/WMMA | wait/WMMA | global/WMMA | ds_store/WMMA | ds_load/WMMA |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline `dbuf-safe` | `ok` | `8.01` | `39.062` | `3.312` | `2.0` | `2.0` | `4.0` |
| unsafe resident reuse | `WRONG rr=nan` | `0.0` | `34.25` | `2.5` | `2.0` | `2.0` | `2.0` |

Interpretation:

- The lever is real: resident reuse cuts `ds_load_b128/WMMA` from `4.0` to `2.0` and also reduces waits and total
  instructions.
- The implementation is not safe: correctness fails with `rr=nan`.
- Therefore the blocker is not "does reuse move the ratio"; it does. The blocker is the missing semantic proof for when
  two WMMA operands may share a resident fragment.

Audit command:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/wmma_frag_key_audit.py \
  --shapes '2,2;4,2;2,4' --loc 2 --unr 2
```

Audit result:

| Shape | Rows | Address-reused groups | Promotion-safe proof groups | Rejected address-only merges |
|---|---:|---:|---:|---:|
| `2x2` | `32` | `16` | `0` | `16` |
| `4x2` | `34` | `16` | `0` | `16` |
| `2x4` | `26` | `4` | `0` | `4` |

Every rejected merge has the same root reason:

```text
missing_proof_key, unprovable:no_proof_metadata
```

We also tested `PREFILL_TC_LOCAL_STAGE_COOP_POST=1` as a possible existing proof-tag path. It does not solve this route:
the audit still reports `missing_proof_key`, and the generated stream no longer matches the `dbuf-safe` LDS path we are
trying to improve. So this is not the shortcut.

## M2 Implementation Scope: Proof-Safe Resident Reuse

The next implementation should be a metadata/proof pass, not another ratio probe.

### Files

| File | Role |
|---|---|
| `tinygrad/codegen/opt/postrange.py` | Attach proof metadata at the current `WITH_LOCAL` LDS staging site. |
| `tinygrad/codegen/late/devectorizer.py` | Preserve proof tags through scalar/vector expansion when possible. |
| `tinygrad/renderer/isa/amd.py` | Use proof keys for resident A/B reuse; fall back when proof is absent. |
| `extra/qk/prefill/wmma_frag_key_audit.py` | Gate promotion with shape-by-shape proof grouping. |
| `test/unit/test_amd_isa_wmma.py` | Add default-off proof-key/fail-closed tests. |

### Required Proof Fields

```text
role: A | B
lds_buffer_id
dbuf_slot
k_phase
logical_row_or_col
byte_start
byte_len = 32
producer_epoch
overwrite_epoch
```

### Implementation Steps

1. Add proof metadata to the current `WITH_LOCAL` LDS load/index carriers.
   - The current route already creates stable local buffers and byte windows.
   - The audit proves contiguity and address grouping, but not slot/phase/epoch.
   - The first code change should make `PREFILL_WMMA_FRAG_KEY_DUMP=1` show proof fields without changing emitted
     instructions.

2. Update `_wmma_frag_reuse_key` to prefer proof keys.

   ```text
   key = _wmma_frag_proof_key(role, carrier)
   if key is None:
     return id(carrier)   # fail closed
   return key
   ```

   Address-only reuse must remain diagnostic-only. It should not be the promotion key.

3. Re-run audit before enabling resident reuse.

   Promotion precondition:

   ```text
   total_promotion_safe_groups > 0
   rejected_address_only_merges are either rejected by proof or split by proof
   no missing role / slot / phase / epoch on promoted groups
   ```

4. Enable resident reuse behind the existing flag and test `2x2`.

   Required first pass:

   ```text
   status = ok
   ds_load/WMMA < 4.0
   no scalar LDS fallback
   no broad wait-skip diagnostic
   ```

5. Expand only after `2x2` passes.

   Order:

   ```text
   2x2 -> 4x2 -> 2x4
   ```

### Acceptance Gate

This phase is complete when:

| Gate | Requirement |
|---|---|
| Correctness | `2x2` generated `dbuf-safe + proof reuse` is `ok`. |
| Density | `ds_load/WMMA` improves from `4.0` toward `2.0`. |
| Safety | Missing proof metadata falls back to carrier identity, not address-only reuse. |
| Generality | `4x2` and `2x4` audits show proof-safe groups before behavior promotion. |
| Default | Default generated route remains unchanged until correctness and perf improve. |

### M2 First Pass Result

Implemented a default-off proof-key skeleton:

```text
PREFILL_WMMA_AB_PROOF_META=1   # opt-in proof metadata attempt
PREFILL_WMMA_AB_PROOF_KEY=1    # resident reuse may use proof keys only
```

Safety result:

| Test | Result |
|---|---|
| Default unit suite | PASS: `python3 -m unittest test.unit.test_amd_isa_wmma` -> `43` tests, `1` expected failure. |
| Baseline `dbuf-safe 2x2` | PASS: `status=ok`, `TFLOPS=8.01`, `ds_load/WMMA=4.0`. |
| Proof-key resident fallback | PASS safe fallback: `status=ok`, `TFLOPS=7.97`, `ds_load/WMMA=4.0`. |
| Audit with proof metadata opt-in | BLOCKED: `promotion_safe_groups=0`; all address-only merges still reject with `missing_proof_key`. |

Important correction: fail-closed cannot mean "fallback to carrier identity" on this route. Carrier identity is already
shared across the unsafe consumers, so it still takes the wrong reuse. The implemented safe behavior is:

```text
if proof key is absent:
  do not reserve/use the resident A/B window
  fall back to the existing non-resident high-fragment WMMA tile path
```

Current blocker:

```text
PREFILL_WMMA_AB_PROOF_META=1 does not survive to the final WMMA operand LOAD/INDEX.
```

Raw dump still shows:

```text
index_buf_op=AFTER
index_buf_tag=None
index_tag=None
proof_key_status=unprovable:no_proof_metadata
```

So the next narrow task is not more resident-reuse tuning. It is to find the exact transformation between postrange
staging and AMD isel that rebuilds the LDS operand `INDEX`/`AFTER` without preserving the proof tag, then preserve the
tag there. Only after `wmma_frag_key_audit.py` reports `promotion_safe_groups > 0` should resident reuse be allowed to
reduce `ds_load/WMMA`.

Follow-up attempt:

- Added default-off tag preservation attempts in `devectorizer.py` for vectorized buffer casts, `CAST(AFTER(...))`, and
  `GEP(AFTER(...))`.
- Added AMD-side extraction from `CAST`, `INDEX`, and `INDEX.src[0]` tags.
- Verified default safety still holds.

Latest result:

| Test | Result |
|---|---|
| `python3 -m unittest test.unit.test_amd_isa_wmma` | PASS: `43` tests, `1` expected failure. |
| Baseline `dbuf-safe 2x2` | PASS: `status=ok`, `ds_load/WMMA=4.0`. |
| `PREFILL_WMMA_AB_PROOF_META=1 PREFILL_WMMA_AB_PROOF_KEY=1 PREFILL_WMMA_CHAIN_AB_RESIDENT=1` | PASS safe fallback: `status=ok`, `ds_load/WMMA=4.0`. |
| `wmma_frag_key_audit.py --shapes '2,2'` | Still blocked: `promotion_safe_groups=0`. |

Raw proof dump still shows:

```text
index_buf_op=AFTER
index_buf_tag=None
index_op=INDEX
index_tag=None
load_index_op=CAST
load_index_tag=None
proof_key_status=unprovable:no_proof_metadata
```

Therefore the current stopping point is below the attempted tag-preservation sites. Either:

1. the active `dbuf-safe` LDS operand load is not produced by the postrange carrier we tagged, or
2. a later rebuild path reconstructs the load/index chain from semantic structure and drops the tagged node entirely.

The next viable probe is to dump the WMMA operand chain immediately after postrange and immediately before AMD isel,
then diff by pointer/base/local byte window. Do not keep tuning resident reuse until that diff names the tag-dropping
transform.

### M3 Two-Point Proof-Chain Probe Result

Added default-off proof-chain diagnostics:

```text
PREFILL_WMMA_PROOF_CHAIN_DUMP=1
```

The dump has three useful phases:

| Prefix | Phase |
|---|---|
| `POSTRANGE_WMMA_PROOF_JSON` | local-stage proof carrier creation in postrange |
| `PREISEL_WMMA_PROOF_JSON phase=after_full_rewrite` | final generic-lowered WMMA operand chain |
| `PREISEL_WMMA_PROOF_JSON phase=after_pre_isel` | after AMD pre-isel, before AMD isel |

Result on `dbuf-safe 2x2`:

| Phase | Result |
|---|---|
| postrange | Creates tagged local stage buffers: `wmma_frag_buffer_proof` exists on `STAGE`. |
| after full rewrite | Final WMMA operand is already `GEP -> LOAD -> CAST -> INDEX -> AFTER`, all tags are gone. |
| after pre-isel / AMD isel | Still untagged. AMD isel is not the tag-dropping layer. |

Conclusion:

```text
The proof metadata disappears between postrange local staging and the final generic lowered WMMA operand chain.
This is not an AMD isel bug.
```

### M3 Shortcut Test: LDS-Descriptor Proof

Tested a default-off shortcut:

```text
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1
```

This synthesizes proof fields from the final AMD LDS descriptor and exact 32-byte window, bypassing the lost tag.

Audit result:

```text
2x2 promotion_safe_groups: 0 -> 16
```

But the generated-kernel small tests reject it:

| Variant | Status | ds_load/WMMA | inst/WMMA | wait/WMMA |
|---|---:|---:|---:|---:|
| A+B descriptor proof reuse | WRONG `rr=1.0e+00` | 2.0 | 36.062 | 2.625 |
| A-only descriptor proof reuse | WRONG `rr=1.0e+00` | 3.0 | 40.875 | 3.562 |
| B-only descriptor proof reuse | RuntimeError / MMU fault | 3.0 | 41.312 | 3.312 |

So exact LDS byte-window identity is not a sufficient proof. It moves the density needle, but it does not prove the
producer epoch / overwrite epoch needed for correctness. Treat `PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1` as a diagnostic
refuted shortcut, not a shippable primitive.

Current primitive requirement:

```text
A valid reuse proof must be carried from the actual LDS-producing store/barrier epoch to the consuming WMMA load.
It cannot be reconstructed from final address identity alone.
```

### M4 Store-Epoch Proof + Resident Reservation Result

Added another default-off proof source:

```text
PREFILL_WMMA_AB_PROOF_FROM_LDS_STORES=1
```

This walks from the final WMMA load's `AFTER(DEFINE_LOCAL, BARRIER)` dependency into the barrier's producer store group,
matches the exact 32-byte LDS load window, rejects ambiguous overwrite coverage, and uses the matched producer-store set
as the proof epoch.

Also fixed the proof-key resident register reservation:

```text
_n_ab_frags(ctx) now counts proof-key A/B resident groups instead of returning 0 in proof mode.
```

Without this, promoted proof groups could allocate low resident A/B VGPRs without reserving them out of the virtual VGPR
pool, creating collisions once proof promotion actually happened.

Small-test results on `dbuf-safe 2x2`:

| Variant | Status | ds_load/WMMA | inst/WMMA | wait/WMMA | Note |
|---|---:|---:|---:|---:|---|
| A+B store-epoch proof reuse | WRONG `rr=nan` | 2.0 | 34.25 | 2.5 | Dense but invalid. |
| A-only store-epoch proof reuse | ok | 3.0 | 37.562 | 3.312 | Correct. |
| B-only store-epoch proof reuse | ok | 3.0 | 37.562 | 3.312 | Correct. |

Current diagnosis:

```text
The proof is sufficient for one operand at a time.
The remaining invalid case is combined A+B residency, not generic proof absence and not simple low-VGPR reservation.
```

Next layer:

```text
Audit combined A+B resident physical layout and WMMA operand lifetime interaction.
The primitive path should promote one operand first, then only enable A+B when the combined resident hazard is named.
```

### M5 Combined A+B Resident Path Audit

Ran the full physical WMMA-chain trace on the same `dbuf-safe 2x2` route for:

```text
A-only store-epoch proof reuse
B-only store-epoch proof reuse
A+B store-epoch proof reuse
```

Trace result:

| Variant | Correct | ds_load/WMMA | Physical behavior |
|---|---:|---:|---|
| A-only | yes on `2x2` | 3.0 | One low resident operand, other operand uses high scratch reloads. |
| B-only | yes on `2x2` | 3.0 | One low resident operand, other operand uses high scratch reloads. |
| A+B | no, `rr=nan` | 2.0 | All K-phase A/B proof groups are loaded into distinct low VGPR ranges before the WMMA block. |

The actual A+B final stream begins by loading 32 LDS fragments for all K phases, then runs the 16 WMMAs. Example shape:

```text
K0: C8  uses A40  B48
K0: C16 uses A40  B104
K0: C24 uses A136 B48
K0: C32 uses A136 B104
K1: C8  uses A56  B64
...
K3: C32 uses A160 B128
```

Conservative waitcnt does not fix it:

```text
AMD_ISA_WAITCNT_CONSERVATIVE=1 -> still WRONG rr=nan
```

A dependency-aware resident-pack probe also does not fix it:

```text
PREFILL_WMMA_RESIDENT_PACK_DEP=1 -> still WRONG rr=nan
```

Cross-shape one-operand result:

| Variant | 2x2 | 4x2 | 2x4 |
|---|---:|---:|---:|
| A-only | ok | wrong `rr=nan` | wrong `rr=nan` |
| B-only | ok | wrong `rr=nan` | RuntimeError |

Current diagnosis:

```text
The current proof/reuse implementation is a useful 2x2 diagnostic, but not the primitive fix.
It promotes proof windows as globally resident fragments across all K phases.
The hand LDS2 primitive is different: phase-scoped A/B reuse.

For each K substep:
  load WM A fragments + WN B fragments
  wait lgkm
  run the WM*WN WMMAs for that K substep
  reuse the same A/B fragment registers for the next K substep
```

So the primitive route should pivot from "proof-key global A/B residency" to:

```text
phase-scoped A/B fragment reuse, ordered K-major, with only WM+WN A/B fragments live at once
```

That is the generated counterpart of hand `compute(buf)`. It gives the same target density:

```text
2x2: each K phase loads 2 A fragments + 2 B fragments = 8 ds_load_b128 for 4 WMMAs
=> ds_load/WMMA = 2.0
```

But it avoids the invalid "all K phases resident at once" layout.

### M6 Phase-Key Small Test

Tested the cheapest possible approximation of phase-scoped reuse:

```text
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
```

This collapses proof windows to a small per-row/per-col key so A/B VGPR ranges are reused across K phases instead of
allocating one resident range per LDS byte window.

Result on `dbuf-safe 2x2`:

```text
status = NotImplementedError: Inc 0: no spills
```

Interpretation:

```text
Key collapsing alone is not the fix.
The current graph still exposes multiple K-phase loads with overlapping pinned A/B destinations without the K-major
ordering edges needed to make those lifetimes legal.
```

Therefore the next implementation should not keep tuning proof keys. It needs to change the WMMA chain lowering shape:

```text
for k_phase in K phases:
  emit/load phase A/B fragments into the reusable A/B ranges
  wait for those fragment loads
  emit all WM*WN WMMAs that consume that phase
```

That is a true K-major/phase-scoped lowering. It cannot be obtained by only changing resident allocation keys.

### M7 True K-Major Phase-Scoped Lowering Test

Implemented a default-off K-major lowering experiment:

```text
PREFILL_WMMA_KMAJOR_PHASE=1
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
PREFILL_WMMA_AB_PROOF_KEY=1
PREFILL_WMMA_AB_PROOF_FROM_LDS_STORES=1
PREFILL_WMMA_CHAIN_AB_RESIDENT=1
```

Unlike M6, this does not only change keys. It lowers sibling WMMA chains together:

```text
for k_phase:
  load/cache phase A row fragments and B column fragments into reusable low VGPR ranges
  emit all WM*WN WMMAs for that phase
  use the last WMMA as the ordering dependency for the next phase's fragment loads
```

The first key attempt reused one A and one B fragment for all four `2x2` subtiles and was wrong:

```text
ds_load/WMMA = 1.0
status = WRONG rr=1.2e+00
```

Correcting the phase key to preserve row/column identity fixed the active shapes:

| Shape | Status | TFLOPS | ds_load/WMMA | inst/WMMA | wait/WMMA |
|---|---:|---:|---:|---:|---:|
| `2x2` | ok | 11.53 | 2.0 | 34.625 | 2.875 |
| `4x2` | ok | 10.19 | 1.5 | 29.188 | 2.188 |
| `2x4` | ok | 10.11 | 1.5 | 29.188 | 2.188 |

Structural trace for `2x2` now shows the intended row/column reuse per K phase:

```text
K phase:
  C8  uses A40 B48
  C16 uses A40 B56
  C24 uses A64 B48
  C32 uses A64 B56
next K phase reuses the same A/B ranges after reloading them
```

This is the first generated route that matches the hand-style phase-scoped A/B fragment primitive on the active shapes.

Verification:

```text
python3 -m unittest test.unit.test_amd_isa_wmma
43 tests OK, 1 expected failure
```
