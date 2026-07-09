# Scope: generated DBUF lifecycle trace to hand-LDS2 parity

Goal: make the generated native-ISA prefill WMMA route produce the hand-LDS2 class of machine-code lifecycle:

```text
global_load_b128 -> ds_store_b128 -> barrier -> ds_load_b128 -> v_wmma
with next-slot staging work visible between current WMMA regions
```

The shared floor is already proven: both generated and hand paths end as `Inst` lists that route through
`assemble_linear -> ELF -> AMDProgram/HSA -> GPU`. The tail-off is before that:

| Path | Tail-off |
| --- | --- |
| generated | `UOps -> isel -> regalloc -> waitcnt/scheduler -> Inst` |
| hand | `Python builder -> fixed Inst list` |

## New tracer

`extra/qk/prefill/kernel_lifecycle_trace.py` is the structural scoreboard. It does not launch the GPU and does not parse
disassembly text. It inspects final RDNA3 instruction fields.

Baseline command:

```bash
PYTHONPATH=. AMD_ISA_WMMA_B128_FRAG=1 \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind all
```

Current baseline:

| Path | global b128 | ds_store_b128 | ds_load_b128 | barriers | WMMA | work between WMMAs | operand origins | D7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| generated route-shaped | 16 | 0 | 0 | 0 | 16 | 0 | global/global | FAIL |
| hand pipe 2x4 | 48 | 0 | 0 | 0 | 32 | 2 | global/global | n/a |
| hand LDS2 2x4 | 48 | 48 | 96 | 4 | 64 | 7 | LDS/LDS | PASS |

The tracer origin classifier is now per-WMMA backward-def based; hand LDS2 reports `('ds_load_b128', 'ds_load_b128')`.

## Generated route matrix

### Non-DBUF, both-side packed WITH_LOCAL

```bash
PYTHONPATH=. AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind generated
```

Result:

| global b128 | ds_store_b128 | ds_load_b128 | barriers | WMMA | operand origins | work between WMMAs |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 8 | 16 | 16 | 2 | 16 | LDS/LDS | 0 |

Meaning: both operands are structurally staged through LDS with packed stores and no scalar LDS stores. The remaining
failure is cadence only: all staging/loads happen before the WMMA block, so there is no hand-like overlap.

### DBUF, full 4x4 route-shaped structural trace

Plain DBUF still fails before final stream:

```text
NotImplementedError: Inc 0: no spills
```

The full address/lifetime bundle now reaches the hand-LDS2-like structural target when paired with address remat and
end-source live-range trimming:

```bash
PYTHONPATH=. AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
REGALLOC_ADDR_REMAT=1 \
REGALLOC_END_NO_SOURCE_LIVE=1 \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind generated
```

Result:

| global b128 | ds_store_b128 | ds_load_b128 | barriers | WMMA | operand origins | work between WMMAs | scalar LDS stores | D7 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 16 | 32 | 128 | 2 | 32 | LDS/LDS | 31 | 0 | PASS |

Meaning: the synthetic 64x64 generated stream now has the desired lifecycle: both WMMA operands come from LDS,
packed stores survive DBUF, and next-slot work is visible between WMMA regions.

### DBUF, smaller `m_up=1`

```bash
PYTHONPATH=. AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind generated --m-up 1
```

Result:

| global b128 | ds_store_b128 | ds_load_b128 | barriers | WMMA | operand origins | work between WMMAs | scalar LDS stores |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 16 | 16 | 32 | 2 | 8 | LDS/LDS | 7 | 32 |

After relaxing the B-tilekey packed-store matcher to accept DBUF's two-slot fragment groups, this case is also clean:

| global b128 | ds_store_b128 | ds_load_b128 | barriers | WMMA | operand origins | work between WMMAs | scalar LDS stores | D7 |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 16 | 20 | 32 | 2 | 8 | LDS/LDS | 7 | 0 | PASS |

Meaning: DBUF cadence exists at the smaller shape, and the packed-store route is now preserved.

## Native execution loop

The central execution harness is `extra/qk/prefill_v2_schedule_search.py` in `WORKER=1` mode. It compiles through the
normal graph matmul path, applies the warmstart opts, checks numeric output against numpy, then times the native kernel.

Real attn_qo shape command:

```bash
PYTHONPATH=. DEV=AMD:ISA WORKER=1 MM=512 OUTF=5120 INF=5120 U0=2 U1=2 LOC=0 UNR=8 \
AMD_ISA_REG_ACCUM=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
REGALLOC_ADDR_REMAT=1 \
REGALLOC_END_NO_SOURCE_LIVE=1 \
python3 extra/qk/prefill_v2_schedule_search.py
```

Result:

| Shape | Schedule | Status | TFLOPS | Notes |
| --- | --- | --- | ---: | --- |
| 512x5120x5120 | u0=2,u1=2,loc=0,unr=8 | ok | 2.98 | first real `DEV=AMD:ISA` DBUF correctness pass |
| 512x5120x5120 | u0=2,u1=2,loc=0,unr=8 without `AMD_ISA_REG_ACCUM=1` | runtime fail | 0 | `local_bytes=65536` + `4096` accumulator LDS reservation -> `69632` |
| 512x5120x5120 | u0=2,u1=4 or u0=4,u1=2 with reclaim | runtime fail | 0 | staged tile footprint `98304` bytes |
| 512x5120x5120 | u0=4,u1=4 with reclaim | runtime fail | 0 | staged tile footprint `131072` bytes |

Meaning: correctness and launch are unblocked for a bounded 2x2 candidate, but performance is not solved. Larger tiles
are blocked by LDS footprint, not register spills or waitcnt.

## Layer decision map

The tracer splits the problem into layers. The key rule is: do not optimize a lower layer until the layer above it has
produced a launchable candidate, and do not revisit NaN/hardware-fault theories while the current failure is a declared
resource footprint failure.

| Layer | Question | Current answer from tracer/runner | Status | Next decision |
| --- | --- | --- | --- | --- |
| L0. Route selection | Are we exercising the generated graph matmul path, not hand asm? | `prefill_v2_schedule_search.py WORKER=1 DEV=AMD:ISA` applies warmstart opts and launches generated native ISA. | Proved | Keep using this as the execution authority. |
| L1. Machine-code lifecycle | Does generated code have the hand-LDS2 lifecycle? | Synthetic full DBUF: `ds_store_b128=32`, `ds_load_b128=128`, WMMA operands LDS/LDS, inter-WMMA work=31. | Proved | Structural D7 remains the regression gate. |
| L2. Packed LDS store cleanliness | Does DBUF preserve packed stores instead of scalar LDS stores? | `m_up=1` DBUF now has scalar LDS stores=0 after B-tilekey matcher relaxation. | Proved | Keep `test_dbuf_withlocal_both_smaller_tile_keeps_packed_lds_stores`. |
| L3. Register pressure | Does full synthetic DBUF get through regalloc? | Requires `REGALLOC_ADDR_REMAT=1 REGALLOC_END_NO_SOURCE_LIVE=1`; then compiles. | Proved under bundle | Treat these flags as part of the native DBUF candidate until replaced by default policy. |
| L4. Accumulator storage | Does the backend avoid wasting LDS on accumulator backing? | 2x2 real shape fails at `69632` without reclaim, passes with `AMD_ISA_REG_ACCUM=1`. | Proved | Keep accumulator reclaim in the candidate bundle. |
| L5. Real-shape LDS footprint | Does a useful real prefill schedule fit under 64 KiB? | 2x2 fits only with reclaim and is slow; 2x4/4x2 are `98304`; 4x4 is `131072`. | Blocking | Need footprint reduction or shape selection. |
| L6. Native correctness | Does a launchable real candidate compute correctly? | 512x5120x5120 `u0=2,u1=2,loc=0,unr=8` returns `status=ok`. | Proved for bounded candidate | Use as correctness baseline, not performance target. |
| L7. Scheduler/waitcnt tuning | Is there a launchable performant candidate to tune? | No. Faster candidates do not launch. | Blocked by L5 | Defer waitcnt/scheduler tuning until below-limit footprint exists. |
| L8. Promotion/default policy | Can this replace current prefill route? | No. Correctness exists only for slow bounded candidate; faster shapes over-allocate LDS. | Blocked by L5/L7 | Keep flag-gated. |

This means the next work is not another tracer for operand origins. The tracer already named the active layer:

```text
L5: real-shape LDS footprint / tile-shape selection
```

## Phased execution scope

This is the ordered plan from the current blocker to a promotable fast-prefill path. Each phase has a sequential gate
that must pass before later phases are meaningful. Work listed as parallel can run while the main implementation is in
progress, but it must not be used to declare completion ahead of the sequence gate.

### Phase 0: lock the failure definition

Goal: make the current failure impossible to misread as a b128, waitcnt, hardware, or handwritten-asm delta.

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| Keep early A `LOC=2` WITH_LOCAL fail-closed unless explicitly unsafe | Sequence | main | worker returns `NotImplementedError: A WITH_LOCAL staging needs a multi-dim LDS key`, not wrong output |
| Preserve known-good controls | Parallel | test/probe | A `LOC=0` still `ok`; B DBUF `LOC=2` still `ok`; unit tests pass |
| Record final-stream evidence | Parallel | trace/doc | doc shows A global addr includes local-y while A LDS addr omits it |

Status: done for the early path. The guard exists; tests pass. The valid path is now the post-stage path, where staging
runs after the planned `LOCAL` opt and can see the local-y range.

### Phase 1: implement A multidim LDS key

Goal: make generated A staging preserve the same identity the hand LDS2 path preserves.

Required shape:

```text
A_LDS_addr = dbuf_slot*A_slot_bytes + lidx1*A_lidy_slice_bytes + lane_fragment_offset
```

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| Find the correct insertion point before/inside LOCAL range construction | Sequence | main | **Done:** `PREFILL_TC_LOCAL_STAGE_POST=1` now actually defers staging until after planned `LOCAL` |
| Add A-specific keyed layout, analogous to B tile-key but using A local-y identity | Sequence | main | **Done via post-stage generic WITH_LOCAL:** `TC_LOCAL_STAGE_POST` sees `AxisType.LOCAL` in A source ranges |
| Update allocation sizing for keyed A slices | Sequence | main | **Done for A-only:** A DBUF `LOC=2` uses `binary_group_segment_bytes=65536` and computes correctly |
| Audit B tile-key for reusable helper extraction | Parallel | review | optional helper plan; no refactor until A works |
| Add trace matcher for A LDS local-y identity | Parallel | trace/test | tracer can report whether A LDS store/read family carries local-y |

Status: passed for A-only. The fix was not a new synthetic A key; it was correcting the gating so `PREFILL_TC_LOCAL_STAGE_POST=1`
prevents early TC-time staging. The post matcher then sees:

```text
src0_ranges includes ((12, AxisType.LOCAL), 2)
stage_ranges = WARP + LOCAL
```

A-only `LOC=2` gates:

| Mode | Result |
| --- | --- |
| non-DBUF | `status=ok`, about 11.5 TFLOPS |
| DBUF | `status=ok`, about 7.7 TFLOPS, `binary_group_segment_bytes=65536` |

### Phase 2: re-enable safe packing over the corrected key

Goal: recover the pressure benefit of packed LDS stores without allowing aliasing.

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| Run A `LOC=2` without unsafe override | Sequence | main | **Done:** A-only non-DBUF worker `status=ok` |
| Run A `LOC=2` DBUF bundle | Sequence | main | **Done:** A-only DBUF worker `status=ok` |
| Keep proof-based b128 group pack fail-closed on missing identity | Sequence | main | no wrong-output path when key proof is absent |
| Inspect b64 fallback pressure | Parallel | probe | confirms whether b64 remains diagnostic-only or can be deleted/deprioritized |
| Unit tests for group pack multidim reject/accept | Parallel | test | tests cover missing-key reject and corrected-key accept if easy to isolate |

### Phase 3: compose A+B DBUF

Goal: both operands use LDS, both are correctly keyed, and DBUF cadence survives in the real generated route.

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| A+B non-DBUF sanity | Sequence | main | **Done:** `PREFILL_TC_LOCAL_STAGE=both LOC=2` worker `status=ok`, about 5.3 TFLOPS |
| A+B DBUF correctness | Sequence | main | **Done for bounded `unr=2`:** `u0=2,u1=2,loc=2` worker `status=ok`; default `unr=8` remains resource-blocked at `131072` |
| Structural trace | Sequence | trace | WMMA operands LDS/LDS, `ds_store_b128 > 0`, `ds_load_b128 > 0`, barriers present, inter-WMMA work visible |
| B regression matrix | Parallel | probe | B-only DBUF still `ok`; no scalar LDS-store regression |
| Address proof matrix | Parallel | probe | global and LDS address proof has zero violations for A-only, B-only, A+B |

### Phase 4: resource fit and schedule viability

Goal: find a real A+B DBUF candidate that fits under 64 KiB and is worth timing.

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| Recompute static LDS model after A keying | Sequence | main | table gate estimates match compiled `binary_group_segment_bytes` closely enough to prefilter |
| Run resource-filtered search | Sequence | harness | below-limit A+B candidates are timed; over-limit candidates are skipped before compile |
| One-operand fallback check | Parallel | search | confirms fallback is still slower or identifies a real win |
| Footprint-reduction branch F1/F4 | Parallel after Phase 1 | implementation/probe | demonstrates lower bytes with unchanged D7/correctness |

Phase 4 found a bounded A+B DBUF route. A keying is correct for A-only, and A+B DBUF now launches when `UNR` is reduced
to 2. The default/high-throughput `UNR=8` shape still does not fit:

| Candidate | `binary_group_segment_bytes` | Status |
| --- | ---: | --- |
| A-only DBUF `u0=2,u1=2,loc=2` | 65536 | correct, diagnostic edge of the limit |
| B-only DBUF `u0=2,u1=2,loc=2` | 32768 | correct |
| A+B DBUF `u0=2,u1=2,loc=2,unr=2` | 32768 | correct; about 8.1 TFLOPS on 4096, 8.9 TFLOPS on 5120 |
| A+B DBUF `u0=2,u1=2,loc=2,unr=4` | 65536 | exact-limit diagnostic edge; rejected by policy |
| A+B DBUF `u0=2,u1=2,loc=2,unr=8` | 131072 | resource-blocked |
| A+B DBUF `u0=4,u1=2,loc=2,unr=2` | 49152 | launches but wrong output (`rr=9.3e-01`) |
| A+B DBUF with `u1=4` | 81920+ | resource-blocked |

### Phase 5: performance tuning

Goal: move from "correct generated DBUF" to "fast prefill candidate."

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| Baseline pinned timing | Sequence | harness | no-staging, one-operand, A+B DBUF measured under the same clock policy |
| Scheduler/waitcnt tuning | Sequence after valid candidate | main | improvement without breaking D7 or correctness |
| Candidate sweep around winning tile | Parallel | search | table of TFLOPS, LDS bytes, correctness, trace status |
| Compare to hand LDS2 lifecycle | Parallel | trace/review | generated lifecycle has same high-level cadence, with expected codegen differences documented |

### Phase 6: promotion and regression gates

Goal: make the route maintainable and impossible to regress silently.

| Work | Parallel? | Owner type | 100% gate |
| --- | --- | --- | --- |
| Convert flags into policy or narrow defaults | Sequence | main | default behavior chosen only for shapes with passing resource/correctness gates |
| Regression tests | Parallel | test | fail-closed A multidim, A-keyed accept, B tile-key, DBUF structural gate |
| Docs and handoff | Parallel | doc | current commands, expected outputs, and reject cases are recorded |
| Cleanup unsafe flags | Sequence | main | unsafe escape hatches remain explicit diagnostics or are removed |

### Parallelization map

Work that can run now:

| Track | Work | Why independent |
| --- | --- | --- |
| P0 trace/test | Add a tracer/test for "A global has local-y but A LDS key does not" | Does not require the implementation fix |
| P1 B regression | Re-run B-only DBUF matrix and record resource/trace results | B path already has tile-key substrate |
| P2 resource model | Prepare the resource table format for post-A-key measurements | The exact bytes update later, but schema can be done now |
| P3 review | Audit B tile-key code for helper extraction to reuse in A | Can produce recommendations without editing critical path |

P2 minimum table contract after A keying:

| Field | Meaning |
| --- | --- |
| `stage` | `A`, `B`, or `both`; must match `PREFILL_TC_LOCAL_STAGE`. |
| `u0`, `u1`, `loc`, `unr` | Schedule knobs used for the compile/runtime row. |
| `static_dbuf_lds_estimate_bytes` | Prefilter estimate from the active model; never treat as authoritative. |
| `static_dbuf_lds_estimate_model` | Model tag such as `pre_a_key_current` or `post_a_key_v1`, so old estimates are not mixed with keyed-A measurements. |
| `a_lidy_slices`, `a_slice_bytes`, `b_tile_slices`, `dbuf_slots` | Minimal factors needed to explain keyed LDS growth after A local-y lands. |
| `binary_group_segment_bytes` | Authoritative ELF group-segment bytes from native compile. |
| `local_bytes` | Backend local allocation summary when available; keep beside ELF bytes to expose accumulator or metadata deltas. |
| `below_limit`, `over_limit`, `lds_limit_bytes` | Resource verdict fields used by the gate. |
| `status`, `message` | `static-over-limit`, `compile-ok`, or compile/runtime error details. |
| `measured` | Optional runtime result, present only for below-limit probes selected with `--resource-run-below-limit`. |

Command flow:

1. Before A keying, keep using the current gate only as a fail-closed prefilter:
   `python3 extra/qk/prefill_v2_schedule_table_gate.py --resource-search --resource-stages both,A,B --resource-u 2,4 --resource-loc 0,2 --resource-unr 8`.
2. After A keying, update the static model tag and factors first, then run the same command without runtime timing and compare every compiled row's `static_dbuf_lds_estimate_bytes` against `binary_group_segment_bytes`.
3. Only when the post-A-key estimate tracks compiled bytes closely enough to reject impossible rows, add `--resource-run-below-limit` to time below-limit candidates.

Work that must be sequential:

```text
S1. Implement A local-y LDS key
S2. Prove A-only non-DBUF correctness
S3. Prove A-only DBUF correctness
S4. Prove A+B DBUF correctness
S5. Recompute resource fit
S6. Tune scheduler/performance
S7. Promote/default
```

S4 now passes only for the bounded `u0=2,u1=2,loc=2,unr=2` candidate. Performance work can use that as the correctness
floor, but must not promote it as the fast route; every larger below-limit candidate must prove correctness first.

## L5 solution space

The valid next branches are mutually exclusive enough to test independently:

| Branch | Idea | Why it could work | First proof | Reject if |
| --- | --- | --- | --- | --- |
| F1. Fragment-scoped LDS allocation | Allocate only the WMMA fragment footprint needed by the current subtile/window, not the table-local tile footprint. | Current 2x4/4x2/4x4 fail because local bytes scale with staged tile count. | A+B DBUF real shape launches with `binary_group_segment_bytes < 65536` and keeps D7 clean. | Address proof or correctness fails, or scalar LDS stores reappear. |
| F2. Resource-aware tile selector | Select the largest DBUF tile shape whose estimated LDS footprint fits before compile. | 2x2 is correct; larger shapes fail only by footprint. A selector prevents invalid candidates. | Table gate filters 2x4/4x2/4x4 and times only below-limit candidates. | Best below-limit candidate remains slower than current baseline. |
| F3. One-operand DBUF fallback | Stage only A or only B through LDS DBUF when A+B exceeds budget. | One-operand candidates can fit far below 64 KiB in related probes. | Real-shape one-operand candidate launches, passes, and beats no-staging/native baseline. | It stays slower or reintroduces operand-origin/global-path hazards. |
| F4. Reduce slot duplication | Keep two-slot DBUF cadence but avoid duplicating full A+B table-local buffers. | Current exact-limit 2x2 and oversized 2x4/4x2 imply slot duplication is too coarse. | Same schedule has lower `local_bytes` with unchanged D7 slot/cadence. | Any slot aliasing breaks correctness. |
| F5. Below-limit schedule search | Expand the search grid around smaller native-ISA DBUF shapes and reject by resource estimate. | The current tested points are sparse. There may be a better below-limit shape than 2x2. | Automated sweep reports a correct candidate with better TFLOPS and `<65536` bytes. | No candidate beats baseline after resource filtering. |

Priority order:

1. F2 first, because it turns the current manual resource facts into an automatic gate and prevents wasting time on
   impossible schedules. **Done:** `prefill_v2_schedule_table_gate.py --resource-search` now prefilters the current DBUF
   staging footprint before full native compile.
2. F5 second, because it may find an existing below-limit win without changing lowering. **Done for current knobs:**
   A+B has one correct bounded candidate (`2x2/unr=2`) and one resource-fitting wrong candidate (`4x2/unr=2`).
3. F1/F4 third, because they are the primitive lowering fixes if the bounded candidate is too slow. **Active next step.**
4. F3 as a fallback if A+B DBUF is too expensive but one operand gives a net win. **Rejected for now:** one-operand DBUF
   is correct but slower than no-staging.

### F2/F5 result

The resource gate now performs a static prefilter for the current packed WITH_LOCAL DBUF staging model. For the
post-stage A+B path it uses the measured grid model `post_stage_both_grid_v2`:

```text
2x2/unr=2  -> 32 KiB
4x2/unr=2  -> 48 KiB
2x4/unr=2  -> 80 KiB
4x4/unr=2  -> 96 KiB
unr=4      -> at least 64 KiB edge
unr=8      -> at least 128 KiB
```

Candidates at or above 64 KiB are marked `static-over-limit` and skipped before full native compilation. Below-limit
candidates still compile and report authoritative ELF `binary_group_segment_bytes`.

Focused production result for `4096x4096`, `stage=both`, `u in {2,4}`, `loc=2`, `unr in {2,4,8}`:

| Candidate class | Resource result | Runtime result |
| --- | --- | --- |
| A+B `2x2/unr=2` | 32768, below limit | correct; about `8.1 TFLOPS` on 4096 and `8.9 TFLOPS` on 5120 |
| A+B `4x2/unr=2` | 49152, below limit | wrong output (`rr=9.3e-01`) |
| A+B `4x2/unr=2` with `PREFILL_TC_LOCAL_STAGE_SPLIT_POST_A=1` | 40960, below limit | wrong output (`rr=nan`) |
| A+B `4x2/unr=2` with targeted waitcnt | 49152, below limit | wrong output improves to `rr=2.2e-01`; scheduler on/off does not fix it |
| A+B `4x2/unr=2` with cooperative post but no GLOBAL | 0 LDS bytes | correct, about 22 TFLOPS, but WMMA operands are global/global; this is not the LDS DBUF route |
| A+B cooperative post with GLOBAL identity | 1179648 bytes | emits LDS but is resource-blocked; full GLOBAL tile identity is too large |
| A+B cooperative post with GLOBAL dropped | small enough | NaN; aliases too aggressively |
| A-only cooperative post with raw GLOBAL identity | 131072 bytes with `NBUF=2`, 65536 with `NBUF=1` | confirms raw A GLOBAL key is too large; `NBUF=1` still NaNs |
| Reject generic packed groups with discontinuous global constants | fallback path | `2x2` NaNs and `4x2` spills; fail-closing the unsafe pack is not enough |
| Load-side fixed A window offsets | diagnostic only | `+128` schedules by WMMA group or within group worsen `4x2` to `rr>1`; final WMMA order is not the missing key |
| A+B `2x2/unr=4` | 65536, exact-limit edge | skipped by policy |
| A+B `2x2/unr=8` | 131072 | skipped by policy |
| A+B with `u1=4` | 81920+ | skipped by policy |
| One-operand DBUF | below-limit in related probes | correct but slower than no-staging/native baseline |

Conclusion: L5 is no longer a hard launch/correctness blocker. It is now a performance and correctness frontier:
`2x2/unr=2` is the correctness floor, while `4x2/unr=2` names the next concrete correctness bug and `unr>=4` names the
remaining footprint problem. Splitting the route as early-B plus post-A reduces `4x2/unr=2` LDS bytes from 49152 to
40960, but still fails correctness, so the missing primitive is not just pass placement. Targeted waitcnt improves the
error but does not fix it, and scheduler on/off has no effect. The failure is a semantic address/layout identity issue
for the extra A upcast group under A+B composition.

### F1/F4 implementation scope

Target primitive:

```text
current:  allocate LDS for all staged table-local/unrolled fragments in the body
target:   allocate LDS only for the A/B WMMA fragments consumed by the current producer/consumer window
```

Required invariants:

| Invariant | Why |
| --- | --- |
| no live-fragment aliasing | Prior attempts that dropped UNROLL/GLOBAL identity produced NaNs. |
| packed-store preserved | `ds_store_b128` must remain, scalar LDS stores must stay zero. |
| two-slot identity preserved | DBUF slot 0/1 must remain visible in address families/cadence. |
| below-limit A+B | `binary_group_segment_bytes < 65536`, not equal. |
| native correctness first | worker status `ok` before any waitcnt/scheduler tuning. |

First implementation target:

1. Add a fragment-window identity to the cooperative A+B staging path instead of using the full table-local tile identity.
2. Keep UNROLL dimensions that are simultaneously live; only reuse LDS slots after a proof that the producing fragment is
   consumed before overwrite.
3. Add a structural gate that the new A+B candidate is below limit and D7-clean before launching.
4. Launch `4096x4096` with the central worker; only then test `5120x5120`.

Diagnostic split probe:

```text
PREFILL_TC_LOCAL_STAGE=both
PREFILL_TC_LOCAL_STAGE_POST=1
PREFILL_TC_LOCAL_STAGE_SPLIT_POST_A=1
```

This makes early TC staging handle B only and the post-stage pass handle A only. It preserves the known-good
`2x2/loc=2/unr=2` route, and it shrinks `4x2/loc=2/unr=2` to 40960 bytes, but the `4x2` result is still NaN. Keep this
as a diagnostic footprint lever, not a promotion route.

Rejected implementation probes:

| Probe | Result | Reason to reject |
| --- | --- | --- |
| Add A `UPCAST` directly to generic `bufferize` ranges | compile-time reshape mismatch in rangeify | `bufferize` cannot accept this identity shape directly |
| Cooperative post without GLOBAL | correct and fast only because it skips LDS; final stream has no `ds_store_b128`/`ds_load_b128` | not the hand-LDS2/DBUF route |
| Cooperative post with full GLOBAL identity | `binary_group_segment_bytes=1179648` | correct identity class but impossible LDS footprint |
| Cooperative post with GLOBAL dropped | NaN | slot reuse aliases live fragments |
| A-only cooperative post with raw GLOBAL identity | `131072` bytes with DBUF, `65536` with single-buffer; single-buffer still NaNs | raw GLOBAL is neither small enough nor sufficient |
| Discontinuity reject in generic b128 pack | `2x2` NaNs, `4x2` spills | the graph-level address still lacks a replacement key; rejecting the pack does not create a legal fallback |
| Load-side immediate offset by WMMA order | `4x2` gets worse (`rr=1.2..1.3`) | the key must come from producer/consumer identity before final WMMA order, not a post-hoc load offset |
| Explicit scalar A tile-key UOp path | compile hangs/pathological graph growth | conceptually right key, wrong layer/form; would also risk losing packed b128 stores |

Next viable primitive:

```text
packed A fragment/window key
  key = (dbuf_slot, local_y, A_upcast_fragment, compressed/global-window class)
  not key = full GLOBAL tile
  not key = generic bufferize range tuple
  not key = scalar per-element UOp staging
```

This needs to be implemented where the existing packed b128 path can carry a proof of the A fragment/window identity,
so it preserves `ds_store_b128` and avoids the scalar graph blow-up. The key missing design piece is the compressed
global-window class: enough to distinguish the live A fragments that currently alias, but bounded so it does not expand
to the raw `GLOBAL` range. Completion for this primitive is:

| Gate | Required |
| --- | --- |
| correctness | `4x2/loc=2/unr=2` returns `status=ok` with `PREFILL_TC_LOCAL_STAGE=both` |
| LDS footprint | `binary_group_segment_bytes < 65536` |
| structure | final stream still has `ds_store_b128 > 0`, `ds_load_b128 > 0`, scalar LDS stores zero |
| regression | `2x2/loc=2/unr=2` remains correct |
| performance | beats the current bounded `2x2` LDS route and is compared against the global/global control |

### Path 2 sidecar resource model: bounded A fragment-window identity

This sidecar model is for the future compressed A window key only. It must not be used to justify the current wrong
`4x2` route, and it must stay separate from renderer/codegen changes until the identity proof exists.

Known anchors:

| Route | Bytes | Meaning |
| --- | ---: | --- |
| raw A `GLOBAL` identity, DBUF/NBUF=2 | 131072 | impossible footprint; preserves too much identity |
| raw A `GLOBAL` identity, `PREFILL_DBUF_NBUF=1` | 65536 | still exact-limit and still wrong/NaN in probes |
| current bounded A+B `2x2/loc=2/unr=2` | 32768 | correctness floor |
| current A+B `4x2/loc=2/unr=2` | 49152 | fits, but wrong output; names the missing A window identity |

Definitions for the proposed sidecar estimate:

```text
NBUF                 = PREFILL_DBUF_NBUF if DBUF else 1        # default 2
A_WINDOW_QUANTUM     = 4096 bytes
A_fragments          = u0
B_fragments          = u1 for stage=both, else 0
raw_A_GLOBAL_bytes   = 65536 * NBUF
compressed_A_bytes   = NBUF * A_WINDOW_QUANTUM * A_fragments
compressed_AB_bytes  = NBUF * A_WINDOW_QUANTUM * (A_fragments + B_fragments)
```

For the first Path 2 primitive, accept only the frontier where the B side is already bounded:

```text
stage in {A,both}
loc == 2
u1 == 2 for stage=both, unless a separate bounded B-window model is enabled
live_UNR_window <= 2, proven by producer/consumer lifetime rather than assumed from schedule UNR
binary_group_segment_bytes < 65536, not equal
```

Expected byte bounds under `future_compressed_a_window_key_v1`:

| Candidate | Estimate | Gate |
| --- | ---: | --- |
| A-only `u0=2,loc=2` | 16384 | accepted if A-only correctness/structure pass |
| A-only `u0=4,loc=2` | 32768 | accepted if extra A upcast has distinct window identity |
| A+B `2x2,loc=2` | 32768 | must remain correct; regression floor |
| A+B `4x2,loc=2` | 49152 | primary Path 2 target; must flip from wrong to correct |
| A+B `2x4,loc=2` | 49152 by A model alone | statically filtered until B-window proof exists |
| A+B `4x4,loc=2` | 65536 | rejected at exact limit even with B proof |
| any raw A `GLOBAL` key | `65536 * NBUF` | rejected; too large for DBUF and not sufficient with `NBUF=1` |

Candidate table fields now required from the gate:

| Field | Required meaning |
| --- | --- |
| `static_dbuf_lds_estimate_model` | `future_compressed_a_window_key_v1` for this sidecar model |
| `static_dbuf_lds_estimate_bytes` | formula result; prefilter only |
| `dbuf_slots`, `a_window_quantum_bytes` | factors used to explain the bound |
| `a_upcast_fragments`, `b_tile_fragments`, `a_lidy_slices` | identity dimensions carried by the candidate |
| `schedule_unr`, `assumed_live_unr_window` | distinguishes total schedule unroll from live LDS window |
| `raw_a_global_identity_bytes` | comparison against the rejected full-identity route |
| `candidate_filter`, `candidate_filter_reasons` | fail-closed filter verdict before native compile |
| `binary_group_segment_bytes`, `local_bytes` | authoritative compile result for accepted rows |
| `below_limit`, `over_limit`, `status`, `message` | resource verdict and skip/compile reason |

Completion table for Path 2:

| Step | Candidate | Required result |
| --- | --- | --- |
| P2.0 model only | `PREFILL_DBUF_A_WINDOW_KEY_MODEL=1` resource search | rejected rows show filter reasons; accepted rows show the estimate factors |
| P2.1 regression floor | A+B `2x2/loc=2/unr=2` | `status=ok`, `binary_group_segment_bytes=32768`, D7 clean |
| P2.2 target correctness | A+B `4x2/loc=2/unr=2` | `status=ok`, `binary_group_segment_bytes=49152` or less, finite RMSE |
| P2.3 no raw fallback | A raw `GLOBAL` identity | statically rejected or compile resource-blocked; never promoted |
| P2.4 no exact-limit promotion | any estimate or ELF size `>=65536` | skipped before runtime timing |
| P2.5 structure | target candidate final stream | LDS/LDS WMMA origins, `ds_store_b128 > 0`, `ds_load_b128 > 0`, scalar LDS stores zero |
| P2.6 performance | target vs bounded `2x2` and global/global control | beats bounded `2x2`; promotion only after correctness/resource/structure gates |

Resource sidecar command:

```bash
PYTHONPATH=. PREFILL_DBUF=1 PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_DBUF_A_WINDOW_KEY_MODEL=1 \
python3 extra/qk/prefill_v2_schedule_table_gate.py \
  --resource-search --resource-stages both,A --resource-u 2,4 --resource-loc 2 --resource-unr 2,4,8
```

### F1/F4 first implementation attempt

Isolation on `4096x4096`, `M=512`, `u0=2,u1=2,loc=2,unr=8`:

| Variant | Result | Meaning |
| --- | --- | --- |
| `PREFILL_TC_LOCAL_STAGE=A`, `WITH_LOCAL=0`, no packed bridge | `ok`, ~27-28 TFLOPS, `binary_group_segment_bytes=0` | Fast, but not the LDS DBUF route; it is effectively the register/no-LDS path. |
| `PREFILL_TC_LOCAL_STAGE=both`, `WITH_LOCAL=0`, no packed bridge | `ok`, ~27 TFLOPS, `binary_group_segment_bytes=0` | Same: useful baseline, not a staged LDS fix. |
| `WITH_LOCAL=1`, `PREFILL_LDS_PACK_WITHLOCAL_B128=0` | `NotImplementedError: Inc 0: no spills` | Safe scalar/vector local staging is too high pressure. |
| early `WITH_LOCAL=1`, `PREFILL_LDS_PACK_WITHLOCAL_B128=1` | used to compile but return wrong output for A/A+B at `loc=2`; now fail-closed for A multi-dim local schedules | Early TC-time staging happens before the planned `LOCAL` opt exists. |
| post-stage `WITH_LOCAL=1`, `PREFILL_TC_LOCAL_STAGE_POST=1` | A-only `loc=2` non-DBUF and DBUF now return `ok`; A+B non-DBUF returns `ok`; A+B DBUF is resource-blocked at 131072 bytes | Deferring staging until after planned `LOCAL` fixes the A keying semantics. |

Root of the fixed sub-blocker:

```text
At loc=2 the final stream has local_size=[32,2,1]. The A global load address includes local-y:

  lidx1 * 131072

but the A LDS store/read key is only the warp/lane-derived address. In the failing stream this showed up as A stores
with a base like:

  lidx0 * 512 + const

so the two local-y rows load different A data and alias into the same LDS slots. The fix is to prevent early TC-time
staging when `PREFILL_TC_LOCAL_STAGE_POST=1`, so the post matcher runs after `LOCAL:0:2` has introduced the real
`AxisType.LOCAL` range. Then A stage ranges become `WARP + LOCAL` instead of just `WARP`.
```

Code changes from this pass:

```text
PREFILL_LDS_PACK_WITHLOCAL_B128_GROUP_ONLY=1
PREFILL_TC_LOCAL_STAGE_A_MULTIDIM_UNSAFE=1
PREFILL_TC_LOCAL_STAGE_POST=1
```

`GROUP_ONLY` disables the unsafe per-store shortcut so probes can force only proof-based group packing. A WITH_LOCAL
with multi-dimensional workitems still raises `NotImplementedError: A WITH_LOCAL staging needs a multi-dim LDS key`
on the early path unless `PREFILL_TC_LOCAL_STAGE_A_MULTIDIM_UNSAFE=1` is set. The supported path is
`PREFILL_TC_LOCAL_STAGE_POST=1`, which makes the key visible before staging.

Completed A-keying gates:

| Gate | Result |
| --- | --- |
| A-only `loc=2`, non-DBUF, post-stage | `ok`, about 11.5 TFLOPS |
| A-only `loc=2`, DBUF, post-stage | `ok`, about 7.7 TFLOPS, `binary_group_segment_bytes=65536` |
| B-only `loc=2`, DBUF | `ok`, about 22 TFLOPS, `binary_group_segment_bytes=32768` |
| A+B `loc=2`, non-DBUF, post-stage | `ok`, about 5.3 TFLOPS |
| A+B `loc=2`, DBUF, post-stage | launch blocked: `binary_group_segment_bytes=131072` |

Next primitive is back to footprint reduction:

1. Avoid full A+B two-slot duplication for the current `LOC=2` layout.
2. Find a fragment/window-scoped layout whose A+B DBUF descriptor is `<65536`.
3. Preserve the post-stage A keying and B tile-key identity while reducing bytes.
4. Only after A+B DBUF launches should scheduler/waitcnt tuning resume.

## Current blockers

| Blocker | Evidence | Required fix |
| --- | --- | --- |
| B0: real-shape larger-tile LDS footprint | with `AMD_ISA_REG_ACCUM=1`, `unr=4/8` and `u1=4` shapes are at/over the 64 KiB edge | shrink staged tile footprint or change tiling; scheduler knobs alone do not fit |
| B1: bounded launch requires accumulator reclaim | 2x2 is `69632` without reclaim and launches with `AMD_ISA_REG_ACCUM=1` | keep reclaim in the native DBUF bundle or reduce local bytes below 64 KiB |
| B2: bounded 2x2 is correct but slow | `2x2/loc=2/unr=2` is about 8-9 TFLOPS | performance work must preserve <=64 KiB LDS while increasing useful work/overlap |
| B3: resource-fitting 4x2 is wrong | `4x2/loc=2/unr=2` is 49152 bytes but returns `WRONG rr=9.3e-01`; split early-B/post-A is 40960 bytes but NaN | diagnose extra-A-upcast address/layout identity before using it as the next performance tile |
| B4: non-DBUF has no overlap | both-side packed route has LDS/LDS operands but `work_between_wmmas=0` | only solved by DBUF/software-pipeline ordering, not waitcnt |

## Completion definition

The generated route is structurally hand-LDS2-like when this command passes D7:

```bash
PYTHONPATH=. AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 \
PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind generated
```

Required structural output:

| Gate | Required |
| --- | --- |
| operands | all WMMA `src0/src1` from `ds_load_b128` |
| stores | `ds_store_b128 > 0`, `ds_store_b16 == 0`, `ds_store_b32 == 0` |
| barriers | `s_barrier > 0` |
| cadence | `global_work_between_wmmas > 0` |
| compile | no `Inc 0: no spills` |

This structural gate now passes with the remat/live-range flags above. The remaining completion definition is native
execution:

| Gate | Required |
| --- | --- |
| launch | real prefill shape launches under `DEV=AMD:ISA` without `group_segment_size` failure |
| correctness | centralized worker status `ok`, finite output, RMSE below harness threshold |
| footprint | `binary_group_segment_bytes <= 65536` |
| performance | recover from the 2x2 correctness baseline toward the old fast-prefill target without reintroducing scalar LDS stores |

Current stopping point: the primitive route is real and correct at `2x2/loc=2/unr=2` with accumulator reclaim, but it is
slow. The next fix should target either the `4x2/unr=2` correctness failure or the staged-tile footprint for `unr>=4`,
not NaN/hardware diagnosis.
