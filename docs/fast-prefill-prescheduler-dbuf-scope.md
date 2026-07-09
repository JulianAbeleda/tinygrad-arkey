# Fast Prefill Pre-Scheduler DBUF Scope

Date: 2026-07-07.

## Objective

Scope the remaining generated-machine-code work from the current 4x4 A+B LDS-staged substrate up to, but not including,
scheduler/waitcnt tuning.

The endpoint of this scope is:

```text
generated 4x4 WMMA route
  -> both A and B staged through wide LDS
  -> two LDS slots under PREFILL_DBUF=1
  -> prologue/body/tail or equivalent modulo cadence visible
  -> verifier-clean
  -> no spills
  -> GPU-correct
  -> structurally ready for scheduler/waitcnt tuning
```

Scheduler tuning starts only after this scope proves the pipeline shape exists. This scope does not require final TFLOPS
win or non-full waitcnt selection.

## Current Starting Point

The old structural blocker is fixed:

| Gate | Current result |
|---|---|
| A-only LDS packed | PASS: `ds_store_b128=8`, `ds_load_b128=8`, no scalar LDS stores. |
| A-only with B tile-key flag present | PASS after scoping devectorizer LOCAL pointer grouping disable to B tile-key slot `993`. |
| B-only tile-key bridge | PASS: `ds_store_b128=8`, `ds_load_b128=8`, no scalar LDS stores. |
| Both A+B LDS native structural | PASS: `ds_store_b128=16`, `ds_load_b128=16`, no scalar LDS stores, no spill. |
| Both central correctness | PASS, finite, same accepted RMSE envelope. |
| Both `SPEC=1` | PASS. |

The remaining pre-scheduler problem is DBUF structure, not basic A/B LDS staging.

## Current DBUF Findings

Latest D2/D3/D5/D6 checks established the current DBUF state:

| Gate | Result | Meaning |
|---|---|---|
| DBUF central correctness | PASS: finite, `rel_rmse_vs_ref=0.00020765016961377114`. | The DBUF flag is not currently a numeric/correctness blocker in the central route-bound harness. |
| DBUF native structural | PASS: final stream emits, `REGALLOC_SPILLS: count=0 stack_size=0`. | Native DBUF is no longer blocked at regalloc. |
| DBUF pressure before DBUF bridge widening | `REGALLOC_DEBUG` reported peak `249` live VGPRs, dominated by `V_IADD` and `V_OFFSET`. | The peeled/two-phase graph expanded address/staging live ranges too broadly. |
| DBUF pressure after bridge widening/address threading and B packed tuple ordering | Both DBUF peak is now `60` live VGPRs. | DBUF pressure is in the no-spill band. |
| B tile-key slot math | Fixed at codegen layer: B tile-key local slot `993` now uses `PREFILL_DBUF_NBUF()` and `(kr % nbuf)` slot indexing. | B tile-key no longer lacks DBUF slot identity in the graph. |
| B tile-key bridge under DBUF | Fixed at renderer layer: the B gather bridge now accepts `16 x GROUP(N stores)` where `N >= 4` and `N % 4 == 0`, not just `N == 4`. | DBUF's two phases no longer bypass B packed `GATED_STORE_B128` lowering. |
| Probe readiness | Strengthened `lds_address_families`, `wmma_lds_operand_families`, `dbuf_gate_summary`, and `resource_summary`. | Probe now separates weak address-family evidence from strict two-operand slot proof and reports LDS group-segment bytes. |

Historical route-bound DBUF probe classification, before the active-shape probe separated current-slot LDS loads from
future-slot staging:

```json
"dbuf_gate_summary": {
  "D2_two_slot_identity": {
    "ok": false,
    "weak_addr_family_ok": true,
    "proof_strength": "addr_register_family_only",
    "store_family_count": 17,
    "load_family_count": 8,
    "src0_lds_family_count": 2,
    "src1_lds_family_count": 1,
    "strict_requirement": "store/load families >= 2 and both WMMA operands observe >= 2 LDS load address families"
  },
  "D3_cadence": {
    "ok": true,
    "prologue_has_staging": true,
    "body_has_next_slot_work": true,
    "tail_region_present": true,
    "wmma_region_count": 32
  },
  "D7_scheduler_readiness": {
    "ok": false,
    "reason": "strict two-operand slot proof is not established",
    "wmma_operands_from_lds": true,
    "scalar_lds_store_count": 0,
    "next_slot_work_near_compute": true
  }
}
```

This classification is superseded for scheduler-readiness. The old broad cadence check counted `ds_load_b128` between
WMMAs as body work. That is current-slot consumption, not next-slot staging. The active generated `2x2` route now reports:

```text
D3_cadence.ok=false
body_has_next_slot_work=false
body_regions_with_current_slot_lds_load=true
D7_scheduler_readiness.ok=false
next_slot_work_near_compute=false
```

The next implementation step is therefore D3, not waitcnt tuning: make generated code expose body `global_load_b128` /
`ds_store_*` next-slot work before current-slot compute. Slot proof remains required, but a scheduler cannot use it until
the body contains future staging work.

Current route-bound resource summary with both operands staged:

```json
{
  "local_bytes": 12288,
  "reg_bytes_per_thread": 512,
  "n_threads": 32,
  "group_segment_unreclaimed_bytes": 28672,
  "group_segment_estimated_bytes": 28672,
  "binary_group_segment_bytes": 28672,
  "over_limit": false
}
```

With `AMD_ISA_REG_ACCUM=1`, the same route-bound descriptor drops to `12288` bytes, proving the accumulator reclaim path
works for accumulator-only `DEFINE_REG` storage.

Latest pressure classification:

| Probe | Peak | Dominant live classes | Status |
|---|---:|---|---|
| Historical DBUF A-only | 89 | `V_OFFSET`, `MOV`, `V_IADD` | Superseded by address-remat and bridge fixes. |
| Historical DBUF B-only after bridge widening | 89 | `V_OFFSET`, `MOV`, `V_IADD` | Superseded. |
| DBUF both current candidate | 60 | `V_IADD`, `V_OFFSET`, small address set | PASS no-spill. |

Rejected/insufficient attempt:

| Attempt | Result | Decision |
|---|---|---|
| Memoize dependency-free `_build_wmma_tile` fragment loads through `_pack_frag`. | Increased pressure by keeping resident MOV/DS_LOAD packs live longer. | Reverted; DBUF needs shorter address/load live ranges, not more residency. |
| Attach K-chain dependencies to b128 load indexes before `isel_index`. | Small B-only reduction only; both still peaks at 137. | Keep only if it remains regression-free; not sufficient as the DBUF fix. |

Primitive that closed this scope:

```text
DBUF required allocator address rematerialization, late DS immediate handling, B tile-key DBUF slot math, and an ordered
packed B store tuple so the b128 data span matches the constrained VGPR order.
```

## Out Of Scope

- Handwritten assembly or route-local custom instruction streams.
- Scheduler latency tuning.
- Targeted/non-full waitcnt policy.
- Final TFLOPS promotion.
- Default-on route selection.
- Direct B `global_load_b128` primitive, unless it is needed to make DBUF fit before scheduler tuning.

## Required Machine Shape Before Scheduler Tuning

The generated stream must expose a real two-slot LDS pipeline shape:

```text
prologue:
  load A/B tile 0 from global
  store A/B tile 0 to LDS slot 0
  barrier / slot visibility edge

body for k tile i:
  start loading A/B tile i+1 from global
  load A/B tile i from LDS slot i&1
  compute WMMA tile i
  store A/B tile i+1 to LDS slot (i+1)&1
  barrier / slot safety edge before that slot is consumed

tail:
  load final staged slot
  compute final WMMA tile
```

Before scheduler tuning, conservative waits are allowed. What must be present is the structural opportunity: two distinct
slot identities and a body where next-slot memory/staging is topologically in the same straight-line region as current-slot
compute.

## Exhaustive Work Packages

### D0. Lock The Current A+B LDS Substrate

Purpose: prevent regressions while DBUF work changes loop/staging structure.

Acceptance:

| Gate | Required evidence |
|---|---|
| D0.1 Unit | `PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py` passes. |
| D0.2 A composition | `local-stage=a` with `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` remains `ok=true`, `ds_store_b128=8`, no scalar LDS stores. |
| D0.3 B bridge | `local-stage=b` remains `ok=true`, `ds_store_b128=8`, no scalar LDS stores. |
| D0.4 Both LDS | `local-stage=both` remains `ok=true`, `ds_store_b128=16`, `ds_load_b128=16`, no scalar LDS stores. |
| D0.5 Correctness | central route-bound `local-stage=both` remains finite/pass. |
| D0.6 Verifier | `SPEC=1` both structural probe remains clean. |

### D1. Define The Authoritative DBUF Candidate Configuration

Purpose: avoid testing many historical DBUF branches interchangeably.

Candidate should start from the known-good generated A+B LDS substrate:

```text
PREFILL_TC_LOCAL_STAGE=both
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1
PREFILL_LDS_PACK_WITHLOCAL_B128=1
AMD_ISA_WMMA_B128_FRAG=1
PREFILL_DBUF=1
```

Acceptance:

| Gate | Required evidence |
|---|---|
| D1.1 Single flag bundle | Probe/docs define one blessed DBUF candidate command. |
| D1.2 Default safe | With `PREFILL_DBUF=0`, current A+B LDS result is unchanged. |
| D1.3 Fail closed | If a kernel has no suitable WMMA/GEMM reduce axis, DBUF peel is a no-op. |

### D2. Prove Reduce Peel And Two-Slot Identity

Purpose: prove `PREFILL_DBUF=1` creates two logical LDS slots rather than just a larger unused local allocation.

Current substrate:

- `postrange.py::_prefill_dbuf_peel` applies a guarded `UNROLL(..., 2)` on a const-even WMMA/GEMM reduce axis.
- `_tc_local_stage_coop_operand` already has `(kr % nbuf)` slot math for the cooperative path.
- The current passing A/B route uses `WITH_LOCAL` plus B tile-key, so slot identity must be verified on the actual selected path, not only the older cooperative diagnostic branch.

Acceptance:

| Gate | Required evidence |
|---|---|
| D2.1 Peel observed | With `PREFILL_DBUF=1`, the route has the expected unroll/peeled K phase or equivalent two-phase body. |
| D2.2 Slot 0/1 observed | Probe can identify two disjoint LDS slot address families for A and B. |
| D2.3 A and B both slotted | Slot identity applies to both operands, including B tile-key layout. |
| D2.4 Slot counts sane | `ds_store_b128`/`ds_load_b128` counts increase as expected for the peeled body without scalar LDS fallback. |
| D2.5 No alias | Slot `(k+1)&1` stores cannot clobber slot `k&1` loads before the current WMMA consumes them. |

Implementation notes:

- If current B tile-key path bypasses `_tc_local_stage_coop_operand` slot math, add equivalent slot indexing to
  `_tc_local_stage_b_src` or factor a shared slot-index helper.
- Do not rely on LDS allocation size alone. The acceptance signal is instruction/address identity.

### D3. Build Prologue/Body/Tail Cadence

Purpose: turn two slots into a visible software pipeline shape.

Current test-backed status:

| Fact | Evidence |
|---|---|
| Current-slot LDS loads are already in the body. | `test_dbuf_withlocal_both_currently_has_only_current_lds_loads_between_wmmas` passes. |
| Future-slot staging is absent from the body. | Same test asserts no `global_load_b128` / `ds_store_*` between adjacent WMMAs. |
| The failure is pre-scheduler. | `test_dbuf_withlocal_both_scheduler_off_still_lacks_future_staging` passes. |
| Existing address-in-loop knob is insufficient. | `test_dbuf_withlocal_both_addr_inloop_flag_does_not_create_future_staging` passes. |
| Desired D3 behavior is locked as the next red test. | `test_dbuf_withlocal_both_d3_target_future_staging_between_wmmas` is `expectedFailure`. |

Required cadence:

```text
slot0 fill
barrier
slot1 prefetch/fill work appears in the same body as slot0 compute
slot0 ds_load -> WMMA
slot1 store
barrier/edge
slot1 ds_load -> WMMA
...
```

Acceptance:

| Gate | Required evidence |
|---|---|
| D3.1 Prologue exists | First slot is filled before the first staged WMMA consumes LDS. |
| D3.2 Body exists | At least one steady-state region contains current-slot `ds_load_b128`/`v_wmma` and next-slot global/staging work. |
| D3.3 Tail exists | Final staged slot is consumed without reading uninitialized LDS. |
| D3.4 Probe visibility | `native_isa_l4_stream_probe.py` reports slot regions and whether next-slot work appears between WMMA groups. |
| D3.5 Conservative correctness | The cadence is correct even with full-drain waits; performance is not judged yet. |

Implementation routes to evaluate in order:

| Route | Layer | Description | Pros | Risks | Stop/acceptance |
|---|---|---|---|---|---|
| D3-A. Graph lifecycle split | `tinygrad/codegen/opt/postrange.py` | Split the staged store graph into prologue/body/tail around the peeled K phase, so one future slot remains topologically near current WMMA. | Correct layer; scheduler sees real work; keeps renderer simple. | Hardest to express verifier-clean effect ordering; may perturb local staging beyond prefill. | Accept if `expectedFailure` flips without scalar LDS fallback or verifier issues. |
| D3-B. Pre-isel pack ordering | `tinygrad/renderer/isa/amd.py` pre-isel packers | Preserve/stitch dependencies in `_pack_withlocal_lds_stores` / `_pack_b_tilekey_lds_stores` so future-slot stores are not all hoisted before the barrier. | Smaller patch surface; close to current packed-store lowering. | Cannot create future global loads if graph has already grouped all producers; can become an ordering hack. | Accept only if `native_isa_l4_stream_probe.py` reports body `global_load_b128` or `ds_store_*`, not just delayed `ds_load_b128`. |
| D3-C. Final-ISA software pipeline | `tinygrad/renderer/isa/amd.py` after isel/pre-regalloc | Pattern-match prologue stores/loads and rewrite final instruction order into prologue/body/tail. | Could be narrow and observable. | High correctness risk around barriers, waitcnt, register lifetimes; not primitive unless A/B prove impossible. | Defer until D3-A/B fail with a named verifier/codegen limitation. |

Updated route classification after tests:

| Route | Status | Why |
|---|---|---|
| D3-A0: plain `AFTER(WMMA, token)` graph anchor | Rejected | Real generated WMMA wrapped in `AFTER` failed before final instruction stream. This is not a legal anchor shape in the current pipeline. |
| D3-A1: operand-level future store group | Rejected as insufficient | It can be verifier-clean after the PTRCAT guard, but it anchors before the consuming WMMA and hoists to prologue. |
| D3-A2: first-class post-WMMA graph/effect token | Primitive design target | This preserves lifecycle semantics in graph/IR, but requires lowering support. It is the clean architectural fix if we are willing to add a new UOp/effect contract. |
| D3-A3: lowering-visible future-stage metadata consumed by `isel_wmma` | Next test path | This is the smallest way to prove the cadence with existing lowering hooks. It uses the real prior-WMMA WAR guard already created in `isel_wmma`, while keeping the future-stage intent explicit on the graph object rather than inferred from final ISA. |
| D3-C: final instruction reordering | Still rejected | It would reorder already-lowered instructions without owning DBUF epoch semantics; keep as last resort only. |

Restructured next path:

```text
P0. Keep the locked red/green tests.
    PASS today:
      - D2 slot identity true.
      - D3 false for default and D3-A flag.
      - PTRCAT verifier obstruction guarded.

P1. Metadata microprobe, no full DBUF semantics.
    Add a tag/sidecar on WMMA operands that says:
      "there exists future-stage work for the opposite DBUF slot."
    In `isel_wmma`, consume that marker only to insert a harmless wide staging-shaped dependency near the existing
    prior-WMMA reload guard.

    Acceptance for the microprobe:
      - compiles with SPEC=1,
      - produces at least one `global_load_b128` or `ds_store_b128` between adjacent WMMAs under a new probe flag,
      - does not allow scalar/narrow LDS stores,
      - D3-A expectedFailure flips only under the probe flag.

P2. Replace harmless/probe staging with real future-slot producer.
    Use the same `isel_wmma` anchor, but feed it actual `(kr+1)%nbuf` future-slot staging metadata from postrange.
    This is correctness work and must go through central GPU correctness before scheduler tuning.

P3. Promote to first-class graph token only if P1/P2 proves metadata is too hidden.
    If metadata becomes brittle, add an explicit post-WMMA effect UOp/lowering contract. Do not use plain AFTER(WMMA).
```

Design rule:

```text
The primitive semantic owner is still the graph/codegen lifecycle.
The practical first test is lowering-visible metadata because the only currently valid post-current-WMMA anchor exists
inside `amd.py::isel_wmma`.
```

P1 concrete implementation sketch:

| Step | File/function | Change | Acceptance |
|---|---|---|---|
| P1.1 Mark | `postrange.py::_tc_local_stage_coop_operand` | Under `PREFILL_DBUF_D3A_POST=1`, attach a metadata tag to staged WMMA operand carriers or proof tags saying this operand has DBUF future-stage eligibility. Do not add future stores yet. | No instruction stream change except metadata; existing tests unchanged. |
| P1.2 Detect | `amd.py::isel_wmma` / `_wmma_frag_proof_*` | Detect that both A and B current operands came from DBUF LDS proof tags and extract enough identity: role, buffer id, slot, `kr`, tile index, tile elems. | Debug-only dump can prove detection on the active 2x2 route. |
| P1.3 Place harmless wide marker | `amd.py::isel_wmma` | Under the same flag, insert a temporary wide staging-shaped `global_load_b128`/`ds_store_b128` dependency on the accumulate-tile `dep=(prev.src[0],)` path, before the next tile's fragment reloads. It may duplicate current source for the microprobe; correctness is not claimed yet. | `test_dbuf_withlocal_both_d3a_flag_future_staging_between_wmmas` flips, scalar/narrow stores remain zero, `SPEC=1` probe compiles. |
| P1.4 Replace marker with real producer | `postrange.py` + `amd.py` | Carry real future global source/index metadata for `(kr+1)%nbuf`; `isel_wmma` emits the actual next-slot stage work at the P1.3 anchor. | Central correctness plus D3 true. |

P1 result:

| Gate | Result |
|---|---|
| P1 metadata/detection | PASS by active-route LDS-fragment detection under `PREFILL_DBUF_D3A_POST=1`. The strict proof-key helper was too narrow for this path because the route carries normalized LDS address proof rather than identical lane proof tags. |
| P1 marker placement | PASS. `amd.py::isel_wmma` now inserts a flag-gated duplicate `global_load_b128` marker on the accumulate-tile prior-WMMA path for DBUF LDS-backed A/B operands. |
| D3-A unit | PASS. `test_dbuf_withlocal_both_d3a_flag_future_staging_between_wmmas` is now a normal green test. Default D3 remains an expected failure. |
| SPEC/native probe | PASS. With `SPEC=1 PREFILL_DBUF_D3A_POST=1`, `D2_two_slot_identity.ok=true`, `D3_cadence.ok=true`, `D7_scheduler_readiness.ok=true`, and `scalar_lds_store_count=0`. |

P1 caveat:

```text
This is a marker probe only. It proves the correct lowering-time anchor exists and can place wide memory work in the
WMMA body. It does not yet stage the real `(kr+1)%nbuf` A/B tile, so it is not a correctness/performance solution.
```

P2 next:

```text
Replace the duplicate global-load marker with real future-slot staging metadata:
  postrange exports future source/index identity for each DBUF LDS operand;
  isel_wmma emits global_load_b128 -> ds_store_b128 to the opposite LDS slot at the proven anchor;
  run central GPU correctness before any scheduler/waitcnt tuning.
```

P2-A placement result:

| Gate | Result |
|---|---|
| Real wide stage emission at `isel_wmma` anchor | PASS behind `PREFILL_DBUF_D3A_POST=1`. The marker was replaced with an emitted `ds_store_b128` stage derived from proven current DBUF LDS producers and ordered through the prior-WMMA dependency path. |
| A-side `global_b128` source | PASS. The helper accepts the existing `NOOP("global_b128", idx)` producer and lowers it as a `global_load_b128` feeding `ds_store_b128`. |
| B-side packed tile-key source | PASS structurally. The helper also accepts the packed `NOOP int.vec(4)` / `V_PACK` bridge shape, including `AFTER` wrappers, so B-side candidates are not rejected just because they are not represented as `global_b128`. |
| Suite-order hygiene | PASS. `_dbuf_withlocal_both_mns` now restores `PREFILL_DBUF_GLOBAL_ADDR_INLOOP`, so the addr-inloop negative test cannot leak into the D3-A positive test and create a false spill. |
| Central stream probe | PASS. With `SPEC=1 PREFILL_DBUF_D3A_POST=1`, the default B-side D3-A route reports `ok=true`, `D2_two_slot_identity.ok=true`, `D3_cadence.ok=true`, `D7_scheduler_readiness.ok=true`, body future staging in 4 of 7 inter-WMMA gaps, and `scalar_lds_store_count=0`. |
| Small central route-bound correctness | PASS. `PREFILL_DBUF_D3A_POST=1` on the existing route-bound gate returns `PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_PASS`, finite output, `max_abs_vs_ref=0.03130340576171875`, and `rel_rmse_vs_ref=0.00020765016961377114`. |
| Target bounded correctness/perf smoke | CORRECT BUT NOT PROMOTED. `DEV=AMD:ISA`, `512x5120x5120`, `u0=2,u1=2,loc=2,unr=2`, `PREFILL_TC_LOCAL_STAGE_POST=1`, DBUF A+B packed runs correctly. Repeated unpinned samples: baseline `7.63/7.75/7.84 TFLOPS`; default B-side D3-A `6.96/6.97/6.97 TFLOPS`. The structural substrate is ready, but current scheduler/waitcnt placement does not yet convert it into speed. |
| B-side future staging diagnostic | PASS after use-site B `V_PACK` rematerialization. The prior spill was one hash-deduped packed B span live across future stores; cloning the B packs at the D3-A use site and making their ordering dependency-sensitive removes the spill. |
| A+B future staging diagnostic | PASS structurally, not performance-promoted. `PREFILL_DBUF_D3A_STAGE_A=1 PREFILL_DBUF_D3A_STAGE_B=1` emits both operands, reaches D2/D3/D7, and remains no-spill/correct, but over-stages this bounded worker and is slower than one-side staging. |
| D3-A audit | PARTIAL. `PREFILL_DBUF_D3A_AUDIT=1` shows default A-side gated off and B-side LDS byte windows from buffer `993`. The active WITH_LOCAL route still lacks explicit `wmma_frag_buffer_proof` metadata, so the audit cannot yet label real `(kr+1)%nbuf` epochs from graph tags; normalized byte-window proof is the authoritative structural proof for now. |
| Scheduler/waitcnt forensic pass | PARTIAL. D3-A inserts body `ds_store_b128` immediately before the next current-slot `ds_load_b128`; targeted waitcnt then drains LGKM for store/load safety. Diagnostic `AMD_ISA_WAITCNT_D3A_SKIP_STORE_LOAD=1` removes 3 body waits and remains correct on the central gate, improving bounded D3-A from about `7.02` to `7.27 TFLOPS`, but baseline remains about `7.8 TFLOPS`. So conservative store/load waits are a blocker, not the whole blocker. |

P2-A caveat:

```text
This proves the lowering-time placement primitive: wide staging can be emitted in the steady-state WMMA body without
scalar/narrow LDS fallback, including the B tile-key packed path. It is still a prototype because it reuses currently
discoverable producers; it has not yet proved explicit graph-tagged `(kr+1)%nbuf` epoch identity or a promotion-grade
same-clock performance win on the target `512x5120x5120` shape. The next blocker has moved to scheduler/waitcnt:
without real overlap, the extra B-side future stores/packs/waits cost more than they hide.
```

Scheduler forensic conclusion:

```text
normal D3-A:   baseline D2/D3/D7 structure, but 37 waits and immediate LGKM drains after body future stores
skip-drain D3-A: 34 waits, central correctness still passes, bounded smoke improves by ~0.25 TFLOPS
remaining gap: extra future-stage packs/stores and too-short lookahead still exceed the hidden latency benefit
```

The primitive next fix is not more operand staging. It is either:

- real LDS byte-window alias tracking in waitcnt, so only overlapping store/load pairs drain; and/or
- a larger lookahead/scheduling window that places future staging far enough ahead of its consuming LDS loads.

Guardrails:

```text
Do not infer future epochs from final instruction order.
Do not accept scalar/narrow LDS as satisfying D3.
Do not make D3 default-on.
Do not tune waitcnt or scheduler until P1.4 passes correctness.
```

Agent audit result:

| Question | Result |
|---|---|
| Can D3-B solve this cleanly? | No. `amd.py` only sees existing store groups, LDS loads, and WMMA chains. `_pack_withlocal_lds_stores`, `_pack_b_tilekey_lds_stores`, `_index_after_dep`, and `_frag_b128_loads` can preserve or delay existing work but cannot synthesize missing future-slot producers without duplicating graph lifecycle semantics in the renderer. |
| Where is the current lifecycle closed? | `postrange.py::_tc_local_stage_coop_operand` builds staged stores, wraps them in `UOp.group(*stores)`, closes the group over `tile_ranges`, creates `UOp.barrier(stage)`, then forces the WMMA operand load through `.after(bar)`. That expresses `all stores -> barrier -> current LDS load -> WMMA`, with no future-stage group left in the body. The older B tile-key path has the same shape. |
| Primitive route | D3-A only: split graph lifecycle so preload/current-slot consumption and future-slot staging are separate effect groups. The renderer should then pack/lower that existing graph work; it should not infer K epochs itself. |

Required D3-A patch shape:

```text
current slot stores -> current_stage_group/end(tile_ranges) -> barrier
barrier -> current slot ds_load_b128 -> WMMA current tile

separate future slot stores, addressed with (kr + 1) % nbuf,
ordered after a current-load/WMMA token,
not included in the current_stage_group before the barrier
```

This must start behind an explicit implementation flag until it flips the 2x2 red test and passes verifier. The future
stores must target only the opposite DBUF slot and must survive DCE through a legal effect/control path; attaching future
stores to the existing pre-barrier stage group is not a fix.

First implementation attempt result:

| Attempt | Result | Meaning |
|---|---|---|
| Add a flag-gated postrange future-stage group with `(kr + 1) % nbuf` slot math. | Rejected; `SPEC=1` hit a verifier failure on `Ops.PTRCAT dtypes.half.ptr(2048, 2).vec(2)` with two `Ops.INDEX` sources. | D3-A is still the right primitive layer, but the next blocker is verifier-clean construction of a second LDS store group, not renderer ordering. |
| Duplicate current source values while targeting the opposite DBUF slot. | Same verifier failure. | The failure is the graph/pointer shape of the second stage group, not the exact future-value substitution. |

Next unblock step before another D3-A implementation attempt:

```text
Build a tiny verifier-only postrange probe for a second LOCAL store group:
  - same local placeholder and scalar half stores,
  - no future K substitution,
  - no renderer packing requirement,
  - prove the graph can carry two disjoint store groups without PTRCAT/vector local pointer verifier failure.

Only after that is verifier-clean, reattach the DBUF future-slot math and packed b128 lowering.
```

Unblock status:

| Gate | Result |
|---|---|
| Verifier-only LOCAL pointer grouping proof | PASS. `tinygrad/codegen/late/devectorizer.py` now disables LOCAL pointer `PTRCAT` grouping for D3-A staging buffers `990/991/993` only when `PREFILL_DBUF_D3A_POST=1`. |
| Unit proof | PASS. `test_dbuf_d3a_local_staging_buffers_do_not_fold_to_ptrcat` verifies the D3-A flag leaves a LOCAL staging stack without `Ops.PTRCAT`. |
| Full D3-A cadence | Still TODO. The future-stage graph rewrite must be retried now that the pointer grouping obstruction is guarded. |

Second implementation attempt result:

| Attempt | Result | Meaning |
|---|---|---|
| Add a separate future-slot store group in `_tc_local_stage_coop_operand` and attach it to the staged LOCAL buffer dependency. | Verifier-clean, but ineffective: the extra `ds_store_b128` work hoisted into the prologue and `D3_cadence.ok` stayed false. The patch was removed. | Operand-level postrange rewrites can create the future producer, but they do not have a post-current-WMMA token to anchor it after the current compute. |
| Attach the future group to `wmma.src[2]`. | Still hoisted. | On this generated route, the useful prior-WMMA dependency is introduced later inside `amd.py::isel_wmma` / fragment reload construction, not as a reliable postrange token on the operand rewrite. |

Current named blocker:

```text
D3-A needs a graph-visible or lowering-visible post-current-WMMA anchor.

The future-stage group must be ordered:
  previous/current WMMA complete enough to have consumed the current slot
  -> future-slot global/LDS staging
  -> next slot LDS consumption

Current `_tc_local_stage_coop_operand` only rewrites WMMA operands, so every dependency it can attach is before the
consuming WMMA. The late WMMA chain builder already creates the true prior-WMMA WAR guard for fragment reloads; D3 must
either expose that guard earlier as a graph token or consume explicit future-stage metadata in the WMMA lowering path.
```

Small design test:

| Probe | Result | Decision |
|---|---|---|
| Wrap real generated `Ops.WMMA` as `wmma.after(UOp(NOOP, void))` behind a temporary probe flag. | Rejected. The route failed before final instruction stream with a lowering/render `KeyError`, so the final D2/D3 probes could not see LDS operands or WMMA regions. Temporary probe code was removed. | A graph-level post-WMMA anchor cannot be represented as a plain `AFTER(WMMA, token)` in the current pipeline. The primitive route needs either a first-class post-WMMA effect token that lowering understands, or explicit future-stage metadata consumed at `isel_wmma` where the true prior-WMMA WAR guard already exists. |

Non-routes:

| Non-route | Reason |
|---|---|
| `_schedule` tuning | Proven insufficient by `AMD_ISA_SCHED=0/1`; no future staging exists in the input stream. |
| `AMD_ISA_WAITCNT_TARGETED` tuning | Waitcnt can preserve overlap but cannot synthesize it. |
| `PREFILL_DBUF_GLOBAL_ADDR_INLOOP=1` alone | Tested; D3 still false. |
| `PREFILL_DBUF_LDS_ADDR_SERIAL=1` | Tested; D3 still false and scalar LDS fallback regressed. |

Minimal first implementation test:

```text
On 2x2 only, make exactly one wide future staging group appear between two adjacent WMMA regions.
Do not judge speed.
Do not expand to 4x2/2x4 until:
  - expectedFailure target flips by seeing global_load_b128 or ds_store_b128 between WMMAs,
  - current diagnostic tests still pass or are intentionally updated,
  - no ds_store_b16/ds_store_b32/ds_store_b64 fallback,
  - native probe D3_cadence.ok=true.
```

### D4. Keep The Lean Staging Contract Under DBUF

Purpose: DBUF must not regress into scalar LDS or full register double-buffering.

Acceptance:

| Gate | Required evidence |
|---|---|
| D4.1 Wide stores | `ds_store_b128` is used for staged A and B fragments. |
| D4.2 Wide loads | `ds_load_b128` feeds WMMA operands. |
| D4.3 No scalar LDS fallback | `ds_store_b16==0`, `ds_store_b32==0`, `ds_store_b64==0` for fragment staging. |
| D4.4 No register DBUF spill | Native probe has `ok=true`; no `Inc 0: no spills`. |
| D4.5 Pressure bounded | `REGALLOC_DEBUG` peak remains in the post-fix band, not the old 169+ live VGPR failure band. |
| D4.6 Operand origins | WMMA `src0` and `src1` originate from `ds_load_b128`. |

### D5. Verifier And Structural Safety

Purpose: keep the generated graph legal while adding DBUF slot/cadence edges.

Acceptance:

| Gate | Required evidence |
|---|---|
| D5.1 `SPEC=1` clean | No malformed `PTRCAT`, vector local pointer, `UNROLL(STACK)`, or bad `AFTER`. |
| D5.2 Cache hygiene | Tests clear `getenv` and `to_program_cache` around env-flag structural probes. |
| D5.3 Effect ordering legal | Any slot-safety edge uses verifier-clean buffer/effect dependencies, not half-value `AFTER` on a void barrier. |
| D5.4 No broad global guards | DBUF-specific pointer or grouping disables are scoped to the affected local buffer/path. |

Allowed ordering fallback if slot safety needs an explicit compiler edge:

```text
STORE(INDEX(AFTER(DEFINE_LOCAL_SLOT, GROUP_OR_BARRIER_DEP), idx), value, gate)
```

Avoid:

```text
AFTER(half_scalar_value, void_barrier)
GROUP(..., BARRIER(...), ...)
STORE(AFTER(INDEX(...), dep), value)
```

### D6. GPU Correctness Before Scheduler Tuning

Purpose: scheduler work should start only from a numerically correct DBUF route.

Acceptance:

| Gate | Required evidence |
|---|---|
| D6.1 Central route-bound | DBUF candidate passes `extra.qk.prefill_graph_gemm_route_bound_stage_gate` or an explicitly documented equivalent central harness. |
| D6.2 Finite | No NaNs/infs. |
| D6.3 Numeric envelope | RMSE/max-abs remain within the existing accepted local-stage envelope unless a new envelope is justified. |
| D6.4 Native probe | The same flag bundle used for correctness also passes the native structural probe. |

### D7. Scheduler-Tuning Readiness Gate

This is the stop line for this scope.

Ready for scheduler/waitcnt tuning means all of these are true:

| Gate | Required evidence |
|---|---|
| D7.1 Two-slot visible | Probe identifies slot 0 and slot 1 for both operands. |
| D7.2 Cadence visible | Probe identifies prologue/body/tail or equivalent modulo cadence. |
| D7.3 Structural opportunity | Next-slot memory/staging work is topologically present near current-slot compute, even if conservative waits still serialize it. |
| D7.4 Correct | GPU/central correctness passes. |
| D7.5 Low pressure | No spills and pressure remains bounded. |
| D7.6 Documented handoff | Scope doc records the exact command, counts, slot evidence, and remaining scheduler question. |

After D7, the next scope is scheduler/waitcnt tuning:

- preserve future global loads across current WMMAs;
- choose minimal `vmcnt`/`lgkmcnt`;
- ensure barriers do not erase overlap;
- measure TFLOPS.

## Commands

Current locked both-side substrate:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Candidate DBUF structural probe:

```bash
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Candidate DBUF verifier probe:

```bash
SPEC=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

D3-A verifier probe, default B-side future staging:

```bash
SPEC=1 \
PREFILL_DBUF=1 \
PREFILL_DBUF_D3A_POST=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 1 --indent 0
```

Candidate DBUF correctness gate:

```bash
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage both --compact
```

D3-A bounded worker smoke:

```bash
DEV=AMD:ISA WORKER=1 \
MM=512 OUTF=5120 INF=5120 U0=2 U1=2 LOC=2 UNR=2 \
PREFILL_DBUF=1 \
PREFILL_DBUF_D3A_POST=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
AMD_ISA_REG_ACCUM=1 \
REGALLOC_ADDR_REMAT=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill_v2_schedule_search.py
```

D3-A waitcnt diagnostic, not default:

```bash
DEV=AMD:ISA WORKER=1 \
MM=512 OUTF=5120 INF=5120 U0=2 U1=2 LOC=2 UNR=2 \
PREFILL_DBUF=1 \
PREFILL_DBUF_D3A_POST=1 \
AMD_ISA_WAITCNT_D3A_SKIP_STORE_LOAD=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
AMD_ISA_REG_ACCUM=1 \
REGALLOC_ADDR_REMAT=1 \
AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill_v2_schedule_search.py
```

Pressure probe:

```bash
REGALLOC_DEBUG=1 \
PREFILL_DBUF=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

## Failure Classification

| Failure | Meaning | Next action |
|---|---|---|
| DBUF no-op | `PREFILL_DBUF=1` changes allocation or flags but not slot identity/cadence. | Fix reduce peel or slot-index plumbing. |
| Slot alias | Slot 0/1 addresses overlap or cannot be distinguished. | Fix LDS address formula for A/B, especially B tile-key. |
| Scalar fallback | DBUF path reintroduces `ds_store_b16`/`ds_store_b32`. | Reconnect packed B128 bridge/matcher to DBUF graph shape. |
| Spill | Two-slot structure duplicates too many live temps. | Add stronger lifetime split or reduce staging breadth before scheduler work. |
| Verifier fail | Bad `PTRCAT`, vector local pointer, or invalid `AFTER`. | Move ordering to buffer/effect-level legal shapes. |
| Correctness fail | Structure compiles but output is wrong/non-finite. | Compare slot address/read contract; do not tune scheduler yet. |
| No body overlap opportunity | Prologue and tail exist but all future work is still outside compute body. | Fix peel/cadence before waitcnt tuning. |

## Completion Definition

This scope is complete: the DBUF candidate is verifier-clean, compiles no-spill, stages both operands through wide LDS,
shows slot identity and a prologue/body/tail cadence, and passes GPU/central correctness. Next scope: scheduler/waitcnt
tuning and measured performance.
