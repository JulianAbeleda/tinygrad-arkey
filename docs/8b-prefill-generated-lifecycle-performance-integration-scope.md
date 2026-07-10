# 8B Prefill Generated Lifecycle Performance Integration Scope

## Status: Deferred by S10.5 (generated transport correct-but-slow; see docs/8b-prefill-s10_5-machine-search-over-backend-atom-scope.md)

Date: 2026-07-09.

## Problem

The generated LDS/DBUF primitive is now structurally correct for the `ffn_gate_up` slice, but it does not make the
whole-prefill route fast.

Current evidence:

| Fact | Status |
|---|---|
| Generated single-buffer LDS route for `ffn_gate_up` | route-bound sampled correctness passes. |
| Generated DBUF route for `ffn_gate_up` | structurally promoted: packed b128 staging, no raw `build_gemm_lds2`, sampled correctness passes, D2/D3/D7 true. |
| Old B-side D2 failure | fixed as a proof-key bug; LDS address family keys are now definition-sensitive. |
| Whole-prefill DBUF smoke | `bench/prefill-whole-synced/lds-dbuf-promoted-smoke.json`: pp512 `205.43 tok/s`. |
| Stored Path1 smoke | `218.12 tok/s`. |
| Stored hand-path authority | `4413.2 tok/s`. |

So the blocker is no longer "can we express LDS/DBUF without a full hand kernel?" The blocker is:

```text
model prefill lifecycle
  -> role selection
  -> pipe roles + ffn_gate_up LDS/DBUF primitive
  -> generated transport attribution
  -> per-role timing
  -> whole-prefill authority
```

The current route does not yet prove which roles are using which primitive in the e2e run, and the measured e2e speed
says the lifecycle is still dominated by ordinary generated transport overhead.

## Non-Goals

- Do not reopen gfx1100 `4x4` development. It is parked because the register budget is the wrong fight for this GPU.
- Do not clone `build_gemm_lds2` as a route-local full instruction list.
- Do not build a new harness when `prefill_pipe_mvp_artifact.py`, `prefill_whole_synced.py`, lifecycle trace, and the
  route manifest already cover the needed gates.
- Do not tune DBUF cadence further until e2e attribution proves the DBUF primitive is selected and role timing proves
  `ffn_gate_up` is still the dominant bottleneck.

## Definition Of 100%

| Layer | Percent | Requirement | Done Signal |
|---|---:|---|---|
| L0. Primitive correctness | 15% | Pipe primitive and LDS/DBUF primitive compile and pass bounded correctness. | Existing artifact verdict reaches `PREFILL_LDS_DBUF_PRIMITIVE_PROMOTED_STRUCTURAL_CORRECTNESS`. |
| L1. Route identity | 15% | Manifest, purity guard, surface audit, and whole-prefill attribution can name pipe-only vs pipe+LDS/DBUF routes. | A distinct route id such as `prefill_wmma_pipe_lds_dbuf_primitive_generated` appears in reports. |
| L2. E2E binding | 15% | Whole-prefill can prove the intended role map: pipe for `attn_qo`, `attn_kv`, `ffn_down`; LDS/DBUF for `ffn_gate_up`. | `prefill_whole_synced.py --require-route` fails closed if any role silently falls back. |
| L3. Per-role timing | 20% | Existing artifact/harness records isolated timing for all four hot GEMM roles under the same route flags. | JSON contains per-role ms/tok or TFLOPS plus route id/provenance for each role. |
| L4. Lifecycle density | 15% | For each slow role, trace explains instruction/WMMA, wait/WMMA, memory ops/WMMA, LDS load/store rates, and WMMA clustering. | Generated-vs-hand delta table identifies the largest ratio gap by role. |
| L5. Performance fix | 15% | The highest-leverage role/lifecycle issue is fixed behind opt-in flags without raw full-kernel fallback. | Whole-prefill smoke moves materially above Path1 and isolated role timing improves for the named bottleneck. |
| L6. Promotion decision | 5% | Same-clock authority either reaches hand parity/5k target or records the exact remaining bottleneck. | Authority artifact says promoted, correct-not-fast with named blocker, or refuted. |

This scope is complete when L0-L6 are represented by durable artifacts and the next action is mechanical promotion or a
single named performance primitive. It is not complete when the only evidence is a standalone DBUF probe.

## Phase Plan

### P0. Bank The Current Finding

Tasks:

- Keep `bench/prefill-pipe-mvp/ffn-gate-up-lds-primitive.json` as the structural correctness authority.
- Keep `bench/prefill-whole-synced/lds-dbuf-promoted-smoke.json` as the current e2e negative performance authority.
- Record that DBUF correctness/cadence is not the current blocker.

Done when:

- Docs point to both artifacts.
- The next phases start from lifecycle/performance integration, not DBUF proof repair.

### P1. Route Identity Split

Purpose: stop hiding the active route behind `prefill_wmma_pipe_primitive_generated`.

Tasks:

- Add a manifest route for the composed generated route:

```text
prefill_wmma_pipe_lds_dbuf_primitive_generated
  roles:
    attn_qo   -> generated pipe primitive
    attn_kv   -> generated pipe primitive
    ffn_down  -> generated pipe primitive
    ffn_gate_up -> generated LDS/DBUF primitive
  provenance: tinygrad_scheduler_generated
```

- Update `pure_search_guard._prefill_gemm_effective` so:

```python
if PREFILL_GRAPH_GEMM and PREFILL_WMMA_PIPE_PRIMITIVE and PREFILL_WMMA_LDS_PRIMITIVE and PREFILL_DBUF:
  return "prefill_wmma_pipe_lds_dbuf_primitive_generated", False
```

- Add a matching `pure_kernel_surface_audit` row.
- Add unit tests for default, pipe-only, raw graph-GEMM, LDS-only, and pipe+LDS/DBUF env maps.

Done when:

- `effective_routes(env)` names the composed route.
- The composed route is strict-pure under the audit.
- Existing pipe-only tests still pass.

### P2. Whole-Prefill Fail-Closed Binding

Purpose: prove the e2e run is using the intended primitives before using timing data.

Tasks:

- Extend `prefill_whole_synced.py --require-route` to accept the composed route id.
- Add an optional role-map field to the whole-prefill artifact:

```json
"prefill_role_routes": {
  "attn_qo": "pipe",
  "attn_kv": "pipe",
  "ffn_down": "pipe",
  "ffn_gate_up": "lds_dbuf"
}
```

- Reuse existing route/sample correctness helpers from `prefill_pipe_mvp_artifact.py`; do not create another harness.
- Fail closed if DBUF flags are requested but the selected route id is still pipe-only.

Done when:

- The existing DBUF smoke reports the composed route id.
- A deliberately missing `PREFILL_DBUF=1` run fails the composed-route requirement.

### P3. Per-Role Timing Attribution

Purpose: find which role is blocking e2e speed.

Tasks:

- Add a per-role timing section to the existing MVP artifact path, using current sampled role runners where possible.
- Measure:
  - `attn_qo`
  - `attn_kv`
  - `ffn_down`
  - `ffn_gate_up`
- Record route flags and route id next to each role timing.
- Include the hand/reference number when an existing artifact already has it; otherwise leave it blank rather than
  inventing a comparison.

Done when:

- One JSON artifact answers: "which role got slower or failed to move?"
- The result can explain why whole-prefill remains near `205 tok/s`.

### P4. Lifecycle Density Diff

Purpose: explain the timing gap in machine terms.

Tasks:

- Reuse `extra/qk/prefill/kernel_lifecycle_trace.py` and `hand_vs_generated_shape_matrix.py`.
- For the slowest role first, compare:
  - `inst/WMMA`
  - `wait/WMMA`
  - `global_load_b128/WMMA`
  - `ds_store_b128/WMMA`
  - `ds_load_b128/WMMA`
  - max consecutive WMMAs without an intervening wait
  - scalar LDS fallback count
- Compare the active generated route against the hand/reference route only as a measurement oracle.

Done when:

- The largest structural gap is named.
- The next fix is one primitive, not broad scheduler tuning.

### P5. First Performance Fix

Pick only after P3/P4. Candidate fixes, ranked by current likelihood:

| Rank | Candidate | Why it may move e2e | Proof required |
|---:|---|---|---|
| 1 | Route/binding fix | The current smoke may not be attributing or requiring the composed route. | E2E route id changes and missing flags fail closed. |
| 2 | Pipe role lifecycle density | Three hot roles still use pipe transport; if they dominate, ffn DBUF cannot move total speed. | Per-role timing shows pipe roles dominate total time. |
| 3 | WMMA cluster scheduling | Hand LDS2 clusters WMMAs behind fewer waits; generated has much higher wait density. | Trace shows wait/WMMA drops and timing improves. |
| 4 | Fragment residency/reuse | Generated previously used about 2x LDS loads per WMMA versus hand. | `ds_load_b128/WMMA` drops without correctness loss. |
| 5 | DBUF stage anchor | D3A worsened prior bounded tests, but a route-specific anchor may still help. | Bounded test beats generated DBUF baseline before e2e. |

Done when:

- A small test moves an isolated role or whole-prefill smoke.
- If no candidate moves, the artifact says "spinning" with the measured reason.

### P6. Authority Run And Decision

Tasks:

- Run same-clock whole-prefill authority with route attribution and per-role timing attached.
- Compare against:
  - Path1 smoke/authority,
  - stored hand-path authority,
  - 5k target.
- Classify:
  - `promote`: passes correctness and reaches the threshold,
  - `correct_not_fast`: correct route, named bottleneck remains,
  - `refuted`: route cannot beat Path1 after the named primitive fix.

Done when:

- The output is a stable artifact under `bench/prefill-whole-synced/`.
- Docs and manifest status reflect the decision.

## Existing Commands To Reuse

Structural artifact:

```bash
PYTHONPATH=. python3 extra/qk/prefill_pipe_mvp_artifact.py \
  --lds-primitive --lds-sample-correctness --sample-cols 16 --compact
```

Current whole-prefill DBUF smoke:

```bash
PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 PREFILL_WMMA_PIPE_PRIMITIVE=1 PREFILL_WMMA_LDS_PRIMITIVE=1 \
PREFILL_CHUNKED=1 AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF=1 PREFILL_DBUF_NBUF=2 PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_DBUF_LDS_CONST_IMM=1 \
PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 PREFILL_DBUF_DIRECT_B128_CHAIN=1 \
PREFILL_DBUF_LDS_ADDR_USE_DEP=1 AMD_ISA_WAITCNT_TARGETED=1 REGALLOC_ADDR_REMAT=1 \
PREFILL_DBUF_D3A_POST=1 PREFILL_DBUF_D3A_AUDIT=1 PREFILL_DBUF_D3A_STAGE_A=1 PREFILL_DBUF_D3A_STAGE_B=1 \
PYTHONPATH=. python3 extra/qk/prefill_whole_synced.py --mode smoke -K 1 --warmups 1 --rounds 1 \
  --start-positions 0 --whole-lengths 512 --max-context 1024 --logits-only --pin-clock \
  --artifact bench/prefill-whole-synced/lds-dbuf-promoted-smoke.json --json
```

Lifecycle diff:

```bash
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py --json
DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py --json
```

## Parallel Work Lanes

| Lane | Can run in parallel? | Owner scope | Files |
|---|---:|---|---|
| A. Route identity | yes | manifest, purity, surface audit, unit tests | `extra/qk/route_manifest.py`, `extra/qk/pure_search_guard.py`, `extra/qk/pure_kernel_surface_audit.py`, route purity tests |
| B. Whole-prefill binding | yes after route id shape is agreed | `prefill_whole_synced.py` route requirement and role-map artifact | `extra/qk/prefill_whole_synced.py`, `test/unit/test_prefill_whole_synced.py` |
| C. Per-role timing | yes | extend existing MVP artifact with role timing fields | `extra/qk/prefill_pipe_mvp_artifact.py`, `test/unit/test_prefill_pipe_mvp_artifact.py` |
| D. Lifecycle diff refresh | yes | produce current generated-vs-hand density table, no codegen changes | `extra/qk/prefill/kernel_lifecycle_trace.py`, `extra/qk/prefill/hand_vs_generated_shape_matrix.py`, docs/artifacts |
| E. Performance fix | sequence after B/C/D | implement exactly one measured primitive fix | selected only after the bottleneck is named |

The immediate parallelizable work is A, B, C, and D. Lane E must wait for the measurements so we do not repeat the
build-before-measure mistake.

## Stop Conditions

Stop and report "spinning" only if all are true:

- the composed route is named and fail-closed,
- per-role timing exists,
- lifecycle density exists for the slowest role,
- one bounded primitive fix was tested,
- no isolated role or whole-prefill smoke moves above the previous Path1 result.

Until then, the next action is known.

## P4 Refresh - 2026-07-09 Task Lane D

Scope: active generated DBUF/pipe route for `m=512,n=5120,k=5120,u0=2,u1=2,loc=2,unr=2`, compared against the
hand LDS2 `wm=2,wn=2,waves_m=1,waves_n=1,bk=64,dbuf=1` measurement oracle. This is a lifecycle-density refresh only;
no codegen changes were made.

Commands run:

```bash
DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py --active-generated --kind generated \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --target AMD:ISA:gfx1100 --json

DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py --kind hand-lds2 \
  --m 512 --n 5120 --k 5120 --wm 2 --wn 2 --waves-m 1 --waves-n 1 --bk 64 --dbuf 1 \
  --target AMD:ISA:gfx1100 --json

DEV=AMD:ISA PYTHONPATH=. python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --skip-hand \
  --hand-reps 1 --hand-iters 1 --json
```

Current density table:

| route | timing note | inst/WMMA | wait/WMMA | global_load_b128/WMMA | ds_store_b128/WMMA | ds_load_b128/WMMA | max WMMA cluster | scalar LDS fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generated active DBUF `2x2` | `7.88 TFLOPS`, unpinned matrix run | 39.062 | 3.312 | 2.0 | 2.0 | 4.0 | 1 | 0 |
| hand LDS2 `2x2` oracle | structural trace only in this refresh | 9.547 | 0.406 | 1.0 | 1.0 | 2.0 | 4 | 0 |

Supporting trace facts:

- Generated active DBUF uses the packed chain (`global_load_b128 -> ds_store_b128 -> barrier -> ds_load_b128 -> WMMA`)
  and has no scalar LDS fallback, so scalar store cleanup is not the current density blocker.
- Generated active DBUF still places all global/LDS store work in the prologue for this trace. `D3_cadence.ok=false`,
  `future_slot_work_before_current_compute=false`, and every body region between WMMAs contains current-slot
  `ds_load_b128` only.
- Generated compute cadence is effectively one WMMA per wait in the body: after the first prologue-heavy WMMA, every
  subsequent WMMA has one `lgkmcnt(0)` wait immediately before it.
- Hand LDS2 keeps `D3_cadence.ok=true` and has future-slot work in body regions
  `between_wmma_195_256`, `between_wmma_298_362`, and `between_wmma_404_433`.
- Hand LDS2 issues four consecutive WMMAs after each LDS-load wait group, giving max WMMA cluster `4` and much lower
  wait density.

Largest structural gap to test next: generated WMMA clustering / LDS fragment residency. The active generated route is
already on packed b128 LDS transport with zero scalar fallback, but it still pays `4.0 ds_load_b128/WMMA` and roughly
one wait per body WMMA, while hand pays `2.0 ds_load_b128/WMMA` and amortizes waits over four-WMMA clusters. The next
small test should therefore prove a generated primitive that groups LDS loads for a four-WMMA cluster or reuses loaded
fragments across adjacent WMMAs, with gates that `max WMMA cluster > 1`, `wait/WMMA` drops, and
`ds_load_b128/WMMA` moves below `4.0`.

## P1-P3 Execution - 2026-07-09

Completed:

- Added `prefill_wmma_pipe_lds_dbuf_primitive_generated` as a distinct route identity.
- `effective_routes(env)` now distinguishes:
  - default scheduler matmul,
  - raw graph-GEMM oracle,
  - pipe-only generated primitive,
  - composed pipe+LDS/DBUF generated primitive.
- `prefill_whole_synced.py` now records:
  - `prefill_role_routes`,
  - `prefill_route_binding_gate`,
  - `--require-route`.
- `prefill_pipe_mvp_artifact.py` now records `per_role_timing` with explicit `not_run` placeholders and
  `--measure-per-role-timing`.
- `postrange._WARMSTART_LOCAL_STAGE_KEYS` now scopes LDS/DBUF rewrites to the warmstart keys that requested them.
  This fixed the first composed-route bug: global LDS/DBUF env flags were contaminating pipe roles.

Verification:

```bash
PYTHONPATH=. pytest -q \
  test/unit/test_prefill_graph_gemm_route.py \
  test/unit/test_prefill_pipe_mvp_artifact.py \
  test/unit/test_prefill_whole_synced.py \
  test/unit/test_pure_kernel_surface_audit.py \
  test/unit/test_qk_route_purity.py
```

Result: `54 passed`.

Route check:

```text
PREFILL_GRAPH_GEMM=1
PREFILL_WMMA_PIPE_PRIMITIVE=1
PREFILL_WMMA_LDS_PRIMITIVE=1
PREFILL_DBUF=1
```

now resolves to:

```text
prefill_wmma_pipe_lds_dbuf_primitive_generated
attn_qo      -> pipe
attn_kv      -> pipe
ffn_down     -> pipe
ffn_gate_up  -> lds_dbuf
binding gate -> PREFILL_ROUTE_BINDING_PASS
```

Per-role sampled correctness after role scoping:

| role | primitive | correctness | compile-included sample note |
|---|---|---:|---:|
| `attn_qo` | pipe | pass | `295.1 ms` |
| `attn_kv` | pipe | pass | `292.1 ms` |
| `ffn_down` | pipe | pass | `312.1 ms` |
| `ffn_gate_up` | LDS/DBUF | pass | `537.0 ms` |

Whole-prefill smoke after route scoping:

```text
bench/prefill-whole-synced/lds-dbuf-promoted-smoke.json
route: prefill_wmma_pipe_lds_dbuf_primitive_generated
binding gate: pass
pp512: 204.99 tok/s
```

Conclusion: route identity, fail-closed binding, and pipe-role correctness are fixed. The e2e speed did not move, so the
remaining blocker is not hidden fallback or cross-role LDS contamination. It is the generated lifecycle density itself.

Next fix lane:

```text
generated active DBUF: max WMMA cluster = 1, wait/WMMA = 3.312, ds_load_b128/WMMA = 4.0
hand LDS2 oracle:       max WMMA cluster = 4, wait/WMMA = 0.406, ds_load_b128/WMMA = 2.0
```

The primitive route to test next is phase-scoped LDS fragment residency / WMMA clustering. It must be proven first on
the existing lifecycle trace harness before another whole-prefill authority run.

## P5 Lane E Bounded Primitive Test - 2026-07-09

Scope: first performance primitive test after P1-P4, limited to the active generated DBUF `2x2` lifecycle trace and
matrix harness. No code patch was required for this bounded test; the smallest existing knob that moved the primitive
was the phase-scoped K-major WMMA path:

```text
PREFILL_WMMA_KMAJOR_PHASE=1
PREFILL_WMMA_AB_PROOF_KEY=1
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1
```

Commands run:

```bash
DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PYTHONPATH=. python3 extra/qk/prefill/kernel_lifecycle_trace.py --active-generated --kind generated \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --target AMD:ISA:gfx1100 --json

DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PREFILL_WMMA_KMAJOR_PHASE=1 PREFILL_WMMA_AB_PROOF_KEY=1 PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1 \
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1 PYTHONPATH=. \
python3 extra/qk/prefill/kernel_lifecycle_trace.py --active-generated --kind generated \
  --shapes 2,2 --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --target AMD:ISA:gfx1100 --json

DEV=AMD:ISA AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 PREFILL_DBUF=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 REGALLOC_ADDR_REMAT=1 \
PREFILL_WMMA_KMAJOR_PHASE=1 PREFILL_WMMA_AB_PROOF_KEY=1 PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1 \
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1 PYTHONPATH=. \
python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py --shapes 2,2 \
  --m 512 --n 5120 --k 5120 --loc 2 --unr 2 --skip-hand --hand-reps 1 --hand-iters 1 --json
```

Result:

| generated active DBUF `2x2` | status | TFLOPS note | instruction count | wait/WMMA | ds_load_b128/WMMA | max WMMA cluster |
|---|---:|---:|---:|---:|---:|---:|
| baseline | trace ok | prior matrix run `7.88`, unpinned | 625 | 3.312 | 4.0 | 1 |
| phase-scoped K-major | trace ok, matrix status `ok` | `12.12`, unpinned | 554 | 2.875 | 2.0 | 3 |

Supporting facts:

- `ds_load_b128` count dropped from `64` to `32` for the same `16` WMMAs.
- Wait count dropped from `53` to `46`.
- The WMMA stream now includes no-wait runs, for example indices `287,291,292`; max cluster is `3`
  by the same no-intervening-wait rule used for the P4 hand comparison.
- The packed LDS chain remains visible and scalar LDS fallback remains `0`.
- `PREFILL_WMMA_CHAIN_AB_RESIDENT=1` with the same proof flags was also probed and failed compile with
  `NotImplementedError: Inc 0: no spills`, so it was not the smallest safe primitive for this lane.

Conclusion: Lane E is not blocked. The phase-scoped K-major residency route is the first primitive that moves the
generated DBUF lifecycle density on the bounded `2x2` trace. The next decision is whether to promote this flag set into
the composed route experiment and run the fail-closed whole-prefill authority, or first add a focused regression test
that asserts the `2x2` structural movement.

## P5 Transfer Check - 2026-07-09

The phase-scoped K-major flags were then tested through the composed route harness:

```text
PREFILL_WMMA_KMAJOR_PHASE=1
PREFILL_WMMA_AB_PROOF_KEY=1
PREFILL_WMMA_AB_PHASE_SCOPED_KEY=1
PREFILL_WMMA_AB_PROOF_FROM_LDS_DESC=1
```

Per-role sampled correctness still passed for all four roles:

| role | primitive | correctness | compile-included sample note |
|---|---|---:|---:|
| `attn_qo` | pipe | pass | `371.7 ms` |
| `attn_kv` | pipe | pass | `368.3 ms` |
| `ffn_down` | pipe | pass | `395.1 ms` |
| `ffn_gate_up` | LDS/DBUF | pass | `553.9 ms` |

But the LDS/DBUF structural verdict downgraded:

```text
verdict: PREFILL_LDS_PRIMITIVE_GENERATED_TRANSPORT_COMPILES_BLOCKED_ON_CORRECTNESS_PERF
D3_cadence.ok=false
body_has_next_slot_work=false
```

Whole-prefill transfer check:

```text
bench/prefill-whole-synced/lds-dbuf-kmajor-phase-smoke.json
route: prefill_wmma_pipe_lds_dbuf_primitive_generated
binding gate: pass
pp512: 204.89 tok/s
```

Conclusion: K-major phase is useful as a bounded lifecycle-density diagnostic, but it is not an e2e promotion by itself.
It reduces LDS reload density in the isolated trace, yet it does not move whole-prefill and breaks the current D3 future
slot cadence gate. The remaining primitive needs both properties at once:

```text
keep D3 next-slot DBUF cadence
and
get phase-scoped LDS fragment residency / WMMA clustering
```

That is the next narrow blocker. Do not promote the K-major flag set globally until a variant preserves D3 cadence and
moves the fail-closed whole-prefill smoke.
