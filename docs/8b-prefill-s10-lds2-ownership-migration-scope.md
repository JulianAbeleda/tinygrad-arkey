# 8B Prefill S10 LDS2 Ownership Migration Scope

Date: 2026-07-09.

## S-Phase Ledger

Repo scan result: there is no committed `S11` or later phase after S10. Later work is documented as either old S10
composed-route work or parked generated DBUF/P4 work, not as a numbered post-S10 phase.

Current ledger:

| Phase | Status | Meaning |
|---|---|---|
| S0 | done | Pin the fast graph-GEMM oracle/baseline. |
| S1 | done | Extract LDS2 register layout. |
| S2 | done | Extract LDS2 memory layout. |
| S3 | done | Extract LDS2 wait policy. |
| S4 | done | Extract LDS2 cadence. |
| S5 | done | Extract LDS2 lifecycle template. |
| S6 | done | Extract LDS2 primitive emitter. |
| S7 | done | Extract shell/epilogue emitter. |
| S8 | done | Make `build_gemm_lds2` a wrapper around `lower_lds2_gemm_kernel`. |
| S9 | done / baseline | Safe search over extracted knobs; keep opt-in; S9 authority path preserves the 4k pp512 band. |
| S10-A | done | Hybrid S9/S10 scope: S10 owns metadata/spec/search gates while S9 emits backend atoms. |
| S10-B | done | Repeatable hybrid role trace over S9 backend atoms. |
| S10-C | done | Isolate the hard DBUF epoch choreography as `DBUFEpochPrimitive`. |
| S10-D | next | Search/control safe S10-owned knobs around the hybrid boundary while preserving pp512 `>=4000`. |
| S10-E | pending | Promotion/rollback gate for the hybrid route. |
| S10-F | pending | Real parameterized epoch primitive interface beyond metadata. |
| S10-G | later | Partial generated replacement around the epoch primitive. |
| S10-H | parked | Full generated DBUF lifecycle replacement. |

So S10 is the active umbrella. Do not create S11 until S10-D/E/F are complete or explicitly retired.

## 2026-07-09 Pivot: Stop Pursuing The Hard DBUF Replacement In S10

S10 is now narrowed to:

```text
search/spec ownership around the proven LDS2 backend atom, while preserving the S9 4k pp512 band
```

The classification target is:

```text
hybrid machine-searched route over hand-tuned backend atoms
```

This is intentionally not `pure_generated`. It is also not "write another full hand kernel." The S9 execution body stays
as the performance baseline, while S10 owns the metadata, role policy, search knobs, classification, and promotion gates
around that body.

The hard generated DBUF replacement path is explicitly parked for later R&D. That includes the P4/B tile-key,
owner-aware rotated-stage rewrite, and generated DBUF lifecycle replacement work. Those are real compiler problems, but
they are not required for the next useful S10 milestone.

Active S10 should instead treat `lower_lds2_gemm_kernel(...)` / `LDS2PrimitiveEmitter` as the backend atom and move
machine-searchable ownership around it:

| Surface | S10 action | Why |
|---|---|---|
| LDS2 ASM backend atom | keep | It is the only route currently proven to hit the 4k pp512 band. |
| `LDS2RegLayout` / `LDS2MemoryLayout` | search/spec-owned | Already extracted and byte-preserving. |
| `LDS2WaitPolicy` | search/spec-owned | S9 proved this is safely searchable; `LGKM_COOP_STORE=2` is a valid opt-in. |
| `LDS2Cadence` / `LDS2LifecycleTemplate` | search/spec-owned, conservative | Keep byte-equivalent defaults; only promote measured non-defaults. |
| route role selection | search/spec-owned | Keep `ffn_gate/up` on LDS2, pipe roles on the fast raw pipe oracle for now. |
| generated DBUF/P4 rotated-stage lowering | parked | This is the hard part; do not block S10 on it. |

The practical milestone is therefore:

```text
S10_MVP_SEARCH_OWNED_LDS2_BACKEND_ATOM
```

not:

```text
S10_GENERATED_DBUF_REPLACEMENT
```

## Fresh 4k-Band Gate

Archived S9 authority proves the path can hit the target band:

| artifact | git | pp512 | pp4096 | note |
|---|---:|---:|---:|---|
| `bench/prefill-whole-synced/raw-hand-s9-combined-best-authority.json` | `b1259638d` dirty | `4413` | `3237` | best S9 combined authority |
| `bench/prefill-whole-synced/raw-hand-s9-wait-store2-authority.json` | `b1259638d` dirty | `4416` | `3237` | `LGKM_COOP_STORE=2` opt-in authority |

Fresh current-head authority preserves that band when the S9 authority methodology is used:

| command surface | git | pp512 | verdict |
|---|---:|---:|---|
| strict default smoke, `PREFILL_V2=1`, `K=1,warmups=1,rounds=1` | `b3f314b7f` | `265` | not the graph-GEMM route; not an S9 authority run |
| graph-GEMM smoke, `PREFILL_GRAPH_GEMM=1`, `K=1,warmups=1,rounds=1` | `b3f314b7f` | `124` | one-shot smoke/capture settings; not valid for S9 throughput gating |
| graph-GEMM smoke, `PREFILL_GRAPH_GEMM=1 PREFILL_LDS2_WAIT_LGKM_COOP_STORE=2`, `K=1,warmups=1,rounds=1` | `b3f314b7f` | `118` | same smoke caveat |
| route dump, `PREFILL_GRAPH_GEMM=1 PREFILL_GRAPH_GEMM_ROUTE_DUMP=1`, `warmups=0` | `b3f314b7f` | `89` | diagnostic run only |
| S9 authority, `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1`, `K=8,warmups=4,rounds=3` | `fa254a410` | `4407` | current head reproduces the 4k band |

So the immediate blocker is not generated DBUF ownership and not the S9 path. The correct active path is:

```text
PREFILL_V2=1 PREFILL_GRAPH_GEMM=1
python3 extra/qk/prefill_whole_synced.py --mode authority -K 8 --warmups 4 --rounds 3 ...
```

The next S10 validation loop is:

1. use the S9 authority path as the baseline escape hatch,
2. ignore one-shot smoke numbers for performance promotion,
3. keep generated DBUF/P4 parked,
4. search over the extracted LDS2 spec knobs only when the S9 authority path stays in the `>=4000 tok/s` pp512 band.

## Hybrid S9/S10 Bare Minimum

The minimum useful S10 phase is a no-performance-regression ownership layer:

```text
S9_FAST_BASELINE_WITH_S10_OWNERSHIP_METADATA
```

It uses:

```text
PREFILL_V2=1 PREFILL_GRAPH_GEMM=1
```

and does not use:

```text
PREFILL_WMMA_PIPE_PRIMITIVE=1
PREFILL_WMMA_LDS_PRIMITIVE=1
PREFILL_DBUF=1
```

Layer ownership:

| Layer | Execution source | S10 responsibility | Promotion risk |
|---|---|---|---|
| role selection | S9 route behavior | describe each role through `PrefillGEMMScheduleSpec` | low |
| pipe roles: `attn_qo`, `attn_kv`, `ffn_down` | S9 `build_gemm_pipe` backend atom | record/spec the selected pipe atom and params | low |
| LDS role: `ffn_gate/up` | S9 `lower_lds2_gemm_kernel` / `build_gemm_lds2` backend atom | record/spec reg layout, memory layout, wait policy, cadence, lifecycle | low |
| wait/layout search | S9 backend atom | choose only already-proven S9-safe spec knobs | medium-low |
| emitted instruction lifecycle | S9 backend atom | preserve byte/perf identity unless a candidate is explicitly measured | low if gated |
| generated DBUF replacement | none in this phase | parked | high; excluded |

Done for this minimum phase means:

| Gate | Required result |
|---|---|
| H0 role trace | one artifact maps every graph-GEMM role to `PrefillGEMMScheduleSpec` and selected backend atom. |
| H1 classification | route is classified as `compiler_primitive_spec_owned__asm_backend_atom` / hybrid, not pure. |
| H2 no primitive flags | authority run proves S10 metadata does not require generated pipe/LDS primitive flags. |
| H3 S9 authority preserved | pp512 remains `>=4000 tok/s` under `K=8,warmups=4,rounds=3,pin_clock`. |
| H4 search boundary | only S9-safe knobs are eligible: wait policy first, then byte-preserving layout/lifecycle metadata. |
| H5 hard path parked | generated rotated DBUF/P4 is not part of this phase's acceptance criteria. |

First implementation should create a route/spec audit artifact using existing code paths:

```text
bench/prefill-s10-lds2-ownership/hybrid-s9-s10-role-trace.json
```

Status: created by:

```bash
PYTHONPATH=. python3 extra/qk/prefill/s10_hybrid_role_trace.py
```

Expected rows:

| role | route family | backend atom | ownership claim |
|---|---|---|---|
| `attn_qo` | `pipe` | `build_gemm_pipe` | S10 records spec/params; S9 emits atom |
| `attn_kv` | `pipe` | `build_gemm_pipe` | S10 records spec/params; S9 emits atom |
| `ffn_down` | `pipe` | `build_gemm_pipe` | S10 records spec/params; S9 emits atom |
| `ffn_gate_up` | `lds` | `lower_lds2_gemm_kernel` / `build_gemm_lds2` | S10 records LDS2 layout/wait/cadence/lifecycle; S9 emits atom |

The pass/fail rule is simple:

```text
If metadata changes the emitted route or drops pp512 below 4000 tok/s, it is not S10 MVP.
If metadata preserves the S9 authority band and produces honest hybrid classification, S10 MVP passes.
```

## Narrow Hand-Coded DBUF Epoch Primitive

The next compromise is to leave only the hardest `ffn_gate_up` DBUF epoch choreography hand-coded, while S10 owns the
rest of the route/spec/search envelope.

Allowed hand-coded primitive:

```text
DBUFEpochPrimitive(
  nbuf=2,
  slot_expr="epoch % 2",
  prologue="produce epoch0 -> slot0",
  body="consume epoch i, produce epoch i+1 into the alternate slot",
  tail="consume final produced epoch",
)
```

Machine/spec-owned around it:

| Surface | Owner |
|---|---|
| role selection | S10 spec/search |
| tile shape, waves, `BK`, `PAD`, `PLRAB` | S10 spec/search |
| LDS reg/memory layout | S10 spec/search |
| wait policy | S10 spec/search |
| backend atom selection | S10 route/spec |
| epoch prologue/body/tail correctness | hand-coded `DBUFEpochPrimitive` |
| WMMA/DS/global-load encoding | backend ASM atoms |

This is still not pure generated. It is:

```text
hybrid compiler primitive with hand-coded DBUF epoch coordinator
```

It becomes a fine-tuned hand kernel again if the epoch primitive hardcodes fixed registers, fixed role-specific shape,
fixed epilogue, or emits the entire `ffn_gate_up` instruction lifecycle. The current trace must expose the primitive as
metadata, not hide it inside a pure/generated claim.

Current gate result:

| artifact | pp512 | verdict |
|---|---:|---|
| `bench/prefill-s10-lds2-ownership/hybrid-s9-s10-pp512-authority.json` | `4389` | pass |
| `bench/prefill-s10-lds2-ownership/hybrid-dbuf-epoch-pp512-authority.json` | `4408` | pass |

Unit gate:

```bash
PYTHONPATH=. pytest -q \
  test/unit/test_prefill_s10_hybrid_role_trace.py \
  test/unit/test_prefill_schedule_spec.py \
  test/unit/test_wmma_lds_spec.py \
  test/unit/test_prefill_wmma_lds2_reg_layout.py
```

Result: `43 passed`.

## Goal

S10 starts after S9 completed as:

```text
S9_COMPLETE_KEEP_OPT_IN
```

S10 is not a performance-tuning phase and not a pure-WMMA roofline phase. S10 is the ownership migration that moves
`ffn_gate/up` from a monolithic hand-kernel lifecycle toward a compiler/search-owned LDS2 primitive.

End state:

```text
ffn_gate/up
  -> PrefillGEMMScheduleSpec
  -> WMMALDSSpec / LDS2GemmSpec-owned layout, memory, wait, cadence, lifecycle
  -> route-owned primitive selection and artifacts
  -> backend emitter implementation
```

The current ASM emitter may remain the backend atom. The thing S10 removes is human ownership of the full kernel
lifecycle as one opaque `build_gemm_lds2(...)` call.

## Current Starting Point

Already built:

| Substrate | Status |
|---|---|
| `PrefillGEMMScheduleSpec` | captures resolved role schedule data |
| `WMMALDSSpec` | exists in `extra/qk/wmma_lds_spec.py` |
| `extract_wmma_lds_spec(...)` | extracts the LDS role spec from the prefill schedule |
| `wmma_lds_slot_identity_proof(...)` | proves A/B LDS slot windows and DBUF window identity |
| `wmma_lds_generated_env_defaults(...)` | points at the generated LDS transport substrate |
| `wmma_lds_postrange_opts(...)` | defines the current generated postrange opts |
| `PREFILL_WMMA_LDS_PRIMITIVE=1` route | exists as opt-in in `route_pf16_graph_gemm` |
| `lower_wmma_lds_spec(...)` | intentionally fails closed; does not call `build_gemm_lds2` |
| S9 knobs | wait/layout/lifecycle/PAD/search/report exist and default stays unchanged |

Current gap:

```text
The fallback default route still emits raw Ops.INS through build_gemm_lds2.
The opt-in LDS primitive route now proves route-bound sampled correctness, but whole-prefill composition does not compile yet.
```

## Current S10 Status

After the first S10 implementation pass:

| Gate | Status | Evidence |
|---|---|---|
| G0 baseline frozen | done | `bench/prefill-s10-lds2-ownership/baseline-freeze.json` |
| G1 spec owns lifecycle data | done | `WMMALDSSpec` serializes reg/memory/wait/cadence/lifecycle and roundtrips JSON |
| G2 route selects spec | done for opt-in trace/runtime surface | `PREFILL_WMMA_LDS_PRIMITIVE=1` route trace selects generated transport without `build_gemm_lds2` |
| G3 fallback explicit | done for trace | route trace records `fallback_reason`, `selected_surface`, and `calls_build_gemm_lds2` |
| G4 route trace | done | `bench/prefill-s10-lds2-ownership/route-trace.json` |
| G5 correctness smoke | done for isolated ffn_gate/up LDS primitive | `prefill_pipe_mvp_artifact.py --lds-primitive --lds-sample-correctness` passes sampled correctness for generated LDS and generated DBUF transports |
| G6 whole-prefill smoke | runs on the canonical `DEV=AMD` authority path; fails before route entry only when forced to `DEV=AMD:ISA` | `s10_compile_capture.py --scenario lds-only --mode authority` reaches the mixed S10 route on `DEV=AMD`; `DEV=AMD:ISA` fails during Q4_K weight realization with `AMD:ISA CAST dtypes.char -> dtypes.float unsupported` |
| G7 classification update | done | route manifest/surface guard classify S10 as spec-owned with ASM backend atom, not strict pure, and do not claim generated-pipe ownership for resource-gated `attn_kv` |

Current result:

```text
S10_PARTIAL_SPEC_OWNED_LDS_PRIMITIVE_VALIDATED
```

After decoupling the S10 LDS migration from the generated pipe primitive:

```text
S10_DECOUPLED_LDS_MIXED_ROUTE_COMPILES
```

Artifact:

```text
bench/prefill-s10-lds2-ownership/compile-capture/report-lds-only.json
```

Route:

```text
prefill_wmma_lds_dbuf_primitive_mixed
```

Role map:

```text
attn_qo     -> raw_pipe_oracle
attn_kv     -> raw_pipe_oracle
ffn_down    -> raw_pipe_oracle
ffn_gate_up -> lds_dbuf
```

Historical smoke result:

```text
captured_failures = 0
WHOLE-PREFILL@512 = 186 tok/s
binding_gate      = PREFILL_ROUTE_BINDING_PASS
```

This is a correctness/route-ownership smoke, not a performance result. It proves the `ffn_gate/up` LDS primitive path can
run inside whole-prefill once the generated pipe primitive is removed from the experiment.

Current-head tiny-slice result:

```text
PYTHONPATH=. DEV=AMD:ISA python3 extra/qk/prefill_pipe_mvp_artifact.py \
  --lds-primitive --lds-sample-correctness --sample-cols 8 --no-artifact --compact

lds_route_sample_correctness.passed      = true
lds_route_sample_correctness.rel_rmse    = 0.0002039360
lds_dbuf_route_sample_correctness.passed = true
lds_dbuf_route_sample_correctness.rel_rmse = 0.0002039362
generated_dbuf_cadence_probe.candidate_ok = true
generated_dbuf_cadence_probe.promoted      = true
```

So the tiny S10 slice works at the role-local level: the runtime opt-in route uses ordinary generated matmul transport,
does not use the hand LDS oracle, installs the warmstart key, and passes sampled correctness for `ffn_gate_up`. The
whole-prefill lds-only smoke is not yet a valid S10 verdict on `DEV=AMD:ISA`, because it fails before the prefill route
while realizing Q4_K weights:

```text
NotImplementedError: AMD:ISA CAST dtypes.char -> dtypes.float unsupported
```

The canonical whole-prefill authority path should use `DEV=AMD`, not `DEV=AMD:ISA`, because model-load fp16
realization still depends on the normal AMD/HIP lowering path for Q4_K byte-to-float dequant. With `DEV=AMD`, the
same lds-only S10 capture reaches the route and passes binding:

```text
PYTHONPATH=. DEV=AMD python3 extra/qk/prefill/s10_compile_capture.py \
  --scenario lds-only --mode authority --json

status = ok
captured_failures = 0
route = prefill_wmma_lds_dbuf_primitive_mixed
binding_gate = PREFILL_ROUTE_BINDING_PASS
role map:
  attn_qo     -> raw_pipe_oracle
  attn_kv     -> raw_pipe_oracle
  ffn_down    -> raw_pipe_oracle
  ffn_gate_up -> lds_dbuf
pp512 = 2336 tok/s
pp4096 = 1974 tok/s
```

Current-head S9 comparison under the same `DEV=AMD` authority methodology:

```text
PYTHONPATH=. DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 python3 extra/qk/prefill_whole_synced.py \
  --mode authority -K 8 --warmups 4 --rounds 3 --whole-lengths 512 --max-context 4608 --json

pp512 = 5107 tok/s
```

So the blocker is no longer "whole-prefill cannot reach S10." It is:

```text
S10_LDS_GENERATED_TRANSPORT_PERF_REGRESSION
```

The generated LDS/DBUF role-local transport is correct, but replacing the fast `ffn_gate_up` backend atom with it
roughly halves whole-prefill pp512 on current head. S10 should therefore keep the generated LDS/DBUF transport as a
correctness/R&D candidate, not the promoted mixed-route default, until its instruction structure is competitive with
the S9 `lower_lds2_gemm_kernel/build_gemm_lds2` backend atom.

After adding the pipe resource gate:

```text
S10_COMPOSED_RESOURCE_GATED_ROUTE_COMPILES
```

Artifact:

```text
bench/prefill-s10-lds2-ownership/compile-capture/report-composed-after-gate.json
```

Role map:

```text
attn_qo     -> pipe
attn_kv     -> pipe_resource_gated_raw_fallback
ffn_down    -> pipe
ffn_gate_up -> lds_dbuf
```

Smoke result:

```text
captured_failures = 0
WHOLE-PREFILL@512 = 221 tok/s
binding_gate      = PREFILL_ROUTE_BINDING_PASS
```

This is still not a promotion/performance result. It proves the original COMGR failure is removed by a pre-COMGR
resource gate, and it keeps the unresolved generated-pipe `attn_kv` work visible as a fallback instead of hiding it.
The composed route classification is therefore partial: `attn_qo` and `ffn_down` use generated pipe transport,
`ffn_gate_up` uses the LDS/DBUF primitive, and `attn_kv` remains explicitly on the raw pipe fallback until the generated
pipe local-staging plan is resource-safe.

The next blocker is not the isolated LDS primitive. The focused ffn_gate/up route sample passes:

```text
PREFILL_LDS_DBUF_PRIMITIVE_PROMOTED_STRUCTURAL_CORRECTNESS
rel_rmse = 0.000203936
finite = true
warmstart_key_present_after_route = true
```

Remaining blocker:

```text
PREFILL_GRAPH_GEMM=1
PREFILL_WMMA_PIPE_PRIMITIVE=1
PREFILL_WMMA_LDS_PRIMITIVE=1
PREFILL_DBUF=1
```

fails whole-prefill smoke with:

```text
tinygrad.device.CompileError: comgr fail 1, ERROR
```

Source capture now identifies the failing generated HIP kernel:

```text
artifact: bench/prefill-s10-lds2-ownership/compile-capture/report.json
source:   bench/prefill-s10-lds2-ownership/compile-capture/failed-001-HIPCompiler-9740011082908df4.cpp
kernel:   r_16_32_32_2_2_2_2_2_128_2_2
shape:    M=512, N=1024, K=4096
role:     attn_kv
family:   pipe
LDS:      69632 bytes declared shared memory
limit:    65536 bytes per workgroup
```

The declaration is:

```text
buf0[2048] half  -> 4096 bytes
buf2[32768] half -> 65536 bytes
total shared     -> 69632 bytes
```

So the S10 whole-route blocker is classified:

```text
S10_BLOCKED_PIPE_PRIMITIVE_ATTN_KV_LDS_OVERFLOW
```

This is not evidence that the isolated `ffn_gate/up` LDS primitive is wrong. The focused `ffn_gate/up` LDS route remains
sample-correct. The composed whole-prefill route fails earlier because `PREFILL_WMMA_PIPE_PRIMITIVE=1` sends the
`attn_kv` pipe role through a generated local-staging path that emits more LDS than gfx1100 permits.

Decoupling is now implemented as:

```text
PREFILL_GRAPH_GEMM=1
PREFILL_WMMA_LDS_PRIMITIVE=1
PREFILL_DBUF=1
PREFILL_WMMA_PIPE_PRIMITIVE unset
```

This selects:

```text
prefill_wmma_lds_dbuf_primitive_mixed
```

The original issue was isolated to the composed generated-pipe route:

```text
PREFILL_WMMA_PIPE_PRIMITIVE=1 + PREFILL_WMMA_LDS_PRIMITIVE=1 + PREFILL_DBUF=1
```

The failure class is generated pipe transport picking up an LDS/local-staging plan that is legal for larger roles
or LDS roles but illegal for small-N `attn_kv`: output shape `512x1024` already forces a smaller tile/workgroup shape, and
the emitted Tensor matmul transport still declares a full `65536` byte local B buffer plus another `4096` byte local A
buffer. That is the exact over-budget resource event.

Implemented mitigation:

```text
pipe_primitive_local_stage_resource_plan(...)
```

If generated pipe local staging is requested for the captured unsafe small-N shape, the route falls back before COMGR to
the existing raw pipe emitter and records the fallback reason on the layer object.

Remaining work for the pipe side is one of:

1. keep `attn_kv` off generated pipe local staging,
2. make generated pipe local staging resource-aware before COMGR,
3. use a smaller local tile for `N=1024`,
4. or leave `attn_kv` on the raw fallback while S10 finishes LDS ownership.

## Non-Goals

| Non-goal | Reason |
|---|---|
| Pure-WMMA practical peak | Deferred explicitly. |
| More S9 tuning | S9 already resolved as opt-in/no default promotion. |
| Reopen 4x4 | Parked on gfx1100 register pressure. |
| Delete all ASM | Too broad; S10 only removes full-lifecycle ownership. |
| Clone `build_gemm_lds2` into a second full instruction list | That preserves the problem under another name. |

## Definition Of 100%

S10 MVP is complete when:

| Gate | Done means |
|---|---|
| G0 baseline frozen | S9 artifacts and route identity are referenced from this doc. |
| G1 spec owns lifecycle data | One serializable LDS2 spec contains shape, memory layout, reg layout, wait policy, cadence, lifecycle, and S9 opt-in selection. |
| G2 route selects spec | `ffn_gate/up` can be routed by spec identity without calling `build_gemm_lds2` on the selected S10 path. |
| G3 fallback is explicit | Unsupported/generated-lowerer failures fall back only when allowed and are recorded as fallback, not misclassified as generated. |
| G4 trace proves lifecycle ownership | A route trace records `role -> PrefillGEMMScheduleSpec -> WMMALDSSpec -> selected lowerer -> emitted surface`. |
| G5 correctness smoke | S10 opt-in route returns finite/correct output on the active role shape or records exact blocker. |
| G6 whole-prefill smoke | Existing whole-prefill harness runs with S10 route flags and records pp512/pp4096 or exact blocker. |
| G7 classification update | Manifest/docs classify the route honestly: compiler primitive with ASM backend atom, not pure generated, not full hand kernel. |

S10 is not complete if the only result is another raw `Ops.INS` kernel behind a new name.

## Phase Plan

### S10.0 Baseline Freeze

Inputs:

```text
bench/prefill-lds2-s9/final-report.json
bench/prefill-lds2-s9/roofline-audit.json
bench/prefill-whole-synced/raw-hand-s9-combined-default-authority.json
bench/prefill-whole-synced/raw-hand-s9-combined-best-authority.json
```

Output:

```text
bench/prefill-s10-lds2-ownership/baseline-freeze.json
```

Must record:

- current route id,
- current role classification,
- S9 default-vs-opt-in decision,
- active shape,
- whole-prefill baseline band.

### S10.1 Spec Completion

File owner:

```text
extra/qk/wmma_lds_spec.py
test/unit/test_wmma_lds_spec.py
```

Work:

1. Add a single serializable `LDS2GemmSpec` or extend `WMMALDSSpec` to carry:
   - `LDS2RegLayout`,
   - `LDS2MemoryLayout`,
   - `LDS2WaitPolicy`,
   - `LDS2Cadence`,
   - `LDS2LifecycleTemplate`,
   - S9 selection label.
2. Add `from_prefill_schedule(...)`.
3. Add `to_json()` / `from_json()` roundtrip.
4. Add `ownership_classification()`:

```text
compiler_primitive_with_asm_backend_atom
```

Done means unit tests can serialize the active `ffn_gate_up` spec and prove legality without calling the hand kernel.

### S10.2 Route And Trace

File owner:

```text
extra/qk/prefill_graph_gemm_route.py
extra/qk/prefill/kernel_lifecycle_trace.py
test/unit/test_prefill_graph_gemm_route.py
test/unit/test_prefill_kernel_lifecycle_trace.py
```

Work:

1. Add route trace fields for the LDS primitive opt-in:

```text
role
route_family
schedule_spec
lds_spec
selected_surface
fallback_reason
classification
calls_build_gemm_lds2
```

2. Ensure S10 opt-in path records whether it used:

```text
generated_transport
asm_backend_atom
fallback_raw_oracle
```

3. Add a smoke command that emits a trace artifact without running whole-prefill:

```text
bench/prefill-s10-lds2-ownership/route-trace.json
```

Done means a test can prove the selected S10 route identity does not silently become the old raw oracle.

### S10.3 Lowering Adapter MVP

File owner:

```text
extra/qk/wmma_lds_spec.py
extra/qk/prefill_graph_gemm_route.py
test/unit/test_wmma_lds_spec.py
```

Allowed first MVP:

```text
lower WMMALDSSpec through existing lower_lds2_gemm_kernel only when explicitly requested as asm_backend_atom
```

This is not pure generated, but it is no longer route-local ownership of a whole hand kernel. It must be classified as:

```text
compiler_primitive_spec_owned__asm_backend_atom
```

Rules:

- The adapter must accept a spec object, not raw loose parameters.
- It must not call `build_gemm_lds2`; if it uses the ASM backend, call the named lowerer boundary.
- It must emit an artifact/classification proving the lifecycle data came from the spec.
- Default route unchanged.

Done means byte parity with current default is proven for the active shape when using default S9 settings.

### S10.4 Search Integration

File owner:

```text
extra/qk/prefill/lds2_s9_combined_search.py
extra/qk/wmma_lds_spec.py
```

Work:

1. Move S9 candidate selection into the spec object.
2. Replace S9 env-only candidate application with spec construction where practical.
3. Keep env knobs as compatibility/CLI overrides only.

Done means the combined candidate can be represented as one spec JSON row.

### S10.5 Correctness And Timing Smoke

Use existing harnesses only:

```text
extra/qk/prefill/hand_vs_generated_shape_matrix.py
extra/qk/prefill_whole_synced.py
extra/qk/prefill_harness.py
```

Artifacts:

```text
bench/prefill-s10-lds2-ownership/micro-smoke.json
bench/prefill-s10-lds2-ownership/whole-smoke.json
```

Done means:

- active role shape is correct, or exact blocker is recorded,
- whole-prefill route runs, or exact blocker is recorded,
- no new harness duplicates existing harness responsibilities.

### S10.6 Classification And Manifest

File owner:

```text
extra/qk/route_manifest.py
extra/qk/pure_search_guard.py
docs/asm-tool-vs-hand-kernel-policy-scope.md
```

Work:

1. Add/adjust route classification for the S10 route.
2. Distinguish:

```text
full_hand_kernel
asm_backend_atom
compiler_primitive_spec_owned
pure_generated
```

3. Ensure guard output does not claim purity for the ASM backend atom.

Done means route census and docs agree on the classification.

## Parallel Work Split

| Lane | Can run in parallel? | Owns | Output |
|---|---|---|---|
| A. Spec completion | yes | `wmma_lds_spec.py`, `test_wmma_lds_spec.py` | serializable ownership spec |
| B. Route/trace | yes | route + lifecycle trace tests | route trace artifact path and fallback classification |
| C. Harness/artifact | yes | new S10 artifact runner using existing harnesses | baseline/micro/whole smoke artifacts |
| D. Classification | yes after A/B shape known | manifest/guard/docs | honest route taxonomy |
| Main integration | sequence after A-C | docs + final tests + active smoke | S10 MVP verdict |

## Expected Blockers

| Blocker | Meaning | Escape hatch |
|---|---|---|
| generated transport too slow/wrong | S10 generated path cannot replace oracle yet | keep ASM backend atom but spec-owned |
| lowerer still needs raw `Ops.INS` | acceptable only if emitted through spec-owned backend atom | classify honestly |
| route trace cannot distinguish fallback | S10 cannot prove ownership | fix trace before performance work |
| whole-prefill route falls back silently | invalid S10 result | add hard gate/failure artifact |

## First Implementation Order

1. Finish spec serialization/ownership classification.
2. Add route trace proof.
3. Add S10 artifact runner that reuses existing harnesses.
4. Add ASM-backend-atom adapter if needed.
5. Run micro/whole smoke.
6. Update route classification.
7. Decide S10 MVP status:

```text
S10_MVP_SPEC_OWNED_ASM_BACKEND_ATOM
S10_BLOCKED_GENERATED_TRANSPORT
S10_BLOCKED_TRACE_OR_CLASSIFICATION
```
