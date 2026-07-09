# 8B Prefill E2E MVP And Lifecycle Ownership Scope

Date: 2026-07-08.

## Goal

Make two pushes, deliberately separated:

1. **E2E MVP push:** prove the prefill route can flow from role/schedule selection into a compiler-owned pipe primitive
   and run end-to-end behind an opt-in flag.
2. **Full lifecycle push:** move the actual middle pipeline ownership out of the hand oracle and into reusable compiler
   primitives until the hand route is only an oracle/escape hatch.

The MVP is not allowed to relabel the hand kernel as generated. The MVP is allowed to be slower, narrower, and
opt-in. Its job is to prove the control plane and replacement seam.

Target end state:

```text
model prefill role
  -> schedule/search selects PrefillGEMMScheduleSpec
  -> extract WMMAPipeSpec
  -> generated/backend-owned pipe lowerer
  -> b128 loads + targeted waitcnt + WMMA + epilogue
  -> whole-prefill run
```

Current state:

```text
model prefill role
  -> schedule/spec selects pipe or LDS
  -> emit_prefill_gemm_from_spec
  -> _emit_schedule
  -> extra/qk/prefill/wmma.py::build_gemm_pipe / build_gemm_lds2
  -> UOp(Ops.INS, ...)
```

## Definitions

| Term | Meaning |
|---|---|
| E2E MVP | A role can route through `WMMAPipeSpec` and execute without using `build_gemm_pipe` as the selected generated path. |
| Full lifecycle ownership | The compiler/backend owns load staging, wait placement, WMMA accumulation, fragment lifetime, and epilogue for the pipe route. |
| Oracle | The current `PREFILL_GRAPH_GEMM=1` raw instruction-list route through `extra/qk/prefill/wmma.py`. |
| Generated candidate | Any opt-in route that executes through tinygrad/codegen/backend primitives rather than route-local full-kernel `Ops.INS`. |
| Promotion | Replacing an oracle role in the default route after correctness, provenance, and performance gates pass. |

## Scope Boundary

MVP includes only the register-resident pipe roles:

| Role | Shape `(M,N,K)` | MVP status |
|---|---:|---|
| `attn_qo` | `512,4096,4096` | first target |
| `attn_kv` | `512,1024,4096` | second target after `attn_qo` |
| `ffn_down` | `512,4096,12288` | third target after role expansion |

MVP excludes:

- `ffn_gate_up` LDS2/DBUF replacement,
- generated 4x4 on gfx1100,
- Q4_K fused decode replacement,
- DBUF LDS ping-pong,
- broad search over LDS fields.

Those remain part of the full lifecycle track after the pipe route is proven.

## E2E MVP Definition Of Done

The E2E MVP is complete when all gates below pass for `attn_qo`.

| Gate | Requirement |
|---|---|
| M0 baseline | Existing oracle and generated-default commands are documented with artifact paths. |
| M1 route seam | `describe_prefill_schedule(...) -> extract_wmma_pipe_spec(...)` succeeds for the target role. |
| M2 opt-in route | An env-guarded generated pipe path can be selected without changing the default route. |
| M3 no hand pipe | The selected MVP path does not call `extra/qk/prefill/wmma.py::build_gemm_pipe`. |
| M4 compiles | The generated candidate compiles on `DEV=AMD:ISA`. |
| M5 runs | The candidate runs for a bounded shape first, then the target role. |
| M6 correctness | Output is finite and passes the existing prefill numerical threshold. |
| M7 trace | Rendered/lifecycle trace shows b128 global loads, WMMA, targeted/non-full waitcnt, and generated route attribution. |
| M8 artifact | Result is written to a stable artifact path under `bench/prefill-pipe-mvp/`. |
| M9 default safety | `PREFILL_GRAPH_GEMM=1` oracle and `PREFILL_GRAPH_GEMM=0` default still behave as before. |

Performance target for MVP:

- Must be measured.
- Must not be used as a promotion gate yet.
- A slower MVP is acceptable if it proves provenance, correctness, and route binding.

Status update, 2026-07-08:

- `attn_qo` route-bound sampled correctness passes on `DEV=AMD:ISA:gfx1100` through
  `prefill_wmma_pipe_primitive_generated` with finite/nonzero output and rel RMSE about `2.1e-4`.
- Existing whole-prefill smoke passes on `DEV=AMD` with `PREFILL_GRAPH_GEMM=1` and
  `PREFILL_WMMA_PIPE_PRIMITIVE=1`; route attribution is pure generated, not the raw graph-GEMM oracle.
- Full whole-model `DEV=AMD:ISA:gfx1100` remains blocked outside the pipe primitive by broader non-GEMM lifecycle
  lowering, currently dynamic `CDIV` from elementwise/index code in the first layer.

## Full Lifecycle Definition Of Done

Full lifecycle ownership is complete role-by-role when:

| Layer | Requirement |
|---|---|
| Route policy | Search/spec owns role selection and schedule parameters. |
| Pipe spec | `WMMAPipeSpec` carries all fields needed by lowering and candidate ledgers. |
| Fragment loads | Generated lowering emits packed b128 global loads for A/B fragments. |
| Fragment lifetime | Compiler metadata distinguishes `F0/F1`, A/B banks, accumulator ranges, and reload safety. |
| Waitcnt | Backend emits targeted `vmcnt(pipe_tm*2 + pipe_tn*2)` and does not full-drain future-stage loads. |
| WMMA | Generated lowering feeds backend-owned WMMA instructions. |
| Epilogue | Generated backend stores accumulators without hand oracle help. |
| Correctness | Role output passes numerical gates across target shapes. |
| Performance | Per-role generated candidate reaches promotion threshold against the current pipe oracle. |
| Provenance | Purity audit shows no route-local raw full-kernel instruction list for the promoted role. |

## MVP Architecture

Use the existing seam:

```python
spec = describe_prefill_schedule(out_f, in_f, role=role)
pipe = extract_wmma_pipe_spec(spec)
```

Add an opt-in branch in:

```text
extra/qk/prefill_schedule_spec.py::emit_prefill_gemm_from_spec
```

Target shape:

```text
if PREFILL_WMMA_PIPE_PRIMITIVE=1 and spec.route_family == "pipe":
  pipe_spec = extract_wmma_pipe_spec(spec)
  return lower_wmma_pipe_spec(pipe_spec)
else:
  return _emit_schedule(...)
```

Fail closed:

- if `extract_wmma_pipe_spec(spec)` returns `None`, use the existing route,
- if generated lowering is incomplete, raise a clear `NotImplementedError` only when the opt-in flag is set,
- never silently fall back to `build_gemm_pipe` while claiming the generated path was selected.

## Minimal Lowering Strategy

The MVP lowerer should start as the narrowest useful generated path.

Allowed for MVP:

- use existing tinygrad/codegen/backend primitives,
- emit a bounded pipe candidate for a diagnostic shape before full 8B role,
- be behind `PREFILL_WMMA_PIPE_PRIMITIVE=1`,
- reuse existing correctness and lifecycle trace harnesses,
- be slower than the oracle.

Not allowed for MVP:

- copy `build_gemm_pipe` instruction lists,
- create a new route-local full GEMM `Ops.INS` emitter,
- call the generated path pure if it still executes through `wmma.py`,
- alter `ffn_gate_up` LDS route,
- change default prefill behavior before gates pass.

## Implementation Phases

### E0. Baseline And Guard

Tasks:

- Keep [docs/8b-prefill-phase0-measurement.md](8b-prefill-phase0-measurement.md) as the baseline command source.
- Add the opt-in env name to docs/tests: `PREFILL_WMMA_PIPE_PRIMITIVE=1`.
- Add a test that the env flag is off by default and current emission still calls `_emit_schedule`.

Done when:

- default behavior is unchanged,
- opt-in branch is visible in tests,
- no generated path claim can hide fallback to `build_gemm_pipe`.

### E1. Pipe Lowerer Stub

Tasks:

- Add `lower_wmma_pipe_spec(spec)` next to `WMMAPipeSpec` or in a backend-owned module.
- Initially return a structured object or raise `NotImplementedError` under opt-in.
- Add tests that:
  - pipe specs enter the new lowerer under opt-in,
  - LDS specs do not enter it,
  - unsupported specs fail closed.

Done when:

- the route seam is executable and test-pinned without runtime behavior change.

### E2. Bounded Generated Pipe Candidate

Tasks:

- Pick the smallest diagnostic pipe shape that exercises:
  - b128 global loads,
  - at least two WMMA steps,
  - targeted/non-full waitcnt,
  - epilogue store.
- Lower it through existing backend primitives, not raw full-kernel `Ops.INS`.
- Run correctness on the diagnostic shape.
- Run lifecycle trace on the diagnostic shape.

Done when:

- one bounded candidate compiles and runs,
- trace proves generated b128 + WMMA + targeted waitcnt,
- no route-local full instruction list is used.

### E3. Route-Bound Pipe-Role MVP

Tasks:

- Bind the generated pipe candidate to `attn_qo`, `attn_kv`, and `ffn_down` only under
  `PREFILL_WMMA_PIPE_PRIMITIVE=1`.
- Run isolated correctness for `M=512,N=4096,K=4096`, `M=512,N=1024,K=4096`, and `M=512,N=4096,K=12288`.
- Run lifecycle trace and store artifact.
- Run whole-prefill smoke with the opt-in flag.

Done when:

- `attn_qo` runs E2E through the generated pipe path,
- artifact records route attribution, correctness, and timing,
- default oracle and default generated routes remain unchanged.

### E4. MVP Report

Tasks:

- Write `bench/prefill-pipe-mvp/latest.json`.
- Record:
  - env,
  - role,
  - shape,
  - `PrefillGEMMScheduleSpec.to_json()`,
  - `WMMAPipeSpec.to_json()`,
  - route attribution,
  - correctness,
  - trace counters,
  - timing.
- Update the scope doc with PASS/BLOCKED status.

Status: schema-ready, runtime result blocked until E2/E3 generated pipe execution exists.

Artifact owner:

- `extra/qk/prefill_pipe_mvp_artifact.py`
- `bench/prefill-pipe-mvp/latest.json`

Schema version:

- `prefill-pipe-mvp-result.v1`

Required top-level fields:

- `env`: Python/platform/git/device plus relevant prefill env flags.
- `role`: model role, initially `attn_qo`.
- `shape`: `{m,n,k}` for the measured role.
- `prefill_gemm_schedule_spec`: `PrefillGEMMScheduleSpec.to_json()`.
- `wmma_pipe_spec`: `WMMAPipeSpec.to_json()`.
- `route_attribution`: selected route, route family, generated-pipe selection, and whether the hand pipe oracle was used.
- `correctness`: finite status, numerical threshold, max absolute error, and max relative error.
- `trace_counters`: b128 global loads, WMMA count, targeted waitcnt count, full waitcnt count, and generated route attribution.
- `timing`: run status, samples, median milliseconds, and TFLOPS.

The validator rejects an artifact that claims `generated_pipe_selected=true` while `uses_hand_pipe_oracle=true`.
The schema-only writer may emit `PREFILL_PIPE_MVP_SCHEMA_READY` with correctness/timing marked `not_run`; the runtime
E3/E4 result must replace those fields with measured values.

Done when:

- the MVP can be rerun from one command sequence,
- failure mode is explicit if performance is poor.

## Full Lifecycle Phases After MVP

### L0. Expand Pipe Roles

Add `attn_kv` and `ffn_down` only after `attn_qo` MVP passes.

Gate:

- same correctness/provenance checks as `attn_qo`,
- measured per-role throughput,
- no hidden fallback.

### L1. Strengthen Fragment Lifetime

Add compiler-visible metadata:

- stage `F0/F1`,
- operand `A/B`,
- bank identity,
- accumulator identity,
- WAR guard for reload-before-consume,
- use-cluster boundaries.

Gate:

- trace can prove the sequence:

```text
load F0
load F1
wait/consume F0
reload F0
wait/consume F1
```

### L2. Strengthen Wait Policy

Make targeted waits a backend policy rather than route behavior.

Gate:

- `pipe_tm=2, pipe_tn=2` emits `vmcnt(8)`-class waits,
- future-stage loads remain outstanding,
- full drains are diagnostic only.

### L3. Search Owns Pipe Specs

Expose only the first safe knobs:

- `pipe_tm`,
- `pipe_tn`,
- logical primitive `tile_m/tile_n`,
- `tile_k/k_step`,
- `wait_policy`,
- `epilogue_policy`.

Do not expose LDS/DBUF knobs in the pipe MVP search.

Gate:

- candidate rows go to `bench/prefill-pipe-spec-search/latest.json`,
- route manifests change only after promotion/refutation.

### L4. Promote Or Keep Dual Route

Per role:

- promote if generated pipe reaches correctness, provenance, and performance thresholds,
- keep dual-route if it is correct but materially slower,
- keep oracle if generated route is blocked or unstable.

### L5. Return To LDS/DBUF

Only after pipe route is resolved:

- extract single-buffer LDS staging,
- then DS offset/address lifetime,
- then DBUF slot ownership,
- then full LDS2 lifecycle.

## Work That Can Run In Parallel

| Workstream | Parallel? | Owner files | Output |
|---|---:|---|---|
| Env guard and route seam tests | yes | `extra/qk/prefill_schedule_spec.py`, tests | opt-in branch pinned |
| Lowerer stub | yes | `extra/qk/wmma_pipe_spec.py` or backend module, tests | `lower_wmma_pipe_spec` contract |
| Measurement wrapper | yes | docs/bench artifacts only | MVP command sequence and artifact schema |
| Trace gate hardening | yes | `kernel_lifecycle_trace.py`, tests | proves b128/WMMA/waitcnt/no raw full kernel |
| Diagnostic generated candidate | no | backend/codegen | starts after lowerer contract |
| Route-bound `attn_qo` | no | prefill route/spec | starts after diagnostic candidate passes |
| Role expansion | no | schedule/search | starts after `attn_qo` MVP passes |

## Blockers To Track

| Blocker | Affects MVP? | Notes |
|---|---:|---|
| No generated pipe lowerer yet | yes | Main MVP blocker. |
| Fragment lifetime model incomplete | yes | MVP can start narrow, but full pipe needs this. |
| Generated structural census bug | partly | `prefill_route_census.py` generated rows hit `AttributeError: 'int' object has no attribute '_fields'`; whole-prefill timing is not blocked. |
| Performance gap vs oracle | no for MVP, yes for promotion | MVP must measure it, not solve it. |
| LDS/DBUF pressure/faults | no | Out of MVP scope. |
| `ffn_gate_up` LDS oracle | no | Keep as oracle until after pipe roles. |

## Stop Conditions

Stop the MVP push if:

- the opt-in branch cannot avoid `build_gemm_pipe`,
- the generated candidate cannot compile even on the diagnostic shape,
- correctness fails in a way that cannot be isolated to layout/indexing,
- route attribution cannot distinguish generated candidate from oracle fallback.

Do not stop the MVP push merely because:

- performance is below oracle,
- only `attn_qo` works,
- LDS/DBUF is still hand-owned,
- strict purity cannot yet be claimed for all prefill roles.

## First Concrete Patch Sequence

1. Add `PREFILL_WMMA_PIPE_PRIMITIVE` env guard and route-seam tests.
2. Add `lower_wmma_pipe_spec(spec)` stub and tests.
3. Wire `emit_prefill_gemm_from_spec` to call the stub only under opt-in for `route_family == "pipe"`.
4. Add artifact schema for `bench/prefill-pipe-mvp/latest.json`.
5. Implement the smallest generated diagnostic pipe lowerer.
6. Run diagnostic correctness and lifecycle trace.
7. Bind to `attn_qo` under opt-in.
8. Run whole-prefill smoke.
9. Decide whether to expand roles or fix the lowerer.
