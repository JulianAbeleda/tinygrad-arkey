# Native-ISA L4 software-pipeline scope

Date: 2026-07-07.

## Superseded Scope Note

The `4x4` generated path described in this document is parked on gfx1100 as of 2026-07-08. The active software-pipeline
work is the same primitive, but applied to fitting shapes: `2x2`, `4x2`, and `2x4`. Do not use this document to justify
new `4x4` spill/NaN work unless `docs/gfx1100-4x4-path-parked-scope.md` is explicitly reopened.

Goal: close the remaining generated-vs-hand performance gap after L3/L5/L6 correctness work. The generated native-ISA
route is now correct and emits direct b128 route-shaped fragments, but it does not expose the hand kernel's load/compute
overlap.

## Current facts

- L3 is closed for the native-ISA prefill shape: route `a @ b.transpose()` emits 16 `global_load_b128`, 0
  `global_load_u16`, 0 `v_pack_b32_f16`, and 16 `v_wmma` for the 4x4 route-shaped probe.
- L5 is closed: 4x4 generated WMMA no longer NaNs on GPU.
- L6 is correctness-valid but not promotable: targeted waitcnt is correct, scalar-pack waits are coalesced, and the old
  ~15 TFLOPS regression is gone; it still does not create hand-class overlap by itself.
- L4 is the remaining blocker: generated code has one live A/B fragment bank and therefore schedules as:

```text
load all A/B fragments
wait
v_wmma all subtiles
```

The hand `build_gemm_pipe` shape is:

```text
prologue: load F0
loop:
  load F1
  wait/use F0
  load F0 for next K
  wait/use F1
tail
```

So the issue is not another waitcnt rule. Codegen must expose at least two pipeline phases or an equivalent overlap
shape before waitcnt can hide latency.

## Parallel audit results

- I0 probe exists at `extra/qk/prefill/native_isa_l4_stream_probe.py`. Baseline route-shaped output confirms
  `global_load_b128=16`, `global_load_u16=0`, `v_pack_b32_f16=0`, `v_wmma=16`, and no b128 loads between WMMAs. Targeted
  waitcnt output confirms `s_waitcnt=10` with `vmcnt` values `12,10,8,6,4,2,0`, still with no overlap.
- The first DBUF compile blocker is understood: `PREFILL_DBUF=1` creates a two-node WMMA chain per subtile whose head
  reads the loop-carried accumulator via `Ops.LOAD`; the old `isel_wmma` chain path assumed a const-zero seed.
- After accepting the accumulator-load chain head, the DBUF route reaches regalloc and fails with `Inc 0: no spills`.
  This moves the blocker from "shape not recognized" to "shape exceeds current register footprint/liveness".
- Smaller register-route DBUF is feasible: with one repeated output upcast (`m_up=1`) the route-shaped DBUF probe compiles,
  exposes b128 loads between WMMAs, and GPU-passes (`rmse=0.001613`, `nan=0`). This proves the renderer can lower the
  unroll-by-2 accumulator-load chain when the footprint fits.
- Current route-shaped pinned ranges are: C `v8..v135`, resident A/B `v136..v199`, high scratch/pool `v200..v255` plus
  low scratch `v1..v7`. Final b128 loads are clustered before WMMAs (`load -> wait -> compute`).
- Literal full F0/F1 A+B banks are rejected for 4x4: C 128 + A/B 64 + second A/B 64 = 256 pinned VGPRs before address,
  scratch, and epilogue temps. This matches the hand `build_gemm_pipe(64,64,64,4,4)` overflow arithmetic.
- Whole-route smoke with `PREFILL_DBUF=1 AMD_ISA_WAITCNT_TARGETED=1 AMD_ISA_WMMA_B128_FRAG=1` remains in the current
  table-local band (33.29/36.74 TFLOPS for the checked schedule-table shapes), so smaller DBUF/interleaving is not yet a
  performance solution.
- LDS pivot status: one-sided A local staging now compiles for full 4x4 and emits wide LDS fragment reads:
  `ds_load_b128=8`, `ds_store_b32=16`, `s_barrier=1`, `v_wmma=16` for
  `PREFILL_TC_LOCAL_STAGE=a PREFILL_TC_LOCAL_STAGE_POST=1 AMD_ISA_WMMA_B128_FRAG=1 --m-up 2`.
- The LDS pivot is not solved yet. Current staging still writes LDS with scalar `ds_store_b32` stores, leaves the B side
  outside the cooperative LDS rewrite, and does not create the hand `build_gemm_lds2` two-slot cadence.
- `PREFILL_TC_LOCAL_STAGE=a PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_DBUF=1 AMD_ISA_WAITCNT_TARGETED=1
  AMD_ISA_WMMA_B128_FRAG=1 --m-up 2` still fails with `Inc 0: no spills`. Combining the current scalar-store LDS stage
  with DBUF increases pressure rather than solving the phase-bank issue.
- `PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_POST=1 --m-up 2` also spills. Full both-side scalar local staging is
  not the implementable hand-class route.
- The existing cooperative-B LDS rewrite still skips the route-shaped operand because it sees an `AxisType.GLOBAL` tile
  range. That is a codegen-shape gate, not an ISA encoding gap.
- Renderer-side wide LDS store/read primitives now exist and are unit-tested. `DS_LOAD_B128` and `DS_STORE_B128` both
  lower with `offset0`; `DS_STORE_B128` selection is intentionally narrow and only fires for a packed
  `int32.vec(4)` carrier backed by contiguous fixed VGPRs.
- Route-shaped cooperative LDS staging across `AxisType.GLOBAL` now preserves the 4x4 expansion shape at the graph level:
  after expansion the WMMA is `float.vec(128)` with `half.vec(256)` operands, matching the baseline route shape. The full
  lowering is still blocked: with verifier on it can still hit the local-pointer `PTRCAT` shape, and when that is bypassed
  it reaches regalloc and fails with `Inc 0: no spills`.
- The fitting diagnostic remains `m_up=1`: cooperative LDS staging compiles and emits 4 WMMAs with wide LDS reads, but
  scalar global-to-LDS stores dominate the stream. This proves the rewrite/lowering path works when the footprint fits,
  and isolates the full 4x4 blocker to live range/register pressure plus scalar staging shape.

## Non-goals

- Do not reopen the solved 4x4 NaN theories.
- Do not make a second hand-written emitter or raw-`Ops.INS` route.
- Do not promote `AMD_ISA_WAITCNT_TARGETED=1` by default until L4 produces a measured win.
- Do not treat HIP cooperative-B staging as the native-ISA L4 blocker. That remains a separate HIP/postrange medium-stage
  route issue.

## Constraints

| Constraint | Why it matters |
|---|---|
| 4x4 C accumulators cost 128 VGPRs | 16 subtiles x 8 VGPRs. |
| Current resident A/B costs 64 VGPRs | 4 A rows + 4 B cols, each 8 VGPRs. |
| Naive second resident A/B bank costs another 64 VGPRs | 128 + 64 + 64 = 256 before address/scratch/epilogue regs, so naive F0/F1 duplication is not viable for 4x4. |
| `PREFILL_DBUF=1` currently fails native-ISA route lowering | For route-shaped 4x4 it reaches `isel_wmma` with `C init lane 0 is Ops.LOAD, expected CONST`. |
| `_schedule` is local basic-block list scheduling | It can reorder exposed independent work; it cannot synthesize prologue/body/tail or alternate banks. |
| `_insert_waitcnt` is now capable of targeted waits | Useful only after the pipeline exposes loads from a future phase. |

## Exhaustive candidate set

| ID | Candidate | Expected payoff | Main blocker | First gate |
|---|---|---:|---|---|
| A | Fix unroll-by-2 rolled-accumulator lowering only | Done for fitting shapes | 4x4 still spills; smaller tile compiles/GPU-passes | `m_up=1 PREFILL_DBUF=1` route-shaped GPU pass |
| B | Phase-key current resident A/B banks literally (`phase 0/1`) | Rejected for 4x4 | 256 pinned VGPRs before scratch/address/epilogue | Only revisit with smaller tile or major lifetime reduction |
| C | Partial second bank: duplicate only the operand side that creates exposed latency | Medium | Need identify whether A-only or B-only overlap moves TFLOPS enough | Disasm shows future-phase b128 loads between WMMA groups; GPU-passes |
| D | Stream/reload fragments instead of full resident A/B for one phase | Medium/high | Reintroduces WAR hazards and may lose row/col reuse | Structural stream has future loads before current WMMAs without extra 64 pinned VGPRs |
| E | LDS DBUF route (`build_gemm_lds2`-style) | High for computed/quantized operands | Native-ISA LDS staging ownership is larger than register route | Generated route has two LDS slots, barriers, targeted `lgkmcnt`, and beats table baseline |
| F | Explicit modulo/software-pipeline pass over final ISA | High but risky | Must preserve labels, hazards, branch offsets, waitcnt and register lifetimes | Behind `AMD_ISA_SWPIPE=1`, byte/disasm proof plus GPU parity |
| G | Reduce tile shape to make literal double-bank fit | Medium | Changes occupancy/tile math and may lose hand target | Search finds smaller WM/WN with better TFLOPS than current 4x4 |

## Recommended order

1. **DONE: compile the DBUF/unroll shape far enough to classify pressure.**
   Fix the current `isel_wmma` rejection so a peeled rolled-accumulator chain whose C source is an accumulator `LOAD`
   lowers to the same pinned in-place C range instead of being treated as a const-seeded unrolled chain. This is a
   necessary diagnostic even if it is not sufficient for performance. Current result after the first unblock: `m_up=1`
   compiles and GPU-passes; `m_up=2` reaches regalloc and spills, confirming the 4x4 footprint problem.

2. **Measure the compiled A-shape structurally.**
   Inspect final ISA for:
   - more than one K-copy in the loop body,
   - whether A/B fragment regs are same-bank or phase-distinct,
   - whether future loads appear before current WMMAs,
   - waitcnt placement around those loads.

3. **Skip literal full phase banks for 4x4.**
   The static budget rejects them. The next implementable register-route experiments are partial-bank and streamed
   fragment designs.

4. **Current pivot: pursue LDS DBUF, but do it as a real hand-shape replacement, not scalar local staging.**
   The first renderer primitive is done: contiguous LDS half fragments lower to two `ds_load_b128` instructions. The
   remaining primitive gap is upstream staging: codegen must express cooperative global b128 loads into packed temporaries
   and wide `ds_store_b128` into two LDS slots, then load the current slot with `ds_load_b128` while the next slot is in
   flight.

5. **Only then promote targeted waitcnt.**
   L6 promotion requires a pipeline shape where nonzero `vmcnt(n)` or `lgkmcnt(n)` waits leave future loads outstanding
   during current WMMAs and improve TFLOPS.

## 100% Definition

This work is 100% complete only when all gates below are satisfied. Anything less is a partial milestone.

| Gate | Required condition | Evidence |
|---|---|---|
| S0. No regression | Existing non-LDS route-shaped native-ISA 4x4 still compiles, GPU-passes, and keeps the L3/L5/L6 fixes. | `test_amd_isa_wmma.py`, I0 harness GPU pass, no NaNs. |
| S1. Generated hand-shape | The generated route, not a handwritten `Ops.INS` emitter, expresses `build_gemm_lds2`-class staging: cooperative global b128 loads, LDS stores, barriers, LDS fragment loads, WMMA. | Final stream contains the expected `global_load_b128`, `ds_store_b128`, `s_barrier`, `ds_load_b128`, `v_wmma` sequence. |
| S2. Both operands staged | A and B fragments both use LDS on the promoted path. A-only staging is diagnostic and does not count. | Final 4x4 route has no promoted-path global fragment read directly into WMMA operand regs. |
| S3. Wide LDS only | Promoted LDS fragment path uses `ds_store_b128` and `ds_load_b128`; scalar per-half LDS staging is absent except fallback/debug paths. | `ds_store_b32`/`ds_store_b16` per-fragment staging count is zero on the promoted path. |
| S4. Two-slot DBUF | `PREFILL_DBUF=1` exposes two LDS slots and a prologue/body/tail or equivalent modulo cadence. | Probe shows alternating slot addresses or equivalent slot identity, with next-slot memory work before current-slot compute completes. |
| S5. Waitcnt is useful | Targeted waitcnt leaves future memory work outstanding across current WMMAs while still draining before the exact consumer/barrier. | Final stream has non-full `vmcnt(n)`/`lgkmcnt(n)` waits around the LDS pipeline and no unnecessary full drain between prefetch and compute. |
| S6. Full 4x4 compiles | Route-shaped `m_up=2` 4x4 with LDS DBUF compiles without spills or verifier failures. | No `Inc 0: no spills`, no `PTRCAT` verifier failure, no regalloc `IndexError`. |
| S7. GPU correctness | Generated LDS DBUF 4x4 is numerically correct on GPU. | I0/custom GPU harness passes: no NaNs, accepted RMSE envelope. |
| S8. Performance win | Same-clock generated path beats the current native-ISA table-local band and moves toward the hand trace. | Schedule-table gate improves over the documented 33.29/36.74 TFLOPS band for the checked shapes. |
| S9. Default/promotion policy | Fast path is behind a clear flag until S0-S8 pass; after passing, default promotion is documented and rollback flag remains. | Env flags documented; tests cover default-off/default-on behavior as appropriate. |

Minimum structural target for the final 4x4 route:

```text
prologue:
  global_load_b128 A/B for slot 0
  wait vm
  ds_store_b128 A/B slot 0
  wait lgkm
  s_barrier

loop:
  global_load_b128 A/B for next slot
  ds_load_b128 A/B current slot
  wait lgkm for current-slot ds_loads
  v_wmma current slot
  wait vm only before storing next-slot globals into LDS
  ds_store_b128 A/B next slot
  barrier/slot-safety edge before next-slot read

tail:
  consume final staged slot
  epilogue stores
```

Current completion state against this definition:

| Gate | Status |
|---|---|
| S0 | Done |
| S1 | Partial: LDS read/write ISA primitives exist, but generated cooperative route shape is blocked |
| S2 | Blocked by route-shaped cooperative B/A staging |
| S3 | Partial: wide LDS lowerers exist; codegen still emits scalar staging for current local path |
| S4 | Blocked |
| S5 | Blocked until S4 exposes overlap |
| S6 | Blocked by spills on the correctly-expanded cooperative LDS 4x4 graph |
| S7 | Pending |
| S8 | Pending |
| S9 | Pending |

## Implementation tasks

### I0. Structural introspection gate

Add a small script or unit helper that builds route-shaped 4x4 native-ISA final streams under combinations of:

```text
PREFILL_DBUF=0/1
AMD_ISA_WAITCNT_TARGETED=0/1
AMD_ISA_WMMA_B128_FRAG=1
```

It should report instruction counts, label/branch regions, b128-to-WMMA ordering, waitcnt immediates, and fragment VGPR
spans. The probe now stores these lean-DBUF acceptance fields from
`extra/qk/prefill/native_isa_l4_stream_probe.py`:

- tracked counts for `global_load_b128`, `ds_store_b128`, `ds_store_b32`, `ds_store_b16`, `ds_load_b128`, `v_wmma`,
  `s_barrier`, and `s_waitcnt`
- `b128_overlap.global_work_overlap` (any staged GMEM/LDS b128 work anywhere) and
  `b128_overlap.global_work_between_overlap` (future-load/compute overlap, i.e. any such work strictly between two
  adjacent `v_wmma` instructions)
- per-region snapshots for `global_load_b128`, `ds_store_*`, `ds_load_b128`, and barriers around each WMMA window
- `waitcnt` and `waitcnt_summary` (`count`, `vmcnt`/`lgkmcnt` sequences, targeted-vs-full split)

Acceptance check for P0/P6 structural baseline:

- P0 (baseline): `global_load_b128=16`, `v_wmma=16`, `ds_store_b128=0`, `ds_store_b32=0`, `ds_store_b16=0`,
  `ds_load_b128=0`.
- P6 candidate (lean LDS DBUF probe shape): `global_work_between_overlap=true`, `ds_store_b128>0`, `ds_store_b16==0`,
  `ds_store_b32==0`.

Recommended current probes:

```bash
AMD_ISA_WMMA_B128_FRAG=1 PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2
AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_WAITCNT_TARGETED=1 PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2
AMD_ISA_WMMA_B128_FRAG=1 PREFILL_DBUF=1 AMD_ISA_WAITCNT_TARGETED=1 PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 1
AMD_ISA_WMMA_B128_FRAG=1 PREFILL_DBUF=1 AMD_ISA_WAITCNT_TARGETED=1 PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2
```

The last command is expected to fail on the current blocker (`Inc 0: no spills`) and provides a required regression baseline for
the same compile path.

### I1. DBUF compile unblock

Owner files:

- `tinygrad/renderer/isa/amd.py::isel_wmma`
- possibly `tinygrad/codegen/opt/postrange.py::_prefill_dbuf_peel`

Acceptance:

- `PREFILL_DBUF=1` route-shaped 4x4 native-ISA lowers without `NotImplementedError`.
- GPU correctness passes.
- No default-path behavior change when `PREFILL_DBUF=0`.

### I2. Fragment phase model decision

Owner files:

- `tinygrad/renderer/isa/amd.py::_n_ab_frags`
- `tinygrad/renderer/isa/amd.py::_ab_base`
- `tinygrad/renderer/isa/amd.py::_pack_frag`
- `tinygrad/renderer/isa/amd.py::isel_wmma`

Decision output:

- reject full phase banks for 4x4 if static budget cannot fit,
- or implement phase keys if measured VGPR use is safe,
- or choose partial/streamed phase design.

### I2b. LDS DBUF primitive split

Owner files:

- `tinygrad/renderer/isa/amd.py::_frag_b128_loads`
- `tinygrad/renderer/isa/amd.py::lower_inst`
- `tinygrad/codegen/opt/postrange.py::_tc_local_stage_*`

Current result:

- DONE: `GLOBAL_LOAD_B128` route-shaped fragments and `DS_LOAD_B128` LDS fragments lower structurally and assemble.
- DONE: full 4x4 A-local staged route compiles with `ds_load_b128=8`, proving the read-side fragment primitive.
- DONE: `DS_STORE_B128` lowerer and narrow selector exist for packed contiguous `int32.vec(4)` values; tests prove
  `offset0` survives lowering.
- PARTIAL: route-shaped cooperative LDS staging across `AxisType.GLOBAL` preserves 4x4 WMMA expansion before lowering.
  Full lowering still needs the local-pointer/vector-load fold fixed without increasing pressure.
- BLOCKED: current local staging uses scalar LDS stores and does not express cooperative packed global-load -> LDS
  `ds_store_b128`.
- BLOCKED: full route-shaped `m_up=2` LDS staging reaches regalloc and spills in the correctly-expanded variants,
  including A-only, B-only, both-side, and `PREFILL_DBUF=1` variants. The next implementation must reduce live VGPR
  pressure, primarily by replacing scalar global-to-LDS staging with packed/wide staging or by changing placement so the
  whole 4x4 staged tile is not live at once.

Acceptance:

- route-shaped 4x4 emits `ds_store_b128` and `ds_load_b128` for staged fragments,
- two LDS slots are visible under `PREFILL_DBUF=1`,
- no scalar per-half LDS staging remains on the promoted route,
- targeted waitcnt produces useful `lgkmcnt(n)`/`vmcnt(n)` gaps around WMMA,
- GPU correctness passes and same-clock TFLOPS beats the current table-local band.

### I2c. Lean LDS DBUF machine-code substrate

This is the missing substrate for making generated code choose the hand-asm-style LDS double buffer instead of the
generic scalar/local staging path.

The target lowering is:

```text
global_load_b128 tempA/tempB
ds_store_b128 tempA/tempB -> LDS slot[next]
kill tempA/tempB
s_barrier / slot-safety edge
ds_load_b128 A/B operand regs <- LDS slot[cur]
v_wmma
```

The important property is not just the instruction names. The global-load temporaries must die immediately after the
`ds_store_b128`, and the WMMA operand registers must be loaded from LDS only near their consumer. Otherwise LDS staging
still increases live VGPR pressure and loses the point of the hand kernel.

#### Substrate layers

| Layer | Requirement | Current state | Completion gate |
|---|---|---|---|
| M0. ISA lowerers | Emit `global_load_b128`, `ds_store_b128`, `ds_load_b128`, `v_wmma`. | Mostly present; lowerers and narrow tests exist. | Unit tests prove offsets, register spans, and instruction mnemonics. |
| M1. Packed staging value | Represent a packed 16-byte fragment as four contiguous VGPRs suitable for `ds_store_b128`. | Partial: renderer can select `DS_STORE_B128` only for fixed contiguous `int32.vec(4)` carriers. | Generated global fragment temp feeds `ds_store_b128` without scalar `ds_store_b16/b32`. |
| M2. Local pointer/vector fold | Lower vectorized LDS addresses without invalid `PTRCAT` and without expanding to pressure-heavy stack loads. | Blocked/fragile. | Route-shaped cooperative staging passes verifier with no local-pointer `PTRCAT` failure. |
| M3. Lifetime boundary | Force `global_load_b128 -> ds_store_b128` temps to die before `ds_load_b128 -> WMMA` operand regs become live. | Missing. | Regalloc sees temps and WMMA operands in disjoint live ranges; full 4x4 does not spill. |
| M4. Two-slot identity | Encode slot 0/1 LDS addresses from the peeled K phase, not as duplicated register banks. | Partial in cooperative placeholder math. | Probe shows alternating LDS slot addresses under `PREFILL_DBUF=1`. |
| M5. Barrier/wait edge | Preserve LDS store/load ordering while allowing global prefetch overlap. | Basic barriers exist; hand-like cadence missing. | Final stream has no full drain between future global load and current WMMA except required slot-safety edge. |
| M6. Selector/cost rule | Choose lean LDS DBUF when register DBUF spills and scalar LDS staging is too heavy. | Missing. | Default-off flag reliably selects lean LDS path; non-selected baseline unchanged. |
| M7. Promotion gate | Prove correctness and speed before default. | Pending. | GPU pass, no NaNs, same-clock TFLOPS beats current native table-local band. |

#### Implementation work packages

| ID | Work package | Files | Acceptance |
|---|---|---|---|
| P0 | Freeze baseline and failing probes | `native_isa_l4_stream_probe.py`, `test_amd_isa_wmma.py` | Baseline 4x4 stays `global_load_b128=16`, `v_wmma=16`; failing LDS route is captured as an expected diagnostic. |
| P1 | Add a generated packed-fragment staging UOp shape | `postrange.py`, possibly renderer pre-match | A/B fragment staging produces a packed temp carrier, not 16 scalar half stores. |
| P2 | Lower packed temp to LDS with `DS_STORE_B128` | `amd.py::isel_store`, `lower_inst` | Final stream has `ds_store_b128` for staged fragments and zero scalar per-fragment LDS stores. |
| P3 | Fix local vector-address lowering | `devectorizer.py`, spec tests | Cooperative route passes verifier; no malformed `PTRCAT`; no vector `LOAD` from raw pointer stack survives. |
| P4 | Add explicit lifetime split | `postrange.py` or AMD pre-regalloc matcher | Global-load temps are consumed by LDS stores before WMMA operand `ds_load_b128`s are introduced. |
| P5 | Add slot-2 DBUF cadence | `postrange.py::_prefill_dbuf_peel`, cooperative staging | `PREFILL_DBUF=1` shows slot alternation and does not duplicate full A/B VGPR banks. |
| P6 | Integrate targeted waitcnt with LDS cadence | `amd.py::_insert_waitcnt`, probe assertions | `lgkmcnt` drains only before consuming current LDS loads; future global loads remain outstanding where safe. |
| P7 | GPU/perf gate | harness/table gate | Correctness pass and measured win over current native table-local band. |

#### Negative tests

These are required to keep the selector honest:

- Full register DBUF for 4x4 remains rejected or diagnostic-only; it must not be selected as the hand-like route.
- Scalar LDS staging remains fallback/debug-only; promoted path must not contain per-fragment `ds_store_b16` or scalarized
  `global_load_u16`.
- A-only or B-only staging does not count as complete; both operands must use LDS on the promoted path.
- A verifier bypass does not count as progress. The path must pass `SPEC=1`.
- A compile that spills does not count as progress, even if the graph shape is hand-like.

#### Exhaustive stop conditions for I2c

- **Solved:** final stream has packed global-to-LDS staging, two LDS slots, wide LDS fragment reads, no spills, GPU
  correctness, and a measured TFLOPS win.
- **Codegen blocked:** packed staging and slot cadence are expressible but generic UOp lowering cannot represent the
  needed lifetime split without invalid pointer/vector forms.
- **Register blocked:** packed staging is correct but still spills after lifetimes are minimized; then the 4x4 generated
  LDS route needs a smaller tile or an explicit final-ISA software-pipeline pass.
- **Scheduler blocked:** low-pressure code compiles, but list scheduling cannot form prologue/body/tail; then escalate to
  a gated final-ISA software-pipeline pass.

### I3. Scheduler/waitcnt integration

Owner files:

- `tinygrad/renderer/isa/amd.py::_schedule`
- `tinygrad/renderer/isa/amd.py::_insert_waitcnt`

Acceptance:

- final stream shows future-phase loads issued before current-phase WMMAs,
- waitcnt before each WMMA waits only for the consumed phase,
- no full drain between future load issue and current compute unless a barrier/store requires it.

### I4. Performance gates

Run in order:

```bash
PYTHONPATH=. python3 -m pytest test/unit/test_amd_isa_wmma.py -q
AMD_ISA_WMMA_B128_FRAG=1 PYTHONPATH=. python3 extra/qk/prefill/gen4x4_i0_harness.py --gpu
AMD_ISA_WAITCNT_TARGETED=1 AMD_ISA_WMMA_B128_FRAG=1 PYTHONPATH=. python3 extra/qk/prefill/gen4x4_i0_harness.py --gpu
AMD_ISA_WAITCNT_TARGETED=1 PYTHONPATH=. python3 -m extra.qk.prefill_v2_schedule_table_gate --run-amd --pin-clock --compact --no-artifact
```

Promotion threshold:

- correctness: no NaNs, route-shaped custom/kernel probes pass,
- structure: future loads overlap current WMMAs in disasm/final stream,
- performance: same-clock TFLOPS beats current table-local band and moves toward the hand trace class.

## Stop conditions

- **Solved:** generated native-ISA route has a proven load/compute overlap cadence and improves same-clock TFLOPS.
- **Design blocked:** all feasible phase-bank/stream/LDS options exceed VGPR or correctness constraints.
- **Escalate:** if UOp/codegen shapes compile and are correct but cannot express the needed prologue/body/tail, add a
  gated final-ISA software-pipeline pass only after a focused review.
