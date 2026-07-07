# Generated Machine-Code LDS DBUF 100% Scope

Date: 2026-07-07.

## Objective

Generate, through codegen, the lean native-ISA machine-code route for 4x4 WMMA prefill:

```text
global_load_b128 -> ds_store_b128 -> LDS slot
LDS slot -> ds_load_b128 -> v_wmma
```

with LDS double buffering, no handwritten assembly path, no spills, GPU correctness, and a measured performance win.

This scope uses the existing working machine-code pattern only as a reference target. The deliverable is generated code.

## Current State

| Area | Status |
|---|---|
| Baseline generated 4x4 | Correct; emits 16 WMMAs and no NaNs. |
| Current generated schedule | Still mostly load-all then compute-all; not hand-class overlap. |
| `global_load_b128` emission | Available. |
| `ds_load_b128` emission | Available. |
| `ds_store_b128` emission | Available and broader after renderer substrate work. |
| Probe visibility | Available; probe can track wide/scalar LDS stores, LDS loads, WMMA windows, barriers, waitcnt. |
| Packed global-to-LDS staging | Missing. |
| Safe local vector pointer lowering | Blocked/fragile. |
| Lifetime split | Missing. |
| Two-slot LDS DBUF cadence | Missing for full 4x4. |
| Full 4x4 LDS DBUF compile | Blocked by verifier/regalloc pressure. |

## 100% Definition

This work is complete only when all gates pass.

| Gate | Required outcome | Evidence |
|---|---|---|
| G0. Baseline protected | Existing generated native-ISA 4x4 route remains correct. | Unit tests pass; baseline probe remains `v_wmma=16`, no NaNs. |
| G1. Packed staging shape | Codegen represents global fragments as packed 16-byte temps. | Generated route has `global_load_b128` feeding LDS staging; no scalar fragment staging. |
| G2. Wide LDS stores | Promoted route stores fragments to LDS with `ds_store_b128`. | Probe: `ds_store_b128 > 0`, scalar per-fragment `ds_store_b16 == 0`, `ds_store_b32 == 0` except unrelated stores. |
| G3. Wide LDS loads | WMMA operands are loaded from LDS with `ds_load_b128`. | Probe: `ds_load_b128 > 0`; WMMA operand regs are fed from LDS loads. |
| G4. Both operands staged | A and B both use LDS on promoted path. | No promoted-path direct global fragment operand into WMMA for either A or B. |
| G5. Two-slot LDS DBUF | Slot 0/1 identity exists under DBUF. | Probe shows alternating LDS slot addresses or equivalent slot identity with `PREFILL_DBUF=1`. |
| G6. Overlap exists | Future memory/staging work occurs between current WMMAs. | Probe: `global_work_between_overlap=true` and/or staged next-slot work appears between adjacent WMMA groups. |
| G7. Waitcnt is correct | Waits drain only the needed memory class before consumers. | No unnecessary full drain between prefetch and compute; required `vmcnt`/`lgkmcnt` waits remain. |
| G8. Verifier clean | UOp lowering is valid. | `SPEC=1` path has no bad `PTRCAT`, vector LDS pointer, or malformed load/store verifier failures. |
| G9. No spills | Full 4x4 compiles in native ISA. | No `Inc 0: no spills`, no regalloc `IndexError`. |
| G10. GPU correctness | Generated route computes correctly. | GPU harness passes: no NaNs, RMSE within accepted envelope. |
| G11. Performance win | Generated route beats current generated native baseline. | Same-clock TFLOPS improves over documented table-local band. |
| G12. Rollout policy | Safe flag/default behavior. | Route remains behind an explicit flag until G0-G11 pass; rollback path documented. |

## Minimum Machine-Code Shape

The final generated route should look like this at the instruction-shape level:

```text
prologue:
  global_load_b128 A/B -> temp regs
  wait vm only as needed before LDS store consumes temps
  ds_store_b128 temp regs -> LDS slot 0
  wait lgkm as needed before slot visibility
  s_barrier

loop:
  global_load_b128 A/B -> temp regs for next slot
  ds_load_b128 A/B <- LDS current slot
  wait lgkm for current ds_load consumers
  v_wmma current slot
  wait vm only before storing next-slot global temps
  ds_store_b128 temp regs -> LDS next slot
  slot-safety barrier/edge
  swap slot

tail:
  consume final LDS slot
  epilogue stores
```

Critical lifetime rule:

```text
global-load temp live range ends at ds_store_b128
WMMA operand regs become live only at ds_load_b128 near v_wmma
```

If these live ranges overlap broadly, the route is not lean and will likely spill.

## Blocker Taxonomy

| Blocker | Meaning | Current status |
|---|---|---|
| B1. Packed staging missing | Codegen does not yet create a packed fragment carrier suitable for `ds_store_b128`. | Open. |
| B2. Scalar LDS fallback | Current staging can devolve into scalar `global_load_u16`/`ds_store_b16`/`ds_store_b32`. | Open. |
| B3. Local vector pointer lowering | Cooperative route can create invalid or pressure-heavy `PTRCAT` / vector pointer forms. | Open. |
| B4. Lifetime pressure | Temps and operand regs overlap too much, leading to no-spill failure. | Open. |
| B5. Slot cadence | Two-slot LDS address identity is not fully generated for 4x4. | Open. |
| B6. Scheduler/waitcnt | Waitcnt can only help after the overlap shape exists. | Deferred until B1-B5. |
| B7. Correctness/perf | GPU correctness and TFLOPS proof require successful compile first. | Pending. |

## Work Packages

| ID | Owner files | Task | Acceptance |
|---|---|---|---|
| P0. Probe baseline | `extra/qk/prefill/native_isa_l4_stream_probe.py` | Keep structural probe reporting wide/scalar LDS stores, LDS loads, barriers, waitcnt, WMMA windows, and overlap. | Probe distinguishes baseline, scalar LDS, and lean LDS DBUF. |
| P1. Packed staging shape | `tinygrad/codegen/opt/postrange.py` | Generate packed A/B fragment staging for cooperative LDS route. | Graph has packed temp carrier; 4x4 WMMA expansion preserved. |
| P2. Wide LDS store select | `tinygrad/renderer/isa/amd.py` | Lower packed fragment temp to `DS_STORE_B128` fail-closed. | Focused tests pass; promoted route can emit `ds_store_b128`. |
| P3. Local pointer lowering | `tinygrad/codegen/late/devectorizer.py` | Fix invalid local vector pointer lowering without increasing pressure. | `SPEC=1` cooperative route passes verifier. |
| P4. Lifetime split | `postrange.py` or AMD pre-regalloc matcher | Ensure global temp regs die at LDS store and WMMA operand regs start at LDS load. | Full 4x4 gets past regalloc without spills. |
| P5. Two-slot DBUF | `postrange.py`, DBUF peel/staging logic | Generate slot 0/1 LDS addresses from DBUF phase. | Probe shows slot alternation with `PREFILL_DBUF=1`. |
| P6. Waitcnt/scheduler | `tinygrad/renderer/isa/amd.py` | Preserve correctness while allowing useful overlap. | Targeted waits do not full-drain future work unnecessarily. |
| P7. GPU/perf gate | harness/table scripts | Prove correctness and speed. | No NaNs; TFLOPS improves. |
| P8. Default policy | docs/tests | Keep default safe until proof. | Flagged route and rollback documented. |

## Required Probe Matrix

Run these after each meaningful implementation change:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0

AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_WAITCNT_TARGETED=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0

AMD_ISA_WMMA_B128_FRAG=1 PREFILL_DBUF=1 AMD_ISA_WAITCNT_TARGETED=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 1 --indent 0

AMD_ISA_WMMA_B128_FRAG=1 PREFILL_DBUF=1 AMD_ISA_WAITCNT_TARGETED=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Expected current state:

| Case | Expected current result |
|---|---|
| Baseline `m_up=2` | Compiles; `v_wmma=16`; `global_load_b128=16`; no LDS DBUF; no overlap. |
| Targeted waitcnt baseline | Compiles; wait profile changes; still no LDS DBUF overlap. |
| DBUF `m_up=1` | Compiles and can show global work between WMMAs, but not final LDS route. |
| DBUF `m_up=2` | Currently fails with no-spill pressure. |

## Negative Tests

These outcomes do not count as completion:

- Falling back to handwritten assembly.
- Full register double buffering for 4x4.
- A-only or B-only LDS staging.
- Scalar LDS staging with many `ds_store_b16`/`ds_store_b32`.
- Passing only with `SPEC=0`.
- Compiling only by spilling.
- Emitting wide LDS loads but scalar LDS stores.
- Producing overlap only for `m_up=1`, not full 4x4.

## Current P1 Finding

The first attempted packed-staging implementation placed explicit `AMDOps.V_PACK` `INS` nodes in
`postrange.py::_tc_local_stage_coop_operand`. This is too early in the pipeline.

Observed failure:

```text
UOp verification failed on Ops.UNROLL dtypes.half
src: Ops.STACK dtypes.half.vec(4)
arg: ((10, 4),)
parent: AMDOps.V_PACK
```

Meaning:

- The expander sees the `V_PACK` inputs under route UPCAST axes.
- It wraps the half inputs as `UNROLL(STACK(...))`.
- That shape fails verifier before the AMD renderer can lower the packed carrier.

Conclusion:

```text
packed staging must not be expressed as early postrange AMDOps.V_PACK over unexpanded half lanes
```

Next viable P1 options:

1. Add a renderer/pre-regalloc matcher that recognizes the LDS store of contiguous half lanes after expansion and lowers
   it directly to `DS_STORE_B128`.
2. Add a target-neutral packed-fragment carrier UOp that survives expansion cleanly and is only converted to `V_PACK` in
   AMD isel.
3. Generate the packed store from already-expanded scalar lanes after the expander, not inside postrange.

The local pointer/readback blockers are improved by:

- folding gated `STORE(GEP(local_ptr), value, gate)`;
- avoiding local pointer `PTRCAT` formation for the cooperative LDS staging path;
- scalarizing `LOAD(STACK(local_ptrs))`;
- using pointer dtype size instead of shape inference for effect-dependent local pointer casts.

Default/focused tests pass, and A-only cooperative LDS staging is now a valid scalar-store control. P1 remains open until
packed staging is moved to a valid pipeline layer.

## Parallel P1 Experiment Plan

Run all three P1 options as mutually-exclusive experiments. They should never be enabled together on the promoted path.

| Branch | Flag | Implementation layer | Hypothesis | First pass gate |
|---|---|---|---|---|
| E1. AMD late matcher | `PREFILL_LDS_PACK_LATE_MATCHER=1` | AMD renderer/pre-regalloc or final matcher | The expanded LDS store pattern is recognizable late enough to lower directly to `DS_STORE_B128`. | SPEC-clean probe with `ds_store_b128>0`. |
| E2. Neutral carrier | `PREFILL_LDS_PACK_CARRIER=1` | UOp/codegen IR before AMD isel | A packed-fragment carrier can survive expansion without invalid `UNROLL(STACK)`. | Graph expands cleanly and AMD isel receives one packed carrier. |
| E3. Post-expander pack | `PREFILL_LDS_PACK_POST_EXPAND=1` | After expander, before devectorizer/regalloc | Packing after UPCAST expansion avoids the early `V_PACK` verifier failure. | SPEC-clean probe with no malformed local pointer/vector nodes. |

Current experiment results, 2026-07-07:

| Probe | Result | Meaning |
|---|---|---|
| A-only scalar control: `PREFILL_TC_LOCAL_STAGE=a PREFILL_TC_LOCAL_STAGE_COOP_POST=1 PREFILL_TC_LOCAL_STAGE_COOP_GLOBAL=1` | `ok=true`; `ds_store_b16=64`, `ds_load_b128=8`, `ds_store_b128=0`, `wmma=16`; WMMA `src0=ds_load_b128`, `src1=global_load_b128`. | The generated LDS staging substrate is now verifier-clean, but it is scalar-store and therefore not a completion candidate. |
| E1 flag: add `PREFILL_LDS_PACK_LATE_MATCHER=1` | Same as scalar control: `ds_store_b128=0`. | The flag is scaffolded only; no late AMD matcher has been implemented yet. |
| E2 flag: add `PREFILL_LDS_PACK_CARRIER=1` | Same as scalar control: `ds_store_b128=0`. | The flag is scaffolded only; no neutral packed carrier has been implemented yet. |
| E3 flag: add `PREFILL_LDS_PACK_POST_EXPAND=1` | Fails verifier on `Ops.UNROLL dtypes.half` over `Ops.STACK dtypes.half.vec(4)`. | The current E3 implementation is still the early `V_PACK` diagnostic, not a true post-expander pass. |
| Both operands: `PREFILL_TC_LOCAL_STAGE=both` | Fails `NotImplementedError: Inc 0: no spills`. | Staging both A and B exceeds the current no-spill register budget before packed stores/lifetime fixes. |

Comparison gates, in order:

1. `SPEC=1` verifier clean.
2. `ds_store_b128 > 0`.
3. Scalar fragment staging absent: no promoted-path `ds_store_b16`; no per-fragment scalar `ds_store_b32`.
4. Full 4x4 `m_up=2` compiles with no spills.
5. GPU correctness.
6. Same-clock TFLOPS win.

Promotion rule:

```text
only one P1 branch can become the default candidate
```

The first branch to pass gates 1-4 becomes the integration candidate for P4/P5 lifetime and slot work. The other branches
remain diagnostic fallbacks until the promoted candidate also passes GPU/perf.

## Stop Conditions

| Stop condition | Meaning |
|---|---|
| Solved | G0-G12 all pass. |
| Codegen blocked | Packed staging and slot cadence are conceptually expressible, but UOp lowering cannot represent the required pointer/vector/lifetime shape cleanly. |
| Register blocked | Lean packed staging is verifier-clean but still spills after lifetime minimization. |
| Scheduler blocked | Low-pressure code compiles, but list scheduling cannot form prologue/body/tail overlap. Escalate to a gated final-ISA software-pipeline pass. |
| Perf blocked | Correct generated LDS DBUF route compiles and passes GPU but does not beat baseline. Revisit tile shape/cost model. |

## Agent Split

Use disjoint ownership to avoid conflicts:

| Agent | Scope |
|---|---|
| P1 agent | `postrange.py`: packed staging shape and slot identity. |
| P2 agent | `amd.py` + tests: renderer `ds_store_b128` substrate. |
| P3 agent | `devectorizer.py` + tests: local pointer/vector lowering. |
| P0/P6 agent | probe + docs: acceptance metrics and waitcnt/overlap visibility. |

Integration order:

1. P0 probe visibility.
2. P2 renderer substrate.
3. P1 packed staging.
4. P3 verifier cleanup.
5. P4/P5 lifetime and DBUF slots.
6. P6 waitcnt.
7. P7 correctness/perf.
