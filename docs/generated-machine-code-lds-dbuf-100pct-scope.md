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
| Packed global-to-LDS staging | A-side packed `ds_store_b128` route available structurally behind `PREFILL_TC_LOCAL_STAGE_COOP_POST=1 PREFILL_LDS_PACK_LATE_MATCHER=1`; direct `global_load_b128 -> ds_store_b128` remains opt-in diagnostic behind `PREFILL_LDS_PACK_GLOBAL_B128=1`. |
| Safe local vector pointer lowering | Fixed for the B tile-key diagnostic path by disabling local pointer grouping under `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`; still flag-gated. |
| Lifetime split | Missing. |
| Two-slot LDS DBUF cadence | Missing for full 4x4. |
| Full 4x4 A-side packed LDS compile | Passes behind `PREFILL_LDS_PACK_LATE_MATCHER=1`; DBUF overlap remains missing. |

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
| B1. Packed staging missing | Codegen does not yet create a packed fragment carrier suitable for `ds_store_b128`. | A-side E1 packed LDS store prototype solved structurally by proof-based late matcher; not numerically valid in the centralized route-bound gate yet. |
| B2. Scalar LDS fallback | Current staging can devolve into scalar `global_load_u16`/`ds_store_b16`/`ds_store_b32`. | The existing centralized passing route is `PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1`, which still uses scalar LDS stores. The packed cooperative route removes scalar LDS stores structurally but fails central numeric correctness. |
| B3. Local vector pointer lowering | Cooperative route can create invalid or pressure-heavy `PTRCAT` / vector pointer forms. | Fixed for the B tile-key path; keep guarded until the promoted path is chosen. |
| B4. Lifetime pressure | Temps and operand regs overlap too much, leading to no-spill failure. | Current blocker for B/both tile-key structural native probes. |
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
| Centralized route-bound gate, existing path: `PREFILL_TC_LOCAL_STAGE=a PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1` | `PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_PASS`; finite; `max_abs_vs_ref=0.03130340576171875`; `rel_rmse_vs_ref=0.0002076508681057021`; shared local and barrier present. | This is the authoritative correctness harness for route-bound A-local staging. Use it instead of creating a new harness. |
| A-only cooperative scalar control: `PREFILL_TC_LOCAL_STAGE=a PREFILL_TC_LOCAL_STAGE_COOP_POST=1 PREFILL_TC_LOCAL_STAGE_COOP_GLOBAL=1` | Structural probe `ok=true`; `ds_store_b16=64`, `ds_load_b128=8`, `ds_store_b128=0`, `wmma=16`; WMMA `src0=ds_load_b128`, `src1=global_load_b128`. Central route-bound gate is non-finite. | The cooperative generated LDS staging substrate is verifier-clean but numerically wrong in the route-bound central gate. It cannot be the correctness base until its LDS layout/read contract is fixed. |
| E1 flag on cooperative route: add `PREFILL_LDS_PACK_LATE_MATCHER=1` | Structural probe `ok=true`; `ds_store_b128=8`, `ds_store_b16=0`, `ds_store_b32=0`, `ds_load_b128=8`, `global_load_b128=8`, `global_load_u16=64`, `wmma=16`; WMMA `src0=ds_load_b128`, `src1=global_load_b128`. Central route-bound gate is non-finite. | Proof-based pre-isel matcher over the cooperative scalar LDS graph emits A-side packed LDS stores using scalar global half loads plus `v_pack`. The instruction shape is useful, but correctness is blocked by the underlying cooperative A-local layout/read mapping. |
| E2 flag: add `PREFILL_LDS_PACK_CARRIER=1` | Fails verifier on `Ops.UNROLL dtypes.half` over `Ops.STACK dtypes.half.vec(4)`. | A postrange carrier is still too early; the carrier must be inserted after expansion or replaced by an E1 pre-isel rewrite. |
| E3 flag: add `PREFILL_LDS_PACK_POST_EXPAND=1` | Fails verifier on `Ops.UNROLL dtypes.half` over `Ops.STACK dtypes.half.vec(4)`. | The current E3 implementation is still the early `V_PACK` diagnostic, not a true post-expander pass. |
| Both operands: `PREFILL_TC_LOCAL_STAGE=both` | Fails `NotImplementedError: Inc 0: no spills`. | Staging both A and B exceeds the current no-spill register budget before packed stores/lifetime fixes. |
| A full-lane LDS layout: add `PREFILL_TC_LOCAL_STAGE_A_FULL_LANE=1` | Structural packed route compiles, but the central route-bound gate remains non-finite. | Full-lane A layout is diagnostic-only. It does not fix the cooperative route-bound correctness failure. |

Follow-up attempt, 2026-07-07:

| Attempt | Result | Decision |
|---|---|---|
| E2 half8 carrier in `postrange.py`, lowered in AMD isel with four late `V_PACK`s plus gated `ds_store_b128`. | Still fails verifier before isel: `Ops.UNROLL dtypes.half` over `Ops.STACK dtypes.half.vec(4)`. | This proves a carrier built in postrange from `src.gep(...)` is still too early. E2 must be inserted after expansion, or E1 must rewrite the already verifier-clean scalar LDS store graph before instruction selection. |
| E1 naive adjacency matcher. | Rejected before implementation. The verifier-clean graph is nested as groups of scalar stores whose apparent LDS address identity is not enough to prove one contiguous 16-byte row. | E1 needs a proof-based group matcher keyed by target LDS base plus constant slot offsets and source lane order, not a linear adjacency pass. |
| E1 proof-based group matcher. | Passes structural compile gate for A-side 4x4: 8 `ds_store_b128`, no scalar LDS stores, no spills. The centralized route-bound gate still fails because the cooperative A-local route fails even before E1. | Keep E1 as a structural emission candidate, but promote only after the cooperative A-local layout/read mapping passes the central gate. |
| E1 direct global-b128 producer: add `PREFILL_LDS_PACK_GLOBAL_B128=1`. | Passes structural compile gate for A-side 4x4: `global_load_b128=16`, `global_load_u16=0`, `ds_store_b128=8`, but the centralized route-bound gate is non-finite. Scheduler-off and conservative-waitcnt runs remain non-finite. | This reaches the desired producer/store instruction shape but its value/layout proof is not correct yet. It is opt-in diagnostic, not the default route. |

## P1-WithLocal Scope

The next integration target is the centralized passing route, not the cooperative route:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE=a \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage a --compact
```

Current structural shape for that route:

| Signal | Current value | Target |
|---|---:|---:|
| `ds_load_b128` | 8 | 8+ |
| `ds_store_b128` | 0 | >0 |
| `ds_store_b32` | 16 | 0 for the promoted A-local fragment stores |
| `global_load_b128` | 8 | 8+ |
| `global_load_u16` | 64 | 0 on the promoted fragment staging path |
| `s_barrier` | 1 | preserved |
| Central gate | PASS | PASS |

Pre-regalloc inspection shows the passing path's LDS writes are already compacted into 16 `AMDOps.DS_STORE` nodes with:

```text
value = Ops.NOOP dtypes.half.vec(4)
esz   = 8
```

That is a different shape from the cooperative E1 matcher, which proves and packs 16 groups of four scalar half stores.
The simple final-lowerer attempt did not fire because the `half.vec4` value is not allocated as a four-register span by
the time `lower_inst` selects `ds_store_b32`. Final address registers also no longer preserve enough symbolic adjacency
to safely combine stores after register allocation.

### 100% for P1-WithLocal

| Gate | Required outcome |
|---|---|
| W0. Central harness retained | The exact route-bound `WITH_LOCAL` command above remains finite and passes. |
| W1. No new harness | All correctness claims use `extra.qk.prefill_graph_gemm_route_bound_stage_gate`. |
| W2. Span/carrier introduced before final lowering | The `half.vec4` LDS store value is represented as an explicit packed carrier or allocated as a b128-capable contiguous span before register allocation loses the proof. |
| W3. Structural lean write | Probe shows `ds_store_b128 > 0` and the promoted fragment stores no longer appear as scalar `ds_store_b32`. |
| W4. Lean producer | The promoted path removes scalar `global_load_u16` fragment staging in favor of a packed producer or provably equivalent pack. |
| W5. Correctness | Central route-bound gate still passes after W2-W4. |
| W6. Promotion | The path stays flag-gated until W0-W5 pass together. |

### Work Items

| ID | Work | Status |
|---|---|---|
| WL0 | Preserve and rerun the passing central `WITH_LOCAL` gate. | Done; still passing. |
| WL1 | Inspect the actual passing-path LDS store carrier. | Done; `DS_STORE esz=8` with `NOOP half.vec4`. |
| WL2 | Remove post-regalloc store combining from the candidate path. | Done; symbolic adjacency is gone too late. |
| WL3 | Add a pre-regalloc packed carrier/span for the `half.vec4` LDS store value. | Done for A-side `WITH_LOCAL`: `PREFILL_LDS_PACK_WITHLOCAL_B128=1` pairs adjacent `half.vec4` local stores into one direct `global_load_b128 -> ds_store_b128` store, with a dependency chain so the b128 loads do not all stay live. |
| WL4 | Re-probe instruction counts and central correctness. | Done: native structural probe has `ds_store_b128=8`, `ds_store_b32=0`, `global_load_u16=0`, `ds_load_b128=8`, `s_barrier=1`, `wmma=16`; centralized route-bound gate passes. |

Latest P1-WithLocal attempt:

| Attempt | Result | Decision |
|---|---|---|
| Per-store `half.vec4 -> 2*v_pack -> ds_store_b64`, behind `PREFILL_LDS_PACK_WITHLOCAL_B64=1`. | Unit-level selection/lowering works. Full native 4x4 structural probe fails with `NotImplementedError: Inc 0: no spills`; `REGALLOC_DEBUG` reports peak pressure at the global-load region before the store stream can drain. | Keep as a diagnostic substrate only. It proves the hook must also change lifetime/scheduling: packed writes have to be formed in a stream that stores and kills fragment temps promptly, or paired as b128 after a proof-based store grouping pass that reduces instruction count and live ranges. |
| Paired direct b128 store group, behind `PREFILL_LDS_PACK_WITHLOCAL_B128=1`. | Passes native structural probe and central correctness for A-side `WITH_LOCAL`: `global_load_b128=16`, `global_load_u16=0`, `ds_store_b128=8`, `ds_store_b32=0`, `ds_load_b128=8`, `v_pack=0`, `wmma=16`. | This is the promoted A-side packed-store substrate. Remaining 100% work is B/both-side staging, two-slot DBUF cadence/overlap, targeted wait/perf proof, and rollout policy. |
| Same paired direct b128 store group for B-only and both operands. | Native structural probes compile: B-only has `ds_store_b128=8`, `ds_load_b128=8`; both has `ds_store_b128=16`, `ds_load_b128=16`; neither has scalar LDS stores or scalar global half loads. Central route-bound gate is numerically wrong for B-only and both (`rel_rmse` about `1.22`). | The next blocker is B-side local layout/read correctness. Pressure and packed-store emission are no longer the blocker for both-side structural compile. |
| B-side bounded tile-key prototype, behind `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`. | Confirms and fixes the scalar B failure mechanism in the centralized route: generic `WITH_LOCAL` stages B by WARP lane only, but B also varies by the output-column tile. The bounded explicit layout (`tile_slot * 256 + row * 16 + frag`) plus local pointer-grouping disable is verifier-clean in the central route, and B-only/both central correctness pass. | Keep opt-in only. The next implementation step is structural: get B-only and both tile-key native probes through no-spill pressure while preserving the packed `ds_store_b128` route. |

Current B-side diagnosis:

```text
A-side WITH_LOCAL works because its staged fragment is reused across the column subtiles covered by the current WARP-only LDS key.
B-side WITH_LOCAL does not: B changes across output-column tile identity, so WARP-only staging aliases distinct B fragments.
The missing identity is small and local to the tile, not the full problem-size GLOBAL axis.
```

Current guarded flags:

| Flag | Purpose | Current result |
|---|---|---|
| `PREFILL_LDS_PACK_WITHLOCAL_B128=1` | A-side direct `global_load_b128 -> ds_store_b128` packing. | Structural and central correctness pass for `PREFILL_TC_LOCAL_STAGE=a`. |
| `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` | Diagnostic B-side bounded tile-key LDS layout. | Central B-only and both correctness pass; native structural probe currently fails with `NotImplementedError: Inc 0: no spills`. |

Latest B tile-key result:

| Gate | Result | Meaning |
|---|---|---|
| Central B-only correctness: `PREFILL_TC_LOCAL_STAGE=b PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` | PASS; finite; `max_abs_vs_ref=0.03130340576171875`; `rel_rmse_vs_ref=0.0002076508681057021`; shared local and barrier present. | The B-side LDS layout/read contract is now correct in the centralized harness. |
| Central both-side correctness: `PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` | PASS with the same correctness envelope. | A+B staged through LDS is numerically valid under the bounded B tile key. |
| Central correctness with `PREFILL_LDS_PACK_WITHLOCAL_B128=1` also enabled | PASS for B-only and both. | The correctness route tolerates the packed-store flag, but this is not yet a native structural proof for B/both. |
| Native structural probe with B tile-key plus B gather bridge | B-only now compiles: `ds_store_b128=8`, `ds_load_b128=8`, no scalar LDS stores, WMMA `src1=ds_load_b128`; pressure peak drops to 72 live VGPRs. | B-only structural packing is unblocked by the bridge. This is not yet the primitive final route because it reverse-engineers the B group in AMD pre-isel and still uses scalar global half loads plus `v_pack`. |
| Native structural probe with both operands plus B gather bridge | PASS after scoping the devectorizer B tile-key guard to local slot `993`: `ds_store_b128=16`, `ds_load_b128=16`, no scalar LDS stores, WMMA `src0/src1=ds_load_b128`; pressure peak drops to 81 live VGPRs. | Both-side native structural staging is unblocked. The next blocker moves back to the broader DBUF goal: two-slot cadence, overlap, waitcnt/perf proof. |

Current native-pressure diagnosis:

| Probe | `REGALLOC_DEBUG` peak | Dominant live classes | Meaning |
|---|---:|---|---|
| B-only tile-key before B gather bridge | 128 live VGPRs | 68 `V_OFFSET`, 39 `MOV`, 8 `DS_LOAD_B128` | B tile-key correctness is fixed, but scalar LDS/global address temps push the native route over the no-spill budget. |
| B-only tile-key after B gather bridge | 72 live VGPRs | 39 `MOV`, 12 `V_OFFSET`, 8 `DS_LOAD_B128` | The bridge removes the B-only no-spill blocker by replacing 64 scalar LDS stores with 8 packed LDS stores and dependency-threaded packs. |
| Both-side tile-key plus B gather bridge before scoped devectorizer guard | 169 live VGPRs | 69 `V_OFFSET`, 64 scalar `GLOBAL_LOAD`, 28 `V_IADD` | The broad B tile-key guard disabled LOCAL pointer grouping for unrelated A staging and broke the A packed baseline. |
| Both-side tile-key plus B gather bridge after scoped devectorizer guard | 81 live VGPRs | 39 `MOV`, 12 `V_OFFSET`, 12 `V_IADD`, 10 `DS_LOAD_B128` | A packed staging survives with B tile-key present; both operands now stage through wide LDS without spills. |
| A-side `WITH_LOCAL` plus `PREFILL_LDS_PACK_WITHLOCAL_B128=1` | 68 live VGPRs | 39 `MOV`, 12 `V_OFFSET`, 8 `GLOBAL_LOAD_B128` | This is the target pressure profile: stream packed loads through the fixed b128 scratch span and store immediately. |

Why the existing A-side matcher does not solve B:

```text
B tile-key lowers to one parent GROUP of 16 fragment groups, each with 4 tile stores.
Each scalar store is GATED_STORE(gate, lds_offset, scalar_global_load, order, local_buf, esz).
The store order is fragment-major across column tiles.
For one tile, fragments 0..3 share one global base with byte immediates 0,2,4,6;
fragments 4..7 then use a different global base, again with immediates 0,2,4,6.
```

So a simple adjacent-store combiner is not sufficient. The next B structural matcher must either:

1. gather by `(tile, half-row)` from the 16x4 parent group and prove the multi-base global-load sequence is the exact B
   fragment row expected by WMMA before emitting packed LDS stores; or
2. move the B tile-key staging shape earlier so the generated stream is already tile-major/row-contiguous when it reaches
   instruction selection.

## Primitive B Tile-Major Scope

The primitive fix is not a scheduler or waitcnt change. The primitive is:

```text
B-side LDS staging must be generated in tile-major, b128-packable row units before scalar address/load expansion creates
long-lived per-half `GLOBAL_LOAD`, `V_OFFSET`, and `V_IADD` nodes.
```

The current B tile-key path is numerically correct but structurally wrong for native ISA:

```text
for frag in 0..15:
  for tile in 0..3:
    gated scalar global_load half
    gated scalar ds_store half
```

The target primitive shape is:

```text
for tile in column_tiles:
  for row_half in 0..1:
    global_load_b128 contiguous B row half
    gated ds_store_b128 to B LDS tile slot

barrier

for each WMMA:
  ds_load_b128 B LDS tile slot
```

### Layer Decision

| Layer | Can express the primitive? | Risk | Decision |
|---|---|---|---|
| `postrange.py::_tc_local_stage_b_src` | Yes. It still knows B is a `CONTRACT`, its fragment axes, WARP lane, and output-column tile ranges. | Must preserve the proven scalar B correctness contract while changing store construction. | Preferred primitive layer. |
| `devectorizer.py` | Partially. It can avoid bad pointer grouping but no longer has enough semantic identity to choose B tile-major row groups cleanly. | Easy to make a local pointer-shape fix that is not a B staging primitive. | Keep only the existing guard; do not put the primitive here. |
| AMD `pre_isel_matcher` | Can rewrite high-level `STORE` groups before instruction selection. | More target-specific and downstream of generic B layout construction, but still before scalar `GLOBAL_LOAD`/`V_OFFSET` blowup. | Acceptable bridge if postrange cannot express a verifier-clean carrier. |
| AMD `pre_regalloc_matcher` | Can gather already-selected `GATED_STORE` groups. | Too late for a primitive: scalar address/load nodes already exist and the proof must reverse-engineer layout from lowered code. | Tactical bridge only, not the promoted primitive. |
| Scheduler/waitcnt | No. Scheduler-off has the same register peak. | Chases a disproven mechanism. | Out of scope until the packed primitive compiles. |

### 100% Definition For The B Primitive

| Gate | Required outcome | Evidence |
|---|---|---|
| BP0. A regression protected | Existing A packed `WITH_LOCAL` path still compiles structurally. | Native probe: `local-stage=a`, `PREFILL_LDS_PACK_WITHLOCAL_B128=1`, `ok=true`, `ds_store_b128=8`, no scalar LDS stores. |
| BP1. B central correctness | B-only route remains numerically correct. | Route-bound gate PASS for `PREFILL_TC_LOCAL_STAGE=b PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`. |
| BP2. Both central correctness | A+B staged route remains numerically correct. | Route-bound gate PASS for `PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`. |
| BP3. Native B no-spill | B-only tile-key route compiles under native ISA. | Native probe `ok=true`; no `Inc 0: no spills`. |
| BP4. Native B packed stores | B-only promoted path emits wide LDS stores. | Probe has `ds_store_b128 > 0`; scalar B staging does not appear as `ds_store_b16`/`ds_store_b32`. |
| BP5. Native B packed loads | B WMMA operand comes from LDS wide loads. | Probe has `ds_load_b128 > 0`; WMMA `src1` origins are `ds_load_b128` for B-staged WMMAs. |
| BP6. Both native no-spill | Both-side tile-key route compiles under native ISA. | Native probe `ok=true`; no spill. |
| BP7. Both native packed A+B | A and B both stage through LDS with wide stores/loads. | Probe has expected higher `ds_store_b128`/`ds_load_b128` counts than A-only and no scalar B staging. |
| BP8. Pressure reduced | B-only pressure approaches the A packed path, not the scalar path. | `REGALLOC_DEBUG` peak moves away from 128/225 and no longer dominated by scalar B `GLOBAL_LOAD`/`V_OFFSET`. |
| BP9. SPEC clean | The new graph is verifier-clean. | `SPEC=1` route/probe does not fail verifier. |

### Implementation Plan

1. Add a new opt-in primitive flag, for example `PREFILL_TC_LOCAL_STAGE_B_TILEMAJOR=1`, leaving the current
   `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` scalar-correct path intact.
2. In `postrange.py::_tc_local_stage_b_src`, split the B staging construction into:
   - scalar-correct fallback: current `for i in range(16)` stores;
   - tile-major primitive path: construct two half-row packed carriers per tile slot.
3. Make the tile-major path produce a carrier that survives expansion without recreating 64 scalar global loads. If an
   early carrier trips the same `UNROLL(STACK)` verifier issue seen in P1, stop and move only the carrier creation to an
   AMD pre-isel matcher while keeping the tile-major row identity explicit in the graph.
4. Teach AMD lowering to select the B tile-major row carrier as `global_load_b128 -> gated ds_store_b128`, using the
   existing `GATED_STORE_B128`/`DS_STORE_B128` substrate.
5. Add focused tests for the B primitive:
   - B tile-major graph does not form local vector pointer `PTRCAT`;
   - B-only native structural probe compiles no-spill;
   - both-side native structural probe compiles no-spill;
   - central correctness remains passing for B and both.
6. Only after BP0-BP9 pass, continue to the next original 100% gates: two-slot DBUF cadence, overlap, targeted waitcnt,
   and measured TFLOPS.

### Bridge Result

Implemented a guarded AMD pre-isel bridge in `tinygrad/renderer/isa/amd.py`:

```text
GROUP(16 fragment groups x 4 tile stores)
  -> gather by tile and half-row
  -> half8 carrier
  -> GATED_STORE_B128
```

The bridge is deliberately fail-closed:

- requires `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` and `PREFILL_LDS_PACK_WITHLOCAL_B128=1`;
- requires exact `16 x GROUP(4 x STORE)` shape;
- requires all stores to be local half scalar stores with the same gate and local dynamic base;
- requires contiguous LDS half offsets for each gathered row;
- threads each row through the previous packed store so the scalar global loads do not all stay live.

It originally solved B-only native structural packing. Both-side staging is now also unblocked by scoping the devectorizer
LOCAL pointer-grouping disable to the B tile-key placeholder (`DEFINE_LOCAL` slot `993`) instead of all LOCAL buffers. A
direct `AFTER` dependency from B scalar values to the A staged operand was tested and rejected because it is not
verifier-clean (`Ops.AFTER dtypes.half` over a void barrier). If ordering is needed again, use a verifier-clean
buffer/effect-level carrier, not a half-value dependency.

### Review Checklist

The implementation is rejected if any of these are true:

- It only changes waitcnt/scheduler behavior.
- It requires spills.
- It proves only A-side packing.
- It passes central correctness but the native probe still has scalar B `GLOBAL_LOAD`/`GATED_STORE` staging.
- It packs B by assuming adjacency in the lowered instruction stream without proving tile and fragment identity.
- It introduces a handwritten assembly path.

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

## Path 2 Fork: A Window Identity Primitive

Date: 2026-07-08.

The current Path 2 fork is no longer blocked on basic `ds_store_b128`/`ds_load_b128` substrate. The blocker is the
combination of A-window proof and native no-spill lifetime at the larger `4x2` candidate. Renderer-only store/load
rewrites are not enough, because they can only select a packed instruction after the graph has already chosen the LDS
identity and lifetime shape.

### Agent Findings

| Agent scope | Result | Decision |
|---|---|---|
| Producer/store audit | Store-only packing is unsafe. Forcing the unsafe multidim bypass shows A stores use `lidx0 * 256 + const`, while the global producer contains `lidx1 * 262144 + ...`. | Do not patch `_pack_withlocal_lds_stores` alone. The A window key must exist before store packing. |
| Consumer/load audit | Raw object identity is unstable after store/load splitting and order deps, but structural LDS keys can line up when a real key exists. | Do not infer the key from late cloned UOps. Pass/register a logical LDS fragment key from the producer stage. |
| Trace/proof gate | Added `extra/qk/prefill/a_fragment_alias_probe.py`. Initial final-register tracking reported a `2x2` A identity failure, but the strengthened probe shows pre-isel symbolic proof is OK while final physical-register equivalence is not yet tracked. | Keep A identity as a gate, but distinguish symbolic LDS proof from final register-family heuristics. |
| Resource/model sidecar | A compressed A-window model fits `2x2` and `4x2` under the 64 KiB LDS cap, but raw A GLOBAL does not. | Keep the target compressed/bounded; do not reintroduce raw GLOBAL-sized LDS. |

### Current Local Probe

Command:

```bash
python3 extra/qk/prefill/a_fragment_alias_probe.py --indent 0
```

Current result in this checkout:

| Case | Compile | LDS bytes | A identity | Meaning |
|---|---|---:|---|---|
| `u0=2,u1=2,loc=2,unr=2` | PASS | 32768 | Pre-isel PASS: `has_bounded_lidx1=true`, `missing_load_count=0`; final physical-register heuristic FAILS. | The current post-stage path has the bounded A key symbolically. The remaining proof gap is final remat/equivalence tracking, not necessarily missing graph identity. |
| `u0=4,u1=2,loc=2,unr=2` | FAIL | n/a | n/a | Fails before final stream with `NotImplementedError: Inc 0: no spills`. |

This changes the promotion bar: a candidate cannot count as Path 2-complete merely because it compiles, fits LDS, and
emits wide LDS ops. It must also prove A producer/store windows match A WMMA load windows at the symbolic LDS-key level,
and either teach the final probe to prove equivalent rematerialized address registers or avoid rematerializing them into
untrackable families.

### Required A Address Shape

For the current `4x2/loc=2/unr=2` Path 2 target, the staged A LDS address must carry:

```text
A_LDS_ADDR =
  dbuf_slot_base
  + a_window_key          # bounded multidim/window identity
  + row_within_fragment
  + fragment_const

Required observed-good symbolic shape:
  lidx0 * 256 + lidx1 * 128 + fragment_const + dbuf_slot_base
```

The exact constants remain implementation-checked, but the invariant is not negotiable: the same bounded window key must
feed the A `ds_store_b128` producer and the A `ds_load_b128` WMMA consumer. Current `2x2` proof keys satisfy this
symbolically; `4x2` still fails before final proof because register pressure spills.

### Layer Decision

| Layer | Can solve A window identity? | Decision |
|---|---|---|
| `postrange.py` A staging | Yes. It still sees the A `CONTRACT`, WARP/LOCAL axes, and the bounded tile/window ranges before store/load identity is lost. | Preferred primitive layer. Add an A-specific post-stage source analogous to B tile-key, but with bounded A window identity. |
| AMD pre-isel store matcher | No by itself. It can pack proven stores, but cannot make existing WMMA loads read a new address family. | Keep as the wide-store selector after the key exists. |
| AMD frag-load matcher | No by itself. It can select `ds_load_b128`, but late inference sees cloned/wrapped expressions and cannot invent the producer key. | Use only as a verifier/probe surface. |
| Scheduler/waitcnt | No. Scheduler-off and conservative waits do not create missing address identity. | Deferred until A identity and no-spill are fixed. |
| Raw GLOBAL cooperative staging | Not acceptable. It can express identity but exceeds LDS or aliases when compressed incorrectly. | Reject for Path 2. |

### Path 2 A-Primitive 100% Gates

| Gate | Required outcome | Evidence |
|---|---|---|
| A0. Probe exists | A producer/consumer identity is measured directly. | `extra/qk/prefill/a_fragment_alias_probe.py` is checked in and runs. |
| A1. Baseline protected | Existing A-only packed `WITH_LOCAL` path still passes. | Native structural probe and central route-bound gate remain passing. |
| A2. B tile-key protected | Existing B-only and both central correctness stay passing. | Central route-bound gate with `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`. |
| A3. Bounded A key represented | A staging graph includes a bounded A window key, not raw problem-size GLOBAL. | `preisel_lds_proof.has_bounded_lidx1=true`; no raw `lidx1 * 262144` LDS key. |
| A4. Store/load same key | A `ds_store_b128` rows cover WMMA A `ds_load_b128` windows. | `preisel_lds_proof.ok=true`; final `a_fragment_window_identity` is advisory until remat-equivalence tracking is fixed. |
| A5. No A aliasing | No A load window is reused by unrelated WMMA A fragments in the symbolic key space. | `preisel_lds_proof.missing_load_count=0`, plus improved final probe when available. |
| A6. Final proof strengthened | Final analyzer no longer reports false missing windows when addresses are rematerialized into different physical regs. | Final probe either proves equivalent address expressions or labels the result as inconclusive instead of failing the primitive. |
| A7. Native `4x2` no-spill | `u0=4,u1=2,loc=2,unr=2` compiles under native ISA. | Alias probe or structural probe returns `ok=true`; no `Inc 0: no spills`. |
| A8. LDS cap | Candidate remains below 64 KiB with DBUF. | `group_segment_bytes < 65536`; exact 65536 remains rejected unless a separate safety proof accepts it. |
| A9. GPU correctness | Candidate is finite and within correctness envelope. | Central/GPU harness passes; no NaNs. |

### Implementation Sequence

1. Preserve the existing post-stage A symbolic key path, because `2x2` already proves bounded `lidx1` coverage before
   isel.
2. Strengthen the probe so final-register rematerialization does not look like an A identity bug. The new
   `preisel_lds_proof` field is the first step.
3. Attack the actual native blocker for `4x2`: pressure/lifetime of the both-side DBUF path. The implementation should
   shorten address and global-load temp lifetimes without changing the proven symbolic LDS key.
4. If `4x2` still needs a new A-specific staging path, add it opt-in in `postrange.py`, for example
   `PREFILL_TC_LOCAL_STAGE_A_WINDOWKEY=1`, and use the same compressed window formula:

   ```text
   slot = dbuf_slot * a_slot_bytes + a_window_key * a_window_bytes
   store_addr = slot + row * row_stride + fragment_const
   load_addr  = slot + row * row_stride + fragment_const
   ```

5. Keep the current generic A `WITH_LOCAL` path as fallback. The new path must fail closed if it cannot identify the
   bounded A window ranges.
6. Preserve packed b128 lowering through the existing AMD store/load selectors. Do not introduce scalar per-half staging
   as the promoted path.
7. Run the probe loop after each slice:

   ```bash
   python3 extra/qk/prefill/a_fragment_alias_probe.py --indent 0
   PYTHONPATH=. python3 -m py_compile tinygrad/codegen/opt/postrange.py tinygrad/renderer/isa/amd.py extra/qk/prefill/a_fragment_alias_probe.py
   PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py
   ```

### Stop Point For This Fork

If `postrange.py` cannot express the A window key without recreating the earlier `UNROLL(STACK)` verifier failure or a
pathological compile, the next valid fallback is not a late renderer guess. The fallback is a small explicit post-expander
carrier that preserves:

```text
logical_lds_key = (operand=A, dbuf_slot, bounded_a_window, row, fragment_const)
```

until AMD pre-isel can lower it to the existing `global_load_b128 -> ds_store_b128 -> ds_load_b128 -> v_wmma` substrate.

## Path 2 `4x2` No-Spill And Correctness Scope

Date: 2026-07-08.

The immediate `4x2` blocker has split into two layers:

| Layer | Status | Evidence |
|---|---|---|
| Structural compile/no-spill | Unblocked by address rematerialization. | `REGALLOC_ADDR_REMAT=1` compiles `u0=4,u1=2,loc=2,unr=2`; LDS is 49152 bytes; symbolic A proof passes. |
| Runtime correctness | Still blocked. | Shared schedule harness returns `WRONG rr=2.2e-01` on `512x5120x5120`. |

### Current Pressure Diagnosis

Baseline failing `4x2` path:

```text
REGALLOC_DEBUG: 2109 uops, PEAK 108 live vregs @ uop 624
  73 V_IADD
  20 MOV
   4 DS_LOAD_B128
```

The pressure is address-dominated, not fragment-data dominated. `REGALLOC_ADDR_REMAT=1` allows this shape to compile by
rematerializing pure address values instead of spilling them. `REGALLOC_NO_LOOP_EXTEND_ADDR=1` alone reduces the apparent
peak to 86 but still spills. The combination compiles but worsens correctness (`rr=7.4e-01`), so it is not the path.

### Runtime Results

Representative command:

```bash
PYTHONPATH=. DEV=AMD:ISA \
REGALLOC_ADDR_REMAT=1 \
PREFILL_DBUF=1 AMD_ISA_WMMA_B128_FRAG=1 AMD_ISA_REG_ACCUM=1 AMD_ISA_WAITCNT_TARGETED=1 \
PREFILL_TC_LOCAL_STAGE=both PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE_POST=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PREFILL_DBUF_LDS_CONST_IMM=1 PREFILL_DBUF_LDS_INDEX_SPLIT=1 PREFILL_DBUF_LDS_STORE_BASE_SPLIT=1 \
PREFILL_DBUF_DIRECT_B128_CHAIN=1 PREFILL_DBUF_LDS_ADDR_USE_DEP=1 \
python3 - <<'PY'
from extra.qk.prefill_v2_schedule_search import _run_config
print(_run_config(512,5120,5120,4,2,2,2))
PY
```

Current result:

```text
status = WRONG rr=2.2e-01
```

Controls:

| Case | Result | Meaning |
|---|---|---|
| `2x2` without remat | PASS, about 6.8 TFLOPS | Existing small bounded DBUF route is numerically valid. |
| `2x2` with `REGALLOC_ADDR_REMAT=1` | PASS, about 9.4 TFLOPS | Address remat is not globally value-corrupting. |
| `4x2` with `REGALLOC_ADDR_REMAT=1` | Compiles, but `WRONG rr=2.2e-01` | The remaining bug is layout/order/remat interaction specific to the larger A-window route. |
| `4x2` with conservative waitcnt | Still `WRONG rr=2.2e-01` | Not a waitcnt drain issue. |
| `4x2` with `REGALLOC_ADDR_REMAT=1 REGALLOC_NO_LOOP_EXTEND_ADDR=1` | Compiles, but `WRONG rr=7.4e-01` | Shortening loop extension this way changes values more, so do not use it as the fix. |
| `4x2` with `REGALLOC_ADDR_REMAT=1 REGALLOC_ADDR_REMAT_NO_END=1` | Spills again | Loop-end address remat is required for structural compile; disabling it is a diagnostic only. |
| `4x2` with `REGALLOC_ADDR_REMAT=1 REGALLOC_ADDR_REMAT_END_NO_EMIT=1` | Still `WRONG rr=2.2e-01` | Emitted loop-end remat instructions are not the cause. |
| non-DBUF `4x2` | PASS, about 9.2 TFLOPS; LDS 24576 bytes | The larger A-window shape is valid without DBUF. The bug is DBUF slot/cadence specific. |

### Diagnostic Added

`tinygrad/codegen/late/regalloc.py` now has default-off diagnostics:

```text
REGALLOC_ADDR_REMAT_NO_END=1
REGALLOC_ADDR_REMAT_END_NO_EMIT=1
REGALLOC_DEBUG_REMAT=1
```

`REGALLOC_ADDR_REMAT_NO_END=1` prevents `can_remat` from treating `Ops.END` as an address-remat user. Disabling END remat
makes `4x2` spill again, so END remat is necessary for the current structural compile. `REGALLOC_ADDR_REMAT_END_NO_EMIT=1`
keeps the allocation effect but suppresses emitted loop-end remat restores; the result is still `WRONG rr=2.2e-01`, so
emitted END remat instructions are not the correctness cause. `REGALLOC_DEBUG_REMAT=1` reports remat composition.

Current `4x2` remat composition:

```text
REGALLOC_REMAT: count=118
  DS_LOAD_B128 <- V_OFFSET : 29
  END <- V_IADD            : 76
  END <- V_IMUL            : 6
  END <- V_OFFSET          : 2
```

### Corrected DBUF Operand Diagnosis

The operand-aware proof corrected the earlier "A imbalance" interpretation. The prior store/load counts mixed A and B
local buffers. After labeling `frag_load_b128_A` and `frag_load_b128_B` at the AMD proof site:

| Case | Runtime | A symbolic proof | Meaning |
|---|---|---|---|
| DBUF `2x2 both` | PASS | PASS | Known-good small both-stage route. |
| DBUF `4x2 A-only` | PASS, about 11.6 TFLOPS | PASS | Larger A window is valid under DBUF when B is not staged. |
| non-DBUF `4x2 both` | PASS, about 9.2 TFLOPS | PASS | Larger A+B staged layout is valid without DBUF. |
| DBUF `4x2 both`, DS immediate fold forced | WRONG `rr=2.2e-01` | PASS | The remaining bug is unsafe LDS constant-offset folding, not missing A coverage. |
| DBUF `4x2 both`, materialized LDS offsets | PASS, about 6.6 TFLOPS on `512x5120x5120` | PASS | Correctness is recovered when DS `offset0` folding is disabled. |

Two negative tests are also banked:

| Test | Result | Meaning |
|---|---|---|
| `PREFILL_DBUF_REDUCE_RANGE_STRICT=1` | `4x2 both` still wrong | Picking actual `AxisType.REDUCE` before fragment `UNROLL` is not sufficient. |
| Skip generic withlocal packing for the smaller B tile-key buffer | `4x2 both` becomes `NaN` | Generic packing is part of the current runnable path; simply disabling it is not a fix. |

Safe route-level fix:

```text
DBUF both-stage with u0>2 is rejected by the schedule-table gate only when
PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1 is also set, unless PREFILL_DBUF_BOTH_U0_GT2_PROVEN=1 overrides it.
```

This prevents the known-wrong immediate-folded `both 4x2` candidate from being selected while preserving the
proven-correct materialized-offset DBUF route. The remaining compiler primitive is a proven DS-offset folding pass, not a
B tile-key bridge, scheduler, or waitcnt fix.

Detailed scope: `docs/dbuf-safe-ds-offset-folding-scope.md`.

### Current Hypotheses

| Hypothesis | Prior | Evidence | Next test |
|---|---:|---|---|
| H1. DS `offset0` folding is unsafe in the DBUF both-stage path. | High | `PREFILL_DBUF_LDS_CONST_IMM=1` plus the legacy fold gives `WRONG rr=2.2e-01`; materialized offsets pass. | Keep folding disabled by default; reintroduce only with an assembler-level offset proof. |
| H1b. B DBUF cadence/tile-major mapping is wrong for `both u0>2`. | Low | Passing materialized-offset route uses the same B cadence; `2x2` and `4x2` have identical B local permutation. | Keep B proof in the tracer, but do not lead with a B packer rewrite. |
| H2. Final global/LDS address proof cannot track remat, hiding an actual out-of-bounds or alias. | Medium | `_final_stream_address_proof` has many `unknown_defs` under remat. | Extend the final analyzer to follow rematerialized `V_OFFSET`/`V_IADD` chains and classify unknowns separately from violations. |
| H3. Address remat at loop end restores/clobbers a loop-carried address needed by the next iteration. | Low | END remat no-emit did not change `rr`; non-DBUF `4x2` is correct. | Keep diagnostics, but do not lead with regalloc. |
| H4. Waitcnt/scheduler. | Low | Conservative waitcnt still wrong. | Do not spend more time here until H1-H3 are clean. |

### Next Work Sequence

1. Keep DS immediate folding disabled unless `PREFILL_DBUF_LDS_CONST_IMM_UNSAFE=1` is set for diagnostics.
2. Use the materialized-offset route as the correctness baseline for DBUF `4x2 both`.
3. Keep `REGALLOC_ADDR_REMAT=1` as a structural compile unlock while debugging, but do not promote it until correctness
   passes.
4. Strengthen the final proof/tracer for remat:
   - preserve symbolic LDS proof as authority for key coverage;
   - add final remat-chain equivalence so physical addr-register families are not false failures;
   - separately report true unknown address ranges.
5. Only after `4x2 both` is both no-spill and numerically correct, promote `REGALLOC_ADDR_REMAT=1` into the Path 2 env bundle.
