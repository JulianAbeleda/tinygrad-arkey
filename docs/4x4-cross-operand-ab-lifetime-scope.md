# 4x4 Cross-Operand A/B Lifetime Scope

## Big Picture

The generated AMD ISA path is trying to reach the same machine-code class as the working 4x4 handwritten path:

```text
global fragment data -> wide LDS store -> LDS barrier -> wide LDS load -> WMMA
```

The target is not a handwritten assembly escape hatch. The target is generated machine code that stages both WMMA
operands through LDS with wide stores/loads, avoids spills, stays verifier-clean, passes GPU correctness, and then becomes
eligible for DBUF/performance work.

## Current State

| Area | State | Meaning |
|---|---|---|
| A-only packed LDS staging | Passes structurally and centrally behind `PREFILL_LDS_PACK_WITHLOCAL_B128=1`. | A-side substrate is usable as the pressure target. |
| A-only packed LDS with B tile-key flag present | Fixed by scoping the devectorizer LOCAL pointer-grouping guard to B tile-key local slot `993`. | A packed staging now survives in the B tile-key composition. |
| B tile-key scalar correctness | Central B-only correctness passes behind `PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` when the packed bridge is not enabled. | The semantic B LDS slot formula is correct. |
| B packed bridge correctness | Passes centrally with `PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 PREFILL_LDS_PACK_WITHLOCAL_B128=1`. | The B tile-key formula and packed-store bridge now agree. |
| Both-side native structural | Passes with wide LDS shape, no scalar LDS stores, no spills, and WMMA operands sourced from `ds_load_b128`. | Generated native ISA has reached the intended A+B LDS-staged substrate. |
| Both-side DBUF substrate | Passes central correctness and structural no-spill under the DBUF flag bundle. | The remaining work is DBUF cadence/perf proof, not basic correctness. |
| Scheduler/waitcnt | Scheduler-off and conservative waitcnt do not fix the pressure failure. | This is not the next layer to tune. |

## Problem Definition

The investigated failure has three layers. The immediate composition failure was that enabling the B tile-key path globally changed
local pointer grouping/lowering enough that the A packed `WITH_LOCAL` path stops compiling no-spill. In that state,
`both` is not really comparing packed A plus packed B; it is scalar A staging plus the B bridge, and pressure spikes before
the intended packed A+B shape is reached.

That composition failure is fixed by narrowing the guard in `devectorizer.py` so `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1`
only disables LOCAL pointer grouping for the B tile-key local placeholder, slot `993`. After that, both-side native
structural staging compiles no-spill.

If a future change reintroduces A/B pressure, the fallback diagnosis remains a cross-operand lifetime/order problem:

```text
A staging stream live
B staging stream live
barrier
LDS loads for A/B
WMMA
```

For B-only and A-only, each side can be made compact enough to compile. If both regress again, the graph may give regalloc
too much independent live work at once: address temps, global-load temps, pack carriers, LDS store operands, and later LDS
load/WMMA operands overlap instead of draining one stage before the other starts.

The fallback fix would be to create a verifier-clean order primitive that makes the intended stream explicit:

```text
stage A rows
store A rows to LDS
kill A staging temps

stage B rows
store B rows to LDS
kill B staging temps

barrier
load A/B rows from LDS
WMMA
```

The primitive must be value-neutral. It should constrain ordering/lifetime only; it must not alter the mathematical
fragment values.

The live blocker was narrower than the fallback pressure diagnosis:

```text
scalar B tile-key store/read     -> PASS
packed B tile-key ds_store_b128  -> now PASS
```

The packed bridge wrote the right LDS address families (`tile*256 + row*16 + frag`) and read B via `ds_load_b128`, but the
first broken form exposed an unsafe fixed-temp/register-span contract: V_PACK values could occupy the fixed b128 temp span in
an order that did not match `ds_store_b128(v232:v235)`. The fix makes the four packed words an explicit int32.vec4 tuple and
sorts/validates it by constrained VGPR index before selecting `ds_store_b128`. That keeps the logical packed-word order tied
to the contiguous data span consumed by the final instruction.

## Non-Goals

- Do not add handwritten assembly.
- Do not treat this as a hardware bug.
- Do not solve it with scheduler or waitcnt changes first.
- Do not accept a route that requires spills.
- Do not accept central correctness alone if the native structural probe still has scalar LDS staging or no-spill failure.
- Do not reintroduce value-level `AFTER` dependencies from half scalar values to a void barrier; that shape is not verifier-clean.

## Layer Scope

| Layer | Role | Decision |
|---|---|---|
| `tinygrad/codegen/opt/postrange.py` | Still has WMMA operand identity, fragment axes, tile ranges, and local staging semantics. | Preferred primitive layer for cross-operand stage ordering. |
| `tinygrad/renderer/isa/amd.py` pre-isel | Can bridge proven store groups before instruction selection. | Acceptable tactical bridge, but not the final generic primitive. |
| `tinygrad/codegen/late/devectorizer.py` | Can avoid invalid local pointer grouping. | Keep as guard/substrate only. |
| AMD scheduler/waitcnt | Controls final instruction movement and hazards. | Out of scope until no-spill packed A+B compiles. |

## Required Work

1. Preserve the locked baselines:
   - A-only packed route remains no-spill.
   - A-only packed route also remains no-spill with `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` present.
   - B-only tile-key plus bridge remains no-spill.
   - B-only and both central route-bound correctness remain finite and within the known RMSE envelope.

2. Keep the composition regression fixed:
   - scope B tile-key local pointer grouping avoidance to the B tile-key buffer/path, or replace it with another
     verifier-clean carrier that preserves A packing;
   - reject any fix where `local-stage=a` plus `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` still emits scalar A LDS staging.

3. Preserve the both-side structural pass:
   - locate A store group;
   - locate B store group;
   - record whether the groups are independent siblings, nested groups, or tied by an order dependency;
   - record which live classes dominate the pressure peak.

4. If both-side pressure regresses, test verifier-clean ordering forms:
   - void/effect-level group ordering, not half-value ordering;
   - `GROUP`/`STORE`/`BARRIER`-level carrier if accepted by `tinygrad/uop/spec.py`;
   - fail closed behind a new flag if the shape is accepted.

5. Only if required, implement the smallest primitive that enforces one staging stream draining before the other:
   - preferably in `postrange.py` where both operands are selected;
   - preserve the existing B tile-key scalar correctness path;
   - keep the AMD B gather bridge only as a bridge if the primitive still needs it.

6. Move next to the remaining larger 100% goal: two-slot DBUF cadence, overlap, waitcnt/perf proof, and rollout policy.

## Latest Gate Results

| Gate | Result |
|---|---|
| Unit suite | `PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py` passes. |
| B packed central | PASS; `rel_rmse_vs_ref=0.0002076508681057021`, `max_abs_vs_ref=0.03130340576171875`. |
| Both packed central | PASS; `rel_rmse_vs_ref=0.00020765016961377114`, `max_abs_vs_ref=0.03130340576171875`. |
| Both packed + DBUF central | PASS with the same RMSE/max-abs envelope. |
| Both packed + DBUF structural | `REGALLOC_SPILLS: count=0 stack_size=0`; peak 60 live VGPRs; WMMA `src0/src1=ds_load_b128`; no scalar LDS stores. |

## Acceptance Criteria

| Gate | Required result |
|---|---|
| AB0. Unit tests | `PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py` passes. |
| AB1. A-only structural | Native probe `local-stage=a` has `ok=true`, `ds_store_b128=8`, `ds_load_b128=8`, no scalar LDS stores. |
| AB2. A-only composition | Native probe `local-stage=a` with `PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` still has `ok=true`, `ds_store_b128=8`, `ds_load_b128=8`, no scalar LDS stores. |
| AB3. B scalar correctness | Route-bound gate `local-stage=b` with `PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1` and without packed bridge passes. |
| AB4. B packed correctness | DONE: route-bound gate `local-stage=b` with `PREFILL_LDS_PACK_WITHLOCAL_B128=1` passes with the same RMSE envelope as scalar B. |
| AB5. Both native no-spill | DONE: native probe `local-stage=both` has `REGALLOC_SPILLS: count=0`. |
| AB6. Both wide LDS | DONE for DBUF route: native probe emits wide `ds_store_b128`/`ds_load_b128` with no scalar `ds_store_b16`/`ds_store_b32`. |
| AB7. Operand origins | DONE: WMMA operands for both A and B are loaded from LDS wide loads, not direct global fragments. |
| AB8. Pressure profile | `REGALLOC_DEBUG=1` no longer shows the both-side peak dominated by simultaneous A+B scalar/global staging. |
| AB9. Verifier clean | `SPEC=1` does not fail on malformed `AFTER`, `PTRCAT`, vector local pointer, or `UNROLL(STACK)` forms. |

## Commands

Baseline unit gate:

```bash
PYTHONPATH=. pytest -q test/unit/test_amd_isa_wmma.py
```

B-only native structural gate:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=b \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Composition regression gate:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=a \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Both-side native structural gate:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Both-side pressure gate:

```bash
REGALLOC_DEBUG=1 \
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 extra/qk/prefill/native_isa_l4_stream_probe.py --m-up 2 --indent 0
```

Both-side central correctness gate:

```bash
AMD_ISA_WMMA_B128_FRAG=1 \
PREFILL_TC_LOCAL_STAGE_B_TILEKEY=1 \
PREFILL_TC_LOCAL_STAGE=both \
PREFILL_TC_LOCAL_STAGE_WITH_LOCAL=1 \
PREFILL_LDS_PACK_WITHLOCAL_B128=1 \
PYTHONPATH=. python3 -m extra.qk.prefill_graph_gemm_route_bound_stage_gate --run-amd --local-stage both --compact
```

## Completion Definition

This scope is complete for the A+B packed correctness/lifetime blocker. The route now stages A and B through wide LDS,
compiles without spills, passes route-bound correctness, and keeps WMMA operands sourced from LDS. The larger LDS DBUF goal
continues from this substrate: prove two-slot cadence, refine overlap, tune targeted waitcnt, and measure performance.
