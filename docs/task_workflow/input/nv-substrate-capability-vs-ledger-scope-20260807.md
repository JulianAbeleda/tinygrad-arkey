# NV substrate capability vs measured ledger - exhaustive cross-reference scope

Date: 2026-08-07
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `e2ea330a6`
Frame: Qwen3-8B-Q4_K_M, d512, RTX 5090 / driver 595.84, native `DEV=NV`
Status: **scope. Contains no new measurement: every microsecond is carried from an existing
record and cited. Authorizes no implementation and changes no policy artifact. Its one
build item is a small delta to an existing tool, specified in section 5.**

Authorities carried:
`nv-decode-parity-campaign-reconciled-ledger-20260805.md`,
`nv-decode-exhaustive-forward-scope-20260805.md`,
`nv-overlap-exhaustive-scope-20260805.md`,
`nv-fusion-exhaustive-scope-20260805.md`,
`nv-host-exhaustive-scope-20260805.md`,
`nv-boundary-free-ordinary-uop-gate-20260805.json`,
`m4-resadd-landing-blocked-record-20260806.md`,
`m4-resadd-s4-gate-run-record-20260806.md`.

## 1. Why this scope exists

The parity ledger is an **attribution** document: it says where elapsed time goes. It does
not say *why* a row is unrecovered, and the two reasons are not interchangeable:

- **Schedule-blocked** - the construction is expressible and measured, and loses on wall.
- **Capability-blocked** - the construction is not expressible at all. No schedule search
  reaches it, and the attribution is unreachable until the substrate changes.

Every NO-GO logged since 2026-08-05 is the second kind. This scope separates the columns so
substrate work is prioritized by the mass it unblocks rather than by ledger row order.

## 2. Assets reused - nothing here is new machinery

This scope composes existing tools. The only build item is section 5's tool delta.

| asset | what it already does | reused for |
| --- | --- | --- |
| `extra/llm_research/decode/nv_boundary_free_ordinary_uop_gate.py` | **Construction gate, not a benchmark.** Establishes ordinary scheduler topology for a realized input and a lazy producer view; advances a candidate only if it yields one replayable ordinary program with no custom-program boundary and no CONTIGUOUS. Emits `verdict` + `reason`. | **The capability oracle.** Already produced the `CONSTRUCTION_GAP` verdict this scope is built on. |
| `extra/llm_research/decode/nv_fusion_population_ledger.py` | CPU-only exhaustive population ledger, schema `tinygrad.nv_fusion_population_ledger.v1`. Classifies every DAG node into disjoint populations (`flash`, `norms`, `residual_cast_contiguous`, `vocab_feedback`, `rope_kv`) against the exact 731-node native role partition; flags `exact=False` on heuristic fallback rather than silently assigning. `--dag` / `--out`. | **The population enumeration.** Supplies the row set and the per-population mass. |
| `extra/llm_research/decode/nv_co_schedule_candidate_scan.py` | Exact critical-path recovery per pair/node/population over the authority DAG, with greedy re-selection. | **Already closed the overlap column.** Not re-run; its verdict is carried in section 4.1. |
| `extra/llm_research/decode/nv_host_recovery_ledger.py` | Host / outside-window workstream ledger. | The host column, currently unclassified (section 6). |
| `/tmp/nv_p4_redirect_on_dag_20260805.json` | Duration-bearing authority DAG, 875 nodes / 4080 edges, digest `49838b8ab2e7118d0c384fb93d2b4c3085b3732f1fe8d5abc69d51d232a6b413`. | Shared input to both ledger and scan. Not regenerated. |
| `m4_resadd_substrate_schedule_gate.py`, `m4_residual_boundary_fold_probe.py`, `m4_resadd_section6_gate.py` | Existing M4 substrate/probe/gate ladder. | Precedent for the proof-stage sequencing critique in section 6. |

**Explicitly not created by this scope:** no new DAG capture, no new census tool, no new
ledger schema, no new measurement harness, no re-derivation of any published number.

## 3. Measured attribution (carried, not re-derived)

From `nv-decode-exhaustive-forward-scope-20260805.md`:

```text
1646.170 us/token
= support-work exposure 1108.082   ( hidden-overlap  445.954
                                   + fusion/dataflow 662.128 )
+ quant cores            302.788
+ outside window         239.805
- llama gaps               8.111
- bridge                   1.143
+ outer reconciliation     4.749
```

Fusion/dataflow family split (`nv-fusion-exhaustive-scope-20260805.md`) - note these family
names are exactly `nv_fusion_population_ledger.py`'s `POP_*` constants, which is what makes
the section 5 join mechanical:

```text
norms                      495.330      POP_NORMS
flash                      163.029      POP_FLASH
residual/cast/contiguous   240.106      POP_RESIDUAL
vocab/feedback              71.215      POP_VOCAB
rope/kv                     17.965      POP_ROPE_KV
llama Q8 pack             -325.517
                         = 662.128
```

Composed same-session bracket: native `5.3242440` vs llama `4.0056768` ms/token, gap
`1318.5672 us`. Booked to date: 270.468 us across five recoveries.

## 4. Capability inventory - what the build cannot express

Each entry: the missing capability, the record naming it, status, and gated rows.

### C1 - No generic cooperative reduction-to-output primitive

**Named by:** `nv-boundary-free-ordinary-uop-gate-20260805.json`, verdict
`CONSTRUCTION_GAP`, reason: *"ordinary RMSNorm has a global reduction result consumed by a
separate epilogue program; existing scheduler lowering has no generic cooperative
reduction-to-output primitive."* Contract shape `[1, 4096]`.

**Status:** OPEN. No implementation, no scope.

**Gates:** `POP_NORMS`, **495.330 us** - the largest family in the fusion bucket. Both 08-06
fusion A/B arms returned NO-GO at this gate without reaching a GPU arm.

**Why highest-value:** a *missing primitive*, not a missing optimization. Until lowering can
emit a cooperative reduction whose result is consumed in-program, no schedule search, no
candidate generation, and no policy artifact can reach this mass.

### C2 - Opaque custom kernels are not fusible from outside

**Named by:** `nv-fusion-residual-ab-record-20260806.md` - *"the admissible ordinary-UOp
in-core construction is not expressible because the consuming q4k GEMV is an opaque custom
program."*

**Status:** WORKED AROUND, not solved. The M4 path inverts it: author the epilogue *inside*
the custom kernel (`q4k_g3_lanemap_gemv_epi_resadd_*`, measured and bitwise-exact) and fold
the input boundary on the ABI side - structurally what llama.cpp does with `has_fusion`
inside `mul_mat_vec_q`.

**Gates:** every in-core epilogue against a custom kernel.

**Scaling limit this raises:** the inversion is per-kernel hand-authoring. No mechanism
*derives* an epilogue into an opaque kernel; each is an emitter change.

### C3 - Typed boundary ABI is producer-rank-limited

**Named by:** `m4-residual-boundary-fold-probe-record-20260806.md`. M5's
`_validated_typed_view` admits only already-flat producers (`Hq*Hd,`); the `(1,1,4096)`
opaque block-output boundary needs the extended contract.

**Status:** PARTIAL. Extended contract exists as a probe-owned dataclass (deliberately not a
`TypedViewRequest`; M5's 27 tests untouched). Production `ResidualViewRequest` scoped
(`m4-resadd-landing-scope-20260806.md` s2.2), **not implemented**.

**Gates:** `POP_RESIDUAL`, 240.106 us; the resadd fold targets ~100 us of it.

### C4 - rangeify could not express flat SPECIAL reads of non-flat opaque outputs

**Named by:** `m4-resadd-landing-blocked-record-20260806.md`,
`ValueError: bad reshape: () -> (1,1,4096)`.

**Status:** **FIXED.** Two 5-line deltas (`3fa55377c`, `2794d6772`, `afbe8fbfa`). Open fold
schedules end to end (1620 kernels); folded subgraph executes on CPU bitwise-equal to the
copy ABI. Unit-locked; hermetic gate 68 green incl. `test_m5_typed_boundary` 27/27.

**Correction carried:** the fold was never broken - it fires on all 36 blocks, layer-0
correctly fails closed. The crash came from inlined flash tile/combine chains (missing
REDUCE arm).

### C5 - weakint SPECIAL is not renderable on NV

**Named by:** `m4-resadd-s4-gate-run-record-20260806.md` -
`UOp verification failed at 31 on Ops.SPECIAL dtypes.weakint 1 [(Ops.CONST, dtypes.weakint,
4096)] gidx0`. `spec.py:490` weakint catch-all precedes the permissive SPECIAL rule at
`spec.py:237`; SPECIAL never fires for weakint.

**Status:** OPEN, fix identified - order `spec_shared`'s SPECIAL rule ahead of the catch-all,
or type the folded index int32 at the emitter / `pm_index_is_shrink` boundary.

**Gates:** the S4 GPU gate, therefore the entire resadd landing.

### C6 - Late STORE selector cannot see through the REDUCE_OUTPUT chain

**Named by:** reconciled ledger. Chain traced: `REDUCE_OUTPUT -> RUNTIME_SCRATCH semantic ->
reshape -> fp16 cast -> contiguous -> opaque CALL`. Exact typed-CALL producer attempt still
produced 0 reducers / 875 programs.

**Status:** OPEN, NO-GO for both exact constructions, with explicit instruction not to
broaden the predicate.

**Gates:** the alternative route into `POP_NORMS`. Note norms carries **two independent**
blockers: C1 on the boundary-free construction, C6 on the wrapper construction.

### C7 - Second lifetime class for owned invocation inputs unproven

**Named by:** reconciled ledger P2b. Generic rule removes 35 copies (874 -> 841) but leaves
35 new per-fused-block copies, missing the <=804 topology gate. **Status:** OPEN.

### C8 - Exact copy ownership unproven for custom epilogue outputs

**Named by:** reconciled ledger, attention-O row. 876 programs, 71 `E_86a2` copies (70 new),
35 fused-O calls. Ledger notes ownership *"remains unproven because the gate already
failed."* **Status:** OPEN, never isolated.

## 4A. Cross-reference (hand-derived - SUPERSEDED by section 8's generated artifact)

**This table is provisional.** It was assembled by reading records, which is exactly the
hand-derivation section 5 exists to replace. Treat it as the shape of the artifact, not as
the artifact.

| population | us/token | blocking class | blockers |
| --- | ---: | --- | --- |
| `POP_NORMS` | 495.330 | **capability** | C1 (primary), C6 (alt route) |
| `POP_RESIDUAL` | 240.106 | **capability** | C2, C3, C5 (C4 cleared) |
| `POP_VOCAB` | 71.215 | mixed | partially booked (P1, P5) |
| `POP_ROPE_KV` | 17.965 | closed | KV already one store/layer; gate/up already fused |
| `POP_FLASH` | 163.029 | **unclassified** | gate never run against it |
| hidden overlap | 445.954 | **structural - not capability** | 4.1 below |
| quant cores | 302.788 | schedule, exhausted | 13 bounded NO-GOs |
| outside window | 239.805 | **unclassified** | `nv_host_recovery_ledger.py` not assessed |

### 4.1 The overlap bucket is not a substrate problem

`nv_co_schedule_candidate_scan.py` closed this with full reachability closure over the
authority DAG:

```text
co-schedule ceiling  17.952 us     greedy realized  10.016 us
target              445.954 us     shortfall        24.8x
serialization slack 598.272 us     reachable        ~3%
```

200 dependency-independent pairs total; flash has zero independent support nodes; 486
support nodes (860.992 us) have no independent host.

**No substrate capability recovers this**, and the reason favours us: tinygrad's quant GEMVs
absorb quantization in-kernel, so llama's 217-node `quantize_q8_1` population *has no
tinygrad counterpart to overlap*. The 445.954 us is llama hiding work we do not perform.
Treat as attribution, not opportunity; remove from the buildable column.

## 5. The one build item: generalize the existing gate

The capability column above should not be a hand-written table. Both halves of the join
already exist as tools; only the gate's parameterization is missing.

**Current state.** `nv_boundary_free_ordinary_uop_gate.py` is hardcoded: `DIM = 4096`, one
`_ordinary()` RMSNorm construction, no `argparse`, no `main()`. It answers the capability
question for exactly one population.

**Delta.** Give it the same CLI shape `nv_fusion_population_ledger.py` already uses
(`--dag` / `--out`, plus `--population`), keyed on the ledger's existing `POP_*` constants,
and one construction per population behind that key. Emit per-population
`{verdict, reason, contract_shape}` under a new schema version of the gate's existing
envelope - no new schema family.

**Join.** Ledger `--out` (population -> mass, with its `exact` flag) joined on population
key against gate `--out` (population -> verdict/reason) produces section 4A mechanically,
including the `exact=False` flags the ledger already tracks and my hand-derived table drops.

**Why this is the right shape:** it is reproduce-from-artifact. The capability column
becomes regenerable from the committed DAG rather than from a reading of the records, and
it re-runs after any topology change - which matters because the overlap verdict is
explicitly scoped *"on the redirect-on topology"* and fusion landings change the graph.

**Estimated delta:** argparse + population dispatch + one construction per unclassified
population. The `flash` and `vocab` constructions are the real work; `norms` already exists.

## 6. Coverage and gaps IN THIS SCOPE

- **Overlap mechanisms (b)-(e) not read.** Only (a) reviewed in full. The parent scope flags
  *"resource-join census augmentation is the CPU-only enabling step"* and (a) defers to (e)
  for resource complementarity. The verdict quoted is (a)'s - decisive for in-graph
  co-scheduling, possibly not for the bucket.
- **`POP_FLASH` (163.029 us) unclassified.** The flash single-stage NO-GO was
  resource/underfill (+82.505 us), which reads as schedule not capability - but the gate has
  never been run against it. Section 5 is what closes this.
- **Host workstream (239.805 us) unclassified.** `nv-host-exhaustive-scope` read only to its
  frame; `nv_host_recovery_ledger.py` not assessed for capability blockers.
- **Capability inventory assembled by grep over `docs/`** plus records read in full.
  Blockers recorded only in `/tmp` artifacts, commit messages, or code comments are absent.
- **Proof-stage warning, carried from the M4 sequence.** The hermetic UOp probe could not see
  rangeify (C4); the CPU host-exec proof could not see NV render (C5). Each stage was blind
  to the next. A capability gate that passes is evidence of expressibility at *that stage
  only*.
- **No mass claimed recoverable.** Clearing a blocker makes a construction expressible;
  whether it wins on wall is separate, and 08-06 is the standing counter-example - fold
  expressible, still 0 credit.

## 7. Non-claims

- Does not assert C1 is worth 495.330 us. Asserts 495.330 us is unreachable while C1 is open.
- Does not reopen the overlap bucket or reorder the campaign's stated
  overlap-first / fusion-second / host-last sequence.
- No policy artifact changes. `decode-q4k-epilogue-resadd-route-policy.json` stays CLOSED.


## 8. Amendment 2026-08-07 - build item implemented, two corrections

The section 5 build item landed (`3f58c3400`, then the v3 delta below). The generated
artifact supersedes section 4A, and it corrects this scope twice.

### 8.1 Correction 1 - C1 gates flash as well as norms

The gate returns `CONSTRUCTION_GAP` for **both** norms and flash, with the same mechanism:

| population | programs | cause |
| --- | ---: | --- |
| norms | 2 | reduction + dependent epilogue |
| flash | 3 | reduction + dependent epilogue |

Section 4A listed flash as **unclassified** and speculated it was schedule-blocked because
the single-stage NO-GO was resource/underfill (+82.505 us). Both are true: the custom-kernel
route lost on resources *and* the ordinary route is not expressible.

```text
norms  495.330
flash  163.029
     = 658.359 us  capability-blocked by C1
fusion/dataflow bucket 662.128 - remainder 3.769
```

**99.4% of the fusion bucket is gated by one missing primitive.** The workstream collapses
to a single build item; C1 is not merely the largest item, it is very nearly the only one.

### 8.2 Correction 2 - this scope under-specified the constructions (v2 false PASS)

v2 reported `residual_cast_contiguous: PASS`, contradicting the S4 GPU gate, which crashes
the residual fold at render one day earlier. Both statements were in the repo simultaneously.

Cause: this scope said "one construction per population" and never required the construction
to instantiate the **real** producer or consumer. `_residual` used
`Tensor.randn(...).realize()` - a plain realized buffer, which
`m4-residual-boundary-fold-probe-record-20260806.md` documents as the trivially-folding
*control* case. The blocked form is `block_output`:
`MS(CONTIGUOUS(GETTUPLE(FUNCTION(precompile=True))))`. The gate was testing the control.

This is a scope defect, not an implementation defect. The v3 delta:

1. **`PASS` -> `ORDINARY_PASS`.** The gate is sound in one direction only.
   `CONSTRUCTION_GAP` is a valid lower bound; a pass is not an upper bound, because no
   ordinary arm instantiates a custom kernel (`contains_custom_kernel` false throughout).
2. **Per-population `scope` field**, stating what the verdict covers. Populations with no
   opaque arm state "absence means not assessed, not plain."
3. **`opaque_producer` arm**, reusing the M4 probe's `_fresh("block_output")` producer form
   rather than reimplementing it. Populations opt in via `OPAQUE_ARMS`; only `POP_RESIDUAL`
   is listed, because only it has a record establishing the real chain.
4. **Split-cause attribution.** A lowered reduce and a precompiled FUNCTION boundary both
   raise the program count; reporting both as "reduction + dependent epilogue" would merge
   C1 with the M4 blocker. The cause is now passed in, never inferred.

Result: `residual_cast_contiguous: OPAQUE_PRODUCER_GAP` - ordinary arm boundary-free,
production producer splits into 3 programs at the precompiled boundary. **The gate now
independently reproduces on CPU what S4 found on GPU**, which is the self-validation the v2
artifact lacked.

### 8.3 Generated capability column (supersedes 4A)

`docs/task_workflow/output/nv-boundary-free-ordinary-uop-gate-20260807.json`, schema
`tinygrad.nv_boundary_free_ordinary_uop_gate.v3`, DAG digest `49838b8a...` (875/4080).

| population | verdict | blocker |
| --- | --- | --- |
| norms | CONSTRUCTION_GAP | C1 |
| flash | CONSTRUCTION_GAP | C1 |
| residual_cast_contiguous | OPAQUE_PRODUCER_GAP | C2/C3/C5 |
| quant_core | ORDINARY_PASS | **not meaningful** - the construction is a dense fp16 matmul, not the Q4_K packed GEMV; the real population *is* a custom kernel |
| vocab_feedback | ORDINARY_PASS | ordinary arms only |
| rope_kv | ORDINARY_PASS | ordinary arms only |
| llama_q8_pack | ORDINARY_PASS | attribution credit, not a tinygrad population |
| other | NO_CONSTRUCTION | ledger fallback |

### 8.4 Still out of scope after v3

The opaque arm closes the *producer* side. It does not instantiate the custom q4k
**consumer**, so C2 (opaque consumer blocks in-core epilogue) and C5 (weakint SPECIAL
render) remain undetectable by this gate. `quant_core`'s pass is the clearest evidence of
that limit and is annotated as such above. Closing it means an arm that builds the real
`q4k_g3_lanemap_gemv_*` program - a larger delta, and not authorized here.

Test coverage: `test/unit/test_nv_boundary_free_ordinary_uop_gate.py`, 8 tests, including a
named regression test for the v2 false PASS. 62 passed across gate / ledger / M4 probe /
M4 landing / M5 typed boundary on this tree.
