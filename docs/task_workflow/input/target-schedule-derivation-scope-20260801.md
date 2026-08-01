# Target schedule derivation: one authority instead of an AMD clone

Date: 2026-08-01

Status: scoped, not implemented. Reviewed 2026-08-01 (Claude); census corrections folded in at their
rows, review questions answered in §8, findings recorded in §9. Written for review before code.

Branch boundary: all work begins on `nvidia-bringup-20260731` per the NVIDIA campaign rule (branch ->
selective merge to `exp` -> promote to `dev` only after NVIDIA is proven end-to-end). This scope does not
authorize promotion by itself. BoltBeam slices begin on BoltBeam `exp` per the BoltBeam campaign rule.

Companion to `prefill-qualification-canonical-path-scope-20260801.md` (C1-C5, the canonical lane and
per-target capability rows, implemented). This scope fixes the last place the canonical path still
inherits AMD values: the minted candidate schedule itself.

## 1. The defect in one sentence

**Both target mints clone AMD's admitted schedule wholesale and only rewrite the target identity, so the
schedule carries AMD literals (`rdna3_*` fragment layouts, `lgkm/vm` waitcnt vocabulary,
`cooperative_row_stride_64_b128` lane mappings, `wmma_f32_16x16x16_f16` instruction family,
`max_lds_bytes: 65536`) that admission then rejects or that a human must hand-retype per campaign (the C5
driver's `_nv_typed` step).**

The identity decoupling is done (CI-C / `candidate-identity-target-decoupling-scope-20260801.md`): no
`gfx1100` string appears in the NV/Metal mint identities, and `build_qwen3_8b_buffer2_candidate_set` stamps
the registry-resolved target into `workload` and `applicability`. What it does **not** do is touch
`schedule` or `static_constraints` at all (`/home/ubuntu/BoltBeam/boltbeam/full_kernel_candidate_set.py:158-206`:
`payload = seed_candidate.to_dict()` is the whole body of the loop). The schedule is AMD's schedule with a
new name on it.

## 2. The schedule is not one thing: exhaustive field census

Every field in the mint payload's `schedule` + `static_constraints`, its ownership class, and its consumer.
Ownership classes: **D** = derivable from declared facts/geometry, **C** = declared emitter contract (names
an existing lowering; cannot be computed), **G** = geometry (search output). The census is the load-bearing
content of this scope; nothing below is asserted from memory.

| field | value in both mints | class | producer | consumer |
| --- | --- | --- | --- | --- |
| `schedule.tile.{m,n,k}` | 128,128,32 | G | search | admission `geometry_divisibility`; codegen `KernelTileGeometry` |
| `schedule.waves.{m,n}` | 4,2 | G | search | admission; codegen `KernelTileGeometry` |
| `schedule.threads` | 256 | D (wm*wn*wave_size) | derive | admission; codegen (operand vector division) |
| `schedule.lane_ownership` | `rdna3_wmma_f32_16x16x16_f16_lds2_static` | C | declared row | admission schema-only; **coincides with `wmma.fragment_layout` in the buffer2 mints but is a distinct field -- the five-buffer row proves it (`rdna3_wave32_direct_wmma_output_tile` vs `rdna3_wave32_signed_i8_direct_global`, runtime_specs.py:355-360)** |
| `schedule.cooperative_load.{a,b}.lane_mapping` | `cooperative_row_stride_64_b128` | C | declared row | admission `capability_lane_map` (hardcoded expected string) |
| `schedule.cooperative_load.{a,b}.vector_width` | 8 | D (vector_bytes/itemsize) | derive | admission `capability_vector` (`*2 == vector_bytes`) |
| `schedule.cooperative_load.{a,b}.alignment` | 16 | D (vector_bytes) | derive | admission `capability_vector` |
| `schedule.lds.banks` | 32 | C (hardware/emitter fact) | declared row | schema-only; not consumed by codegen |
| `schedule.lds.padding` | 16 | C (transport contract) | declared row | stride arithmetic (`80 = tk*2 + 16`) |
| `schedule.lds.load_vector_width` | 8 | D | derive | admission `capability_vector` (**hardcoded `!= 8`**) |
| `schedule.lds.store_vector_width` | 8 | D | derive | admission `capability_vector` (**hardcoded `!= 8`**) |
| `schedule.lds.strides.{a,b}` | 80 | D (tk*itemsize + padding) | derive | admission; codegen `KernelTileGeometry` windows |
| `schedule.lds.windows.{a,b}` | [0,10240],[10240,20480] | D (tile*stride) | derive | admission; codegen `KernelTileGeometry` windows |
| `schedule.pipeline.buffer_count` | 2 | G/shape | search/row | admission `capability_pipeline`; codegen `KernelStage1PipelinePlan` |
| `schedule.pipeline.stage_count` | 1 | G/shape | search/row | admission; codegen |
| `schedule.pipeline.epoch_graph` | AMD barrier names (`before_fragment_load`, `after_wmma_before_slot_reuse`) | C | declared row | schema-only; not consumed by codegen |
| `schedule.wmma.instruction_family` | `wmma_f32_16x16x16_f16` | D (tc descriptor) | derive (`_instruction_family_for`, runtime_specs.py:70) | admission `capability_tc` |
| `schedule.wmma.fragment_layout` | `rdna3_wmma_f32_16x16x16_f16_lds2_static` | C | declared row | admission `capability_tc` |
| `schedule.wmma.accumulator_ownership` | `wmma_accum_wm_x_wn_8_vgprs` | C | declared row | schema-only |
| `schedule.dependency_policy.waitcnt.{lgkm,vm}` | 0,0 | C (AMD vocabulary) | declared row | schema-only; **`lgkm` is AMD-specific** |
| `schedule.dependency_policy.barriers` | AMD barrier names | C | declared row | schema-only |
| `schedule.epilogue.lane_mapping` | `wmma_accumulator_scalar_b16` | C | declared row | schema-only |
| `schedule.epilogue.vector_width` | 1 | C | declared row | schema-only |
| `schedule.residency.{preload,resident,reuse}` | a/b preloaded, accumulator resident, reuse a=4/b=2 | C/G | declared row / policy | admission `candidate_storage_kind` (resident marker only) |
| `schedule.numerical_mode` | `ieee_fp16_acc_fp32` | C | declared row | schema-only |
| `static_constraints.max_lds_bytes` | **65536 (AMD's value)** | D (**transport-dependent**: `shared_max` for LDS transport, `0` for direct-global -- the five-buffer row is 0, runtime_specs.py:375) | derive | admission `capability_lds` (checked against capability AND payload) |
| `static_constraints.max_vgpr_per_thread` | 256 | C | declared row | schema-only |
| `static_constraints.allow_spill` | false | C | declared row | schema-only |

Verified consumer census (rg over `tinygrad/` and `extra/llm_research/`): the fields codegen actually reads
are `tile`, `waves`, `threads`, `lds.windows/strides`, `pipeline.buffer_count/stage_count`, and the packed
`operand_sources`. Everything else (`lane_ownership`, `cooperative_load`, `lds.banks/padding`,
`epoch_graph`, `dependency_policy`, `epilogue`, `residency.reuse`, `numerical_mode`,
`accumulator_ownership`) is **admission vocabulary or documentation**: validated against the capability row
or schema-validated, never consumed by the lowering. The compile resolves its tensor-core descriptor from
`device` + `arch` (`_resolve_tensor_core`, runtime_specs.py:473), **not** from the payload's
`wmma.instruction_family`.

That last fact is the key to the design: the payload's contract vocabulary is a **self-consistency gate with
the capability row**, not a codegen input. Retyping is therefore a vocabulary problem, not a lowering
problem.

## 3. Where the AMD-isms live (every hardcoded site)

1. **BoltBeam builder** — `build_qwen3_8b_buffer2_candidate_set` clones the seed's schedule byte-for-byte;
   only `workload`/`applicability` are rewritten (full_kernel_candidate_set.py:158-206). The seed itself is
   AMD's (`boltbeam/data/full_kernel_candidates.json`).
2. **Lane payload injection** — `_payload_for_config` hardcodes wave 32 and stride 80:
   `schedule["threads"] = g["wm"] * g["wn"] * 32`, `a_end, b_end = g["tm"] * 80, (g["tm"] + g["tn"]) * 80`,
   `strides = {"a": 80, "b": 80}` (precontract_probe_lane.py:150-158).
3. **Admission hardcoded literals** — `any(schedule["lds"][x] != 8 for x in ("store_vector_width","load_vector_width"))`
   and `expected_lane_mapping = "wave_contiguous_b128" if register else "cooperative_row_stride_64_b128"`
   (runtime_specs.py:538-542). The `8` is a vector-width literal; the lane-mapping strings are AMD/Metal
   emitter contract names.
4. **The capability rows** (C1, correct) — `FullKernelCapability` declares per-target `fragment_layout`,
   `instruction_family` (derived from `tc`), `max_lds_bytes` (from `Renderer.shared_max`: cuda.py:12 =
   49152, cstyle.py:481 MetalRenderer = 32768, cstyle.py:583 HIPRenderer = 65536), `vector_bytes = 16`.
   These exist; nothing consumes them to build a schedule.
5. **static_constraints in the mints** — `max_lds_bytes: 65536` is cloned from AMD into both NV and Metal
   mints even though the capability rows declare 49152/32768.
6. **The C5 driver's `_nv_typed`** — scratchpad manual retyping of three fields
   (`instruction_family`, `fragment_layout`, `lane_ownership`) from the capability row
   (scratchpad/c5_nv_canonical_lane_probe.py). It is the human doing what a function should do, and it is
   only partial (it does not retype `waitcnt`, `epilogue`, `residency`, `cooperative_load`,
   `static_constraints`).

## 4. Proposed: one derive authority, three input classes

One pure function that assembles a target's schedule from three inputs -- **declared row, geometry, shape** --
instead of cloning another target's literal:

```text
derive_target_schedule(row, geometry, pipeline_shape) -> schedule dict
```

- **Declared row** (already exists as `FullKernelCapability`, extended with the C-class fields it does not
  yet carry): `fragment_layout`, `lane_ownership`, `accumulator_ownership`, `cooperative_load.lane_mapping`,
  `epilogue.lane_mapping`, `waitcnt`/`barrier` vocabulary, `lds.banks`, `lds.padding`, `residency`,
  `numerical_mode`, `max_vgpr_per_thread`, `allow_spill`. Each field carries the same citation discipline
  the rows already use ("names this repo's emitter contract; CUDA -> `cuda.py` `mma.sync`, Metal ->
  `simdgroup_matrix`"). Two naming disciplines apply to the row and the derive function together:
  the **family string is vocabulary, not hardware** -- the derive function must not fabricate a `wmma_`
  prefix for every target (the current `_instruction_family_for` does exactly that; CUDA's fragment layout
  is `cuda_mma_*` and Metal's is `simdgroup_matrix_*`, so the emitted family literal should come from the
  row, with the `tc` dims as the only derived part); and the row's **backend spelling is authoritative**
  (`"Metal"`, not the `"METAL"` map key that `_tensor_core_family` reaches via `.upper()`) -- the derive
  function keys on the row's declared spelling, never on an uppercased lookup alias.
- **Geometry** (search output, caller-supplied): `tile`, `waves`, `buffer_count`/`stage_count`. For the
  mechanism test these may be the AMD-seed geometry; for promotion they come from the search.
- **Derived** (computed, never stored as a literal): `instruction_family` (structural part only -- dims and
  dtypes from the target's `tc` descriptor; the vocabulary prefix comes from the declared row, see above),
  `threads` (`wm*wn*wave_size`), `cooperative_load.vector_width`/`alignment` (from `vector_bytes`),
  `lds.load/store_vector_width` (from `vector_bytes`), `lds.strides` (`tk*itemsize + padding`),
  `lds.windows` (`tile*stride`), `static_constraints.max_lds_bytes` (transport-dependent: `Renderer.shared_max`
  for LDS transport, `0` for direct-global -- no uniform formula, the five-buffer row is 0).

Consumers, in order of slices:

1. The **mint builder** (BoltBeam) stops cloning: it receives the per-target typed schedule template (or
   the row + geometry) and stamps `workload`/`applicability` onto it, exactly as it already stamps identity.
2. The **lane** (`_payload_for_config`) calls the same derive function: the hardcoded 32/80 disappear.
3. **Admission** checks against the row instead of hardcoded literals: the `!= 8` becomes
   `vector_bytes // itemsize`; `expected_lane_mapping` becomes the row's declared field.
4. The **C5 driver's `_nv_typed` is deleted**: the mint is already typed, so
   `run_precontract_probe(ProbeConfig(..., device=device))` is the whole probe, one call, any target.

## 5. Where the authority lives (the cross-repo decision)

Three options, to be decided in review. The constraint: BoltBeam mints candidate artifacts through a
deliberately tinygrad-free path (the mint modules are pure schema/data validation; see the correction
below); tinygrad owns the capability rows and admission.

- **A. Tinygrad is the authority; BoltBeam receives the typed template as input (recommended).** The
  derivation function lives next to the capability rows it reads. The mint command computes the typed
  schedule via tinygrad's derive function (or a tinygrad-published JSON snapshot of the per-target contract
  rows) and passes it to BoltBeam's builder. BoltBeam gains no new vocabulary; its job stays "validate +
  stamp identity". Cross-repo drift is bounded to the snapshot, pinned by a BoltBeam test that fails if the
  snapshot stops matching tinygrad's table.
- **B. BoltBeam's registry owns the contract vocabulary.** `targets.json` gains `fragment_layout`,
  `lane_mapping`, `waitcnt` names, etc. Tinygrad's capability rows become a derived view over it. Inverts
  the dependency: the runtime would read a research repo's data file, and tinygrad would need the registry
  reachable at load.
- **C. A shared JSON contract table both repos read.** Neutral schema, two readers. Cleanest in theory;
  adds a third artifact that must be kept in sync with `FullKernelCapability`, and there is no existing
  shared-data channel between the repos (artifacts flow one way: BoltBeam -> tinygrad bench).

Recommendation A: it matches the existing dependency direction (BoltBeam consumes tinygrad evidence;
INTEGRATION_TINYGRAD.md), keeps one authority, and makes the mint input explicit -- the mint cannot silently
inherit another target's vocabulary because it must be given a typed schedule.

**Correction from review (2026-08-01): BoltBeam *does* import tinygrad.** The blanket claim "BoltBeam must
not import tinygrad runtime code" is false: `boltbeam/runtime/amd_runtime_bridge.py:10-11` imports
`tinygrad.engine.realize.get_runtime` and `tinygrad.uop.ops`. The claim that option A actually rests on is
narrower and true: **the candidate-set mint path is deliberately tinygrad-free.** `search/spec.py:9` states
"pure schema/data validation: no hardware execution, no tinygrad imports", `full_kernel_candidate_set.py`
imports only `boltbeam` modules and stdlib, and INTEGRATION_TINYGRAD.md scopes the evidence-integration
package the same way. The doc should say that, not the stronger false claim; the recommendation is unchanged.

## 6. Hard constraints

- **AMD byte-identity is a test, not an aspiration.** `derive_target_schedule` on the AMD row + the
  seed geometry must reproduce the promoted template byte-for-byte. Existing pins that must not move:
  `ANCHOR_SINGLE_BUFFER_CANDIDATE_HASH` (`579b909f...`, runtime_specs.py:28), the promoted artifact's
  canonical/legacy identities (`51b05622...`/`5585ac26...`), and BoltBeam's pinned AMD set_hash
  (`e9839825993c...`, test_full_kernel_candidate_set.py). AMD admission outcomes identical before/after;
  the AMD rendered-source-equality scratchpad must stay green.
- **The v1 schema shape does not change.** Values are retyped; field structure, keys, and
  `static_constraints` shape stay exactly as today. (BoltBeam's v2 candidate schema is a different,
  backend-neutral shape; adopting it at the tinygrad boundary is out of scope -- see review question 4.)
- **No dtype/precision work.** The accumulator-carrier boundary (`dtypes.float.vec(8)` vs CUDA's `vec(4)`,
  kernel_pipeline.py:181) and the operand lane-layout derivation (kernel_lds.py:171) are the next NV
  campaign step, governed by `wmma-carrier-shape-awareness-scope-20260731.md` and
  `dtype-authority-decomposition-scope-20260731.md`. This scope changes *payload vocabulary*, not lowering.
- **No geometry search in this scope.** Geometry stays caller-supplied; search is the follow-on slice the
  canonical-path scope already sequenced.
- **No `PACKED_WMMA_ROUTES` changes, no new flags/subsystem, no `dev`/`master` changes.**

## 7. Slices with verification

Each slice keeps the NV e2e bench green (same ratchet discipline as C1-C5) and adds tests at module
boundaries. Commit prefixes: `[codegen]` `[nn]` `[test]` `[docs]` `[repo]`; BoltBeam slices use BoltBeam's
prefixes on BoltBeam `exp`.

| slice | change | verification |
| --- | --- | --- |
| T1 | Census pins: tests that assert the current AMD literals are *declared facts with citations*, one per C-class field (a contract table in `extra/llm_research/`); no behavior change | tests pass against today's tree; the census table in this doc becomes a code object, not prose |
| T2 | `extra/llm_research/target_schedule.py::derive_target_schedule(row, geometry, shape)`; AMD row + seed geometry -> byte-identical schedule vs the promoted template; non-AMD rows produce typed schedules (no `rdna3_*`, no `lgkm`, `max_lds_bytes` = declared); family string read from the row, never fabricated | unit tests: AMD equality; NV row -> row-declared family + `cuda_mma_*` layout + `max_lds_bytes: 49152`; Metal row -> `simdgroup_matrix` + 32768; a test asserting the derive function never emits a `wmma_` prefix not present in the row |
| T3 | Admission consumes the row: `capability_vector` uses `vector_bytes // itemsize`; `expected_lane_mapping` read from the row; `capability_lds` unchanged (already row-based) | full unit suite failing-ID set identical before/after; AMD admission outcomes identical; NV e2e ratchet unchanged (~156-158 tok/s, first token 50994) |
| T4 | BoltBeam builder accepts the typed schedule template (option A): stops cloning `schedule`/`static_constraints`, validates the template's pipeline family, stamps identity | BoltBeam test: AMD template -> byte-identical set_hash (`e9839825993c...`); NV/Metal templates -> mints with zero AMD vocabulary strings |
| T5 | Lane `_payload_for_config` uses `derive_target_schedule`; hardcoded 32/80 removed | existing lane unit tests updated; M1b/M1c dispatch still admits with `active_lds_bytes == 25600`; AMD probe config byte-identical payload before/after |
| T6 | Re-mint sm120 + m4_10c candidate sets via the typed path; C5 driver becomes one-call (`mint_path`, `device`), `_nv_typed` deleted; Metal driver written (admission-level runnable on this box; GPU run on a Mac) | NV mint as committed now ADMITS through the lane (the old `capability_tc` skip for the typed mint must disappear); NV-typed buffer2/buffer1 probes fail only at the known lowering boundaries (kernel_pipeline/kernel_lds), recorded in the bring-up doc; NV e2e ratchet unchanged; docs row added |

## 8. Review questions

1. **Authority location.** Is option A (tinygrad derives; BoltBeam receives the typed template) right, or
   does the review prefer B or C? The decisive fact: the candidate-set mint path is deliberately
   tinygrad-free (BoltBeam as a whole does import tinygrad in `runtime/amd_runtime_bridge.py`, but the
   mint modules do not), and tinygrad must not read a research repo's data file at load. Is the
   pinned-snapshot sync test enough to make A safe?
2. **What belongs in the row vs the derive function.** The census splits D and C by "computable" vs
   "names an existing lowering". Is `residency.reuse (a=4,b=2)` really a declared contract, or is it a
   geometry/search field? Is `lds.padding = 16` a transport constant of b128 (derivable) or a per-target
   emitter choice (declared)?
3. **Should `lane_ownership` stay in the schema at all?** It is *not* a duplicate of
   `wmma.fragment_layout` -- the five-buffer row carries different values for the two
   (`rdna3_wave32_direct_wmma_output_tile` vs `rdna3_wave32_signed_i8_direct_global`), so the field has
   distinct meaning. In the buffer2 mints the values coincide, which is what made it look redundant. The
   question is narrower: is the coincidence in the LDS family a coincidence to preserve, and does the
   derive function emit the two fields from one declared value for LDS transport while keeping them
   distinct for direct-global?
4. **v2 as the authoring format.** BoltBeam's v2 candidate schema is backend-neutral by construction
   (`plan_kind`, `transforms`, `mapping.lane_policy`, `memory.a/b/c`, `compute.family`, `numerical_mode`).
   Is "retype v1 values" the right immediate move, or should the typed-schedule work author in v2 and
   downgrade to v1 at the tinygrad boundary? (v2 changes canonical identity shape and requires a
   tinygrad-side v1 downgrade; larger blast radius.)
5. **`static_constraints.max_lds_bytes` authority.** Today admission checks the payload value AND the
   capability row. The derive rule is transport-dependent (LDS -> `shared_max`, direct-global -> 0), so a
   typed mint carries the target's declared value and the two always agree. Should admission stop checking
   the payload value (row is the authority), or keep both as a self-consistency gate?
6. **Sequencing vs the fact-refactor scope.** `target-fact-type-refactor-scope-20260801.md` (F3) removes
   the all-defaults `FullKernelCapability()` constructor and builds rows from `TargetFacts`. T2/T3 extend
   the same rows. Should F land before T (typed facts first, then derive), or after (derive first, then
   lift the extended row into `Fact[T]`)? F's own §8 says to wait if the table's future shape is being
   decided -- this scope is exactly that decision.

---

## 9. Review resolutions (Claude, 2026-08-01)

Answers to §8, with the checks that produced them. Q1/Q3/Q5's corrections are already folded into §2/§5
above; what follows is the decision for each.

1. **Authority location -- A.** The corrected justification in §8.1 is the right one: the mint modules are
   deliberately tinygrad-free (four of them say so in their docstrings) even though
   `boltbeam/runtime/amd_runtime_bridge.py:10-11` imports `tinygrad.engine.realize` and `tinygrad.uop.ops`.
   A matches the existing dependency direction and keeps one authority. The pinned-snapshot sync test is
   sufficient **if it fails loudly** -- a warn-on-drift test is not a gate.

2. **`residency.reuse` is derivable (D), not a declared contract.** Checked against the seed
   (`BoltBeam/boltbeam/data/full_kernel_candidates.json`, `rows[0]/search_row/full_kernel_candidate`):
   tile `(128,128,32)`, waves `(m=4, n=2)`, AMD tc dims `(16,16,16)`. `derive_precontract_shape_factors`
   computes `sm = 128//(4*16) = 2` and `sn = 128//(2*16) = 4`. The seed carries `reuse: {a: 4, b: 2}` --
   i.e. `reuse.a == sn` and `reuse.b == sm`, which is exactly the right semantics (an A fragment is reused
   across all N subtiles, a B fragment across all M subtiles). These are the standard register-blocking
   reuse factors, already computed by an existing helper. Reclassify the census row C/G -> D and derive
   them; do not carry them as declared literals.

   **`lds.padding` is the more interesting half: D where bank facts are declared, C where they are not.**
   The padding exists to avoid LDS bank conflicts, and bank structure is *already* a declared per-target
   `Renderer` fact (`lds_bank_dwords`, `lds_bank_cycle_lanes`) -- deliberately `None` on Metal because
   Apple does not publish it. So padding is derivable exactly where those facts exist and must be declared
   (or absent) where they do not. This is the fact-refactor scope's absence taxonomy showing up in a second
   place independently, which is evidence the taxonomy is real rather than an abstraction looking for a use.

3. **Keep `lane_ownership`.** Resolved in §8.3 as amended: it is not a duplicate. For the LDS family the
   derive function may emit both fields from one declared value; for direct-global they stay distinct.

4. **Retype v1 now; do not author in v2.** v2 changes canonical identity shape and needs a tinygrad-side
   downgrade -- a second blast radius stacked onto a scope whose hard constraint is byte-identity against
   pinned hashes. Revisit v2 when geometry search lands and the mint is being rewritten anyway.

5. **Keep both checks.** They are different claims: the capability row is the *device* budget, the payload's
   `static_constraints` is the *candidate author's* self-imposed budget. Collapsing them removes the ability
   to mint a candidate that promises to fit in less than the device allows, which is a legitimate search
   constraint. The redundancy costs one comparison.

6. **T before F.** `target-fact-type-refactor-scope-20260801.md` §8 says to do F0 only and wait while the
   table's shape is being decided; this scope *is* that decision, and T2 adds roughly a dozen fields to the
   row. Lifting into `Fact[T]` first means lifting a row that is about to change. Sequence T1-T4, then F3.
   F0 is inert and can land at any point.

### Withdrawn finding

An earlier review round flagged `_instruction_family_for`'s `wmma_` prefix (`runtime_specs.py:101`) as an
AMD mnemonic leaking into NV/Metal rows. **That was wrong and was never acted on.** `TensorCore.__str__`
(`tinygrad/codegen/opt/tc.py:96`) names every descriptor `WMMA_*` regardless of vendor -- `cuda_81616`
stringifies to `WMMA_8_16_16_half_float` in upstream tinygrad. The prefix follows upstream's own
cross-target convention. Recorded here so it does not resurface as a finding in a later pass.
