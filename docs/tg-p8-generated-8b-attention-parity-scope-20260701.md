# TG-P8 Scope: Generated 8B Decode Attention Parity

Date: 2026-07-01.

Goal: replace the last default `external_handwritten_kernel`, `decode_attention_owned_two_kernel`, with a generated/search-owned 8B decode-attention route **only if** the generated route is correct, route-bound, and not slower than the owned HIP route. This is the remaining blocker to `TINYGRAD_DEFAULT_PURITY_PASS`.

This is not a route-policy plumbing task. TG-P2 through TG-P7 already proved that BoltBeam can emit policy and tinygrad can consume it. TG-P8 is a codegen/search-quality task: make generated 8B attention competitive with the owned two-kernel route, or produce a precise blocker.

## Current State

TG-P7 terminal state:

- `bench/tg-p7-pure-search-default/summary.md`: `TG_P7_BLOCKED_PURITY_DEBT_REMAINING`.
- `bench/pure-machine-search-default-path-census/summary.md`: `TINYGRAD_DEFAULT_PURITY_FAIL`.
- Default hot routes: 4 of 5 are `machine_authored_generated`; the sole final purity debt is `decode_attention_owned_two_kernel`.

TG-P5 already tested the obvious generated replacement:

- `bench/tg-p5-attention-generated-default/latest.json`: `TG_P5_REFUTE_GENERATED_ATTENTION_SLOWER`.
- Geometry: `Hq=32`, `Hkv=8`, `Hd=128`, `G=4`.
- Candidate route: generated G5/G-block-tile generalized to 8B.
- Result:

| ctx | owned tok/s | generated tok/s | generated / owned |
|---:|---:|---:|---:|
| 512 | 107.8 | 94.4 | 87.6% |
| 4096 | 97.9 | 93.6 | 95.6% |

TG-P5 proved the candidate is:

- token-identical;
- route-bound;
- generated UOp, not external HIP/ASM;
- resource-sane enough to run;
- slower than owned, so not promotable.

TG-P6 guard:

- `bench/tg-p6-pure-search-diagnostic/summary.md`: `PURE_MACHINE_SEARCH_ONLY=1` fails the normal default because attention is still external-handwritten, and passes only when generated attention is forced.

TG-P7 strict gate:

- `bench/tg-p7-pure-search-default/final_census.json`: strict final default purity fails only on `decode_attention_owned_two_kernel`.

## Code Sites

Primary tinygrad sites:

- `tinygrad/llm/model.py`
  - `_SUPPORTED_QK_ROUTE_IDS` and `_load_qk_route_policy` route-policy validation.
  - generated G5/G4 attention dispatch around the `DECODE_FLASH_BLOCK_TILE_G5_8B` branch.
  - owned attention dispatch via `DECODE_ATTN_AMDGCN_TILE`.
  - `PURE_MACHINE_SEARCH_ONLY` model-init guard.
- `extra/qk_flash_decode.py`
  - generated flash routes, including `flash_decode_g5_block_tile` and `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`.
- `extra/qk_owned_flash_decode_graph_node.py`
  - owned route oracle, not a template to copy.
- `extra/pure_machine_search_default_path_census.py`
  - final purity census and route attribution.
- `extra/qk_route_manifest.py`
  - route provenance, default/fallback classification, and replacement-scope metadata.

Primary BoltBeam sites:

- `/home/ubuntu/BoltBeam/boltbeam/data/candidates.json`
  - current candidate ledger and refuted/reopen conditions.
- `/home/ubuntu/BoltBeam/boltbeam/policy/emit.py`
  - route-policy emission.
- `/home/ubuntu/BoltBeam/tests/test_policy_guard.py`
  - route-policy and purity tests.

## Non-Negotiable Guardrails

1. No handwritten kernel.
   - Do not add HIP/ASM/inline ISA as the candidate implementation.
   - Do not copy owned HIP logic into a new external kernel.
   - Generated UOp or tinygrad scheduler/codegen output only.

2. Do not force purity with a slowdown.
   - Owned HIP stays default unless generated attention is not slower on protected contexts.
   - The TG-P5 generated route is already correct-but-slower; do not promote it as-is.

3. Do not re-chase refuted attention-combine paths.
   - `docs/attention-combine-reachability-audit-20260701.md`
   - `/home/ubuntu/BoltBeam/docs/attention-combine-closure-stress-test-20260701.md`
   - Hq-only fused combine, split-preserving merge that collapses `Hq*Hd`, FLASH_L regressions, and 14B wholecache regressions are ledgered. Reopen only with a new mechanism that preserves the required parallelism or introduces a genuinely new coordination primitive.

4. Owned route is an oracle, not a default target to delete.
   - Keep rollback/oracle flags.
   - Use owned route for correctness and W==D comparison.

5. BoltBeam judges promotion.
   - If a candidate passes, add/update the BoltBeam candidate and emit route policy.
   - If it fails, ledger the exact blocker with a reopen condition.

## Working Hypothesis

The G4 generated route loses to owned at ctx512 because it does not match the owned route's effective work decomposition and/or scheduling quality.

TG-P5 says the blocker is not correctness, route binding, or basic resources. The next pass must identify the concrete delta:

- per-kernel wall split: generated tile vs generated gmax/combine vs owned tile vs owned combine;
- launch count and workgroup geometry;
- LDS usage, VGPR count, scratch bytes;
- static instruction count by category;
- memory traffic and redundant K/V reads;
- occupancy-sensitive parallelism at `Hq=32`, `Hkv=8`, `G=4`, `Hd=128`;
- whether generated G4 can use a better split/staging geometry than the 14B-derived G5 K-only path.

## Phase TG-P8.0: Evidence Refresh

Purpose: make the baseline undeniable before changing code.

Run or create a small authority tool that writes:

- `bench/tg-p8-generated-8b-attention-parity/baseline.json`
- `bench/tg-p8-generated-8b-attention-parity/summary.md`

Measure at minimum:

- ctx512 and ctx4096;
- owned default;
- current generated G4 candidate (`DECODE_FLASH_BLOCK_TILE_G5_8B=1`, K-only);
- optional current generated wholecache fallback (`DECODE_ATTN_AMDGCN_TILE=0` without G4 candidate), if cheap.

Required fields:

- token/logit equivalence;
- route-bound kernels;
- tok/s and wall ms;
- attention-kernel wall split;
- kernel names and counts;
- selected env / route policy.

Verdicts:

- `TG_P8_0_PASS_BASELINE_PINNED`
- `TG_P8_0_BLOCKED_AUTHORITY_MISSING`
- `TG_P8_0_BLOCKED_ROUTE_ATTRIBUTION`

Stop if route attribution is not clean.

## Phase TG-P8.1: Owned-vs-Generated Delta Audit

Purpose: decide what can be searched.

Use static and runtime metadata to compare:

- owned two-kernel route:
  - `owned_flash_tile_gqa_whole`
  - `owned_flash_combine`
- generated G4 route:
  - `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`
  - `flash_state_gmax_32_128`
  - `flash_state_combine_32_128`

Capture:

- VGPR count;
- scratch/private bytes;
- LDS bytes;
- static instruction count;
- barrier count;
- global-load/store count;
- DS/LDS instruction count;
- workgroups launched per context;
- estimated bytes read/written;
- per-kernel wall share.

Write:

- `bench/tg-p8-generated-8b-attention-parity/delta_audit.json`

Classify the delta into one primary class:

| class | meaning | next |
|---|---|---|
| `SPLIT_GEOMETRY_MISMATCH` | generated route launches too little/much or wrong split shape | TG-P8.2 geometry search |
| `COMBINE_OVERHEAD` | generated extra gmax/combine dominates vs owned | TG-P8.3 combine lifecycle |
| `RESOURCE_PRESSURE` | VGPR/LDS/scratch/occupancy gap is decisive | TG-P8.4 resource variant |
| `MEMORY_TRAFFIC` | redundant K/V traffic or materialization dominates | TG-P8.5 memory-layout variant |
| `INSTRUCTION_SCHEDULING` | static instruction bloat or schedule/waitcnt gap dominates | TG-P8.6 codegen primitive |
| `LOW_AMDAHL_OR_AT_CEILING` | route is close enough or gap cannot move whole model | stop/defer |

Verdicts:

- `TG_P8_1_PASS_DELTA_CLASSIFIED`
- `TG_P8_1_BLOCKED_METADATA_MISSING`
- `TG_P8_1_BLOCKED_MULTI_CAUSE_UNRESOLVED`

Do not implement a candidate until TG-P8.1 selects a class.

## Phase TG-P8.2: Geometry Search

Run only if TG-P8.1 selects `SPLIT_GEOMETRY_MISMATCH`.

Search generated parameters, not handwritten kernels:

- `L` / split length;
- `TK`;
- `WARPS=G` vs alternate legal workgroup shapes if expressible;
- K-only vs KV-both staging only as a controlled comparison;
- whether combine split dimensions can preserve both `Hq*S` partial parallelism and enough output-d parallelism.

Use bounded enumeration. Abort if candidate count explodes.

Acceptance:

- every candidate route-bound;
- no hidden owned fallback;
- correctness first;
- W==D only for candidates with clean route binding.

Verdicts:

- `TG_P8_2_PASS_GEOMETRY_CANDIDATE_SELECTED`
- `TG_P8_2_REFUTE_GEOMETRY_SPACE`
- `TG_P8_2_BLOCKED_SEARCH_SPACE_EXPLOSION`

## Phase TG-P8.3: Combine Lifecycle

Run only if TG-P8.1 selects `COMBINE_OVERHEAD`.

Do not repeat refuted 14B combine collapses. The valid question is specific to 8B G4:

- Can generated attention match owned's two-kernel lifecycle without external HIP?
- Can it reduce from three generated kernels to two while preserving the parallelism that mattered?
- Can it avoid collapsing combine work from `Hq*Hd` to `Hq`?

Any candidate must explicitly report workgroup geometry for partial and combine phases.

Verdicts:

- `TG_P8_3_PASS_COMBINE_CANDIDATE_SELECTED`
- `TG_P8_3_REFUTE_COMBINE_LIFECYCLE`
- `TG_P8_3_BLOCKED_PRIMITIVE_MISSING`

## Phase TG-P8.4: Resource Variant

Run only if TG-P8.1 selects `RESOURCE_PRESSURE`.

Possible generated variants:

- reduce live accumulator state;
- narrower LDS footprint;
- lower register-pressure schedule;
- reduce per-head state kept across G heads;
- alternative staging that improves occupancy without bringing back known V-staging bloat.

Acceptance:

- no scratch spill;
- VGPR/LDS improvement shown in resource artifact;
- token/logit equivalent.

Verdicts:

- `TG_P8_4_PASS_RESOURCE_CANDIDATE_SELECTED`
- `TG_P8_4_REFUTE_RESOURCE_VARIANTS`
- `TG_P8_4_BLOCKED_REGALLOC_OR_SCHEDULER`

## Phase TG-P8.5: Memory-Traffic Variant

Run only if TG-P8.1 selects `MEMORY_TRAFFIC`.

Investigate:

- whether generated G4 rereads V/K more than owned;
- whether K-only staging is suboptimal for G4 despite helping G5;
- whether assigned-KV slice/materialization differences explain the ctx512 loss;
- whether the owned route benefits from a cache-warming side effect that the generated route lacks.

Acceptance:

- measured drop in traffic proxy or per-kernel wall;
- no refuted bypass of beneficial cache warm unless replaced by a measured equivalent.

Verdicts:

- `TG_P8_5_PASS_MEMORY_CANDIDATE_SELECTED`
- `TG_P8_5_REFUTE_MEMORY_VARIANTS`
- `TG_P8_5_BLOCKED_MEMORY_ORACLE`

## Phase TG-P8.6: Codegen Primitive

Run only if TG-P8.1 selects `INSTRUCTION_SCHEDULING`.

This is the deepest path. The output should be a generic codegen primitive or lowering improvement, not an 8B special-case.

Candidate examples:

- better local-memory staging lowering;
- better wait/barrier placement for generated attention;
- better schedule for online softmax state;
- better vectorized load lowering in the attention tile;
- generic route-spec parameterization that lets search author owned-like decomposition.

Acceptance:

- generic primitive name and tests;
- applies to at least one microgate beyond the exact 8B shape, or explains why the shape is the minimal legal proof;
- no inline HIP/ASM.

Verdicts:

- `TG_P8_6_PASS_GENERIC_PRIMITIVE`
- `TG_P8_6_BLOCKED_CODEGEN_CAPABILITY`
- `TG_P8_6_REFUTE_PRIMITIVE_NO_WD_MOVEMENT`

## Phase TG-P8.7: Promotion Gate

Run only after TG-P8.2/3/4/5/6 selects a candidate.

Protected contexts:

- ctx512;
- ctx4096.

Optional if cheap:

- ctx128 to ensure short-context threshold behavior unchanged;
- ctx1024/2048 for slope sanity.

Promotion requirements:

1. token/logit equivalent to owned;
2. route-bound generated attention;
3. no hidden owned fallback;
4. generated route is at least 98% of owned at every protected context;
5. no protected-context regression >1%;
6. rollback to owned remains available;
7. BoltBeam candidate and route policy updated;
8. `PURE_MACHINE_SEARCH_ONLY=1` passes without forcing the slower TG-P5 route;
9. `extra/pure_machine_search_default_path_census.py --strict-final-default` passes.

Verdicts:

- `TG_P8_7_PASS_GENERATED_ATTENTION_PROMOTED`
- `TG_P8_7_REFUTE_CANDIDATE_SLOWER`
- `TG_P8_7_BLOCKED_ROUTE_ATTRIBUTION`
- `TG_P8_7_BLOCKED_POLICY_OR_CENSUS`

## Phase TG-P8.8: Ledger and Handoff

Always run after TG-P8.7 or any blocker/refutation.

Update:

- tinygrad docs and bench artifacts;
- BoltBeam candidate manifest;
- route policy emitter if promoted;
- default-path census;
- pure-search guard results.

If blocked/refuted, record:

- exact blocker class;
- reopen condition;
- do-not-retry axis tags;
- whether owned remains default.

Verdicts:

- `TG_P8_8_PASS_LEDGER_UPDATED`
- `TG_P8_8_BLOCKED_LEDGER_DRIFT`

## Expected Outcomes

Best case:

- generated 8B attention reaches >=98% of owned at ctx512 and ctx4096;
- owned HIP becomes rollback/oracle;
- strict default purity passes;
- tinygrad reaches `TINYGRAD_DEFAULT_PURITY_PASS`.

Honest likely case:

- TG-P8.1 identifies a scheduling or lifecycle delta;
- a bounded generated candidate improves but may still not clear owned;
- owned remains default with a sharper blocker than TG-P5.

Bad case:

- evidence shows the owned two-kernel route relies on a backend coordination/scheduling feature not currently expressible by tinygrad generated UOps;
- classify as `EMITTER_BLOCKED` or `PRIMITIVE_MISSING`, not as a generic failure.

## Claude Execution Prompt

Use this exact task framing:

> You are continuing the tinygrad pure-machine-search migration at TG-P8. The only remaining final-default purity debt is 8B decode attention: `decode_attention_owned_two_kernel`. TG-P5 already proved the generated G4/G5 block-tile replacement is correct and route-bound but slower (87.6% of owned at ctx512, 95.6% at ctx4096), so do not promote it as-is and do not re-chase refuted attention-combine paths. Start with TG-P8.0 evidence refresh and TG-P8.1 owned-vs-generated delta audit. Do not implement a new route until the delta is classified. Any implementation must be generated UOp/tinygrad codegen only, no handwritten HIP/ASM. Promotion requires token/logit equivalence, route-bound generated attention, >=98% of owned at ctx512 and ctx4096, rollback to owned, BoltBeam ledger/policy update, and strict final-default purity pass. If it cannot pass, produce a precise blocker and reopen condition.

