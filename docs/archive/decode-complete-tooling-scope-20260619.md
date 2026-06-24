# Decode Complete Tooling Scope

Date: 2026-06-19

## Purpose

Build the complete decode tooling layer needed to explain and price tinygrad's in-model MMVQ lifecycle loss.

The target question is narrow:

```text
Why does tinygrad retain ~44% HBM in-model when its standalone MMVQ surface can reach ~76%, while llama retains ~54%?
```

The tooling must separate four things that are currently easy to conflate:

1. **program identity** - which kernel object actually launched;
2. **body attribution** - whether the kernel body is visible and matches the expected primitive surface;
3. **lifecycle accounting** - main kernel, reduce, glue, activation producer, scheduling boundaries;
4. **timing authority** - how much each lifecycle component costs in role-local and W==D decode terms.

ATT now solves visibility. It does not solve timing by itself.

## Current Instrument Inventory

| Layer | Current status | Authority level | Notes |
|---|---|---:|---|
| HCQ attribution ledger | working | high for program identity | Captures eager launches, HCQGraph construction/replay, program names, launch shapes, and graph boundaries. |
| ROCprofiler ATT through HCQ | working | high for body visibility | AQLprofile v2 packets replay through tinygrad HCQ and produce decodable SQTT/ATT body packets. |
| Role-joined ATT, Q4_K | working | high for one real role | `blk.0.attn_output` uses `q4k_coop_partial_4096_4096` plus reduce/glue in-model. |
| Role-joined ATT, Q6_K | partial | high for surface, not full model activation | `ffn_down` and `lm_head` launch intended Q6_K coop surfaces through `q6_surface_fallback`; full model activation capture hit a 4.68 GB AMD allocation failure. |
| Native PMC/roofline | working but coarse | high for aggregate, weak for short kernels | Good for model-level HBM/VALU classification; per-kernel timing remains fragile. |
| llama HIP trace/capture | working | high for comparison contracts | Separate HIP-only process can capture llama MMVQ grids/kernargs and rocprofiler traces. |
| Primitive ledger | working | high for provenance/search memory | Reconstructs verdicts and prevents stale refutations from re-entering as live routes. |

## Missing Tooling

The missing tooling is not "can we see kernels?" We can. The missing tooling is a complete, repeatable join from
decode role -> launched programs -> ATT bodies -> lifecycle components -> timing/Amdahl -> decision gate.

| Gap | Why it matters | Required output |
|---|---|---|
| full-model Q6 activation capture | The Q6 ATT result validates the surface, but not the full activation path due to model-load memory failure. | Either full in-model Q6 role capture, or a documented no-copy/minimal-loader fallback accepted as equivalent by program identity + activation shape. |
| reliable role timing | ATT packet counts are not timing. ProfileGraphEvent/short-kernel timings can be stale or perturbed. | A timing authority for role-local main/reduce/glue costs, or a bounded A/B surrogate with explicit caveats. |
| reduce/glue Amdahl ledger | Q4/Q6 roles visibly pay separate reduce/glue, but the model-level cost is unpriced. | Per-role main/reduce/glue share and projected W==D movement. |
| ATT metric extraction | We currently record packet classes, but not a standardized per-role metric table. | Body packets, `VALUINST`, `INST`, wave starts/ends, trace bytes, program hash, grid/local, and role label in one schema. |
| cross-role lifecycle atlas | Decode roles are still split across individual docs/artifacts. | One role matrix covering Q4_K/Q6_K high-share roles and any small roles kept for completeness. |
| llama comparison join | llama's lifecycle is understood conceptually, but not joined into the same role schema. | Side-by-side tinygrad vs llama row for comparable MMVQ surfaces: activation format, grid/local, VGPR if known, body metrics, timing/throughput if reliable. |

## Non-Goals

- Do not build direct-output/reduce fusion until the reduce/glue ledger clears the movement gate.
- Do not infer speed from ATT packet counts alone.
- Do not start another standalone kernel search from this tooling phase.
- Do not change model defaults or route flags.
- Do not reopen q8 quality work unless lifecycle accounting shows q8 is still a top remaining decode lever.

## Role Set

Minimum role matrix:

| Role group | Weight format | Why included |
|---|---|---|
| `attn_q/o` | Q4_K | Existing Q4 role-joined ATT; smaller but clean full in-model proof. |
| `ffn_gate/up` | Q4_K | High-share Q4 roles; sibling activation reuse candidate. |
| `ffn_down` | Q6_K | High-share Q6 role; Stage 1 surface ATT passed. |
| `lm_head` | Q6_K | Huge output projection; Stage 1 surface ATT passed. |
| `attn_k/v` | Q6_K or role-specific | Include only if share justifies the extra harness work. |
| attention/KV | non-MMVQ | Track separately for long-context slope; not part of the base MMVQ retention gate. |

## Phase DCT-0 - Tooling Source of Truth

Goal: freeze the instrumentation contract before more probes are written.

Work:

- collect current HCQ, ATT, PMC, llama-trace, and primitive-ledger artifacts into one index;
- define the role-level JSON schema;
- tag each field with authority:
  - `measured`;
  - `inferred`;
  - `surface_fallback`;
  - `unsupported`;
  - `not_timing_authority`.

Deliverable:

- `bench/qk-decode-complete-tooling/schema.json`
- `bench/qk-decode-complete-tooling/instrument_inventory.json`

Pass:

- every existing Q4/Q6 ATT artifact can be represented without losing provenance.

## Phase DCT-1 - Full-Model Role Harness Fix

Goal: remove or explicitly retire the 4.68 GB AMD allocation blocker for Q6 role capture.

Work options, in order:

1. avoid copying full GGUF metadata/storage to AMD during model construction;
2. build a minimal block/role loader that keeps non-target tensors on CPU and only materializes the target role;
3. reuse the existing `q6_surface_fallback`, but add a formal equivalence proof:
   - same primitive class;
   - same weight storage;
   - same program identity;
   - same activation shape and dtype;
   - same decode flags.

Deliverable:

- `bench/qk-decode-complete-tooling/q6_capture_equivalence.json`

Pass:

- full in-model Q6 capture works, or the fallback is explicitly accepted as role-surface equivalent.

Kill:

- if neither full capture nor equivalence can be proven, Q6 timing must be labeled `surface_only` and cannot drive a
  build gate alone.

## Phase DCT-2 - Unified HCQ + ATT Role Runner

Goal: make one runner produce comparable role rows for all high-share decode roles.

Work:

- extend the existing role-join probe into a multi-role runner;
- wrap each role interval with:
  - ATT start/stop;
  - HCQ program capture;
  - graph/eager mode tag;
  - decode primitive flags;
  - artifact hashes.

Artifacts:

- `bench/qk-decode-complete-tooling/roles/{role}.json`
- `bench/qk-decode-complete-tooling/role_atlas.json`

Pass:

- each high-share role has:
  - main primitive program;
  - reduce/glue programs;
  - launch geometry;
  - ATT body packets > 0 for the main program or an explicit unsupported reason.

## Phase DCT-3 - Timing Authority

Goal: produce defensible role-local timing without trusting ATT as a timer.

Preferred order:

1. same-process interleaved role-local A/B where only the role lifecycle changes;
2. HCQ graph timestamp validation against repeated eager timing for the same role;
3. model-level W==D A/B if role-local timing is not stable;
4. coarse Amdahl inference from aggregate model timing only as a last resort.

Work:

- run a timing audit on known stable cases and known bad cases;
- compare device events, host wall, graph replay, and PMC aggregate where possible;
- label timing source per role.

Artifacts:

- `bench/qk-decode-complete-tooling/timing_audit.json`
- `bench/qk-decode-complete-tooling/timing_policy.md`

Pass:

- the timing policy can distinguish a real `>=5%` W==D projected win from measurement noise.

Kill:

- if role timing remains unstable, no implementation phase can be funded from role-local measurements; only full W==D
  A/B can promote changes.

## Phase DCT-4 - ATT Metric Extractor

Goal: turn decoded ATT output into role-level resource rows.

Metrics:

- body-like packet count;
- `VALUINST`;
- `INST`;
- wave start/end count;
- nonzero trace bytes;
- optional wave duration / occupancy if the decoded format supports it reliably;
- program descriptor hash and code-object hash;
- grid/local and target CU/SIMD settings.

Artifact:

- `bench/qk-decode-complete-tooling/att_metrics.json`

Pass:

- Q4 full in-model, Q6 surface, and one standalone surface can be compared in the same table.

Boundary:

- ATT metrics are explanatory. They can justify a hypothesis, but not prove a speedup.

## Phase DCT-5 - Reduce/Glue Amdahl Ledger

Goal: price the visible lifecycle tax.

Work:

- classify each role interval into:
  - main MMVQ;
  - reduce;
  - elementwise/glue;
  - activation producer;
  - layout/reshape;
  - scheduler gap if observable;
- attach timing authority from DCT-3;
- compute local role movement and W==D projected movement for:
  - remove glue only;
  - fuse reduce into main;
  - direct output;
  - q8 producer reuse;
  - combined lifecycle candidate.

Artifact:

- `bench/qk-decode-complete-tooling/reduce_glue_ledger.json`

Build gate:

- a single role or shared primitive must project `>=5%` W==D movement, or `>=10%` local role movement on a high-share
  role.

Kill:

- if reduce/glue totals low single-digit W==D movement, do not build direct-output/reduce-fusion.

## Phase DCT-6 - llama Comparison Importer

Goal: place llama's MMVQ lifecycle into the same schema.

Work:

- ingest existing llama launch capture rows;
- add comparable fields:
  - q8 activation producer;
  - consumer kernel symbol;
  - grid/local;
  - kernarg size;
  - role shape;
  - throughput/timing where reliable;
  - trace metadata if available.

Artifact:

- `bench/qk-decode-complete-tooling/llama_join.json`

Pass:

- at least one Q4_K and one Q6_K llama role can be compared to tinygrad in the same role schema.

## Phase DCT-7 - Complete Decode Lifecycle Atlas

Goal: produce the decision document for the next implementation move.

Output:

- role matrix;
- lifecycle cost matrix;
- ATT/resource matrix;
- timing authority labels;
- tinygrad vs llama comparison;
- open/closed primitive rows;
- build/no-build decisions.

Artifacts:

- `bench/qk-decode-complete-tooling/summary.md`
- `docs/decode-complete-tooling-result-20260619.md`

Completion criteria:

| Question | Required answer |
|---|---|
| Are all high-share roles using intended native primitives? | yes/no, with program evidence |
| What is the reduce/glue tax? | role-local and W==D Amdahl |
| Is the `76% -> 44%` loss attributable to a bounded lifecycle tax? | yes/no |
| Is there a buildable next primitive? | direct-output/reduce fusion, q8 lifecycle, scheduler/resource, or close |
| Can the next build be promoted without weak timing? | timing authority label and gate |

## Expected Outcomes

Likely outcomes, based on current evidence:

1. **Runtime/cache identity remains closed.** Q4 and Q6 already launch the intended primitives.
2. **Reduce/glue is real but may be below the W==D build gate.** This must be priced before any fusion work.
3. **If no bounded lifecycle tax clears, the remaining gap is scheduler/resource contract preservation.** That means
   project-level AMD backend work, not another small kernel patch.
4. **q8 remains a research route, not the main parity route, unless the ledger shows `ffn_gate/up` dominates the
   remaining priced gap.**

## Immediate Execution Order

Run the tooling in this order:

1. DCT-0 schema/inventory.
2. DCT-1 Q6 capture equivalence.
3. DCT-2 multi-role HCQ+ATT atlas.
4. DCT-3 timing policy.
5. DCT-5 reduce/glue Amdahl ledger.
6. DCT-6 llama join.
7. DCT-7 final lifecycle atlas.

DCT-4 can run in parallel with DCT-2 after the role artifacts exist.

