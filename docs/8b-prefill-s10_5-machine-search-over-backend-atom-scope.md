# 8B Prefill S10.5 Machine Search Over Backend Atom Scope

Date: 2026-07-10.

## Decision

S10.5 accepts the hybrid boundary:

```text
ffn_gate_up = machine-searched compiler primitive + hand-coded reusable DBUF backend atom
```

Classification:

```text
compiler_primitive_spec_owned__asm_backend_atom
```

This is a hybrid compiler primitive route with a hand-coded backend atom. It is not pure generated code, because the DBUF
epoch coordinator/prologue/body/tail atom remains hand-coded. It is not a full fine-tuned hand kernel, because the full
role lifecycle must not be an opaque role-specific instruction list selected by a flag; searchable compiler/spec metadata
owns the safe schedule, route, legality, and wait-policy decisions around the atom.

The purpose of S10.5 is to move every safe decision around the `ffn_gate_up` backend atom into searchable/spec-owned
metadata while preserving the S9 4k pp512 band.

## Current Facts

| Path | Route | Clock | pp512 | pp4096 | Meaning |
|---|---|---:|---:|---:|---|
| S9 authority | `prefill_pipe_role_selective_generated` | pinned | `~4413` | `~3237` | fast hybrid/raw backend atom baseline |
| S9 authority | `prefill_pipe_role_selective_generated` | unpinned | `~5111` | `~3677` | same route under boost clocks |
| S10 generated composed | `prefill_wmma_pipe_lds_dbuf_primitive_generated` | pinned | `~1332` | `~1189` | correct route, slow generated transport |
| S10 generated composed | `prefill_wmma_pipe_lds_dbuf_primitive_generated` | unpinned | `~1514` | `~1341` | not a clock regression, still structurally slow |

So S10.5 must not promote the slow generated LDS/DBUF transport. It keeps the fast backend atom and makes the surrounding
contract machine-owned.

## Non-Goals And Prior Work Boundary

S10.5 is not a restart of the older generated LDS/DBUF route. Do not duplicate or reopen the prior generated transport
work unless a candidate first proves that it can match the backend atom contract and authority timing gates in this doc.

Out of scope for S10.5:

- copying the S10 generated composed LDS/DBUF transport into a second route,
- adding another generated postrange DBUF stage-movement experiment,
- replacing the backend atom's hard epoch choreography before checker coverage can compare generated output against the
  atom contract,
- creating a new benchmark or trace harness when an existing one below already covers the gate,
- broad 4x4 DBUF shape work, which remains parked separately on gfx1100 VGPR pressure.

## Ownership Boundary

| Surface | S10.5 owner | Classification |
|---|---|---|
| role detection, `ffn_gate_up = 512x12288x4096` | compiler/spec | machine-owned |
| route policy, `ffn_gate_up -> LDS2/DBUF` | compiler/search | machine-owned |
| tile shape, waves, `BK`, `PAD`, `DBUF`, `PLRA/B`, `LEANADDR` | spec/search | machine-search-owned |
| LDS byte windows and A/B layout keys | `WMMALDSSpec` + checker | machine-owned proof |
| wait policy | spec/search | machine-search-owned |
| DBUF epoch template identity | `DBUFEpochPrimitive` metadata | machine-owned contract |
| prologue/body/tail instruction choreography | backend atom | hand-coded reusable primitive |
| physical registers and instruction cadence inside atom | backend atom | hand-coded primitive |

The path becomes a fine-tuned hand kernel again if S10.5 hides role-specific shape, fixed registers, output epilogue, or
full instruction-stream decisions inside a new opaque emitter. The path remains acceptable if the hand-coded part is a
small reusable atom parameterized by the spec.

## Searchable Knobs

Initial safe search space:

| Knob | Source today | Candidate values | Gate |
|---|---|---|---|
| `wait.lgkm_after_coop_store` | S9 wait search | default, `2` | authority throughput and correctness |
| `wait.lgkm_after_frag_load` | S9 wait search | default, `2` | authority throughput and correctness |
| `pad` | schedule/spec | current legal values only | `WMMALDSSpec.legality_errors == []` |
| `bk/tile_k` | schedule/spec | legal divisors already supported by backend atom | LDS <= 64 KiB, no perf regression |
| `wm/wn/waves_m/waves_n` | schedule/spec | legal extracted S9 candidates | accum/temp VGPR budget and authority timing |
| `plra/plrab` | schedule/spec | existing supported modes | legality + authority timing |
| `leanaddr` | schedule/spec | existing supported modes | legality + authority timing |
| `dbuf_epoch_primitive.nbuf` | metadata only for now | `2` only | no execution change |
| `dbuf_epoch_primitive.slot_expr` | metadata only for now | `epoch % 2` only | checker P1-P2/P5 |

Deferred search space:

| Knob | Why deferred |
|---|---|
| generated postrange DBUF stage movement | previous S10 generated transport was correct but slow and wait-heavy |
| replacing backend atom prologue/body/tail | this is the hard epoch choreography; keep it until checker P1-P8 can compare generated vs atom |
| 4x4 DBUF shape | parked on gfx1100 VGPR pressure |

## Existing Harnesses To Reuse

No new benchmark harness should be created.

| Need | Existing tool |
|---|---|
| E2E authority timing | `extra/qk/prefill_whole_synced.py` |
| S9 report aggregation | `extra/qk/prefill/lds2_s9_report.py` |
| S9 search artifacts | `extra/qk/prefill/lds2_s9_*_search.py` |
| route/lifecycle trace | `extra/qk/prefill/s10_hybrid_role_trace.py` |
| baseline freeze | `extra/qk/prefill/s10_baseline_freeze.py` |
| DBUF lifecycle proof | `extra/qk/prefill/dbuf_epoch_lifecycle_checker.py` |
| LDS spec exporter | `extra/qk/prefill/dbuf_s10_lds_spec_exporter.py` |
| surface classification | `extra/qk/pure_kernel_surface_audit.py`, `extra/qk/pure_search_guard.py`, `extra/qk/route_manifest.py` |

## Done Gates

S10.5 is done only when all gates pass:

| Gate | Required result |
|---|---|
| classification | route is recorded as `compiler_primitive_spec_owned__asm_backend_atom`; not pure generated; not full hand-kernel ownership |
| candidate schema | every promoted candidate serializes `PrefillGEMMScheduleSpec`, `WMMALDSSpec`, `DBUFEpochPrimitive`, wait policy, selected backend atom, and expected classification |
| legality/proof | `WMMALDSSpec` legality passes, slot identity proves active_buffers=2, DBUF lifecycle P1 passes, and P5 wait proof passes when explicit waits are present |
| harness reuse | timing, traces, reports, baseline freeze, lifecycle proof, LDS export, and classification use the existing harnesses listed above |
| authority timing | pinned `pp512 >= 4000` through `extra/qk/prefill_whole_synced.py --mode authority --pin-clock` |
| route binding | S10.5 report confirms `route_attribution.prefill_route_family == prefill_pipe_role_selective_generated`; the generic pure-route binding gate is expected to reject this hybrid route |
| promotion report | final report says either `S10_5_HYBRID_SEARCH_OWNED_BACKEND_ATOM_READY` or `S10_5_HYBRID_SEARCH_BLOCKED_WITH_EXACT_REASON` |

Default policy stays unchanged until a candidate satisfies every gate and matches or beats the current S9 authority band.

## Phases

### P0. Scope And Classification

Done means this doc exists and the route is described as:

```text
compiler_primitive_spec_owned__asm_backend_atom
```

This explicitly means hybrid compiler primitive plus hand-coded backend atom; it does not mean pure generated, and it does
not mean full fine-tuned hand-kernel ownership.

### P1. Spec Candidate Rows

Add a machine-readable candidate record for `ffn_gate_up` that serializes:

```text
PrefillGEMMScheduleSpec
WMMALDSSpec
DBUFEpochPrimitive
wait policy
selected backend atom
expected classification
```

Output target:

```text
bench/prefill-s10_5-machine-search/ffn-gate-up-candidates.json
```

### P2. Search Runner

Build a small runner over existing S9-safe knobs. It may call existing S9 search helpers, but its output must be S10.5
candidate rows, not only env strings.

Output target:

```text
bench/prefill-s10_5-machine-search/search-report.json
```

Done means every candidate states:

```text
spec_json
env_overrides
legality
checker_status
expected_backend_atom
promotion_status
```

### P3. Checker Gate

For every candidate, run existing proof layers that do not require generated replacement:

```text
WMMALDSSpec legality
slot identity proof with active_buffers=2
DBUFEpochPrimitive P1 lifecycle proof
P5 wait proof when explicit waits are present
```

Candidates that cannot prove these are rejected before timing.

### P4. Authority Timing Gate

Use only the existing authority harness:

```bash
PYTHONPATH=. DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 \
python3 extra/qk/prefill_whole_synced.py \
  --mode authority -K 8 --warmups 4 --rounds 3 --pin-clock \
  --artifact bench/prefill-s10_5-machine-search/<candidate>-authority.json \
  --json
```

Promotion gate:

```text
pinned pp512 >= 4000
route_attribution.prefill_route_family == prefill_pipe_role_selective_generated
classification remains hybrid/backend-atom
```

Do not compare pinned candidates to unpinned baselines.

Do not use the generic `--require-route` binding gate as the S10.5 acceptance gate. That gate enforces pure/generated
route provenance and correctly rejects `prefill_pipe_role_selective_generated` as external/hybrid. S10.5 instead records
that rejection and applies its own hybrid route/performance gate in:

```text
extra/qk/prefill/s10_5_machine_search.py --authority-artifact ...
```

### P5. Report And Default Policy

Write a final report that says one of:

```text
S10_5_HYBRID_SEARCH_OWNED_BACKEND_ATOM_READY
S10_5_HYBRID_SEARCH_BLOCKED_WITH_EXACT_REASON
```

Default policy remains unchanged unless a candidate passes P1-P4 and matches or beats the current S9 authority band.

## Parallel Work

| Lane | Can run parallel? | Files |
|---|---:|---|
| candidate serializer | yes | `extra/qk/prefill/s10_5_machine_search.py`, tests |
| checker integration | yes | `extra/qk/prefill/s10_5_machine_search.py`, DBUF checker tests |
| classification audit | yes | docs/manifest/audit only |
| authority timing | sequence after candidates pass legality | benchmark artifacts |

## Stop Conditions

Stop and report if:

- a candidate cannot be represented without raw shape-specific instruction-list ownership,
- checker P1/P2 fails for the backend atom metadata,
- authority route falls off `prefill_pipe_role_selective_generated`,
- pinned pp512 drops below `4000`,
- or repeated candidates only change metadata while producing identical timing and no new ownership boundary.
