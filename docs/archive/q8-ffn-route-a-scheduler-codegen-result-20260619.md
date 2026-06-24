# q8 FFN Route A scheduler/codegen result (2026-06-19)

Executed Route A A0/A1 from `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md`.

Verdict:

- **A0 schedule contract extraction: PASS**
- **A1 AMD DSL capability map: FAIL_A1_NO_BOUNDED_FEATURE**

Decision: **do not start A2 for q8 decode**. Route A remains a project-level AMD scheduler/codegen roadmap item, not a
bounded q8 primitive follow-up.

## Artifacts

Probe:

- `extra/q8_ffn_route_a_schedule_contract.py`

Outputs:

- `bench/q8-ffn-amd-scheduler-project/oracle_contract.json`
- `bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json`
- `bench/q8-ffn-amd-scheduler-project/route_a_result.json`

## A0 — schedule contract extraction

A0 normalized the hipcc/LLD `q8_mmvq_gateup` oracle against COMGR and tinygrad AMD DSL/ASM.

The contract is concrete, not just "LLVM is better":

| feature | oracle | tinygrad ASM |
|---|---:|---:|
| dot4 | `16` | `16` |
| global loads | `11` | `22` |
| `global_load_b128` top-count | `4` | `0` |
| `global_load_b32` top-count | not top | `16` |
| DS total | `7` | `10` |
| waitcnt | `20` | `17` |
| `s_clause` | `3` | `0` |
| `s_delay_alu` | `30` | `0` |

The dynamic contract joins DSO:

| variant | median ms |
|---|---:|
| full ASM | `0.166649` |
| reduction-only | `0.153344` |
| synthetic-dot | `0.150879` |
| load/wait-only | `0.152562` |
| grouped-wait load-only | `0.151725` |

A0 verdict: **PASS_A0**.

## A1 — capability map

A1 mapped the concrete oracle features against the current AMD DSL/assembler surface and DSO movement evidence.

| feature | status | measured/estimated movement | A2 gate? | action |
|---|---|---:|---|---|
| native dot4 | expressible now | `0us` | no | closed as non-blocker |
| vector/coalesced loads | mnemonics expressible, not proven as standalone scheduler feature | `~14.1us` | no | do not run standalone A2 |
| waitcnt grouping | expressible now | `~0.84us` | no | closed as standalone |
| reduction rewrite | expressible now | `~13.3us` | no | closed as standalone |
| `s_clause` / `s_delay_alu` scheduling annotations | mnemonics exist, semantics are scheduler-level | unknown | no | project-level scheduler |
| local-y descriptor | small runtime/assembler feature, low EV | unknown | no | ergonomics/compiler roadmap |
| register/live-range scheduler | renderer scheduler feature | unknown | no | project-level Route A |

The A2 entry gate required one feature with credible `>=30us` movement. No feature clears it.

A1 verdict: **FAIL_A1_NO_BOUNDED_FEATURE**.

## Interpretation

This closes the "maybe Route A has a small first feature" question for q8 decode.

The oracle contract names real differences:

- vector/coalesced load shape;
- scheduler markers;
- resource/codegen encoding;
- wait/reduction details.

But the measured standalone candidates do not move enough. DSO already showed the q8-shaped kernels are
body-insensitive at this granularity; A1 confirms that the named features do not justify a bounded A2 proof.

## Decision

Do not execute A2 for q8 decode unless new PMU/SQTT evidence identifies a `>=30us` bounded feature.

Route A remains valid only as a broader compiler roadmap:

- latency-aware instruction scheduling;
- register/live-range scheduling;
- semantic placement of scheduling annotations;
- vector/coalesced load selection as part of a scheduler, not a one-off q8 fix;
- descriptor/local-id cleanup as supporting infrastructure.

Route B remains the practical research path:

- reproducible artifact/import route;
- graph-safe;
- `115.24us` isolated lifecycle;
- default off;
- policy-bound.
