# Decode Native Renderer DNR-3 Scheduler/Resource Scope - 2026-06-20

## Verdict

`BLOCKED_DNR3_NEEDS_BROAD_SCHEDULER_RESOURCE_MODEL_AND_ATTRIBUTION`

DNR-3 is the next step after DNR-2 correctness. It is not a single decode patch. The native q8/Q4_K gate/up lowering is
correct, but the remaining gap is scheduler/resource behavior.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3_scheduler_resource_scope.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3_scheduler_resource_scope_result.json
```

## Timing Context

| row | time |
|---|---:|
| native tinygrad AMD DSL gate/up, historical | `166.649us` |
| hipcc/LLD q8 artifact oracle consumer | `93.54us` |
| native minus oracle | `73.109us` |
| original target gate | `60us` |

The native stream is correct but slow. DNR-3 is about explaining and closing that scheduling/resource gap.

## Closed As Standalone Fixes

| feature | decision | movement |
|---|---|---:|
| dot4 instruction selection | closed; already matched | `0us` |
| global load shape/coalescing | below standalone gate | `14.087us` |
| waitcnt grouping | below standalone gate | `0.837us` |
| reduction topology | below standalone gate | `13.305us` |

These should not be reopened one at a time.

## Required Capabilities

| capability | why needed | status |
|---|---|---|
| semantic schedule IR | encode def/use, memory space, lane role, dependency groups, and legal reorder boundaries | missing |
| `s_clause` / `s_delay_alu` policy | oracle has `s_clause=3`, `s_delay_alu=30`; native has none | missing semantics |
| coalesced load lowering policy | native grouped global loads `22`; oracle `11`, including `global_load_b128` | missing policy, not missing opcode |
| register live-range/resource policy | choose instruction order and register reuse without serializing loads/dot/reduction | missing |
| branch/exec policy | oracle has branch/exec control absent from native stream | missing |
| hardware attribution | static diffs are not timing authority; local SQTT decode remains unusable | blocked tooling |

## What This Means

DNR-3 can proceed only as broad backend scheduler/resource work. It needs a semantic representation of the decode MMVQ
stream and a correctness-preserving emitter for compound candidates. BEAM/search stays blocked because there is no legal
native knob space to search yet.

## Do Not Do

- Do not start BEAM/search.
- Do not static-copy `s_delay_alu` or `s_clause`.
- Do not reopen standalone waitcnt, global-load, or reduction patches.
- Do not claim performance from instruction-count similarity.

## Minimum Unblock

Start DNR-3A only if the branch accepts a backend implementation task:

1. add semantic schedule IR for decode MMVQ def/use and legal reordering;
2. add resource/live-range ledger for VGPR/SGPR/private/LDS;
3. add correctness-preserving coalesced load lowering;
4. add semantic `s_clause` / `s_delay_alu` insertion policy;
5. time a compound candidate against the q8 oracle and explain the movement.
