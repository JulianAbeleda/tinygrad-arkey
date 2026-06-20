# Decode Native Renderer Project Scope - 2026-06-20

## Verdict

`PASS_DECODE_NATIVE_RENDERER_PROJECT_SCOPE_READY_BROAD_BACKEND_REQUIRED`

This scope answers the current decode question after the P7/P8 closeout:

- imported llama Q4 MMVQ routing was measured and closed as a speed path;
- the q8 fused artifact remains the only measured decode upside;
- native tinygrad does not have a bounded N2 scheduler/renderer patch ready;
- BEAM/search is still premature because there is no native lowerable decode schedule space to search.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_project_scope.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_project_scope_result.json
```

## What Happened

The old next step was ambiguous because several decode routes were open at once. That is no longer true.

| route | current state | consequence |
|---|---|---|
| current default decode | keep promoted | no default change from this project |
| imported llama Q4 MMVQ graph route | closed as speed path | P7d/P7e measured local losses; do not continue as a performance route |
| q8 fused artifact route | correct, measured, default-off | useful oracle/research flag; not native tinygrad |
| native tinygrad MMVQ renderer | project-level work | no bounded N2 feature cleared the start gate |

The decisive tooling result is that `n2_candidate_count=0`. The largest isolated timing movement is about `14us`,
below the `30us` gate required to justify a one-off native decode patch.

## Oracle Contract

The project oracle is the hipcc/LLD q8 artifact pair.

| component | contract |
|---|---|
| q8 producer | `global=[1,1,1]`, `local=[1024,1,1]`, kernarg `32`, LDS `4096`, private `0` |
| fused gate/up consumer | `global=[12288,2,1]`, `local=[32,4,1]`, kernarg `40`, LDS `16`, private `0` |
| work decomposition | 128 threads per row; y selects gate/up; 16 Q4_K blocks; `sub=tid&7`; `kb=tid/8` |
| lifecycle timing | producer median about `21.7us`, gate/up consumer median about `93.54us`, lifecycle about `115.24us` |
| correctness | producer, gate, and up correctness all pass |

## Native Equivalents Missing

| oracle primitive | tinygrad state | missing native equivalent |
|---|---|---|
| q8 activation producer lifecycle | artifact import exists | native schedule contract plus quality/promotion policy |
| packed Q4_K + q8 dot4 consumer | `v_dot4_i32_iu8` already exists | address/data-format lowering and resource scheduling |
| global load shape/coalescing | expressible and measured | standalone movement is below N2 gate |
| waitcnt grouping | expressible and measured | standalone movement is below N2 gate |
| reduction topology | expressible and measured | standalone movement is below N2 gate |
| `s_clause` / `s_delay_alu` | mnemonics exist | semantic insertion policy and timing attribution |
| register live-range/resource policy | classified as renderer scheduler work | compound backend model, not a one-off patch |

## Project Tracks

| track | purpose | exit gate |
|---|---|---|
| DNR-0 oracle preservation | keep q8 artifacts as correctness/timing authority | artifact route stays correct, default-off, and reproducible |
| DNR-1 schedule contract object | bind producer and fused gate/up oracle contract into a native AMD schedule object | structural gates pass against launch/resource/ISA counts |
| DNR-2 address and data-format lowering | lower block_q8_1, packed Q4_K, min/scale correction, and y-role selection | native kernel runs and matches q8 oracle numerically |
| DNR-3 scheduler/resource model | model `s_clause`, `s_delay_alu`, register lifetimes, ordering, and wait/resource policy | native timing closes toward oracle without scratch/private spills |
| DNR-4 timing authority | one-clock interleaved timing against default, artifact oracle, and native candidate | W==D, dNLL, lifecycle, and clock provenance table |
| DNR-5 search/BEAM enablement | search only legal knobs in a lowerable native schedule space | correctness-preserving candidates can be compared to oracle |

## Explicit Non-Goals

- No default routing change.
- No BEAM/search until native lowering and scheduler knobs exist.
- No standalone load-shape, waitcnt, reduction, or dot4 patch.
- No promotion of the lossy q8 route without W==D, dNLL, and lifecycle policy gates.
- No performance claim from static instruction-count similarity alone.

## Next Action

Start DNR-1: bind the q8 producer and fused gate/up oracle into the native decode schedule-object path and make the
structural gate executable. Only after DNR-1 passes should DNR-2 attempt runnable native lowering.
