# Decode q8 both-lanes execution result - 2026-06-19

Purpose: execute `decode-q8-both-lanes-execution-scope-20260619.md`.

Artifacts:

- `extra/qk_decode_q8_both_lanes_execution.py`
- `bench/qk-decode-mmvq-large-project/q8_both_lanes_execution.json`

## Verdict

**BOTH_ACCEPTED_ARTIFACT_RESEARCH_AND_NATIVE_PROJECT_CHARTERED**.

## Lane 1 - Artifact Research Flag

Decision: **ACCEPT_FOR_RESEARCH_FLAG_ONLY**.

Operational state:

- flag: `Q8_FFN_HANDWRITTEN=1`;
- default: off;
- dependency accepted only for research-flag use: external hipcc/LLD HSACO;
- fallback: flag off returns to default tinygrad decode;
- runtime: tinygrad AMD HCQ / `AMDProgram`;
- no in-process HIP runtime.

Evidence:

| item | value |
|---|---:|
| min W==D speedup | `1.051x` |
| median W==D speedup | `1.059x` |
| dNLL | `+0.002887` |
| lifecycle | `115.24us` |

Maintenance actions:

- keep `Q8_FFN_HANDWRITTEN` default off;
- keep artifact source strings and hashes documented;
- rerun W==D and dNLL if source strings, toolchain, model shape, or graph route changes;
- do not generalize beyond gfx1100/Qwen3-8B dense Q4_K gate/up without a new scope.

## Lane 2 - Native Transfer Project

Decision: **FUND_AS_PROJECT_LEVEL_BACKEND_WORK**.

This is not a bounded q8 patch. Current native failures remain:

| route | measured |
|---|---:|
| COMGR lifecycle | `177.72us` |
| AMD DSL consumer | `166.65us` |
| artifact oracle lifecycle | `115.24us` |

Native phase plan:

| phase | deliverable | gate |
|---|---|---|
| N0 | oracle diff tool | labels instruction groups, load widths, wait placement, resources, timing |
| N1 | attribution | assigns `>=30us` movement to one scheduler feature |
| N2 | scheduler feature | improves q8 consumer by `>=25us` and stays correct |
| N3 | native rebuild | consumer `<=75us`, path to `<=60us`, max_abs `<=2e-3` |
| N4 | model gate | W==D `>=3%`, dNLL `<=0.01`, no external artifact |

Start gate:

Do not start native transfer until one bounded feature has `>=30us` measured or attributed movement, or the project
explicitly funds the whole AMD backend scheduler effort.
