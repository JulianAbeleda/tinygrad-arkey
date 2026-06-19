# Decode next 1-2 execution result - 2026-06-19

Purpose: execute the high-level decode next steps:

1. use the q8 artifact route as the research answer;
2. start the native AMD scheduler/codegen project path.

Artifacts:

- `extra/qk_decode_next12_execution.py`
- `bench/qk-decode-next12/execution.json`

## Verdict

**NEXT_1_COMPLETE_NEXT_2_CHARTERED_N1_BLOCKED_ON_ATTRIBUTION**.

## 1 - q8 Research Answer

Status: **COMPLETE_RESEARCH_ANSWER**.

The measured decode kernel/lifecycle win is the q8 fused FFN artifact route:

- flag: `Q8_FFN_HANDWRITTEN=1`;
- default: off;
- dependency: external hipcc/LLD HSACO;
- fallback: flag off returns to default tinygrad decode.

Evidence:

| metric | value |
|---|---:|
| min W==D speedup | `1.051x` |
| median W==D speedup | `1.059x` |
| dNLL | `+0.002887` |
| lifecycle | `115.24us` |

Decision: this is done as a research answer. Do not spend more decode time on imported Q4 routing.

## 2 - Native Scheduler Project

Status: **ACTIVE_PROJECT_CHARTER**.

N0 is complete:

- oracle contract artifact: `bench/q8-ffn-amd-scheduler-project/oracle_contract.json`;
- verdict: `PASS_A0`;
- named features: global load shape, scheduler markers, wait/reduction details, resource contract, work decomposition.

N1 is the first real work item and is currently blocked:

- PMC profile runnable: yes;
- SQTT capture runnable: yes;
- SQTT decode usable: no;
- A2 reopen: no.

Blocker:

```text
SQTT capture is runnable, but the local decoder failed on every instruction-trace blob.
```

Bounded-feature state:

- A2 candidates: `0`;
- largest measured standalone delta: `15.77us`;
- verdict: `FAIL_A1_NO_BOUNDED_FEATURE`.

## Next Native Work

Do this before any native codegen patch:

1. make SQTT decode usable for RDNA3 HCQ instruction traces, or add another attribution path;
2. use attribution to assign `>=30us` movement to one scheduler feature;
3. only then implement an N2 scheduler-feature proof.

Start gate for compiler code changes:

```text
>=30us attributed feature or explicit whole AMD backend scheduler funding
```
