# Decode N1 native scheduler attribution scope - 2026-06-19

Purpose: decide whether the native q8/MMVQ scheduler lane earns an N2 implementation patch.

This scope follows the accepted two-lane decode decision:

- artifact lane: complete as `Q8_FFN_HANDWRITTEN=1`, default off, research-only;
- native lane: continue only if attribution identifies a bounded compiler/runtime feature worth implementing.

## Gate

Start N2 only if one feature has credible `>=30us` attributed movement on the q8 gate/up oracle gap.

The gate is intentionally high because the completed q8 artifact route already gives the small research win. Native work
should not begin from "LLVM is better" or "scheduler likely matters"; it needs a concrete primitive-sized feature.

## Inputs

- `bench/q8-ffn-amd-scheduler-project/oracle_contract.json`
- `bench/q8-ffn-amd-scheduler-project/dsl_capability_map.json`
- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json`

## Feature Buckets

N1 must classify:

- dot4 instruction selection;
- global-load shape/coalescing;
- waitcnt grouping;
- reduction topology;
- `s_clause` / `s_delay_alu` scheduler markers;
- register/live-range/resource scheduling;
- local-y descriptor / launch-contract effects.

## Output

- script: `extra/qk_decode_n1_attribution.py`
- artifact: `bench/q8-ffn-amd-scheduler-project/n1_attribution.json`

## Decision Rules

- If any bounded bucket is `>=30us`, start N2 for that exact feature.
- If all bounded buckets are below `30us`, do not start N2.
- If the only remaining movement is scheduler/resource behavior without hardware attribution, classify it as a
  project-level backend/tooling route, not a q8 decode patch.
