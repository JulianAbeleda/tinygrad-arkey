# Decode Native Tooling Completion Result

Date: 2026-06-19

Scope:

- `docs/decode-native-tooling-completion-scope-20260619.md`

Artifacts:

- Role join probe: `extra/qk_att_inmodel_role_join.py`
- Readiness probe: `extra/qk_decode_native_tooling_readiness.py`
- Role artifacts:
  - `bench/qk-att-inmodel-role-join/ffn_gate.json`
  - `bench/qk-att-inmodel-role-join/ffn_up.json`
- Readiness artifacts:
  - `bench/qk-decode-native-tooling/readiness.json`
  - `bench/qk-decode-native-tooling/feature_attribution.json`
  - `bench/qk-decode-native-tooling/feature_join.json`
  - `bench/qk-decode-native-tooling/ablation_matrix.json`

## Verdict

`TOOLING_NOT_READY`, with DTR-1 visibility complete.

The missing q8/native `ffn_gate/up` body-evidence row is now filled for the default native path. The remaining blocker
is not role visibility; it is still feature-level scheduler/resource attribution.

Current start gate:

| Gate | Current |
|---|---:|
| N2 candidate count | `0` |
| Max timing-grade movement | `14.087us` |
| Required movement | `>=30us` |
| Feature join | visibility/static only |
| Ablation verdict | `NO_N2_ABLATION` |

Do not start native scheduler/renderer implementation.

## DTR-1 Result

`ffn_gate` and `ffn_up` now have in-model ATT body attribution.

| Role | Verdict | Main program | Body packets | Notes |
|---|---|---|---:|---|
| `ffn_gate` | `PASS_INMODEL_ROLE_JOIN_NATIVE_COOP` | `q4k_gemv_partial_12288_4096_1` | `47,143` | launches expected native Q4_K path |
| `ffn_up` | `PASS_INMODEL_ROLE_JOIN_NATIVE_COOP` | `q4k_gemv_partial_12288_4096_1` | `47,132` | same binary/hash as gate |

Both roles also launch glue `E_32_32_4n2`. The main program hash is `236fd9e8841b577f`.

The attempted `ffn_gateup_pair` rerun hit the known model-load class:

```text
MemoryError: Allocation of 4.68 GB failed on AMD. Used: 0 B
```

This does not invalidate the individual `ffn_gate` / `ffn_up` captures; it means the pair/fused boundary remains a
separate lifecycle capture if needed.

## DTR-2 Result

`feature_join.json` was generated and joins the in-model role evidence to the q8 oracle contract.

Joined facts:

- role: `ffn_gate/up`;
- program: `q4k_gemv_partial_12288_4096_1`;
- hash: `236fd9e8841b577f`;
- native timing authority: existing q8 native/oracle contract, not same interval;
- native/oracle gap: `73.109us`;
- ATT: body-visible, not timing authority;
- ISA diffs: load shape, scheduler markers, reduction shape;
- resource attribution: still missing.

Verdict:

```text
PASS_VISIBILITY_JOIN_NO_COUNTER_GRADE_ATTRIBUTION
```

The join is useful but not enough to start N2 because it is still static/visibility grade for the largest suspected
scheduler/resource bucket.

## DTR-3 Result

`ablation_matrix.json` was generated from existing measured ablations.

| Feature | Movement | Decision |
|---|---:|---|
| dot4 instruction selection | `0us` | closed |
| global load shape/coalescing | `14.087us` | below gate |
| waitcnt grouping | `0.837us` | below gate |
| reduction topology | `13.305us` | below gate |
| `s_clause` / `s_delay_alu` scheduler markers | unknown | project-level/unattributed |
| register/live-range/resource scheduler | unknown | project-level/unattributed |
| local-y descriptor / launch contract | unknown | low EV / closed for decode speed |

Verdict:

```text
NO_N2_ABLATION
```

No single feature has timing-grade `>=30us` movement.

## What Changed

Before this pass:

```text
q8 ffn_gate/up role-joined body evidence was missing.
```

After this pass:

```text
q8 ffn_gate/up body evidence exists, and it confirms the expected native Q4_K program.
```

The remaining problem is narrower:

```text
which scheduler/resource feature explains the remaining 73.109us q8 native-to-oracle gap?
```

The current tooling still cannot answer that with timing/counter-grade evidence.

## Final Decision

Do not start a native q8 scheduler/codegen N2 patch.

Allowed next work:

- acquire counter/timeline-grade attribution for scheduler markers or register/resource behavior;
- explicitly fund broad AMD backend scheduler work without bounded attribution;
- keep the q8 artifact route as the default-off research answer.

Disallowed next work:

- manually inserting `s_clause` / `s_delay_alu` from static diff alone;
- q8-specific renderer changes without `>=30us` attributed movement;
- treating ATT packet count as timing;
- reopening load-shape, waitcnt grouping, or reduction topology as standalone N2 features.

## Outcome Label

The correct outcome is:

```text
TOOLING_NOT_READY_FOR_N2
```

Not because role visibility is missing anymore, but because counter/timing-grade scheduler-resource attribution is
still missing.
