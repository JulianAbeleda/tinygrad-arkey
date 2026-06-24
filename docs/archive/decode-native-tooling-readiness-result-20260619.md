# Decode Native Tooling Readiness Result

Date: 2026-06-19

Scope:

- `docs/decode-native-tooling-readiness-scope-20260619.md`

Artifacts:

- Probe: `extra/qk_decode_native_tooling_readiness.py`
- Directory: `bench/qk-decode-native-tooling/`
- Readiness: `bench/qk-decode-native-tooling/readiness.json`
- Feature attribution: `bench/qk-decode-native-tooling/feature_attribution.json`

## Verdict

`TOOLING_NOT_READY`.

The existing tooling is now frozen into one readiness artifact. It is sufficient to say what not to build next, but it
is not sufficient to start native scheduler/renderer implementation.

Current state:

```text
ATT/HCQ visibility exists; timing policy exists; native scheduler feature attribution still does not.
```

## Start Gate

| Gate | Current |
|---|---:|
| N2 candidate count | `0` |
| Max timing/dynamic-attribution movement | `14.087us` |
| Required q8 feature movement | `>=30us` |
| Max projected W==D movement | `0%` |
| Verdict | `TOOLING_NOT_READY` |

No feature clears the native implementation start gate.

## Missing Rows

The generated readiness artifact names four blocking rows:

1. q8 `ffn_gate/up` role-joined ATT/PMC/body evidence.
2. timing-grade feature attribution `>=30us`.
3. counter/timing join that converts scheduler-resource unknowns into a bounded feature.
4. bytes/math/overhead bucket classification for q8 `ffn_gate/up`.

These are tooling gaps, not kernel implementation tasks.

## What We Have

| Layer | Status |
|---|---|
| AQLprofile ATT replay through tinygrad HCQ | `PASS_BODY_ATTRIBUTION` |
| Primitive ATT atlas | `PASS_ATT_PRIMITIVE_ATTRIBUTION` |
| Complete decode tooling | `COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS` |
| N1 native attribution | `N1_COMPLETE_NO_N2_START` |
| q8 artifact research route | measured `1.05-1.06x`, default off |

This is enough to reject fallback/runtime-cache explanations and reject direct-output/reduce-fusion from current
evidence. It is not enough to identify the native AMD scheduler feature to implement.

## Feature Attribution State

| Feature | Movement | Decision |
|---|---:|---|
| dot4 instruction selection | `0us` | already matched; closed |
| global load shape/coalescing | `14.087us` | below N2 gate |
| waitcnt grouping | `0.837us` | closed as standalone target |
| reduction topology | `13.305us` | below N2 gate |
| `s_clause` / `s_delay_alu` scheduler markers | unknown | static diff only |
| register/live-range/resource scheduler | unknown | project-level until attributed |
| local-y descriptor / launch contract | unknown/low EV | do not reopen for decode speed |

The only plausible remaining large movement is scheduler/resource behavior, but it is still unattributed.

## Role Readiness

| Role | State |
|---|---|
| `attn_q/o` | in-model ATT body join exists; reduce/glue visible but below build gate |
| `ffn_down` | Q6 surface fallback ATT plus runtime identity; visibility only |
| `lm_head` | Q6 surface fallback ATT plus runtime identity; visibility only |
| `ffn_gate/up` | runtime identity only; missing role-joined ATT/body evidence |
| `attn_k/v` | runtime identity only |

The key gap is still q8/native `ffn_gate/up`, because it is the only measured decode speed route and the native
scheduler oracle target.

## Decision

Do not start native scheduler/renderer implementation from current evidence.

Allowed next work:

- DTR-1: fill q8 `ffn_gate/up` role-joined body evidence;
- DTR-2: join timing, ATT/PMC/SQTT, ISA/resource metadata, and oracle rows into feature attribution;
- DTR-3: run dynamic ablations only to price named features;
- DTR-4: rerun readiness and start N2 only if a bounded feature clears the gate.

Disallowed next work:

- q8-specific scheduler/codegen patch without a `>=30us` attributed feature;
- reduce/glue fusion from packet visibility alone;
- old env knob searches;
- imported Q4 routing as a speed route.

## Next Concrete Command

Current readiness can be regenerated with:

```bash
python3 extra/qk_decode_native_tooling_readiness.py
```

The next missing artifact should be a q8 `ffn_gate/up` role-join result that updates
`bench/qk-decode-native-tooling/readiness.json` from "missing role-joined body evidence" to either a concrete bucket
classification or a new blocker.
