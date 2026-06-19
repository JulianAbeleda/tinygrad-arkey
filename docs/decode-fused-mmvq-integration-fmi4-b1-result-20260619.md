# Decode fused-MMVQ integration FMI-4 B1 result - 2026-06-19

Purpose: execute the first bounded Track-B discriminator from
`decode-fused-mmvq-integration-fmi1-fmi2-result-20260619.md`.

No defaults changed. No model route was wired. No prefill files were touched. This is an in-model eager measurement
surface over existing env knobs only.

Artifacts:

- `extra/qk_decode_fmi4_knob_probe.py`
- `bench/qk-decode-fused-mmvq-integration/fmi4_b1_knob_probe.json`
- `bench/qk-decode-fused-mmvq-integration/fmi4_b1_summary.md`

## Verdict

`FAIL_B1_NO_ENV_KNOB_CLEARS_GATE`.

Existing launch-shape env knobs do not close Track B. The next Track-B surface is **runtime/cache identity** or a
renderer/scheduler project, not `Q4K_COOP_RT` / `Q6K_COOP_RT` tuning.

## Method

The probe loads the target model once, keeps the same warm eager decode site, and sweeps:

- default;
- `Q6K_COOP_RT={1,2,8,16}`;
- `Q6K_FFN_DOWN_COOP=0`, `Q6K_LM_HEAD_COOP=0`;
- `Q4K_COOP_RT={4,8,32}`;
- `Q4K_ATTN_QO_COOP=0`.

Gate:

```text
one high-share role group moves >=10% relative isolated in-model
```

Important interpretation note: this B1 probe is a same-process **relative** knob sweep. Its absolute role percentages
should not replace the aggregate PMU/tok-s authority from `decode-bandwidth-bound-pmu-learning-20260619.md`.

## Results

Best observed relative movement:

| role | best config | relative vs baseline | absolute in this probe |
|---|---|---:|---:|
| `ffn_gate/up` | default | `1.000x` | `37.2%` HBM |
| `ffn_down` | `q6_rt16` | `1.004x` | `86.0%` HBM |
| `attn_k/v` | `q4_rt8` | `1.014x` | `28.7%` HBM |

No row reaches the `>=1.10x` gate.

## Decision

Close B1:

```text
existing env launch-shape knobs are not the integration lever
```

Track B remains live, but only at the deeper surfaces named in FMI-4:

- B2 runtime/cache route: ensure the in-model route uses the intended compiled program/metadata and graph-safe launch
  identity;
- B3 renderer/scheduler project: preserve the low-VGPR, large-grid MMVQ launch contract natively;
- B4 artifact/import only if a mature decode MMVQ artifact family is identified.

Track A q8 replay remains secondary because it is lossy and EV-capped by `ffn_gate/up` reuse count `2`.
