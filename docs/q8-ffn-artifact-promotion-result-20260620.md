# q8 FFN Artifact Promotion Result

Date: 2026-06-20

Artifact:
`bench/q8-ffn-artifact-promotion/promotion_result.json`

Command:

```bash
python3 extra/q8_ffn_artifact_promotion_execute.py --rerun-quality
```

Verdict:
`PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN`

## Decision

The q8 FFN handwritten/artifact route is promoted from research-only evidence to a hardened opt-in candidate.

It remains default-off:

```bash
Q8_FFN_HANDWRITTEN=1
```

No default behavior changed.

## Gate Results

| Gate | Verdict | Key result |
|---|---|---|
| Q8P-1 quality | `PASS_Q8P1_QUALITY_PROMOTION_GATE` | 4 windows, `max dNLL 0.002225`, `mean dNLL 0.000017`, threshold `0.01` |
| Q8P-2 default safety | `PASS_Q8P2_DEFAULT_SAFETY_GATE` | flag default-off, fallback declared, route guarded by flag |
| Q8P-3 coverage | `PASS_Q8P3_COVERAGE_GATE` | limited to Qwen3-8B `4096->12288` Q4_K gate/up on gfx1100 |
| Q8P-4 performance | `PASS_Q8P4_PERFORMANCE_GATE` | W==D min speedup `1.0507x`, host-sync `0.0%` |
| Q8P-5 artifact ownership | `PASS_Q8P5_ARTIFACT_OWNERSHIP_GATE` | hashes, source module, rebuild command, kernargs, fallback, no HIP runtime recorded |
| Q8P-6 model policy | `PASS_Q8P6_MODEL_POLICY_GATE_HARDENED_OPT_IN` | hardened opt-in accepted; default-on rejected for this pass |

## Quality Matrix

| Window | Tokens | dNLL |
|---|---:|---:|
| systems | 96 | `-0.001397` |
| hardware | 96 | `+0.000525` |
| quality | 96 | `+0.002225` |
| decode | 96 | `-0.001284` |

## Performance Matrix

| ctx | baseline tok/s | q8 tok/s | speedup |
|---:|---:|---:|---:|
| 128 | `79.5` | `84.5` | `1.063x` |
| 512 | `73.0` | `77.4` | `1.060x` |
| 1024 | `71.3` | `75.4` | `1.058x` |
| 4096 | `65.1` | `68.4` | `1.051x` |

## Policy Boundary

Default-on is not accepted in this pass because the route is lossy and externally owned. The accepted promotion is:

- hardened opt-in candidate;
- supported model set: Qwen3-8B Q4_K_M-style dense FFN, `dim=4096`, `hidden=12288`, gfx1100;
- rollback: unset `Q8_FFN_HANDWRITTEN`;
- fallback: existing default tinygrad decode.

Do not mix this small q8 route with the larger MMVQ contract-preservation project.
