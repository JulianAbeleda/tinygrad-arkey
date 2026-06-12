# QK Policy Memory Cap Runs

Date: 2026-06-12

Purpose: test whether generated Q4_K/Q6_K policy can scale to Qwen3-32B when
primitive packed-weight storage is treated as a capped resource.

## Result

| model | policy | reference mode | cap | reference tok/s | generated tok/s | gain | verdict |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3-32B-Q4_K_M | `32b-1536mb/policy.json` | generic fused baseline | `1536 MB` | `3.44` | `4.16` | `20.98%` | accept |

Correctness: `32b-1536mb/output-ab.json` reports `match=true` for a 32-token
greedy output A/B.

The full explicit Q4/Q6 primitive baseline is not the reference here because it
does not fit in VRAM for 32B. This run compares the capped generated policy
against the generic fused graph baseline.

## Storage

| metric | value |
|---|---:|
| selected primitive tensors | `144` |
| selected bytes | `1,600,389,120` |
| cap bytes | `1,610,612,736` |
| capped primitive tensors | `304` |

Selected tensor roles:

| role | tensors |
|---|---:|
| `attn_k` | `64` |
| `attn_v` | `64` |
| `ffn_down` | `16` |

## Artifacts

- `32b-1536mb/decision.json`: final machine-readable decision.
- `32b-1536mb/README.md`: per-run report.
- `32b-1536mb/policy.json`: tensor-scoped capped generated policy.
- `32b-1536mb/policy-parity.json`: explicit/generic/generated policy parity.
- `32b-1536mb/decode-summary.md`: repeated decode samples.
- `32b-1536mb/output-ab.json`: greedy correctness A/B.
- `32b-1536mb/profile-report.md`: DEBUG=2 profile report.

## Interpretation

This run proves the 32B blocker is storage architecture, not generated-search
semantics. The same search output can be lowered into a tensor-scoped policy
that fits by selecting high-benefit tensors and leaving the rest on the fused
graph.

The next architectural step is shared/lazy primitive storage so full explicit
and full generated policies can fit, after which 32B can be measured as a true
explicit-vs-generated scaling point.
