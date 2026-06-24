# q8 FFN Artifact Promotion Scope

Date: 2026-06-20

Artifact:
`bench/q8-ffn-artifact-promotion/scope.json`

Command:

```bash
python3 extra/q8_ffn_artifact_promotion_scope.py
```

Verdict:
`PASS_Q8_FFN_ARTIFACT_PROMOTION_SCOPE_READY`

## Purpose

Scope what it would take to promote the q8 FFN handwritten/artifact route from default-off research to a default
candidate.

Current route:

- flag: `Q8_FFN_HANDWRITTEN=1`;
- scope: Qwen3-8B Q4_K_M dense FFN gate/up, `dim=4096`, `hidden=12288`, gfx1100;
- primitive: fused RMSNorm+q8 producer plus fused Q4_K x q8 gate/up consumer;
- current status: `PASS_RESEARCH`, default off.

## Current Evidence

| Gate | Current evidence | Promotion interpretation |
|---|---:|---|
| W==D speed | `1.051x-1.063x` over ctx128/512/1024/4096 | good enough to scope promotion |
| minimum speedup | `1.0507x` | clears research/perf floor |
| NLL baseline | `2.855476` over 160 tokens | too narrow for default |
| NLL q8 route | `2.858363` over 160 tokens | dNLL passes research |
| dNLL | `+0.002887` | under `<=0.01`, but single-window only |
| default behavior | unchanged | required and currently satisfied |
| fallback | flag off returns to default decode | required and currently satisfied |

## Promotion Gates

| Phase | Gate | Minimum pass |
|---|---|---|
| Q8P-1 | Quality promotion | Multi-window or task-quality eval passes `dNLL <=0.01` on every accepted window, reports mean/max dNLL, and includes W==D greedy sanity. |
| Q8P-2 | Default safety | Route remains flag-gated, fallback returns byte-identical existing decode, unsupported paths fall back, and default-off/default-on behavior is isolated in subprocess tests. |
| Q8P-3 | Coverage | Routed tensors/layers are enumerated; route is limited to Qwen3-8B `4096->12288` Q4_K gate/up on gfx1100; no accidental lm_head, attention, Q6, prefill, or unsupported model routing. |
| Q8P-4 | Performance | Clean W==D decode reproduces `>=3%` speedup at ctx512/1024/4096 with host-sync `<=5%` and no per-step host Tensor contamination. |
| Q8P-5 | Artifact ownership | Manifest records source, build command, code hashes, kernargs, supported arch, no HIP runtime, fallback, and maintenance owner; or a native tinygrad-owned replacement is selected. |
| Q8P-6 | Model policy | Explicit decision accepts or rejects a lossy `~3-6%` default route; names fallback, quality threshold, supported model set, release flag, and rollback criteria. |

## Why Q8P-1 Comes First

The current route already has a real W==D speed win and a passing research dNLL, but it is lossy and the quality
evidence is only one 160-token window. Default promotion requires broader quality confidence before spending more
time on policy and ownership.

## Stop Rules

- Do not default the route from current evidence.
- Do not broaden routing before coverage is enumerated.
- Do not claim native tinygrad ownership while the win depends on external hipcc/LLD artifacts.
- Do not mix this small q8 route with the larger MMVQ contract-preservation project.

## Next

Run Q8P-1: quality promotion gate.
