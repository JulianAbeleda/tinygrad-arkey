# 8B Prefill Path 1 Mixed MVP Scope

Date: 2026-07-09.

## Decision

Path 1 is the active E2E MVP.

Use the normal AMD backend for the surrounding model graph, and use the generated pipe primitive for the selected hot
fp16 prefill GEMM route. Do not require full `DEV=AMD:ISA` model lifecycle ownership for this MVP.

## What This Proves

The MVP proves:

- role selection can bind all pipe-eligible fp16 roles to `prefill_wmma_pipe_primitive_generated`,
- the selected prefill GEMM route is pure/generated, not the raw `prefill_pipe_role_selective_generated` oracle,
- the route executes with finite/nonzero correctness,
- the generated stream has b128 loads, WMMA, targeted waitcnt, and no full wait drain,
- whole-prefill smoke can run through the mixed lifecycle and report the generated route.

## What This Does Not Prove

The MVP does not prove:

- full whole-model `DEV=AMD:ISA` lifecycle ownership,
- `attn_kv` or `ffn_down` generated pipe coverage,
- `ffn_gate_up` replacement,
- promotion-level speed versus the raw oracle.

The current full-AMD:ISA blocker remains dynamic `CDIV` from non-GEMM lifecycle code. That belongs to Path 2.

## Required Command

Use the existing whole-prefill harness:

```sh
PYTHONPATH=. DEV=AMD \
  python3 extra/qk/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --mode smoke \
  --path1-mvp \
  --artifact bench/prefill-whole-synced/path1-mixed-mvp-smoke.json \
  --json
```

`--path1-mvp` sets:

```text
PREFILL_V2=1
PREFILL_GRAPH_GEMM=1
PREFILL_WMMA_PIPE_PRIMITIVE=1
PREFILL_ROUTE=fp16
PREFILL_CHUNKED=0
```

It also implies `--logits-only` and fails closed unless route attribution reports:

```text
prefill_route_family=prefill_wmma_pipe_primitive_generated
prefill_route_pure=true
prefill_route_rolled_back=false
prefill_route_provenance=tinygrad_scheduler_generated
```

It also fails closed if `PREFILL_ROUTE` is not `fp16` or `PREFILL_CHUNKED` is enabled. This is intentional: the
previous `auto + chunked` smoke could report the generated route id while the model entry lifecycle selected the
direct-packed Q4/Q6 prefill route before reaching `route_pf16_graph_gemm`.

## Completion Bar

Path 1 MVP is complete when these artifacts exist and pass:

| Gate | Artifact | Requirement |
|---|---|---|
| Route correctness + trace | `bench/prefill-pipe-mvp/latest.json` | correctness pass, `uses_hand_pipe_oracle=false`, b128 + WMMA + targeted waitcnt |
| Whole smoke | `bench/prefill-whole-synced/path1-mixed-mvp-smoke.json` | `PATH1_MIXED_PREFILL_MVP_PASS` |

## Current Result

Status: complete for all pipe roles.

- `bench/prefill-pipe-mvp/latest.json`:
  - route: `prefill_wmma_pipe_primitive_generated`,
  - correctness: pass, finite/nonzero, sampled rel RMSE about `2.1e-4`,
  - trace: `global_load_b128=32`, `wmma=8`, `targeted_waitcnt=11`, `full_waitcnt=0`,
  - `uses_hand_pipe_oracle=false`.
- `bench/prefill-whole-synced/path1-mixed-mvp-smoke.json`:
  - verdict: `PATH1_MIXED_PREFILL_MVP_PASS`,
  - whole-prefill smoke: about `220 tok/s` at pp512,
  - route attribution: pure generated pipe primitive, not rolled back.

All-pipe-role completion requires:

- `bench/prefill-pipe-mvp/path1-all-pipe-roles.json`,
- verdict `PATH1_PIPE_ALL_ROLES_PASS`,
- roles exactly `attn_qo`, `attn_kv`, and `ffn_down`,
- excluded role `ffn_gate_up`.

Current all-role result:

| Role | Shape `(M,N,K)` | Correctness | rel RMSE | Trace |
|---|---:|---|---:|---|
| `attn_qo` | `512,4096,4096` | pass | `0.0002075` | `global_load_b128=32`, `wmma=8`, `targeted_waitcnt=11`, `full_waitcnt=0` |
| `attn_kv` | `512,1024,4096` | pass | `0.0002065` | `global_load_b128=32`, `wmma=8`, `targeted_waitcnt=11`, `full_waitcnt=0` |
| `ffn_down` | `512,4096,12288` | pass | `0.0002148` | `global_load_b128=32`, `wmma=8`, `targeted_waitcnt=11`, `full_waitcnt=0` |

Rollup verdict: `PATH1_PIPE_ALL_ROLES_PASS`.

## Next After MVP

After all pipe roles pass:

1. Keep `ffn_gate_up` on the existing LDS/raw oracle until a separate primitive exists.
2. Run promotion timing against the current raw graph-GEMM oracle.
3. Decide whether the next primitive is `ffn_gate_up` LDS ownership or Path 2 full AMD:ISA lifecycle.
