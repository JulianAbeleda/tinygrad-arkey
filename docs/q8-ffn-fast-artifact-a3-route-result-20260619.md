# q8 FFN fast artifact A3 route result (2026-06-19)

This executes the next gate after `q8-ffn-fast-artifact-vs-raw-code-result-20260619.md`.

Goal:

- take the passing hipcc/LLD q8 lifecycle primitive;
- route one dense FFN block through the real model buffers;
- then test whether the primitive can become Tensor-visible for TinyJit/HCQGraph capture.

Verdict:

- **A3 eager one-block route: PASS.**
- **A3 Tensor-visible runtime-cache injection: BLOCKED_UNSAFE.**

## A3a — eager one-block fast-artifact route: PASS

Probe:

- `extra/q8_ffn_oneblock_route.py --fast-artifact`

Artifact:

- `bench/q8-ffn-handwritten-oracle/oneblock_fast_artifact_route.json`

Route:

`ffn_norm input -> hipcc/LLD q8 producer -> hipcc/LLD fused gate/up consumer -> silu(gate)*up -> existing ffn_down`

Result:

| check | value | verdict |
|---|---:|---|
| producer | 20.99 us | pass |
| fused gate/up | 100.38 us | pass |
| lifecycle | **121.38 us** | pass vs <=129.2 us |
| route vs graph q8 proxy max_abs | 0.00137 | pass |
| route vs graph q8 proxy mean_abs | 4.44e-5 | pass |
| HIP runtime in process | absent | pass |
| default changed | no | pass |

This proves the passing isolated primitive survives a real one-block route when launched directly through HCQ.

## A3b — Tensor-visible injection: BLOCKED_UNSAFE

Probe:

- `extra/q8_ffn_fast_artifact_inject.py`

Artifact:

- `bench/q8-ffn-handwritten-oracle/fast_artifact_inject.json`

Attempted mechanism:

1. create placeholder `custom_kernel` PROGRAM nodes for producer and fused gate/up;
2. capture their `runtime_cache` keys;
3. replace those runtimes with the precompiled q8 artifacts;
4. use a probe-local `AMDComputeQueue.exec` patch so the artifact launch dims override placeholder dims, following the
   same pattern that worked for the Tensile route.

The dry-run contract is:

| program | placeholder launch | artifact launch | kernarg |
|---|---|---|---:|
| producer | global `(1,1,1)`, local `(1024,1,1)` | global `(1,1,1)`, local `(1024,1,1)` | 32 B |
| fused gate/up | global `(1,12288,1)`, local `(128,1,1)` | global `(12288,2,1)`, local `(32,4,1)` | 40 B |

The dim mismatch was handled by the queue-exec override, but execution still caused an AMD MMU fault during eager
route output copyout:

`MMU fault: 0x76205F713000 | NotPresent=1 ReadOnly=0 NoExecute=0 imprecise=0`

So this is not just a launch-dim problem. The remaining unsafe mismatch is likely the Tensor placeholder
`ProgramInfo.globals/outs/ins` contract versus the artifact kernarg buffer order/lifetime.

The probe is now safe by default: it writes the contract and `BLOCKED_UNSAFE` verdict unless explicitly run with
`--execute`.

## What this means

The q8 route is fast enough and correct enough at the eager HCQ level. The blocker is graph integration, not kernel
economics.

Do **not** continue by rerunning unsafe runtime-cache swaps. The next safe step is one of:

1. build a contract verifier that prints `ProgramInfo.globals`, `outs`, `ins`, resolved buffer order, and kernarg
   pointer offsets before launch; or
2. bypass placeholder `custom_kernel` swapping and emit explicit precompiled `Ops.PROGRAM` nodes whose
   `ProgramInfo` exactly matches the artifact launch and buffer contract.

Until that exists, W==D decode cannot be measured honestly because direct HCQ calls are outside TinyJit/HCQGraph and
would add per-token Python launch overhead.

## Current q8 status

| gate | status |
|---|---|
| q8 quality proxy | PASS (`dNLL +0.00165`) |
| isolated fast lifecycle | PASS (`114.12 us`) |
| eager one-block route | PASS (`121.38 us`) |
| Tensor-visible graph injection | BLOCKED_UNSAFE |
| W==D decode sweep | blocked on graph injection |

The next research unit is graph integration, not more q8 kernel tuning.
