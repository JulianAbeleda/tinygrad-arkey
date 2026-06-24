# q8 FFN fast artifact A3 route result (2026-06-19)

This executes the next gate after `q8-ffn-fast-artifact-vs-raw-code-result-20260619.md`.

Goal:

- take the passing hipcc/LLD q8 lifecycle primitive;
- route one dense FFN block through the real model buffers;
- then test whether the primitive can become Tensor-visible for TinyJit/HCQGraph capture.

Corrected verdict after the contract audit:

- **A3 eager one-block route: PASS.**
- **A3 Tensor-visible runtime-cache injection: PASS.**

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

## A3b — Tensor-visible injection: PASS after contract audit

Probe:

- `extra/q8_ffn_fast_artifact_inject.py`
- `extra/q8_ffn_injection_contract_audit.py`

Artifact:

- `bench/q8-ffn-handwritten-oracle/fast_artifact_inject.json`
- `bench/q8-ffn-handwritten-oracle/fast_artifact_contract_audit.json`

Attempted mechanism:

1. create placeholder `custom_kernel` PROGRAM nodes for producer and fused gate/up;
2. capture their `runtime_cache` keys;
3. replace those runtimes with the precompiled q8 artifacts;
4. use a probe-local `AMDComputeQueue.exec` patch so the artifact launch dims override placeholder dims, following the
   same pattern that worked for the Tensile route.

Initial attempt:

- placeholder `custom_kernel` bodies optimized away most inputs;
- producer exposed only `(norm_out, x)`;
- fused gate/up exposed only `(gate, up)`;
- swapped-runtime execution caused an AMD MMU fault.

The principle-aligned fix was to stop executing and audit the Tensor `PROGRAM` contract first.

Contract audit result:

| program | `globals` | `outs` | `ins` | artifact arg order |
|---|---|---|---|---|
| producer | `[0,1,2,3]` | `[0,1]` | `[2,3]` | `norm_out, q8, x, w` |
| fused gate/up | `[0,1,2,3,4]` | `[0,1]` | `[2,3,4]` | `gate, up, gate_words, up_words, q8` |

Two concrete bugs were fixed:

1. placeholder loads now keep all artifact input buffers alive in `ProgramInfo.globals`;
2. runtime-key install now uses real Q4_K storage shape/dtype: `uint32[7077888]`, not dummy `uint8`.

The remaining launch-dim mismatch is deliberate and handled by `Q8ArtifactRunner`:

| program | placeholder launch | artifact launch |
|---|---|---|
| producer | global `(1,1,1)`, local `(1024,1,1)` | global `(1,1,1)`, local `(1024,1,1)` |
| fused gate/up | global `(1,12288,1)`, local `(32,4,1)` | global `(12288,2,1)`, local `(32,4,1)` |

Execution after the audit:

| path | max_abs vs q8 proxy | mean_abs | verdict |
|---|---:|---:|---|
| eager injected node | 0.00137 | 4.44e-5 | PASS |
| TinyJit replay calls 2-3 | 0.00137 | 4.44e-5 | PASS |

No HIP runtime is loaded in-process.

## What this means

The q8 route is now graph-integrable as a research primitive. The next question is not kernel economics or graph
capture; it is full decode truth.

Next gate:

1. route dense FFN gate/up behind `Q8_FFN_HANDWRITTEN=1`, default off;
2. run W==D decode sweep;
3. rerun dNLL with the actual route.

## Current q8 status

| gate | status |
|---|---|
| q8 quality proxy | PASS (`dNLL +0.00165`) |
| isolated fast lifecycle | PASS (`114.12 us`) |
| eager one-block route | PASS (`121.38 us`) |
| Tensor-visible graph injection | PASS |
| W==D decode sweep | next |

The next research unit is W==D decode and quality, not more q8 kernel tuning.
