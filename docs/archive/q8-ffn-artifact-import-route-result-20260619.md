# q8 FFN artifact/import route result (2026-06-19)

Executed Route B from `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md`.

Verdict: **PASS_RESEARCH / policy-bound**.

The hipcc/LLD q8 decode artifact route is now reproducible, HCQ-loadable, fixed-launch wrapped, graph-safe by the
existing injection proof, and still default-off. This does not make q8 a native tinygrad codegen win; it makes the
external-artifact route a clean research route.

## Artifacts

Probe:

- `extra/q8_ffn_artifact_import_route.py`

Outputs:

- `bench/q8-ffn-amd-scheduler-project/artifact_build_manifest.json`
- `bench/q8-ffn-amd-scheduler-project/artifact_loader.json`
- `bench/q8-ffn-amd-scheduler-project/artifact_policy_boundary.json`
- `bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json`
- `bench/q8-ffn-amd-scheduler-project/result.json`

## B1 — reproducible artifact build

The executor rebuilds the producer and fused gate/up kernels from pinned source strings in
`extra.q8_ffn_fast_artifact_probe` using:

1. `hipcc -c -emit-llvm --cuda-device-only -O3 -mcumode --offload-arch=<arch>`;
2. `hipcc -target amdgcn-amd-amdhsa -mcpu=<arch> -O3 -mllvm -amdgpu-internalize-symbols`;
3. `/opt/rocm/llvm/bin/ld.lld -flavor gnu -shared`.

| artifact | hash | kernarg | LDS | key check |
|---|---|---:|---:|---|
| producer `q8_rmsnorm_side` | `dd119afa0ef41c8d...` | `32` | `4096` | loads in `AMDProgram` |
| fused gate/up `q8_mmvq_gateup` | `9d00b0723a6aa92d...` | `40` | `16` | `16` dot4, loads in `AMDProgram` |

Build gates:

- producer loads in `AMDProgram`: PASS;
- gate/up loads in `AMDProgram`: PASS;
- gate/up has `16` dot4 ops: PASS;
- no unresolved relocations: PASS.

## B2 — fixed-launch artifact loader

The loader is deliberately narrow:

- `AMDProgram` loads the linked HSACO;
- `FixedLaunchRunner` pins artifact launch dimensions;
- no in-process HIP runtime;
- no default/model route change.

Measured isolated route:

| metric | value |
|---|---:|
| fused gate/up consumer | `93.54us` |
| producer + gate/up lifecycle | `115.24us` |
| lifecycle gate | `<=129.2us` PASS |
| no in-process HIP runtime | PASS |

Correctness gates all pass against the q8 proxy:

- producer correct;
- gate correct;
- up correct.

## B3 — graph-safe research route

Existing graph injection proof was rerun into this route's artifact directory:

- `bench/q8-ffn-amd-scheduler-project/artifact_graph_route.json`

Result: **PASS**.

| check | result |
|---|---:|
| eager max_abs vs q8 proxy | `0.001373` |
| TinyJit replay max_abs | `0.001373` |
| TinyJit replay correct | PASS |
| no in-process HIP runtime | PASS |
| default changed | false |

The W==D and dNLL authority remains the A4 result because this is the same graph-injected artifact route:

- W==D decode `1.051-1.063x`;
- dNLL `+0.002887`;
- default off.

## B4 — maintenance boundary

The route is explicitly research-only.

Supported boundary:

- model family: Qwen3-8B Q4_K_M-style dense FFN block;
- `dim=4096`, `hidden=12288`;
- gate/up weights: Q4_K;
- activation side-channel: q8_1;
- GPU arch: `gfx1100`;
- runtime: tinygrad AMD HCQ / `AMDProgram`;
- no in-process HIP runtime.

Non-goals:

- not a default route;
- not a portable tinygrad backend feature;
- not validated for other hidden sizes, quant formats, or GPU archs;
- not a replacement for native scheduler/codegen ownership.

Policy gate:

- the project must explicitly accept the external hipcc/LLD HSACO dependency before using this beyond research.

## Decision

Route B is now clean enough to keep as a research flag:

- B1 reproducible artifact build: PASS;
- B2 fixed-launch loader: PASS;
- B3 graph-safe injection: PASS;
- B4 maintenance boundary: documented.

Route A remains the only path to native tinygrad ownership. Do not call Route B a compiler win; it is a controlled
artifact/import bridge.
