# Decode q8 research route hardening result - 2026-06-19

Purpose: execute the small-path hardening pass from `decode-large-small-paths-scope-20260619.md`.

No model default changed. No new kernel was built by this pass; it consolidates the existing measured authority into a
single small-path decision artifact.

Artifacts:

- `extra/qk_decode_path_split.py`
- `bench/qk-decode-path-split/small_q8_hardening.json`

## Verdict

`PASS_RESEARCH_HARDENED_EXISTING_EVIDENCE`.

The q8 FFN artifact route is clean enough to keep as a default-off research flag. It should not be confused with the
large parity path.

## Measured Authority

W==D decode:

| ctx | baseline tok/s | q8 tok/s | speedup |
|---:|---:|---:|---:|
| 128 | `79.5` | `84.5` | `1.063x` |
| 512 | `73.0` | `77.4` | `1.060x` |
| 1024 | `71.3` | `75.4` | `1.058x` |
| 4096 | `65.1` | `68.4` | `1.051x` |

Quality:

- baseline NLL: `2.855476`;
- q8 route NLL: `2.858363`;
- dNLL: `+0.002887`;
- gate: `<=0.01`;
- tokens: `160`.

Artifact route:

- producer hash: `dd119afa0ef41c8dbf5de6ec365f8c04fd3b7018553dee3cf179bdde99ae8682`;
- gate/up hash: `9d00b0723a6aa92d54f18e152678352d6b19d04ace9cbf605637c6abcf0287a5`;
- `16` dot4 ops;
- fused gate/up consumer: `93.54us`;
- producer + gate/up lifecycle: `115.24us`;
- no in-process HIP runtime.

## Boundary

Supported:

- Qwen3-8B Q4_K_M-style dense FFN block;
- `dim=4096`, `hidden=12288`;
- Q4_K gate/up weights;
- q8_1 activation side-channel;
- gfx1100;
- tinygrad AMD HCQ / `AMDProgram`.

Non-goals:

- not a default route;
- not a portable backend feature;
- not validated for other hidden sizes, quant formats, or GPU archs;
- not a replacement for native scheduler/codegen ownership.

## Decision

The small path is complete as a research flag.

Optional future hardening is policy-driven, not required for the current decode conclusion:

- multi-window dNLL or task eval before any non-research use;
- shape/architecture portability only if leaving Qwen3-8B/gfx1100;
- native compiler ownership only if funding the broader AMD scheduler/codegen project.
