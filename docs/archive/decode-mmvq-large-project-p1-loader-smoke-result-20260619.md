# Decode MMVQ large project P1 loader smoke result - 2026-06-19

Purpose: execute P1 from `decode-mmvq-large-project-scope-20260619.md`.

No kernels were launched. No model route or default changed.

Artifacts:

- `extra/qk_decode_mmvq_p1_loader_smoke.py`
- `bench/qk-decode-mmvq-large-project/p1_loader_smoke.json`
- `bench/qk-decode-mmvq-large-project/p1_loader_smoke_summary.md`

## Verdict

`PASS`.

Selected llama.cpp Q4_K and Q6_K MMVQ descriptors load through tinygrad's AMD HCQ path without HIP runtime and without
launching a kernel.

## Loaded Targets

| type | ncols | bools | VGPR | SGPR | LDS | kernarg | wave |
|---|---:|---|---:|---:|---:|---:|---:|
| Q4_K | `1` | `0/0` | `23` | `24` | `0` | `144` | `32` |
| Q6_K | `1` | `0/0` | `26` | `24` | `128` | `144` | `32` |

Loader details:

- Q4_K descriptor offset: `0x74840`;
- Q6_K descriptor offset: `0x74e40`;
- descriptor kernarg size matches metadata: `144`;
- no unsupported relocations;
- no dispatch pointer SGPR;
- no private segment SGPR;
- no kernel launch.

## Why This Matters

P0 proved that llama.cpp's built gfx1100 object contains descriptors and metadata. P1 proves tinygrad can actually load
selected descriptors through HCQ. That removes the first source/object import kill gate.

The next unknown is no longer object loading. It is the real launch contract:

```text
What exact 144-byte kernarg, grid, local size, and fusion flag combination does llama use for the Qwen3-8B roles?
```

## Next Phase

P2: kernarg and launch capture.

Gate:

- capture one Q4_K and one Q6_K llama launch in a separate HIP-only process;
- record symbol, grid, local size, and all `144` kernarg bytes;
- classify buffer pointer slots vs scalar slots;
- no in-process HIP inside tinygrad;
- no model route changes.

Only after P2 should P3 attempt an HCQ launch with tinygrad-owned buffers.
