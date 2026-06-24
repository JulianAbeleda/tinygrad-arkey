# AMD Broad Backend BB-5a.10 Tensile Layout Audit

Date: 2026-06-19

Generator:

- `extra/qk_amd_bb5a10_tensile_layout_audit.py`

Artifact:

- `bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json`

## Verdict

`PASS_TENSILE_LAYOUT_AUDIT_CANDIDATE_SPEC_READY_NOT_BITEXACT`.

We have enough selected-kernel evidence to start the BB-5a.10 pure-tinygrad staged-LDS candidate. We do **not** have
enough to claim a bit-identical clone of Tensile's exact LDS layout.

## Selected Authority Kernel

The audit isolates the selected rocBLAS authority symbol from `/tmp/td_all.txt`:

- shape: `M=512`, `N=12288`, `K=4096`, fp16->fp32 `ffn_gate/up`;
- schedule: `MT128x128x16`, `MI16x16x16x1`, `PGR1`, `PLR1`, `1LDSB0`, `LRVW16`, `TT4_64`, `WGM8`;
- launch: global `[512, 96, 1]`, local `[128, 1, 1]`;
- resource envelope: `VGPR=256`, `LDS=25088`, `scratch=0`;
- isolated disassembly body: lines `282071..289317`, `7247` lines.

Selected-function instruction evidence:

| instruction | count |
|---|---:|
| `buffer_load_b64` | `24` |
| `ds_store_b64` | `32` |
| `ds_store_b128` | `0` |
| `ds_load_b128` | `40` |
| `v_wmma` | `80` |
| `s_waitcnt` | `545` |
| `s_barrier` | `6` |

Correction from the broader oracle wording: this selected rocBLAS authority function uses `ds_store_b64` for
global-to-LDS stores and `ds_load_b128` for LDS-to-WMMA operand reads. Do not require `ds_store_b128` for the first
candidate. The larger extraction corpus contains `ds_store_b128`, but the selected function we are mapping does not.

## Layout Evidence

The audit extracts concrete LDS offset families:

| path | unique offsets | range | representative offsets |
|---|---:|---:|---|
| `ds_store_b64` | `14` | `0..17248` | `0, 256, 512, 768, 288, 576, 864, 16384, 16640, 16672, 16896, 16960, 17152, 17248` |
| `ds_load_b128` | `16` | `0..18736` | `0, 16, 32, 48, 2304, 2320, 2336, 2352, 16384, 16400, 16416, 16432, 18688, 18704, 18720, 18736` |

Register-handoff inference passes:

| inference | result |
|---|---:|
| `buffer_load_b64` data registers later stored to LDS | `32 / 32` LDS stores examined |
| `ds_load_b128` destination registers later feed `v_wmma` source operands | `80 / 80` WMMAs examined |

That is enough to define structural acceptance probes for Tinygrad: nonzero LDS, visible LDS stores, `ds_load_b128`,
WMMA fed by LDS-loaded registers, waits/barriers, and scratch-free resources.

## What Is Still Missing

This audit does not reconstruct Tensile's symbolic tensor-layout source:

- disassembly gives offsets, VGPR ranges, waits, barriers, and handoff windows;
- it does not give the source-level mapping from logical A/B tile coordinates to every LDS byte lane;
- a bit-identical clone would require Tensile generator metadata/source reconstruction, not only selected-kernel ISA.

Therefore BB-5a.10 should implement a **non-bitexact Tensile-class candidate**, not a Tensile clone.

## BB-5a.10 Candidate Scope

Implement the first pure-tinygrad candidate against the selected authority contract:

| track | minimum pass |
|---|---|
| B LDS layout | authority-shape ELF reports nonzero LDS; source/disasm show selected-kernel-compatible LDS stores and `ds_load_b128` reads |
| B wide LDS reads | `ds_load_b128` destination VGPRs feed `v_wmma` source operands |
| C K-loop scheduler | prologue plus steady-state staged movement over `depthU=16`, with producer/consumer LDS stages |
| C waits/barriers | correctness-preserving `vmcnt`/`lgkmcnt` waits and barriers over staged LDS traffic |
| D resource policy | reject scratch/private spills and bad VGPR/SGPR/LDS envelope before timing |

Candidate gate remains unchanged: correctness plus `>=60 TFLOPS` pure tinygrad authority prefill before q8 transfer can
reopen.
