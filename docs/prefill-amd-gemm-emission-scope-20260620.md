# Prefill AMD GEMM Emission — Scope (first structural ISA emission)

Date: 2026-06-20

## The one question this answers

Can the renderer **emit** a structurally Tensile-shaped GEMM loop — actual ISA assembled into an ELF — that
matches the lowering plan?

**Answer: YES, structurally.** Verdict `PASS_GEMM_STRUCTURAL_EMISSION`. A minimal label/branch resolver plus a
hand-emitted unrolled-by-2 K-loop produces a real instruction stream (232 insts) that assembles via
`tinygrad/renderer/amd/elf.py:assemble_linear` into a **1879-byte ELF** with **`group_segment_fixed_size =
25088`** and **`private_segment_fixed_size = 0`**, containing every required op class with alternating LDS
slot offsets and a validated backward branch.

This is **STRUCTURAL ONLY**. It is explicitly **not runnable, not correct, not fast** — the global addresses
are placeholders. It proves the *shape* is emittable; correctness and timing come only after.

This builds **no production kernel**, changes **no routing/defaults**, makes **no performance claim**, runs
**no** BEAM/search, and does **no** timing.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_emission_probe.py` | branch resolver + structural K-loop emission + assemble + inspect |
| `bench/amd-broad-backend-roadmap/amd_gemm_emission_result.json` | emitted-ISA evidence (`bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_emission_probe.py
```

Inputs: `amd_gemm_lowering_plan_result.json`, `amd_gemm_schedule_object_structural_result.json`, the encoder
`tinygrad/renderer/amd/elf.py:assemble_linear`, and the RDNA3 primitives in
`tinygrad.runtime.autogen.amd.rdna3.ins`.

## 1. Label / branch-offset resolver

`assemble_linear` is a straight-line encoder with no label table. The probe adds a minimal two-pass resolver
over a stream of `("inst",·) | ("label",name) | ("branch",name,build_fn)` items:

- **Pass 1** assigns byte offsets by summing `len(inst.to_bytes())` (labels are zero-size; branches are 4 B).
- **Pass 2** computes `simm16 = (target_byte − (branch_byte + 4)) // 4`, asserts dword alignment and 16-bit
  range, fills `s_cbranch_scc0(simm16 & 0xFFFF)`, then **validates** by decoding the emitted bytes back and
  checking `branch_byte + 4 + decoded·4 == target_byte`.

Measured result (the loop's single backward branch):

| field | value |
|---|---|
| branch byte | `1240` |
| target byte (`loop_head`) | `556` |
| `simm16` | `−172` |
| backward | `true` |
| decode round-trips | `true` |

This is exactly the byte-offset capability the lowering plan flagged as `to_build`; it is now built and
validated.

## 2. Emitted unrolled-by-2 K-loop (structural)

`loop_head:` then two sub-iterations with alternating slots, closing on the backward branch:

| sub-iter | read slot | write slot | emitted ops |
|---|---|---|---|
| A | 0 (A0/B0) | 1 (A1/B1) | `s_waitcnt lgkm → s_barrier → 8×global_load_b128 → 8×ds_load_b128(slot0) → s_waitcnt vm → 8×ds_store_b128(slot1) → s_waitcnt lgkm → 16×v_wmma → s_sub` |
| B | 1 (A1/B1) | 0 (A0/B0) | same shape, slots swapped, `→ s_sub → s_cmp_eq_i32 → s_cbranch_scc0 loop_head` |

Emitted instruction counts (from the assembled stream):

| op class | count |
|---|---:|
| `global_load_b128` | 16 |
| `ds_store_b128` | 16 |
| `ds_load_b128` | 16 |
| `v_wmma_f32_16x16x16_f16` | 32 |
| `s_waitcnt` | 8 |
| `s_barrier` | 2 |
| `s_sub_u32` (counter) | 2 |
| `s_cbranch_scc0` (backward) | 1 |
| `global_store_b128` (epilogue) | 1 |
| total | 232 |

## 3. Fixed register ledger (authority shape)

| role | allocation |
|---|---|
| thread id + address VGPRs | `v[0:7]` (placeholder addressing) |
| A/B global-load temps | `v[8:39]` (32 VGPR = 8×`global_load_b128`) |
| accumulator fragments | `v[64:191]` (16×8 = 128 VGPR; one wave's 16 WMMA outputs) |
| A/B LDS-read fragments | `v[192:223]` (32 VGPR = 8×`ds_load_b128`) |
| loop counter SGPRs | `s[16]` (K counter), `s[17]` (saved init) |
| kernarg / pointers | `s[0:1]` base; `s[4:5]=A`, `s[6:7]=B`, `s[8:9]=C` |
| scratch / private | `0` (no `DEFINE_REG` spills) |

Fit: **max VGPR 224 / 256**, **max SGPR 18 / 106** — within budget, no spill. (Authority SGPR budget is 58;
the structural emission uses 18.)

## 4. Slot alternation in the encoded offsets

The DS 16-bit offset is `offset1<<8 | offset0`; slot bases are multiples of 256, so the slot lives in the
`offset1` high byte: slot 0 → `{0x00 (A0), 0x10 (B0)}`, slot 1 → `{0x40 (A1=16384), 0x50 (B1=20480)}`. Both
families are present in the emitted `ds_store`/`ds_load` stream, and the two sub-iterations swap read/write
slots — so slot alternation is visible at the **encoded-byte** level, not just in source.

## 5. Structural gates (all 12 pass)

| gate | result |
|---|---|
| visible `global_load` | ✅ (16) |
| visible `ds_store` | ✅ (16) |
| visible `ds_load_b128` | ✅ (16) |
| visible `v_wmma` | ✅ (32) |
| visible `s_waitcnt` | ✅ (8) |
| visible `s_barrier` | ✅ (2) |
| visible loop counter decrement | ✅ (2) |
| visible backward branch | ✅ (validated, simm16 −172) |
| slot alternation offsets present | ✅ |
| LDS bytes 25088 | ✅ (from ELF descriptor) |
| scratch/private 0 | ✅ (from ELF descriptor) |
| no performance claim | ✅ |

The verdict maps emission failures to a precise blocker: opcode encode failure →
`BLOCKED_GEMM_EMISSION_OPCODE`; branch resolution failure → `BLOCKED_GEMM_EMISSION_BRANCH_RESOLUTION`; VGPR/SGPR
over budget → `BLOCKED_GEMM_EMISSION_REGISTER_ALLOCATION`; assembly failure →
`BLOCKED_GEMM_EMISSION_ADDRESS_MODEL`. None triggered.

## What is explicitly NOT claimed

- **Not runnable**: `addressing_mode = STRUCTURAL_EMISSION_ONLY`; global addresses are placeholders, not the
  real per-thread tiled A/B/C arithmetic. The kernel must not be launched as-is.
- **Not correct**: fragment↔WMMA operand mapping and accumulator↔output indexing are structural, not verified.
- **Not fast / not timed**: no performance number, by rule.
- **Not bit-exact Tensile**: non-bitexact slot/offset model, as scoped.

## Verdict

`PASS_GEMM_STRUCTURAL_EMISSION` — the renderer emits a structurally Tensile-shaped GEMM loop that assembles to
a real ELF with the correct LDS/scratch envelope, op classes, slot alternation, and a validated backward
branch. The shape is real and inspectable.

## Next (gated; this pass does not authorize a runnable kernel)

1. Build the **address-expression model** (real per-thread tiled A/B/C global addresses from `WG[32,4,1]` /
   `TT[4,64]` + kernarg strides) to make the emitted loop runnable, plus the correct fragment↔WMMA operand and
   accumulator↔output mapping.
2. Verify **correctness** (relative RMSE vs `a@b`) under the structural gate — still no timing.
3. Only after correctness, **time** vs the `≥60 TFLOPS` pure-tinygrad authority under the PTM-1 interleaved
   one-clock harness.

Order stays **contract → K-loop → lowering plan → emission → (address model + correctness) → timing →
search**, with BEAM still out of the picture until correctness and timing exist.
