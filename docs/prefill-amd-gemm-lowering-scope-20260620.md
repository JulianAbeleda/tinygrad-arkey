# Prefill AMD GEMM Lowering — Scope (first lowering plan)

Date: 2026-06-20

## The one question this answers

Does a **renderer-side lowering plan** for the reconstructed K-loop exist, and can it be **structurally
validated**?

**Answer: YES.** Verdict `PASS_GEMM_LOWERING_PLAN_READY`. Every symbolic K-loop phase maps to a concrete
RDNA3 ISA op class that already exists in `tinygrad.runtime.autogen.amd.rdna3.ins`; the two LDS slots map to
the A0/B0 and A1/B1 regions; slot alternation and the five dependency edges are preserved; and the resource
invariants hold (LDS 25088, scratch/private 0). The plan is complete enough to implement emission — which is
the next pass, behind this structural gate.

This builds **no production kernel**, changes **no routing/defaults**, makes **no performance claim**, and
runs **no** BEAM/search.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_lowering_probe.py` | loads the K-loop + schedule-object results, builds the lowering plan, runs the structural gate |
| `bench/amd-broad-backend-roadmap/amd_gemm_lowering_plan_result.json` | machine-readable lowering plan (`bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_lowering_probe.py
```

Inputs: `amd_gemm_kloop_reconstruction_result.json` (the symbolic template) and
`amd_gemm_schedule_object_structural_result.json` (LDS regions + resource envelope).

## Lowering plan — unrolled-by-2 K-loop

Each sub-iteration lowers to this ordered ISA op-class sequence (slots alternate between sub-iterations):

| order | phase | ISA op class | edge / note |
|---:|---|---|---|
| 1 | wait before buffer reuse | `s_waitcnt` (lgkmcnt) | drain prior reads before overwrite |
| 2 | barrier | `s_barrier` | buffer-reuse protection across waves |
| 3 | global_load A (next K) | `global_load_b128` | GLVW4 fp16 |
| 4 | global_load B (next K) | `global_load_b128` | |
| 5 | lds_read A (read slot) | `ds_load_b128` | LRVW16 operand fragments |
| 6 | lds_read B (read slot) | `ds_load_b128` | |
| 7 | wait before LDS store | `s_waitcnt` (vmcnt) | store only after global load lands |
| 8 | lds_store A (write slot) | `ds_store_b128` | into the *other* slot |
| 9 | lds_store B (write slot) | `ds_store_b128` | |
| 10 | wait before WMMA | `s_waitcnt` (lgkmcnt) | WMMA operands must be LDS-loaded VGPRs |
| 11 | wmma consume (×16) | `v_wmma_f32_16x16x16_f16` | one wave's 16 output fragments |
| 12 | counter decrement | `s_sub_u32` | `s5 -= 1` |
| 13 | branch (end of body) | `s_cbranch_scc0` | back to loop head; `simm16` byte-offset resolved at emit |

`buffer_swap` is not an opcode: it is **compile-time slot-offset selection** (sub-A reads slot0/writes slot1;
sub-B reads slot1/writes slot0). Every op class above was checked to exist in the autogen RDNA3 ins module.

## Symbolic slot → LDS region map

| slot | regions | byte bases |
|---|---|---|
| slot 0 | A0, B0 | `0`, `4096` |
| slot 1 | A1, B1 | `16384`, `20480` |

Sub-A: read slot0 (A0/B0), write slot1 (A1/B1). Sub-B: read slot1, write slot0. Matches the schedule object's
LDS layout exactly; total stays `25088`.

## Dependency-edge → waitcnt lowering

| edge (from → to) | via | lowered to |
|---|---|---|
| lds_read[prev] → barrier | lgkmcnt | `s_waitcnt` lgkmcnt |
| global_load[next_k] → lds_store[other_slot] | vmcnt | `s_waitcnt` vmcnt |
| lds_store[other_slot] → lds_read[other_slot]@next_iter | barrier | `s_barrier` |
| lds_read[this_slot] → wmma | lgkmcnt | `s_waitcnt` lgkmcnt |
| wmma → counter/branch | wmma_dependency | `s_waitcnt` lgkmcnt (WMMA reads LDS-loaded VGPRs) |

All five reconstructed edges are preserved and given a concrete waitcnt mechanism.

## Emission-capability ledger

The emission path is `assemble_linear` (`tinygrad/renderer/amd/elf.py:14`), a **straight-line encoder** that
concatenates `inst.to_bytes()` and already sizes `group_segment_fixed_size` from `DEFINE_LOCAL` (elf.py:41).
The existing `extra/gemm/rdna3_wmma_matmul.py` proves the LDS/WMMA/waitcnt primitives on this exact path.

| capability | status | basis |
|---|---|---|
| LDS offset lowering | **present** | `DEFINE_LOCAL` → group_segment; `ds_store_b128`/`ds_load_b128` immediate offsets (rdna3_wmma_matmul.py) |
| WMMA operand packing | **present** | `v_wmma_f32_16x16x16_f16(vdst,src0,src1,src2)` over 8-VGPR ranges (rdna3_wmma_matmul.py) |
| waitcnt scheduler | **present** | manual edge-driven `s_waitcnt`; the reconstruction supplies explicit edges (automatic dep-group scheduler is optional future work) |
| VGPR allocation model | **present (fixed shape)** | static hand-allocation suffices for the authority shape (acc 16×8 + A/B frags + addr regs); a general allocator is not required for first emission |
| branch/counter emission | **to build** | `assemble_linear` has no label table; add a minimal byte-offset pass (sum `inst.to_bytes()` sizes between branch and target) to fill `s_cbranch_scc0` `simm16`. Uses existing `s_sub_u32`/`s_cmp`/`s_cbranch`; no new infra |
| address-expression model | **to build** | derive per-thread A/B/C addresses from `WG[32,4,1]`/`TT[4,64]` + kernarg strides via existing `v_add_nc_u32`/`v_lshlrev_b32`/`s_mov`; the structural slot/offset model substitutes the unreconstructed per-element address VGPR evolution (non-bitexact) |
| output store path | **to build (simple)** | `global_store_b128` of the accumulator VGPRs for alpha=1/beta=0 first emission; full beta·C + bounds (GW_* path) deferred |

None of the "to build" items is a **blocker**: each is constructible with existing RDNA3 primitives and the
existing encoder. They are the enumerated work items for the emission pass, not unknowns that stop the plan.

## Structural gates (all pass)

| gate | result |
|---|---|
| all K-loop phases lower to ISA op classes | ✅ |
| slot alternation preserved (A:0→1, B:1→0) | ✅ |
| dependency edges preserved (5/5 → waitcnt) | ✅ |
| slot → LDS region mapped (slot0=A0/B0, slot1=A1/B1) | ✅ |
| LDS bytes remain 25088 | ✅ |
| scratch/private remain 0 | ✅ |
| no missing stage from `AMDGemmScheduleObject` | ✅ |
| no performance claim | ✅ |
| no truly-blocking capability | ✅ |

## Remaining unknown (substituted, not blocking)

- Exact per-element address-VGPR evolution → substituted by the structural slot/offset model (non-bitexact).
- General VGPR allocator → first emission uses fixed hand-allocation for the authority shape.
- Full output epilogue (beta·C + bounds) → first emission targets alpha=1/beta=0.

## Verdict

`PASS_GEMM_LOWERING_PLAN_READY` — the renderer-side lowering plan exists and is structurally validated. The
plan is complete enough to implement emission.

## Next (gated; this pass does not authorize emission)

Implement ISA emission behind the existing structural gate:

1. Build `branch_counter_emission` (the byte-offset resolution pass over `assemble_linear`) and the
   `address_expression_model`; hand-allocate VGPRs; emit the unrolled-by-2 body.
2. Validate the emitted kernel against `amd_gemm_schedule_object_structural` (nonzero LDS, scratch 0, visible
   global_load/ds_store/ds_load_b128/v_wmma, waits + barriers, WMMA fed from LDS) — structural only.
3. Only then time against the `≥60 TFLOPS` pure-tinygrad authority under the PTM-1 interleaved one-clock
   harness.

Order stays **contract → K-loop → lowering plan → emission → timing → search**, with BEAM still out of the
picture until emission exists and passes its structural gate.
