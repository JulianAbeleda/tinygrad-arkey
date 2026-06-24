# Prefill AMD GEMM K-loop Reconstruction — Scope

Date: 2026-06-20

## The one question this answers

Can we turn the structural GEMM schedule object into a **lowerable repeated K-loop template**?

**Answer: YES.** Verdict `PASS_KLOOP_TEMPLATE_RECONSTRUCTED_FOR_LOWERING`. The selected rocBLAS ffn_gate/up
function disassembles into a recoverable CFG whose main summation loop is a clean **unrolled-by-2,
double-buffered software pipeline** with alternating LDS slots and ordered vmcnt/lgkmcnt/barrier dependency
edges. Lowering is the next step; the branch does **not** need a new CFG/address tool to proceed.

This makes **no performance claim**, changes **no routing/defaults**, builds **no kernel**, and runs **no**
BEAM/search. It reconstructs structure from the captured disassembly only.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_kloop_reconstruction_probe.py` | parses the selected function, resolves the CFG, extracts the symbolic K-loop template, emits the gate |
| `bench/amd-broad-backend-roadmap/amd_gemm_kloop_reconstruction_result.json` | machine-readable reconstruction (`bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_kloop_reconstruction_probe.py
```

Inputs: `ffn_gate_up_contract.json`, `ffn_gate_up_schedule_template.json`,
`bb5a10_tensile_layout_audit_result.json`, and the captured disassembly `/tmp/td_all.txt` (selected function
lines `282071..289317`). If the disasm is absent the probe degrades to
`BLOCKED_KLOOP_TEMPLATE_NEEDS_DISASM_OR_CFG` and names the missing artifact rather than guessing.

## 1. Regions (precise CFG, not the first/last-WMMA heuristic)

The prior segmentation was "before/through/after the first/last `v_wmma`." This pass recovers the real
control flow from labels plus a **resolved backward branch**: `s_cbranch_scc0 LoopBeginL_1` at file line
`282707` encodes `simm16=0xFE28` → target address `0x2f4ca8` → file line `282442` (the loop head, which the
disassembler also names `label_0013`).

| region | bounds / label | role |
|---|---|---|
| kernel prologue | `282071` → `ShadowInitStart_10` | args, alpha check, address setup |
| shadow-init prefetch | `ShadowInitStart_10` (local 194) | PGR fill: prefetch first K tiles into LDS slot 0 |
| loop preheader | `openLoopL_12` / `label_0013` (local 367–371) | s5 compare, enter loop |
| **main K-loop body** | head `282442` → branch-back `282707` | **steady state, unrolled ×2, hardware loop on `s5`; 2 WMMA clusters** |
| even/odd exit | `LoopEndL_evenexit_4` / `LoopEndL_oddexit_3` (local 639/642) | drain, `v_xor 0x4000` slot swap |
| steady drain | `LoopEndL_2` → `Summation_End_OptNLL_16` (local 646–759) | final-iteration drain; 1 WMMA cluster |
| OptNLL | `OptNLL_End_15` (local 1264) | optimized no-load-loop last iteration (no global prefetch); 1 cluster |
| tail loop | `TailLoopBeginL_6`..`TailLoopEndL_7` (local 1415–1671), `s5 -= 16` | K-remainder loop; 1 cluster |
| global-write epilogue | `GW_*` / `label_GW_End_21` | beta·C load + store output |

## 2. Symbolic repeated K-loop template (lowerable)

The main loop body is **unrolled by 2** (PLR1 double-buffer). Each sub-iteration reads one LDS slot and
writes the *other*, so the two sub-iterations alternate slots and the loop carries no copy. Slot identity is
the LDS byte offset: slot 0 = low buffer (`< 8192 B`, offsets `0..2352`), slot 1 = second buffer
(`>= 16384 B`, offsets `16384..18736`) — exactly the A0/B0/A1/B1 layout the schedule object already encodes.

| sub-iter | global loads | LDS read | LDS write | WMMA | observed phase order |
|---|---:|---|---|---:|---|
| **A** | 8 ×`buffer_load_b64` | 8 ×`ds_load_b128` ← **slot 0** | 8 ×`ds_store_b64` → **slot 1** | 16 | `wait_lgkmcnt → barrier → global_load → lds_read(slot0) → (vmcnt→lds_store(slot1))×8 → wait_lgkmcnt → wmma×16 → s5−1` |
| **B** | 8 ×`buffer_load_b64` | 8 ×`ds_load_b128` ← **slot 1** | 8 ×`ds_store_b64` → **slot 0** | 16 | `wait_lgkmcnt → barrier → global_load → lds_read(slot1) → (vmcnt→lds_store(slot0))×8 → wait_lgkmcnt → wmma×16 → s5−1 → branch` |

Loop counter: `s5 = SizesSum(K) >> log2(DepthU) = K // DepthU = 256`, decremented once per sub-iteration,
exit when `s5 == 1`; the tail loop steps `s5 -= 16`.

### Ordered dependency edges (one sub-iteration)

1. `lds_read[prev]` → **barrier** via `lgkmcnt` — all waves finish reading a buffer before it is overwritten.
2. `global_load[next_k]` → `lds_store[other_slot]` via `vmcnt` — store only after the global load lands.
3. `lds_store[other_slot]` → `lds_read[other_slot]@next_iter` via **barrier** — next iter reads after store completes.
4. `lds_read[this_slot]` → `wmma` via `lgkmcnt` — WMMA operands must be LDS-loaded VGPRs.
5. `wmma` → `kcounter_dec / branch` — loop control after compute issue.

These five edges + the two alternating slots are the lowering contract: a renderer can emit the body from
this without the per-element address map.

## 3. Why `v_wmma = 80`, not `256`

`v_wmma = 80` is **static code size**, not the dynamic K-slice count. The function emits **5 WMMA clusters of
16**, one per distinct scheduled region:

| cluster | region | n |
|---|---|---:|
| 1 | `label_0013` (main loop, even half) | 16 |
| 2 | `label_0014` (main loop, odd half) | 16 |
| 3 | `LoopEndL_2` (steady drain) | 16 |
| 4 | `OptNLL_End_15` (optimized no-load last iter) | 16 |
| 5 | `TailLoopBeginL_6` (tail loop) | 16 |

Each cluster of **16 WMMA = one wave's full 128×128 output update for one `DepthU=16` K-slice** (the wave owns
a 64×64 sub-tile = `(64/16)²` = 16 of the 16×16×16 fragments; `DepthU=16` = one MI K-step ⇒ one WMMA per
fragment). The **symbolic K-loop dynamically executes 256 slices** (`K // DepthU`) through the `s5` hardware
loop — the main body (32 WMMA, 2 slices) repeats ~127× plus pipeline fill/drain/tail. So `80 ≠ 256`: 80 is
the emitted footprint across rolled regions, not the trip count. `sum(clusters) = 80` matches the audit's
`v_wmma` exactly.

## 4. Opcode evidence per symbolic phase

| symbolic phase | opcode |
|---|---|
| global load (next K) | `buffer_load_b64` (GLVW4 fp16 pairs) |
| wait before LDS store | `s_waitcnt vmcnt` |
| LDS store (other slot) | `ds_store_b64` |
| buffer-reuse barrier | `s_barrier` |
| LDS read (this slot) | `ds_load_b128` (LRVW16 operand fragments) |
| wait before WMMA | `s_waitcnt lgkmcnt` |
| WMMA consume | `v_wmma_f32_16x16x16_f16` |
| slot alternation | LDS offset (slot0 `<8192 B` vs slot1 `>=16384 B`); `v_xor 0x4000` in the even/odd drain |

## 5. What remains unknown (explicit)

- **Per-element address VGPR evolution** — A/B base offsets are carried in address VGPRs (not full immediates);
  not reconstructed. Lowering can use the **structural slot model** (the offset families above) instead, but a
  *bit-identical* Tensile clone cannot be claimed.
- **Branch/loop-counter micro-structure beyond `s5`** — even/odd PLR exits and OptNLL guard predicates are
  labeled but not symbolically modeled; a lowering can roll its own counter.
- **Bank-conflict rationale beyond `LdsPadB=8`/128 B** — not replayed from the Tensile generator.
- **Tail-loop dynamic entry for K=4096** (a clean multiple of DepthU) — emitted but may be skipped at runtime;
  not evaluated here.

None of these block lowering: they block a *bit-exact clone*, which we already declared out of scope
(non-bitexact native schedule object). The first item is the only one a lowering must consciously substitute
(structural slot model in place of the exact address arithmetic).

## 6. Gate

`PASS_KLOOP_TEMPLATE_RECONSTRUCTED_FOR_LOWERING` — all four lowering checks pass:

| check | result |
|---|---|
| alternating LDS slots (A:0→1, B:1→0) | ✅ |
| ordered dependency edges (global_load, lds_store, lds_read, wmma, barrier per sub-iter) | ✅ |
| hardware loop counter recovered (`s5`, unroll ×2) | ✅ |
| WMMA clusters sum matches audit (`5×16 = 80`) | ✅ |

The alternative `BLOCKED_KLOOP_TEMPLATE_NEEDS_DISASM_OR_CFG` is emitted only if the disasm is missing or the
loop-head branch cannot be resolved — neither is the case here.

## Lowering readiness — verdict

**Ready for lowering.** The symbolic template (unrolled-by-2 body, two alternating slots, five ordered
dependency edges, recovered `s5` counter) is sufficient to drive a renderer-side lowering of the selected
GEMM schedule object. The remaining unknowns are bit-exactness details a non-bitexact lowering substitutes,
not blockers.

## Next (still gated, not authorized here)

1. **Lower** the symbolic template through the `AMDGemmScheduleObject` path: emit the unrolled-by-2 body with
   the A0/B0/A1/B1 slot offsets and the five ordered vmcnt/lgkmcnt/barrier edges; this is the net-new K-loop
   scheduler capability named in `prefill-sw-pipeline-codegen-charter-20260620.md` (the codegen wall) — now
   with a concrete, validated target.
2. Gate the lowered output through the existing structural gate (`amd_gemm_schedule_object_structural`).
3. Only then time against the `≥60 TFLOPS` pure-tinygrad authority under the PTM-1 interleaved one-clock
   harness. **No BEAM/search until the schedule object lowers to ISA.**
