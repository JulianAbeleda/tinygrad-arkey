# MMVQ deep-linearizer — Phase 4-6 microkernel + scheduler assessment + VERDICT (2026-06-18)

Built the Q4_K block-dot microkernel through the new first-class `_sdot4` op and measured it. **Verdict: B/C —
first-class dot4 is achieved at the codegen layer (native v_dot4) but does NOT lift the ~52% ceiling; the binding
layer is scheduling/scale-hoist/register, not the dot4 representation.** RX 7900 XTX, Q4_K ffn_gate/up.

## Phase 4/5 — microkernel performance (% HBM peak)

| representation | % peak | dot4 | notes |
|---|---|---|---|
| base fp | 40 | no | |
| fp coop | 48 | no | coalesced fp dequant |
| pure-UOp int reduce | 2.2 | scalarized | (no auto-dot4) |
| visible `_dp4a` udot4 | 46 | native | unsigned + bias overhead |
| opaque asm signed dot4 | 52 | native | prior custom_kernel (asm volatile) |
| **first-class `_sdot4` helper microkernel** | **49** | **native v_dot4_i32_iu8** | non-volatile, renderer-owned |
| llama / READRAW | 70 | native | target |

The first-class `_sdot4` microkernel reaches **49% (1.31× base, 1.13× fp coop)** — **does not beat the opaque
asm 52%.** Native dot4, non-volatile (compiler-schedulable), renderer-owned — yet the same ceiling.

**Correctness caveat (honest):** the `_sdot4` *helper* is disasm-validated (emits native v_dot4_i32_iu8, test
passes). The block *microkernel* has an unresolved rel-err ≈5.4 vs base (a scale/pack-convention bug; the
reference pure-UOp `intdot` kernel with the same convention is correct at rel 0.006, so the bug is in the
microkernel's partial-affine/pack, not the dot instruction). The **perf** result (49%) is structure-determined —
the kernel does identical dot4 work regardless of the value bug — so the ceiling finding holds.

## Phase 6 — scheduler assessment

Why 49% (≈52%) not 70%, despite native schedulable dot4:
- **dot4 count / extract**: correct (8 native v_dot4 dot + 8 qsum per block, packed `0x0F0F0F0F` extract).
- **repeated scale decode**: each of the 8 lanes recomputes `_q4k_group_params` for all 8 groups → ~8× redundant
  6-bit scale decode. llama decodes scales once per group/warp. **Not hoisted.**
- **accumulators / unroll**: the per-lane `psd`/`psm` + the stage-2 `.sum` reduction + the partials buffer add
  overhead vs llama's single register-tight accumulator chain over the row.
- **occupancy**: the coop lane decomposition + partials round-trip.

So: **the dot4 representation is solved and is NOT the bottleneck** (49% with native schedulable dot4 ≈ 52% with
opaque asm ≈ 48% fp coop). The 49→70% gap is the **scheduling/scale-hoist/register layer**.

## Verdict: B/C — dot4 foundation built; the next layer is scale-hoist + register/unroll

The arc made the next step un-hand-wavy by localizing the failure:
- **UOp representation**: int-reduce scalarizes (no auto-dot4) — known.
- **renderer lowering / ISA**: SOLVED — `_sdot4` helper emits native `v_dot4_i32_iu8` (the only RDNA3 signed path;
  the builtin scalar-fallbacks).
- **linearizer scheduling / register allocation / scale-hoist**: **THIS is where 49→70% is lost** — redundant
  per-lane scale decode, partials/reduction overhead, no register-tight unroll.

**Next task (if the deep arc continues): a scale-decode hoist transform** (compute each group's 6-bit scale once
per workgroup, shared via LDS or restructured so it isn't recomputed per lane) **and/or a register-tight
single-accumulator row decomposition** (drop the partials/stage-2 reduction). These are real linearizer/codegen
transforms, not dot4 work. **Do NOT** invest more in dot4 representation — it is solved and is not the constraint.

## Recommendation
The dot4 foundation is built and the failing layer is precisely identified (scheduling/scale-hoist). Whether to
fund the scale-hoist/register transforms next is a deep-linearizer investment decision; the prize (~+8-12% decode
if 70% reached) is real but the transforms are nontrivial. The lower-risk alternative remains **14B/32B**. Either
way, this arc delivered durable capability (`_sdot4` renderer lowering + ISA map) and an exact next-layer target.

## Files / commits
`[codegen]` `tinygrad/renderer/cstyle.py` (`_sdot4` helper); `[test]` `test/external/test_sdot4_lowering.py` +
`extra/q4_k_gemv_primitive.py` probes (`_sdot4_op`, `q8_signed_pack_u32_kernel`, `q4k_coop_sdot4_partial_kernel`);
`[docs]` this + `qk-mmvq-deep-linearizer-arc-20260618.md`; `bench/qk-mmvq-deep-linearizer/baseline.json`. No
`[nn]`, no routing, no defaults.
