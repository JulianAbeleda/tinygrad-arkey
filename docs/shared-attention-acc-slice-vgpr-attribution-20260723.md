# Shared-attention accumulator-slice ISA/VGPR attribution

Date: 2026-07-23

## Question and evidence boundary

This report compares the same causal `gfx1100` attention geometry (`Hq=8`, `Hkv=2`, `Q=32`, `KV=64`, `Hd=128`) in three compile-only variants:

| Variant | Output blocks | HIP VGPR | SGPR | LDS | Scratch/spills |
|---|---|---:|---:|---:|---:|
| full | `0..7` | 254 | 26 | 512 B | 0 |
| low | `0..3` | 244 | 26 | 512 B | 0 |
| high | `4..7` | 245 | 26 | 512 B | 0 |

The committed artifact contains the final HIP source, tinygrad AMD-ISA program, and parsed HIP binary resource metadata for each descriptor. The descriptors were compiled but never launched; no replay was run.

Two allocation domains must not be conflated:

- `hip_resources.vgpr` is LLVM/clang metadata from the compiled HIP binary. It is the authoritative allocation for `254/244/245`.
- The `vN` references below are the explicit registers in tinygrad's final AMD-ISA program. They provide exact semantic ownership and instruction attribution, but are not the HIP compiler's physical-register numbering.

Consequently, `HIP VGPR - (highest AMD-ISA vN + 1)` is only a cross-domain numerical margin. It is not a count of unused physical registers.

## Exact high-water comparison

| Variant | AMD-ISA highest reference | Distinct referenced VGPRs | Referenced ranges | HIP allocation |
|---|---:|---:|---|---:|
| full | `v234` | 203 | `v0:v95`, `v128:v234` | 254 |
| low | `v234` | 171 | `v0:v39`, `v72:v95`, `v128:v234` | 244 |
| high | `v234` | 203 | `v0:v95`, `v128:v234` | 245 |

All variants have the same explicit high-water register. The numerical margins are 19, 9, and 10 respectively, but cannot be interpreted as physical dead ranges because the HIP and AMD-ISA allocators differ.

`v234` is not a PV accumulator. In every variant it is referenced by the online-softmax rescale/mask sequence, specifically `v_cndmask_b32_e32` and `v_mul_f32_e32`. The immediately lower tail has these roles:

| Range | Instruction families/stage | Survives both halves? |
|---|---|---|
| `v233` | row reduction and rescale (`ds_bpermute_b32`, add/mul/cndmask), address arithmetic, and V `global_load_u16` staging | Yes |
| `v232` | address arithmetic plus `ds_load_b128` before PV | Yes |
| `v228,v230` | LDS probability publication plus max/reduction arithmetic | Yes |
| `v227,v229,v231` | online max/exp/rescale arithmetic | Yes |
| `v216:v226` | QK/softmax conversion, LDS publication, and V load/pack temporaries | Yes |

Thus halving PV ownership does not touch the explicit high-register tail `v216:v234`.

## Fixed ABI versus generated ranges

| Range | Ownership | Full | Low slice | High slice |
|---|---|---|---|---|
| `v8:v39` | fixed PV C blocks 0..3 in full/low | PV WMMA + normalize/store | PV WMMA + normalize/store | Not PV C; reused by normalize/store epilogue |
| `v40:v71` | fixed PV C blocks 4..7 | PV WMMA + normalize/store | Completely unreferenced | PV WMMA; result is staged through the low bank for drain |
| `v72:v79` | fixed online maximum `m` | Present | Present | Present |
| `v80:v87` | fixed online normalization `l` | Present through reciprocal/drain | Present | Present |
| `v88:v95` | fixed QK C fragment | 8 QK WMMAs | Same 8 QK WMMAs | Same 8 QK WMMAs |
| `v96:v127` | reserved gap in this generated program | Unreferenced | Unreferenced | Unreferenced |
| `v128:v199` | allocator-assigned address/load/mask/exp/reduction temporaries | All 72 referenced | All 72 referenced | All 72 referenced |
| `v200:v207` | fixed/reused WMMA A fragment | QK and PV | QK and PV | QK and PV |
| `v208:v215` | fixed/reused WMMA B fragment | QK and PV | QK and PV | QK and PV |
| `v216:v234` | allocator-assigned softmax/repack/V staging | All 19 referenced | All 19 referenced | All 19 referenced |

The current program does not reference `v96:v103`; the prior ledger's attribution of an alpha lease to that range is not true for this exact capture. Alpha/rescale values exist, but are assigned within generated temporary ranges instead of that fixed gap.

## Stage attribution

The full program emits 16 WMMAs: QK at lines 119 and 327-333, then PV at lines 1003, 1006, ..., 1024. Both sliced programs emit 12 WMMAs: the same eight QK operations and four PV operations.

- QK stage: `v88:v95` C, `v200:v207` A, `v208:v215` B.
- Online-softmax stage: persistent `v72:v87` plus generated `v216:v234`; this stage owns the highest reference.
- Probability publication and V preparation: `ds_store_b16`, `ds_bpermute_b32`, `ds_load_b128`, `global_load_u16`, and `v_pack_b32_f16` reuse `v200:v233`.
- PV stage: four or eight fixed eight-register C blocks plus reused `v200:v215` operands.
- Drain: `v80:v87` reciprocals and output normalization. Low consumes `v8:v39`; high also references `v8:v39` for epilogue staging even though its PV WMMAs target `v40:v71`.

The low slice is the clean structural deletion: `v40:v71` disappears completely, yet HIP allocation falls by only 10. The high slice deletes the same number of semantic PV values but retains textual references across `v0:v95` and saves only 9. This proves register pressure is controlled by overlapping non-PV temporaries and phase lifetimes, not the fixed PV-C count alone.

## Ranked next reducible ownership sets

1. **Generated online-softmax/repack/V tail, `v216:v234` (19 explicit registers).** It contains the high-water reference and survives unchanged in both halves. Split or reschedule probability reduction/publication and V packing so this whole semantic tail is not live across PV accumulation.
2. **Generated address/load/mask/exp region, `v128:v199` (72 explicit registers).** Every register remains referenced after slicing. It spans prologue, loop, and epilogue, so shorten lifetimes or recompute indices rather than reserve another fixed bank.
3. **Coexistence of `m/l` and QK C, `v72:v95` (24 fixed registers), with PV C.** A score-state/PV phase boundary is useful only if it removes this simultaneous ownership from the PV pass; renumbering it is not a reduction.
4. **High-slice drain staging through `v8:v39`.** Directly drain `v40:v71` or use a bounded transient. This is worth fixing for symmetry, but the observed low/high allocation difference bounds the likely isolated gain to about one HIP VGPR, so it is not the primary roofline lever.
5. **WMMA operands `v200:v215` (16 fixed/reused registers).** They are required by both contractions. Optimize their lifetime only after the generated tail and phase coexistence are removed; deleting them would require a different fragment ABI.

## Falsifiable next gate

The next score-state/PV ownership implementation passes this resource hypothesis only if both four-block PV variants prove all of the following from fresh compile-only captures:

1. HIP metadata allocation is `<=192 VGPR` for both low and high, with zero scratch and zero VGPR/SGPR spills.
2. The AMD-ISA semantic census no longer assigns the eliminated online-softmax/repack/V coexistence to the complete `v216:v234` tail; an unchanged `v234` high-water with the same instruction-family census rejects the claimed lifetime split even if registers were cosmetically renamed.
3. PV kernels do not simultaneously own loop-carried `m/l`, QK C, four PV C fragments, and the current V staging set. The report must identify the new owner of each state or explicit recomputation.
4. Low and high output ownership remains exactly disjoint and complete over blocks `0..7`; high-slice epilogue staging is separately recorded.

`<=192` requires a 52/53-register reduction from the measured slices and is intentionally stronger than the 10-register gain already obtained. If either slice remains above 192, or the `v216:v234` semantic tail survives unchanged, the proposed phase split has not broken the pressure mechanism and should not proceed to replay under that claim.

## Artifact index

- `docs/artifacts/shared-attention-acc-slice-vgpr-attribution-20260723/summary.json`
- `docs/artifacts/shared-attention-acc-slice-vgpr-attribution-20260723/full.amdisa.s`
- `docs/artifacts/shared-attention-acc-slice-vgpr-attribution-20260723/low.amdisa.s`
- `docs/artifacts/shared-attention-acc-slice-vgpr-attribution-20260723/high.amdisa.s`
- Matching `full.hip.cpp`, `low.hip.cpp`, and `high.hip.cpp` preserve the compiled source used to obtain HIP resource metadata.
