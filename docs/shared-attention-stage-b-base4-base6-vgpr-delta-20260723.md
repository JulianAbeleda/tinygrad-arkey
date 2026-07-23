# Stage B base4/base6 physical VGPR delta

Date: 2026-07-23
Source commit: `9f87563c5`

## Exact result

Compile-only HIP metadata reports Stage B base4 at 192 VGPR and base6 at 197 VGPR. Physical HIP-binary disassembly shows this is not five additional referenced VGPR values:

- base4 highest reference is the main V address `v190:v191`.
- base6 highest reference is the main V address `v195:v196`.
- base6 does not reference `v192:v194`.

Therefore the `197 - 192 = 5` delta is allocator high-water/lifetime placement. It must not be entered in the live-state ledger as five semantic values.

## Producer/consumer attribution

Base4 uses one V address pair:

- `0x1f28`: `v_lshlrev_b64 v[190:191], 1, v[96:97]`
- `0x1f4c`: `v_add_co_u32 v190, vcc_lo, s22, v190`
- `0x1f5c`: `v_add_co_ci_u32_e64 v191, null, s23, v191, vcc_lo`
- `0x1f78..0x2070`: all 32 V half-loads consume `v[190:191]`; their immediate offsets fit.

Base6 needs three concurrent V address pairs because its block-6/7 offsets cross the immediate-address boundary:

- adjusted pair A: `v86:v87` is shifted into `v101:v102`, then based with `s22:s23` at `0x1f5c..0x1f80`; `global_load_d16_hi_b16 v88, v[101:102], off` consumes it at `0x1fa8`.
- adjusted pair B: `v90:v91` is shifted in place, then based at `0x1f64..0x1f94`; `global_load_d16_hi_b16 v92, v[90:91], off` consumes it at `0x1fb0`.
- main pair: `v195:v196` is based from `v88:v89` at `0x1f9c/0x1fc0`; the remaining 30 V half-loads consume it at `0x1ff8..0x20e0`.

The semantic live-state increase is two extra 64-bit adjusted addresses, four address components. The fifth allocated slot is placement/fragmentation caused by the changed schedule: the QK/PV fragment region shifts upward, while the final physical range contains the unreferenced hole `v192:v194` before `v195:v196`.

## Why base6 creates the boundary case

The generated HIP source differs only in constants and the kernel name:

- V block element bases change from `+64/+80` to `+96/+112`.
- output element offsets all change by `+32`, corresponding to a `+64` byte shift from Hd blocks 4/5 to blocks 6/7.

Base4 can encode every V load from one base address with an immediate offset. Base6 has two boundary loads that LLVM lowers through separately adjusted addresses, which lengthens address lifetimes across the adjacent WMMA preparation.

## Drain and output mapping

The drain is not the five-VGPR source. Both variants use output address `v16:v17` and the same data registers `v0:v8`.

- base4 store byte offsets begin `128,160,640,672,...` and end `3712,3744`.
- base6 offsets begin `192,224,704,736,...` and end `3776,3808`.
- every base6 store offset is the corresponding base4 offset plus 64 bytes, exactly mapping output blocks 6/7 instead of 4/5.

No additional drain address pair or high drain data range appears.

## Live-state ledger action

Record this row:

| Ownership | Base4 | Base6 | Action |
|---|---|---|---|
| V global-load addresses at PV preparation | one 64-bit pair, `v190:v191` | main `v195:v196` plus adjusted `v101:v102` and `v90:v91` | rematerialize/serialize the two boundary loads or normalize the V base before the clause; do not change output drain |

A falsifiable fix must compile base6 at `<=192 VGPR`, retain zero spills/scratch, and remove simultaneous ownership of both adjusted address pairs. Merely renumbering the main pair below `v195:v196` without removing the two special address paths does not pass.

No renderer source was edited and no kernel was launched or replayed for this attribution.
