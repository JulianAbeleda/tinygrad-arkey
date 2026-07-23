# LDS PV microgate: 197 versus 192 VGPR attribution

## Result

The five-VGPR excess is real, but the current microgate does **not** show QK/softmax/P-repack values live at the same time as the synthetic LDS PV rotation.

- The microgate high-water mark is the QK fragment window. Its last K load writes `v193:v196` at PC `0x19a0`; the final QK WMMA consumes `v189:v196` at PC `0x1ac0`.
- Synthetic LDS accumulator initialization does not begin until PC `0x2a4c`. Its rotating loads use `v9:v16` and its LDS addresses use `v17`/`v18`, after the attention temporaries have died.
- Therefore, 197 versus 192 cannot be attributed to simultaneous LDS-rotation pressure. The microgate proves that LDS rotation is legal and keeps the aggregate allocation at 197, but not that rotation causes the high-water mark.
- Compared with the corrected 192-VGPR Stage B slice, the extra pressure is attributable to loop-carried online state being made live before the QK fragment preload. Stage B loads final statistics later instead.

Artifacts used for this comparison were physical HIP binaries compiled for `gfx1100`, not tinygrad virtual-register text:

- Current LDS microgate: `/tmp/lds_microgate.co`, `/tmp/lds_microgate.disasm`
- Corrected Stage B, base 4: `/tmp/stage_b_4.co`, `/tmp/stage_b_4.disasm`

Both kernels report 26 SGPRs and zero scratch/spills. The microgate reports 197 VGPRs and 8704 bytes LDS; Stage B reports 192 VGPRs and 512 bytes LDS.

## Exact excess intervals

The following physical values are born before Q/K fragment loading and remain available until the first QK WMMA or online merge. Five such intervals are sufficient to explain the measured delta; all nine are shown because they form one scheduling unit.

| Value | Producer | Last demonstrated consumer | Meaning |
| --- | --- | --- | --- |
| `v73` | zero at `0x1720` | QK C input `v73:v80` at `0x1a5c` | zero-initialized online-`l` / QK-C seed alias |
| `v74` | copy of zero at `0x1738` | QK C input `v73:v80` at `0x1a5c` | same |
| `v75` | copy of zero at `0x1774` | QK C input `v73:v80` at `0x1a5c` | same |
| `v76` | copy of zero at `0x1790` | QK C input `v73:v80` at `0x1a5c` | same |
| `v77` | copy of zero at `0x179c` | QK C input `v73:v80` at `0x1a5c` | same |
| `v78` | copy of zero at `0x1858` | QK C input `v73:v80` at `0x1a5c` | same |
| `v79` | copy of zero at `0x1898` | QK C input `v73:v80` at `0x1a5c` | same |
| `v80` | copy of zero at `0x18a4` | QK C input `v73:v80` at `0x1a5c` | same |
| `v132` | `-inf` at `0x1744` | old/new max compare at `0x1b38` | loop-carried old `m` representative |

The allocator has coalesced zero-valued logical state with the initial QK C fragment, so the first eight rows are physical interval evidence, not a claim that eight independent semantic `l` values are all consumed by that WMMA. They nonetheless occupy eight physical registers throughout K preload (`0x1928` through `0x19a0`). `v132` is a separate, unambiguous old-maximum interval spanning the same region.

At the peak, Q fragments occupy `v9:v72`, K fragments extend through `v196`, and the early online-state/zero-seed intervals remain resident. The corrected Stage B slice instead loads `v109:v116` final statistics at `0x1830..0x1868`; it does not preserve monolithic loop-carried old `m/l` across the equivalent fragment preload. Its highest registers, `v190:v191`, are the late V base address produced at `0x1f28..0x1f5c` and consumed by V loads at `0x1f78..0x2070`.

## P publication and LDS rotation are later phases

The direct attention path publishes P with `ds_store_b16` from `0x1c7c` through `0x264c`, reloads packed P into `v140:v147` beginning at `0x2764`, and consumes it in PV WMMA at `0x29c4`. The synthetic accumulator rotation then starts at `0x2a4c` and processes one reusable `v9:v16` window per block:

| Block | Loads | Stores |
| --- | --- | --- |
| 0 | `0x2acc`, `0x2ad4` | subsequent block boundary |
| 1 | `0x2b68`, `0x2b70` | `0x2bf8`, `0x2c00` |
| 2 | `0x2c08`, `0x2c10` | `0x2ca8`, `0x2cb0` |
| 3 | `0x2cb8`, `0x2cc0` | `0x2d48`, `0x2d50` |
| 4 | `0x2d58`, `0x2d60` | `0x2da0`, `0x2da8` |
| 5 | `0x2db0`, `0x2db8` | `0x2e00`, `0x2e08` |
| 6 | `0x2e18`, `0x2e20` | `0x2ec4`, `0x2ecc` |
| 7 | `0x2edc`, `0x2ee4` | `0x2f80`, `0x2f88` |

This is the desired bounded eight-VGPR rotating window, but it is sequential rather than integrated with the online loop.

## Ranked kill/rematerialize plan

1. **Read loop-carried `m/l` after QK.** Keep the state in its backing representation until the final QK WMMA, then materialize only the row being merged. This directly shortens `v73:v80`/`v132`-class intervals across the peak and mirrors Stage B's late-statistics schedule.
2. **Use a fresh zero QK-C seed at first use.** Emit/rematerialize the zero seed immediately before the first QK WMMA, or use a zero-C WMMA form when available. Do not let the allocator tie QK C initialization to online-`l` state born before fragment loading.
3. **Kill each QK fragment bank before P repack.** Preserve the existing ordering in which softmax/P publication follows the final QK use; do not retain Q/K views for address or masking calculations that can be rematerialized.
4. **Create LDS rotation addresses at the PV boundary.** In an integrated kernel, materialize the per-block LDS read/write address only after the current PV WMMA operands are ready, and reuse one address pair rather than carrying it through QK/softmax.
5. **Rotate one accumulator block.** Retain the demonstrated `v9:v16` load/compute/store window and immediately overwrite it for the next block. Never reconstruct all accumulator blocks in VGPRs.

The first change is the highest-confidence route to the five-register target. The fourth and fifth changes prevent the future integrated implementation from creating a new overlap, but cannot explain or repair the current 197 peak because their instructions execute later.

## Falsifiable gate

Recompile the same physical HIP microgate after late online-state materialization. Accept the pressure fix only if all conditions hold:

- VGPR count is at most 192 and the highest referenced VGPR is at most `v191`.
- Scratch and spill counts remain zero.
- LDS remains 8704 bytes and the rotating load/store sequence remains present.
- The nine QK WMMAs, typed waits, and numerical checks remain unchanged.

If this gate does not reach 192, the current evidence rejects online-state lifetime as the complete explanation and the next step is instruction-level live-in/live-out accounting at the exact QK high-water PC, not renderer changes.
