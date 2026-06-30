# RA0 — Loop-Carried Physical Accumulator Design (AMD ISA backend)

**Verdict: `AMD_ISA_REGALLOC_ACCUM_RA0_PASS_DESIGN_READY`**

## Core insight
A loop-carried accumulator is modeled like the AMD backend's **existing fixed system registers** (KARG `s0:1`,
workgroup-id `s2/s3/s4`) that are already emitted directly in `lower_inst` and never allocated/tracked: a **reserved
physical VGPR** the virtual-register linear-scan regalloc **never sees** (no tuple-tag) and **never allocates**
(reserved out of the pool). Because the accumulator is *not a virtual register*, the single-def SSA invariant is
**not weakened** — it simply never applies to the accumulator.

## How it clears the three N5A walls
| N5A wall | resolution |
|---|---|
| 1. single-def assert `lr[v][0]==i` | accumulator write carries **no tuple-tag** → never enters `live_range` → assert never evaluated for it (untouched for real vregs) |
| 2. no fixed-reg operand class | `ACCUM_READ/UPDATE` reference `v[pin]` as a **fixed `Reg`** in `lower_inst` (exactly like `_S[2]`/KARG), not via `alloc_vregs` |
| 3. no real move / loop carry | explicit **`v_mov_b32`** for init + final read; **in-place `v_add`/`v_fma`** for the per-iter update; pinned reg implicitly live across `RANGE/END` (reserved, untracked) |

## Representation
- `ACCUM_POOL = v240..v255` reserved; `_vpool` returns only `v1..v239`.
- One pinned VGPR per accumulator **element**. A SIMD VGPR holds per-lane values, and each of the 4 warps has
  wave-private VGPRs → **one VGPR == the full per-(warp,lane) accumulator** (no 128-register explosion; satisfies the
  stop-condition).
- New AMDOps: `ACCUM_INIT` (`v_mov v[pin], imm`), `ACCUM_UPDATE` (`v_<op> v[pin], v[pin], delta`), `ACCUM_READ`
  (`v_mov vvirt, v[pin]`). Pinned index encoded in the AMDOp **arg**, not a tag.

## Defs/uses, live range, scheduler/waitcnt
- The pinned reg appears as a real `Reg` operand in lowered instructions → **scheduler/waitcnt order RAW/WAW/WAR
  correctly** (in-place RMW chain serialized). But it has no virtual tag → **regalloc ignores it** (no def tracking,
  no allocation). Delta inputs are normal tracked vregs.
- `ACCUM_INIT` ordered before the loop (existing AFTER/range placement); `ACCUM_UPDATE` in the body; `ACCUM_READ`
  after `END`. Implicit liveness (reserved) means loop live-in/out machinery never touches it.

## Fallback (default unchanged)
`AMD_ISA_REG_ACCUM=0` (default) → the existing `Ops.DEFINE_REG` LDS path is byte-identical. The pinned path activates
only with the flag set AND a qualifying accumulator (per-thread scalar, **compile-time index**, total elements ≤ pool
size); otherwise LDS fallback.

## Staging
- **RA1**: the 3 AMDOps + reserved pool + 3 microkernels (single / two-independent / nested accumulators) behind the
  flag, NOT wired to the tile.
- **RA2**: wire the `DEFINE_REG` accumulator load/store to the pinned path (compile-time index, accumulator-only, LDS
  fallback kept); prove `lds_accum_stage` DS count drops + token match.
- **RA3**: W==D ctx512/ctx4096 + before/after PC-source + N4.

## Open risks (carried to RA1)
1. scheduler must see `v[pin]` as a Reg for ordering yet regalloc must not allocate it — compatible (reserve + emit as fixed Reg), but verify in the microgate.
2. `ACCUM_READ` must sit **outside** any EXEC-masked region.
3. pool overflow (>16 elements) → LDS fallback.
