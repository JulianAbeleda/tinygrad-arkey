# Arm-A codegen modulo scheduler — result (2026-06-28)

**Commit:** `268777937` (codegen pass) + the docs commit that follows.

Executed the Codex prompt (`docs/codex-brief-decode-attention-modulo-scheduler-20260628.md` → prompt): implement or
honestly refute the Arm-A (UOp-level, in `linearizer.py`) codegen scheduler for the decode block tile. **Built the
pass, measured it, and reached an honest terminal.**

## Verdict
`SEARCH_BLOCKED_BY_CODEGEN__SCHEDULING` — Arm A is **wirable** (UOp order does reach the block-tile schedule) but
**cannot push the schedule out of LLVM's envelope toward owned**. The path to owned-quality scheduling is **Arm B**
(tinygrad emits scheduled ISA directly, bypassing LLVM's MachineScheduler).

## Why not `SCHEDULER_NOT_WIRABLE` (the preflight nuance)
Render path: the AMD device uses `[HIPRenderer, AMDLLVMRenderer, HIPCCRenderer]` (`tinygrad/runtime/ops_amd.py:1026`)
— tinygrad emits HIP-C / LLVM-IR, and **LLVM's MachineScheduler produces the final ISA schedule from the dependency
DAG**. So I tested whether UOp order reaches the emitted schedule, with a default-off `SCHED_MODULO_PROBE` (a valid,
maximally-different within-block reverse-toposort) and an instrumented `SCHED_LIST`:

| kernel | probe reorder | waitcnt schedule under reorder |
|---|---|---|
| **matmul** (control) | 352/366 | **byte-identical** (9→9, every lgkmcnt/vmcnt unchanged) — LLVM fully fixes simple schedules |
| **block tile** | 462/968 | **changes** (51→42, lgkmcnt distribution shifts) — the lgkmcnt-heavy schedule is order-sensitive |

So order is NOT fully erased for the block tile → Arm A is wirable. (The matmul shows it CAN be erased; the block
tile shows it partially survives.)

## The measurement — every UOp reorder stays in LLVM's envelope
Built `SCHED_MODULO` (`tinygrad/codegen/late/linearizer.py`): a default-off, cache-keyed, within-basic-block
critical-path latency list-scheduler (latency model: LOAD=20/40, else 1; respects `_STRUCTURAL` scoping → valid C).
Block-tile total `s_waitcnt` (owned target = **21**):

| config | reorder | total waitcnt | lgkmcnt(0) full-drains |
|---|---|---|---|
| natural (no sched flags) | — | 51 | 5 |
| `SCHED_LIST` (Layer 1) | 16/968 | 51 | 5 (inert here) |
| **`SCHED_MODULO`** (critical-path) | **297/968** | **52** | **5** (no improvement) |
| aggressive reverse probe | 462/968 | 42 | 8 (fewer waits, MORE full stalls) |
| **owned** (hand-asm, bypasses LLVM) | — | **21** | — |

Every UOp-level reorder lands in **42–52** total waitcnt; the only "fewer waits" (42) comes with *more* full drains.
**None approaches owned's 21.** LLVM re-optimizes the schedule from the dep DAG within a tight quality envelope,
regardless of the order we feed it. Owned's 21 comes from hand-scheduled assembly that bypasses LLVM.

Correctness (gate 1): `SCHED_MODULO` is a valid topo permutation by construction and produces valid C (the tile
compiles); microgate `BLOCK_TILE_MICROGATE_PASS` (see commands). W==D / route-bound (gates 3) NOT run: the hotloop
schedule metric (waitcnt) did not move toward owned (gate-2 classification), so a W==D run would only re-measure the
existing 35.0/6.7 — not spent.

## Commands
```
# preflight (render path + does UOp order reach the schedule)
DEV=AMD JIT=1 CACHELEVEL=0 <stack> SCHED_MODULO_PROBE=1  python3 <in-process compile of matmul / block tile>
# the Arm-A pass
DEV=AMD JIT=1 CACHELEVEL=0 <stack> SCHED_MODULO=1 SCHED_LIST_REPORT=1  python3 <block-tile compile> -> waitcnt 52
DEV=AMD JIT=1 DECODE_ATTN_BLOCK_TILE=1 DECODE_FAST_EXP2=1 SCHED_MODULO=1  python3 extra/qk_decode_attention_block_tile_microgate.py
```
`<stack>` = `DECODE_STAGE_COALESCE=4 COALESCED_LOAD_LOWERING=1 SCHED_UNROLL=8 DECODE_FAST_EXP2=1`.

## Code (default-off, cache-keyed, shipped defaults unchanged)
- `tinygrad/codegen/late/linearizer.py`: `SCHED_MODULO` (critical-path within-block latency scheduler) +
  `SCHED_MODULO_PROBE` (preflight reorder probe). Both default-off, hooked next to the `SCHED_LIST` model.
- `extra/qk_codegen_list_scheduler.py`: `SCHED_LIST_REPORT` reorder-count instrumentation (+ `getenv` import).

## Next: Arm B (scoped)
Owned-quality scheduling requires bypassing LLVM's MachineScheduler — tinygrad emitting **scheduled AMD assembly
itself** on the `Ops.INS → assemble_linear` path (`tinygrad/renderer/amd/elf.py`). The dormant
`extra/qk_asm_scheduler.py` already builds a register def/use DAG over `list[Inst]` and can reorder fence-delimited
regions — mature it into a latency/modulo scheduler on that path. This is the only lever that can reach owned's 21,
because it owns the instruction schedule instead of handing source to LLVM. It is the larger, comgr-independent
build the perf-state has also named for prefill GEMM — one foundation, two kernels.

## Generality (gate 4)
Not reached — the Arm-A pass did not progress on decode, so the prefill-GEMM generality proof is moot. Recorded as
decode-pattern measurement only; Arm B is where generality (and BubbleBeam binding) would be proven.
