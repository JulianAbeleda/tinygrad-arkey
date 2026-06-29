# AMD ISA backend — exhaustive roadmap to end-to-end (2026-06-29)

Where we are: `AMDISARenderer` (opt-in `DEV=AMD:ISA`) is built on the verified assemble foundation; scalar-elementwise
isel is complete; it reaches register allocation and is blocked there (`AMD_ISA_INC0_BLOCKED_REGISTER_OR_ABI`). This
lists every increment from here to "run the full decode e2e via the native backend," each with goal + work + gate.
The end state is pure machine search for AMD: the machine owns isel/regalloc/**scheduling**, generated kernels reach
hand-ASM quality, and the hand kernels retire.

## Inc 0 — finish the minimal kernel (running correct)
**0a. Register-model integration.** Fix the `regalloc.py:118` `reals[i]` KeyError: faithfully model the AMD register
file on `LinearScanRegallocContext` — fixed entry registers (`v0`=tid, `s[0:1]`=kernarg), 64-bit pointers as SGPR
*pairs*, VGPR data. Study how `reals` is keyed per program point; likely add a register-class/alignment notion (pairs)
or pre-color entry regs the way the allocator expects.
**0b. Vec + consecutive registers.** The real generated elementwise kernel is vectorized (`STACK`/`CAST`/b128). Add
vec isel (GEP/STACK, vec dtypes → `global_load_b128`/`global_store_b128`) + **consecutive/aligned multi-register
allocation** (b128 = 4 aligned VGPRs) the single-register allocator doesn't provide.
**Gate:** `DEV=AMD:ISA` trivial elementwise (`a+b`) compiles + runs numerically correct on gfx1100.

## Inc 1 — general op coverage (elementwise + reductions)
Port the LLVM isel map (Phase-0 method, per-op) for: casts (f16/f32/int), `sub/div/max/min/exp2/log2/recip`,
`cmplt`/`where` (compare+select via exec mask), the `RANGE`/`END` reduce **loop** (control flow → labels/branches),
gated load/store (exec-mask predication or scratch trick), CONST/strided index forms. Correct-but-conservative
`s_waitcnt` (drain after each memory op).
**Gate:** a GEMV and a sum-reduction kernel run correct vs numpy through `DEV=AMD:ISA`.

## Inc 2 — the decode-attention tile's special primitives
isel for the ops the block tile uses: `v_dot2acc_f32_f16` (fdot2 packed dot), `ds_bpermute_b32` (cross-lane reduce),
LDS staging (`DEFINE_LOCAL` → `ds_load`/`ds_store` + group-segment size in the descriptor), `s_barrier`, the
online-softmax ALU (`v_exp`, `v_max`, `v_fma`). These are CUSTOMI today — map each to its rdna3 inst.
**Gate:** the generated block tile compiles via `AMDISARenderer` + `qk_decode_attention_block_tile_microgate.py`
→ `BLOCK_TILE_MICROGATE_PASS` (token-correct).

## Inc 3 — the scheduler (THE lever — this is the whole point)
Mature `extra/qk_asm_scheduler.py` (reg def/use DAG over the `Inst` stream, foundation-verified) + `renderer/amd/
schedule.py` (latency metadata) into a real latency/modulo scheduler on the `Inst` stream, run inside the renderer
before assemble. Three pieces: (a) **consumer-only `s_waitcnt`** insertion (track vmcnt/lgkmcnt per outstanding memory
op; insert minimal, not drain-all); (b) latency list-scheduling to fill load/reduce shadows; (c) **cross-iteration
software pipelining** of the online-softmax recurrence — the thing LLVM's MachineScheduler cannot do (Arm-A finding).
**Gate:** block-tile `s_waitcnt` drops toward owned's 21 (vs LLVM's 42–52); `qk_decode_hotloop_schedule_diff.py`
exposed-latency drops toward owned.

## Inc 4 — register-allocation / occupancy quality
Make regalloc match LLVM/owned quality: occupancy-aware allocation (minimize VGPR toward owned's 64 → high wave
occupancy), correct spill/fill under pressure, cross-class copies. Use the Phase-0 LLVM regalloc model as the bar.
**Gate:** block-tile vgpr/occupancy ≈ owned; no spills on the hot path.

## Inc 5 — route binding + W==D promotion
Bind the `AMDISARenderer`-compiled block tile into the decode route; run route-bound W==D (`qk_decode_route_
attribution_wd.py`): route_bound + token_match + tok/s.
**Gate:** W==D rises materially from 35.0/6.7 toward owned (103.8/94.6); promotable threshold approached. (If the
scheduler can't beat the LLVM envelope even here → that's the honest abstraction-limit terminal, recorded.)

## Inc 6 — BubbleBeam / search binding (pure machine search)
Lift the searchable decisions — schedule (pipeline depth, list priority), regalloc (occupancy vs spill), waitcnt
placement — into BubbleBeam/FutureSight as a candidate space; evaluator = route-bound W==D + token-match. Generality
proof: the same backend + scheduler moves the prefill GEMM hot loop.
**Gate:** the machine *selects* a competitive schedule; Phase-3 purity gate (generated ≥ promotion threshold of owned).

## Inc 7 — pure-default retirement + RUN IT (the final step)
Reclassify the owned hand-ASM tile to fallback; make the native-backend route the default for decode attention.
**FINAL STEP — run the full decode e2e:** run the full model decode (the synced W==D harness / `model.generate`) with
`AMDISARenderer` producing the scheduled attention tile, measure whole-model tok/s vs owned at ctx512/4096, confirm
token-match, and record the e2e number. This is "run it": pure machine search producing a competitive AMD decode
kernel end-to-end, no LLVM in the final mile, no hand-ASM required.

## Issue brief for Codex — Inc 0 (the immediate next prompt to format)

**Goal of the next prompt:** unblock Inc 0 so `DEV=AMD:ISA python3 -c "from tinygrad import Tensor;
(Tensor.empty(64)+Tensor.empty(64)).numpy()"` runs numerically correct on gfx1100. Two sub-parts, gate each.

**The exact blocker (0a — register/ABI integration).** `tinygrad/renderer/isa/amd.py` (the `AMDISARenderer`) emits
real rdna3 `Inst`s and reaches the framework allocator, which throws at `tinygrad/codegen/late/regalloc.py:118`:
`ndefs = tuple(ctx.reals[i][v] for v in x.tag)` → **`KeyError: 4`**. `reals` (built in `LinearScanRegallocContext.
__init__`, `regalloc.py:12-108`) maps `program_point -> {virtual_reg: real_reg}`; program point 4 has a def
(`x.tag` is a tuple) but no `reals` entry, i.e. some def's virtual register was never assigned a real one. Already
ruled out: (a) `.rtag()` on immediate operands; (b) x86-style `alloc_vregs` seeding the fixed `v0` (`TID`) as a
constrained vreg. **What Codex's prompt must direct:** study how `reals` and `live_range` are keyed
(`regalloc.py:19-30` builds live ranges from `u.tag` defs + `s.reg` uses; `:32-108` allocates and fills `reals`),
then make the AMD register model faithful to it — specifically the three things x86 doesn't have: (1) **fixed entry
registers** (`v0`=workitem id, `s[0:1]`=kernarg ptr) that must be pre-colored/live-in without a def line; (2)
**64-bit pointers = SGPR pairs** (currently faked with an even-aligned single-register pool — the allocator's
interference model doesn't know `s4` clobbers `s5`); (3) confirm every emitted def's virtual reg actually flows into
`live_range` so `reals[point]` exists. Reference: how `x86.py` (`:828-862`, `:332-357`) pins ABI/fixed regs and how
`regalloc.py` consumes `.reg`/`.tag`. **Gate 0a:** the kernel reaches `post_regalloc` + `assemble_linear` without a
regalloc error (even if numerics not yet checked).

**The downstream blocker (0b — vec + consecutive registers).** The real generated `a+b` kernel is vectorized:
`STORE(CAST(INDEX(PARAM,CONST)), STACK(add0..add3))` — a vec4/b128. The current isel only covers scalar. Codex's
prompt must add: vec-dtype isel (`STACK`/`GEP` → `global_load_b128`/`global_store_b128` + per-lane `v_add_f32`) and
**consecutive/aligned multi-VGPR allocation** (b128 needs 4 aligned VGPRs) — the framework allocator is
single-register, so this needs a register-class/alignment extension OR a scalarization pass that splits the b128 into
4 scalar b32 ops (simpler first cut — emit 4 `global_load_b32` + 4 `v_add` + 4 `global_store_b32`, sidestepping
consecutive-reg alloc for Inc 0). **Gate 0b (= Inc 0 done):** trivial elementwise runs correct on gfx1100, matches
numpy; foundation test still `INC0 ALL_PASS`; default `HIPRenderer` byte-identical.

**Constraints for the prompt:** default-off (`DEV=AMD:ISA`); no edits to `tinygrad/runtime/autogen/**`; correctness
first; do not start Inc 1 until Inc 0 runs correct; bracketed-prefix commits. **Verdicts to emit:**
`AMD_ISA_INC0_PASS_TRIVIAL_KERNEL_RUNS` / `..._BLOCKED_REGISTER_OR_ABI` / `..._BLOCKED_ISEL_MODEL_INCOMPLETE` /
`..._BLOCKED_ASSEMBLE_OR_ELF` / `..._BLOCKED_RUNTIME_ROUTE`. Files: `tinygrad/renderer/isa/amd.py` (the renderer),
`tinygrad/renderer/isa/x86.py` (template), `tinygrad/codegen/late/regalloc.py` (the allocator), `tinygrad/renderer/
isa/__init__.py` (framework), `bench/amd-isa-backend-inc0/latest.json` + `docs/amd-isa-backend-inc0-result-20260628.md`
(current state).

## Risk / where it could stall (honest)
- Inc 0b/4 (register pairs, occupancy) and Inc 2 (special-op isel) are mechanical-but-substantial.
- **Inc 3 (scheduler) is the make-or-break**: it's the lever that must beat LLVM's 42–52 envelope toward 21. If a
  proper modulo scheduler on the `Inst` stream still can't approach owned, that is the true abstraction limit — but
  now provable with the machine owning the schedule (unlike Arm A/B, where LLVM owned it or no backend existed).
- One backend serves decode attention **and** prefill GEMM — so the investment amortizes across both hand kernels.
