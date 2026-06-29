# AMD ISA backend — Inc 0 result (2026-06-29)

Executed the Codex prompt (audit-first AMD ISA backend) using LLVM AMDGPU as the solved map. **Phase 0 reached
`LLVM_MODEL_READY_FOR_INC0`; Inc 0 verdict = `AMD_ISA_INC0_BLOCKED_REGISTER_OR_ABI`** — a real `AMDISARenderer` is
built, opt-in, and processes generated UOps through isel into register allocation, but the register-model
integration with the framework allocator is not yet complete (and a vec gap follows). No fake completion.

## Phase 0 — LLVM model: REACHED
`LLVM_MODEL_READY_FOR_INC0` (`bench/amd-llvm-backend-model/latest.json`). Extracted LLVM's AMDGCN model for the
trivial elementwise kernel by disassembling the HIPRenderer (comgr/LLVM) output: ABI = `s[0:1]` kernarg ptr at
entry, `v0` = workitem id; sequence `s_load` kernarg ptrs → `v_lshlrev` byte offset → `s_waitcnt lgkmcnt(0)` →
`global_load` → `s_waitcnt vmcnt(0)` → `v_add_f32` → `global_store` → `s_endpgm`. This is the map Inc 0 reproduces.

## What was built
`tinygrad/renderer/isa/amd.py` — `AMDISARenderer(ISARenderer)`, wired opt-in at `tinygrad/runtime/ops_amd.py:1026`
(appended to the renderer list; selected by **`DEV=AMD:ISA`**; default stays `HIPRenderer`). Emits real rdna3 `Inst`
objects (via `renderer/amd/dsl.py` + `rdna3/ins.py`) so the `do_assemble` path → `assemble_linear` runs.
**isel is complete for the scalar elementwise path:** PARAM→`s_load_b64` from kernarg[i*8]; SPECIAL→`v0`;
CAST(ptr)→passthrough; INDEX→`v_lshlrev_b32` byte offset; LOAD→`global_load_b32`; ADD/MUL→`v_add/mul_f32_e32`;
STORE→`global_store_b32`; SINK→`s_endpgm`; conservative `s_waitcnt` drains after memory ops; immediates `.rtag()`'d;
fixed entry regs (TID=v0) seeded as constrained vregs (x86 `alloc_vregs` analog).

## Gates
1. **Foundation (gate 1): PASS** — `PYTHONPATH=. python3 extra/qk_asm_scheduler_inc0_test.py` → `INC0 ALL_PASS`
   (assemble byte-faithful, runs correct, legal reorder runs correct).
2. **Default path unchanged: PASS** — `Device['AMD'].renderer` = `HIPRenderer`; default add numerically correct.
3. **Compiles through AMDISARenderer: PARTIAL** — isel completes; reaches `LinearScanRegallocContext`.
4. **Runs correct on gfx1100: NOT REACHED.**

## Exact blocker
`tinygrad/codegen/late/regalloc.py:118` — `ndefs = tuple(ctx.reals[i][v] for v in x.tag)` → **`KeyError: 4`**
(program point 4 carries a def with no `reals` entry). The fix-attempts that did NOT resolve it: (a) `.rtag()` the
immediate operands; (b) port x86's `alloc_vregs` to seed the fixed `v0` register as a constrained vreg. Root cause:
the AMD register model (fixed entry registers + 64-bit pointers as SGPR *pairs* + VGPR data) does not yet map
faithfully onto the framework's single-register linear-scan live-range analysis, so some program point's def is
never assigned a real register.

**Downstream gap (also required for full Inc 0):** the real generated elementwise kernel is **vectorized**
(`STORE(CAST(INDEX(PARAM,CONST)), STACK(...))` — vec4/b128), so beyond the regalloc fix, Inc 0 needs **vec isel +
consecutive-register allocation** (b128 = 4 aligned VGPRs) that the framework single-register allocator doesn't
provide. Inc 0 requires BOTH.

## Commands run
```
# Phase 0 (LLVM model): disasm HIPRenderer output for out=a+b -> bench/amd-llvm-backend-model/latest.json
DEV=AMD JIT=1 CACHELEVEL=0 python3 <hook+_disasm of (a+b)>
# build + wire AMDISARenderer; iterate isel/regalloc
DEV=AMD:ISA JIT=1 CACHELEVEL=0 python3 -c "from tinygrad import Tensor; (Tensor.empty(64)+Tensor.empty(64)).numpy()"
# gates
DEV=AMD JIT=1 python3 extra/qk_asm_scheduler_inc0_test.py        # INC0 ALL_PASS
DEV=AMD JIT=1 python3 -c "...default add..."                     # default HIPRenderer correct
```

## Artifacts
- `bench/amd-llvm-backend-model/latest.json` — Phase-0 LLVM model (`LLVM_MODEL_READY_FOR_INC0`).
- `bench/amd-isa-backend-inc0/latest.json` — Inc-0 state + exact blocker (`AMD_ISA_INC0_BLOCKED_REGISTER_OR_ABI`).
- `tinygrad/renderer/isa/amd.py` — the renderer (opt-in, default unchanged).

## Verdict
`AMD_ISA_INC0_BLOCKED_REGISTER_OR_ABI`. Honest progress: a real, opt-in AMD ISA renderer that takes generated UOps
through complete scalar-elementwise isel into register allocation, on the verified assemble foundation, with the
precise next blockers identified (register-model integration, then vec/consecutive-register support). Per the
prompt, Inc 1 is NOT started (Inc 0 does not yet run correct).
