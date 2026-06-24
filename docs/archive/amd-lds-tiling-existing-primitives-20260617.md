# LDS tiling — existing-primitives inventory (Phase 1, 2026-06-17)

**Verdict: the missing primitive IS expressible directly in `Tensor.custom_kernel`, WITHOUT BEAM.** LDS
(`AddrSpace.LOCAL`), a workgroup barrier (`Ops.BARRIER`), workgroup/thread dims (`UOp.special`), and
register accumulators (`AddrSpace.REG`) are all available as UOp-builder ops and render correctly on the HIP
(gfx1100) backend. There is even a working UOp **LDS flash-attention** example in-repo. → **GREENLIGHT Phase 2.**

## The primitives (file:line)

| primitive | how | where |
|---|---|---|
| address spaces | `AddrSpace.GLOBAL / LOCAL / REG` (LOCAL = LDS) | `tinygrad/dtype.py:52` |
| LDS buffer | `UOp.placeholder(shape, dtype, slot, addrspace=AddrSpace.LOCAL)` → `Ops.DEFINE_LOCAL` | `tinygrad/uop/ops.py:1053` |
| register buffer | `…addrspace=AddrSpace.REG` → `Ops.DEFINE_REG` | same |
| barrier | `UOp.barrier(*stores)` → `Ops.BARRIER`; order reads via `buf.after(barrier)` | `tinygrad/uop/ops.py:531` |
| grid / thread dims | `UOp.special(n, "gidx0")` (block), `UOp.special(n, "lidx0")` (thread) | `extra/gemm/amd_uop_matmul.py:41,58` |
| HIP render: LDS | `__attribute__((shared, aligned(16)))` prefix | `tinygrad/renderer/cstyle.py:368,190` |
| HIP render: barrier | `__builtin_amdgcn_fence(RELEASE,"workgroup"); __builtin_amdgcn_s_barrier(); …fence(ACQUIRE…)` | `tinygrad/renderer/cstyle.py:370` |

LDS can ALSO be introduced by the optimizer (`OptOps.LOCAL/GROUP`, `tinygrad/codegen/opt/`), but that path
needs the heuristic/BEAM search — **not required here**; we author LOCAL directly in the kernel AST.

## Templates (closest → furthest)

- **`extra/gemm/amd_uop_matmul.py`** — the canonical pattern: `UOp.special` grid/thread, cooperative
  `copy(local[…tid], global[…tid])` GLOBAL→LDS load, `barrier = UOp.barrier(stores)`, `local.after(barrier)`,
  then LDS→REG tiles + FMA. Times via `GlobalCounters.time_sum_s` under `Context(DEBUG≥2)` (the authoritative
  GPU-time source — exactly our measurement rule).
- **`extra/gemm/amd_flash_attention.py`** — a UOp **flash attention** with LDS Q/K/V tiles + barrier
  (`QP_lds`/`KV_lds` placeholders, `qk_load_barrier`). **Significant:** UOp LDS flash exists in-repo without
  BEAM — so the Phase-5 "needs BEAM/LDS we can't express" framing was too pessimistic; the locality primitive
  is reachable. (Whether it actually wins on prefill shapes is what Phases 3+ must measure — don't assume.)
- `test/amd/test_custom_kernel.py:68 custom_lds_sync` — smallest test using LDS (`DEFINE_LOCAL` + `s_barrier`).
- `extra/gemm/amd_matmul.py` — raw HIP `__shared__` (loads .cpp/.s), NOT UOp; the path we explicitly avoid.

## Minimal recipe (for Phase 2)

```
gid = UOp.special(NB, "gidx0"); tid = UOp.special(BLOCK, "lidx0")
lds = UOp.placeholder((BLOCK,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
store = lds[tid].store(x.reshape(NB,BLOCK)[gid][tid])     # cooperative GLOBAL->LDS
lds   = lds.after(UOp.barrier(store))                      # barrier, then reuse
out   = y.reshape(NB,BLOCK)[gid][tid].store(lds[(tid+1)%BLOCK]*2+1)  # cross-lane LDS read
return out.sink(arg=KernelInfo(name="...", opts_to_apply=()))
```

## Measurement (locked to this arc)

Use `GlobalCounters.time_sum_s` inside `Context(DEBUG>=2)` (kernels run with wait) — the same authoritative GPU
time the matmul example uses. Never wall-clock around `.realize()` (the Phase-5 trap). Report compile vs exec
separately, kernel/program count, bytes R/W.

## What's still unproven (Phases 2–3 will test)

Expressibility ≠ a win. Phase 2 proves the LDS round-trip compiles/runs/replays; **Phase 3 proves LDS reuse
actually beats redundant global reads on GPU time** (the real primitive proof). Only then does reopening
flash-prefill make sense.
