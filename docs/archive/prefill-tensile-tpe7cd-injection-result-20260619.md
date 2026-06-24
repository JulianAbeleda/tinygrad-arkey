# TPE-7c RESULT — precompiled Tensile kernel INJECTED into tinygrad's realize path (eager PASS); JIT path = bounded dim-emission

Research-only execution of Track 1 (`prefill-tensile-tpe7cd-injection-and-codegen-oracle-scope-20260619.md`): inject
the precompiled Tensile kernel as a tinygrad graph node driven by `TensileRunner`. **Result: the injection works — the
precompiled Tensile kernel runs through tinygrad's realize path and produces the correct GEMM (rel_err 3.7e-4), with
NO UOp surgery.** Eager realize is a full PASS; the in-model JIT path needs one bounded extra step (emit the Tensile
launch dims in the PROGRAM UOp). Nothing shipped/banked/defaulted; external artifact research-only. Probe:
`extra/qk_tensile_inject.py`; artifact: `bench/qk-tensile-extraction/inject.json`.

## The injection (no UOp surgery) [M]
1. Build a trivial `custom_kernel(C, A, B, fxn=…)` referencing all three buffers → tinygrad codegens a valid
   `Ops.PROGRAM` + populates `runtime_cache`; capture its key via a `get_runtime` hook.
2. **Overwrite `runtime_cache[(key, 'AMD')] = TensileRunner`** — now realize uses the precompiled Tensile kernel.
3. `TensileRunner.__call__` **forces the Tensile launch grid** ((4,96,1)/(128,1,1)) and `fill_kernargs((C,A,B))`
   writes the Tensile kernarg with those buffers' VAs (the eager path `exec_kernel` calls `rt(*bufs, global_size,
   local_size, …)`, so the runner controls dims).
4. A fresh same-shaped `custom_kernel` hits the same key → `TensileRunner` runs on the new buffers.

**Verified: rel_err 3.7e-4 vs the tinygrad fp16 oracle** — the precompiled rocBLAS Tensile kernel executed inside
tinygrad's realize path, on tinygrad-owned buffers, no copies, no HIP runtime. This **refutes the earlier worry that
in-model injection was blocked** — it is feasible without core `ops_amd.py` edits or hand-built UOps.

## What remains for in-model JIT capture (bounded) [I]
The eager path (`run_linear`/`exec_kernel`) invokes `rt(*bufs, global_size, local_size)` = `HCQProgram.__call__`, so
the `TensileRunner.__call__` dim-override fixes dims there. But TinyJit/`HCQGraph` (graph/hcq.py:175) does
`q.exec(runtime, args, ast.arg.global_size, ast.arg.local_size)` — it reads dims from the **PROGRAM UOp's
`ProgramInfo`, not the runner**. The trivial kernel currently emits `global=(128,96,4)` (tinygrad's axis ordering put
the LOCAL lane on grid dim0), so under JIT the Tensile kernel would launch the wrong workgroup grid. The fix is
bounded and identified: make the trivial kernel **emit `ProgramInfo.global_size=(512,96,1)`/`local_size=(128,1,1)`**
(workgroups (4,96,1)) by structuring its ranges so the GLOBAL M-tile range and the LOCAL 128-thread range merge onto
one grid dim (a real-GEMM-style tiling reshape), or by replacing the captured PROGRAM UOp's `ProgramInfo` dims. Then
HCQGraph would `q.exec` with the right grid and the injected node is JIT-capturable — enabling TPE-7d (warm pp512 +
dNLL) at the projected ~1.40× pp.

## Verdict + status
**TPE-7c = injection PROVEN (eager PASS, rel 3.7e-4); JIT capture = one bounded dim-emission step away.** Combined
with the runtime half (TPE-7b `TensileRunner`, conformant) and the rebindable keystone (TPE-7a), the in-model route is
now demonstrated end-to-end except the JIT launch-dim plumbing. The full chain TPE-1→7c is feasible; ~1.40× pp (~95%
llama) is reachable. TPE-7d (the measured warm pp512) awaits the dim-emission fix + the external-artifact policy for
any landed route.

## Files
`extra/qk_tensile_inject.py`, `extra/qk_tensile_runtime.py` (TensileRunner + `__call__` dim-override),
`bench/qk-tensile-extraction/inject.json`, this doc. Provenance: `prefill-tensile-tpe7a-rebindable-node-result-20260619.md`,
`prefill-tensile-tpe7bcd-result-20260619.md`. No kernel/model/default changes; no runtime files modified.
