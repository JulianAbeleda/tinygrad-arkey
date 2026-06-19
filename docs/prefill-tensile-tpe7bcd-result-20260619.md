# TPE-7b/c/d RESULT — TensileRunner conforms to the graph protocol (PASS); in-model capture is a deep UOp-injection build

Research-only execution of TPE-7b/c/d (`prefill-tensile-tpe7-inmodel-route-scope-20260619.md`) — building the
in-model route purely to learn, nothing shipped/banked/defaulted, external artifact research-only. **TPE-7b: PASS** —
built `TensileRunner`, a runtime object that conforms to the exact HCQGraph protocol, validated for all 3 roles.
**TPE-7c/d: the remaining seam is a deep tinygrad-internals UOp-injection build** (precisely characterized below), the
same deep-codegen/runtime-capability class that has walled this project; the runtime half is now done, the achievable
number is the validated ~1.40× pp. Probe: `extra/qk_tensile_runtime.py`; artifact:
`bench/qk-tensile-extraction/runtime.json`. No model/runtime-default change; decode untouched.

## TPE-7b — TensileRunner (PASS) [M]
`HCQGraph` (graph/hcq.py:42,92) needs a runtime exposing `.dev`, `.kernargs_alloc_size`, and
`fill_kernargs(bufs, vars, argsbuf) -> HCQArgsState`, then execs the returned args_state with launch dims from the
PROGRAM UOp. `TensileRunner(NamedAMDProgram)` provides:
- **named-descriptor** load (resolves `<symbol>.kd` from the 305-kernel object) — already in `NamedAMDProgram`;
- a **rebindable Tensile-layout `fill_kernargs(bufs)`** — writes the captured 128-byte template and substitutes the
  CURRENT call's buffer VAs at the fixed offsets (D@16, C@24, A@32, B@40), overriding tinygrad's pointers-first
  `CLikeArgsState`.

Validated by mirroring the HCQGraph call path (`fill_kernargs(bufs) → exec(args, tensile_dims)`) for **all 3 roles ×
3 distinct buffer rebindings** — every result correct. So the runtime half of the in-model node is **done and
conformant**: drop a `TensileRunner` in as the runtime for a Tensile PROGRAM UOp and HCQGraph can fill+exec it.

## TPE-7c — why in-model capture is a deep build (the precise seam) [M]
The model forward is TinyJit-captured (`model.py:946-949`); TinyJit (jit.py:269-296) at capture (`cnt==1`) collects
`_linears` (UOps appended via the `add_linear`/`capturing` hook when kernels realize) and **immediately** builds the
executable with `jit_lower` — **there is no post-capture window to swap a runtime in.** Therefore the Tensile kernel
must enter as a `CALL`+`Ops.PROGRAM` UOp *through the realize path* during capture. That forces three coordinated
internals manipulations, none of which the model's `custom_kernel(fxn=…)` supports (it CODEGENs UOps from `fxn`, so
it cannot carry a precompiled HSACO):

1. **Precompiled `Ops.PROGRAM` UOp** — construct one with the Tensile HSACO as its lib (`src[4].arg`, per
   `get_runtime` realize.py:111), short-circuiting the `do_render`/`do_compile` codegen pipeline
   (codegen/__init__.py:195-199).
2. **Tensile `ProgramInfo`** — HCQGraph takes launch dims from `ast.arg.global_size/local_size` and buffer roles
   from `.outs/.ins/.vars`; these must be set to Tensile's (`global=(4,96,1)` etc., out/in buffer indices), not a
   codegen-derived kernel's.
3. **Bind `TensileRunner`** — make `get_runtime` return the `TensileRunner` (done in 7b) for that PROGRAM UOp, so
   the custom kernarg layout + named descriptor are used.

Each is feasible individually but together they are a fragile, multi-piece UOp/runtime build (construct or
post-process the PROGRAM UOp + ProgramInfo + lib + runtime binding so it survives `jit_lower` and HCQGraph), squarely
in the deep-codegen/runtime-capability class. The runtime-swap-after-capture shortcut is ruled out (no window); the
hand-built-PROGRAM-UOp path is the only one and is a dedicated build, not a probe.

## TPE-7d — gates (not run) [I]
Warm pp512 ≥1.25×/≥1.35× + dNLL ≤0.01 require 7c first. The expected number is grounded: TPE-5 weighted **~1.40× full
pp** (~95% llama) from the per-role HCQ throughput (66.8/68.9/58.9 TFLOPS), and TPE-6b's measured/projected **~1.74×
FFN-block matmul** in a single dispatch. So the gate would clear *if* 7c lands; no new measurement is possible without
the UOp-injection build.

## Verdict + recommendation
**TPE-7b PASS; TPE-7c/d = a deep tinygrad-internals UOp-injection build (named, characterized, runtime-half done).**
The entire extraction→runtime chain is now proven end to end (TPE-1 selection → 2 contract → 3 HCQ launch → 4 perf →
5 shape matrix → 6 block → 6b graph analysis → 7a rebindable node → 7b conformant runtime), all PASS/resolved,
pointing at ~1.40× pp (~95% llama). What is left is exactly one capability: injecting a precompiled, Tensile-dim,
custom-kernarg `Ops.PROGRAM` UOp into the JIT graph. Options:
- **build it** as a dedicated research session (construct the precompiled PROGRAM UOp + ProgramInfo, bind
  `TensileRunner`, capture in a one-block harness, then in-model behind `PREFILL_TENSILE_GEMM=1`); or
- **use the extracted kernels as a codegen-transfer oracle** — the proven-faster schedule to imitate in a future
  pure-tinygrad GEMM (sidesteps both the external-artifact dependency and the injection build).

Either way the research conclusion is firm: the mature backend's prefill speed **is** reachable from tinygrad
(correct, no-copy, no-HIP, rebindable, conformant runtime); the only thing between here and ~95%-of-llama prefill is a
single deep UOp-injection capability.

## Files
`extra/qk_tensile_runtime.py`, `bench/qk-tensile-extraction/runtime.json`, this doc, scope
`prefill-tensile-tpe7-inmodel-route-scope-20260619.md`, keystone `prefill-tensile-tpe7a-rebindable-node-result-20260619.md`.
Reuses `qk_tensile_hcq_launch.py` + `kernarg_all.jsonl`. No kernel/model/default changes; no runtime files modified.
