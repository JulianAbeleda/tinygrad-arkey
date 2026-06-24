# Scope — TPE-7 in-model Tensile route (research flag), and the keystone it depends on

TPE-6b showed the reachable prefill win (~1.74× FFN-block matmul, ~1.40× weighted full pp ≈ 95% llama) requires the
extracted Tensile kernel to run as a node **inside the model's single forward graph** (PREFILL_V2 forward is
TinyJit-captured: `model.py:946-949`). A manual `prg()` launch inside the jitted forward is NOT captured → wrong on
replay. So TPE-7 is genuine tinygrad **runtime surgery**, gated by the external-artifact policy decision.

## Why this is not a probe — the graph-protocol bridge
tinygrad's JIT/graph is UOp-based. A captured kernel is an `Ops.PROGRAM` UOp; `get_runtime` (realize.py:111) builds
the runner via `Device['AMD'].runtime(function_name, lib_bytes, …)` from the UOp's `ProgramInfo` + `src[4].arg` (lib);
`HCQGraph` (graph/hcq.py:42) calls `runtime.fill_kernargs(hcq_bufs[j], call.arg.vars, argsbuf)`. Injecting the
precompiled Tensile kernel as a captured node requires four pieces:

1. **Named-descriptor `AMDProgram`** — stock `AMDProgram` loads the FIRST `.rodata` descriptor of a multi-kernel
   object; the Tensile object has 305. Need named-symbol resolution (the `NamedAMDProgram` prototype already does
   this for `<symbol>.kd`). Core change to `ops_amd.py` or a runtime subclass selected for this kernel.
2. **Tensile-layout, rebindable `fill_kernargs(bufs, vars, argsbuf)`** — must write the 128-byte Tensile kernarg with
   the CURRENT call's buffer VAs at the fixed offsets (16=D, 24=C, 32=A, 40=B), NOT tinygrad's pointers-first
   `CLikeArgsState`. Must work with buffers that change per call (HCQGraph re-binds input buffers on replay).
3. **Precompiled-`Ops.PROGRAM` injection from the model** — the model's `custom_kernel(fxn=…)` CODEGENS UOps; it
   cannot wrap a precompiled HSACO. Need a path that emits an `Ops.PROGRAM` UOp carrying the Tensile lib + symbol +
   the custom runtime, from the prefill Linear, producing the output Tensor so it joins the realize/JIT graph.
4. **Capture + measurement** — route eligible prefill matmuls (ffn_gate/up, ffn_down, attn_q/o) behind
   `PREFILL_TENSILE_GEMM=1` inside the PREFILL_V2 forward; verify TinyJit captures the nodes (no recompile storm, no
   extra realizes/copies), measure warm pp512/pp1024, run dNLL ≤ 0.01, confirm decode untouched + clean fallback.

Layout (from TPE-6): run the block in `[feature,T]` space so weights stay natural `[out,in]` (B, no transpose) and
the FFN intermediate feeds `ffn_down` directly — only block entry/exit transposes, which are tinygrad ops already in
the graph.

## Phases
- **TPE-7a — keystone (this turn, probe-local, zero default/runtime-default risk):** prove a single
  `NamedAMDProgram` is **rebindable** — re-fed different (A,B,C) buffers per launch via a graph-style
  `fill_kernargs(bufs)` that substitutes their VAs into the Tensile template, correct each time. This is the
  capability every captured-node replay needs (each layer = different weights/activations; JIT replay = new input
  buffers). If rebinding fails, the whole in-model route is blocked here.
- **TPE-7b — named-descriptor runtime** (core): promote `NamedAMDProgram` to a runtime selectable for the Tensile
  `Ops.PROGRAM`, behind a flag; no default. Risk: touches `ops_amd.py`.
- **TPE-7c — model injection** (core): emit the precompiled `Ops.PROGRAM` from the prefill Linear behind
  `PREFILL_TENSILE_GEMM=1`; verify JIT capture.
- **TPE-7d — gates:** warm pp512 ≥ 1.25× (research) / ≥1.35× (strong) vs PREFILL_V2, dNLL ≤ 0.01, decode unchanged,
  fallback clean.

## Gates / kill (from the parent scope, TPE-7)
- research pass ≥1.25× warm pp512; strong ≥1.35× pp512+pp1024; no default without policy review.
- KILL: full pp gain <1.15×; quality/fallback fails; JIT can't capture without material overhead; artifact coupling
  too brittle.

## Hard prerequisites (NOT my call)
- **External-artifact policy (TPE-0):** routing the Tensile HSACO into the model ships an external ROCm-artifact
  dependency. The parent scope requires explicit authority before a model route. TPE-7b/c/d should not land a model
  route until that decision is made. TPE-7a (keystone) is policy-neutral (probe-only).

## Constraints
No model default; decode untouched; research flag only; reuse committed captures + `NamedAMDProgram`; keep TPE-7a
probe-local. TPE-7b/c are core-runtime changes — small, reviewed, flag-gated, fallback to PREFILL_V2 on any
unsupported shape/device.

## Deliverables (this turn = TPE-7a)
`extra/qk_tensile_rebindable_node.py` (rebindable-node keystone proof), `bench/qk-tensile-extraction/rebindable_node.json`,
result in `prefill-tensile-tpe7a-rebindable-node-result-20260619.md`. TPE-7b/c/d are scoped here and pending the
external-artifact policy decision.
