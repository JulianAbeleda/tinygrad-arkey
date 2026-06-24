# TPE-7a RESULT — rebindable Tensile node keystone holds (PASS); in-model route gated on policy + core runtime

Executed TPE-7a of `prefill-tensile-tpe7-inmodel-route-scope-20260619.md`: prove the capability the whole in-model
JIT route depends on — that a single extracted Tensile kernel node can be **re-fed different buffers per launch**
(different layers; JIT replay with re-bound inputs) via the graph-protocol `fill_kernargs(bufs)`. **Verdict: PASS.**
Probe-local, zero model/runtime-default change. Probe: `extra/qk_tensile_rebindable_node.py`; artifact:
`bench/qk-tensile-extraction/rebindable_node.json`.

## Result [M]
One `NamedAMDProgram` built once, then launched against **4 distinct (A,B,C) buffer sets** (different random
weights + activations) by writing the captured 128-byte Tensile kernarg template into a fresh kernarg buffer and
substituting the current buffers' VAs at the fixed offsets (16=D, 24=C, 32=A, 40=B):

| binding | rel_err | correct |
|---|---:|---|
| 0 | 3.9e-4 | ✓ |
| 1 | 3.7e-4 | ✓ |
| 2 | 3.6e-4 | ✓ |
| 3 | 3.4e-4 | ✓ |
| replay of #0 | 3.9e-4 | ✓ (stable) |

Distinct rel_errs confirm each binding did genuinely different work (not a stale buffer). Gates: all bindings
correct, one node serves many buffers, replay stable → **PASS**.

## Why this is the keystone
The in-model JIT route needs a *captured* node, and a captured node must (a) serve every layer (different weight +
activation buffers) and (b) survive TinyJit replay (HCQGraph re-binds input buffers each call). TPE-7a proves exactly
that: the Tensile kernarg is **rebindable** — the captured scalars/strides/WGM stay fixed while the 4 buffer VAs are
re-substituted per call, correct every time. This is the buffer-binding contract a graph node requires, validated
without touching the model or runtime defaults.

## What remains for the full in-model route (TPE-7b/c/d) — core runtime + policy
TPE-7a is the only piece that is policy-neutral and low-risk. The rest is genuine tinygrad runtime surgery and is
**gated on the external-artifact policy decision (TPE-0), which is not an engineering call:**

1. **TPE-7b (core):** select a named-descriptor runtime for the Tensile `Ops.PROGRAM` (stock `AMDProgram` uses the
   first `.rodata` descriptor of the 305-kernel object) — touches `ops_amd.py`, flag-gated.
2. **TPE-7c (core):** emit a precompiled-`Ops.PROGRAM` UOp from the prefill Linear behind `PREFILL_TENSILE_GEMM=1`
   (the model's `custom_kernel(fxn=…)` codegens UOps and cannot wrap a precompiled HSACO), with the rebindable
   Tensile `fill_kernargs` from TPE-7a as the graph runtime; verify TinyJit captures it (no recompile storm / extra
   realizes/copies).
3. **TPE-7d (gates):** warm pp512 ≥1.25× (research)/≥1.35× (strong) vs PREFILL_V2, dNLL ≤0.01, decode unchanged,
   clean fallback. Expected ~1.74× FFN-bucket → ~1.40× full pp (~95% llama) from TPE-5/6b.

**Decision required before TPE-7b/c/d:** routing the Tensile HSACO into the model ships an external ROCm-artifact
dependency; the parent scope (TPE-0) requires explicit authority for that. The technical path is now de-risked end to
end (selection→contract→HCQ launch→perf→shape matrix→block→rebindable node, all PASS/REDIRECT-resolved); the gate is
the artifact-policy decision, then the bounded core-runtime work above.

## Verdict + recommendation
**TPE-7a PASS.** The full extraction→in-model chain is technically proven except the final core-runtime injection,
which should proceed **only after the external-artifact policy decision**. Recommended: decide artifact policy; if
accepted, build TPE-7b/c behind `PREFILL_TENSILE_GEMM=1` (no default) and run the TPE-7d gates; if declined, retain
the extracted kernels as a **codegen-transfer oracle** (the schedule to imitate in a future pure-tinygrad GEMM) and
rest the route at PREFILL_V2.

## Files
`extra/qk_tensile_rebindable_node.py`, `bench/qk-tensile-extraction/rebindable_node.json`, this doc, scope
`prefill-tensile-tpe7-inmodel-route-scope-20260619.md`. Reuses `qk_tensile_hcq_launch.py` + `kernarg_all.jsonl`. No
kernel/model/default changes; no runtime files modified.
