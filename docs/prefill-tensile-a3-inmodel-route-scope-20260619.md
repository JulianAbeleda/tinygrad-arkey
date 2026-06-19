# Scope - A3/A4 in-model Tensile prefill route + pp512 measurement

Continues `prefill-tensile-research-measurement-scope-20260619.md` after A0/A1 PASS (the injected Tensile node is
JIT/HCQGraph-capturable with correct dims). A2 stalled only on a *standalone-harness* `get_runtime` key-capture quirk;
this scope does A2/A3 **in-model**, where the model's own realize/JIT compiles the kernels natively and the runtime
route is installed once and globally — sidestepping that quirk. Goal: a measured warm pp512/pp1024 + dNLL behind
`PREFILL_TENSILE_GEMM=1`, clean fallback to PREFILL_V2. Research-only; no default, no ship, policy stays open.

## Injection point (confirmed)
`_pf16(lin, x)` (model.py:38-46) is the PREFILL_V2 fp16 matmul: `out[T,out] = x[T,in].cast(f16).linear(w.transpose())`
with `w = lin._pf16_w` realized `[out,in]` fp16. This is the single per-linear site to route. Eligible roles (TPE-5,
ubatch T=512): ffn_gate/up (in=4096,out=12288), ffn_down (in=12288,out=4096). attn_q/o optional later.

## Layout (TPE-5/6)
The extracted kernel computes col-major `C[T,out]=A[T,in]·B[in,out]` ⇒ row-major buffers: A=`x^T` `[in,T]`, B=`w`
`[out,in]` (natural — no weight transpose/copy), C=`out^T` `[out,T]`. So a per-linear route needs **x→[in,T] (entry)
and out^T→[T,out] (exit) transposes** — two tinygrad ops per routed linear, inside the JIT graph. (The transpose-free
alternative is a `[feature,T]` whole-FFN-block restructure; defer unless the transposes erase the win.)

## Phases

### B0 - flag + path plumbing
- add `PREFILL_TENSILE_GEMM = getenv("PREFILL_TENSILE_GEMM", 0)`; require `PREFILL_V2` on (the route lives in `_pf16`).
- confirm `_pf16` is on the prefill path for the eligible linears (it is: ffn_gate/up/down via `_pf16_w`).
- gate: flag off ⇒ byte-identical to current PREFILL_V2 (no import/behavior change).

### B1 - in-model injection mechanism (robust, global, install-once)
Avoid A2's per-warmup hook dance. When `PREFILL_TENSILE_GEMM=1`, install ONCE at model setup:
- a **global `get_runtime` router**: for a PROGRAM ast whose output-buffer shape matches an eligible role's trivial
  kernel (identify by the kernel's output size / a sentinel `KernelInfo.name`), return that role's `TensileRunner`;
  else the real runtime. (Routes by shape on every realize ⇒ no key-capture step ⇒ not the A2 quirk.)
- the **`AMDComputeQueue.exec` dim-override** for `TensileRunner` (A1), and the **rebindable `fill_kernargs`** (A1).
- build per-role `TensileRunner`s once from `bench/qk-tensile-extraction/kernarg_all.jsonl`.
- gate: a standalone in-model-style smoke (route one `_pf16` call) is correct under realize+JIT before wiring the model.

### B2 - wire `_pf16`
- when flag on and `(out,in,T)` ∈ eligible set: build `C[out,T] = custom_kernel(C_zeros, x.transpose()→[in,T], w
  (natural [out,in]), fxn=generic_trivial)`, then return `C.transpose()→[T,out]`. The router makes it Tensile.
- unsupported shape/device/role ⇒ fall back to the normal `_pf16` `.linear` (silent, with a one-time diag).
- keep weights as the existing `_pf16_w` `[out,in]` (no extra realize/copy).

### B3 - correctness + quality
- one prefill forward (T=512): output finite; per-routed-linear rel_err ≤ 2e-2 vs the `.linear` result;
- **dNLL ≤ 0.01** vs PREFILL_V2 (flag off) on a fixed prompt set.

### B4 - measurement (TPE-7d)
- warm pp512 and pp1024 (clean model.generate / CLI --benchmark prefill path, per the bench/README harness rule);
- baseline = PREFILL_V2 (flag off); flag-on = routed; report speedup;
- fallback A/B: flag-off == current PREFILL_V2; unsupported-shape path == PREFILL_V2;
- decode smoke: ctx512 W==D unchanged (route is prefill-only);
- graph attribution if timing doesn't transfer (no per-op host sync, one graph).
- artifacts: `bench/qk-tensile-extraction/inmodel_measurement.json`; doc updates the existing
  `prefill-tensile-inmodel-measurement-result-20260619.md`.

## Gates / verdicts (from the parent scope)
| gate | threshold |
|---|---|
| correctness | rel_err ≤ 2e-2 per routed linear; model outputs finite |
| quality | dNLL ≤ 0.01 |
| research speed | warm pp512 ≥ 1.25× PREFILL_V2 |
| strong speed | pp512 and pp1024 ≥ 1.35× PREFILL_V2 |
| fallback | flag-off / unsupported == current PREFILL_V2 |
| decode | unchanged (prefill-only route) |
| graph | no per-op host-sync wall (TPE-6 naive); one HCQGraph |

- **PASS_RESEARCH**: speed/quality/fallback pass; external artifact stays research-only.
- **PASS_STRONG_POLICY_GATED**: strong speed; only the TPE-0 artifact policy blocks landing.
- **REDIRECT_GRAPH_BOUNDARY**: kernels fast but graph/transpose overhead eats the win in-model.
- **KILL**: dims/capture/correctness/quality/fallback fail in-model.

## Stop conditions
- per-call compile/re-key in the measured path;
- transposes erase the Amdahl value (then either restructure to `[feature,T]` block, or REDIRECT);
- dNLL fails; fallback/off changes behavior; decode touched;
- the global `get_runtime` router proves unreliable in-model (then reassess the install mechanism).

## Non-goals
No default; no decode route; no new extraction/PMU/q8/pure-codegen; no HIP runtime in-process; no per-shape recompile.
Only the eligible TPE-5 roles. External artifact remains research-only and explicit.

## Deliverables
A flag-gated `_pf16` route + install-once router (probe-local module imported only when the flag is set; model edit
minimal and guarded), `bench/qk-tensile-extraction/inmodel_measurement.json`, and the result doc with routed roles,
pp512/pp1024 baseline-vs-flag, dNLL, fallback, decode check, artifact-dependency statement, and verdict.
