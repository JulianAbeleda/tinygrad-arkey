# Scope - Option A research measurement: in-model Tensile prefill route

Purpose: finish the bounded external-Tensile route only far enough to produce an **in-model research measurement**.
This is not a default path and not a shipping decision. The output is a measured warm pp512/pp1024 + dNLL result
behind `PREFILL_TENSILE_GEMM=1`, with clean fallback to PREFILL_V2.

## Current state

The technical arc is exhausted through TPE-7c:

| step | status |
|---|---|
| external GEMM ceiling | real: ~1.5-1.7x tinygrad |
| in-process HIP bridge | KILL: HIP runtime cannot coexist with tinygrad HCQ/KFD |
| Tensile extraction | PASS: selected kernel + launch contract recovered |
| HCQ launch | PASS: 66.9 TFLOPS, correct, no HIP runtime, no copies |
| shape matrix | PASS: ffn_gate/up 66.8, ffn_down 68.9, attn_q/o 58.9 TFLOPS; weighted ~1.397x pp512 |
| runtime protocol | PASS: `TensileRunner` conforms to HCQGraph-style `fill_kernargs` |
| rebindability | PASS: one node can bind different current buffers |
| eager tinygrad injection | PASS: `runtime_cache` swap runs precompiled Tensile through realize, rel_err ~3.7e-4 |
| pure tinygrad codegen transfer | project-level: software-pipelined K-loop is not UOp-expressible |

The single remaining engineering gap for Option A is **JIT launch dims**:

- eager `HCQProgram.__call__` lets `TensileRunner.__call__` override launch dims;
- HCQGraph calls `q.exec(runtime, args, ast.arg.global_size, ast.arg.local_size)`;
- therefore the captured `Ops.PROGRAM` must carry Tensile launch dims in its `ProgramInfo`.

For ffn_gate/up, desired hardware launch:

- workgroups/global grid: `(4, 96, 1)`
- local/workgroup: `(128, 1, 1)`
- equivalent ProgramInfo dims: global elements `(512, 96, 1)`, local `(128, 1, 1)`, depending on tinygrad's
  `ProgramInfo.launch_dims` convention already observed in TPE-7c.

## Decision boundary

This scope assumes **external artifacts are acceptable for a research measurement only**:

- allowed: run a precompiled rocBLAS/Tensile HSACO through tinygrad HCQ behind `PREFILL_TENSILE_GEMM=1`;
- not allowed: default route, shipped route, artifact policy change, decode route change;
- after measurement, policy remains open: accept external artifact, reject it and keep oracle only, or rest.

## Goals

1. Make injected Tensile `Ops.PROGRAM` nodes JIT-capturable by emitting or patching correct launch dims.
2. Prove correctness under TinyJit/HCQGraph for at least one role.
3. Route the PREFILL_V2 high-share linear roles behind `PREFILL_TENSILE_GEMM=1` in a research path.
4. Measure warm pp512 and pp1024, dNLL, fallback, and decode untouched.
5. Produce a clear PASS/REDIRECT/KILL result.

## Non-goals

- No model default.
- No broad model refactor.
- No pure-tinygrad renderer work.
- No HIP runtime in-process bridge.
- No new Tensile selection/extraction work unless an unsupported shape is encountered.
- No q8 decode work.

## Required source assets

Use existing committed/proven assets:

- `extra/qk_tensile_runtime.py` (`TensileRunner`, graph-protocol kernarg fill);
- `extra/qk_tensile_inject.py` (eager injection proof and runtime_cache swap pattern);
- `bench/qk-tensile-extraction/kernarg_all.jsonl`;
- `bench/qk-tensile-extraction/shape_matrix.json`;
- `bench/qk-tensile-extraction/inject.json`;
- `extra/qk_hcq_attribution.py` for runtime/graph attribution if routing behavior is unclear.

Do not re-scope TPE-1 through TPE-7c.

## Phases

### A0 - preflight / invariants

Check:

- tree state and current commits;
- `bench/qk-tensile-extraction/kernarg_all.jsonl` has roles `ffn_gate_up`, `ffn_down`, `attn_q_o`;
- `TensileRunner` can still run the eager smoke;
- `extra/qk_hcq_attribution.py --include-tensile` still classifies `graph_rebind_ok`.

Gate:

- if eager injection or `TensileRunner` fails, stop and fix only that regression;
- do not proceed to model routing until preflight passes.

### A1 - JIT-dim minimal proof

Goal: one tiny JIT graph node launches the precompiled Tensile kernel with correct dims.

Approach options, in order:

1. **Shape the trivial custom kernel's ranges** so its emitted `ProgramInfo` already has the Tensile dims.
2. If range shaping is brittle, **patch the captured `ProgramInfo` dims** on the `Ops.PROGRAM` UOp and re-key/update
   `runtime_cache` consistently.
3. Keep the patch probe-local unless the exact hook is needed for model route.

Required proof:

- TinyJit captures the injected node;
- HCQGraph uses the Tensile dims, not trivial-kernel dims;
- output matches tinygrad fp16 oracle, rel_err <= 2e-2;
- no copies and no HIP runtime.

Artifact:

- `bench/qk-tensile-extraction/jit_dim_proof.json`

Gate:

- PASS: correct under TinyJit, graph captured, dims correct.
- KILL: no way to carry Tensile dims through `Ops.PROGRAM` without per-call recompile or invalid graph capture.

### A2 - one-block graph route

Goal: route one FFN block in `[feature, T]` space using injected Tensile nodes under TinyJit.

Roles:

- gate;
- up;
- down.

Rules:

- weights stay natural `[out, in]`;
- no per-matmul transposes/copies;
- only block entry/exit layout ops if already part of the graph route;
- fallback to PREFILL_V2 if any role/shape unsupported.

Required measurements:

- correctness vs tinygrad fp16 FFN block;
- graph attribution: program count, graph count, graph replays;
- wall/device or wall/proxy timing;
- compare against PREFILL_V2 block baseline.

Artifact:

- `bench/qk-tensile-extraction/one_block_graph_route.json`

Gate:

- research PASS if block speedup >= 1.25x and correctness passes;
- strong PASS if block speedup >= 1.5x and graph attribution shows no per-op host sync;
- REDIRECT if kernels are fast but graph boundary still eats the win;
- KILL if correctness/fallback fails.

### A3 - in-model research flag

Add a research-only route behind:

```bash
PREFILL_TENSILE_GEMM=1
```

Implementation constraints:

- affects PREFILL_V2 prefill path only;
- decode path untouched;
- unsupported device/shape falls back silently or with an explicit diagnostic to PREFILL_V2;
- no default changes;
- no artifact dependency unless flag is set;
- external artifact paths are explicit and documented.

Eligible roles:

- `ffn_gate/up`;
- `ffn_down`;
- `attn_q/o` if shape/layout matches TPE-5 without copies.

Keep the first route conservative. It is acceptable to route only FFN first if that isolates risk; but the final
measurement must state exactly which roles were routed.

### A4 - TPE-7d measurement

Run:

- warm pp512;
- warm pp1024 if feasible;
- dNLL <= 0.01 against current PREFILL_V2 baseline;
- fallback/off A/B;
- decode smoke to confirm unchanged behavior;
- HCQ attribution on the research path if timing does not transfer.

Required artifact:

- `bench/qk-tensile-extraction/inmodel_measurement.json`

Required doc:

- `docs/prefill-tensile-inmodel-measurement-result-20260619.md`

Report:

- routed roles;
- model/hardware/backend;
- pp512/pp1024 baseline vs flag;
- speedup;
- dNLL;
- graph attribution classification;
- fallback behavior;
- artifact dependency statement;
- verdict.

## Gates

| gate | threshold |
|---|---|
| correctness | rel_err <= 2e-2 for block/linear; model outputs finite |
| quality | dNLL <= 0.01 |
| research speed | warm pp512 >= 1.25x PREFILL_V2 |
| strong speed | warm pp512 and pp1024 >= 1.35x PREFILL_V2 |
| fallback | unsupported/off path equals current PREFILL_V2 behavior |
| decode | no decode regression / route untouched |
| graph | no per-op host sync wall like TPE-6 naive route |

Verdicts:

- **PASS_RESEARCH**: speed/quality/fallback pass, but external artifact remains research-only.
- **PASS_STRONG_POLICY_GATED**: strong speed pass; only TPE-0 artifact policy blocks landing.
- **REDIRECT_GRAPH_BOUNDARY**: role kernels work but graph routing loses speed.
- **KILL**: dims/capture/correctness/quality/fallback fail.

## Stop conditions

Stop immediately if:

- JIT dims cannot be made correct without core UOp surgery larger than this scope;
- any route requires per-call compile/re-key in the measured path;
- any role requires layout copies that erase expected Amdahl value;
- dNLL fails;
- fallback/off path changes behavior;
- decode path is touched.

## Expected decision after result

If PASS:

- keep route research-only;
- present TPE-0 policy decision: accept external HSACO dependency, keep as measurement/oracle only, or reject.

If REDIRECT/KILL:

- close Option A as non-transfer;
- retain extracted Tensile as codegen oracle for Option B.

If no policy decision is made:

- do not land default;
- retain artifact and docs as evidence.

## Claude execution prompt

Use this exact scope. Do not reopen extraction, PMU, q8 decode, pure-tinygrad codegen, or general search. The task is
only Option A research measurement:

1. Prove JIT launch dims for an injected Tensile node.
2. Route one block if dims pass.
3. Route PREFILL_V2 behind `PREFILL_TENSILE_GEMM=1` only if one-block passes.
4. Measure pp512/pp1024 + dNLL + fallback.
5. Commit docs/artifacts.

Keep every change flag-gated and research-only.
