# NV DeepSeek 240 tok/s continuation scope

Date: 2026-08-23  
Repo: `/home/ubuntu/tinygrad-arkey`  
Branch: `nvidia-bringup-20260731`  
Backend: `DEV=NV`  
GPU: RTX 5090, `sm_120`  
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`  
Workload: single-token decode, depth 512

## Mission

Work autonomously toward a measured **240 tok/s**. Do not stop at a projected
ceiling or at a renamed hypothesis. Continue until either:

1. a production-default reverse bracket proves at least 240 tok/s with exact
   token correctness, or
2. a current, zero-residual ledger identifies the exact remaining mechanism and
   the missing observable or blocked route.

Every performance claim in reports must be labeled `[MEASURED]`, `[INFERRED]`,
`[UNMEASURED]`, or `[INVALIDATED]`. Do not trust prior conclusions without
checking their raw evidence and SHA manifest.

## Current authority

The current production route has already passed a fresh control/candidate/
control composition bracket:

```text
control A       4810.981188 us/token
candidate       4697.288625 us/token
control C       4824.558875 us/token
control midpoint4817.770031 us/token
measured recovery120.481406 us/token
current speed   212.888770 tok/s
```

The 240 target is `4166.666667 us/token`. The measured current gap is therefore
`530.621958 us/token`.

All three arms reproduced token-stream SHA:

```text
f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905
```

Authority report:
`docs/task_workflow/output/nv-booked-composition-wall-result-20260823.md`

Raw bracket:
`docs/task_workflow/evidence/nv-booked-composition-wall-20260823/`

The prior `4677.993 us/token` value is a superseded cross-session projection,
not the current endpoint. The old `4771.423 us/token` and `604.756 us/token`
figures are historical pre-composition values.

## Routes already active

Do not redo these as if they were open candidates:

1. Q6_K FFN-down four-warp fp16 direct consumer, active on 18 applicable
   blocks.
2. Semantic Q/K REDUCE_OUTPUT RMSNorm+RoPE fusion, active on all 36 blocks.

Their composition is measured and token-exact. Any new candidate must be
bracketed with both routes enabled in the production candidate arm.

## Required reading before work

Read these completely and distinguish current authority from historical scope:

1. `docs/task_workflow/output/nv-booked-composition-wall-result-20260823.md`
2. `docs/task_workflow/evidence/nv-qk-norm-rope-booked-ledger-20260823.json`
3. `docs/task_workflow/output/nv-qk-norm-rope-semantic-fix-result-20260823.md`
4. `docs/task_workflow/output/nv-q6-ffn-down-confirm-result-20260823.md`
5. `docs/task_workflow/output/nv-full-token-role-census-result-20260822.md`
6. `docs/task_workflow/output/nv-r-predecessor-conditioned-exact-kv-o-result-20260823.md`
7. `docs/task_workflow/input/nv-r-pc5-partition-handoff-scope-20260823.md`
8. `docs/task_workflow/output/nv-gateup-fourwarp-profile-closure-result-20260823.md`
9. `docs/task_workflow/output/nv-installed-islands-phase5-ffn-result-20260822.md`
10. `docs/task_workflow/output/nv-installed-islands-phase11-handoff-result-20260822.md`

The Phase 11 and remaining-route ledgers predate the accepted routes. Their
tooling remains useful, but their baseline and remaining-gap numbers are not
current authority.

## Non-negotiable measurement protocol

For every measurement:

- Use one fresh process per arm.
- Serialize through `/tmp/gpu-bench.lock`.
- Lock GPU clocks and retain `nvidia-smi` state per arm.
- Use reverse order `control A -> candidate -> control C`.
- Use settled continuous decode windows, not first-token timing.
- Validate token stream SHA; use output/logit SHA where numerically relevant.
- Keep node sum, union, wall, useful body, overlap, and host gap separate.
- Retain raw JSON, timestamp/counter output, command lines, and hashes.
- Run `py_compile` and `git diff --check`.
- Reset clocks and verify idle state after every GPU session.

Do not compare tinygrad HCQ command wall with llama pure kernel duration. Do
not promote a projected ceiling. Do not infer cache state without a positive
eviction control, reverse bracket, and direct counters.

## Authorization and code boundaries

Production changes are authorized for this continuation, but they must be
minimal, generic, and target-gated.

- Shape, dtype, quantization, capability, and `NV/sm_120` gating are allowed.
- Model-name, prompt, layer-number, token-position, or Qwen-only hardcoding is
  not allowed.
- Unsupported targets and shapes must fail closed.
- Preserve AMD and Metal behavior unless separately measured.
- Preserve all unrelated dirty-worktree changes.
- Do not weaken exact token/logit correctness.
- Do not commit unless explicitly requested.

## Phase 0: rebuild the current ledger

Before choosing another kernel, profile the **current production candidate**.
The old full-token census was before the accepted composition and must be
rebased.

Close these identities with zero unexplained remainder:

```text
wall = device_union + host_gap
device_union = node_sum - overlap
useful_body = sum(command_interval - dependency_spin)
device_union = useful_body - useful_overlap
```

Produce a disjoint current census with exact row ownership. Compare against a
same-domain llama PDL-off authority where available. Reuse existing capture,
wait-exit, profile, and counter tooling rather than inventing a new timing
domain.

Required output:

- current wall, node_sum, union, overlap, useful_body, host_gap;
- row-by-row count and measured wall contribution;
- zero residual closure;
- raw evidence and SHA manifest;
- ranked recoverable terms based on current installed graph, not historical
  ceilings.

## Phase 1: partition the large residual

The historical `R` label is not a recovery claim. Re-measure current Q, O, K/V,
flash-score, and flash-combine rows using exact production cubins.

Partition each installed-vs-clean difference as:

```text
installed - clean = admission + dependency_wait + useful_body

admission = grid_start - predecessor_exit
wait      = wait_exit - grid_start
body      = kernel_exit - wait_exit
```

Use the existing `%globaltimer`/wait-exit machinery. Require forward and
reverse closure to the timestamp quantum. If closure fails, fix the harness and
label the result `[UNMEASURED]`.

Decision rules:

- Admission dominant: investigate QMD placement, launch batching, and submit
  structure.
- Wait dominant: construct the smallest dependency-legal scheduler candidate.
- Body dominant: investigate memory access, occupancy, instruction mix, and
  generated SASS.
- No term over the useful wall threshold: close the row and move on.

Do not extrapolate occurrence zero to all occurrences without a count-weighted
model. The retained K/V and O occurrence-0 closures are evidence, not proof of
a generic scheduler mechanism.

## Phase 2: candidate loop

For each candidate, perform this sequence:

1. Name the exact kernel row, occurrence count, current measured cost, and
   mechanism being attacked.
2. Build the smallest generic admission or backend change.
3. Run a bit-exact microgate against the installed kernel.
4. Run a structural profile and check node count, union, overlap, and all new
   materialization/copy/cast kernels.
5. Reject any candidate that merely moves work into an uncounted row.
6. Run a fresh control/candidate/control wall bracket with both accepted routes
   enabled in every candidate arm.
7. Require candidate below both controls and identical token SHA.
8. Confirm stable wins with an independent bracket and depth-128 non-regression.
9. Promote only through a closed-default target policy.
10. Re-run a composed bracket after promotion and update the current endpoint.

The historical `+50 us/token` bar is a prioritization threshold, not a reason
to discard stable generic wins. A positive 10–50 us result may be retained when
it has three passing brackets, exact correctness, no structural regression, and
low implementation risk. Do not spend time on sub-10 us work unless it enables
a larger fusion.

## Candidate priority

### 1. Current Q/O/K/V/flash admission-wait-body partition

This is mandatory first because it is the only historical pool large enough to
explain hundreds of microseconds. Use the retained Stage 3 instrumentation and
production cubins. Do not label cache, PDL, or scheduler causality until the
partition closes.

### 2. Generic NV quantized GEMV streaming

The old gate/up four-warp spelling failed because it added an output-cast launch.
Do not retry it unchanged. Investigate the actual memory/codegen cause using:

- real DRAM bytes and sectors;
- vector load width and coalescing;
- memory-level parallelism;
- occupancy and register pressure;
- generated PTX/SASS;
- predecessor-conditioned cache/working-set controls.

The goal is a reusable kernel/codegen improvement across applicable Q4_K/Q6_K
shapes, not a model-specific spelling.

### 3. Vocab tail topology

Re-measure the current installed row first. Historical 58.3 us is a ceiling,
not booked recovery. Look for reduction topology, launch count, and final-logit
materialization without changing output selection.

### 4. Flash combine body/topology

First separate body, admission, and wait. The historical 46.1 us ceiling is not
evidence of a viable candidate. Preserve exact softmax/PV numerics and cache
layout.

### 5. Provider-preserving attention norm work

Preserve the existing shared-Q8 provider advantage. Any norm optimization must
be evaluated with the provider path included; do not claim an isolated norm win
if it removes or regresses the shared provider.

### 6. Small launch eliminations and semantic fusions

Only pursue these after the current census shows a real wall contribution. Use
existing semantic scheduler markers so no opaque boundary materializes copies.

## Closed findings that must not be re-litigated blindly

- Naive Q/K opaque fusion: failed due to materialization kernels.
- Naive gate/up four-warp route: failed due to an extra output cast.
- Generic Q/K/V two-queue overlap: wall-negative.
- Prior broad PDL placement: wall-neutral or negative.
- Large llama PDL overlap: dependency-wait shadow, not automatically useful
  concurrency.
- Broken cache-flush and rotation probes: cache conclusions invalid.
- Precision-blocked Q4 down variants: do not retry without a new precision path.

## Promotion and terminal rules

Declare `240_MEASURED` only when all are true:

- production-default candidate is at or below `4166.666667 us/token`;
- candidate is below both fresh reverse-bracket controls;
- token SHA matches every arm;
- an independent confirmation bracket passes;
- depth 128 does not regress;
- profile confirms intended routes and no unexpected materialization;
- all evidence hashes verify;
- GPU clocks are reset.

If 240 is not reached, the final report must include the current measured
throughput, exact remaining microseconds, a zero-residual wall ledger, every
candidate verdict, remaining named terms, and the exact missing observable or
mechanism. “More optimization is needed” is not an acceptable terminal result.

## Required artifacts

- One findings-first report under `docs/task_workflow/output/`.
- One authoritative ledger JSON.
- Raw arm/timestamp/counter JSON.
- `sha256.txt` covering every retained evidence file.
- Source and tests for each landed candidate.
- `py_compile`, relevant tests, and `git diff --check` output.
- Final GPU reset state.

Final verdict must remain `240_UNMEASURED` until a measured reverse bracket
proves 240 tok/s.
