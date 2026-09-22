# Phase 1 Scope: Packed Q4_K x Q8 Gate/Up Tensor-Core Primitive

Date: 2026-08-29

## Objective

Build and qualify one isolated NVIDIA primitive for the Qwen3-8B prefill gate/up projection:

```text
M=512, N=12288, K=4096
weight=canonical packed Q4_K
activation=FP16 input, converted to Q8 records on device
accumulation=signed INT8 tensor-core path with FP32 accumulation
output=FP32
role=single gate or up projection
```

This phase succeeds only if the candidate is fully correct and materially faster than tinygrad's current installed compiler route. It does not include model integration.

## Why This Is First

Measured representative service times:

```text
tinygrad gate/up projection: 409.312 us
llama gate/up projection:    219.200 us
per-projection gap:          190.112 us
population:                  72 projections
projected regional gap:      13.688064 ms
measured gate/up gap:        13.57 ms
```

The independent regional oracle also passed:

```text
control A: 65.298799 ms
control B: 65.301318 ms
oracle:    44.830975 ms
removable service: 20.468-20.470 ms
```

The representative kernel delta explains the measured gate/up regional gap almost exactly. This makes the single-projection packed matmul the highest-confidence substrate lever.

Authority ledger:

```text
docs/task_workflow/output/nv-prefill-substrate-test-ledger-20260829.md
```

## Allowed Files

The implementation agent may create or modify only:

```text
extra/llm_research/prefill/nv_packed_qk_q8_streamk.py
extra/llm_research/prefill/nv_packed_qk_q8_gate_qualify.py
docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.md
docs/task_workflow/output/nv-packed-qk-q8-streamk-gate-phase1-result-20260829.json
```

Do not modify the installed model route or existing model harness. In particular, do not modify:

```text
extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py
```

If a required change cannot fit inside the allowed files, record the blocker and STOP. Do not broaden the patch.

## Existing Substrate To Reuse

Inspect and reuse the established packed-weight, binding, and native-program conventions rather than inventing a second ABI:

```text
extra/llm_research/prefill/nv_compiler_q4k_pp512_binding.py
extra/llm_research/prefill/nv_native_program_uop.py
tinygrad packed Q4_K transforms and renderer/codegen paths
```

Use current qualified programs only as controls or sources of ABI/layout facts. The new candidate must execute its own kernel body and produce its own materialized output.

## Required Primitive Contract

The candidate must:

1. Consume canonical packed Q4_K weights directly. No expanded or repacked hot-path weight tensor is allowed.
2. Consume FP16 activations and create the qualified Q8 record representation on device.
3. Use signed INT8 tensor-core accumulation for the matmul body.
4. Accumulate into FP32 and materialize an FP32 output for qualification.
5. Be shape-parametric in its implementation contract, while Phase 1 qualifies only `M=512, N=12288, K=4096`.
6. Use a persistent, split-K, or stream-K schedule. Approximately 170 CTAs is a measured llama clue, not a hard-coded correctness requirement.
7. Use deterministic partial reduction/fixup if K work is split.
8. Place records, output, and workspace in graph-owned/device-owned storage.
9. Allocate no memory and perform no host synchronization inside a captured/replayed candidate route.
10. Expose enough metadata for the qualifier to report launch count, grid/block geometry, workspace bytes, input pointers, output pointer, and whether any copy/expansion kernel ran.

## Qualification Inputs

Use real Qwen3-8B canonical packed gate/up weights and real-shape activation storage. Test at least two deterministic activation cases so a cached or input-independent result cannot pass:

```text
case 0: original deterministic activation
case 1: rotated/permuted deterministic activation with the same shape
```

The candidate and reference must use the same weight identity and logically identical activation values. Record model path, tensor name, dtype, shape, storage pointer identity where available, and a stable input checksum.

## Correctness Gates

All correctness gates are mandatory:

1. Full materialized output comparison, not a sampled scalar and not a reduction-only sink.
2. Every output value must be finite.
3. Compare every output element against the installed compiler reference using `rtol=0.02, atol=0.5`.
4. Report maximum absolute error, maximum relative error over numerically meaningful reference values, mismatch count, and first mismatch coordinates.
5. Repeat the exact candidate launch 20 times for each activation case and require byte-identical candidate outputs across repeats.
6. Change the activation between cases and prove that the output changes.
7. Poison output and workspace before at least one launch and prove the candidate overwrites all logically required output elements.

Any correctness or determinism failure is `FAIL`. Do not performance-tune an incorrect route.

## Structural Gates

The qualifier must fail closed unless all are true:

1. The timed candidate contains the candidate Q8 producer, candidate matmul, required deterministic fixup, and final materialized output.
2. No expanded Q4_K weight buffer or hot-path weight copy exists.
3. No cached reference output is copied or aliased into the candidate result.
4. Output and workspace pointers remain stable across timed replays.
5. No allocation occurs inside timed replays.
6. Launch inventory and geometry are recorded.
7. Timed work is synchronized at the measurement boundary.

## Performance Protocol

Serialize GPU ownership for every benchmark command:

```bash
flock -w 1200 /tmp/gpu-bench.lock <benchmark-command>
```

Use an R9 control/candidate/control sequence in the same process where possible:

```text
control A: installed tinygrad compiler route
candidate: new isolated primitive
control B: installed tinygrad compiler route
```

For each arm:

1. Warm up before collecting samples.
2. Collect at least 40 synchronized service-time samples per repetition.
3. Report minimum, median, p95, mean, and raw samples.
4. Report candidate deltas against both controls.
5. Time the complete single-projection service contract, including Q8 production and required reduction/fixup.

## Decision Thresholds

Correctness and structural gates must pass before applying performance thresholds.

```text
FAIL: candidate does not beat both controls on both minimum and median
STOP: candidate median is slower than 307.0 us, even if statistically faster
ITERATE: candidate median is <=307.0 us but >250.0 us
PHASE-1 PASS: candidate median is <=250.0 us and beats both controls on min and median
STRETCH: approach or beat llama's 219.2 us representative service time
```

`307.0 us` is approximately a 25% reduction from the 409.312 us baseline. A smaller isolated win is not sufficient evidence to fund model integration.

## Ordered Work Packet

Execute in this order. Do not skip ahead.

### P0: Freeze Facts

1. Identify the exact canonical Q4_K gate/up tensor and existing reference invocation.
2. Record packed layout, Q8 record layout, scale interpretation, output ordering, and reference service boundary.
3. Record the current control time using the qualification protocol.

STOP if the candidate cannot consume the exact canonical weight representation.

### P1: Build Correct Candidate

1. Implement the direct packed Q4_K consumer.
2. Implement or bind the on-device FP16-to-Q8 record producer.
3. Implement signed INT8 tensor-core accumulation and FP32 output.
4. Add deterministic reduction/fixup where required.
5. Expose launch and storage metadata to the qualifier.

Do not tune performance until all correctness gates pass.

### P2: Tune Only the Scheduler

Once correct, tune bounded scheduler variables:

```text
CTA count / persistent occupancy
N tile width
K work partition
stages / prefetch distance
reduction ownership
```

Do not change quantization semantics, output semantics, or weight representation while tuning.

### P3: Run Fail-Closed Qualification

1. Run both activation correctness cases.
2. Run determinism and poison tests.
3. Run structural inventory.
4. Run serialized R9 control/candidate/control performance measurement.
5. Emit Markdown and JSON evidence.

### P4: Decide

Apply the thresholds exactly. Report `FAIL`, `STOP`, `ITERATE`, or `PHASE-1 PASS`.

Do not integrate into the model in this phase, including after `PHASE-1 PASS`.

## Explicit Non-Goals

Do not implement or test any of the following in Phase 1:

```text
Q6_K support
FFN down projection
gate/up activation-record reuse
paired gate/up projection API
SiLU(gate) * up fusion
residual fusion
Q/O reshape or conversion epilogues
K/V cache-ready layout
model routing or policy changes
default enablement
full 72-projection model timing
```

These are gated follow-on phases. The prior ledger already showed that standalone Q8 reuse, fused SiLU/mul/cast, residual, Q/O copy, and K/V layout are not first-order levers by themselves.

## Required Evidence

The Markdown and JSON results must contain:

```text
status and exact threshold applied
commands and environment variables
GPU identity and software/runtime identity
model and tensor identity
input shapes, dtypes, checksums, and relevant pointers
kernel launch inventory and geometry
workspace size and pointer stability
correctness metrics for both activation cases
20-repeat determinism result
poison/overwrite result
all raw timing samples
min, median, p95, and mean for every R9 arm
deltas against both controls
observed failure or bottleneck if not PASS
exact files changed
```

Do not claim projected model tok/s as a measured result. If Phase 1 passes, report only the arithmetic projection separately:

```text
72 * (control_projection_us - candidate_projection_us)
```

Model-level validation belongs to the next gated phase.

## Promotion Gate

Only a `PHASE-1 PASS` authorizes a separate integration scope. That later scope will add a default-off model route, perform 72-projection end-to-end A/B/A timing, and compare realized model savings with the isolated arithmetic projection before extending the primitive to FFN down.
