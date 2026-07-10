# Fable Prompt: Compiler Scheduling Review For Clustered WMMA Consumers

I want a compiler/scheduling design review. Please reason from the math and pseudocode below, identify flaws in the
current theory, and suggest the smallest robust primitive to test next.

This is not a request for vendor-specific assembly code. Treat the operations as abstract compiler IR nodes:

```text
async_shared_load(fragment)
wait_shared_loads(...)
matrix_op(A_fragment, B_fragment, accumulator)
```

## Problem

We have a generated matrix-kernel path that is numerically correct but slow. The slow path appears to issue too much
synchronization per unit of matrix work.

The fast reference shape batches shared-memory fragment loads and then executes a burst of matrix ops. The generated
shape has improved fragment reuse, but still places waits too frequently.

The bounded test shape is:

```text
M=512, N=5120, K=5120
logical tile shape = 2x2
```

## Useful-Work Math

For the matrix op in this test:

```text
1 matrix_op = 16 * 16 * 16 FMAs = 8192 FLOPs
useful_flops = matrix_op_count * 8192
flops_per_overhead(kind) = useful_flops / count(kind)
```

Measured structural counters:

| Route | Quality gate | matrix ops | useful FLOPs | waits/op | max op burst | shared loads/op | inst/op | FLOPs/wait |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fast reference `2x2` | pass | 64 | 524288 | 0.406 | 4 | 2.0 | 9.547 | 20165 |
| Generated baseline `2x2` | fail | 16 | 131072 | 3.312 | 1 | 4.0 | 39.062 | 2473 |
| Generated K-major `2x2` | fail | 16 | 131072 | 2.875 | 3 | 2.0 | 34.625 | 2849 |
| Generated K-major + wait coalesce | fail | 16 | 131072 | 2.562 | 4 | 2.0 | 34.312 | ~3197 |
| Generated `4x4` math oracle + wait coalesce | fail/wrong output | 64 | 524288 | 1.141 | 16 | 1.0 | 23.078 | ~7182 |

Interpretation:

```text
K-major improved fragment reuse:
  shared loads/op: 4.0 -> 2.0
  max op burst:    1 -> 3

But it did not fix wait amortization:
  fast reference: 0.406 waits/op
  generated:      2.562-2.875 waits/op
```

## Why We Think The Fast Reference Is Fast

The fast reference counters cover a larger scheduling window than the generated K-major probe:

```text
fast reference:      64 matrix ops
generated K-major:   16 matrix ops
```

That difference is important. Per-op normalization is useful for seeing density, but it can make fixed overhead look
like a per-op instruction problem. The strongest non-confounded signal is wait amortization: the fast reference does 4x
the matrix work with fewer absolute waits.

Fast reference `2x2` raw counts:

```text
matrix_op_count       = 64
useful_flops          = 64 * 8192 = 524288
wait_count            = 26
shared_load_count     = 128
instruction_count     = 611
non_matrix_inst_count = 611 - 64 = 547
max_matrix_op_burst   = 4
average_ops_per_wait  = 64 / 26 ~= 2.46
```

Derived math:

```text
waits/op              = 26 / 64  = 0.406
shared_loads/op       = 128 / 64 = 2.0
instructions/op       = 611 / 64 = 9.547

FLOPs/wait            = 524288 / 26  ~= 20165
FLOPs/shared_load     = 524288 / 128 = 4096
FLOPs/instruction     = 524288 / 611 ~= 858
```

Generated K-major raw counts:

```text
matrix_op_count       = 16
useful_flops          = 16 * 8192 = 131072
wait_count            = 46
shared_load_count     = 32
instruction_count     = 554
non_matrix_inst_count = 554 - 16 = 538
max_matrix_op_burst   = 3
average_ops_per_wait  = 16 / 46 ~= 0.35
```

Derived math:

```text
waits/op              = 46 / 16  = 2.875
shared_loads/op       = 32 / 16  = 2.0
instructions/op       = 554 / 16 = 34.625

FLOPs/wait            = 131072 / 46  ~= 2849
FLOPs/shared_load     = 131072 / 32  = 4096
FLOPs/instruction     = 131072 / 554 ~= 237
```

This is the key inference:

```text
Generated K-major already matches the fast reference on shared-load amortization:
  FLOPs/shared_load = 4096 in both

The real structural gap is wait amortization:
  absolute waits        = 46 generated vs 26 fast reference
  average ops/wait      = 0.35 generated vs 2.46 fast reference
  FLOPs/wait            = 2849 generated vs 20165 fast reference

The instruction/op gap is mostly a density/window artifact:
  non-matrix inst count = 538 generated vs 547 fast reference
  FLOPs/instruction    = 237 generated vs 858 fast reference because the fast reference spreads similar fixed overhead
                         across 4x the matrix work.
```

Additional existing-flag tests:

```text
4x2 and 2x4 legal generated windows:
  wait coalesce improves max burst to 8 and waits/op to 1.781, but timing is worse and P8 still fails.

4x4 generated:
  moves much closer to the math target (64 ops, max burst 13-16, waits/op 1.141-1.375),
  but output is wrong (`rr=nan`), so it is only a math oracle, not a route.
```

So the next design should not primarily target fewer shared loads. It should target:

```text
1. fewer waits per cluster,
2. larger matrix-op bursts after each wait,
3. more matrix work per fixed scheduling/lifecycle window.
```

## Current Generated Shape

Approximate shape:

```python
for phase in phases:
    async_shared_load(fragment_group_0)
    wait_shared_loads(partial)
    matrix_op(...)

    wait_shared_loads(partial)
    matrix_op(...)

    async_shared_load(fragment_group_1)
    matrix_op(...)
```

Desired shape:

```python
for cluster in phase:
    async_shared_load(all_fragments_needed_by_cluster)
    wait_shared_loads(cluster_ready)

    matrix_op(...)
    matrix_op(...)
    matrix_op(...)
    matrix_op(...)
```

## Failed Small Tests

### Test 1: Wait Coalescing Only

Idea:

```python
if next_consumer_is_matrix_op:
    wait_for_all_outstanding_shared_loads()
```

Result:

```text
waits:      46 -> 41
wait/op:    2.875 -> 2.562
max burst:  3 -> 4
speed:      12.24 -> 11.88
```

This improved structure slightly but slowed execution. It waits for more work but does not improve where loads are
placed.

### Test 2: Dependency-Only Preload

Idea:

```python
for phase in phases:
    packs = materialize_all_fragment_packs_for_phase()
    for matrix_op in phase:
        matrix_op.depends_on(packs)
```

Result:

```text
wait/op: 2.812
max burst: 3
shared loads/op: 2.0
```

With wait coalescing:

```text
wait/op: 2.562
max burst: 4
speed: 11.55
```

This also failed. Adding dependencies did not force the final stream into the desired load/wait/op cluster.

## Current Theory

The missing primitive may need to represent a cluster explicitly, not just adjust waits or dependencies.

Possible abstraction:

```python
class MatrixConsumerCluster:
    ops: list[MatrixOp]                 # e.g. 4 adjacent matrix ops
    required_a_fragments: list[Fragment]
    required_b_fragments: list[Fragment]
    shared_memory_windows: list[ByteRange]
    barrier_epoch: int
    resident_register_plan: RegisterPlan

def lower_cluster(cluster):
    # choose stable resident registers for all fragments used by the cluster
    for fragment in cluster.required_fragments:
        emit_async_shared_load(fragment, into=cluster.resident_register_plan[fragment])

    emit_wait_for_cluster_fragments(cluster)

    for op in cluster.ops:
        emit_matrix_op(op, using=cluster.resident_register_plan)
```

Safety constraints:

```text
1. All fragments are from the same barrier/synchronization epoch.
2. Shared-memory byte windows are exact and non-overwritten before the cluster consumes them.
3. Resident fragment registers are not reused until the cluster finishes.
4. The cluster does not increase register pressure past the allocator limit.
5. The lowering has a fallback to the current per-op path.
```

Success gate:

```text
waits/op <= 1.0, target about 0.4
max op burst >= 4
shared loads/op <= 2.0
inst/op materially closer to 9-12 than 34+
correctness preserved
```

Important constraint:

```text
Do not propose "just use 4x4" as the answer. 4x4 currently demonstrates the desired amortization direction but is wrong
on the GPU in this generated route. The useful design is: get 4x4-like scheduling-window amortization inside a legal
2x2/4x2/2x4 path, or explain why that is impossible.
```

## Question

Please review this as a compiler scheduling problem.

1. Is the explicit `MatrixConsumerCluster` abstraction the right primitive, or is there a smaller one?
2. Why did dependency-only preloading fail to change the final stream?
3. What is the smallest next experiment that could prove or disprove the cluster abstraction?
4. Are there known compiler techniques for this pattern, such as software-pipelined consumer groups, modulo-scheduled
   load/compute clusters, or pressure-aware rematerialized fragment residency?
5. What invariants should the implementation prove before rewriting the stream?

Please answer in pseudocode and design terms. Avoid assuming access to special hardware counters; use only final-stream
instruction counts, wait counts, shared-load counts, and correctness/timing smoke tests.
