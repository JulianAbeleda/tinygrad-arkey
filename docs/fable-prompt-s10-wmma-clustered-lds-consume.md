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

Interpretation:

```text
K-major improved fragment reuse:
  shared loads/op: 4.0 -> 2.0
  max op burst:    1 -> 3

But it did not fix wait amortization:
  fast reference: 0.406 waits/op
  generated:      2.562-2.875 waits/op
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

