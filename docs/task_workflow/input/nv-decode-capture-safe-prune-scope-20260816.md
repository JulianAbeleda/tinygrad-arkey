# NV decode capture-safe schedule prune: scope and stress test (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731` (HEAD `9013df75e`)
Status: **fix implemented and stress-tested on CPU + NV.**

## 1. Problem

`df3dca075` ("keep full schedule during capture so decode replay retains
KV-cache readers") made `create_linear_with_vars` skip the liveness prune
(`_drop_dead_schedule_items`) while a TinyJit is capturing. The change was a
correctness workaround, not a fix: it stopped the prune from dropping the
attention/KV-reader chain, but it also carried the composite-internal dead copy
kernels into the captured decode graph.

The wall consequence on the NV native decode route:

| condition | decode wall | token sha |
| --- | ---: | --- |
| before `df3dca075` | 200.89 tok/s | `227ad3ce...` |
| with `df3dca075` (skip prune) | 83.87 tok/s | `227ad3ce...` |
| revert `df3dca075` | 200.89 tok/s | `227ad3ce...` |
| **capture-safe prune (this fix)** | **197.29 tok/s** | `227ad3ce...` |

The token stream and the estimate-based device sum are unchanged. The slowdown
is entirely launch overhead from 72 dead, unnamed, no-estimate kernels that are
kept in the captured graph (668 nodes unpruned vs 596 pruned).

## 2. Root cause

The prune performs backward buffer liveness over the resolved schedule and
roots liveness at written buffers present in the JIT call args. During capture
a feedback buffer is represented as two distinct `BUFFER` UOps:

1. The fresh write twin (`A`), produced by the block-output store.
2. The realized prior-step seed twin (`B`), read by the final norm / LM-head
   consumer as an external call-arg buffer.

In the eager/ignore graph the block-output store and the final-norm reader
refer to the same `Buffer` object. During capture they do not: item 18 writes
`A`, item 19 reads `B`. The memory planner later re-aliases `A` and `B` (write
at item 18, read at item 19, disjoint lifetimes), but that substitution happens
in `jit_lower`, after `_drop_dead_schedule_items` has already run.

At the prune site there is no signal that `A` is live:

- `A` and `B` are different BUFFER UOps with different `Buffer.base`.
- Neither has a memory-semantic owner.
- `has_precompiled_output_identity()` is false for both.
- `A` has no reader in the schedule, so it looks dead and its whole producer
  chain (block output, attention, FFN residual store) is dropped.

The earlier `df3dca075` workaround avoided that wrong drop by disabling the
prune entirely during capture. That preserved the live chain but also preserved
the 72 dead copy kernels the prune exists to remove.

## 3. Fix

`tinygrad/schedule/__init__.py`, `_drop_dead_schedule_items`.

After the existing root set is built and only while
`capturing and CAPTURING`, add the write twin of a feedback buffer to the
liveness roots when all of the following hold:

- The write buffer has no reader in the schedule and is not already rooted.
- A read-only call-arg seed exists (`in i_ids`, `in call_arg_bufs`, `not in
  writes`).
- The seed matches the written buffer on `dtype`, `arg` (numel, not bytes), and
  `device`.
- The last write index precedes the seed's first read index.

The match is deliberately narrow. `dtype`, numel, and device are the stable
identity fields that survive the capture split; the ordering constraint encodes
the memory planner's aliasing precondition (write lifetime ends before the
aliased read lifetime begins).

The prune then remains active during capture:

```python
linear = _drop_dead_schedule_items(linear, linear_call.src[1:])
```

The temporary `DROP_FULL` debug block and the capture skip were removed.

### Why this is sound

Liveness roots can only add work, not remove it. The added root is justified by
the alias the planner will construct: the written buffer is genuinely consumed,
just through a substitution target that does not yet exist at prune time. The
prune is still free to drop the 72 dead copy chains because those writers do not
match a read-only seed triple.

### Residual risk

If a genuinely dead buffer happens to share `(dtype, numel, device)` with an
unrelated read-only seed consumed later in the same capture, the fix will
conservatively keep its producer chain. That is a performance-only over-keep,
never a correctness loss. The NV census below shows the real decode graph does
not hit that case: the kernel count returns exactly to 596.

## 4. Stress-test matrix

### CPU correctness

```text
PYTHONPATH=. .venv/bin/python -m pytest test/unit/test_llm_decode_correctness.py -q
6 passed
```

This includes the original oracle replay test that failed under the reverted
prune (`[3,3,0,0] != [3,3,3,3]`) and a new
`test_capture_schedule_prune_is_active_during_jit_capture` regression test that
asserts the prune is actually invoked while the JIT is capturing.

### NV decode census (RTX 5090)

```text
flock -w 600 /tmp/gpu-bench.lock timeout 900 PYTHONPATH=. DEV=NV \
  .venv/bin/python extra/llm_research/decode/route_kernel_census.py \
  --depth 512 --out /tmp/census_capture_safe_prune.json
```

| run | tok/s | kernels/token | token sha | graph groups |
| --- | ---: | ---: | --- | ---: |
| 1 | 197.289 | 596 | `227ad3ce...` | 5 |
| 2 | 198.004 | 596 | `227ad3ce...` | 5 |
| 3 | 198.875 | 596 | `227ad3ce...` | 5 |

The census `git_commit` field records HEAD (`9013df75e`); the fix was applied as
an uncommitted worktree change for these runs and is committed with this doc.

The 596 kernel count is the acceptance signal: it is the pruned graph, down
from 668 with the workaround, and it returns the route to its pre-regression
wall instead of the 83.87 tok/s workaround wall.

## 5. Why the alternatives were not chosen

1. **Root at the substitution-target seed before pruning.** The cleanest
   long-term fix conceptually, but the substitution target is produced later by
   `jit_lower`; threading that identity backward into the schedule prune is a
   larger structural change with wider blast radius.
2. **Narrow the prune during capture to only the documented composite-internal
   copy chains.** Preserves correctness but duplicates a second prune mode and
   risks leaving other dead work in the capture graph.
3. **Move backward liveness into `jit_lower` after memory planning.** Sound,
   but changes the ownership boundary between scheduling and lowering and
   touches the hot JIT path for every backend.

The landed fix keeps the existing prune as the single liveness authority and
adds the smallest capture-specific root rule that makes it correct.

## 6. Acceptance and rollback

Acceptance: the CPU correctness file is green, the new capture-active test
passes, and the NV census reports 596 kernels/token with token sha
`227ad3ce...` and a decode wall near the 200 tok/s baseline rather than the
83.87 tok/s workaround wall.

Rollback: revert the change to `_drop_dead_schedule_items` and restore the
`capturing and CAPTURING` skip if a future model shows a correctness regression.
Before doing so, check whether the model's feedback buffer matches a different
read-only seed triple; the fix can be tightened to require the seed consumer to
be on the live path without reintroducing the 72 dead kernels.
