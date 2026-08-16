# NV decode wall regression: capture-time schedule prune vs KV-cache readers (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `a6e79b67d`)
Status: **measured. Production NV decode is 2.3x below its recorded baseline.**

## 1. The regression

`df3dca075` ("keep full schedule during capture so decode replay retains
KV-cache readers") changed `create_linear_with_vars` to skip the liveness prune
while a TinyJit is capturing. On the NV native decode path this regressed the
whole token wall:

| census (`route_kernel_census.py`, DEV=NV, d512) | tok/s |
| --- | ---: |
| before (`765f03f30`, Q4 fp16 promotion record) | 198.86 |
| HEAD `a6e79b67d` | **83.87** |
| HEAD with the `df3dca075` schedule change reverted | **200.89** |

All three runs keep 596 kernels/token, 5 graph groups, and token sha
`227ad3ce...`. The regression is not kernel count or token stream: the device
kernel time still sums to ~5.1 ms, while the wall is 11.9 ms, i.e. ~6.8 ms of
host/launch/replay overhead is added by carrying the unpruned schedule into
capture and memory planning.

## 2. The correctness tradeoff that forced the workaround

Reverting `df3dca075` restores 200.89 tok/s but fails
`test_generate_jit_replay_matches_full_prefix_greedy_oracle` (CPU, JIT=1):

```text
assert [3, 3, 0, 0] == [3, 3, 3, 3]
```

The liveness prune (`_drop_dead_schedule_items`) roots backward liveness at the
written buffers present in the JIT call args plus fail-safe items. During
capture it drops the attention/KV-reader chain (items 5 and 8..18 of the
23-item decode schedule) because the final vocab reader consumes an external
block-output seed buffer, not the freshly-written block-output buffer that the
residual store produces. The prune cannot see that substitution (it happens
later, in `jit_lower`'s input substitution), so it treats the freshly-written
block output as dead. Replay then omits the KV-cache readers and the LM head
samples from a non-attention input.

## 3. Why the workaround is expensive

Carrying the full resolved schedule into capture means the memory planner sees
the unpruned live set. This is the same mechanism recorded in
`nv-overlap-planner-serialization-root-cause-20260815.md`: the planner aliases
independent fan-out live ranges and collapses the decode DAG to width 1. On the
CUDA backend that record measured ~179 -> 187 tok/s with the planner disabled;
on the NV native HCQ graph the same serialization is far more expensive
(200.9 -> 83.9 tok/s).

## 4. Fix direction (not yet landed)

The workaround should not be the fix. Three concrete options, in order of
preference:

1. Make the liveness prune capture-safe: root liveness at the true output /
   substitution-target buffers so the block-output seed chain survives the
   prune, then keep the prune active during capture.
2. Narrow `_drop_dead_schedule_items` during capture to its documented purpose
   (composite-internal copy chains) instead of full backward liveness.
3. Fix the memory planner to keep independent fan-out live ranges in distinct
   arena slots (the `nv-overlap-planner-serialization-root-cause` next action),
   so the full-schedule capture no longer serializes.

Until one of these lands, the Q6 four-warp fp16 kernel win is wall-masked and
the route stays closed (see
`decode-q6k-ffn-down-fp16-geometry-route-policy.json`, empty promoted_targets).

## Evidence

- census regression: `/tmp/census_q6_wired_*.json` (83.87 tok/s)
- census with Q6 route closed: `/tmp/census_q6_disabled_*.json` (83.34 tok/s)
- census with schedule reverted: `/tmp/census_schedule_reverted_*.json` (200.89 tok/s)
- failing correctness test: `test/unit/test_llm_decode_correctness.py::test_generate_jit_replay_matches_full_prefix_greedy_oracle`
- liveness trace: `DROP_TRACE` dump of the 23-item capture schedule (kept
  `[0..4,6,7,19..22]`, dropped `[5,8..18]`; vocab reads an external seed buffer,
  not the freshly-written block output)
