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

All three runs keep token sha `227ad3ce...`. The regression is not the token
stream and not the estimated device work: the estimate-based device sum stays
~5.14 ms, while the wall is 11.9 ms, i.e. ~6.8 ms of unaccounted per-token
cost appears when the unpruned schedule is carried into capture.

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

## 3. Why the workaround is expensive: 72 dead kernels stay in the graph

Measured at the `jit_lower` boundary, the decode graph carries 668 nodes with
the workaround active versus 596 with the prune active. The extra 72 items are
unnamed (`name="test"`), no-estimate (`mem`/`flops` are `None`) SINK kernels,
two per transformer block (36 blocks). They are the composite-internal dead
items `_drop_dead_schedule_items` exists to remove.

Because those 72 kernels have no mem/FLOP estimates they contribute nothing to
the estimate-based device sum (hence "device time identical"), but they are
still launched on every replayed token and cost ~94 us each (~6.8 ms total).
That is the entire 200.9 -> 83.9 tok/s regression: dead launch work, not memory
planner serialization. This supersedes the planner-serialization hypothesis
from `nv-overlap-planner-serialization-root-cause-20260815.md`, which the
earlier Route B3.2 duration-weighted gate had already marked
NOT_MECHANISM_SCALE (~0.6%).

## 4. Fix direction (not yet landed)

The workaround should not be the fix. Three concrete options, in order of
preference:

1. Make the liveness prune capture-safe: root liveness at the true output /
   substitution-target buffers so the block-output seed chain survives the
   prune, then keep the prune active during capture (removes the 72 dead
   kernels without dropping the KV-reader chain).
2. Narrow `_drop_dead_schedule_items` during capture to its documented purpose
   (composite-internal copy chains) instead of full backward liveness.
3. Keep the capture prune active but add the block-output seed (the substitution
   target) to the liveness roots, so only the genuinely dead 72 kernels are
   dropped.

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
