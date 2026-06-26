# Decode score-broadcast full-model MMU scope

Goal: isolate why `DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1` works in direct gates but MMU-faults inside full model decode route capture.

Known pass:

| Gate | Status |
|---|---|
| Score reuse primitive | pass |
| Standalone chain | pass |
| Direct eager route | pass |
| Direct constant TinyJit route | pass |
| Direct variable-bound TinyJit route | pass |
| Minimal variable-bound chain, 1/2/4 chunks | pass |

Known fail:

| Gate | Status |
|---|---|
| Full model route, 1 chunk | AMD MMU fault |
| Full model route, 4 chunks | blocked by same class |
| W==D | blocked |

Note: one-chunk and reduced-chunk routes are liveness/materialization diagnostics only. They duplicate earlier PV chunks into later output-column ranges and are not correctness or W==D candidates.

Next isolation gates:

| Gate | Purpose |
|---|---|
| `extra/qk_decode_score_broadcast_model_cache_view_gate.py` | Reproduce the model's `assigned_kv = cache.after(slice.store(stack(k,v)))` view outside the full transformer. |
| attention-only model gate | If cache-view passes, call only model `_attention` with real model cache state before full block/FFN. |
| materialization audit | If attention-only passes but full model fails, inspect full graph call order and buffer lifetimes. |

Decision rule:

Do not run W==D until the full model route gate returns `SCORE_BROADCAST_ROUTE_CLEAN__WD_NEXT`.

Current result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_model_cache_view_latest.json` | `SCORE_BROADCAST_MODEL_CACHE_VIEW_READY__ATTENTION_ONLY_NEXT` | The model-shaped `assigned_kv = cache.after(slice.store(stack(k,v)))` path passes in eager and variable-bound TinyJit. |

Interpretation:

The full-model MMU fault is not caused by the `assigned_kv` cache update/view by itself. The next isolation layer is model `_attention` with real projected q/k/v and block cache state.

Attention-only gate:

| Gate | Purpose |
|---|---|
| `extra/qk_decode_score_broadcast_attention_only_gate.py` | Prefill real model cache, then call only first block `_attention(block.attn_norm(x), start_pos)` under the score-broadcast route. |

Attention-only result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_attention_only_latest.json` | `SCORE_BROADCAST_ATTENTION_ONLY_READY__FULL_ROUTE_NEXT` | First-block `_attention` with real model cache passes in eager and variable-bound TinyJit. |

Updated interpretation:

The full-model MMU fault is now isolated beyond `_attention`. The score-broadcast route, model-shaped cache view, and real first-block `_attention` all pass. The remaining fault is in full block/forward materialization after attention is combined with residual and FFN graph work.

Staged block diagnostic:

| Gate | Purpose |
|---|---|
| `extra/qk_decode_score_broadcast_block_stage_gate.py` | Runs first block stages in order: attention, residual, ffn_norm, ffn, full_block. The first failing stage identifies where the MMU enters. |
| `extra/qk_decode_score_broadcast_block_depth_gate.py` | Runs sliced full-model forwards at increasing block depths in isolated child processes. The first failing depth identifies whether the MMU is a multi-block/materialization boundary. |
| `extra/qk_decode_score_broadcast_depth1_tail_gate.py` | Uses the failing one-block sliced model and isolates prefill, first block, output norm, output head, logits slicing, argmax, and full forward. |
| `extra/qk_decode_score_broadcast_depth1_step_gate.py` | Uses the one-block sliced model and varies repeated TinyJit decode step count to find whether cache reuse after the first score-broadcast decode triggers the fault. |
| `extra/qk_decode_score_broadcast_jit_phase_gate.py` | Separates eager repeated calls from TinyJit first-call capture and second-call replay variants: same token/position, incremented position, changed token, and normal progression. |

Decision rule:

If `attention` passes and `residual` fails, inspect attention-output lifetime/residual materialization. If `ffn` or `full_block` fails, inspect FFN interaction and memory-planner aliasing after attention.

Staged block result:

| Artifact | Verdict | Meaning |
|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_block_stage_latest.json` | `SCORE_BROADCAST_BLOCK_STAGES_READY__FULL_MODEL_NEXT` | First-block attention, residual, ffn_norm, ffn, and full_block all pass with one score-broadcast chunk. |

Updated interpretation:

The full-model MMU fault is beyond the first block. The remaining suspect is multi-block graph materialization, cumulative memory-planner pressure, or a later block-specific interaction, not the first block attention/residual/FFN path.

Block-depth bisection gate:

| Artifact | Meaning |
|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_block_depth_latest.json` | Sliced full-model liveness/materialization sweep across block depths. Reports `last_pass_depth` and `first_fail_depth`. |

Decision rule:

If the first failing depth is `1`, the problem is full-forward materialization around logits/norm after the first block. If depth `1` passes and a later depth fails, inspect cross-block buffer lifetime, aliasing, and memory-planner pressure at the transition from `last_pass_depth` to `first_fail_depth`. Reduced chunks remain diagnostic-only; full promotion requires a clean all-four-chunk route gate before W==D.

Current block-depth result:

| Artifact | Verdict | Boundary | Meaning |
|---|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_block_depth_latest.json` | `SCORE_BROADCAST_BLOCK_DEPTH_FAIL_BOUNDARY_FOUND` | `last_pass_depth=0`, `first_fail_depth=1` | The fault is not multi-block accumulation. It appears when the first block is exercised through full `m.forward`, after the standalone first-block stage gate already passed. Next isolation target is full-forward wrapper materialization around first-block output, final norm, logits, and TinyJit capture. |

Depth-1 tail isolation:

| Artifact | Meaning |
|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_depth1_tail_latest.json` | One-block sliced model tail sweep. Reports `last_pass_stage` and `first_fail_stage` across `prefill_only`, `embed_only`, `block_only`, `output_norm`, `output_head`, `logits_slice`, `argmax_no_gumbel`, and `full_forward`. |

Current depth-1 tail result:

| Artifact | Verdict | Boundary | Meaning |
|---|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_depth1_tail_latest.json` | `SCORE_BROADCAST_DEPTH1_TAIL_PASS` | `last_pass_stage=full_forward`, `first_fail_stage=null` | One-shot one-block prefill, block, final norm, output head, logits slice, argmax, and full forward are live. The depth-1 failure is therefore not a simple final-norm/logits/full-forward tail bug. Since the block-depth gate uses `capture_decode` and repeats decode through one TinyJit, the next suspect is repeated decode/cache-update behavior across multiple decode steps under the score-broadcast route. |

Depth-1 repeated-decode step gate:

| Artifact | Meaning |
|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_depth1_step_latest.json` | One-block sliced model repeated-decode sweep. Reports `last_pass_steps` and `first_fail_steps` across explicit TinyJit decode step counts. |

Current repeated-decode result:

| Artifact | Verdict | Boundary | Meaning |
|---|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_depth1_step_latest.json` | `SCORE_BROADCAST_DEPTH1_STEP_FAIL_BOUNDARY_FOUND` | `last_pass_steps=1`, `first_fail_steps=2` | A one-block model can run one score-broadcast decode step, then MMU-faults on the second repeated TinyJit decode step. The blocker is now narrowed to cache update/reuse or captured-buffer lifetime across repeated decode invocations, not first-block math, final norm, logits, output head, or one-shot full forward. |

Next isolation target:

Build a reuse-mode gate that compares second-step behavior under unchanged token/start_pos inputs, incremented start_pos only, changed token only, and normal token+start_pos progression. If unchanged replay passes but incremented start_pos fails, the issue is cache-position/update reuse. If unchanged replay fails, the issue is repeated invocation/lifetime of the captured score-broadcast custom-kernel chain itself.

JIT phase gate:

| Artifact | Meaning |
|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_jit_phase_latest.json` | Diagnostic mode matrix for eager x2, TinyJit first call, TinyJit same/same replay, same-token incremented-position replay, changed-token same-position replay, and normal replay. |

Current JIT phase result:

| Artifact | Verdict | First failing mode | Meaning |
|---|---|---|---|
| `bench/qk-decode-primitive-space/score_broadcast_jit_phase_latest.json` | `SCORE_BROADCAST_JIT_PHASE_FAIL` | `jit_replay_same_same` | Eager repeated calls pass and the TinyJit first call passes. The first replay with the same token and same `start_pos` already MMU-faults, so the issue is not token mutation and not `start_pos` advancement. The current blocker is TinyJit replay/lifetime of the captured score-broadcast custom-kernel chain. |

Next fix target:

Audit the score-broadcast chain under TinyJit replay for captured intermediate lifetime and buffer aliasing. The likely pressure point is the multi-consumer custom-kernel chain `state -> pv0..pv3 -> combine`: one-shot execution is live, but replay reuses captured buffers or bind-time launch metadata in a way that becomes invalid on the second invocation.

Long-term synthesis:

| Path | Role | Long-term value | Promotion status |
|---|---|---|---|
| Persistent scratch buffers | Route-level lifetime primitive. Keep `state`, PV chunks, and combine output as realized held buffers so TinyJit does not memory-plan them as anonymous temporaries. | Unblocks capture and W==D measurement without disabling global compiler/runtime machinery. | Candidate unblocker, not itself a speed primitive. |
| Disable memory planner globally | Runtime diagnostic to test whether alias planning is the failure mode. | Useful falsifier only. It would hide real bugs and harm unrelated graphs. | Not promotable. |
| Disable graph batching | Runtime diagnostic to test whether HCQ graph batching is the failure mode. | Useful falsifier only. It gives no search primitive and may regress whole-decode overhead. | Not promotable. |
| Fuse into fewer kernels | Replace `state -> PV chunks -> combine` with a smaller generated lifecycle, eventually one q.k ownership path feeding all PV columns. | Real speed path. Reduces launch count, temp lifetime pressure, memory traffic, and scheduler alias surface. | Long-term target after capture is stable. |

Primitive map:

| Primitive | Current status | Why it matters |
|---|---|---|
| StableRouteScratch | Implemented for score-broadcast route. | Makes captured custom-kernel DAGs safe enough to benchmark without globally disabling memory planning. |
| CapturePhaseGate | Implemented as `extra/qk_decode_score_broadcast_jit_phase_gate.py`. | Separates no-JIT warmup, capture execution, and true replay so failures are not mislabeled. |
| MultiConsumerLifetime | Open compiler/runtime primitive. | TinyJit needs correct lifetime handling for custom-kernel outputs consumed by multiple downstream kernels. |
| ScoreReuseAcrossPVColumns | Partially implemented as 32-column chunked score-broadcast. | Removes per-output-column q.k recompute, but still performs one q.k pass per PV chunk plus state pass. |
| FusedScorePVLifecycle | Not implemented. | Long-term performance primitive: fewer launches and fewer intermediate buffers. |

Execution plan:

| Step | Gate | Stop condition |
|---|---|---|
| 1 | Add persistent scratch to the score-broadcast route. | Stop if syntax/import fails. |
| 2 | Fix JIT phase gate labels and require warmup/capture/replay phases in the varjit chain gate. | Stop if capture execution still MMU-faults. |
| 3 | Run `score_broadcast_jit_phase_latest.json`. | If capture passes but replay fails, inspect true replay buffer mutation; if capture fails, inspect held scratch coverage. |
| 4 | Run route gate only after the one-block JIT phase passes. | W==D remains blocked until route gate returns `SCORE_BROADCAST_ROUTE_CLEAN__WD_NEXT`. |
| 5 | After W==D is measurable, start fused-kernel primitive work. | Do not start fusion while the route cannot be captured cleanly. |

## 2026-06-26 resolution update: graph-batch barrier fixed the capture fault

Final diagnosis: the score-broadcast full-model fault was not caused by multi-block accumulation and was not fixed by persistent scratch alone. The failure boundary was TinyJit capture execution under default `JIT=1` HCQ graph batching.

Evidence:
- `score_broadcast_block_depth_latest.json`: depth 0 passed, depth 1 failed, so the blocker was not long-chain accumulation.
- `score_broadcast_depth1_tail_latest.json`: one-shot prefill/block/norm/output/logits/full_forward passed, so eager route math and full-model tail were viable.
- `score_broadcast_depth1_step_latest.json`: step 1 passed, step 2 failed; later phase labeling showed this was capture execution, not true replay.
- `score_broadcast_jit_phase_latest.json` after the graph-prefix barrier: `SCORE_BROADCAST_JIT_PHASE_PASS`, `chunks=4`, all capture and replay modes passed, with one state kernel, four PV chunk kernels, and one combine kernel present.

Implemented primitive:
- `JIT_NO_GRAPH_KERNEL_PREFIXES`: generic compiler barrier for kernels whose names must stay out of HCQ graph batching.
- Score-broadcast installs only its own prefixes when `DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1` is active.

Meaning:
- `NO_MEMORY_PLANNER=1` remains diagnostic-only and did not solve the fault.
- Global `JIT=2` remains diagnostic-only and is no longer required for this route.
- Persistent scratch remains useful as a route-lifetime primitive, but the decisive fix is the local graph-batch barrier.
- The next performance decision is no longer "can it survive TinyJit?" It is now "does the unfused six-kernel score-broadcast route win W==D enough to promote, or should work move directly to fused score+PV lifecycle?"

Next gate:
- Run the canonical decode W==D benchmark with `DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1` and `DECODE_ATTN_SCORE_BROADCAST_CHUNKS=4` under default `JIT=1`.
- Keep `score_broadcast_jit_phase_latest.json` as the preflight capture/replay gate for this route.
