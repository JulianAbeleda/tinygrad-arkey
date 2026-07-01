# Qwen3 14B/32B Reduce Source Resolution And Elimination Scope

Date: 2026-06-30

Status: execution scope for Claude. This continues from LDR0-LDR2. The next route is not GEMV, split-K, or topology
tuning; the measured bottleneck is unfused `r_*` reduction kernels. First resolve their source, then eliminate the
dominant class.

## Why This Is The Next Step

Model-driven attribution localized the 14B decode gap:

| bucket | ctx128 baseline | ctx128 G3-anyshape | ctx512 G3-anyshape | read |
|---|---:|---:|---:|---|
| `reduce_partial` | 52.35% | 56.93% | 47.48% | dominant bottleneck |
| `q4k_gemv` | 28.64% | 22.21% | 23.26% | G3-anyshape works; FFN is not the limiter |
| `q6k_gemv` | 14.14% | 15.43% | 14.68% | secondary |
| `lm_head` | 1.90% | 2.07% | 1.97% | not the drag |
| `attention` | absent/low | absent/low | 5.38% | secondary at ctx512 |

The single hottest row in the current G3-anyshape artifact is:

| kernel | calls/token | pct gpu-compute | current class |
|---|---:|---:|---|
| `r_8_32_4_20_4_2_32` | 40.0 | 38.59% | `reduce_other` |

So the immediate blocker is source attribution for `r_*` rows. Do not implement a reduce optimization until the hot
row is mapped to a source operation and its removability is proven.

## Source Citations

Read these before starting:

| claim | source |
|---|---|
| LDR result and target redirect | `docs/qwen-14b-32b-ldr-attribution-result-20260630.md` |
| Claude continuation scope that selected reduce elimination | `docs/qwen-14b-32b-model-driven-decode-route-continuation-scope-20260630.md` |
| current 14B baseline attribution | `bench/qk-decode-role-attribution/qwen3-14b-baseline/latest.json`, `summary.md` |
| current 14B G3-anyshape attribution | `bench/qk-decode-role-attribution/qwen3-14b-g3anyshape/latest.json`, `kernel_taxonomy.json`, `summary.md` |
| model-driven classifier | `extra/qk_decode_role_profile.py` |
| model-driven capture wrapper | `extra/qk_decode_role_attribution_modular.py` |
| Q4_K FFN split-K refutation | `docs/qwen-14b-32b-split-k-sk-result-20260630.md` |
| G3-anyshape route binding | `docs/qwen-14b-32b-truegen-q1432-result-20260630.md` |

## Non-Negotiables

- Do not optimize GEMV in this scope.
- Do not assume `r_32_4_1187`/vocab reduce is the bottleneck. It is ~0.7% combined and sampling/gumbel-adjacent.
- Do not classify reduce kernels from product alone. Shape product is useful but insufficient; use graph/event position.
- Do not hand-write a model-specific 14B route.
- Do not introduce a default-on behavior change until token-match, route-bound, memory, W==D, and rollback gates pass.
- If source resolution remains ambiguous for the hottest row, stop with a blocker. Do not implement a guessed fusion.
- GPU runs must be serialized.

## Definitions

`r_*` kernel source classes:

| class | examples | removable? |
|---|---|---|
| `rmsnorm_reduce` | reduce over hidden dim to compute sum of squares | maybe, via fused RMSNorm or route-count reduction |
| `attention_softmax_reduce` | max/sum over valid KV tokens | maybe, via existing/generated flash route or fused attention path |
| `attention_combine_reduce` | partial split combine for attention state | maybe, if combine can be fused/eliminated |
| `qk_coop_partial_combine` | reduce over partial GEMV lanes/workgroups | maybe, by preferring direct single-pass route |
| `sampling_or_vocab_reduce` | gumbel/argmax/top-k over vocab | usually not target; already measured tiny |
| `elementwise_generated_reduce` | reduction introduced by generic generated graph lowering | maybe, by rewrite/fusion |
| `unknown_reduce` | unresolved | not optimizable |

## Phase RSR0: Preserve Ordered Profile Events

Goal: build the missing trace primitive. The current tool aggregates per-kernel durations, which hides graph position.

Build:

```text
extra/qk_decode_reduce_source_trace.py
```

Requirements:

- Reuse `extra/qk_decode_role_attribution_modular.py` loading/capture structure.
- Capture `Compiled.profile_events` in order for one eager decode step, not only aggregates.
- Emit every `ProfileRangeEvent` with:
  - ordinal index;
  - kernel name;
  - duration;
  - calls if aggregated later;
  - bucket/classifier output from `extra.qk_decode_role_profile.classify_kernel`;
  - nearest previous and next non-reduce kernels;
  - local window of +/- 5 kernel names;
  - ctx and route flags.
- Run at least:

```bash
DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. \
python3 extra/qk_decode_reduce_source_trace.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --id qwen3-14b-g3anyshape \
  --ctx 128
```

Artifacts:

```text
bench/qwen-14b-32b-truegen/reduce_source_trace/ordered_events.json
bench/qwen-14b-32b-truegen/reduce_source_trace/reduce_windows.json
bench/qwen-14b-32b-truegen/reduce_source_trace/latest.json
bench/qwen-14b-32b-truegen/reduce_source_trace/summary.md
```

Pass:

```text
RSR0_PASS_ORDERED_REDUCE_TRACE
```

Block:

```text
RSR0_BLOCKED_NO_ORDERED_PROFILE_EVENTS
RSR0_BLOCKED_CAPTURE_FAILED
```

## Phase RSR1: Resolve Hot Reduce Sources

Goal: classify the hot `r_*` rows by source, starting with the row that owns most of the wall.

Use a layered resolver:

1. Kernel event position:
   - previous and next non-reduce kernels;
   - repeated pattern across layers;
   - calls/token count.
2. Shape factors:
   - product;
   - dimensions embedded in the `r_*` name;
   - ctx-dependent factors such as `start_pos`;
   - model profile facts: hidden, ffn, kv heads, q heads, head dim, layer count, vocab.
3. Differential route runs:
   - baseline vs `DECODE_Q4K_G3_ANYSHAPE=1`;
   - flash/non-flash if needed;
   - optionally disable or force specific route flags only to identify source, not to claim speed.
4. Micro-isolation where cheap:
   - isolate RMSNorm forward if the hot row sits around norm kernels;
   - isolate attention call if the hot row sits around attention kernels;
   - isolate Q6/Q4 coop partial if the hot row sits around those GEMVs.

Minimum rows to resolve:

| kernel | current pct @ctx128 G3 | required verdict |
|---|---:|---|
| `r_8_32_4_20_4_2_32` | 38.59% | must be firm, not ambiguous |
| `r_8_8_16_2_20_4_2_32n1` | 6.51% | firm or explicitly secondary |
| `r_5_2_8_16_4_28start_pos2B129` | 4.11% | firm or explicitly secondary |
| `r_5_2_4_28start_pos2B129n1` | 2.32% | firm or explicitly secondary |

Output schema for each reduce row:

```json
{
  "kernel": "r_...",
  "source_class": "rmsnorm_reduce|attention_softmax_reduce|...",
  "confidence": "firm|likely|ambiguous",
  "evidence": {
    "calls_per_token": 40,
    "pct_gpu_compute": 38.59,
    "event_windows": ["..."],
    "shape_factors": [8, 32, 4, 20, 4, 2, 32],
    "matched_profile_facts": ["layers=40", "hidden=5120"],
    "differential_runs": ["..."]
  },
  "removable_by": ["fuse_rmsnorm", "route_direct", "..."],
  "do_not_target_reason": null
}
```

Artifacts:

```text
bench/qwen-14b-32b-truegen/reduce_source_resolution/latest.json
bench/qwen-14b-32b-truegen/reduce_source_resolution/source_rows.json
bench/qwen-14b-32b-truegen/reduce_source_resolution/summary.md
```

Pass:

```text
RSR1_PASS_HOT_REDUCE_SOURCE_RESOLVED
```

Block:

```text
RSR1_BLOCKED_HOT_ROW_AMBIGUOUS
RSR1_BLOCKED_TRACE_INSUFFICIENT
```

## Phase RSR2: Select One Reduce-Elimination Candidate

Goal: pick a single implementation target from resolved source rows.

Selection rules:

| resolved top class | candidate |
|---|---|
| `rmsnorm_reduce` owns >= 15% | generated/fused RMSNorm decode route |
| `attention_softmax_reduce` or `attention_combine_reduce` owns >= 10% | generated/owned flash route guard fix or combine fusion |
| `qk_coop_partial_combine` owns >= 10% | route to direct generated single-pass GEMV or generated partial combine |
| `elementwise_generated_reduce` owns >= 10% | graph rewrite/fusion to remove generic reduce |
| no firm source owns >= 10% | stop; no safe implementation target |

For the selected candidate, write:

```text
bench/qwen-14b-32b-truegen/reduce_candidate_selection/latest.json
bench/qwen-14b-32b-truegen/reduce_candidate_selection/summary.md
```

Pass:

```text
RSR2_PASS_CANDIDATE_SELECTED
```

Stop:

```text
RSR2_STOP_NO_FIRM_TIER_B_TARGET
```

## Phase RSR3: Minimal Implementation For Selected Candidate

Do exactly one branch.

### Branch A: Fused RMSNorm

Use only if RSR2 selects `rmsnorm_reduce`.

Constraints:

- Decode-only first.
- Generated route, not hardcoded model name.
- Must handle hidden dim from profile.
- Preserve numerical behavior within existing token-match tolerance.
- Keep rollback flag.

Likely files:

```text
tinygrad/llm/model.py
extra/qk_*rmsnorm* or new generated route helper
```

### Branch B: Attention Reduce / Combine

Use only if RSR2 selects attention reduce/combine.

Constraints:

- First check route guards; a missing guard may be enough.
- If a new route is needed, reuse the generated attention substrate; no hand-ASM tile.
- Keep current flash route as rollback.

Likely files:

```text
tinygrad/llm/model.py
extra/qk_flash_decode.py
extra/qk_decode_attention_*.py
```

### Branch C: QK Coop Partial Combine

Use only if RSR2 selects coop partial combine.

Constraints:

- Do not repeat the refuted Q6_K half-warp route blindly.
- Candidate must be generated from quant/profile/target facts.
- Prove role-local correctness before model route binding.

Likely files:

```text
extra/q4_k_gemv_primitive.py
extra/q6_k_gemv_primitive.py
tinygrad/llm/model.py
extra/qk_candidate_evaluator.py
```

### Branch D: Generic Generated Reduce Fusion

Use only if RSR2 selects a generic generated reduce.

Constraints:

- Rewrite must be local and guarded.
- Do not change unrelated reductions globally.
- Add a reproducer showing the specific graph pattern removed.

Pass:

```text
RSR3_PASS_IMPLEMENTED_DEFAULT_OFF
```

Block/refute:

```text
RSR3_BLOCKED_NUMERIC
RSR3_BLOCKED_ROUTE_BINDING
RSR3_REFUTED_ROLE_LOCAL
```

## Phase RSR4: Correctness And Route-Bound Gate

Required after any implementation:

- role-local correctness if applicable;
- 14B token match at ctx128 and ctx512;
- route attribution proves the new route fires;
- hot reduce row decreases in the attribution artifact;
- rollback flag restores old route;
- memory does not exceed the existing 24GB fit envelope.

Artifacts:

```text
bench/qwen-14b-32b-truegen/reduce_correctness/latest.json
bench/qwen-14b-32b-truegen/reduce_correctness/token_match.json
bench/qwen-14b-32b-truegen/reduce_correctness/route_attribution.json
```

Pass:

```text
RSR4_PASS_CORRECT_ROUTE_BOUND
```

Block:

```text
RSR4_BLOCKED_TOKEN_MISMATCH
RSR4_BLOCKED_ROUTE_NOT_BOUND
RSR4_BLOCKED_MEMORY
```

## Phase RSR5: W==D, Llama, And 32B Transfer

Measure:

- 14B W==D at ctx128 and ctx512;
- llama.cpp matched-depth ratio;
- before/after bucket table;
- 32B transfer only if 14B passes;
- memory fit.

Threshold policy:

| verdict | condition |
|---|---|
| `RSR5_PASS_TIER_A` | >= 5% W==D gain, no protected-context regression > 1%, token-match |
| `RSR5_PASS_TIER_B` | >= 2% W==D gain, no protected-context regression > 1%, token-match |
| `RSR5_PASS_EQUIV_GENERATED_REPLACEMENT` | speed-equivalent but removes hand route or hardcoded path |
| `RSR5_REFUTED_LOW_MOVEMENT` | correct but < 2% and does not improve purity |
| `RSR5_REFUTED_REGRESSION` | any protected-context regression beyond guard |

Artifacts:

```text
bench/qwen-14b-32b-truegen/reduce_wd/latest.json
bench/qwen-14b-32b-truegen/reduce_wd/per_ctx.json
bench/qwen-14b-32b-truegen/reduce_wd/llama_compare.json
bench/qwen-14b-32b-truegen/reduce_wd/memory_fit.json
```

## Phase RSR6: Promotion Or Ledger

If RSR5 passes:

- add profile-scoped route policy;
- default remains off until promotion doc is updated;
- rollback flag documented;
- update route manifest/search ledger.

If RSR5 fails:

- record exact refutation with source class, kernel row, model profile, route flags, and W==D result;
- do not keep experimental flags on by default.

## Expected First Commands

Start with trace, not implementation:

```bash
DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. \
python3 extra/qk_decode_reduce_source_trace.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --id qwen3-14b-g3anyshape \
  --ctx 128
```

If the trace tool does not exist yet, build RSR0 first. Then resolve the hottest row:

```text
r_8_32_4_20_4_2_32
```

Do not proceed to fusion until this row has a firm source class.

## Deliverable For Claude

Return:

1. RSR phase verdict reached.
2. Ordered trace summary for the top reduce rows.
3. Firm source classification for `r_8_32_4_20_4_2_32`, or an explicit ambiguity blocker.
4. Candidate selected by RSR2, if source resolution passes.
5. Any implementation gated default-off.
6. Correctness, route-bound, W==D, llama, and memory-fit tables if RSR4/RSR5 are reached.
7. A precise ledger entry for any refuted candidate.
