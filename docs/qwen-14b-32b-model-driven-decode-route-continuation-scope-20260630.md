# Qwen3 14B/32B Model-Driven Decode Route Continuation Scope

Date: 2026-06-30

Status: execution scope for Claude. This supersedes the old split-K-first continuation. The new entry point is the
model-driven decode attribution tool committed in `66ec751ff`.

## Why This Scope Exists

The 14B/32B route work has reached a sharper frontier:

- `DECODE_Q4K_G3_ANYSHAPE=1` correctly binds generated G3 to large Q4_K shapes and gives a real +8-9% on 14B.
- KT proved `words_per_group` tuning cannot beat the shipped `bg=4,wpg=8` topology for the target large-K shapes.
- SK4A refuted split-K as the FFN lever: 14B `ffn_down` direct G3 is ~355 GB/s role-local, already above the full-model
  decode average of ~243 GB/s, while the split-K partial substrate is ~53-58 GB/s.
- Therefore the remaining 14B/32B gap is not proven to be Q4_K FFN topology. The next step is full-decode role
  attribution using the new model-driven tooling, then target the measured bucket.

Do not continue from 8B constants or from the stale assumption "large model gap = Q4_K FFN split-K." Use the model
profile and measured kernel buckets.

## Source Citations

Read these before changing code:

| claim | source |
|---|---|
| model-driven classifier + wrapper | `extra/qk_decode_role_profile.py`, `extra/qk_decode_role_attribution_modular.py` |
| legacy W1 now uses the shared classifier | `extra/amd_isa_weight_path_route_attribution.py` |
| 14B/32B route miss and G3-anyshape binding | `docs/qwen-14b-32b-truegen-q1432-result-20260630.md` |
| topology/words-per-group axis exhausted | `docs/qwen-14b-32b-shape-tuned-topology-kt-result-20260630.md` |
| split-K FFN refuted and target redirected | `docs/qwen-14b-32b-split-k-sk-result-20260630.md` |
| 8B system residual example, now historical not reusable as-is | `docs/amd-isa-system-residual-to-bandwidth-ceiling-scope-20260629.md` |
| benchmark baseline vs llama | `bench/models/qwen/amd-rx7900xtx-gfx1100.md` |
| route/profile search substrate | `extra/qk_candidate_evaluator.py`, `extra/qk_lanemap_template.py`, `bench/qk-search-spaces/` |

## North Star

Use the pure-search foundation to continue the route:

```text
profile from GGUF -> model-driven role attribution -> selected measured target -> generated candidate route
-> correctness + route-bound proof -> W==D + llama comparison -> default-off or profile-scoped promotion
```

No hand-written large-model route should be introduced. A generated route may replace an existing route only after it
is token-equivalent, route-bound, memory-safe, and at least tier-promotable under the repo's residual threshold policy.

## Non-Negotiables

- Do not hardcode `14B`, `32B`, `5120`, `17408`, `25600`, or `151936` in route selection logic. Those values may appear
  only in artifacts, summaries, or profile data loaded from GGUF.
- Do not copy the old 8B classifier. Use `extra.qk_decode_role_profile`.
- Do not optimize a bucket until attribution shows it is a wall-share target.
- Keep `DECODE_Q4K_G3_ANYSHAPE` and any new route default-off until promotion gates pass.
- Keep current direct G3, existing Q6_K routes, and shipped attention routes as rollback.
- If a candidate loses role-local or W==D, ledger it as refuted and stop. Do not mutate the route until it "almost"
  wins.
- GPU runs must be serialized. Do not run 14B and 32B captures concurrently.
- Memory must be part of the verdict. A route that requires extra full-weight fp16 storage is invalid for 14B/32B on
  the 24GB card unless llama.cpp also needs equivalent memory and both fit.

## Phase LDR0: Profile Sanity Gate

Goal: prove the model profile is derived from GGUF and not hardcoded.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_role_attribution_modular.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --id qwen3-14b \
  --profile-only

PYTHONPATH=. python3 extra/qk_decode_role_attribution_modular.py \
  --model /home/ubuntu/models/Qwen3-32B-Q4_K_M.gguf \
  --id qwen3-32b \
  --profile-only
```

Expected profile facts, from the current local GGUFs:

| model | hidden | ffn | vocab | key role facts |
|---|---:|---:|---:|---|
| qwen3-14b | 5120 | 17408 | 151936 | mixed Q4_K/Q6_K `ffn_down`; Q6_K lm_head `151936 x 5120` |
| qwen3-32b | 5120 | 25600 | 151936 | Q4_K `attn_qo` includes `5120 x 8192` and `8192 x 5120`; Q6_K lm_head |

Artifacts:

```text
bench/qk-decode-role-attribution/qwen3-14b/latest.json
bench/qk-decode-role-attribution/qwen3-14b/profile.json
bench/qk-decode-role-attribution/qwen3-32b/latest.json
bench/qk-decode-role-attribution/qwen3-32b/profile.json
```

Pass:

```text
LDR0_PASS_MODEL_PROFILE_PINNED
```

Block:

```text
LDR0_BLOCKED_PROFILE_PARSE
LDR0_BLOCKED_ROLE_AMBIGUITY
```

## Phase LDR1: Full Decode Bucket Attribution

Goal: measure where 14B decode time goes under both current shipped route and G3-anyshape route.

Run 14B first. Keep contexts short and comparable:

```bash
# shipped/current baseline
DEV=AMD JIT=1 PYTHONPATH=. QK_ATTR_STEPS=4 \
python3 extra/qk_decode_role_attribution_modular.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --id qwen3-14b-baseline \
  --ctxs 128,512 \
  --capture

# generated G3-anyshape candidate
DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. QK_ATTR_STEPS=4 \
python3 extra/qk_decode_role_attribution_modular.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --id qwen3-14b-g3anyshape \
  --ctxs 128,512 \
  --capture
```

Only after 14B attribution is stable, run 32B:

```bash
DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. QK_ATTR_STEPS=4 \
python3 extra/qk_decode_role_attribution_modular.py \
  --model /home/ubuntu/models/Qwen3-32B-Q4_K_M.gguf \
  --id qwen3-32b-g3anyshape \
  --ctxs 128,512 \
  --capture
```

Artifacts:

```text
bench/qk-decode-role-attribution/qwen3-14b-baseline/{latest.json,summary.md,kernel_taxonomy.json}
bench/qk-decode-role-attribution/qwen3-14b-g3anyshape/{latest.json,summary.md,kernel_taxonomy.json}
bench/qk-decode-role-attribution/qwen3-32b-g3anyshape/{latest.json,summary.md,kernel_taxonomy.json}
```

Acceptance:

- capture succeeds;
- bucket total has no unknown/uncategorized bucket above 10% without a listed reason;
- `q4k_gemv`, `q6k_gemv`, `lm_head`, `attention`, `reduce_partial`, and `norm_rope_elementwise` are separated;
- the G3-anyshape run shows Q4_K roles using `generated_g3` where structurally eligible;
- artifacts record model id, profile, ctx, route flags, and classifier provenance.

Pass:

```text
LDR1_PASS_FULL_DECODE_ATTRIBUTED
```

Block:

```text
LDR1_BLOCKED_CAPTURE
LDR1_BLOCKED_UNKNOWN_BUCKET_GT_10PCT
LDR1_BLOCKED_ROUTE_FLAGS_NOT_REFLECTED
```

## Phase LDR2: Target Selection Rules

Goal: choose exactly one next implementation target from measured role/bucket data.

Selection rules:

| measured result | next target | notes |
|---|---|---|
| `lm_head` or Q6_K bucket >= 10% and role-local route is below Q4_K/G3 efficiency | `LDR3_Q6K_ROUTE` | Must not reuse the refuted 8B half-warp blindly. Re-design from profile. |
| `attention` >= 10% at ctx512 and grows with ctx | `LDR3_ATTENTION_ROUTE` | Check whether existing 8B owned/generated attention route guards exclude 14B/32B shapes. |
| `reduce_partial` >= 15% and is not sampling/gumbel argmax | `LDR3_REDUCE_ELIMINATION` | First role-resolve reduce rows; do not repeat the 8B lm_head/gumbel confusion. |
| `norm_rope_elementwise` or `other` >= 10% | `LDR3_FUSION_OR_RUNTIME_OVERHEAD` | Kernel-count/scheduling/fusion target, not GEMV. |
| No bucket >= 10%, but W==D still far from llama | `LDR3_INTER_KERNEL_SCHEDULING` | Measure program count, host sync, graph capture, and launch shape. |
| Q4_K FFN remains dominant AND in-model Q4_K role-local rate is below full-model average | Reopen Q4_K route | This contradicts SK4A and needs fresh evidence. |

Output:

```text
bench/qwen-14b-32b-truegen/ldr2_target_selection/latest.json
bench/qwen-14b-32b-truegen/ldr2_target_selection/summary.md
```

Pass:

```text
LDR2_PASS_TARGET_SELECTED
```

Stop:

```text
LDR2_STOP_NO_TIER_B_TARGET
```

## Phase LDR3: Candidate Design For Selected Target

Do only the selected branch.

### Branch A: Q6_K / lm_head Route

Use this branch only if LDR2 selects Q6_K or lm_head.

Requirements:

- Build a role-local microbench for the actual profile shape, usually `rows=vocab`, `cols=hidden`.
- Compare current Q6_K route, generated candidate route, and a correctness reference.
- Preserve Q6_K dequant semantics. No quant demotion.
- Do not assume the 8B direct half-warp route applies; it regressed W==D and the prior lm_head reduce premise was partly sampling/gumbel.
- Candidate must be generated from quant/profile/target facts.

Candidate axes to expose:

| axis | examples |
|---|---|
| row grouping | rows per wave/workgroup |
| K-pos grouping | 16-lane Q6_K natural group, 32-lane packed group, two-row warp |
| reduction shape | in-warp, partials+reduce, hybrid |
| packed load shape | coalesced ql/qh/scales loads |
| output policy | direct final store vs partials |

Pass:

```text
LDR3_Q6K_PASS_CANDIDATE_READY
```

Refute:

```text
LDR3_Q6K_REFUTED_ROLE_LOCAL
LDR3_Q6K_REFUTED_WD_REGRESSION
```

### Branch B: Attention Route

Use this branch only if LDR2 selects attention.

Requirements:

- Derive heads, kv heads, head dim, and ctx from the model profile or runtime model config.
- Check whether existing flash route guards are 8B-specific.
- Attribute tile vs combine vs non-flash SDPA separately.
- If writing a candidate, use existing generated attention/TG substrate; no new hand-tile.

Pass:

```text
LDR3_ATTN_PASS_CANDIDATE_READY
```

Refute:

```text
LDR3_ATTN_REFUTED_LOW_WALL_SHARE
LDR3_ATTN_BLOCKED_SHAPE_UNSUPPORTED
```

### Branch C: Reduce/Fusion/Runtime Route

Use this branch only if LDR2 selects reduce/other/inter-kernel overhead.

Requirements:

- Role-resolve every hot reduce row before implementing anything.
- Separate real mathematical reductions from sampling/gumbel argmax and RMSNorm.
- Measure program count per token and host sync.
- Prefer graph/fusion/route-count reductions over per-kernel micro-optimizations if the wall is launch/program-count.

Pass:

```text
LDR3_RUNTIME_PASS_CANDIDATE_READY
```

Refute:

```text
LDR3_RUNTIME_REFUTED_NOT_WALL
```

## Phase LDR4: Minimal Correctness Gate

For the selected candidate:

- role-local numerical correctness vs reference;
- token match for 14B at ctx128 and ctx512;
- route attribution proves the candidate fired;
- rollback flag restores prior route byte-identically;
- no hidden fallback.

Artifacts:

```text
bench/qwen-14b-32b-truegen/ldr4_correctness/latest.json
bench/qwen-14b-32b-truegen/ldr4_correctness/token_match.json
bench/qwen-14b-32b-truegen/ldr4_correctness/route_attribution.json
```

Pass:

```text
LDR4_PASS_CORRECT_ROUTE_BOUND
```

Block:

```text
LDR4_BLOCKED_NUMERIC
LDR4_BLOCKED_ROUTE_BINDING
LDR4_BLOCKED_OOM
```

## Phase LDR5: W==D, Llama, And Memory Gate

Goal: decide whether the route should remain experimental, be profile-scoped, or be promoted.

Measure:

- 14B W==D at ctx128 and ctx512;
- 32B transfer if 14B passes;
- llama.cpp matched-depth comparison from existing benchmark method;
- VRAM peak and whether the route fits the 24GB card;
- route counts and bucket deltas before/after.

Pass levels:

| tier | condition |
|---|---|
| TIER_A | >= 5% W==D gain, no protected-context regression >1%, token-match, route-bound |
| TIER_B | >= 2% W==D residual gain, no protected-context regression >1%, clean rollback |
| EQUIV | within +/-1%, useful only if it replaces hand code with generated code |
| REFUTED | regression beyond guard or no live wall movement |

Artifacts:

```text
bench/qwen-14b-32b-truegen/ldr5_wd/latest.json
bench/qwen-14b-32b-truegen/ldr5_wd/per_ctx.json
bench/qwen-14b-32b-truegen/ldr5_wd/llama_compare.json
bench/qwen-14b-32b-truegen/ldr5_wd/memory_fit.json
```

Pass:

```text
LDR5_PASS_TIER_A
LDR5_PASS_TIER_B
LDR5_PASS_EQUIV_GENERATED_REPLACEMENT
```

Stop:

```text
LDR5_REFUTED_WD_REGRESSION
LDR5_REFUTED_LOW_MOVEMENT
LDR5_BLOCKED_MEMORY
```

## Phase LDR6: Promotion / Ledger

If LDR5 passes:

- add a profile-scoped route policy, not a global unconditional flag;
- generated route selected from model profile and target feature descriptors;
- rollback flag remains available;
- update route manifest and refuted/open frontier ledger.

If LDR5 fails:

- record the candidate as refuted with model profile, route, role, shape, and measured reason;
- update the next target selection only if LDR1/LDR2 data show another bucket.

Do not merge a route that only works because of a local 14B/32B name branch.

## Deliverable For Claude

Claude should return:

1. Exact LDR verdict reached.
2. The measured 14B bucket table before/after G3-anyshape.
3. Whether 32B transfer was run or deferred.
4. The selected target bucket and why.
5. Any implemented route, guarded by flag/profile policy.
6. Token-match, route-bound, W==D, llama, and memory-fit tables if implementation reaches LDR4/LDR5.
7. A precise stop reason if the measured data refutes the next candidate.

## First Command To Run

Start here:

```bash
DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. QK_ATTR_STEPS=4 \
python3 extra/qk_decode_role_attribution_modular.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --id qwen3-14b-g3anyshape \
  --ctxs 128,512 \
  --capture
```

Then run the flag-off baseline and compare. The rest of the route is selected from that delta, not from prior guesses.
