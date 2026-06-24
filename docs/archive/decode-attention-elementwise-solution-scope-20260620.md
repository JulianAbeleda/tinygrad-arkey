# Decode Attention + Elementwise Solution Scope

Date: 2026-06-20

Executor: Claude

## Objective

Use the new timed current-route attribution to pursue the actual remaining decode gap:

1. **Rank 1: attention flash-decode efficiency**;
2. **Rank 2: elementwise / FFN activation fusion**.

This scope supersedes the earlier Q6/MMVQ-first build expectation in
`docs/decode-role-tensor-kernel-attribution-solution-scope-20260620.md`.

Do **not** spend implementation time on Q6 `ffn_down`, `lm_head`, broad MMVQ quality, Q4 `ffn_gate/up`, q8
lifecycle, or persistent/host runtime unless a new timed artifact contradicts the current attribution.

## Starting Evidence

Read these first:

- `docs/decode-current-route-attribution-result-20260620.md`
- `docs/decode-role-tensor-kernel-attribution-result-20260620.md`
- `bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json`
- `extra/qk_decode_current_route_attribution.py`
- `bench/README.md`
- `docs/decode-q8-model-route-timing-audit-result-20260620.md`

Current timed decode state:

| mode | ctx512 | ctx1024 | ctx4096 | host-sync |
|---|---:|---:|---:|---:|
| baseline | `68.5 tok/s` | `66.9 tok/s` | `61.2 tok/s` | `0.0%` |
| q8 opt-in | `72.8 tok/s` | `71.0 tok/s` | `64.5 tok/s` | `0.0%` |

Baseline ctx1024 gap versus llama:

| family | gap ms/tok | conclusion |
|---|---:|---|
| attention | `+2.73` | Rank 1, ctx-growing |
| elementwise | `+1.83` | Rank 2, flat |
| weight-GEMV | `+0.41` | drop as build |
| rmsnorm | `-0.12` | drop |
| glue/other | `-0.13` | drop |

The decisive point: **attention + elementwise explain essentially the whole gap**. Weight-GEMV/MMVQ is at or above
llama parity in-model and must not be the next build target.

## Measurement Policy

1. Full-model W==D decode timing is the promotion authority.
2. PROFILE GPU timestamps are acceptable for attribution and local cost ranking, but not final promotion.
3. Same-process interleaved A/B is acceptable for local candidate gates.
4. Every candidate must report:
   - local ms/token movement;
   - full W tok/s;
   - D dispatch ceiling;
   - host-sync percentage;
   - correctness/dNLL or exact greedy policy where relevant;
   - ctx512/1024/4096.
5. No default decode behavior changes unless the full W==D gate passes and the owner explicitly accepts the policy.
6. Restore GPU perf state to `auto` after any controlled-clock run.

## Deliverable 1: Attention Cost Split

Before building attention changes, split the current attention cost into actionable buckets.

Create:

- `extra/qk_decode_attention_cost_split.py`
- `bench/qk-decode-attention-elementwise/attention_cost_split.json`
- `docs/decode-attention-cost-split-result-20260620.md`

Run target:

```bash
PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_cost_split.py \
  --modes baseline,q8 \
  --ckpts 512 1024 2048 4096 \
  --nmeas 20 \
  --warmups 8 \
  --out bench/qk-decode-attention-elementwise/attention_cost_split.json
```

The split must classify attention kernels into:

| bucket | examples / expected content |
|---|---|
| `partial_compute` | `flash_partial_coop_vec` or equivalent main attention compute |
| `reduce_fixup` | reduce/fixup kernels such as `r_2_8_128...`, `r_1024_16...`, start_pos-dependent reduce rows |
| `softmax_stats` | max/den/prob/gmax/combine-style stat kernels |
| `qk_scores_or_other` | any non-flash attention leftovers |
| `unclassified_attention` | must be small or explicitly explained |

For each ctx and mode, report:

- ms/token;
- % wall;
- calls/token;
- top program names;
- ctx slope;
- q8 deltas.

Pass gate:

- Classifies at least `90%` of attention ms/token into named buckets.
- Shows which bucket owns at least `1.0ms/token` at ctx1024 or `2.0ms/token` at ctx4096.

Stop condition:

- If attention cannot be split into actionable buckets, do not build. Fix instrumentation/mapping first.

## Deliverable 2: Attention Candidate A/B

Only after Deliverable 1 identifies the dominant attention bucket, test bounded candidates. Do not rewrite all
attention at once.

Likely target:

- reduce/stat/fixup overhead around the current `gqa_coop_vec` / flash-decode path, not the weight-GEMV route.

Possible candidate classes:

1. **Reduce/stat fusion:** combine max/den/prob/gmax/combine or reduce-fixup kernels where shape/lifecycle permits.
2. **Chunk policy tuning:** adjust `FLASH_L` or split policy only if cost split shows fixed overhead dominates.
3. **GQA grouped tile reshape:** reduce repeated reductions across GQA groups if current program geometry is doing
   avoidable per-head/per-group work.
4. **Short-context attention bypass:** if ctx512/1024 fixed overhead dominates, add a faster path for small KV.

Create candidate-specific scripts/docs, for example:

- `extra/qk_decode_attention_candidate_ab.py`
- `bench/qk-decode-attention-elementwise/attention_candidate_ab.json`
- `docs/decode-attention-candidate-ab-result-20260620.md`

Local gate:

- Same-process interleaved A/B.
- Candidate improves attention-local ms/token by:
  - `>=0.5ms/token` at ctx1024, or
  - `>=1.0ms/token` at ctx4096.
- No correctness regression under the existing decode policy.

Full-model gate:

```bash
PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
```

or a candidate-specific W==D harness over ctx `512,1024,4096`.

Promotion threshold:

- ctx4096 speedup `>=5%`;
- ctx1024 speedup `>=3%`;
- no ctx512 regression worse than `1%`;
- host-sync remains non-target;
- default unchanged unless owner-approved.

Expected benefit:

| target | recovered ms/tok | projected ctx1024 |
|---|---:|---:|
| partial attention fix | `0.7-1.0` | `~70-72 tok/s` baseline |
| large attention fix | `~2.0` | `~77 tok/s` baseline |
| ctx4096 attention fix | `2.0-4.0` | material long-context win |

## Deliverable 3: Elementwise Cost Split

The attribution says elementwise is a flat `+1.83ms/token`. Split it before building.

Create:

- `extra/qk_decode_elementwise_cost_split.py`
- `bench/qk-decode-attention-elementwise/elementwise_cost_split.json`
- `docs/decode-elementwise-cost-split-result-20260620.md`

Run target:

```bash
PYTHONPATH=. .venv/bin/python extra/qk_decode_elementwise_cost_split.py \
  --modes baseline,q8 \
  --ckpts 512 1024 4096 \
  --nmeas 20 \
  --warmups 8 \
  --out bench/qk-decode-attention-elementwise/elementwise_cost_split.json
```

Classify elementwise into:

| bucket | expected content |
|---|---|
| `ffn_activation` | `silu(gate) * up`, especially `E_49152_32_3` class |
| `rope` | RoPE elementwise kernels |
| `residual_add` | block residual additions |
| `casts_copies` | cast/copy/layout cleanup |
| `unclassified_elementwise` | must be small or explained |

Pass gate:

- Classifies at least `90%` of elementwise ms/token.
- Identifies any repeated program family `>=0.25ms/token`.
- Confirms whether `E_49152_32_3` owns `~1.4ms/token` as claimed.

## Deliverable 4: FFN Activation Fusion Candidate

Likely highest-value elementwise build:

Fuse or remove the separate FFN activation multiply:

```text
ffn_out = ffn_down(silu(gate) * up)
```

Current suspected separate kernel:

```text
E_49152_32_3
```

Potential approaches:

1. Fuse `silu(gate) * up` into the `ffn_down` input lifecycle if current tensor scheduling allows it.
2. Fuse activation into q8 producer/consumer route only if it also benefits baseline or is explicitly q8-only.
3. Fuse gate/up epilogue output layout so the activation product does not round-trip through a standalone elementwise
   kernel.
4. If true fusion is hard, test a targeted custom elementwise kernel only if it reduces launches or improves memory
   traffic enough to pass W==D.

Create:

- candidate implementation in the narrowest existing model/kernel path;
- `extra/qk_decode_ffn_activation_fusion_ab.py`
- `bench/qk-decode-attention-elementwise/ffn_activation_fusion_ab.json`
- `docs/decode-ffn-activation-fusion-result-20260620.md`

Local gate:

- Removes or materially shrinks `E_49152_32_3`.
- Recovers `>=0.5ms/token` local elementwise time at ctx1024.
- Same logits/greedy or dNLL within existing decode policy.

Full-model gate:

- W==D ctx512/1024/4096.
- ctx1024 speedup `>=3%`.
- no ctx4096 regression.
- q8 route either still works or is explicitly marked incompatible/default-off.

Expected benefit:

| target | recovered ms/tok | projected ctx1024 |
|---|---:|---:|
| half of FFN activation overhead | `~0.7` | `~70 tok/s` baseline |
| full FFN activation overhead | `~1.4` | `~74 tok/s` baseline |
| all elementwise gap | `~1.8` | `~76 tok/s` baseline |

## Stacking Policy

After one attention candidate and one elementwise candidate pass independently:

Create:

- `extra/qk_decode_attention_elementwise_stacked_timing.py`
- `bench/qk-decode-attention-elementwise/stacked_timing.json`
- `docs/decode-attention-elementwise-stacked-result-20260620.md`

Run:

- baseline;
- q8 only;
- attention candidate only;
- elementwise candidate only;
- attention + elementwise;
- q8 + attention + elementwise if compatible.

Required ctx:

- `512,1024,4096`.

Promotion gate:

- stacked ctx1024 reaches at least `80 tok/s`, or recovers `>=2.5ms/token`;
- ctx4096 improves `>=8%`;
- no correctness regression;
- no default change without owner approval.

Projection from current attribution:

| action | recovered ms/tok | approximate ctx1024 |
|---|---:|---:|
| attention large win | `~2.0` | `~77 tok/s` |
| elementwise FFN activation win | `~1.4` | `~74 tok/s` |
| stacked | `~3.4` | `~86-88 tok/s` |
| stacked + q8 | possibly higher, but must be measured |

## Explicit Drops / Do Not Reopen

These are closed by current timed attribution:

| lane | decision |
|---|---|
| Q6 `ffn_down` / `lm_head` | drop |
| full MMVQ family quality | drop |
| Q4 `ffn_gate/up` role build | drop |
| q8 lifecycle | close; keep existing opt-in |
| host/persistent runtime | close while host-sync remains `0%` |
| prefill | solved/rested; do not mix with this decode scope |

Reopen only if a new timed W==D artifact contradicts `docs/decode-current-route-attribution-result-20260620.md`.

## Final Report Format

Claude should finish with:

- `docs/decode-attention-elementwise-result-20260620.md`

It must include:

1. Current W/D baseline table.
2. Attention cost split.
3. Elementwise cost split.
4. Candidate A/B results.
5. Full W==D timing for any candidate.
6. Correctness/dNLL/greedy status.
7. Stacked projection or measured stacked result.
8. Explicit lanes dropped.
9. Exact commands.
10. Artifact paths.
11. Whether decode default behavior changed.

## Success Criteria

Minimum useful success:

- identify the exact attention bucket and exact elementwise kernel family to build next, with no implementation if
  not enough evidence.

Strong success:

- one candidate recovers `>=0.5ms/token` locally and `>=3%` full-model at ctx1024.

Best success:

- attention + elementwise stacked route reaches `>=80 tok/s` at ctx1024 with correctness intact.

