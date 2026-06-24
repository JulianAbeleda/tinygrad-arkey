# Decode Role/Tensor/Kernel Attribution Solution Scope

Date: 2026-06-20

Executor: Claude

## Objective

Explain and then reduce the current Qwen3-8B decode gap versus llama.cpp by using current-route role/tensor/kernel
attribution, not stale lifecycle microbench decisions.

Current known numbers:

| route | ctx512 | ctx1024 | note |
|---|---:|---:|---|
| llama.cpp reference | `98.6 tok/s` | `97.6 tok/s` | `bench/qk-llama-token-primitive-accounting/provenance.json` |
| latest q8 tinygrad route | `72.6 tok/s` | `70.5 tok/s` | `docs/decode-q8-model-route-timing-audit-result-20260620.md` |
| gap at ctx1024 | | `~3.93 ms/token` | `10.25ms llama` vs `14.18ms tinygrad` |

The q8 lifecycle artifacts are local context only. They are superseded as whole-decode decision authority by:

- `docs/decode-q8-clock-authority-result-20260620.md`
- `docs/decode-q8-model-route-timing-audit-result-20260620.md`
- `docs/decode-q8-lifecycle-band-attribution-result-20260620.md` current-status table

Do not start more q8 lifecycle work unless the current-route attribution table proves it owns material token time.

## Non-Negotiable Measurement Policy

1. Promotion authority is full-model `W==D` timing at ctx `512,1024`, with ctx `4096` added for attention-sensitive
   changes.
2. ATT/HCQ packet visibility can prove identity, shape, and call count. It is not timing authority.
3. Same-process interleaved role A/B is acceptable for local yes/no gates before full-model timing.
4. Every result must report token effect in `ms/token` and projected `tok/s`, not only local microseconds.
5. Every script must write JSON under `bench/qk-decode-role-tensor-kernel-attribution/`.
6. Every result doc must state whether decode default behavior changed. Default should remain unchanged unless explicitly
   gated by full W==D timing and correctness.
7. Restore GPU performance state to `auto` after any controlled-clock experiment.

## Deliverable 0: Current-Route Attribution Table

This is the required first deliverable. Do not build kernels before this lands.

Create:

- `extra/qk_decode_current_route_attribution.py`
- `bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json`
- `docs/decode-current-route-attribution-result-20260620.md`

Run target:

```bash
PYTHONPATH=. python3 extra/qk_decode_current_route_attribution.py \
  --modes baseline,q8 \
  --ckpts 512 1024 4096 \
  --nmeas 20 \
  --warmups 8 \
  --out bench/qk-decode-role-tensor-kernel-attribution/current_route_attribution.json
```

The table must have one row per role/tensor/kernel family:

| field | requirement |
|---|---|
| `role` | semantic bucket: `ffn_gate/up`, `ffn_down`, `lm_head`, `attn_q/o`, `attn_k/v`, `attention_flash`, `rmsnorm`, `rope`, `elementwise`, `reduce/glue`, `other` |
| `tensor_family` | `Q4_K`, `Q6_K`, `q8_activation`, `fp/elementwise`, `attention`, etc. |
| `program_or_kernel` | tinygrad program name or llama kernel family where available |
| `calls_per_token` | measured or inferred from current route |
| `ms_per_token` | current-route timing estimate |
| `share_pct` | share of current token time |
| `effective_bw_GBs` | where weight traffic is known |
| `llama_analogue` | matching llama family if known |
| `llama_share_pct` | from `llama_runtime.json` if comparable |
| `gap_ms_per_token` | tinygrad row minus llama analogue where defensible |
| `confidence` | `timed`, `proxy`, `identity_only`, or `inferred` |
| `next_action` | `build`, `audit_more`, or `drop` |

Inputs to join:

- `bench/qk-decode-complete-tooling/role_atlas.json`
- `bench/qk-decode-complete-tooling/llama_join.json`
- `bench/qk-decode-complete-tooling/reduce_glue_ledger.json`
- `bench/qk-decode-complete-tooling/timing_audit.json`
- `bench/qk-llama-token-primitive-accounting/llama_runtime.json`
- `bench/qk-llama-token-primitive-accounting/provenance.json`
- `bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_result.json`
- `bench/qk-decode-block-map/result.json` only as stale/proxy role anatomy, clearly marked

Pass gate:

- Produces ctx `512,1024,4096` rows.
- Separates `W` wall tok/s from `D` dispatch ceiling.
- Identifies enough row mass to explain at least `2.5ms/token` of the `~3.93ms/token` ctx1024 gap, or explicitly says
  the current tooling cannot attribute the gap.

Stop condition:

- If the table cannot attribute current-route timing by role, do not build. Add only the missing instrumentation needed
  to make the table trustworthy.

## Lane 1: Q6 Big Roles (`ffn_down`, `lm_head`)

Priority: highest build candidate if Deliverable 0 confirms material current-route share.

Why:

- Existing artifacts show weak isolated Q6 bandwidth:
  - `ffn_down_q6k ~129.7 GB/s`
  - `lm_head_q6k ~91.8 GB/s`
  - `10-14%` peak class
- Older block map had:
  - `ffn_down`: `18.4%` proxy GPU time
  - `lm_head`: `13.2%` proxy GPU time
- If these still own `25-30%` of current decode, a `1.5x` local win can move whole decode by `~1.10-1.15x`.

Scope:

1. Build fresh role-local timing for `ffn_down` and `lm_head` on the current route.
2. Confirm exact program identity in-model, not only surface fallback.
3. Compare current tinygrad geometry against llama Q6 geometry:
   - tinygrad `q6k_coop_partial_4096_12288`
   - tinygrad `q6k_coop_partial_151936_4096`
   - llama `Q6_K fusion_true/fusion_false` families in `llama_join.json`
4. Identify whether loss is:
   - low occupancy;
   - bad memory coalescing;
   - reduce/glue overhead;
   - excessive VGPR/LDS;
   - shape/parts policy;
   - output projection special case.

Suggested files:

- `extra/qk_decode_q6_role_timing_audit.py`
- `bench/qk-decode-role-tensor-kernel-attribution/q6_role_timing.json`
- `docs/decode-q6-role-timing-result-20260620.md`

Build gates:

- Local role A/B candidate must show `>=1.25x` on `ffn_down` or `lm_head`, or `>=0.5ms/token` projected recovery.
- Full model W==D must show:
  - ctx512 and ctx1024 minimum speedup `>=1.03x` for one-role change;
  - correctness/dNLL unchanged within existing decode policy;
  - no ctx4096 regression over `1%`.

Expected token benefit:

| outcome | approximate whole-decode effect from `70.5 tok/s` |
|---|---:|
| Q6 roles still `~25%` share, `1.25x` local | `~74-75 tok/s` |
| Q6 roles still `~25%` share, `1.5x` local | `~78-81 tok/s` |
| Q6 roles still `~30%` share, `2.0x` local | `~88-90 tok/s` |

Stop/drop:

- Drop as first build if current-route Q6 share is `<12%` or projected full-model gain is `<3%`.

## Lane 2: Full MMVQ Family Quality

Priority: highest strategic lane. Build only after Deliverable 0 tells which role/tensor family owns the gap.

Why:

- llama decode-only runtime is `73.4%` MMVQ.
- llama effective MMVQ is `626 GB/s`, about `70%` HBM peak.
- Older tinygrad aggregate matvec was `349 GB/s`, about `39%` peak.
- Reduce ledger says tinygrad in-model weight GEMV was `44%` vs llama `54%`; projected `44 -> 54` is `1.187x` e2e.

Scope:

1. Create a current-route MMVQ ledger by family:
   - Q4 `ffn_gate/up`
   - Q4 `attn_q/o`
   - Q4/Q6 `attn_k/v`
   - Q6 `ffn_down`
   - Q6 `lm_head`
2. For each family, record:
   - effective bytes/token;
   - ms/token;
   - GB/s;
   - tinygrad program shape;
   - llama workgroup/grid/VGPR/LDS/scratch analogue;
   - reduce/glue count;
   - candidate limiting reason.
3. Rank the largest recoverable family by `gap_ms_per_token`, not by microbench speed.

Suggested files:

- `extra/qk_decode_mmvq_family_gap_ledger.py`
- `bench/qk-decode-role-tensor-kernel-attribution/mmvq_family_gap_ledger.json`
- `docs/decode-mmvq-family-gap-ledger-result-20260620.md`

Build gates:

- Any proposed MMVQ implementation must have a row-level projected recovery `>=0.7ms/token` or full-model projected
  speedup `>=5%`.
- Same-process interleaved role timing must show `>=1.15x` on the selected high-share role before full model wiring.
- Full W==D must confirm `>=1.05x` minimum speedup over ctx512/1024.

Expected token benefit:

| outcome | approximate whole-decode effect from `70.5 tok/s` |
|---|---:|
| recover `0.7ms/token` | `~74.2 tok/s` |
| recover `1.5ms/token` | `~78.8 tok/s` |
| recover `2.5ms/token` | `~85.6 tok/s` |
| recover full `44% -> 54%` ledger projection | `~83-84 tok/s` |

Stop/drop:

- Do not pursue generic renderer work unless this ledger identifies a specific role/family and a specific codegen defect.

## Lane 3: Q4 `ffn_gate/up` Exact Role Join + Timing

Priority: high audit priority, medium standalone build priority.

Why:

- `ffn_gate/up` is still `runtime_identity_only` in `role_atlas.json`.
- Older proxy share was `14.3%`.
- q8 route already gives `~1.06x` whole-model speedup, so this role is proven to matter, but not enough by itself to
  explain `70 -> 100`.

Scope:

1. Add fresh ATT/HCQ role join for `ffn_gate/up` on the current route.
2. Record exact current program names, call counts, geometry, and any q8 route replacement behavior.
3. Run role-local A/B for:
   - current native Q4;
   - q8 research route;
   - any existing imported llama Q4 path, if still present;
   - no new implementation until timing shows headroom.
4. Decide whether remaining gap is kernel quality, activation lifecycle, or already mostly recovered by q8.

Suggested files:

- `extra/qk_decode_q4_gateup_role_join_timing.py`
- `bench/qk-decode-role-tensor-kernel-attribution/q4_gateup_role_join_timing.json`
- `docs/decode-q4-gateup-role-join-timing-result-20260620.md`

Build gates:

- Fresh current-route `ffn_gate/up` share must be `>=10%`.
- Candidate must project `>=0.5ms/token` or `>=3%` full-model speedup.
- Full W==D must beat existing q8 route, not only baseline.

Expected token benefit:

| outcome | approximate whole-decode effect from `70.5 tok/s` |
|---|---:|
| `25%` local win on `14%` share | `~73 tok/s` |
| `50%` local win on `14%` share | `~75-76 tok/s` |
| `2x` local win on `14%` share | `~82 tok/s` |

Stop/drop:

- If current q8 route already collapses this role's recoverable share below `0.5ms/token`, do not build more gate/up
  variants.

## Lane 4: Attention Context-Slope Attribution

Priority: medium; high only if ctx4096 remains weak.

Why:

- llama decode attention share is `7.5%`.
- Older tinygrad attention after `gqa_coop_vec` was `~13-18%`, with larger importance at long context.
- This is likely a ctx4096 slope issue more than a ctx512/1024 issue.

Scope:

1. Split attention into:
   - `attn_flash_partial`
   - `attn_flash_reduce`
   - `attn_flash_max`
   - `attn_flash_prob`
   - `attn_qk_scores`
   - `attn_other`
2. Report ctx scaling for `512,1024,2048,4096`.
3. Compare tinygrad attention share against llama attention share and kernels:
   - `flash_attn_tile`
   - `flash_attn_stream_k_fixup_general`
   - `flash_attn_combine_results`
4. Identify whether the slope is partial compute, reduce/fixup, or extra tinygrad small ops.

Suggested files:

- `extra/qk_decode_attention_slope_attribution.py`
- `bench/qk-decode-role-tensor-kernel-attribution/attention_slope_attribution.json`
- `docs/decode-attention-slope-attribution-result-20260620.md`

Build gates:

- ctx4096 attention must be `>=15%` current-route share, or ctx1024 attention must be `>=10%`.
- Candidate must improve ctx4096 by `>=5%` without regressing ctx512/1024 over `1%`.

Expected token benefit:

| outcome | effect |
|---|---|
| attention `18% -> 12%` at long ctx | about `1.07x` long-context decode |
| attention `18% -> 7.5%` at long ctx | about `1.13x` long-context decode |
| ctx512/1024 only | likely low single-digit tok/s unless current attribution says otherwise |

Stop/drop:

- If attention is `<10%` at ctx1024 and `<15%` at ctx4096, do not build attention changes in this pass.

## Lane 5: Small-Op / Reduce / Glue Ledger

Priority: medium-low; useful cleanup only if current table shows large residual.

Why:

- Older maps had about `~1000 programs/token`.
- Host sync is already effectively `0%`, so launch count alone is not the issue.
- Previous reduce/glue standalone route failed the build gate:
  - stage2 tax `~6.8us`
  - `~10%` on one Q4 surface
  - not enough for standalone direct-output/reduce-fusion.

Scope:

1. Recompute current-route counts and time for:
   - elementwise residual/cast/rope;
   - rmsnorm;
   - reduce/glue;
   - tiny attention fixups;
   - tail ops.
2. Group by repeated program name and by per-layer lifecycle.
3. Identify any repeated op with:
   - `>=0.25ms/token` total current-route cost;
   - simple fusion target;
   - no correctness risk.

Suggested files:

- `extra/qk_decode_smallop_current_ledger.py`
- `bench/qk-decode-role-tensor-kernel-attribution/smallop_current_ledger.json`
- `docs/decode-smallop-current-ledger-result-20260620.md`

Build gates:

- A proposed small-op fusion must recover projected `>=0.3ms/token` or `>=2%` full-model speedup.
- Full W==D must confirm `>=1.02x` and no ctx regression.

Expected token benefit:

| outcome | approximate whole-decode effect from `70.5 tok/s` |
|---|---:|
| recover `0.3ms/token` | `~72.0 tok/s` |
| recover `0.7ms/token` | `~74.2 tok/s` |
| halve a true `15%` small-op share | `~76 tok/s` |

Stop/drop:

- Do not pursue if the largest repeated small-op family is `<0.2ms/token`.

## Lane 6: q8 Lifecycle / RMSNorm / Activation Quant

Priority: low for the llama gap, keep as research-only.

Why:

- q8 lifecycle is DPM/perf-state sensitive.
- Controlled `manual_peak` lifecycle can hit `58.04us` median, `9/10` pass.
- In-model q8 route gives stable `~1.06x`, around `70-72 tok/s`, with `0.0%` host-sync residual.
- llama q8 activation quant is only `3.8%` decode.

Scope:

1. Do not add more lifecycle probes unless Deliverable 0 shows q8/RMSNorm/activation owns `>=0.5ms/token`.
2. If reopened, separate:
   - activation quantization;
   - RMSNorm;
   - q8 consumer MMVQ;
   - clock lane sensitivity;
   - whole-model W/D.

Suggested files only if reopened:

- `extra/qk_decode_q8_current_role_share.py`
- `bench/qk-decode-role-tensor-kernel-attribution/q8_current_role_share.json`
- `docs/decode-q8-current-role-share-result-20260620.md`

Build gates:

- Must beat existing q8 route, not baseline.
- Must show `>=0.5ms/token` current recoverable time.

Expected token benefit:

| outcome | effect |
|---|---|
| existing q8 route | `~1.06x` proven |
| extra q8 lifecycle work | likely `+1-4 tok/s` unless it unlocks broader MMVQ coverage |

Stop/drop:

- Drop if it remains only a microbench win with no W==D model-route movement.

## Lane 7: Host Runtime / Persistent Decode

Priority: lowest for current gap.

Why:

- W==D says decode is GPU-work bound.
- q8 model-route timing reports host-sync residual `0.0%`.
- Persistent lifecycle cannot recover a `~3.93ms/token` gap if wall and dispatch are already aligned.

Scope:

1. Only run a sanity audit:
   - W tok/s;
   - D dispatch ceiling;
   - host-sync percentage;
   - programs/token.
2. Do not implement persistent/on-device lifecycle unless W and D diverge by `>=5%`.

Suggested files:

- extend Deliverable 0 rather than a separate script.

Build gate:

- W/D divergence `>=5%` at ctx512 or ctx1024.

Expected token benefit:

- Near zero under current artifacts.

Stop/drop:

- If host-sync remains `0.0%`, explicitly close this lane.

## Recommended Execution Order

1. Deliverable 0: current-route attribution table.
2. If Q6 roles are still high share: execute Lane 1 first.
3. In parallel only after Deliverable 0: execute Lane 3 role-join timing for `ffn_gate/up`.
4. Build the MMVQ family gap ledger from the completed Lane 1/Lane 3 facts.
5. Run attention slope attribution if ctx4096 remains materially worse than ctx512/1024.
6. Run small-op ledger only after weight-MMVQ mass is accounted for.
7. Close q8 lifecycle and host runtime unless current-route attribution reopens them.

## Final Report Format

Claude should finish with a single result doc:

- `docs/decode-role-tensor-kernel-attribution-result-20260620.md`

It must include:

1. Current route W/D table by ctx.
2. Ranked role/tensor/kernel attribution table.
3. Token math:
   - current `ms/token`;
   - llama `ms/token`;
   - gap `ms/token`;
   - attributed gap `ms/token`;
   - unattributed residual.
4. Build recommendations ranked by expected full-model tok/s.
5. Explicit drops: lanes that should not receive implementation work.
6. Exact commands used.
7. Artifact paths.
8. Whether default decode behavior changed.

## Expected Decision Outcomes

Likely outcomes, based on existing artifacts:

| lane | likely decision | reason |
|---|---|---|
| Q6 `ffn_down` / `lm_head` | build candidate if still high share | weak Q6 bandwidth and large roles |
| full MMVQ quality | strategic root-cause lane | only family with enough mass to approach llama |
| Q4 `ffn_gate/up` | audit first, build only if q8 did not already capture most headroom | high share but q8 whole-model win is only `~1.06x` |
| attention | context-slope candidate | more valuable at ctx4096 than ctx512/1024 |
| small ops | cleanup only | host sync is not the lever |
| q8 lifecycle | keep closed unless current table reopens | lifecycle sensitivity does not explain whole-model gap |
| persistent runtime | close unless W/D diverges | current W/D says GPU-bound |

