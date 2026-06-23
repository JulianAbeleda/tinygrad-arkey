# Decode Context-Slope Audit — Scope / Claude Prompt (2026-06-23)

## Mission

Explain why the buffer-identity KV read gain is smaller at ctx4096 than at ctx512/1024/2048.

Current observed post-fix table:

| ctx | old default | new default | delta | vs llama.cpp |
|---:|---:|---:|---:|---:|
| 512 | 86.7 | 102.9 | +18.7% | 105% |
| 1024 | 86.2 | 101.3 | +17.4% | 104% |
| 2048 | 84.9 | 98.7 | +16.3% | 104% |
| 4096 | 82.9 | 94.2 | +13.3% | 102% |

Hypothesis:

```text
The buffer-identity fix removed a mostly fixed / MAXC-shaped materialization tax.
As ctx grows, real ctx-linear attention/KV-scan work becomes a larger share, so percent gain shrinks.
```

This audit must verify or refute that hypothesis with synced W==D and kernel/time-tax evidence.

## Required Reading

Read first:

1. `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
2. `docs/decode-campaign-final-synthesis-20260623.md`
3. `docs/post-owned-attention-default-audit-result-20260623.md`
4. `docs/machine-code-translation-roadmap-result-20260623.md`
5. `bench/qk-decode-eval/HARNESS_GUIDE.md`
6. `structure/Development/performance-primitive-research-principles.md`
7. `structure/Development/session-handoff.md`

Inspect tools:

- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_time_tax_audit.py`
- `extra/qk_decode_materialization_check.py`
- `extra/qk_decode_route_fire_check.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_harness_contract.py`
- `bench/qk-owned-tile-buffer-identity-kv-read/`
- `bench/qk-post-parity-hardening/`
- llama reference artifacts used by post-parity docs

## Harness Requirements

Follow:

- `bench/qk-decode-eval/HARNESS_GUIDE.md`

Rules:

- clean synced W==D is the only authority for percentage claims;
- `.item()` / synchronization must be inside timed decode loop;
- PROFILE/GPU kernel timestamps are attribution only;
- DEBUG/stdout timings are debugging only;
- no raw-dispatch/no-sync headline numbers;
- record repeats, spread, git status, env, hardware;
- stamp artifacts with `extra.qk_harness_contract.stamp()` where applicable.

## Required Artifact Directory

```text
bench/qk-decode-ctx-slope-audit/
```

Required artifacts:

- `authority.json`
- `wd_by_ctx.json`
- `kernel_attribution_by_ctx.json`
- `slope_fit.json`
- `llama_comparison.json`
- `decision.json`

Required result doc:

- `docs/decode-ctx-slope-audit-result-20260623.md`

## Configs To Measure

### Config A — New Default Whole-Cache Route

Current default:

```text
DECODE_ATTN_KV_IDENTITY=1/default
owned_flash_tile_gqa_whole fires
E_49152 absent
```

### Config B — Old Slice/Materialization Route

Fallback comparator:

```text
DECODE_ATTN_KV_IDENTITY=0
slice route fires
E_49152 present
```

### Config C — Optional gqa/legacy comparator

Only if cheap and already supported:

```text
DECODE_ATTN_AMDGCN_TILE=0
```

Use for context, not required for main verdict.

### Config D — llama reference

Use existing llama refs from post-parity artifacts. Refresh only if incompatible.

## Contexts

Required:

- 512
- 1024
- 2048
- 4096

Optional if cheap:

- 3072
- 6144 or max supported, only if harness already supports it safely

## Phase 0 — Authority Lock

Record:

- HEAD;
- git status;
- GPU/arch;
- model path;
- default flags;
- route config flags;
- current owned tile candidate state;
- llama reference source;
- harness commands.

Artifact:

- `bench/qk-decode-ctx-slope-audit/authority.json`

Verdicts:

- `CTX_SLOPE_AUTHORITY_LOCKED`
- `CTX_SLOPE_AUTHORITY_INCOMPLETE_STOP`

## Phase 1 — W==D By Context

Run synced W==D for Config A and Config B at required contexts.

Required table:

| ctx | config | tok/s | ms/token | repeats | spread % | tokens match | route | E_49152 |
|---:|---|---:|---:|---:|---:|---|---|---|

Compute:

| ctx | old ms | new ms | saved ms | delta % |
|---:|---:|---:|---:|---:|

Artifact:

- `bench/qk-decode-ctx-slope-audit/wd_by_ctx.json`

Verdicts:

- `CTX_SLOPE_WD_MEASURED`
- `CTX_SLOPE_WD_UNSTABLE`
- `CTX_SLOPE_CORRECTNESS_FAIL`

Stop on correctness failure.

## Phase 2 — Route / Materialization Confirmation

For each context/config:

- confirm route fires:
  - whole route has `owned_flash_tile_gqa_whole`;
  - slice route has old/slice tile or materialization path;
- confirm materialization:
  - Config A: `E_49152` absent;
  - Config B: `E_49152` present;
- confirm buffer identity:
  - Config A: whole-buffer identity preserved;
  - Config B: sliced views/materialization.

Can reuse:

- `extra/qk_decode_route_fire_check.py`
- `extra/qk_decode_materialization_check.py`

Append to:

- `wd_by_ctx.json`

Verdicts:

- `CTX_SLOPE_ROUTE_CONFIRMED`
- `CTX_SLOPE_ROUTE_MISMATCH_STOP`

## Phase 3 — Kernel Attribution By Context

Use PROFILE/GPU timestamps or existing time-tax tooling for attribution only.

For Config A and Config B at each required context, collect:

- owned tile us/token;
- combine us/token;
- materialization/copy us/token;
- non-attention residual;
- total GPU-busy if available;
- top kernels.

Required table:

| ctx | config | tile us | combine us | materialization us | non-attn us | total gpu us | top residual |
|---:|---|---:|---:|---:|---:|---:|---|

Artifact:

- `bench/qk-decode-ctx-slope-audit/kernel_attribution_by_ctx.json`

Verdicts:

- `CTX_SLOPE_KERNEL_ATTRIBUTION_READY`
- `CTX_SLOPE_KERNEL_ATTRIBUTION_LIMITED`

Do not use these timings as promotion authority.

## Phase 4 — Slope Fit / Decomposition

Fit or tabulate a simple model:

```text
ms/token(ctx) = fixed_ms + slope_ms_per_ctx * ctx
```

Do this for:

- Config A;
- Config B;
- delta old-new;
- llama reference if enough points exist.

Required outputs:

| config | fixed_ms | slope_ms_per_1k_ctx | fit_error | interpretation |
|---|---:|---:|---:|---|

Also compute:

| ctx | saved ms | saved % | materialization share | attention ctx-linear share |
|---:|---:|---:|---:|---:|

Artifact:

- `bench/qk-decode-ctx-slope-audit/slope_fit.json`

Verdicts:

- `CTX_SLOPE_FIXED_TAX_CONFIRMED`
- `CTX_SLOPE_CTX_LINEAR_ATTENTION_DOMINATES`
- `CTX_SLOPE_MODEL_INCONCLUSIVE`

## Phase 5 — Llama Comparison

Compare Config A slope to llama reference.

Questions:

| question | answer |
|---|---|
| Is tinygrad still above llama at all measured ctx? | |
| Does tinygrad have worse ctx-linear slope than llama? | |
| Is ctx4096 margin lower due to higher slope or lower fixed advantage? | |
| Is there evidence of remaining long-context attention inefficiency? | |
| Is it worth a bounded long-ctx tile policy search? | |

Artifact:

- `bench/qk-decode-ctx-slope-audit/llama_comparison.json`

Verdicts:

- `CTX_SLOPE_LLAMA_MARGIN_EXPLAINED`
- `CTX_SLOPE_LONG_CTX_TILE_GAP_FOUND`
- `CTX_SLOPE_LLAMA_REF_INSUFFICIENT`

## Phase 6 — Decision

Write:

- `bench/qk-decode-ctx-slope-audit/decision.json`
- `docs/decode-ctx-slope-audit-result-20260623.md`

Required result doc sections:

1. Verdict.
2. Authority/config.
3. W==D by context.
4. Materialization/route confirmation.
5. Kernel attribution.
6. Slope model.
7. Llama comparison.
8. Decision:
   - no action;
   - long-ctx tile policy search;
   - artifact only.
9. Files changed.
10. Git status.

Allowed final verdicts:

- `CTX_SLOPE_FIXED_TAX_EXPLAINS_DELTA`
- `CTX_SLOPE_LONG_CTX_ATTENTION_GAP_ACTIONABLE`
- `CTX_SLOPE_AUDIT_INCONCLUSIVE`
- `CTX_SLOPE_NO_ACTION_DECODE_MAINTENANCE`

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

The owner wants to understand why the buffer-identity decode gain is smaller at ctx4096 than shorter contexts.

Read and execute:

```text
docs/decode-ctx-slope-audit-scope-20260623.md
bench/qk-decode-eval/HARNESS_GUIDE.md
docs/owned-tile-buffer-identity-kv-read-result-20260623.md
docs/decode-campaign-final-synthesis-20260623.md
```

Use synced W==D for performance authority. PROFILE/GPU timings are attribution only.

Measure:

- new default whole-cache route;
- old slice/materialization route via `DECODE_ATTN_KV_IDENTITY=0`;
- ctx 512/1024/2048/4096;
- route fire;
- `E_49152` presence/absence;
- kernel attribution;
- simple fixed+slope model;
- llama comparison from existing refs.

Do not:

- change defaults;
- implement kernels;
- start machine search;
- use no-sync/raw-dispatch headline numbers;
- treat PROFILE timing as promotion authority.

Final response must include:

- final verdict;
- whether fixed materialization tax explains the shrinking % gain;
- whether a long-ctx attention slope gap remains;
- whether any action/search is recommended;
- artifacts written;
- files changed;
- git status.
