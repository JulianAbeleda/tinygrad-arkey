# Post-Owned-Attention Holistic Primitive Audit — Scope / Claude Prompt (2026-06-23)

## Mission

Run a fresh post-owned-attention decode audit using the new GPU primitive model:

```text
model primitive -> tensor graph / runtime lifecycle -> HIP or tinygrad lowering -> LLVM AMDGPU -> AMDGCN ISA
-> resources / occupancy / memory movement -> whole-decode W==D transfer
```

The previous time-tax and gap maps are stale because the decode stack changed materially:

- `Q4K_GEMV_WARP` is default-eligible.
- owned AMDGCN attention is default-eligible.
- FO2 native fp16 cache shipped for the owned route.
- owned attention + FO2 measured roughly:
  - `+13.1%@ctx512`
  - `+16.0%@ctx1024`
  - `+18.8%@ctx2048`
  - `+23.2%@ctx4096`
- runtime-KV is deferred and should only be reconsidered from fresh residual-tax data.

This audit must answer two levels of questions:

1. **Decode outcome:** where is tinygrad now vs llama, and what residual gap remains?
2. **Primitive lifecycle:** for each remaining lane, which layer is the limiter: algorithm, work decomposition,
   memory movement, ISA/codegen, runtime/graph lifecycle, or W==D transfer?

The goal is not merely to rank timing buckets. The goal is to understand the GPU holistically enough to choose the
next bounded primitive or declare that no bounded 8B primitive remains.

## Preconditions

Do not run this audit until one of these states is true:

1. default flip is complete and committed; or
2. owner explicitly asks for a pre-flip audit using candidate env flags.

If default flip is not complete, the audit must record the exact env flags used and must not call the result
`default`. Use:

- `POST_PROMOTION_CANDIDATE_AUDIT`

not:

- `POST_DEFAULT_AUDIT`

## Required Reading

Read these first, in order:

1. `docs/amd-gpu-holistic-primitive-model-20260623.md`
2. `docs/owned-tile-post-promotion-four-step-result-20260623.md`
3. `docs/post-owned-attention-promotion-synthesis-20260623.md`
4. `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`
5. `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`
6. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
7. `docs/decode-ffn-gemv-warp-result-20260622.md`
8. `docs/tinygrad-vs-llama-decode-time-tax-diff-result-20260622.md`
9. `docs/decode-gap-audit-consolidated-20260622.md`
10. `docs/runtime-managed-kv-cache-result-20260623.md`
11. `structure/Development/performance-primitive-research-principles.md`
12. `structure/Development/session-handoff.md`

Inspect tools/artifacts:

- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_tinygrad_vs_llama_time_tax.py`
- `extra/qk_decode_audit_common.py`
- `extra/qk_owned_tile_short_ctx_probe.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- `bench/qk-tinygrad-vs-llama-time-tax/`
- `bench/qk-decode-kernel-probe/`
- `bench/qk-decode-attention-route-b-b4/`
- `bench/qk-decode-attention-route-b-b5-combine/`
- `bench/qk-decode-eval/candidates.json`
- llama rocprof traces already captured under the relevant bench directories

## Primitive Model To Apply

Every candidate lane must be classified across these layers:

| layer | audit question |
|---|---|
| algorithmic primitive | Is the math/semantic target still worth optimizing? |
| work decomposition primitive | Is the work split across waves/workgroups/CUs effectively? |
| memory movement primitive | Are bytes, dtype conversions, LDS staging, cache materialization, and HBM round trips on the critical path? |
| ISA primitive | Did the compiler/code object emit the intended AMDGCN instructions/resources? |
| runtime / graph primitive | Does the route work inside TinyJit/HCQ/model replay with correct buffers, vars, fallbacks, and lifecycle? |
| W==D transfer primitive | Does it improve whole-decode token/s with byte-identical output and stable repeats? |

Do not call a lane a primitive unless it has a lifecycle classification. A local kernel win is not enough.

## Phase Plan

### Phase 0 — Authority Lock

Record:

- HEAD commit;
- working tree status;
- GPU/arch;
- ROCm/HIP compiler path and version if used;
- model path;
- default flags;
- exact candidate env flags if this is not default-on;
- whether owned attention is actually default-on or forced by env;
- whether Q4K GEMV warp is default-on or forced by env;
- candidate registry state for both wins;
- llama reference artifact/version used;
- whether any benchmark artifacts are reused vs freshly captured.

Artifact:

- `bench/qk-post-owned-attention-default-audit/authority.json`

Verdicts:

- `AUTHORITY_LOCK_DEFAULT`
- `AUTHORITY_LOCK_CANDIDATE_FLAGS`
- `AUTHORITY_LOCK_INCOMPLETE`

### Phase 1 — W==D Confirmation

Run canonical decode W==D with the current default or candidate configuration.

Required ctx:

- 512
- 1024
- 2048
- 4096

Required checks:

- tok/s;
- wall ms/token;
- `.item()` inside the timed window;
- route fire counts:
  - owned AMDGCN attention nodes;
  - Q4K GEMV warp route if detectable;
- byte-identical or deterministic token stream;
- repeated spread;
- fallback behavior on unsupported ctx/shape if cheap;
- confirm native fp16 cache path when owned route is active;
- confirm no accidental gqa fallback when owned route is expected.

Artifact:

- `bench/qk-post-owned-attention-default-audit/wd.json`

Verdicts:

- `POST_DEFAULT_WD_CONFIRMED`
- `POST_PROMOTION_CANDIDATE_WD_CONFIRMED`
- `POST_DEFAULT_WD_REGRESSION`
- `POST_DEFAULT_ROUTE_NOT_FIRING`
- `POST_DEFAULT_AUDIT_ENV_FORCED`

Stop if correctness fails. Do not continue into primitive ranking on invalid output.

### Phase 2 — Post-Owned Tinygrad-vs-Llama Time-Tax Diff

Run or update:

- `extra/qk_tinygrad_vs_llama_time_tax.py`

under the exact authority configuration from Phase 0.

Required ctx:

- 512
- 1024
- 2048
- 4096

Required output:

- tinygrad tok/s and ms/token;
- llama reference tok/s and ms/token;
- tinygrad % of llama;
- gap ms/token;
- per-bucket wall-normalized table;
- raw GPU-busy table;
- top kernels by time;
- rendered-source classification for ambiguous buckets;
- route identity confirmation.

Artifact:

- `bench/qk-post-owned-attention-default-audit/time_tax.json`

If existing llama traces are stale/incompatible, refresh only the minimum needed. Record exact llama version/command.

Verdicts:

- `TIME_TAX_DIFF_REFRESHED`
- `TIME_TAX_DIFF_REUSED_LLAMA_ORACLE`
- `TIME_TAX_DIFF_NEEDS_LLAMA_REFRESH`
- `TIME_TAX_DIFF_UNSTABLE`

### Phase 3 — Corrected Bucket Re-Audit

Do not trust old bucket labels.

For the top residual buckets, re-render / fingerprint actual tinygrad kernels and classify by source/AST and runtime
role:

- attention;
- cache/copy/materialization;
- q8 quant;
- FFN/GEMV residual;
- projection/lm_head;
- RMSNorm/RoPE/genuine small ops;
- sampling / logits;
- host/graph/runtime overhead.

Required table:

| bucket | tinygrad ms | llama ms | gap ms | confidence | evidence | actionable? |
|---|---:|---:|---:|---|---|---|

Artifact:

- `bench/qk-post-owned-attention-default-audit/corrected_buckets.json`

Verdicts:

- `POST_DEFAULT_BUCKETS_CORRECTED`
- `POST_DEFAULT_BUCKETS_UNSTABLE`
- `POST_DEFAULT_BUCKETS_NEED_LLAMA_REFRESH`

### Phase 4 — Holistic Primitive Classification

For each top residual bucket from Phase 3, fill a lifecycle table:

| lane | algorithm | work decomposition | memory movement | ISA/codegen | runtime/graph | W==D transfer | blocker |
|---|---|---|---|---|---|---|---|

Use these blocker labels where applicable:

- `ALGORITHM_NOT_WORTH_IT`
- `WORK_DECOMPOSITION_GAP`
- `MEMORY_MOVEMENT_GAP`
- `ISA_CODEGEN_GAP`
- `RUNTIME_GRAPH_LIFECYCLE_GAP`
- `WD_TRANSFER_REFUTED`
- `MEASUREMENT_ARTIFACT`
- `NEEDS_PRIMITIVE_AUDIT`

Artifact:

- `bench/qk-post-owned-attention-default-audit/primitive_lifecycle.json`

Rules:

- A bucket with only timing evidence is `NEEDS_PRIMITIVE_AUDIT`.
- A bucket with local speedup but no W==D transfer is `WD_TRANSFER_REFUTED`.
- A bucket with a dtype/layout/copy issue is `MEMORY_MOVEMENT_GAP` unless runtime semantics are the actual wall.
- A bucket that requires TinyJit/HCQ/cache lifecycle redesign is `RUNTIME_GRAPH_LIFECYCLE_GAP`.

### Phase 5 — ISA / Code Object Primitive Audit

Add an ISA/resource audit for the promoted routes and top residual candidate lanes.

Minimum kernels to inspect if available:

- owned attention tile;
- owned attention combine;
- Q4K GEMV warp kernel;
- any top residual tinygrad-generated kernel;
- any llama oracle kernel if disassembly/metadata is already available.

Inspect with available local tools (`llvm-objdump`, `roc-objdump`, `amdllvm-objdump`, code object metadata, or existing
ROCm tooling). Do not block the whole audit if one tool is absent; record absence explicitly.

Required fields per kernel:

- symbol;
- code object path/hash if available;
- gfx target;
- group segment / LDS bytes;
- private segment / scratch bytes;
- VGPR count;
- SGPR count;
- kernarg size/layout if available;
- key instruction flags:
  - `has_v_dot2`;
  - `has_lds`;
  - `has_cross_lane`;
  - `has_vector_global_load`;
  - `has_spill`;
- notes on `s_waitcnt` / obvious memory dependency structure if inspected;
- source-level intent vs observed ISA.

Artifact:

- `bench/qk-post-owned-attention-default-audit/isa_primitive_audit.json`

Verdicts:

- `ISA_PRIMITIVES_CONFIRMED`
- `ISA_PRIMITIVE_GAP_FOUND`
- `ISA_AUDIT_PARTIAL_TOOLING_LIMIT`
- `ISA_AUDIT_NOT_RUN`

This phase does not need a perfect disassembler pipeline. It must at least establish whether we have enough tooling to
audit AMDGCN primitives systematically.

### Phase 6 — Stack / Interaction Analysis

Quantify whether the promoted wins stack:

Configurations:

1. baseline default before flip if reproducible;
2. Q4K GEMV warp only;
3. owned attention only;
4. both routes;
5. both + FO2 if FO2 is not part of default state already.

If historical artifacts are sufficient, use them; otherwise run a bounded in-process A/B.

Required table:

| config | ctx512 | ctx1024 | ctx2048 | ctx4096 | tokens match | notes |
|---|---:|---:|---:|---:|---|---|

Artifact:

- `bench/qk-post-owned-attention-default-audit/stacking.json`

Verdicts:

- `STACKING_CONFIRMED`
- `STACKING_OVERLAP_LIMITED`
- `STACKING_REGRESSION`
- `STACKING_NOT_MEASURED`

### Phase 7 — Runtime-KV Reopen Decision

Using fresh post-owned residual data, decide runtime-KV status.

Required questions:

| question | answer |
|---|---|
| Is a full-MAXC copy still present in default/candidate path? | |
| How many ms/token does it cost after FO2? | |
| Does gqa also pay it? | |
| Is it on the critical path? | |
| Is opaque-append NaN still the blocker? | |
| Is this a memory-movement gap or runtime/graph lifecycle gap? | |
| Would runtime-KV plausibly clear >=5% W==D now? | |

Verdicts:

- `RUNTIME_KV_REOPEN_JUSTIFIED`
- `RUNTIME_KV_DEFER_INCREMENTAL`
- `RUNTIME_KV_RETIRED_FOR_NOW`
- `RUNTIME_KV_NEEDS_NEW_DIAGNOSTIC`

Artifact:

- `bench/qk-post-owned-attention-default-audit/runtime_kv_decision.json`

### Phase 8 — Next Primitive Ranking

Rank next work from fresh data and lifecycle evidence, not stale assumptions.

Required table:

| rank | lane | residual gap ms | lifecycle blocker | expected W==D if solved | confidence | boundedness | first gate |
|---:|---|---:|---|---:|---|---|---|

Candidate lanes to consider:

- residual attention efficiency;
- cache/materialization/runtime-KV;
- q8 quant / small ops;
- remaining GEMV/projection/lm_head;
- host/graph overhead;
- native tinygrad codegen learning from owned AMDGCN tile;
- ISA primitive audit/tooling as an enabling step;
- cross-model/generalization instead of more 8B.

Allowed decisions:

- `NEXT_PRIMITIVE_RUNTIME_KV`
- `NEXT_PRIMITIVE_RESIDUAL_ATTENTION`
- `NEXT_PRIMITIVE_SMALL_OPS_Q8`
- `NEXT_PRIMITIVE_GEMV_RESIDUAL`
- `NEXT_PRIMITIVE_NATIVE_CODEGEN`
- `NEXT_PRIMITIVE_ISA_AUDIT_TOOLING`
- `NEXT_PROJECT_GENERALIZE_PROMOTED_ROUTES`
- `NO_BOUNDED_8B_PRIMITIVE_REMAINS`

Artifact:

- `bench/qk-post-owned-attention-default-audit/next_primitive.json`

No next primitive claim is valid unless it includes:

- residual gap ms;
- lifecycle blocker;
- plausible W==D transfer;
- first bounded gate.

### Phase 9 — Result Doc / Synthesis

Write:

- `docs/post-owned-attention-default-audit-result-20260623.md`

Required sections:

1. Verdict.
2. Authority/config.
3. Current tinygrad vs llama tok/s.
4. Current % of llama.
5. W==D confirmation.
6. Time-tax diff.
7. Corrected bucket map.
8. Holistic primitive lifecycle table.
9. ISA/code-object primitive audit.
10. Stacking analysis.
11. Runtime-KV decision.
12. Next primitive ranking.
13. Project synthesis update.
14. Artifacts and commands.
15. Files changed.
16. Working tree status.

Update:

- `docs/README.md`
- `structure/Development/session-handoff.md`
- candidate notes if needed.

Do not rewrite historical docs; add superseding notes.

## Boundaries

- No implementation in this audit.
- No default flip unless explicitly authorized in a separate task.
- No 14B/32B unless the result decision is generalization, and even then scope only.
- No runtime-KV implementation.
- No new kernels.
- No stale bucket labels without rendered-source validation.
- No next primitive claim without gap-ms + lifecycle blocker + bounded first gate.
- No ISA claims without disassembly/metadata evidence or an explicit `tooling unavailable` note.
- Do not treat local kernel speedup as success without W==D transfer.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/post-owned-attention-default-audit-scope-20260623.md` completely and execute it.

This is now a holistic GPU primitive audit, not only a timing audit. Read
`docs/amd-gpu-holistic-primitive-model-20260623.md` first and apply its lifecycle model:

```text
model primitive -> graph/runtime lifecycle -> HIP/tinygrad lowering -> LLVM AMDGPU -> AMDGCN ISA
-> resources/occupancy/memory movement -> W==D transfer
```

Run a fresh post-owned-attention default/candidate audit. The prior tinygrad-vs-llama gap maps are stale because
Q4K_GEMV_WARP, owned AMDGCN attention, and FO2 native fp16 cache changed the decode path materially.

If the owned route has not been flipped default-on, run this as a candidate audit with explicit env flags and do not
call it default. Confirm W==D, route firing, corrected time-tax buckets, primitive lifecycle classification, ISA/code
object evidence, stacking, runtime-KV status, and the next primitive ranking from fresh data.

Write all required artifacts under `bench/qk-post-owned-attention-default-audit/` and the result doc
`docs/post-owned-attention-default-audit-result-20260623.md`. Update README/session handoff with superseding notes.

Do not implement new kernels, runtime-KV, 14B/32B, or default flips in this audit. Report final verdict, commands,
artifacts, files changed, and git status.
