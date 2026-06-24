# Post-Default Runtime-KV + ISA Primitive Course — Exhaustive Scope / Claude Prompt (2026-06-23)

## Mission

Execute the next course of action after `POST_DEFAULT_AUDIT_COMPLETE`.

The post-owned-attention holistic audit changed the project state:

- owned AMDGCN attention is default-on for the validated shape;
- Q4K GEMV warp is default-eligible and can be forced by env;
- FFN weight-GEMV is now at/near llama parity;
- attention is near llama parity;
- the leading residual is no longer attention/GEMV math;
- the leading bounded residual is runtime/cache lifecycle:
  - `E_49152` / full-MAXC KV materialization is still about `~1.5 ms/token`;
  - it was historically mislabeled as norm/small-op work;
  - FO2 removed fp32->fp16 cast tax but did not remove full-cache materialization;
  - gqa pays the materialization too;
  - runtime-KV must be re-diagnosed now that the owned tile dtype bug is fixed.

This scope has five ordered goals:

1. **Do not reopen attention or GEMV.**
2. **Run a new Runtime-KV diagnostic first, not implementation.**
3. **If the diagnostic passes, write a bounded Runtime-KV implementation scope.**
4. **If Runtime-KV fails again, classify it as core runtime work and pivot to small-op/activation fusion scope.**
5. **Build/formalize an AMDGCN ISA primitive audit tool so every future kernel carries code-object evidence.**

The core question:

```text
Can tinygrad avoid full-MAXC KV cache materialization while preserving cross-token decode correctness,
now that owned AMDGCN attention + native fp16 cache are default-on and real-cache-correct?
```

## Required Reading

Read these first, in order:

1. `docs/post-owned-attention-default-audit-result-20260623.md`
2. `docs/post-owned-attention-default-audit-scope-20260623.md`
3. `docs/amd-gpu-holistic-primitive-model-20260623.md`
4. `docs/post-owned-attention-promotion-synthesis-20260623.md`
5. `docs/owned-tile-post-promotion-four-step-result-20260623.md`
6. `docs/owned-amdgcn-tile-short-ctx-result-20260623.md`
7. `docs/owned-amdgcn-tile-real-cache-revalidation-result-20260623.md`
8. `docs/runtime-managed-kv-cache-result-20260623.md`
9. `docs/runtime-kv-graphrunner-arg-patch-result-20260623.md`
10. `docs/runtime-kv-opaque-read-result-20260623.md`
11. `docs/runtime-kv-buffer-identity-rebase-result-20260623.md`
12. `docs/kv-cache-copy-elimination-result-20260622.md`
13. `docs/decode-gap-audit-consolidated-20260622.md`
14. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
15. `docs/decode-ffn-gemv-warp-result-20260622.md`
16. `structure/Development/performance-primitive-research-principles.md`
17. `structure/Development/session-handoff.md`

Inspect relevant code/tools:

- `tinygrad/llm/model.py`
- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_time_tax_audit.py`
- `extra/qk_tinygrad_vs_llama_time_tax.py`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_owned_flash_decode.hip`
- prior Runtime-KV probes under `extra/qk_*kv*.py`
- `bench/qk-post-owned-attention-default-audit/`
- `bench/qk-decode-eval/candidates.json`

## Non-Negotiable Boundaries

- No attention tile work in this task.
- No GEMV work in this task.
- No new model default flip.
- No 14B/32B.
- No broad TinyJit/HCQ redesign unless the diagnostic explicitly proves that is the remaining wall; if so, stop and scope it.
- No one-off kernel implementation before the Runtime-KV diagnostic gates pass.
- No timing claims unless `.item()` is inside the timed window.
- No correctness claims from position-written proxies; token correctness is authority.
- No local-kernel-only success. W==D transfer is authority.
- No ISA claims without disassembly/metadata evidence or explicit tooling limitation.

## Required Artifacts Directory

Write artifacts under:

```text
bench/qk-post-default-runtime-kv-course/
```

Required artifacts:

- `authority.json`
- `maxc_shrink_ab.json`
- `e49152_critical_path.json`
- `opaque_append_fixed_tile.json`
- `runtime_kv_diagnostic_decision.json`
- `isa_tooling_inventory.json`
- `isa_primitive_audit_tool_result.json`
- `fallback_small_ops_decision.json`
- `next_course_decision.json`

Required docs:

- `docs/post-default-runtime-kv-diagnostic-result-20260623.md`
- if diagnostic passes:
  - `docs/runtime-kv-implementation-scope-post-owned-attention-20260623.md`
- if diagnostic fails:
  - `docs/runtime-kv-core-runtime-blocker-result-20260623.md`
  - `docs/small-ops-activation-fusion-scope-20260623.md`
- always:
  - update `docs/README.md`
  - update `structure/Development/session-handoff.md`

## Phase 0 — Authority Lock

Record:

- HEAD commit;
- git status;
- GPU/arch;
- ROCm/HIP toolchain availability;
- model path;
- default flags;
- explicit env flags used for Q4K warp if default-off;
- whether owned attention is default-on and firing;
- whether native fp16 cache is active;
- current candidate registry state;
- current post-default audit artifact hashes/paths.

Artifact:

- `bench/qk-post-default-runtime-kv-course/authority.json`

Verdicts:

- `AUTHORITY_LOCKED`
- `AUTHORITY_INCOMPLETE_STOP`

Stop if default/current route state is unclear.

## Phase 1 — Confirm Baseline Before Diagnostics

Run a short baseline decode confirmation under the current post-default configuration.

Required:

- ctx `1024` and `4096` at minimum;
- owned attention route fires;
- Q4K warp state recorded;
- native fp16 cache state recorded;
- token stream sane/deterministic;
- no accidental fallback to gqa;
- `E_49152` / full-MAXC materialization kernel presence checked.

Artifact fields:

- tok/s;
- ms/token;
- route nodes;
- top kernels;
- `E_49152` count/time if visible;
- tokens sample.

Append this to:

- `authority.json`

Verdicts:

- `BASELINE_CONFIRMED`
- `BASELINE_ROUTE_NOT_FIRING_STOP`
- `BASELINE_CORRECTNESS_FAIL_STOP`

## Phase 2 — MAXC-Shrink A/B Under Owned Attention + Q4K

Purpose:

Determine whether the full-MAXC materialization is still on the W==D critical path after owned attention + FO2.

Run a controlled A/B that changes only the effective max-context/cache materialization size where possible.

Requirements:

- current owned attention default route active;
- Q4K warp state explicit;
- compare at least two MAXC/materialization sizes;
- use same prompt/token path;
- token correctness / deterministic token sample;
- `.item()` inside timing window;
- repeated passes or tight spread;
- record top kernel changes, especially `E_49152` or equivalent materialization kernels;
- record whether attention/GEMV kernels are unchanged.

Preferred contexts:

- ctx1024;
- ctx2048 or ctx4096 if cheap.

Required table:

| config | MAXC/effective cache size | ctx | tok/s | ms/token | E_49152 ms | tokens match | delta vs baseline |
|---|---:|---:|---:|---:|---:|---|---:|

Artifact:

- `bench/qk-post-default-runtime-kv-course/maxc_shrink_ab.json`

Verdicts:

- `MAXC_SHRINK_TRANSFERS`
- `MAXC_SHRINK_NO_TRANSFER`
- `MAXC_SHRINK_INCONCLUSIVE`
- `MAXC_SHRINK_INVALID_CORRECTNESS`

Interpretation:

- If shrinking materialization reduces W==D by approximately the measured copy time, Runtime-KV remains high-value.
- If shrinking materialization does not transfer, classify the copy as overlapped/off-critical and deprioritize Runtime-KV.

## Phase 3 — E_49152 Critical-Path Attribution

Purpose:

Confirm whether the materialization kernel is a true wall-clock bottleneck or an overlapped GPU-busy artifact.

Measure:

- `E_49152` wall/GPU-busy contribution;
- launch order around KV store/attention read;
- whether dependent attention waits on it;
- whether removing/reducing it changes subsequent owned tile inputs;
- whether it appears once per layer/token or in a subset;
- whether it is identical under gqa and owned route.

Possible methods:

- existing time-tax tooling;
- kernel trace ordering;
- controlled MAXC shrink from Phase 2;
- targeted DEBUG/kernel-name capture;
- bounded graph inspection.

Required table:

| kernel/name | role | count/token | us/token | dependency evidence | critical-path verdict |
|---|---|---:|---:|---|---|

Artifact:

- `bench/qk-post-default-runtime-kv-course/e49152_critical_path.json`

Verdicts:

- `E49152_ON_CRITICAL_PATH`
- `E49152_OVERLAPPED`
- `E49152_NOT_PRESENT`
- `E49152_ATTRIBUTION_INCONCLUSIVE`

## Phase 4 — Re-Test Opaque Append With Fixed fp16 Owned Tile

Purpose:

Prior Runtime-KV experiments were contaminated by the owned tile fp32/fp16 dtype bug. Re-test the opaque append path
now that:

- owned tile reads real cache correctly;
- fp16 route cache exists;
- native fp16 cache removes the cast path;
- GraphRunner arg-patching was previously proven correct.

Do **not** implement a production route here. Build/probe only as needed.

Required probes:

1. **Standalone/microbench:**
   - opaque append into fp16 cache;
   - owned tile reads persistent cache;
   - changing `start_pos`;
   - multi-step append/read;
   - real nonzero K/V;
   - numpy reference or gqa comparison.

2. **Model-local diagnostic:**
   - no full production route;
   - verify append inputs finite;
   - verify written cache finite;
   - verify owned tile input finite;
   - verify token correctness for at least several decode steps;
   - verify persistence across decode steps;
   - verify no `E_49152` if attempting copy-free path.

3. **Failure isolation if it fails:**
   - dtype mismatch;
   - append input NaN;
   - append output NaN;
   - tile read NaN;
   - persistence lost;
   - graph lifecycle/purity wall;
   - alias/dependency wall.

Required table:

| probe | correctness | persistence | E_49152 removed | route/tile finite | verdict | blocker |
|---|---|---|---|---|---|---|

Artifact:

- `bench/qk-post-default-runtime-kv-course/opaque_append_fixed_tile.json`

Verdicts:

- `OPAQUE_APPEND_FIXED_TILE_PASS`
- `OPAQUE_APPEND_CORRECTNESS_FAIL`
- `OPAQUE_APPEND_PERSISTENCE_FAIL`
- `OPAQUE_APPEND_RUNTIME_GRAPH_BLOCKED`
- `OPAQUE_APPEND_INCONCLUSIVE`

## Phase 5 — Runtime-KV Diagnostic Decision

Combine Phases 2-4.

Required decision table:

| question | answer | evidence |
|---|---|---|
| Does MAXC/materialization shrink transfer to W==D? | | |
| Is `E_49152` on the critical path? | | |
| Does opaque append work with the fixed fp16 owned tile? | | |
| Does it preserve cross-token persistence? | | |
| Does it remove the materialization kernel? | | |
| Is the remaining blocker model-route code, or core TinyJit/HCQ lifecycle? | | |
| Is expected W==D >=5% if solved? | | |

Artifact:

- `bench/qk-post-default-runtime-kv-course/runtime_kv_diagnostic_decision.json`

Allowed verdicts:

- `RUNTIME_KV_DIAGNOSTIC_PASS_SCOPE_IMPLEMENTATION`
- `RUNTIME_KV_CORE_RUNTIME_BLOCKED`
- `RUNTIME_KV_DEFER_OVERLAPPED`
- `RUNTIME_KV_INCONCLUSIVE_NEEDS_NARROWER_PROBE`

Decision rules:

- If MAXC shrink transfers and opaque append works with fixed tile, write the implementation scope.
- If MAXC shrink transfers but opaque append fails at graph/persistence, classify as core runtime blocker.
- If MAXC shrink does not transfer, defer Runtime-KV and pivot.
- If correctness is invalid, stop and do not rank Runtime-KV as active implementation.

## Phase 6A — If Runtime-KV Diagnostic Passes: Scope Implementation

Only run this phase if verdict is:

- `RUNTIME_KV_DIAGNOSTIC_PASS_SCOPE_IMPLEMENTATION`

Do not implement yet unless explicitly authorized separately.

Write:

- `docs/runtime-kv-implementation-scope-post-owned-attention-20260623.md`

Required scope sections:

1. Mission.
2. Exact primitive:
   - runtime-managed KV cache lifecycle;
   - opaque append;
   - owned attention read;
   - no full-MAXC materialization.
3. Required route/state guards.
4. Correctness gates:
   - token authority;
   - multi-step decode;
   - multiple prompts;
   - finite K/V;
   - fallback.
5. W==D gates:
   - ctx512/1024/2048/4096;
   - no regression;
   - expected >=5% if copy tax transfers.
6. Runtime graph requirements:
   - persistence;
   - dependency ordering;
   - start_pos patching;
   - no hidden fallback.
7. Failure modes and stop rules.
8. Artifacts.
9. Default policy.

Verdict:

- `RUNTIME_KV_IMPLEMENTATION_SCOPE_READY`

## Phase 6B — If Runtime-KV Fails: Classify Core Runtime Work

Only run this phase if verdict is:

- `RUNTIME_KV_CORE_RUNTIME_BLOCKED`

Write:

- `docs/runtime-kv-core-runtime-blocker-result-20260623.md`

Required classification:

| layer | finding |
|---|---|
| algorithm | KV append/read semantics are valid |
| work decomposition | not the blocker |
| memory movement | materialization tax is real |
| ISA/codegen | not primary unless append kernel fails locally |
| runtime/graph lifecycle | blocker |
| W==D | potential transfer, blocked by lifecycle |

Required final label:

- `RUNTIME_GRAPH_LIFECYCLE_GAP`

Required stop rule:

- Do not attempt broad TinyJit/HCQ alias/persistence redesign in this task.
- Write a separate core runtime capability scope only if owner explicitly asks.

## Phase 7 — Small-Ops / Activation Fusion Fallback Scope

Run this phase if Runtime-KV is blocked, overlapped, or deferred.

Purpose:

Prepare the next practical lane without implementing it.

Use the post-default audit finding:

- small ops / norm / q8 / activation-like residuals remain around `~1-1.2 ms` class;
- some are unfused small kernels where llama fuses;
- prior bucket labels were often wrong, so this must start with rendered-source evidence.

Write:

- `docs/small-ops-activation-fusion-scope-20260623.md`

Required contents:

1. Mission.
2. Required corrected bucket map.
3. Candidate kernels by name/source fingerprint.
4. Fusion opportunities:
   - q8 quant adjacent kernels;
   - genuine RMSNorm/RoPE if any;
   - activation/copy remnants;
   - sampling/logits if measurable.
5. Lifecycle classification.
6. First bounded gate:
   - prove one fusion removes a measured kernel group;
   - prove token correctness;
   - prove >=1-2% W==D before expanding.
7. Stop rules:
   - no broad codegen rewrite first;
   - no stale labels;
   - no local-only success.

Verdict:

- `SMALL_OPS_FUSION_SCOPE_READY`
- or `SMALL_OPS_FUSION_NOT_JUSTIFIED`

Artifact:

- `bench/qk-post-default-runtime-kv-course/fallback_small_ops_decision.json`

## Phase 8 — Build/Formalize AMDGCN ISA Primitive Audit Tool

Purpose:

Turn the successful manual ISA inspection from the post-default audit into reusable tooling.

Build or scope a tool:

- `extra/qk_amdgpu_isa_primitive_audit.py`

Minimum behavior:

- accept one or more code objects (`.co`, `.hsaco`) or discover cached code objects from the owned routes;
- run available disassembly/metadata tools:
  - `llvm-objdump`;
  - `roc-objdump`;
  - `amdllvm-objdump`;
  - `readelf`;
  - `clang-offload-bundler` if needed;
- parse or record:
  - symbol;
  - gfx target;
  - group segment / LDS bytes;
  - private segment / scratch;
  - VGPR/SGPR if available;
  - kernarg size/layout if available;
  - instruction flags:
    - `has_v_dot2`;
    - `has_lds`;
    - `has_cross_lane`;
    - `has_vector_global_load`;
    - `has_spill`;
  - instruction counts where easy;
  - tooling gaps.

Initial audited kernels:

- owned attention tile;
- owned attention combine;
- Q4K GEMV warp if code object available;
- one residual tinygrad-generated kernel if available.

Artifacts:

- `bench/qk-post-default-runtime-kv-course/isa_tooling_inventory.json`
- `bench/qk-post-default-runtime-kv-course/isa_primitive_audit_tool_result.json`

Verdicts:

- `ISA_PRIMITIVE_AUDIT_TOOL_READY`
- `ISA_PRIMITIVE_AUDIT_TOOL_PARTIAL`
- `ISA_PRIMITIVE_AUDIT_TOOL_BLOCKED`

Do not overbuild. A partial tool that reliably emits evidence for owned `.co` kernels is useful.

## Phase 9 — Final Next-Course Decision

Write:

- `bench/qk-post-default-runtime-kv-course/next_course_decision.json`
- `docs/post-default-runtime-kv-diagnostic-result-20260623.md`

Required result doc sections:

1. Verdict.
2. Authority/config.
3. Baseline confirmation.
4. MAXC-shrink A/B.
5. `E_49152` critical-path attribution.
6. Opaque append with fixed owned tile.
7. Runtime-KV decision.
8. If pass: implementation scope summary.
9. If fail: core runtime blocker classification.
10. Small-ops fallback decision.
11. ISA audit tool result.
12. Updated next primitive ranking.
13. Artifacts and commands.
14. Files changed.
15. Git status.

Allowed final verdicts:

- `RUNTIME_KV_IMPLEMENTATION_SCOPE_READY`
- `RUNTIME_KV_CORE_RUNTIME_BLOCKED_SMALL_OPS_NEXT`
- `RUNTIME_KV_DEFER_SMALL_OPS_NEXT`
- `RUNTIME_KV_INCONCLUSIVE_REPEAT_DIAGNOSTIC`
- `ISA_TOOL_READY_RUNTIME_KV_PENDING`

Update:

- `docs/README.md`
- `structure/Development/session-handoff.md`

Do not rewrite historical docs. Add superseding notes.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read and execute:

```text
docs/post-default-runtime-kv-and-isa-next-course-scope-20260623.md
```

This is the next course after `POST_DEFAULT_AUDIT_COMPLETE`.

The post-default audit found:

- owned AMDGCN attention + FO2 moved tinygrad to ~88-89% of llama;
- FFN weight-GEMV is at/near parity;
- attention is near parity;
- the leading residual is the runtime/cache lifecycle, especially `E_49152` / full-MAXC KV materialization at about `~1.5 ms/token`;
- runtime-KV must be re-diagnosed now that the owned tile fp16 dtype bug is fixed;
- ISA inspection works and should be formalized into reusable tooling.

Execute the scope in order:

1. Authority lock.
2. Baseline confirmation.
3. MAXC-shrink A/B under owned attention + Q4K.
4. `E_49152` critical-path attribution.
5. Re-test opaque append with the fixed fp16 owned tile.
6. Runtime-KV diagnostic decision.
7. If Runtime-KV passes, write the implementation scope only.
8. If Runtime-KV fails, classify the core runtime blocker and write the small-ops/activation fusion fallback scope.
9. Build/formalize the AMDGCN ISA primitive audit tool or a partial first version.
10. Write the final result doc and update README/session handoff.

Hard boundaries:

- Do not reopen attention.
- Do not reopen GEMV.
- Do not flip defaults.
- Do not implement production Runtime-KV unless explicitly authorized after the diagnostic.
- Do not do 14B/32B.
- Do not use position-written proxy as correctness; token correctness is authority.
- Do not make local-only performance claims; W==D is authority.
- Do not make ISA claims without disassembly/metadata evidence or explicit tooling-limited notes.

Write required artifacts under:

```text
bench/qk-post-default-runtime-kv-course/
```

Write required docs:

```text
docs/post-default-runtime-kv-diagnostic-result-20260623.md
docs/runtime-kv-implementation-scope-post-owned-attention-20260623.md   # only if diagnostic passes
docs/runtime-kv-core-runtime-blocker-result-20260623.md                 # only if blocked
docs/small-ops-activation-fusion-scope-20260623.md                      # if Runtime-KV blocked/deferred
```

Final response must include:

- final verdict;
- Runtime-KV diagnostic decision;
- MAXC-shrink result;
- `E_49152` critical-path verdict;
- opaque append fixed-tile verdict;
- ISA audit tool verdict;
- next primitive recommendation;
- commands run;
- artifacts written;
- files changed;
- git status.
