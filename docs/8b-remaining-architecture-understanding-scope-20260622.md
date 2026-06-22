# 8B Remaining Architecture Understanding — Exhaustive Scope / Claude Prompt (2026-06-22)

## Mission

The bounded 8B primitive ladder is exhausted. The remaining work is **understanding**, not another opportunistic
kernel patch.

This scope exhaustively audits the two remaining architectural lanes:

1. **Runtime-managed / two-graph KV cache**
2. **Codegen-level fused attention**

The goal is to produce a decision-grade explanation of what each would require, what it could recover, why it is
not a bounded primitive, and what exact first gate would justify funding it later.

This is audit/scope work only unless explicitly redirected. Do not implement either architecture in this pass.

## Current Project State

Latest committed state:

- `22ffdaaa6 [docs] KV-cache stateful-JIT capability: scope + result (KV_RUNTIME_MANAGED_CACHE_REQUIRED)`
- `7dbc08089 [test] KV-cache stateful-JIT capability probe: opaque RDNA3 append kernel + microprobe`
- `273a7dc12 [docs] KV-cache copy elimination: implementation scope + result (KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED)`
- `14962ef7a [test] q4k_gemv_warp promotion hardening + same-lever expansion -> Q4K_GEMV_WARP_READY_FOR_OWNER_DEFAULT_DECISION`

Current 8B lane status:

| Lane | Final bounded status | Why |
|---|---|---|
| Weight GEMV | **Closed / won** | `Q4K_GEMV_WARP` is lossless, W==D pass, default-eligible. |
| FFN activation | **Closed** | `silu` is fused into gate/up GEMV; old "activation" bucket was KV copy. |
| Norm/Rope | **Closed** | genuine RMSNorm/qk-norm near parity or faster than llama. |
| Attention bounded Route B | **Closed for bounded work** | B5 combine 2.4x local win did not transfer; Route B saturates ~+5.7%@4096. |
| KV-cache copy local fix | **JIT-blocked** | `.assign()` and slice-`.after()` both fail under same-graph pure-function semantics. |
| KV opaque append write | **Microprobe pass** | custom append node writes symbolic-offset K/V and replays with changing `start_pos`. |
| Same-graph KV read after write | **Blocked** | attention reduce over mutated buffer reintroduces read-after-write / symbolic-size wall. |

Therefore:

**No bounded 8B model/primitive lever remains.**

The only remaining levers are architectural:

- remove KV lifecycle tax by taking KV state out of pure tensor value semantics;
- close attention tail by building a codegen/runtime capability that can express llama-class fused flash.

## Required Reading

Read these before making claims:

### Consolidated 8B state

1. `docs/decode-gap-audit-consolidated-20260622.md`
2. `docs/8b-exhaustion-next-implementation-decision-20260622.md`
3. `docs/tinygrad-vs-llama-decode-time-tax-diff-result-20260622.md`
4. `structure/Development/session-handoff.md`
5. `structure/Development/performance-primitive-research-principles.md`

### GEMV closure

6. `docs/decode-ffn-gemv-warp-result-20260622.md`
7. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`

### KV-cache closure and stateful-JIT probe

8. `docs/kv-cache-copy-elimination-result-20260622.md`
9. `docs/kv-cache-stateful-jit-capability-result-20260622.md`
10. `extra/qk_kv_cache_state_token.py`
11. `extra/qk_kv_append_microprobe.py`

### Attention closure / codegen wall

12. `docs/attention-tail-after-b5-audit-result-20260622.md`
13. `docs/b4-cheaper-combine-result-20260622.md`
14. `docs/decode-attention-route-a-route-b-full-execution-scope-20260621.md`
15. `docs/decode-attention-primitive-spec-and-route-scope-20260621.md`
16. `docs/fused-flash-concrete-gate-result-20260621.md`
17. `docs/matmul-pv-diagnostic-result-20260621.md`
18. `docs/native-fused-flash-linearizer-scope-20260621.md`
19. `docs/post-matmul-pv-decode-strategic-scope-20260621.md`
20. `docs/llama-flash-attn-tile-oracle-result-20260621.md`
21. `docs/low-level-decode-attn-attribution-result-20260621.md`

### External reference anchors

Use only for framing, not as implementation authority:

- tinygrad docs: `https://docs.tinygrad.org/`
- tinygrad repo: `https://github.com/tinygrad/tinygrad`
- TensorRT-LLM KV cache docs: `https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html`
- NVIDIA KV cache reuse blog: `https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/`
- vLLM KV cache docs: `https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/core/kv_cache_utils.html`
- OpenXLA KV-cache discussion: `https://groups.google.com/a/openxla.org/g/openxla-discuss/c/_PmzjktC0_M`
- vAttention paper: `https://arxiv.org/html/2405.04437v2`

## Work Product

Write one decision-grade result doc:

- `docs/8b-remaining-architecture-understanding-result-20260622.md`

Write supporting artifacts:

- `bench/qk-8b-remaining-architecture-understanding/runtime_kv.json`
- `bench/qk-8b-remaining-architecture-understanding/fused_attention_codegen.json`
- `bench/qk-8b-remaining-architecture-understanding/decision.json`

No default changes. No model/kernel implementation. No 14B/32B work.

## Lane 1 — Runtime-Managed / Two-Graph KV Cache

### Question

What exactly would it take to remove the full-MAXC KV-copy tax without violating tinygrad's pure TinyJit graph model?

### Known facts

- Copy tax is real: `E_49152`, about `1.4 ms/token` at MAXC 4608.
- MAXC shrink transfers to wall: about `+1.5 ms` / `+8 tok/s`.
- Local `.assign()` failed with scheduler read-after-write hazard.
- Slice-scoped `.after()` failed with symbolic-size / alias resolution.
- Opaque custom append write passed: symbolic-offset, capture/replay, changing `start_pos`.
- In-model failed because same-graph attention read does not see/persist the opaque write unless the cache UOp is
  repointed; repointing reintroduces the read-after-write reduce failure.

### Required analysis

Produce a table:

| design | what changes | why it might work | exact blocker/risk | estimated recoverable ms | first gate | verdict |
|---|---|---|---|---:|---|---|
| A. separate append graph + attention graph | | | | | | |
| B. runtime KV object / pointer+length | | | | | | |
| C. state token in HCQ graph only | | | | | | |
| D. alias-aware same-graph mutation | | | | | | |
| E. paged/block-table cache | | | | | | |

Required points to answer:

1. **Two-graph decode feasibility**
   - Can append be realized as its own TinyJit/HCQ graph before the attention graph?
   - Does the extra graph/launch overhead eat the `~1.4 ms` savings?
   - Can the append graph be captured once and replayed with changing `start_pos`?
   - Can the attention graph read the persistent cache buffer without `.after()` full-copy?
   - Can the two graphs preserve ordering without host sync?

2. **Runtime KV object shape**
   - What object owns the cache buffer?
   - How is buffer lifetime managed across prompts, batch, reset, and repeated generation?
   - How are `start_pos`, `max_context`, dtype, Hkv, Hd, and batch encoded?
   - Does this remain compatible with current `model.generate` and server paths?

3. **Pointer/length attention interface**
   - Does existing `gqa_coop_vec` / B4 owned attention accept a pointer+length style cache read?
   - If not, what minimum adapter is needed?
   - Can native tinygrad attention read from persistent buffer without functional `.after()`?
   - Does this imply external attention graph nodes are required?

4. **Correctness hazards**
   - stale KV between prompts;
   - repeated generation calls;
   - JIT replay with different `start_pos`;
   - batch >1;
   - prefill-to-decode handoff;
   - speculative/multi-token decode;
   - server concurrency.

5. **Performance model**
   - Current post-warp token_ms at ctx512/1024/2048/4096.
   - Copy tax by MAXC.
   - Projected tok/s after removing copy.
   - Extra cost of append graph / ordering.
   - Break-even graph overhead.

6. **Implementation blast radius**
   - Files likely touched.
   - Whether core tinygrad must change.
   - Whether this is fork-local or upstreamable.
   - Test matrix required.

### Required verdicts for Lane 1

Choose exactly one:

- `RUNTIME_KV_TWO_GRAPH_BOUNDED_GATE_READY`
- `RUNTIME_KV_RUNTIME_OBJECT_SCOPE_READY`
- `RUNTIME_KV_REQUIRES_ATTENTION_INTERFACE_REWRITE`
- `RUNTIME_KV_UNBOUNDED_DEFER`
- `RUNTIME_KV_NOT_WORTH_FUNDING`

### Lane 1 first-gate scope if funded

If the verdict is not defer, write the first gate in the result doc:

- microbenchmark two-graph append+read ordering;
- no full-MAXC copy;
- append graph overhead < `0.25 ms/token`;
- byte-identical 64-token generation;
- W==D `>= +5% @ctx1024`;
- no stale cache leakage across two prompts in one process;
- default-off flag only.

## Lane 2 — Codegen-Level Fused Attention

### Question

What exact codegen/runtime capability would be needed to close the remaining attention tail, given that bounded
Route B/B5 attention is exhausted?

### Known facts

- Attention gap is real and ctx-growing.
- B4/B5 owned AMDGCN graph-node route is correct and fires in-model, but W==D saturates:
  - about `+0.23%@1024`;
  - about `+5.66-5.7%@4096`;
  - below `+7%@4096` promotion gate.
- Cheaper combine does not move W==D; combine overlaps.
- Existing `gqa_coop_vec` is the best tinygrad-native split path.
- Llama's tile has LDS + `v_dot2` + online softmax/PV in one tile.
- tinygrad current attention has scalar fp16 loads, no `v_dot2`, no LDS in the flash partial path.
- Prior concrete fused-flash gate failed because tiled-GEMM codegen and `.set/.after` online-softmax fusion are
  mutually exclusive in current tinygrad.
- Matmul-PV is blocked by symbolic split/layout and is dominated unless folded into full fused flash.

### Required analysis

Produce a table:

| route | target capability | why previous work failed | what would be different now | first gate | risk | verdict |
|---|---|---|---|---|---|---|
| A. native linearizer coupled multi-reduce | | | | | | |
| B. AMDGCN/HSACO owned fused tile as graph node | | | | | | |
| C. renderer/codegen primitive for LDS+v_dot2 flash | | | | | | |
| D. improve split attention further | | | | | | |
| E. do nothing / rest attention | | | | | | |

Required points to answer:

1. **What "fused attention" actually means here**
   - q·k score;
   - online max/den;
   - softmax probability;
   - PV accumulation;
   - GQA V reuse;
   - LDS-staged K/V;
   - `v_dot2_f32_f16`;
   - split-KV combine only if needed.

2. **Why Route B did not clear W==D**
   - separate tile+combine;
   - split-KV economics;
   - overlap/off-critical-path combine;
   - attention share / W==D saturation.

3. **Why current tinygrad codegen cannot express llama-class tile**
   - `.set/.after` register-state fusion vs tiled-GEMM path;
   - coupled reductions;
   - multiple accumulators;
   - dynamic/symbolic KV length;
   - LDS scheduling;
   - vector dot instruction selection;
   - layout constraints for PV.

4. **What capability would change the answer**
   - a raw fused AMDGCN graph-node with a single kernel;
   - or native linearizer support for coupled online-softmax+PV reductions with LDS;
   - or a renderer primitive/schedule template for flash decode.

5. **Performance model**
   - current attention gap at ctx512/1024/2048/4096;
   - llama tile oracle local advantage;
   - expected W==D if attention tail were cut by 25/50/75/100%;
   - why `+5.7%@4096` is current bounded ceiling;
   - what must be true to exceed it.

6. **Implementation blast radius**
   - native codegen files;
   - custom AMDGCN route files;
   - model route / graph-node integration;
   - evaluator/candidate registry;
   - correctness and dNLL/greedy gates.

### Required verdicts for Lane 2

Choose exactly one:

- `FUSED_ATTENTION_AMDGCN_SINGLE_TILE_GATE_READY`
- `FUSED_ATTENTION_LINEARIZER_CAPABILITY_SCOPE_READY`
- `FUSED_ATTENTION_RENDERER_TEMPLATE_SCOPE_READY`
- `FUSED_ATTENTION_UNBOUNDED_DEFER`
- `FUSED_ATTENTION_REST`

### Lane 2 first-gate scope if funded

If the verdict is not rest/defer, write the first gate:

- one concrete Qwen3-8B/gfx1100 ctx1024 fused-flash single-tile candidate;
- graph-node integrated, default-off;
- byte-correct vs `gqa_coop_vec` / numpy;
- local attention A/B `>=1.5x` vs `gqa_coop_vec`;
- W==D expected projection before in-model route;
- stop if local gate misses.

## Cross-Lane Decision

After Lane 1 and Lane 2, produce a ranked decision:

| rank | lane | expected gain | confidence | implementation size | risk | why now / why not |
|---:|---|---:|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |

Decision options:

- `FUND_RUNTIME_KV_FIRST`
- `FUND_FUSED_ATTENTION_FIRST`
- `FUND_NEITHER_PROMOTE_GEMV_AND_GENERALIZE`
- `FUND_BOTH_AS_SEPARATE_PROJECTS`

Ranking rule:

1. Prefer the lane with the clearest first gate and highest chance of W==D transfer.
2. Break ties in favor of lower blast radius.
3. Do not rank raw theoretical upside above measured transfer.
4. If both are unbounded, recommend promoting/hardening the GEMV win and moving to generalization.

## Non-Goals

- No source implementation unless explicitly redirected.
- No default changes.
- No 14B/32B.
- No reopening closed bounded lanes.
- No name-based bucket claims.
- No broad "optimize decode" recommendation without first-gate criteria.

## Required Result Doc Structure

`docs/8b-remaining-architecture-understanding-result-20260622.md` must include:

1. Executive summary.
2. Current 8B closure table.
3. Lane 1: runtime-managed KV cache analysis.
4. Lane 1: first gate if funded.
5. Lane 2: codegen-level fused attention analysis.
6. Lane 2: first gate if funded.
7. Cross-lane ranking.
8. Recommendation.
9. Required future docs if funded.
10. Artifacts and commands.
11. Working tree status.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/8b-remaining-architecture-understanding-scope-20260622.md` completely and execute it as an audit/scope
task. Do not implement either architecture. The user wants full understanding of the two remaining architectural
lanes:

1. runtime-managed / two-graph KV cache;
2. codegen-level fused attention.

Use the latest committed results as authority:

- `KV_RUNTIME_MANAGED_CACHE_REQUIRED`
- `KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`
- `ATTENTION_BOUNDED_LEVER_EXHAUSTED_NO_REOPEN`
- `B5_COMBINE_LOCAL_PASS_WD_FAIL`
- `Q4K_GEMV_WARP_READY_FOR_OWNER_DEFAULT_DECISION`

For each lane, produce a decision-grade table covering what would change, why prior bounded work failed, what the
first funded gate would be, expected gain, confidence, blast radius, and stop condition. Then rank the two lanes
against the alternative of promoting/hardening the GEMV win and moving to generalization.

Write:

- `docs/8b-remaining-architecture-understanding-result-20260622.md`
- `bench/qk-8b-remaining-architecture-understanding/runtime_kv.json`
- `bench/qk-8b-remaining-architecture-understanding/fused_attention_codegen.json`
- `bench/qk-8b-remaining-architecture-understanding/decision.json`

Do not change defaults. Do not move to 14B/32B. Do not reopen FFN activation, norm/rope, bounded attention, or
weight-GEMV. Report final verdict, artifacts, commands, and git status.
