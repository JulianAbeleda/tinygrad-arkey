# Fused Flash Single-Tile Owned AMDGCN — Implementation Scope / Claude Prompt (2026-06-22)

## Mission

Execute the next funded 8B architecture step:

**Build and gate a single fused owned AMDGCN decode-attention tile.**

This is the follow-on from:

- `docs/8b-remaining-architecture-understanding-result-20260622.md`
- verdict: `FUND_FUSED_ATTENTION_FIRST`
- lane verdict: `FUSED_ATTENTION_AMDGCN_SINGLE_TILE_GATE_READY`

The tile is the convergence substrate for the two remaining architecture lanes:

1. **Attention tail:** test the one untried bounded attention lever: one fused tile, no separate tile -> combine
   part/meta HBM round-trip.
2. **Runtime KV:** provide the opaque cache read required to remove the full-MAXC KV-copy tax in the immediate
   follow-on.

This is an implementation task, but it is gate-first. Do not broaden into native tinygrad linearizer/renderer work.
Do not change defaults.

## External Research Check

Online references support the project decision:

- FlashInfer exposes single-request decode and append/prefill attention APIs over KV cache, including precompiled
  JIT modules and paged KV-cache formats. This reinforces that decode attention is commonly treated as a specialized
  kernel/runtime interface, not a generic tensor expression.
  - `https://docs.flashinfer.ai/api/attention.html`
  - `https://flashinfer.ai/2024/02/02/introduce-flashinfer.html`

- vLLM's paged attention kernel is explicitly designed around its KV-cache storage format, with key/value caches
  stored in blocks and the attention kernel aware of that layout. This supports the "opaque attention read" framing:
  efficient KV cache removal needs the attention side to understand the cache interface.
  - `https://docs.vllm.ai/en/latest/design/paged_attention/`

- TensorRT-LLM's generation attention path treats KV cache as a first-class kernel input for generation, with
  per-layer KV caches. This aligns with the runtime-managed KV conclusion.
  - `https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html`
  - `https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html`

- Flash-Decoding describes the split-KV idea: load K/V in parallel, then rescale/combine partial results. Our B4/B5
  work implemented and optimized this split form, then measured that the separate combine overlaps and W==D saturates.
  The remaining untested lever is therefore not "cheaper combine" again, but a single fused tile that avoids the
  part/meta round-trip entirely.
  - `https://pytorch.org/blog/flash-decoding/`

- AMD's ROCm attention overview emphasizes FlashAttention's purpose: reduce memory movement between SRAM and HBM via
  tiling. This matches the target: LDS-staged K/V + vector dot + online softmax/PV in one kernel.
  - `https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/model-acceleration-libraries.html`

- vAttention argues that preserving contiguous attention-kernel compatibility while changing KV memory management can
  be preferable to page-table-aware kernel rewrites. This is relevant for the follow-on runtime-KV lane: first build a
  strong opaque contiguous attention reader, then use it to remove the functional full-cache copy.
  - `https://arxiv.org/html/2405.04437v2`

### What the online check changes

It does **not** reopen generic attention fusion or native linearizer work. It strengthens the decision that the next
bounded implementation should be an **owned specialized attention/KV interface**:

- a single opaque decode-attention graph node;
- native cache-buffer reads;
- explicit shape guards;
- default-off route;
- measured W==D gate.

## Required Reading Before Editing

Read these in order:

1. `docs/8b-remaining-architecture-understanding-result-20260622.md`
2. `docs/8b-remaining-architecture-understanding-scope-20260622.md`
3. `docs/kv-cache-stateful-jit-capability-result-20260622.md`
4. `docs/kv-cache-copy-elimination-result-20260622.md`
5. `docs/decode-gap-audit-consolidated-20260622.md`
6. `docs/attention-tail-after-b5-audit-result-20260622.md`
7. `docs/b4-cheaper-combine-result-20260622.md`
8. `docs/decode-attention-route-b-b3-owned-amdgcn-result-20260621.md`
9. `docs/decode-attention-route-b-b4-external-graph-node-result-20260621.md`
10. `docs/decode-attention-route-a-route-b-full-execution-scope-20260621.md`
11. `docs/fused-flash-concrete-gate-result-20260621.md`
12. `docs/matmul-pv-diagnostic-result-20260621.md`
13. `docs/llama-flash-attn-tile-oracle-result-20260621.md`
14. `docs/low-level-decode-attn-attribution-result-20260621.md`
15. `docs/decode-ffn-gemv-warp-result-20260622.md`
16. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
17. `structure/Development/performance-primitive-research-principles.md`
18. `structure/Development/session-handoff.md`

Required code to inspect:

- `extra/qk_owned_flash_decode.hip`
- `extra/qk_owned_flash_decode_graph_node.py`
- `extra/qk_b4_decode_eval.py`
- `extra/qk_b4_combine_ab.py`
- `extra/qk_kv_cache_state_token.py`
- `extra/qk_kv_append_microprobe.py`
- `tinygrad/llm/model.py` attention route
- `bench/qk-decode-eval/candidates.json`
- `bench/qk-decode-eval/binding_templates.json`

## Current Local Baseline

Post-`Q4K_GEMV_WARP` baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Known numbers from the decision artifact:

| ctx | post-warp tok/s | token_ms | projected no-KV-copy tok/s | projected no-copy+attention-tail tok/s |
|---:|---:|---:|---:|---:|
| 512 | 76.1 | 13.146 | 85.3 | 97.6 |
| 1024 | 74.0 | 13.509 | 82.7 | 95.7 |
| 2048 | 71.0 | 14.092 | 78.9 | 93.6 |
| 4096 | 67.0 | 14.927 | 74.0 | 91.5 |

Current B4/B5 attention route:

- default-off;
- two graph nodes: `owned_flash_tile_gqa` -> `owned_flash_combine`;
- correct and byte-identical;
- W==D saturates around `+5.66-5.7% @ctx4096`;
- below the `+7% @ctx4096` promotion gate;
- cheaper combine does not move W==D materially.

Current KV state:

- full-MAXC copy is real, about `1.4 ms/token`;
- local `.assign()` and slice-`.after()` are JIT-blocked;
- opaque append write works as a microprobe;
- same-graph functional attention read is the wall;
- a single opaque attention read is the missing substrate.

## Target Primitive

Build one owned AMDGCN graph node:

```text
owned_flash_single_tile_gqa
```

Functional target:

- T=1 decode;
- Qwen3-8B Qwen shape first: `B=1`, `Hq=32`, `Hkv=8`, `G=4`, `Hd=128`;
- native tinygrad KV layout `[Hkv, MAXC, Hd]`;
- fp16 Q/K/V input;
- fp32 accumulation;
- one output tensor `[Hq, Hd]` fp32;
- no `part` tensor;
- no `meta` tensor;
- no separate combine node;
- online softmax and PV finalized inside the tile.

Performance target:

- local attention A/B `>=1.5x` vs `gqa_coop_vec` at ctx1024;
- correct vs numpy/reference with `rel_rmse <= 1e-3` or stricter if existing B3/B4 tolerances are tighter;
- W==D projection before model route;
- if model route is attempted, W==D gate:
  - `>= +5% @ctx1024`;
  - `>= +7% @ctx4096`;
  - no ctx512 regression;
  - greedy byte-identical.

Architecture target:

- one graph node, not tile+combine;
- native cache read;
- eligible to become the opaque attention read for runtime-KV follow-on.

## Important Design Constraint

Classic Flash-Decoding split-KV uses multiple split workgroups, each writes partial `(m, denom, acc)`, and a combine
kernel merges them. B4/B5 already optimized that shape and proved the combine overlaps. This task must therefore
explicitly decide how a "single tile" handles the split dimension.

Allowed approaches:

1. **One workgroup per `(query head, output tile)` over full KV**
   - simplest single-node shape;
   - may under-occupy or lose split parallelism;
   - likely slower at long context.

2. **One kernel containing both split partial and cross-split reduction**
   - multiple workgroups still cannot globally synchronize inside one ordinary kernel;
   - if implemented, it must use a legal single-kernel strategy, not assume grid-wide sync.

3. **Single node but internally persistent/cooperative**
   - only if legal on target runtime and no hidden host sync;
   - must be documented carefully.

4. **Admit impossibility of true folded combine without grid sync**
   - if this is the finding, classify honestly and redirect to runtime-KV opaque read using the existing two-node tile,
     or declare `SINGLE_TILE_GLOBAL_REDUCTION_BLOCKED`.

This is the central risk. Do not handwave it.

## Phase Plan

### Phase 0 — Baseline Lock

Before editing:

- confirm `git status --short`;
- confirm latest commits include `872d3eea4` and `3fb5dd982`;
- run or inspect:
  - B4/B5 latest artifacts;
  - post-warp W==D baseline;
  - B3 owned tile local result;
  - KV stateful append microprobe result.

Write a pre-edit baseline section in the result doc with:

- commit hash;
- GPU/arch;
- model path;
- env flags;
- baseline tok/s;
- current B4/B5 route W==D;
- known copy tax.

### Phase 1 — Feasibility Reconciliation

Before coding the full tile, answer:

| question | required answer |
|---|---|
| Can cross-split combine be folded into one ordinary GPU kernel without grid-wide sync? | yes/no with mechanism |
| If no, what single-node shape remains? | full-KV per head, persistent strategy, or stop |
| Does the remaining shape have enough parallelism at ctx1024/4096? | estimate workgroups/waves/CU occupancy |
| Does it preserve GQA V reuse and coalesced V loads? | yes/no |
| Can it still serve as opaque KV read for runtime-KV follow-on? | yes/no |

Allowed verdicts:

- `SINGLE_TILE_FEASIBILITY_PASS`
- `SINGLE_TILE_NEEDS_PERSISTENT_GRID_UNSUPPORTED`
- `SINGLE_TILE_GLOBAL_REDUCTION_BLOCKED`
- `SINGLE_TILE_PARALLELISM_INSUFFICIENT`
- `SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ`

If feasibility fails, stop and write result. Do not build a doomed kernel.

### Phase 2 — Local Kernel Prototype

If Phase 1 passes, add the minimal prototype.

Likely files:

- `extra/qk_owned_flash_decode.hip`
  - add `owned_flash_single_tile_gqa` or equivalent;
  - reuse B3/B5 helper code where possible;
  - preserve existing B4/B5 kernels.

- `extra/qk_owned_flash_decode_graph_node.py`
  - add compile/registry path for `single_tile`;
  - output one graph node;
  - no `part`/`meta` allocation for this variant;
  - keep current variants intact.

- New local A/B tool:
  - `extra/qk_fused_flash_single_tile_ab.py`

Artifact:

- `bench/qk-fused-flash-single-tile/local_ab.json`

Local gate:

- correctness vs numpy/reference;
- local attention A/B vs `gqa_coop_vec`;
- disassembly/resource notes:
  - `v_dot2` present;
  - LDS use present if intended;
  - VGPR count;
  - LDS bytes;
  - spills;
  - workgroups.

Stop if:

- correctness fails;
- local A/B < `1.05x` vs `gqa_coop_vec`;
- no `v_dot2`/LDS despite claiming them;
- workgroup count is clearly insufficient.

Promising local result threshold:

- `>=1.5x` vs `gqa_coop_vec` @ctx1024.

Borderline:

- `1.05x-1.5x`: write projection; only continue to W==D if it plausibly unlocks KV-copy follow-on.

### Phase 3 — Graph-Node Integration

If local prototype passes:

- integrate as default-off graph-node route;
- suggested env:
  - `DECODE_ATTN_AMDGCN_SINGLE_TILE=1`
- keep existing `DECODE_ATTN_AMDGCN_TILE` two-node route untouched;
- fallback to `gqa_coop_vec` on unsupported shape/device/exception;
- strict guard:
  - AMD/gfx1100;
  - B=1;
  - T=1;
  - Hq=32/Hkv=8/Hd=128;
  - dtype/layout as validated.

Required graph identity:

- captured TinyJit graph contains exactly the single fused tile node for attention;
- no `owned_flash_combine`;
- no `part`/`meta` intermediate allocation;
- no unexpected full-MAXC copy introduced by the route.

Artifact:

- `bench/qk-fused-flash-single-tile/graph_route.json`

### Phase 4 — W==D Gate

Run W==D under post-warp baseline:

Baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 PYTHONPATH=. .venv/bin/python <harness>
```

Candidate:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 DECODE_ATTN_AMDGCN_SINGLE_TILE=1 PYTHONPATH=. .venv/bin/python <harness>
```

Required ctx:

- 512
- 1024
- 2048
- 4096

Required timing discipline:

- `.item()` inside timed window;
- repeated or in-process A/B if needed;
- report spread;
- tokens match.

Promotion gate:

- `>= +5% @ctx1024`;
- `>= +7% @ctx4096`;
- no ctx512 regression beyond noise;
- greedy byte-identical.

Interpretation:

- If W==D clears: `FUSED_FLASH_SINGLE_TILE_WD_PASS`.
- If local passes but W==D saturates `<= +5.7%@4096`: `FUSED_FLASH_SINGLE_TILE_STRUCTURAL_ATTENTION_REST`.
- If local fails: `FUSED_FLASH_SINGLE_TILE_LOCAL_FAIL`.
- If impossible due to global reduction/sync: `FUSED_FLASH_SINGLE_TILE_REDUCTION_BLOCKED`.

### Phase 5 — Runtime-KV Follow-On Decision

Even if attention does not clear W==D, decide whether the single tile is sufficient as an opaque cache read for the
runtime-KV follow-on.

Required table:

| requirement for runtime-KV opaque read | single tile status |
|---|---|
| reads persistent cache pointer directly | |
| does not require `assigned_kv` functional full-copy | |
| can be ordered after opaque append | |
| supports symbolic `start_pos` | |
| T=1 decode supported | |
| fallback safe | |

If yes, write the next scope stub:

- `docs/runtime-kv-opaque-read-followon-scope-20260623.md`

Do not implement the runtime-KV follow-on in this task unless explicitly asked.

## Result Doc

Write:

- `docs/fused-flash-single-tile-result-20260622.md`

Required sections:

1. Verdict.
2. Online research summary and why it supports the scope.
3. Baseline lock.
4. Feasibility reconciliation, especially cross-split combine / grid-sync question.
5. Kernel design.
6. Local A/B result.
7. Graph-node identity.
8. W==D result.
9. Runtime-KV follow-on decision.
10. Default / candidate registry decision.
11. Artifacts and commands.
12. Working tree status.

Allowed final verdicts:

- `FUSED_FLASH_SINGLE_TILE_WD_PASS`
- `FUSED_FLASH_SINGLE_TILE_LOCAL_PASS_WD_FAIL`
- `FUSED_FLASH_SINGLE_TILE_STRUCTURAL_ATTENTION_REST`
- `FUSED_FLASH_SINGLE_TILE_LOCAL_FAIL`
- `FUSED_FLASH_SINGLE_TILE_REDUCTION_BLOCKED`
- `FUSED_FLASH_SINGLE_TILE_SCOPE_REDUCED_TO_OPAQUE_READ`

## Candidate Registry

Only if graph-node route exists and correctness passes:

- add candidate:
  - `decode_attention_owned_amdgcn_single_tile`
- mark:
  - `default_on=false`;
  - `default_eligible=true` only if W==D and correctness gates pass;
  - otherwise `default_eligible=false`;
  - include shape/device guard metadata;
  - include relation to runtime-KV follow-on.

Do not register if Phase 1 feasibility fails before a route exists.

## Non-Goals

- No default change.
- No 14B/32B.
- No native tinygrad linearizer/renderer surgery.
- No page-table/paged attention.
- No runtime-KV copy removal implementation yet.
- No more combine-only tuning.
- No activation/norm/GEMV work.
- No claim from local A/B alone.

## Acceptance Checklist

- [ ] Required docs read.
- [ ] Online research summarized.
- [ ] Cross-split/global-reduction feasibility answered explicitly.
- [ ] Local kernel correctness measured if built.
- [ ] Local A/B vs `gqa_coop_vec` measured if built.
- [ ] Graph route identity verified if integrated.
- [ ] W==D ctx512/1024/2048/4096 measured if local gate passes.
- [ ] Runtime-KV opaque-read follow-on status decided.
- [ ] Result doc written.
- [ ] Artifacts written.
- [ ] Defaults unchanged.
- [ ] Working tree status reported.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/fused-flash-single-tile-implementation-scope-20260622.md` completely and execute it.

This is the next funded architecture task after `FUND_FUSED_ATTENTION_FIRST`. The goal is to build and gate a
**single fused owned AMDGCN decode-attention tile**:

- one graph node;
- no separate `owned_flash_combine`;
- no `part`/`meta` HBM round-trip;
- native tinygrad KV layout;
- default-off;
- strict Qwen3-8B/gfx1100 guard;
- correctness and W==D gated.

Start with Phase 1 feasibility. The central question is whether the cross-split combine can legally be folded into
one ordinary GPU kernel without grid-wide sync. If not, classify honestly before coding. If feasible, build the local
prototype, then graph-node route, then W==D. Use the post-`Q4K_GEMV_WARP` baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Hard gates:

- local correctness;
- local A/B vs `gqa_coop_vec`;
- graph identity: one fused node, no combine, no part/meta;
- W==D `>= +5%@ctx1024` and `>= +7%@ctx4096` for promotion;
- tokens byte-identical;
- defaults unchanged.

Even if attention W==D saturates, decide whether the single tile can serve as the opaque attention read for the
runtime-KV follow-on. Do not implement runtime-KV in this task.

Write:

- `docs/fused-flash-single-tile-result-20260622.md`
- `bench/qk-fused-flash-single-tile/local_ab.json`
- `bench/qk-fused-flash-single-tile/graph_route.json` if graph-integrated
- `bench/qk-fused-flash-single-tile/wd.json` if W==D reached

Do not change defaults. Do not move to 14B/32B. Do not edit native tinygrad codegen/renderer. Do not reopen
activation, norm/rope, GEMV, combine-only tuning, or paged attention. Report final verdict, commands, artifacts,
files changed, default status, and git status.
