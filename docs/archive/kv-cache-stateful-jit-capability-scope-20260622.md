# KV-Cache Stateful JIT Capability — Scope / Claude Prompt (2026-06-22)

## Mission

The bounded local implementation probe ended with:

**`KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`**

The full-`max_context` KV-cache copy is real and transferable, but the two local fixes failed:

- in-place `.assign()` hit a scheduler read-after-write hazard;
- slice-scoped `.after()` hit symbolic alias/size resolution failure.

This scope decides whether to fund a **bounded core/JIT capability** for stateful KV append in tinygrad decode.
It is not another attention, activation, norm/rope, or GEMV kernel task.

## External Research Summary

Online references shift the understanding in one important direction: the failure is not surprising. Efficient
LLM decode treats KV cache as **runtime-managed state**, not as an ordinary pure tensor value that is fully
returned every step.

Relevant references:

- tinygrad's public docs describe tensors as lazy and TinyJit as a pure-function replay mechanism. This matches
  the local failure: the current decode graph wants functional value semantics, so the full-buffer `.after()`
  becomes the safe but expensive dependency form.  
  Sources: `https://docs.tinygrad.org/` and `https://github.com/tinygrad/tinygrad`.

- The OpenXLA StableHLO KV-cache discussion explicitly frames KV cache as a difficult state-management problem:
  caches are often runtime-managed, not simply returned from a pure function; optimized implementations may use
  custom ops or compiler/runtime support so attention operates on mutable / paged KV state without inducing costly
  dependency chains.  
  Source: `https://groups.google.com/a/openxla.org/g/openxla-discuss/c/_PmzjktC0_M`.

- TensorRT-LLM treats KV cache as a system with block pools, reuse, eviction, and GQA-aware pools. It is not a
  normal tensor copy in the model graph.  
  Sources: `https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html` and
  `https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/`.

- vLLM's cache manager / PagedAttention split cache storage into block tables and runtime-managed blocks; the
  attention kernel follows the table instead of assuming a pure contiguous tensor value is rebuilt every step.  
  Sources: `https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/core/kv_cache_utils.html` and
  `https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/`.

- vAttention argues for retaining virtual contiguity while adding dynamic physical allocation; the key lesson for
  this project is that KV-cache management is a runtime/memory capability, and preserving contiguous attention
  kernels can be preferable to paged-kernel rewrites.  
  Source: `https://arxiv.org/html/2405.04437v2`.

### How this shifts the local interpretation

The local failure is no longer best interpreted as "find a clever `.assign()` expression." It is better classified
as:

**tinygrad lacks an explicit stateful decode-buffer dependency primitive.**

The safe functional fallback copies/rematerializes the full buffer. Avoiding that requires one of:

1. an explicit state token / side-effect dependency in the JIT graph;
2. alias-aware in-place update semantics for a known buffer slice;
3. a runtime-managed KV cache object outside pure tensor value semantics;
4. a paged/block-table cache route, which requires attention kernels that understand page tables.

For this repo, option 1 is the most bounded first probe.

## Current Local State

Read these first:

1. `docs/kv-cache-copy-elimination-result-20260622.md`
2. `docs/kv-cache-copy-elimination-implementation-scope-20260622.md`
3. `docs/decode-gap-audit-consolidated-20260622.md`
4. `docs/8b-exhaustion-next-implementation-decision-20260622.md`
5. `docs/decode-ffn-gemv-warp-result-20260622.md`
6. `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
7. `structure/Development/performance-primitive-research-principles.md`
8. `structure/Development/session-handoff.md`

Also inspect:

- `extra/qk_kv_cache_copy_probe.py`
- `tinygrad/llm/model.py` around the KV-cache update in `_attention`
- `tinygrad/engine/jit.py`
- `tinygrad/engine/realize.py`
- `tinygrad/ops.py` for `Ops.AFTER`, `Ops.STORE`, `Ops.ASSIGN` / related UOps
- existing custom-kernel tests and Route B B4 graph-node code:
  - `extra/qk_owned_flash_decode_graph_node.py`
  - `test/amd/test_custom_kernel.py`

## Corrected 8B Lane Status

Do not reopen closed lanes while doing this capability decision.

| Lane | Status | Reason |
|---|---|---|
| Weight GEMV | Closed / won | `Q4K_GEMV_WARP` is lossless and W==D pass. |
| FFN activation | Closed | `silu` is fused into gate/up GEMV; old bucket was KV copy. |
| Norm/Rope | Closed | genuine norm/qk-norm is near parity or faster than llama. |
| Attention | Closed for bounded work | Route B/B5 saturates below W==D promotion gate; deeper route is codegen-level. |
| KV-cache copy | Real but JIT-blocked | local `.assign()` and slice-`.after()` probes failed; needs stateful graph capability. |

## Target Capability

Build or decisively refute a minimal stateful decode-buffer capability:

> A JIT-captured decode graph can append the current token's K/V into a persistent KV buffer, establish a dependency
> from subsequent attention reads to that append, and avoid materializing/copying the full MAXC buffer.

The target is not a general mutation system. It is a narrow decode-state primitive for:

- one known cache buffer,
- one slice append per layer per token,
- symbolic `start_pos`,
- T=1 decode,
- Qwen3-8B/gfx1100 first,
- default-off only.

## Candidate Designs To Evaluate

Evaluate these in order. Stop as soon as one is either proven viable enough for a W==D probe or refuted with a
specific blocker.

### Design A — Explicit State Token / Opaque Append Node

Create a tiny opaque graph node that:

- takes `cache_kv`, `k`, `v`, `start_pos`;
- writes only the target slice in-place;
- writes or returns a small dependency token/sentinel;
- makes attention's K/V read depend on that token without applying `.after()` to the full `cache_kv`.

Possible implementation shapes:

- tinygrad `Tensor.custom_kernel` / fully formed `Ops.PROGRAM` similar to Route B B4;
- a tiny AMD kernel that writes K/V and a one-element flag;
- a tiny tensor dependency injected into attention as `zero * flag` or another no-op barrier, if that is enough to
  enforce graph order.

Questions to answer:

- Can HCQGraph preserve the append node before attention?
- Can attention read the same cache buffer after the append without full-buffer copy?
- Does this avoid scheduler alias analysis entirely by making the append opaque?
- Is the barrier strong enough under TinyJit replay?

Pass criteria:

- byte-identical greedy;
- graph contains the append node before attention;
- `E_49152` full-copy disappears and no replacement full-copy appears;
- W==D `>= +5% @ctx1024` under post-warp baseline.

### Design B — Alias-Aware Slice Mutation In Scheduler

If Design A fails because tinygrad cannot express the ordering, inspect the smallest core scheduler change:

- recognize `cache[:, :, :, start_pos:start_pos+1, :].assign(...)` as a stateful slice append;
- allow same-buffer read after write when the read prefix includes the written slice;
- emit a dependency edge without producing a full value copy.

This is higher risk. Do not implement broadly until a microprobe proves the precise transform.

Required microprobes:

- non-symbolic slice append + read prefix;
- symbolic `start_pos` append + read prefix;
- repeated TinyJit replay with different `start_pos`;
- alias-negative test showing unrelated slices do not get incorrectly reordered.

Stop if this expands into broad alias analysis.

### Design C — Runtime-Managed KV Buffer Object

If A and B fail, scope but do not implement a runtime-managed KV object:

- cache lives outside pure Tensor value semantics;
- append is a runtime side effect;
- attention receives buffer pointer + length;
- similar in spirit to TensorRT-LLM / vLLM owning KV storage in the runtime.

This is likely a larger design and should become a separate project if chosen.

### Design D — Paged / Block-Table KV Cache

Only document. Do not build in this phase.

Paged cache could avoid fixed MAXC materialization and improve serving behavior, but it requires attention kernels
that follow block tables. That reopens attention kernel/layout work and is explicitly out of this bounded scope.

## Required Phases

### Phase 0 — Evidence Lock

Reproduce or inspect:

- `KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`;
- the exact `.assign()` scheduler `KeyError`;
- the exact slice-`.after()` symbolic size failure;
- current post-warp W==D baseline;
- copy kernel identity and cost.

Write:

- `docs/kv-cache-stateful-jit-capability-result-20260622.md` draft with baseline evidence.

### Phase 1 — Design A Microprobe

Build the smallest append-node proof outside the full model first.

Minimum microprobe:

- allocate a cache tensor;
- append one token K/V into a slice using an opaque custom node;
- read prefix including the appended token;
- verify bytes;
- run TinyJit capture/replay with changing `start_pos`;
- inspect graph node order.

Artifact:

- `bench/qk-kv-cache-stateful-jit/design_a_microprobe.json`

Verdicts:

- `DESIGN_A_MICROPROBE_PASS`
- `DESIGN_A_ORDERING_FAIL`
- `DESIGN_A_ALIAS_FAIL`
- `DESIGN_A_JIT_REPLAY_FAIL`
- `DESIGN_A_NOT_EXPRESSIBLE`

### Phase 2 — Design A In-Model Probe

Only if Phase 1 passes:

- add default-off flag `KV_CACHE_STATE_TOKEN=1`;
- integrate the opaque append node in `model.py`;
- preserve fallback to current functional `.after()` path;
- run correctness and kernel identity checks;
- run W==D at ctx512/1024/2048/4096.

Artifact:

- `bench/qk-kv-cache-stateful-jit/design_a_wd.json`

### Phase 3 — Design B Microprobe

Only if Design A fails for expressibility/order reasons:

- create minimal scheduler/alias probes;
- do not change full model;
- identify exact tinygrad files that would need changes;
- prove whether a small alias rule is possible.

Artifact:

- `bench/qk-kv-cache-stateful-jit/design_b_microprobe.json`

### Phase 4 — Decision

Write one of:

- `KV_STATE_TOKEN_WD_PASS`
- `KV_STATE_TOKEN_LOCAL_PASS_WD_FAIL`
- `KV_STATE_TOKEN_NOT_EXPRESSIBLE`
- `KV_ALIAS_RULE_BOUNDED_SCOPE_READY`
- `KV_ALIAS_RULE_UNBOUNDED_DEFER`
- `KV_RUNTIME_MANAGED_CACHE_REQUIRED`
- `NO_KV_STATEFUL_JIT_CAPABILITY_FUNDED`

## Gates

### Correctness

- greedy byte-identical vs post-warp baseline;
- at least 40 decode steps at ctx1024;
- ideally 64-token natural prompt;
- repeated generation calls do not leak stale KV state;
- capture/replay with changing `start_pos` works.

### Kernel Identity

- rendered source / AST fingerprint confirms no full-MAXC copy remains;
- no equivalent replacement full-buffer copy appears;
- append work is O(T) for T=1 or O(actual written slice), not O(MAXC).

### W==D

Baseline:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1
```

Candidate:

```bash
DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 KV_CACHE_STATE_TOKEN=1
```

Required ctx:

- 512
- 1024
- 2048
- 4096

Pass threshold:

- `>= +5% @ctx1024`;
- no regression elsewhere;
- tokens match;
- tight spread.

## Non-Goals

- No default change.
- No 14B/32B.
- No more attention tile/combine work.
- No FFN activation work.
- No norm/Rope work.
- No broad alias-analysis rewrite unless Design B microprobe proves a bounded rule.
- No paged attention implementation.

## Result Doc Requirements

Write:

- `docs/kv-cache-stateful-jit-capability-result-20260622.md`

Must include:

1. Online research summary with links.
2. Local blocker recap.
3. Corrected 8B lane table.
4. Design A result.
5. Design B result if reached.
6. W==D result if reached.
7. Final verdict.
8. Next funded scope, if any.
9. Files changed, artifacts, commands, and git status.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Read `docs/kv-cache-stateful-jit-capability-scope-20260622.md` completely and execute it.

The local KV-cache copy elimination probe ended `KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED`. Do not retry the same
`.assign()` or slice-`.after()` model patch as if it were new. The online research changes the framing: efficient
KV decode usually needs runtime-managed state, custom ops, scatter/update support, or a paged/block-table cache.
For tinygrad, the most bounded next probe is an explicit state-token / opaque append node in the JIT graph.

Start with Design A as a microprobe:

- opaque append node writes current K/V into the persistent cache slice;
- node emits or writes a tiny dependency token;
- attention read depends on the token without applying `.after()` to the full cache;
- TinyJit capture/replay works with changing `start_pos`.

Only if the microprobe passes should you integrate behind `KV_CACHE_STATE_TOKEN=1` and run W==D. If Design A fails,
evaluate whether a bounded alias-aware scheduler rule is scopeable; do not implement a broad alias rewrite.

Do not change defaults. Do not move to 14B/32B. Do not reopen attention, norm/rope, activation, or weight-GEMV.

Write `bench/qk-kv-cache-stateful-jit/*.json` artifacts and
`docs/kv-cache-stateful-jit-capability-result-20260622.md`. Report final verdict, commands, files changed, artifacts,
default status, and git status.
