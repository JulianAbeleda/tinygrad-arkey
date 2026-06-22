# KV-Cache Stateful JIT Capability — Result (2026-06-22)

## Verdict: **KV_RUNTIME_MANAGED_CACHE_REQUIRED**

The bounded opaque-append-node capability (**Design A**) is **PROVEN** for the in-place symbolic-offset KV
**write** (microprobe `DESIGN_A_MICROPROBE_PASS`), but it is **necessary-but-insufficient in-model**: the
full-`max_context` KV copy is the functional scheduler's *resolution* of the same-graph read-after-write
hazard between the KV append and the attention reduce, plus the symbolic-size requirement on the read side.
A bounded scheduler alias rule (**Design B**) is **unbounded** (symbolic-range alias analysis). The only
viable elimination is a **runtime-managed / two-graph KV cache** (**Design C**) — a separate, larger project.
**Recommendation: do not fund a bounded JIT capability; keep the canonical copy as the safe default.** No
default change; the gated in-model integration was reverted (`model.py` is byte-clean; default decode
byte-identical).

## 1. Online research summary
The local failure is not surprising: efficient LLM decode treats the KV cache as **runtime-managed state**,
not an ordinary pure tensor value returned every step.
- tinygrad docs (`https://docs.tinygrad.org/`, `https://github.com/tinygrad/tinygrad`): tensors are lazy,
  TinyJit is pure-function replay → the decode graph wants functional value semantics, so the full-buffer
  `.after()` is the safe-but-expensive dependency form.
- OpenXLA StableHLO KV-cache discussion (`https://groups.google.com/a/openxla.org/g/openxla-discuss/c/_PmzjktC0_M`):
  caches are runtime-managed, often via custom ops / compiler-runtime support so attention operates on
  mutable/paged KV without costly dependency chains.
- TensorRT-LLM (`https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html`,
  `https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/`):
  KV cache is a block-pool system, not a model-graph tensor copy.
- vLLM PagedAttention (`https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/core/kv_cache_utils.html`): block tables +
  runtime-managed blocks; attention follows the table, not a rebuilt contiguous value.
- vAttention (`https://arxiv.org/html/2405.04437v2`): KV management is a runtime/memory capability; preserving
  contiguous attention kernels can beat paged-kernel rewrites.

The shift: the local interpretation is no longer "find a clever `.assign()`/`.after()` expression" — **tinygrad
lacks an explicit stateful decode-buffer dependency primitive**, and the safe functional fallback copies the
full buffer. Confirmed by this probe.

## 2. Local blocker recap (from the prior phase)
`KV_CACHE_COPY_ELIMINATION_JIT_BLOCKED` (`docs/kv-cache-copy-elimination-result-20260622.md`): in-place
`.assign()` → scheduler read-after-write `KeyError`; slice-scoped `.after()` → symbolic alias/size
`ValueError: eval failed to be a single number`. The copy (`E_49152`, ~1.4 ms/token, O(MAXC)) is real and
transfers (+1.5 ms / +8 tok/s).

## 3. Corrected 8B lane status (all prior closures intact)
| Lane | Status |
|---|---|
| Weight GEMV | Closed/won (`Q4K_GEMV_WARP`, lossless W==D pass). |
| FFN activation | Closed — silu fused into gate/up GEMV; old bucket was the KV copy. |
| Norm/Rope | Closed — genuine norm at parity. |
| Attention | Closed for bounded work — B5 saturates below the W==D gate; deeper is codegen-level. |
| KV-cache copy | Real but core-JIT-blocked — this doc decides the capability question. |

No lane was reopened; no attention/activation/norm/weight-GEMV work; no 14B/32B.

## 4. Design A — explicit state token / opaque append node
**Microprobe: `DESIGN_A_MICROPROBE_PASS`** (`extra/qk_kv_append_microprobe.py`,
`bench/qk-kv-cache-stateful-jit/design_a_microprobe.json`). A raw-RDNA3 `custom_kernel`
(`extra/qk_kv_cache_state_token.py`) writes the current token's K/V into the persistent cache slice **in
place**, with `start_pos` as a `ProgramInfo` **runtime scalar var** (computed in-ISA, never baked into a
captured index). Results:
- **byte-correct** symbolic-offset append at positions 0/3/9 (slice correct + rest zero), and at MAXC=16 and
  MAXC=4608; the `kv_append_node` model entry places K/V correctly.
- **capture/replay with changing `start_pos`** (2,7,11,4) all correct from ONE compiled linear —
  **bypassing the `bind mismatch on start_pos` that killed `.assign()`** (the var is patched per replay, not
  baked). This is the hard part, and it passed.

**In-model integration: `DESIGN_A_IN_MODEL_FAIL_READ_SIDE`.** The opaque WRITE works, but inside
`@function(precompile=True)` the read/persistence side fails with two **mutually exclusive** failures:
- **(a) without repointing `cache_kv.uop`** → `@function` does not track the opaque write as a cache mutation
  (it is not a tinygrad `STORE`), so the KV does not persist correctly across the precompiled calls → garbage
  tokens.
- **(b) with `cache_kv.uop` repointed to the after-node** (mirroring `Tensor.assign`'s simple branch, so
  `@function` tracks it) → the same-graph attention reduce over the just-mutated buffer **reintroduces the
  read-after-write hazard** → `KeyError` on the flash `REDUCE` uop — the **identical** failure as the original
  `.assign()` probe.

**Root cause:** the full-MAXC copy is the functional scheduler's resolution of (i) the same-graph
read-after-write hazard between the append and the attention reduce, and (ii) the symbolic-size requirement —
a symbolic read-prefix `[0:start_pos+T]` cannot be a buffer (`rangeify.py:397` "no zero sized or symbolic
sized buffers"; `ops.py:443` `eval failed to be a single number`). The opaque node escapes the symbolic-alias
analysis for the **write**, but attention's **read** of the mutated buffer in the same pure-function graph is
the wall. (The microprobe passed precisely because its read was a *separate* `run_linear`/`.numpy()`, not a
same-graph reducing consumer.)

## 5. Design B — alias-aware slice mutation rule
**`KV_ALIAS_RULE_UNBOUNDED`.** A bounded rule recognizing
`cache[start_pos:start_pos+1].assign(...)` + a same-graph read of `[0:start_pos+1]` as a safe ordered
read-after-write would need symbolic-**range** alias analysis (does the read prefix include the written
slice?) under symbolic `start_pos` — which hits the same `_eval`/`_min_max` symbolic-resolution wall
(`ops.py:440-445`) and the symbolic-sized-buffer reject (`rangeify.py:397`). The relevant seams:
`rangeify.py:91-98` (`fix_store_hazard`), `callify.py:54-57` (materialization decision). Per the scope ("Stop
if this expands into broad alias analysis"), this is **unbounded** — not a bounded scheduler change.

## 6. W==D result
**Not reached.** In-model correctness failed at Phase 2, so the W==D ctx512/1024/2048/4096 sweep did not run.

## 7. Final verdict
**`KV_RUNTIME_MANAGED_CACHE_REQUIRED`.** The bounded opaque-node (A) is proven for the write but insufficient
in-model; the bounded alias rule (B) is unbounded. The viable elimination is **Design C — runtime-managed /
two-graph KV state** (cache outside pure-tensor value semantics; append as a side effect realized separately,
like `examples/gpt2.py`'s `.assign().realize()`; attention receives a pointer+length). This avoids the
single-pure-function read-after-write hazard but is a separate, larger project (the vLLM/TRT-LLM model).
**Design D (paged/block-table)** is documented-only — it reopens attention layout (page-table-aware kernels),
out of this bounded scope.

## 8. Next funded scope
**Recommend: none now.** Do **not** fund a bounded JIT capability — none suffices. Keep the canonical
full-buffer copy as the safe functional default. Revisit **Design C (runtime-managed KV)** as a separate
project **only if** the ~1.4 ms/token (which grows with `max_context`, so larger at longer-context or
larger-model serving) justifies taking the KV cache out of pure-tensor semantics. If reopened, the
deliverables already exist as a head start: the proven opaque append kernel (`extra/qk_kv_cache_state_token.py`)
is the write half; the missing half is a runtime-managed read path (separate-graph append + pointer/length
attention) that sidesteps the same-graph hazard.

## 9. Files changed, artifacts, commands, git status
- **Source: none shipped.** The gated `KV_CACHE_STATE_TOKEN` route (`model.py` `_attention` + `_init_state`)
  was added, probed, and **reverted**; `model.py` is byte-clean and default decode is byte-identical
  (`[279, 1156, 22148, 18495, 1033, 5798, …]`).
- **New (kept): `extra/qk_kv_append_microprobe.py`** (Design A microprobe, PASS),
  **`extra/qk_kv_cache_state_token.py`** (the proven opaque append kernel + node — the write half of a future
  Design C).
- **Artifacts:** `bench/qk-kv-cache-stateful-jit/{design_a_microprobe,latest}.json`.
- **Commands:** microprobe `DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_kv_append_microprobe.py`;
  in-model repro requires re-applying the gated route (this doc §4) then
  `... KV_CACHE_STATE_TOKEN=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 ...`.
- **git status:** `model.py` clean; new `extra/` tools + `docs/` + `bench/` artifacts (the bench JSON is not
  committed — it carries timing/non-deterministic fields and references the reverted route).
