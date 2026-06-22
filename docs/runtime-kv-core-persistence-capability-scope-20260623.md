# Runtime-KV Core Persistence Capability — Design Scope (2026-06-23)

## Verdict: `RUNTIME_KV_CORE_CAPABILITY_SCOPE_READY_DESIGN_A`

This is a **core tinygrad runtime/lifecycle design scope**, not a kernel or model task and not an implementation
patch. Goal: enable **persistent mutable decode KV state across TinyJit replay without full-MAXC `.after()`
materialization**. Recommended design: **A (runtime-managed KV object) with the append run as a pre-graph runtime
side-effect** (the vLLM "update KV before graph replay" model). Do **not** implement without explicit owner
authorization + a separate design review.

## Why this is the prize
MAXC-shrink proved the materialization is on the W==D critical path (**+11.8%@1536 / +12.9%@1280 → ~llama
parity**). `E_49152` ≈ 1.5ms/token. It is `RUNTIME_GRAPH_LIFECYCLE_GAP` (proven NOT ISA, NOT the owned tile, NOT
arg-patching).

## The core problem (precisely)
The canonical `assigned_kv = Tensor(cache_kv.after(store))` does **two** jobs at once:
1. **ordering** — the attention read happens after the KV write;
2. **`@function` cross-replay persistence** — materializing the full cache makes the mutated buffer a tracked,
   carried-forward value.

Removing the materialization (opaque append, microbench-proven) removes job 2 → the full-model decode **bakes from
step 1**. The needed capability decouples these: a **runtime-owned mutable buffer** that persists across replay and
is **read** by the pure graph as a stable input, with the **write** done as a runtime side-effect ordered before
the captured graph (exactly how vLLM updates KV blocks before CUDA-graph replay — see
`docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`).

## Required design questions (answers below per design)
| question | requirement |
|---|---|
| What owns mutable KV state? | an explicit runtime-managed buffer/object, not a pure-Tensor `.after()` value |
| How does it persist across TinyJit replay? | a stable realized buffer input; the captured graph never re-materializes it |
| How are writes ordered before reads? | the append is a runtime side-effect executed before the captured read graph (or an explicit dependency token) |
| How is full-cache `.after()` avoided? | the read graph takes `cache_kv` as a pre-existing input; no `after(store)` materialization |
| How is aliasing represented? | bounded: the append writes only `[start_pos]`; the read uses `[0:start_pos+1]` — no general symbolic alias analysis |
| How is `start_pos` handled? | runtime var (already proven patchable), not a baked symbolic index |
| Fallback | default-safe: any failure/unsupported shape → canonical materialized path (current default), no silent corruption |
| Smallest proof | a toy mutable-buffer replay before any model wiring |

## Candidate designs (evaluated against prior evidence)
### Design A — Runtime-managed KV object, append as pre-graph side-effect (RECOMMENDED)
The cache is a runtime-owned realized buffer. Each decode step: the **append kernel runs as a runtime mutation
(outside the TinyJit-captured pure graph)**, then the captured decode graph reads the buffer as a stable input.
This matches vLLM (static buffers + metadata update before replay).
- **Pros**: semantically honest (mutable KV is not pure dataflow); production-aligned; bounded alias (write
  `[start_pos]`, read prefix); no per-layer extra graph if the append is batched once per step.
- **Risk**: needs a tinygrad contract for "run this side-effecting kernel before the captured graph, on a buffer the
  graph reads" — a new but **narrow** runtime API. The prior RUNTIME_KV attempt put the append *inside* the
  captured graph (bypassing `@function`) and baked; moving it *outside* is the fix.
- **Prior status**: the in-graph variant baked; the **outside-the-graph (pre-replay) variant is untried** and is
  the actual capability.

### Design D — Two-graph decode split (FALLBACK)
Per layer/step: an append graph (writes cache, realized) and a separate attention/read graph, with an explicit
runtime-state boundary between them.
- **Pros**: cleanest read-after-write separation; closest to runtime-managed cache systems.
- **Risk**: per-layer graph boundary (36×) → launch overhead that could eat the win; must measure.

### Design B — State-token dependency primitive (PARTIAL)
Append returns a lightweight ordering token; attention depends on it without materializing.
- **Addresses ordering but NOT cross-replay persistence** (the harder half). Useful as a sub-primitive of A/D, not
  a standalone fix.

### Design C — Bounded KV alias rule (TOO BROAD — reject)
Special-case symbolic append/read alias so the pure graph reads a prefix without materializing.
- **Prior status**: attempted (repoint `cache_kv.uop`) → `REDUCE` read-after-write hazard / general symbolic alias
  analysis. **`TOO_BROAD`** per the stop rules — do not pursue.

## Minimal proof ladder (do these in order; stop at the first failure)
1. **Toy buffer proof** — a runtime-owned realized buffer; an opaque append at a runtime `start_pos` run as a
   pre-replay side-effect; a captured TinyJit graph reads the prefix; replay with changing `start_pos`; **assert no
   full-buffer materialization in the captured graph** and correct accumulation. (No model.)
2. **One-layer KV proof** — one TransformerBlock: append K/V (pre-graph), owned tile reads the persistent buffer;
   numeric reference vs the materialized path; **multi-step persistence**, token-correctness proxy.
3. **Full-model shadow proof** — default-off flag; ctx1024 real prefill; **byte-identical tokens vs baseline**;
   `E_49152` absent/reduced; multi-prompt.
4. **W==D proof** — ctx512/1024/2048/4096; no regression; expected **≥+5%, likely parity-class** (MAXC-shrink
   predicts ~+11%).

## Gates
- **Correctness**: token-authority byte-identical multi-step (NOT position-written), 2 prompts, finite K/V.
- **Graph identity**: no `E_49152`/full-MAXC copy; append before read; route fires; default fallback intact.
- **W==D**: ≥+5%@ctx1024 (canonical harness, `.item()` in window, repeated); no ctx512 regression.
- **ISA guard**: the append kernel passes `extra/qk_isa_primitive_audit.py` (no spills, expected stores).

## Stop rules (honored)
- Requires general symbolic alias analysis → `TOO_BROAD` (this is why Design C is rejected).
- Requires rewriting all TinyJit buffer semantics → `TOO_BROAD`.
- No describable toy proof → do not implement.
- Cannot preserve token correctness → stop, revert, classify.

## Boundaries
No attention/GEMV work; no machine search; no 14B/32B; no default flip; no production implementation without owner
authorization. This doc is the design scope only.
