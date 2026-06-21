# Increment 0 Result: forcing concrete chunks captures the GPU-throughput prize — but is MASKED end-to-end

Date: 2026-06-20
Repo: `/home/ubuntu/tinygrad-arkey`, branch `qk-prefill-flag-leak-resolution`. GPU gfx1100 (RX 7900 GRE).
Model Qwen3-8B-Q4_K_M. Scope: `docs/prefill-flash-wmma-attention-scope-20260620.md` (Increment 0).
Harness: `extra/qk_prefill_concrete_kv_increment0.py` + e2e/dispatch probes (`/tmp/inc0_e2e.py`, `/tmp/inc0_dispatch.py`).

## Question

Does forcing every prefill chunk CONCRETE (`PREFILL_CONCRETE_KV=1`, so the shipped fusion attention fires on all
chunks, not just start_pos=0) collapse the SYMBOLIC-regime attention cost — capturing the prize with **no new
kernel**? And at what compile / materialization tax?

## Result 1 — pure GPU prefill forward: YES, a large win (the llama-comparable metric)

Per-`start_pos` synced interleaved arbiter (K=8 forwards/one sync, clock pinned high), symbolic (today, SDPA) vs
concrete (Increment 0, fusion):

| start_pos | KV | symbolic ms (today) | concrete ms (Inc 0) | speedup | % llama (170ms) | byte-identical |
|---:|---:|---:|---:|---:|---:|:--:|
| 0 | 512 | 259 | 153 | 1.69× | 66 → 111 | rmse 0.0 |
| 512 | 1024 | 398 | 162 | 2.45× | 43 → 105 | rmse 0.0 |
| 1536 | 2048 | 680 | 192 | 3.55× | 25 → 89 | rmse 0.0 |
| 3072 | 3584 | 1024 | 234 | 4.38× | 17 → 73 | rmse 0.0 |

The symbolic forward **degrades catastrophically with context** (259→1024 ms) — the symbolic-start_pos codegen
penalty compounds — while concrete degrades gently (153→234 ms) and **holds 73–111% of llama at every context**.
Confirmed through the REAL `model.__call__` dispatch (warmstart install + jit keying), warm: a 1536-token,
3-chunk prefill forward is **558 ms concrete vs 1213 ms symbolic = 2.17×**. The Increment-0 win is real, survives
real dispatch, and is byte-identical.

**On the standard prefill-throughput metric (what llama pp512 = 3020 tok/s measures), Increment 0 captures the
prize with zero new kernel code.**

## Result 2 — end-to-end generate(): NO change off-vs-on, but for a DIFFERENT reason than first hypothesized

`generate()` time to first token, 1536-token prompt, `PREFILL_CONCRETE_KV` 0 vs 1: cold 24.46 vs 24.91 s; warm
7.16 vs 7.10 s — **identical off-vs-on**. I first hypothesized a large host overhead; **that was WRONG.** A
class-method timing wrapper on `__call__` attributes it precisely:

- **Non-forward (host/setup) overhead = 21.8 ms — negligible.** The warm ~5.6 s IS forward compute.
- BUT it is **~12 forward calls**, each `start_pos=symbolic UOp`, `T=32` (BIND toks 1..32), ~498 ms each — i.e.
  the **`else` branch (model.py:1288): chunk_size=32 SYMBOLIC decode-style chunks**, NOT the prefill-v2 512-chunks.

So in this e2e run **neither ck arm engaged the concrete prefill-v2 path** — the prompt was processed as slow
32-token symbolic chunks — which is exactly why ck=0 and ck=1 were identical. Why generate() fell into the
32-chunk path here: the e2e harness warmed on a prior repeated-filler prompt, so prefix-caching left a large
`start_pos`; with `prompt_len - start_pos < PREFILL_UBATCH(512)` the prefill-v2 guard (model.py:1279) is false →
the 32-token symbolic remainder path. This is a **prompt/prefix-cache artifact of the harness**, not a property
of Increment 0.

Cold (24.5 vs 24.9 s) is also a wash, but for the real reason: the per-distinct-start_pos **compile tax** (~5 s
per concrete jit, measured `concrete_capture_s`) — ck=1 compiles 3 concrete jits where ck=0 compiles fewer, and a
single cold pass runs each chunk once, so compile offsets the run saving. Only amortizes with jit-caching.

## Verdict

- **Increment 0 delivers a real prefill-v2 forward win:** 2.17× warm on a clean 3-chunk dispatch (558 vs 1213 ms),
  1.7–4.4×/chunk, 73–111% of llama, byte-identical, no new kernel. Solid (arbiter + real-dispatch probe).
- **There is NO host-overhead problem** (21.8 ms non-forward) — correcting the earlier hypothesis. Warm prefill is
  forward-compute-bound, as the measured map says ([[inference-perf-measured-map]]).
- **It is not a clean e2e ship today**, for two integration reasons (NOT host overhead):
  1. **Cold compile tax** — one jit per distinct start_pos (~5 s each); needs jit-caching to amortize.
  2. **generate() chunk scheduling** — generate() can fall into the slow 32-token symbolic `else` path (here via a
     prefix-cache artifact) so the concrete prefill-v2 fast path isn't even exercised. The win only lands if
     generate() actually routes the prefill through concrete prefill-v2 512-chunks.

## Implication for the roadmap

The earlier "attribute a 6.5 s host overhead first" step is **withdrawn** — there is no such overhead. The real
gating questions for whether the attention prize lands e2e are narrower and concrete:
1. **Does a genuine COLD first prompt (no prefix cache, prompt ≥512) engage prefill-v2 512-chunks** for the bulk,
   or does it too fall into the 32-token path after chunk@0? (Determines if Increment 0 / flash can ever help e2e
   as generate() is written.)
2. **Jit-caching** to amortize the per-start_pos compile tax (server/repeated-prompt use case).

The flash kernel (Increment 2) stays attractive precisely because it **compiles once** (no per-start_pos tax) and
is symbolic-capable — it would also serve the 32-token symbolic remainder path, sidestepping both integration
issues above.

## Recommended next steps (revised)
1. **Clean cold-first-prompt e2e check** (cheap): one fresh-model generate() on a ≥1024-token prompt, trace the
   per-call start_pos/T like `/tmp/inc0_trace.py`, confirm whether the bulk goes through prefill-v2 512-chunks
   (concrete vs symbolic) or the 32-token `else` path. This tells us if generate()-as-written can realize the
   forward win at all.
2. If prefill-v2 512-chunks ARE engaged cold → ship Increment 0 with **jit-caching** (amortize compile) for the
   throughput win; and/or build **Increment 2 (flash)** for the symbolic/long-ctx + 32-token tail.
3. If generate() keeps falling into 32-token chunks → the lever is generate()'s **prefill chunk scheduling**
   (route more through concrete prefill-v2), orthogonal to and prerequisite for any attention-kernel work.

## Notes
- The probe's attention-share buckets read 0.0% (PROFILE env init-timing in the subprocess — profiling captured no
  ProfileGraphEvents); the perf numbers (burst arbiter) are unaffected. Attention-share mechanism is already
  established in `docs/prefill-branch-b-tc-attention-result-20260620.md` (concrete attention ~5% vs symbolic ~47%).
- Iron law held: synced same-process interleaved arbiter, clock pinned, byte-identical correctness per row.
