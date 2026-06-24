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

## Result 3 — cold chunk-scheduling RESOLVED: prefill-v2 512-chunks DO engage cold

A clean cold trace (fresh model, empty cache, 1024-token varied prompt; `/tmp/inc0_cold2.py`) shows generate()
makes exactly **2 forward calls: `start_pos=int, T=512` (concrete chunk@0) + `start_pos=UOp, T=512` (symbolic
chunk@512)**. So on a genuine cold prompt the prefill-v2 path engages properly, and chunk@512 is symbolic only
because ck=0 is default — with `PREFILL_CONCRETE_KV=1` it becomes concrete → Increment 0's per-chunk win applies
to every chunk past the first. The warm "12× 32-token" pathology was purely a **prefix-cache artifact** of the
repeated-filler harness (resume left `prompt_len - start_pos < 512` → the 32-token `else` path); it is not a
property of cold prefill and not Increment 0's concern.

Concrete cold arithmetic, 1024-prompt: ck=0 forward ≈ 153 (concrete@0) + 398 (symbolic@512) = 551 ms; ck=1 ≈ 153
+ 162 = 315 ms → **1.75× on the cold forward**, plus a one-time ~5 s compile for the extra concrete jit.

## Implication for the roadmap (finalized)

- The earlier "attribute a 6.5 s host overhead first" step is **withdrawn** — there is no host overhead (21.8 ms).
- Cold chunk-scheduling question **resolved**: prefill-v2 512-chunks engage cold; Increment 0 makes chunks 2+
  concrete → a real cold forward win (~1.75× on 1024-prompt, scaling with context).
- **Sole remaining blocker = the per-start_pos compile tax** (cold, ~5 s/jit). Two ways it pays off:
  (a) **jit-caching / repeated prompts / server** — compile amortizes across generations → the full 1.7–4.4×
  forward win lands; (b) **long prompts** — many chunks each saving 0.2–0.8 s eventually exceed the one-time
  compile. For a one-shot short cold prompt, compile dominates → no e2e win.
- **Increment 2 (flash) is the cleaner general answer**: compiles ONCE (no per-start_pos tax), symbolic-capable
  (serves chunk@512+ directly without forcing concrete), no score materialization, and also covers the 32-token
  remainder path. It sidesteps the compile tax entirely.

## Recommended next steps
1. **Ship Increment 0 gated on jit-caching** for the repeated-prompt/server use case (the forward win is real and
   byte-identical; only the cold compile tax blocks the one-shot case). Confirm `prefill_v2_jits` persist across
   generations and measure a warm-second-generation e2e on a clean (non-prefix-colliding) prompt.
2. **Build Increment 2 (flash)** for the one-shot / long-context / symbolic case — it removes the compile tax and
   the score materialization that Increment 0 still pays at long KV. This is the durable lever.
3. Increment 1 (warmstart TC-opts on the attention matmuls) remains optional, gated on a per-kernel attribution
   of the explicit path's residual cost.

## Notes
- The probe's attention-share buckets read 0.0% (PROFILE env init-timing in the subprocess — profiling captured no
  ProfileGraphEvents); the perf numbers (burst arbiter) are unaffected. Attention-share mechanism is already
  established in `docs/prefill-branch-b-tc-attention-result-20260620.md` (concrete attention ~5% vs symbolic ~47%).
- Iron law held: synced same-process interleaved arbiter, clock pinned, byte-identical correctness per row.
