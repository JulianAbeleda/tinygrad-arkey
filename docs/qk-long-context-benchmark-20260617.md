# tinygrad long-context decode benchmark (2026-06-17)

`extra/qk_long_context_bench.py`, artifact `bench/qk-long-context-20260617/result.json`. Qwen3-8B-Q4_K_M,
gfx1100. Prefills incrementally to each checkpoint and measures decode tok/s over a fixed NTOK window at that
context depth (decode degradation curve, separated from prefill). Wall = DEBUG=0 perf_counter (real e2e); GPU =
time_sum_s @ DEBUG=2 (authoritative). Same tokens across configs; greedy argmax recorded per checkpoint.

## Decode degradation vs context (the headline)

| ctx | baseline decode | flash-decode | flash speedup |
|---:|---:|---:|---:|
| 512 | 39.0 tok/s | 41.0 | 1.05× |
| 1024 | 28.7 | 35.3 | 1.23× |
| 2048 | 19.1 | 27.9 | 1.46× |
| 4096 | **11.4** | **19.7** | **1.73×** |

- **Baseline decode degrades 3.4× from ctx 512→4096** (39.0 → 11.4 tok/s): the attention/KV-cache read grows
  linearly with context and dominates long-context decode.
- **Flash-decode (already implemented, gated by `FLASH_DECODE`) recovers most of it — 1.73× at 4096**, and the
  benefit grows monotonically with context (1.05× → 1.73×). At short context it barely helps (why it isn't
  default), but for long context it is a clear, already-built, low-risk win.

## Caveats (measurement honesty)

- **Prefill numbers in the artifact are unreliable** (cold compile folded into the first segment; first-token
  latency at ctx 512 = ~4.2 s = JIT compile, then ~21 ms warm). Use the banked PREFILL_V2 prefill numbers
  (~2486 tok/s) and llama-bench pp512 (~2742–3104 tok/s) for prefill comparison, NOT this harness's prefill.
- The decode tok/s here (39 @ ctx512) is below the banked short-context ~64 because (a) decode is measured at a
  *already-512-deep* context, and (b) no ffn_down demotion in this run. The CURVE (relative degradation + flash
  recovery) is the clean, valuable result, not the absolute ctx-512 number.
- **llama long-context decode is not cleanly isolatable** from llama-bench `-pg` (combined pp+tg throughput).
  llama uses flash attention by default, so it degrades gracefully like tinygrad's flash-decode config. Short
  decode: llama tg128 ≈ 80–100 tok/s (thermal-noisy) vs tinygrad ~54–64.

## Implication for priorities

**Long context flips the priority.** At short context the gap vs llama is small-op/fusion overhead (Phase-1
census: ~38% non-GEMV). At long context, the **attention/KV-cache read dominates and decays decode 3.4×**, and
**flash-decode is the highest-value, lowest-risk lever (already built, 1.73× @ 4096)** — making it default for
long context (or auto-enabling above a context threshold) is the obvious near-term win for long-context decode.
