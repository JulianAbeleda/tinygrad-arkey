# Increment 0 SHIPPED: PREFILL_CONCRETE_KV + precompile-at-load — clean e2e prefill win

Date: 2026-06-20. Repo `/home/ubuntu/tinygrad-arkey`. gfx1100, Qwen3-8B. Scope:
`docs/prefill-increment0-and-flash-execution-scope-20260620.md` (Part A).

## What shipped

1. **Verified the e2e win lands** (A2). Earlier "compile tax dominates / no e2e change" was a broken harness
   (prompts too short → only one prefill-v2 chunk + the 32-token path). With genuine ≥1024-token prompts,
   `PREFILL_CONCRETE_KV=1` makes chunk@512+ concrete (`int:512`, the fusion path) instead of symbolic
   (`UOp:512`, SDPA):

   | | chunk schedule | cold prefill | warm prefill |
   |---|---|---:|---:|
   | ck=0 (default) | int:512, **UOp:512** | 10038 ms | 4966 ms |
   | ck=1 | int:512, **int:512** | 8526 ms | **3457 ms** |

   → **1.44× warm, 1.18× cold** e2e prefill, byte-identical. (For short prompts ck=0 also compiles a symbolic
   jit, so compile counts match and concrete wins outright; the per-start_pos tax only accumulates for *long
   cold* prompts.)

2. **Killed the compile tax (A1).** New `Transformer.precompile_concrete_prefill_jits()` (model.py), called from
   `from_gguf` when `PREFILL_V2 and PREFILL_CONCRETE_KV`, precompiles the per-start_pos concrete prefill jits
   (`ceil(max_context/UBATCH)` of them) ONCE at load. So even the FIRST cold generation is warm-fast:

   | | load | FIRST-gen prefill | tok0 |
   |---|---:|---:|---|
   | ck=0 | 12.5 s | 9.10 s | 3143 |
   | ck=1 (+precompile) | 23.8 s | **3.52 s** | 3143 |

   → **first cold generation prefill 2.58× faster, byte-identical** (same tok0). Compile moved to load (+11.3 s
   for ctx1536; scales with max_context). Opt-in via `PREFILL_CONCRETE_KV`; default (ck=0) unchanged — no
   regression. KV pollution from the dummy precompile passes is safe (a fresh model's first generation starts at
   start_pos=0 and overwrites the cache in chunk order before any position is read).

## Combined with Increment 0's per-chunk numbers

The concrete prefill forward is 1.7–4.4×/chunk faster than symbolic (73–111% of llama at every context,
byte-identical) — see `docs/prefill-concrete-kv-increment0-result-20260620.md`. With precompile, that forward win
now lands on every generation with no warmup. Net: prefill goes from collapsing to 17% of llama at long context
(symbolic) to holding 73–111% (concrete), e2e.

## How to use
`PREFILL_V2=1 PREFILL_CONCRETE_KV=1` (gfx1100). Trades a one-time load cost (~5 s × ceil(max_context/512) jits)
for 1.7–4.4×/chunk faster prefill on every generation. Recommended for servers / repeated prompts / long context.
Left opt-in (not default-on) because the load cost is real and only amortizes with reuse or long prompts.

## Gates
Byte-identical greedy (tok0 match + per-chunk rel_RMSE 0.0 + dNLL 0.0 from the Branch B / Increment 0 gates);
synced warm prefill improvement (1.44×) and first-cold-gen improvement (2.58×); default path unchanged.

## Not done here (→ Part B)
The cold one-shot case still pays the precompile cost at load. **Increment 2 (flash kernel)** removes it entirely
(compiles once, symbolic-native, no per-start_pos jits, no score materialization) — the durable answer, built next.
