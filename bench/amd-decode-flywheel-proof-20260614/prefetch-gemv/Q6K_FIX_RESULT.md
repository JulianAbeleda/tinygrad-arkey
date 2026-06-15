# FIX: enable the Q6_K primitive → 2.2× decode (23→51 tok/s), byte-identical output

Date: 2026-06-15. Acts on BREAKDOWN_RESULT (the decode bottleneck is the Q6_K matmuls running as a slow
generated fp-dequant reduce, ~59% of GPU work, because only Q4_K primitives were installed). The Q6_K
primitive was already fully built (`extra/q6_k_gemv_primitive.py`, `Q6KPrimitiveLinear`, `_install_q6k_
primitives`, `_q6k_policy`) but gated behind a `Q6K_PRIMITIVE` flag that nothing set.

## Result (RX 7900 XTX, full clock, Qwen3-8B Q4_K_M, sustained tok/s)
| config                                              | tok/s | vs llama.cpp (105.7) |
|-----------------------------------------------------|------:|---------------------:|
| Q4K_PRIMITIVE only (Q6_K falls back) — the old path |  23.1 |        22%            |
| **Q4K_PRIMITIVE (Q6_K now auto-on) — ffn_down**     |**50.8**|       **48%**         |
| + Q6K_COVER_MORE (attn_v + lm_head)                 |  53.1 |        50%            |

**2.2× decode speedup**, and the output is **byte-identical** — a 20-token greedy continuation matches the
pure-fp baseline exactly (`[323, 358, 1079, 1588, 311, 1492, 498, 448, 697, 4755, 13, 5209, 2666, 1910, 311,
2548, 752, 4113, 11, 323]`). Q6_K dequant is exact (unpack self-test max_abs=0; GEMV max_abs=9.7e-4 < 1e-2).

## Mechanism (confirmed by profile)
The Q6_K `ffn_down` (18/36 layers, mixed-quant Q4_K_M) was `r_32_32_4_48` = **59% of GPU decode work** at
~38 GB/s. With the primitive it becomes `q6k_gemv_partial_4096_12288` = ~3.4 ms (16%). non-GEMV GPU work
dropped 37 ms → 14 ms/token; `r_32_32_4_48` is gone entirely.

## Changes (tinygrad/llm/model.py)
1. **Q4K_PRIMITIVE now implies Q6K_PRIMITIVE** (set `Q6K_PRIMITIVE=0` to opt out). The Q6_K matmuls were the
   bottleneck whenever the Q4_K fast path was on; since Q6_K dequant is exact, enabling it is a pure win.
2. **`_set_module_at` top-level fix**: handle a dotted-path-free module name (e.g. `output`) — required to
   primitivize the Q6_K lm_head; previously crashed with a tuple-unpack error.
3. **`_q6k_policy` + `Q6K_COVER_MORE`**: re-evaluated the stale "attn_v/output lose to the fused graph"
   comment — empirically they now WIN (+5%, 50.8→53.1), gated behind `Q6K_COVER_MORE` pending broader
   validation (ffn_down, the decisive win, is unconditional).

## Audit items fixed (tinygrad/renderer/cstyle.py) — not impacting any solution, hardened anyway
- **`_dp4a` gate** tightened from substring (`"_dp4a" in arg`) to a real-call regex (`(?<!\w)_dp4a\s*\(`),
  so `my_dp4a(` can't trigger a stray (harmless) helper. Verified: the real `dot = _dp4a(...)` call still
  matches; adversarial names don't; Q4K_VDOT decode still emits + runs the helper.
- **CUSTOM/QK_BLOCK_DOT `.format()`** wrapped in `_render_arg_format` that raises an actionable error
  (naming the op, the brace-doubling rule, and src count) instead of a bare IndexError. Success path is
  byte-identical. QK_BLOCK_DOT compile-gate test passes (5/5).
Neither audit item was impacting a current solution (both were latent/fail-loud); the Q6_K path uses neither
(it builds UOps, not format-string CUSTOM args).

## Status vs the mission
Decode went from 22% → 48–50% of llama.cpp with identical output, by fixing the diagnosed bottleneck. The
remaining gap is now: the Q6_K lm_head + attention reduces, and ~25 ms/token host/sync (the next levers).
The standalone-kernel WIN (Q4_K int-dot 76% > llama 57%) is unaffected.

Repro: `DEV=AMD Q4K_PRIMITIVE=1 ...` (Q6_K auto-on) vs `Q6K_PRIMITIVE=0` (opt out); exactness via 20-token
greedy continuation vs no-flags baseline; `extra/q6_k_gemv_primitive.py <gguf> --tensor <Q6_K ffn_down>` self-test.
