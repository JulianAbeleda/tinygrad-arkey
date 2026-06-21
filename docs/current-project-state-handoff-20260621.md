# Current Project State — Handoff (2026-06-21)

Canonical, high-signal snapshot. If anything elsewhere contradicts this file, this file (and the linked
reconciliation result) wins. Machine: gfx1100 RX 7900 XTX 24GB, Qwen3-8B-Q4_K_M.

## 1. Canonical numbers (clean-wall, PROFILE=0, auto clock)

| metric | value | source |
|---|---|---|
| decode @ctx≈0 | **~85–86 tok/s** (empty-KV peak; a contextual number, see §3 on `87.6`) | CLI `--warmup --benchmark` |
| decode @ctx 512 / 1024 / 4096 | **68.1 / 66.4 / 60.7 tok/s** (≈ **~67% llama** — the steady-state headline) | `extra/qk_decode_runtime_overhead.py` (W) |
| q8 opt-in @ctx 512 / 1024 / 4096 | **72.8 / 70.9 / 64.3 tok/s** (~+7%, default-OFF, dNLL-gated) | same harness, `Q8_FFN_HANDWRITTEN=1` |
| prefill (opt-in fast path) | concrete-KV **73–111% of llama pp512**; warm prefill **0.17–1.6 s** | `docs/prefill-policy-integration-result-20260620.md` |
| VRAM | default ~5–6 GB; **`PREFILL_V2` adds ~+14 GB fp16** (≈19–21 GB), resident through decode | `docs/decode-prefill-headline-reconciliation-result-20260621.md` |

## 2. Decided policies (do not re-open)

- **Global `PREFILL_V2` default: OFF** (decided 2026-06-21). It is **not** flipped to `auto` — the +14 GB fp16
  prefill state stays resident during decode for zero decode benefit; the common decode/short-prompt user must not pay it.
- **`PREFILL_V2=auto`**: opt-in (VRAM-gated; enables only where it fits — 24GB+ on; ≤16GB / unknown off).
- **`PREFILL_SERVER_PROFILE=1`**: opt-in (⇒ `PREFILL_V2=auto` + concrete-KV precompile; the server/long-prompt profile).
- **`PREFILL_REMAINDER_FIX`**: default-ON but only active under `PREFILL_V2`; byte-identical (kills the 32-token trap).
- **q8 FFN (`Q8_FFN_HANDWRITTEN=1`)**: opt-in, default-off.

## 3. Closed lanes

- **Prefill kernels** — solved. Flash-prefill v2 was correct but ~15× too slow; not reopened.
- **Prefill default-owner call** — closed: global default stays OFF (§2).
- **Bounded decode fusion** — closed. FFN activation producer-fusion built & byte-exact but ~0% (work-conserved,
  not launch-recoverable); attention reduce/stat microfusion is a no-go (intrinsic O(KV) QK/softmax). See
  `docs/decode-fusion-build-result-20260620.md`.
- **`87.6` ambiguity** — reconciled. `87.6` is a **numeric coincidence**: a real **ctx≈0 decode tok/s** (~11.4 ms)
  AND, separately, a real **ctx4096 decode ms/token** (=11.4 tok/s). **Never quote `87.6` bare.** The decode headline
  is the **curve** (~86 @ctx≈0 → ~61 @ctx4096), characterized as **~67% llama**, not the ctx≈0 peak.

## 4. Open frontier (Claude 1 only)

- **Decode is the frontier.** The single live lever is **fused + coop-optimized in one primitive** —
  `docs/decode-fused-coop-primitive-roadmap-scope-20260621.md` (Claude 1).
- **No tactical decode patch** until that roadmap returns `BRIDGE_FIRST` or `LINEARIZER_FIRST` (else `ROADMAP_ONLY`).

## 5. Where to start

`docs/README.md` · `bench/README.md` · `docs/decode-prefill-headline-reconciliation-result-20260621.md` (headline
authority) · `docs/decode-fused-coop-primitive-roadmap-scope-20260621.md` (Claude 1's live lane).

## Consistency guardrail
Run `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py` — it fails if a canonical doc
re-opens a closed question (bare `87.6`, an open `PREFILL_V2=auto` owner call, "flip global PREFILL_V2=auto",
"decode headline 87", or bounded decode fusion as current work).
