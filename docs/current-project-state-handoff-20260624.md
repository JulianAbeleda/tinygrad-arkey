# Current Project State — Handoff (2026-06-24)

Canonical, high-signal snapshot. If anything elsewhere contradicts this file, this file wins.
Machine: gfx1100 RX 7900 XTX 24GB, Qwen3-8B-Q4_K_M. Supersedes the 2026-06-21 handoff (now in
`docs/archive/`), whose `~67% llama` decode baseline and "bounded decode RESTED / capped at backend
ceiling" frontier were overturned by the 06-22→06-24 campaign (owned attention tile + buffer-identity fix).

## 1. Canonical numbers (clean-wall, PROFILE=0, auto clock, W==D)

| metric | value | source |
|---|---|---|
| decode @ctx 512 / 1024 / 2048 / 4096 | **101.6 / 99.8 / 97.4 / 92.9 tok/s** (≈ **100.6–104.0% of llama** — at/above parity) | `prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md` |
| decode full-stack envelope (non-search) | 104.0 / 102.1 / 99.6 / 95.1 tok/s | `decode-q4k-gemv-warp-promotion-result-20260624.md` |
| llama reference (same ctx) | 97.71 / 97.39 / 95.00 / 92.37 tok/s | `bench/qk-post-parity-hardening/authority.json` |
| prefill @ctx 512 / 1024 / 2048 / 4096 / 8192 | **3597 / 3505 / 3263 / 2784 / 2217 tok/s** (`eightwave` promoted; ~+3% over prior) | `prefill-eightwave-promotion-result-20260624.md` |
| q8 FFN opt-in | ~+7% decode, **default-OFF, dNLL-gated** | `Q8_FFN_HANDWRITTEN=1` |
| VRAM | default ~5–6 GB; **`PREFILL_V2` adds ~+14 GB fp16** (≈19–21 GB), resident through decode | handoff history |

**⚠ Flag-stack caveat (do not mis-state the parity claim).** The at/above-llama decode numbers above are the
**canonical default stack with `Q4K_GEMV_WARP*` enabled** (promoted default 2026-06-24). A fresh *default-off*
run reads **below** llama — the two families are reconciled in `decode-parity-no-regression-audit-result-20260623.md`.
Always quote the canonical-stack number with the stack stated, not bare.

## 2. Decided policies (do not re-open)

- **Global `PREFILL_V2` default stays OFF** (decided 2026-06-21, unchanged). Not flipped to `auto` — the +14 GB
  fp16 prefill state stays resident during decode for zero decode benefit; the common decode/short-prompt user
  must not pay it.
- **`PREFILL_V2=auto`**: opt-in (VRAM-gated; 24GB+ on, ≤16GB / unknown off).
- **`PREFILL_SERVER_PROFILE=1`**: opt-in (⇒ `PREFILL_V2=auto` + concrete-KV precompile; server/long-prompt profile).
- **`PREFILL_REMAINDER_FIX`**: default-ON but only active under `PREFILL_V2`; byte-identical (kills the 32-token trap).
- **q8 FFN (`Q8_FFN_HANDWRITTEN=1`)**: opt-in, default-off.
- **`Q4K_GEMV_WARP*`**: promoted **default-ON** (the warp GEMV that lands decode weight-GEMV at/below llama).
- **`eightwave` prefill**: promoted default.

## 3. What changed since the 06-21 handoff (the parity win)

The 06-21 frontier called bounded decode RESTED and decode "capped at tinygrad's backend ceiling (~67% llama),
closable only by a deep, separately-funded codegen capability." The 06-22→06-24 campaign refuted that:

- **Attention was not exhausted.** The owned hand-AMDGCN decode-attention tile (north-star lane, not a bounded
  patch) added +12–22% and entered the decode path — see `decode-campaign-final-synthesis-20260623.md` and
  `post-owned-attention-promotion-synthesis-20260623.md`.
- **Buffer identity was the actual wall**, not a runtime-KV core block; resolving it unblocked W==D promotion.
- **Weight-GEMV** reached at/below llama via the `Q4K_GEMV_WARP` warp kernel (promoted default,
  `decode-q4k-gemv-warp-promotion-result-20260624.md`).
- **Prefill** held no-regression and promoted `eightwave` (`prefill-long-context-no-regression-audit-result-20260623.md`,
  `prefill-eightwave-promotion-result-20260624.md`).

Net: tinygrad decode now runs at/above llama parity across ctx 512–4096 as the **default** route; prefill ~114% of
llama pp512. Coverage map: `gpu-lifecycle-primitive-coverage-tracker-20260624.md`.

## 4. Where to start

1. `docs/README.md` — curated navigation map.
2. `docs/current-project-state-handoff-20260624.md` — this file (canonical current state).
3. `bench/README.md` — bench/evaluator map.
4. `docs/decode-campaign-final-synthesis-20260623.md` — how decode reached parity.
5. `docs/prefill-decode-next-workstreams-codex-scope-20260624.md` — next-work map.

Historical provenance (the full 06-16→06-22 probe log, superseded results, completed scopes) lives in
`docs/archive/` and is indexed by `docs/provenance-index-20260624.md`. It is kept for history, **not authority**.

## Consistency guardrail
Run `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py` — it scans the canonical
START-HERE docs and fails if one re-opens a closed question (bare `87.6` with no context, an open
`PREFILL_V2=auto` owner call, a "flip global PREFILL_V2=auto" proposal, or bounded decode fusion presented as
current work).
