# Retained pp512 prefill evidence index

This audit reuses identity-qualified prefill evidence. It contains no decode
ledger measurements and changes no production route.

- `../nv-prefill-corrected-tile-20260828/` — corrected tinygrad FP16
  correctness and R7 wall authority.
- `../nv-prefill-restoration-20260828/` — retained llama pp512 R5 wall
  authority.
- `../nv-prefill-q4-imma-gate-20260828/` — extracted llama packed-MMQ
  discriminator and service estimates.
- `../nv-prefill-roofline-counters-20260828/` — llama MMQ physical counters;
  cache-hot replay is labeled and not used as cold wall authority.
- `../nv-q8-compact-producer-20260828/` — exact compact Q8 producer evidence.
- `../nv-q4-imma-combined-chain-20260828/` — full real-output packed-v4
  oracle, read-only checks, and R9 primitive timing.
- `../nv-compiler-packed-fragment-20260828/` — typed compiler fragment and
  group-correction gates, K64 geometry sweep, 72-real-weight proxy, fresh
  compiler-packed model/control R9, logits comparison, and profile census.

The lifecycle totals and compiler-ownership findings are explained in:

- `../../output/nv-llama-prefill-lifecycle-audit.md`;
- `../../output/nv-q4-imma-combined-chain-integration-20260828.md`;
- `../../output/nv-llama-mmq-warp-ownership-audit.md`;
- `../../output/nv-compiler-packed-fragment-gate-result.md`;
- `../../output/nv-compiler-packed-prefill-model-integration.md`.

Production HCQ kernels remain invisible to the existing CUPTI counter route.
No inferred hardware counter is substituted for that observation wall.
