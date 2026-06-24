# Decode MMVQ large project P7c one-role route scope - 2026-06-19

Purpose: integrate the P7b graph-safe imported Q4_K MMVQ route into one real model role behind a research flag.

## Target

Route only `TransformerBlock.attn_output`:

- default off: `DECODE_MMVQ_IMPORT_Q4=0`;
- AMD only, Q4_K primitive storage only;
- decode only: `B=1`, `T=1`, hidden dim `4096`;
- persistent per-block q8/out side buffers;
- no Q6, no FFN, no default behavior change.

## Gates

1. Import path compiles and falls back silently if ineligible.
2. One-block graph route still passes.
3. One forward/decode smoke runs with the flag.
4. No default route changes when flag is unset.

If the flag route works, the next phase is timing + dNLL. If it faults or fails graph capture, stop at P7b and keep the
imported path as a probe-only primitive.
