# Phase N0b -- native-matmul opt-space characterization (2026-06-15)

`extra/qk_beam_log.py`, `beam_log.jsonl` (1385 records = 277 candidate schedules x 5 shapes),
`n0b_summary.json`. Each record: (matmul shape, opt schedule, device_us, tflops, valid). The opt
space is TC + UPCAST/LOCAL/UNROLL combos -- the same space tinygrad BEAM searches.

## Landscape (per shape)
  shape            valid  best_TF  worst_TF  spread   within-10%-of-best
  4096x4096x64     187    15.2     0.14      111x     10
  4096x4096x256    262    26.1     0.16      163x      3
  4096x4096x512    274    24.3     0.11      223x      9
  4096x14336x256   262    21.9     0.12      178x      4
  14336x4096x256   262    27.4     0.13      214x      2

## The three properties that matter for the loop
1. RUGGED: 111-223x spread between best and worst valid config -> config choice matters enormously
   (the OPPOSITE of the flat Q4_K GEMV space, where opts barely moved the device metric).
2. SHARP optima: only 2-10 configs (of ~250) within 10% of best -> the good region is narrow; a
   selector that finds it is worth a lot.
3. NO universal winner: 0 configs are in the top-5 of ALL 5 shapes. Each shape's best config is
   often RANK 130-211 (near-worst) on the others. So a deterministic "always use config X" lookup
   FAILS -- unlike the GEMV space where a lookup tied the model.
4. BUT structured, not random: configs cluster by shape-FAMILY. The 4096x4096x256 best ranks 1/4/3/4
   across the four "attn-shaped" matmuls (transfers within family) but rank 144 on the N=64 shape.
   So shape features predict good configs.

## Implication for N1 (the loop's make-or-break)
This is the strongest positive signal for the loop in the whole investigation. The native-matmul
space is rich (rugged) + has no lookup solution (no universal winner) + has learnable structure
(family clustering). These are exactly the conditions under which a learned cost model / cross-kernel
transfer CAN beat a deterministic baseline -- the conditions ABSENT in the GEMV space (flat) and the
fused-WMMA space (no competitive point). N1 is now well-motivated: train a cost model on some shapes'
(config->time) logs, test whether it predicts good configs on held-out shapes with fewer trials than
BEAM, and whether accumulated experience transfers across kernels (the flywheel).

Pre-registered for N1: deterministic baselines to beat = (a) best-single-config (the lookup that
fails here), (b) BEAM's own trial count. If a cost model cannot beat those, the structure seen here
is not exploitable and the loop thesis closes even on its best substrate.
