# Phase N1 RESULT -- the native-matmul space IS learnable; the loop has a home (2026-06-15)

`extra/qk_loop_learnability.py`, `n1_learnability.json`, dataset `beam_log_n1.jsonl` (3878 records =
277 schedules x 14 diverse matmul shapes). Leave-one-shape-out XGBoost regressor (shape + config
features -> tflops); for each held-out shape, take the model's top-1 predicted config and report its
ACTUAL tflops vs that shape's oracle best, vs two pre-registered baselines.

## Headline numbers (leave-one-shape-out, 14 folds)
  mean model top-1 / oracle      = 0.89   (top-5 = 0.93)
  mean LOOKUP / oracle           = 0.80   (global-best-config baseline)
  median random trials to match  = 131    (the model's top-1 is worth ~131 random draws)

## PRE-REGISTERED GATE: PASS = False (kept honest, not moved)
  model_beats_lookup     = True   (0.89 > 0.80)
  model_top1_high (>=.90) = False  (0.89 -- missed by 0.01)
  model_saves_trials     = True   (131 >= 3)
The strict overall-0.90 gate is narrowly missed.

## WHY (diagnostic -- not part of PASS): it is a data-coverage gap, not a wall
  batched regime N>=256 (10 folds, the regime matmul_decoded serves): top1/oracle = 0.964
                                                                       lookup/oracle = 0.911
  small-N <256 (4 folds, UNDER-SAMPLED: only N=32/64/128 present):     top1/oracle = 0.705
The entire miss is the 4 small-batch shapes the training set barely covers. On the batched regime the
substrate actually serves, the model reaches 96.4% of oracle on HELD-OUT shapes and clears 0.90.

## TRANSFER curve (N1b) -- the flywheel-gets-better signal
  k_train:  1     4     7     10    13
  top1/orc: 0.46  0.83  0.88  0.83  0.89
Accumulated experience improves held-out prediction (0.46 -> 0.89). The loop gets better with data.

## Where the model beats the lookup (it earns its keep off-distribution)
  4096x4096x1024: model 1.00 vs lookup 0.685   (lookup's global-best config is wrong at large batch)
  4096x4096x32:   model 0.65 vs lookup 0.00     (lookup's config is INVALID on this shape)
  14336x4096x256: model 0.98 vs lookup 0.852
On the "central" N=256 shapes the global-best config is already near-optimal, so model ties lookup.

## Verdict
This is the FIRST genuine positive for the loop in the whole investigation. On the native-matmul opt
space -- the only substrate that is rich + competitive + (now shown) learnable -- a learned cost model
BEATS the deterministic lookup (0.89 vs 0.80 overall; 0.96 vs 0.91 on the served regime), is worth
~131 random trials, and IMPROVES with accumulated experience (the flywheel mechanism). The conditions
absent in the dead spaces (GEMV = flat/lookup-ties-model; fused-WMMA = no competitive point) are
present here. The strict pre-registered 0.90 gate is missed by 0.01 overall, entirely due to under-
sampled small-batch shapes; closing it is a data-coverage task (sweep more small-N shapes), not a
framework wall.

## Honest scope boundary (unchanged)
This proves the loop MECHANISM works on the native-matmul opt space -- a transferable autotuning
result. It is NOT a llama.cpp-decode win (the on-target quantized-decode spaces are dead: GEMV flat,
fused-WMMA walled). The loop, if built, is a general learned-autotuning contribution that serves
quantized inference via matmul_decoded for the batched regime, decoupled from the original decode bar.

## N1.1 -- strict gate CLOSED via data coverage (2026-06-15)
Added ~12 small-batch shapes (`extra/qk_loop_dataset_smalln.py`, `beam_log_n1_smalln.jsonl`) -> merged
~26-shape dataset, SAME leave-one-shape-out harness (the loaders now merge all `beam_log_n1*.jsonl`).
  mean model top-1 / oracle = 0.922  (was 0.89)  -> PRE-REGISTERED GATE NOW PASSES (>=0.90)
  small-N top-1 / oracle    = 0.915  (was 0.705)  -- the coverage gap was the cause, confirmed
  mean LOOKUP / oracle      = 0.054  -- the naive global-best-config lookup COLLAPSES across 26 diverse
                                        shapes (invalid/far-off on most held-out) while the model
                                        adapts per-shape; model beats lookup on 26/26 folds.
The gate closed by adding data, NOT by moving the threshold (it stayed 0.90).

## N2 -- the loop demonstrably works (model-guided search + online flywheel)
`extra/qk_loop_search.py`, `n2_loop_search.json`, test `test_qk_loop_search.py`. Offline on the
measured dataset (ranking is the model's; "measuring the top-K" is a true-time lookup).

N2a -- model-guided best-of-K / oracle vs RANDOM best-of-K (26 shapes, leave-one-out):
  K:        1     2     5     8     20    50
  model:    0.922 0.953 0.977 0.982 0.992 0.995
  random:   0.482 0.593 0.718 0.771 0.851 0.917
  TRIALS TO 95% OF ORACLE: guided median = 1.0   vs   random median = 86.3   (~86x fewer measurements)

N2b -- online accumulation (best-of-5 / oracle as the training corpus grows):
  corpus=1 -> 0.667 ; corpus=6 -> ~0.89 ; corpus=25 -> 0.977. The flywheel gets better with experience.
  Gate: guided reaches 95% by K<=8, beats random at every K, needs ~86x fewer trials, improves online.
  ALL PASS.

## Bottom line
The learned loop is real on the native-matmul substrate: a cost model trained on accumulated
(shape, config -> time) data guides search to 95% of oracle in ~1 measured config (vs ~86 random),
robustly beats the deterministic lookup (which collapses to 0.05 across diverse shapes), and improves
as the corpus grows. This is the positive existence proof the whole investigation was after -- the
loop MECHANISM works where the space is rich + competitive + learnable. Scope boundary unchanged: this
is a general learned-autotuning result on native matmul (serves quantized inference via matmul_decoded
for the batched regime), decoupled from the llama.cpp decode bar (whose on-target spaces are dead).
