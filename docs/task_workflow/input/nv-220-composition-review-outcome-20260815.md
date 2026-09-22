# NV 220 composition review outcome: the fold map does not validate against measurement

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `d40b938af`)
Status: **review record. Fresh production baseline measured; the composition
fold map reconciled against the per-fold GPU records. The composition claim is
NOT validated by measurement.**

This closes the review checkpoint named in
`nv-220-composition-scope-20260815.md` section 8: does the F1-F5 fold map
compose to ~557 us wall without adding body work? Answer, grounded in a fresh
run plus the already-booked per-fold A/B records: no, on the evidence available.

## 1. Fresh production baseline (this run)

`DEV=NV` decode census at d512 on Qwen3-8B-Q4_K_M / RTX 5090, production
policies, same harness as the attribution record:

| quantity | measured |
| --- | ---: |
| wall | **193.49 tok/s = 5.168 ms/token** |
| kernels/token (prime) | **596** |
| graph groups/token (replay) | 5 |
| token sha (3 reps) | `227ad3ce...` identical |

The composition scope pinned its baseline at 196 tok/s / 5.102 ms. That is
stale: the committed record `c56d33c14` already names 193.5 tok/s after the
ffn-norm site close, and this fresh run reproduces it (193.49). The real gap to
220 (4.545 ms) is therefore **~623 us**, not the ~557 us the composition scope
subtracted from 196.

## 2. Fold-map reconciliation

Each composition row is checked against the latest measured record, not against
the fold map's own census-to-wall assumption.

| # | composition row | composition claim | measured reality | validates? |
| --- | --- | --- | --- | --- |
| F1 | fp32 q/k reduce-output (240.5 us) | body-free removal into Q/K GEMV | BOOKED cooperative-body replacement, +38.7/+43.2 us wall (waived bar); the GEMV-absorption spelling is not shipped | partial |
| F2 | FFN-down reduce-output (151.5 us) | body-free removal into FFN-down GEMV | ffn-norm residual-bind body-free A/B correctness-clean but wall FLAT (NOT_PROMOTED, `nv-reduce-output-ffn-residual-bind-outcome-20260813.md`) | no |
| F3 | M1 norm chains (229.5 us) | reduce+epilogue -> one body | body-adding fold NO-GO +81.9 us; body-free primitive fold measured FLAT (same ffn residual-bind record) | no |
| F4 | attn K/V extras + residual (~247.6 us) | M4 typed-view residual fold | M2b/M2c/M2d residual/cast/contiguous exhausted at ~193 tok/s; M4 landed; remainder not a composed body-free removal | partial |
| F5 | vocab aux (57.3 us) | single-pass cross-tile max | two-program chain NO-GO -1.11%; single-pass mechanism not built | no |
| F6 | flash score (166.5 us) | out of scope (structural) | still out of scope; structural floor 90 us | n/a |

## 3. Why the composition arithmetic fails

The composition scope priced F1-F5 at ~926 us census and applied a ~0.6
census-to-wall map to reach 557 us. The measured body-free norm folds do not
behave that way. The ffn-norm residual-bind removed 199 kernels and was
bitwise-exact, but the fused bodies carried enough per-launch cost to make wall
FLAT: candidate +10.68 us vs bracket median but -3.30 us vs control A, inside
noise, NOT_PROMOTED. That is a body-free removal mapping to ~0 wall, not 0.6-1.0.

The only booked wall gain in the F1-F5 map is F1 (fp32 q/k), and even that
booked below the +50 us bar. The "all class deltas at 1:1" step 7 of the
08-12 ladder (`219.8 tok/s`) is an unverified ceiling, not a composed result.

## 4. What IS validated

- The primitive substrate (reduce-output, typed-view admission, epilogue
  authoring) renders and is correctness-clean; the body-free construction is
  expressible.
- Exact-output gates hold bitwise across every fold tried.
- Kernel-count removal is real (F3 body-free removed 199 kernels).

What is NOT validated: that removing those kernels removes wall. The measured
norm folds are launch-cost-bound, and the fused body's own launch cost cancels
the removal.

## 5. Next measured step

Do not re-litigate F3 (both the body-adding and body-free spellings are
measured). The remaining rows with an unmeasured wall question are F2's
FFN-down-side absorption and F5's single-pass vocab max; the structural
flash-score floor (F6) is the largest single untouched item and sits on the
critical path. Any further 220 claim must come from a measured composed wall
bracket, not from extending the 0.6/1.0 census map.

## Evidence

- fresh census: `/tmp/census_prod_20260815.json` (this run, git `d40b938af`)
- `nv-reduce-output-ffn-residual-bind-outcome-20260813.md` (F2/F3 body-free
  FLAT, 199 kernels removed, NOT_PROMOTED)
- `nv-m1-norm-epilogue-generic-primitive-scope-20260812.md` (M1 body-adding
  NO-GO +81.9 us; body-free reopen contract)
- `nv-decode-gap-attribution-same-session-20260812.md` (ladder step 7 = 219.8
  tok/s at 1:1, explicitly labeled a ceiling)
- `nv-vocab-top1-fusion` A/B and `nv-220-composition-scope-20260815.md` (the
  scope under review)
