# Decode integration diagnostic summary

Status: `LOCALIZED_NO_SINGLE_TAX`

Decode does not mirror the prefill Tensile finding as one clean layout tax. The ledger has four parts:

- stage2 partial reduce: measured `6.8us` / `10%` on the Q4_K ffn_gate/up surface; removing it only reaches `~53-54%` peak on that micro-surface.
- q8 activation lifecycle: max Q4_K activation reuse is `2`; useful but lossy and native-producer-walled.
- existing env knobs: `FAIL_B1_NO_ENV_KNOB_CLEARS_GATE`, no passing rows.
- MMVQ contract preservation: tinygrad `76%` standalone -> `44%` in-model, while llama `57%` -> `54%`.

Potential model:

- `44 -> 54%` over the weight-GEMV bucket: `1.187x` decode.
- `44 -> 60%`: `1.293x`.
- `44 -> 76%`: `1.557x` theoretical, not earned by current evidence.
