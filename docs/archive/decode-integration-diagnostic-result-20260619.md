# Decode integration diagnostic result - 2026-06-19

Purpose: run the prefill-style localization pass for decode after the Tensile diagnostic found a clean
layout-conversion tax. This pass does not build kernels and does not change routes; it consolidates the already
measured decode evidence into a tax ledger and potential model.

Artifacts:

- `extra/qk_decode_integration_diag.py`
- `bench/qk-decode-integration-diagnostic/tax_ledger.json`
- `bench/qk-decode-integration-diagnostic/potential.json`
- `bench/qk-decode-integration-diagnostic/result.json`
- `bench/qk-decode-integration-diagnostic/summary.md`

## Verdict

`LOCALIZED_NO_SINGLE_TAX`.

Decode does **not** have the prefill shape where one layout transpose cancels one fast kernel. The measured local tax
exists, but it is too small:

| tax / limiter | measured state | decision |
|---|---|---|
| Q4_K global partials + stage2 reduce | `6.8us`, `10%` of the Q4_K ffn_gate/up surface; removing it only reaches `~53-54%` peak | real but insufficient; closed as standalone route |
| q8 activation lifecycle | max Q4_K activation reuse is `2` (`gate+up`); expected decode EV `~3-6%` | useful but lossy and native-producer-walled |
| existing launch-shape env knobs | FMI-4 B1: no role reaches `>=1.10x` relative movement | closed |
| MMVQ in-model contract preservation | tinygrad `76%` standalone -> `44%` in-model; llama `57%` -> `54%` | live, project-level |

So the decode analogue to the prefill layout tax is a **contract-preservation problem**, not a single conversion pass:
preserve llama-like activation format reuse plus low-VGPR, high-grid MMVQ occupancy inside the model.

## Potential

Using the measured weight-GEMV share (`~85%`) and the authority aggregate (`44%` current in-model HBM):

| target | modeled decode speedup | interpretation |
|---|---:|---|
| `44% -> 54%` | `1.187x` | close tinygrad to llama-like in-model retention |
| `44% -> 60%` | `1.293x` | better-than-llama retention, still below tinygrad standalone |
| `44% -> 76%` | `1.557x` | theoretical full standalone transfer; not earned by current evidence |

The realistic bounded wins remain smaller: stage2-only is low single digits, and q8 gate/up is `~3-6%` with quality
and producer-codegen caveats. The large win requires preserving the MMVQ contract in-model.

## Next

Do **not** keep tuning `Q4K_COOP_RT` / `Q6K_COOP_RT`; FMI-4 B1 already closed that surface.

The next bounded diagnostic is **B2 runtime/cache identity**:

1. Prove whether the in-model route uses the intended compiled program, specialization metadata, and graph-safe launch
   identity for the target role.
2. Compare that program identity to the standalone fast surface and the role kernel selected in-model.
3. If B2 finds no wiring/cache bug, stop treating this as a small primitive edit and choose between:
   - B3 renderer/scheduler project: native low-VGPR, high-grid MMVQ contract preservation;
   - B4 artifact/import: import a mature MMVQ family if one exists.

q8 replay stays secondary because it is lossy and activation reuse is capped at `2`.
