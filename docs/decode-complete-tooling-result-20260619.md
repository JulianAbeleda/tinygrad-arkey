# Decode Complete Tooling Result

Date: 2026-06-19

Scope:

- `docs/decode-complete-tooling-scope-20260619.md`

Artifacts:

- Probe: `extra/qk_decode_complete_tooling.py`
- Directory: `bench/qk-decode-complete-tooling/`
- Schema: `bench/qk-decode-complete-tooling/schema.json`
- Inventory: `bench/qk-decode-complete-tooling/instrument_inventory.json`
- Role atlas: `bench/qk-decode-complete-tooling/role_atlas.json`
- Q6 equivalence: `bench/qk-decode-complete-tooling/q6_capture_equivalence.json`
- Timing policy: `bench/qk-decode-complete-tooling/timing_audit.json`
- ATT metrics: `bench/qk-decode-complete-tooling/att_metrics.json`
- llama join: `bench/qk-decode-complete-tooling/llama_join.json`
- Reduce/glue ledger: `bench/qk-decode-complete-tooling/reduce_glue_ledger.json`
- Summary: `bench/qk-decode-complete-tooling/summary.md`

## Verdict

`COMPLETE_TOOLING_PASS_WITH_EXPLICIT_GAPS`.

DCT-0 through DCT-7 are now artifact-backed. The tooling is complete enough to prevent the wrong next build:

```text
Do not fund direct-output/reduce-fusion from ATT visibility alone.
```

The visible reduce/glue lifecycle is real, but the currently priced tax does not clear the build gate. The next decode
implementation choice remains either:

1. keep the measured q8 fused artifact as a default-off research route; or
2. fund project-level native scheduler/renderer work.

## Phase Results

| Phase | Result | Artifact |
|---|---|---|
| DCT-0 schema/inventory | PASS | `schema.json`, `instrument_inventory.json` |
| DCT-1 Q6 capture/equivalence | PASS for visibility, not timing | `q6_capture_equivalence.json` |
| DCT-2 multi-role atlas | PASS with explicit ffn_gate/up ATT gap | `role_atlas.json` |
| DCT-3 timing policy | PASS | `timing_audit.json`, `timing_policy.md` |
| DCT-4 ATT metrics | PASS | `att_metrics.json` |
| DCT-5 reduce/glue Amdahl ledger | NO BUILD GATE | `reduce_glue_ledger.json` |
| DCT-6 llama join | PASS | `llama_join.json` |
| DCT-7 final atlas | PASS | `result.json`, `summary.md` |

## Role Coverage

| Role | Capture | Verdict |
|---|---|---|
| `attn_q/o` | full in-model activation ATT | `PASS_INMODEL_ROLE_JOIN_NATIVE_Q4K_COOP` |
| `ffn_down` | Q6 surface fallback ATT | `PASS_INMODEL_ROLE_JOIN_NATIVE_COOP` |
| `lm_head` | Q6 surface fallback ATT | `PASS_INMODEL_ROLE_JOIN_NATIVE_COOP` |
| `ffn_gate/up` | runtime/cache identity only | `PASS_RUNTIME_IDENTITY_ATT_MISSING` |
| `attn_k/v` | runtime/cache identity only | `PASS_RUNTIME_IDENTITY_ATT_MISSING` |

The Q6 boundary is explicit: full model activation capture still fails on the 4.68 GB AMD allocation issue, but
runtime/cache identity already saw the same Q6 programs in-model. Therefore the Q6 surface fallback is acceptable for
program/lifecycle visibility, not as a timing promotion authority.

## Timing Policy

The timing policy is now explicit:

- ATT/SQTT packet counts are visibility evidence, not timing evidence.
- Same-process interleaved A/B is acceptable for role-local yes/no gates.
- Full W==D ctx sweep is required for final decode promotion.
- Q6 surface fallback cannot promote a timing build by itself.
- Non-interleaved clock-confounded timing remains provenance only.

This matters because the ATT atlas makes reduce/glue visible, but visibility is not a speedup.

## Reduce/Glue Ledger

The current priced reduce/glue authority is still the decode integration diagnostic:

| Tax | Value | Decision |
|---|---:|---|
| Q4_K partials plus stage-2 reduce | `6.8us`, about `10%` of the Q4_K ffn_gate/up surface | real but insufficient |
| Best case if removed | about `53-54%` peak | still below llama-class retention |
| Build gate | `>=5%` W==D or `>=10%` high-share local movement | not cleared |

The role atlas shows reduce/glue is present in Q4 and Q6 lifecycles, but the only priced evidence does not justify a
direct-output/reduce-fusion implementation. Reopen only if the timing policy prices a larger cross-role reduce/glue
share.

## llama Join

The llama join puts the comparison into the same tooling frame:

- llama MMVQ decode share: `73.4%`;
- llama MMVQ effective bandwidth: `626 GB/s`, about `70%` HBM peak;
- dominant contract: q8_1 activation producer plus low-VGPR `wg32` MMVQ consumers;
- captured Q4/Q6 kernargs: `144` bytes, with real grids for Q4, Q6 ffn_down, and Q6 lm_head.

This reinforces the current decode story: tinygrad's issue is not a hidden fallback. It is preserving the MMVQ
lifecycle contract in-model.

## Remaining Gaps

- Fresh ATT body attribution for `ffn_gate/up` is still missing. Runtime identity exists, but the exact high-share role
  has not been body-traced.
- Full-model Q6 activation capture remains blocked by the 4.68 GB AMD allocation issue.
- There is still no reliable per-kernel graph replay timing authority.

These are tooling gaps, not evidence for a bounded kernel build.

## Decision

Decode tooling is now complete enough to answer the immediate question:

```text
No, reduce/glue visibility does not by itself justify the next build.
```

The measured decode paths remain:

- q8 fused artifact route: default-off research flag, about `1.05-1.06x` W==D, dNLL `+0.002887`;
- native transfer: project-level AMD scheduler/renderer work;
- direct-output/reduce-fusion: closed unless a new timing-grade ledger clears the build gate.

