# System Fusion SF4 — Ledger Report

Date: 2026-07-01. Covers SF0-SF4 for the 14B/32B decode system-fusion track.

## Fragment resolved

SF0 identified 14 kernels in the "other" elementwise bucket at ctx512 (12.76% of GPU time). All 14 were attributed to named fragment classes with known producers and consumers. Unknown% = 0.0%.

The bucket was dominated by E_49152_32_3 (6.69%, attention_elementwise, EMITTER_BLOCKED) and the silu_gate pair (1.26%, REACHABLE_NOW).

## Candidate selected (SF2): decode_silu_gate_fusion

- **Root cause**: `.contiguous()` on model.py:1017 (marked `# TODO: remove the need for this contiguous`) forces materialization of silu(gate) as a standalone kernel before the gate×up multiply.
- **Fix**: remove `.contiguous()` behind `DECODE_FUSE_SILU_GATE` flag (default-off).
- **Amdahl**: 1.26% at ctx512 (E_136_32_4 + E_136_32_4n1, 40 launches × 2 kernels → 40 launches × 1 kernel).
- **Why selected**: cleanest REACHABLE_NOW path — one-line change, no new primitive, no handwritten kernel, root cause documented.

## SF3 result

| | ctx128 | ctx512 |
|--|--------|--------|
| baseline | 52.3 tok/s | 50.0 tok/s |
| fused | 52.4 tok/s | 50.4 tok/s |
| delta | +0.1 (+0.2%) | +0.4 (+0.8%) |

**Correctness: PASS** (rel_rmse=0.00e+00, pearson=1.0, identical logit vectors).

**W==D verdict: SF3_LOW_AMDAHL_NO_MOVEMENT** — ctx512 delta 0.4 tok/s < 0.5 tok/s significance threshold.

## SF4 decision: DEFER_NOT_DEFAULT_ON

`decode_silu_gate_fusion` is correct and the code change is kept in the repo. It is NOT promoted to default-on because the W==D improvement (0.4 tok/s) is within noise.

**do_not_retry: False** — the root cause is real, the correctness is proven. The lever is deferred, not refuted.

## What this teaches about the system-fusion track

The elemental silu_gate fusion (1.26% Amdahl) is too small to move the W==D needle individually. The ~12% elementwise bucket is the correct aggregate target but needs:
1. A broader multi-group fusion pass covering all REACHABLE_NOW fragments simultaneously (~4.9% aggregate Amdahl)
2. Or a scheduler capability that naturally fuses post-GEMV epilogue ops (residual add, rmsnorm scale, silu gate) without requiring per-op flag proliferation

Individual piece by piece (1.26%, 1.28%, 0.90%, 1.46%) each falls below the noise floor. Aggregated (~4.9%), the improvement should be detectable.

## Updated BoltBeam status

`decode_silu_gate_fusion`: status=refuted_no_movement, do_not_retry=False, evidence=bench/system-fusion-sf3/latest.json.

## Reopen condition (precise)

1. **Aggregate multi-fusion pass**: a scheduler change that fuses silu_gate + rmsnorm_scale × 2 + residual_adds + qk_norm_scale into a single pass (aggregate ~4.9% Amdahl → ~2.4 tok/s at ctx512). This is the only reopen path worth pursuing now.
2. **E_49152_32_3 (6.69%)**: EMITTER_BLOCKED. Reopens if a scheduler/UOp primitive allows elementwise→flash_reduce fusion without a global barrier. Not currently achievable.
3. **E_5_2_2_16_4_4n1 (1.46%, qk_norm_scale)**: REACHABLE_NOW alone but same Amdahl problem. Bundle with the aggregate pass.

## Current frontier

The system-fusion track has mapped the full 12.76% elementwise bucket (SF0_PASS). The REACHABLE_NOW total is 4.90% (silu_gate 1.26% + rmsnorm_scale 1.28% + residual_adds 0.90% + qk_norm_scale 1.46%). The 6.69% EMITTER_BLOCKED E_49152 is the remaining locked piece.

Next action: design a multi-group fusion scheduler pass rather than individual levers.
