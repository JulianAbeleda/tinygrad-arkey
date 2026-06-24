# Decode Parity / No-Regressions Audit — Result (2026-06-24)

## 1) Scope and decision

Goal was to determine whether current decode performance is a regression and whether any decode kernel/search work is required.

**Decision: `DECODE_HARNESS_RECONCILIATION_ONLY`**.

There is no new decode-kernel behavior change in this pass. The measured gap is explained by flag-stack variance between runs.

## 2) Authority lock

- HEAD: `2c937539b844ee82fdb273d91a9528ee014a73a4`
- Branch: `qk-prefill-flag-leak-resolution`
- GPU: `RX 7900 XTX / gfx1100`
- Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- Probe matrix:
  - `extra/qk_decode_route_fire_check.py`
  - `extra/qk_decode_materialization_check.py`
  - `extra/qk_decode_runtime_overhead.py`
  - `extra/qk_decode_time_tax_audit.py`

## 3) Artifact reconciliation (source-of-truth)

Two decode families are now in conflict by design:

- canonical flag-stack artifact (`Q4K_GEMV_WARP*` enabled): tinygrad above llama at measured ctx.
- fresh default run (`Q4K_GEMV_WARP*` not set, therefore default-off): tinygrad below llama at measured ctx.

Primary reconciliation output:

- `bench/qk-decode-parity-no-regression-audit/artifact_reconciliation.json`

## 4) Decode W==D by context

- Default config artifact: `bench/qk-decode-parity-no-regression-audit/wd_decode_by_ctx.json`

Required ctx sample:

- `A=DECODE_ATTN_KV_IDENTITY=1` tok/s: `89.2/87.5/85.7/82.2` at `512/1024/2048/4096`
- `B=DECODE_ATTN_KV_IDENTITY=0` tok/s: `77.0/76.5/75.8/74.1`

Route-comparator margin stays positive for A vs B.

## 5) Route and materialization proof

- `route_fire_by_ctx.json`
- `route`: default uses `owned_flash_tile_gqa_whole` and removes slice route
- `materialization`: default has no `E_49152`
- `DECODE_ATTN_KV_IDENTITY=0` has `E_49152_32_3` and `E_49152_32_3n1`

## 6) Time-tax

- `time_tax_by_ctx.json`
- Shared-tax picture at required ctxs: FFN gate-up + norm/rope small-ops are top shares, with attention path not the only dominant contributor.
- No single decode-only primitive dominates enough to explain a sub-2x delta.

## 7) Decision and next step

Current status is not a kernel correctness or route-firing issue.

Next action is explicit harness reconciliation:

1. Fix one canonical flag stack for all decode parity claims.
2. Re-run the same W==D matrix under that single stack.
3. If that canonicalized run is still below llama, proceed to bounded decode levers.

No project defaults, source files, or defaults were changed in this decode audit. 
