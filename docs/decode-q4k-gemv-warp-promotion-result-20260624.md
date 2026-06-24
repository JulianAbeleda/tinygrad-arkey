# Decode Q4K GEMV Warp Promotion Result (2026-06-24)

## Decision

`DECODE_PROMOTE_Q4K_GEMV_WARP_FFN`

`Q4K_GEMV_WARP` and `Q4K_GEMV_WARP_DOWN` are now default-on for the guarded Qwen3-8B Q4_K FFN decode shapes on gfx1100.

## Authority

Primary prior authority:

- `docs/decode-ffn-gemv-warp-result-20260622.md`
- `docs/q4k-gemv-warp-promotion-hardening-result-20260622.md`
- `docs/decode-ctx-slope-audit-result-20260623.md`
- `bench/qk-decode-parity-no-regression-audit/artifact_reconciliation.json`

Core promotion evidence:

| route | ctx512 | ctx1024 | ctx4096 | correctness |
|---|---:|---:|---:|---|
| `Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1` | +9.75% | +9.63% | +8.55% | byte-identical |

Canonical decode stack evidence:

| ctx | tinygrad canonical | llama | tinygrad vs llama |
|---:|---:|---:|---:|
| 512 | 104.0 | 97.71 | +6.5% |
| 1024 | 102.1 | 97.39 | +4.8% |
| 2048 | 99.6 | 95.0 | +4.8% |
| 4096 | 95.1 | 92.37 | +3.0% |

Default-off artifact:

| ctx | tinygrad default-off | llama | tinygrad vs llama |
|---:|---:|---:|---:|
| 512 | 89.2 | 97.71 | -8.8% |
| 1024 | 87.5 | 97.39 | -10.2% |
| 2048 | 85.7 | 95.0 | -9.8% |
| 4096 | 82.2 | 92.37 | -11.1% |

## Code Change

Changed `tinygrad/llm/model.py`:

- `Q4K_GEMV_WARP` now defaults to enabled for guarded FFN gate/up.
- `Q4K_GEMV_WARP_DOWN` now defaults to enabled for guarded Q4_K FFN down.
- `Q4K_GEMV_WARP_PROJ` remains default-off/research-only because the promotion hardening audit found local speedup but no W==D transfer.

Escape hatches:

```bash
Q4K_GEMV_WARP=0
Q4K_GEMV_WARP_DOWN=0
```

## Rationale

The prior decode parity gap was a flag-stack authority mismatch: canonical warp-enabled artifacts are above llama,
while default-off artifacts are below llama. The promoted FFN warp route is lossless, guarded to validated shapes/arch,
and fallback-safe.

Decode attention defaults are unchanged.
