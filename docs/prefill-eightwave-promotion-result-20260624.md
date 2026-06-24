# Prefill Eightwave Promotion Result (2026-06-24)

## Decision

`PREFILL_PROMOTE_EIGHTWAVE_ONLY`

`PREFILL_GEMM_8WAVE` is now promoted as the default prefill graph-GEMM emit layout for tile-divisible roles.

## Authority

Promotion artifact:

- `/tmp/prefill-emits/emit-search-20260624-112043.json`
- `/tmp/prefill-emits/emit-search-20260624-112043.md`
- `/tmp/prefill-emits/emit-search-20260624-112043.csv`

Command scope:

- candidates: `baseline_current_default`, `eightwave`, `eightwave_old_plra`
- contexts: `512,1024,2048,4096,8192`
- strict mode: enabled
- confirm: `--confirm-k 1 --confirm-repeats 6`

## Result

| candidate | 512 | 1024 | 2048 | 4096 | 8192 | decision |
|---|---:|---:|---:|---:|---:|---|
| `eightwave` | +3.2% | +3.0% | +2.8% | +2.4% | +1.9% | promote |
| `eightwave_old_plra` | -10.3% | -10.3% | -9.5% | -8.3% | -6.7% | reject |

Confirm block for `eightwave`:

| ctx | confirm delta |
|---:|---:|
| 512 | +3.10% |
| 1024 | +2.84% |
| 2048 | +2.67% |
| 4096 | +2.28% |
| 8192 | +1.85% |

## Code Change

Changed `extra/qk_prefill_graph_gemm_route.py`:

- default enables the 8-wave layout when no explicit emit-style override is set
- `PREFILL_GEMM_8WAVE=0` disables the new default
- explicit `PREFILL_GEMM_CFG_*`, `PREFILL_GEMM_DBUF`, `PREFILL_GEMM_PLRA`, or `PREFILL_GEMM_PLRAB` overrides suppress default eightwave unless `PREFILL_GEMM_8WAVE=1` is also explicitly set

This preserves the old route escape hatch:

```bash
PREFILL_GEMM_8WAVE=0 PREFILL_GEMM_DBUF=0 PREFILL_GEMM_PLRA=1
```

## Rationale

`eightwave` is consistently positive across the full long-context envelope and passes confirm. The combined
`eightwave_old_plra` path is feasible but significantly regresses all contexts, so promotion is limited to
`eightwave` alone.

Decode is unchanged.
