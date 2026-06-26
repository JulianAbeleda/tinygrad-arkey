# Decode Attention Online-State+PV Tile P8 Numeric Scope

## Goal

Isolate the P7 token mismatch outside the full model.

P7 route binding worked, but tokens diverged. P8 compares the P5 scalar-state tile against the P7 x-lane token-sharded tile on deterministic tensors and reports where the numeric mismatch appears:

- per-split `m`;
- per-split `l`;
- per-split PV;
- final combine output.

## Why this is required

P7 proved the generated x-lane route can compile and fire without owned attention or `E_49152`, but correctness failed:

```text
owned: [315, 24231, 6009, 979, 220, 576]
P7:   [315, 119523, 119523, 313, 296, 296]
```

That means the next issue is numeric/dataflow correctness inside the x-lane online-softmax merge, not route binding.

## Tool

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p8_numeric.py
```

## Gate

P8 passes only if scalar P5 and x-lane P7 match within tolerance on deterministic data:

| Quantity | Tolerance |
|---|---:|
| per-split `m` | `1e-4` |
| per-split `l` | `1e-4` |
| per-split PV | `2e-3` |
| final output | `2e-3` |

## Expected verdicts

```text
ONLINE_STATE_PV_P8_NUMERIC_PASS
ONLINE_STATE_PV_P8_FAIL__M
ONLINE_STATE_PV_P8_FAIL__L
ONLINE_STATE_PV_P8_FAIL__PV
ONLINE_STATE_PV_P8_FAIL__OUT
ONLINE_STATE_PV_P8_FAIL__CAPTURE
```

## Next decision

- If `m` fails: fix cross-lane max merge.
- If `l` fails: fix online denominator merge/rescale.
- If PV fails but `m/l` pass: fix `acc[D]` rescale/sum merge.
- If only final output fails: fix `flash_state_combine` use of state columns.
- If P8 passes: rerun P7 in-model gate, then only after that do W==D/ISA attribution.
