# Decode Attention Online-State+PV Tile P8 Numeric Result

## Verdict

`ONLINE_STATE_PV_P8_FAIL__NAN`

P8 isolated the P7 token mismatch outside the full model by comparing P5 scalar-state tile output against P7 x-lane tile output on deterministic tensors.

Artifact:

- `bench/qk-decode-attention-online-state-pv-p8-numeric/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p8_numeric.py
```

## Shape

| Field | Value |
|---|---:|
| `Hq` | 32 |
| `Hkv` | 8 |
| `Hd` | 128 |
| `MAXC` | 512 |
| `L` | 64 |
| `Tc` | 128 |
| `Sval` | 2 |
| `W` | 130 |

## Result

| Quantity | Error / status |
|---|---:|
| `m_max_abs` | 0.03939542919397354 |
| `l_max_abs` | 115589.5546875 |
| `pv_max_abs` | 112063.359375 |
| `out_max_abs` | 1.0282398462295532 |
| scalar has NaN | true |
| x-lane has NaN | true |

## Interpretation

P8 found the numeric failure before the full-model route gate.

The intended x-lane route fired in P7, but P8 shows the isolated online-state tile path is not numerically clean. Because both scalar and x-lane isolated paths report NaNs, the immediate blocker is not only x-lane merge. The online-state tile recurrence or custom-kernel compilation/cache path needs a smaller scalar numeric proof before x-lane can be trusted.

During P8, both scalar and x-lane tile code were patched to preserve old state on invalid token shards rather than computing `-inf - -inf`. The deterministic microcase was then changed to `Tc=128`, `L=64` so active splits have no invalid tail tokens. NaNs still appeared.

## Decision

Do not rerun P7 in-model or W==D yet.

Next required step:

```text
P9: scalar online-state tile standalone numeric proof
```

P9 should remove x-lane entirely and compare:

- score buffer from generated score kernel;
- scalar `flash_online_state_pv_tile_whole_cache_*` output columns `PV/l/m`;
- direct NumPy reference for the same deterministic q/cache/score inputs.

Only after scalar P5 numeric output is proven clean should x-lane P7 be debugged again.
