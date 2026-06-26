# Decode Attention Online-State+PV Tile P11 X-Lane Merge Result

## Verdict

`ONLINE_STATE_PV_P11_FAIL__MERGE`

P11 tested the synthetic x-lane online-softmax merge primitive directly, outside the full Q/K/V attention tile.

Artifact:

- `bench/qk-decode-attention-online-state-pv-p11-xlane-merge/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p11_xlane_merge.py
```

## Result

| Case | Generated | NumPy reference |
|---:|---:|---:|
| 0 | `0.826359748840332` | `0.08638609945774078` |
| 1 | `1.0` | `1.470588207244873` |
| 2 | `0.9999997615814209` | `0.9044954180717468` |
| 3 | `-2.127318859100342` | `-0.024110715836286545` |

Max absolute error:

```text
2.103208065032959
```

No NaNs were present.

## Interpretation

The x-lane failure is now isolated below the full attention tile.

The staged cross-lane merge sequence used by P7 does not currently compute the intended log-sum-exp merge in this generated UOp context:

```text
gm  = max_lane(m_lane)
w   = exp(m_lane - gm)
den = sum_lane(l_lane * w)
num = sum_lane(acc_lane * w)
out = num / den
```

Since the synthetic merge is wrong, do not debug P7 per-token state generation yet. The merge primitive itself must be fixed or replaced first.

## Decision

Next step:

```text
P12: repair or replace x-lane merge primitive
```

P12 should test smaller components independently:

| Component | Check |
|---|---|
| `warp_reduce_max(m)` | compare generated max vs NumPy max |
| `_warp_reduce_sum_staged(x)` | compare generated sum vs NumPy sum |
| gated store from `lane==0` | confirm stored lane receives full broadcast result |
| full LSE merge | compare full `num/den` formula |

Only after P12 passes should P7 be retried.
