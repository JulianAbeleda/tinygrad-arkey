# Decode Attention Online-State+PV Tile P10 X-Lane Output Result

## Verdict

`ONLINE_STATE_PV_P10_FAIL__XLANE_REF`

P10 compared final outputs only, avoiding raw intermediate state-column assertions.

Artifact:

- `bench/qk-decode-attention-online-state-pv-p10-xlane-output/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p10_xlane_output.py
```

## Result

| `Tc` | scalar vs NumPy | x-lane vs NumPy | x-lane vs scalar | Verdict |
|---:|---:|---:|---:|---|
| 128 | `2.24e-08` | `1.0107` | `1.0107` | fail |
| 130 | `3.73e-08` | `1.0107` | `1.0107` | fail |
| 32 | `3.73e-08` | `1.0511` | `1.0511` | fail |
| 256 | `1.49e-08` | `0.9974` | `0.9974` | fail |

No NaNs were present in scalar, x-lane, or NumPy final outputs.

## Interpretation

This cleanly localizes the current blocker.

Scalar online-state output is correct. X-lane output is finite but wrong by about `1.0` across all cases.

So the problem is not:

- scalar recurrence;
- final combine kernel in the scalar path;
- NaN/tail handling;
- route binding;
- materialization.

The problem is the P7 x-lane final-output merge:

```text
flash_online_state_pv_tile_xlane_whole_cache_32_128
```

Likely bug class:

| Candidate | Why |
|---|---|
| lane-sharded online-softmax merge formula | x-lane partials are finite but merge to wrong output |
| staged cross-lane sum/max usage | scalar is correct, x-lane only changes cross-lane merge |
| lane/store ownership | only `lane==0` stores merged state; wrong lane value or gated merge would corrupt output |
| per-lane token-shard recurrence | each lane computes only a token shard; local recurrence may be correct but global merge wrong |

## Decision

Next step:

```text
P11: x-lane merge microproof
```

P11 should avoid the full Q/K/V path and test the online merge itself:

- construct synthetic per-lane partial states `(m_lane, l_lane, acc_lane[D])`;
- merge them with the same staged cross-lane functions used by P7;
- compare against NumPy merge formula;
- then apply the fix to `flash_online_state_pv_tile_xlane_whole_cache_32_128`.
