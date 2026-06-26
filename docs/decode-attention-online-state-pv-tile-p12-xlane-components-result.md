# Decode Attention Online-State+PV Tile P12 X-Lane Components Result

## Verdict

`ONLINE_STATE_PV_P12_FAIL__MAX`

P12 decomposed the x-lane merge into component tests:

- `warp_reduce_max(m)`;
- `_warp_reduce_sum_staged(x)`;
- denominator sum;
- full LSE output.

Artifact:

- `bench/qk-decode-attention-online-state-pv-p12-xlane-components/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_state_pv_p12_xlane_components.py
```

## Result

| Component | Max error |
|---|---:|
| max | `7.53878927230835` |
| sum | `2.7978341579437256` |
| denominator | `18.178207397460938` |
| LSE output | `1.0957602262496948` |

No NaNs were present.

## Interpretation

The failure is below the full LSE formula.

The generated component kernel does not reliably broadcast the correct cross-lane max/sum values for this shape. Some cases match, but others return lane-local or otherwise partial values. That explains P11 and P10: the online-softmax merge is fed by incorrect cross-lane component reductions.

A follow-up attempt to rewrite the component kernel as multiple gated stores from a single global axis hit a UOp verification failure on `Ops.AFTER` with multiple stores. That was not pursued further in P12.

## Decision

Do not retry P7 or W==D.

The blocker is now:

```text
Cross-lane reduction helper/store composition is not reliable for the decode-attention generated UOp shape.
```

Next step:

```text
P13: define a safer attention-local cross-lane reduction/store primitive or park x-lane token sharding as SEARCH_BLOCKED_BY_CODEGEN.
```

P13 should choose one:

| Option | Meaning |
|---|---|
| repair | build a tiny attention-local reducer that emits one output per component with a verified store contract |
| park | classify P7 as `SEARCH_BLOCKED_BY_CODEGEN` because the current generated UOp/store machinery cannot express the required cross-lane merge safely |
