# Decode Attention Online-State+PV Tile P7 X-Lane Result

## Verdict

`ONLINE_STATE_PV_XLANE_FAIL__TOKEN_MISMATCH`

P7 introduced a token-sharded x-lane online-state+PV tile:

```text
flash_online_state_pv_tile_xlane_whole_cache_32_128
```

Artifact:

- `bench/qk-decode-attention-online-state-pv-xlane/latest.json`

Tool:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_online_state_pv_xlane_gate.py
```

## Route Signature

The intended generated route fired:

```text
flash_score_whole_cache_32_128
flash_online_state_pv_tile_xlane_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
```

Route hygiene:

| Check | Result |
|---|---|
| owned tile absent | pass |
| owned combine absent | pass |
| `E_49152` absent | pass |
| x-lane tile present | pass |
| state gmax/combine present | pass |
| external max/den absent | pass |
| old prob/partial stages absent | pass |
| token sample matches owned baseline | fail |

Owned token sample:

```text
[315, 24231, 6009, 979, 220, 576]
```

P7 token sample:

```text
[315, 119523, 119523, 313, 296, 296]
```

## What Was Tried

P7 changed the tile from P5's serial-per-`d` token loop to a token-sharded local-lane form:

| P5 | P7 |
|---|---|
| each `d` lane runs full serial token loop | local wave lanes shard token positions |
| no useful cross-lane reduction site | cross-lane merge sites for `m/l/acc[D]` exist |
| token-correct | token mismatch |

P7 also fixed an invalid-lane online-update issue during the attempt:

| Issue | Fix |
|---|---|
| empty token shards could form `-inf - -inf` in the correction factor | invalid shards now preserve prior `m/l/acc` and use correction `1.0` |

The fix did not resolve the token mismatch.

## Interpretation

P7 proves the next primitive boundary is now genuinely in the token-sharded online-softmax merge, not route selection.

The project has reached this state:

| Layer | Status |
|---|---|
| generated whole-cache lifecycle | solved |
| online `m/l/PV` state inside tile | solved structurally in P5 |
| token-sharded x-lane route identity | solved structurally in P7 |
| token-sharded online-softmax merge correctness | not solved |

The failure is likely one of:

| Possible cause | Why plausible |
|---|---|
| cross-lane online-softmax merge math bug | route fires and materialization is clean, but tokens diverge immediately |
| staged `ds_bpermute` interaction with the generated loop/store shape | x-lane route depends on staged cross-lane max/sum under generated UOp control |
| lane/store ownership mismatch | only `lane==0` stores merged state; if merge or gate is misplaced, partial state is wrong |

## Decision

Do not proceed to W==D or promotion.

Next required step is a small correctness microgate for the x-lane online-softmax merge outside the full model path:

```text
P8: isolated x-lane online-state tile numeric gate
```

P8 should compare P5 scalar-state tile vs P7 x-lane tile on a tiny deterministic attention problem and report max error for:

- per-split `m`;
- per-split `l`;
- per-split PV;
- final combine output.

Only after P8 is numerically correct should P7 be re-run in-model.
