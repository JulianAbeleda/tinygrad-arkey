# Decode Attention Online-State+PV Tile P5 Result

## Verdict

`ONLINE_STATE_PV_TILE_STRUCTURAL_ROUTE_CLEAN`

P5 rewrites the structural online-PV tile so per-split `m` and `l` live in the tile output lifecycle.

Artifact:

- `bench/qk-decode-attention-online-state-pv-tile/latest.json`

Tool:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_attention_online_state_pv_tile_gate.py
```

## Candidate

```text
flash_score_whole_cache_32_128
flash_online_state_pv_tile_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
```

Flag:

```text
DECODE_ATTN_ONLINE_STATE_PV_TILE=1
```

## Gate Result

| Check | Result |
|---|---|
| owned tile absent | pass |
| owned combine absent | pass |
| `E_49152` absent | pass |
| token sample matches owned baseline | pass |
| `flash_online_state_pv_tile_whole_cache_32_128` present | pass |
| `flash_state_gmax_32_128` present | pass |
| `flash_state_combine_32_128` present | pass |
| external `flash_max_32` absent | pass |
| external `flash_den_32` absent | pass |
| old `flash_prob_32` absent | pass |
| old partial-PV stages absent | pass |
| P2 `flash_online_pv_tile_whole_cache_32_128` absent | pass |

Token sample matched owned baseline:

```text
[315, 24231, 6009, 979, 220, 576]
```

## What Changed

P5 changes the online tile output width from `Hd+1` to `Hd+2`:

| Column | Meaning |
|---:|---|
| `0..Hd-1` | unnormalized PV accumulator |
| `Hd` | per-split denominator `l` |
| `Hd+1` | per-split max `m` |

This removes the separate external per-split max and denominator stages from the generated attention lifecycle.

## Current Route Signature

Before P5 structural rewrite:

```text
flash_score_whole_cache_32_128
flash_max_32
flash_online_pv_tile_whole_cache_32_128
flash_gmax_32
flash_den_32
flash_combine_32_128
```

After P5 structural rewrite:

```text
flash_score_whole_cache_32_128
flash_online_state_pv_tile_whole_cache_32_128
flash_state_gmax_32_128
flash_state_combine_32_128
```

## Interpretation

P5 completes the dataflow rewrite required by P4.

The generated route now has a real in-tile online-state site for `m/l/PV` lifecycle work. This is still not a speed promotion claim. It creates the correct structural target for the next lowerings.

## Next Step

Proceed to P6: bind lowerings to real in-tile sites.

P6 should test, in order:

| Target | Purpose |
|---|---|
| cross-lane max/add over online state | determine whether `m/l` can use generated cross-lane reduction |
| packed-dot/`v_dot2` score direct-fusion path | determine whether score can move closer to the tile lifecycle |
| resource/ISA audit | prove emitted instructions or classify `SEARCH_BLOCKED_BY_CODEGEN` |

Do not promote P5 until W==D proves transfer.
