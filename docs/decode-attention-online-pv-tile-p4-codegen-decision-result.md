# Decode Attention Online-Softmax+PV Tile P4 Codegen Decision Result

## Verdict

`ONLINE_PV_TILE_P4_NEEDS_DATAFLOW_REWRITE_BEFORE_CODEGEN`

P4 exhaustively checked the plausible reduction/dot/codegen targets after P3.

Artifact:

- `bench/qk-decode-attention-online-pv-p4-codegen-decision/latest.json`

Tool:

```bash
PYTHONPATH=. python3 extra/qk_decode_attention_online_pv_p4_codegen_decision.py
```

## Current Route Signature

```text
flash_score_whole_cache_32_128
flash_max_32
flash_online_pv_tile_whole_cache_32_128
flash_gmax_32
flash_den_32
flash_combine_32_128
```

## Lowerings Available In Repo

| Lowering | Available |
|---|---|
| `extra/qk_fdot2_lowering.py` | yes |
| `extra/qk_warp_reduce_lowering.py` | yes |
| `extra/qk_lane_partition_reduce.py` | yes |

## Exhaustive Target Matrix

| Target | Current owner | Bindable now? | Classification |
|---|---|---|---|
| score dot | `flash_score_whole_cache_32_128` | no | prior no-transfer as standalone score program |
| per-split `m` | `flash_max_32` | no | no in-tile lane-owned reduction site |
| online `l` / denominator | `flash_den_32` plus denominator lane contribution | no | partial contribution only, not online state |
| PV `acc[D]` | register `c[G]` in online tile | no | accumulator present but no cross-lane combine site |
| final combine | `flash_combine_32_128` | no | not next speed target; combine lever already refuted |
| LDS staging | none in generated online tile | no | requires dataflow/resource plan first |

## Interpretation

This is not a missing-file problem. The codegen lowerings exist, but the current P2/P3 route does not expose the right dataflow site.

Directly flipping lowerings now would either:

- rerun A3.1-style standalone score `v_dot2`, already no-transfer;
- attempt cross-lane reduction on external metadata programs, not the online tile;
- optimize combine, already refuted as the bounded lever;
- add LDS without a proven reuse target, which repeats the decode LDS trap.

## Decision

The next implementation must be a dataflow rewrite:

```text
flash_online_state_pv_tile_whole_cache_32_128
```

Required change:

- move per-split `m` update into the tile lifecycle;
- move online `l` update into the tile lifecycle;
- keep `Hkv*S` workgroups;
- keep `Hd+1` local lanes;
- preserve whole-cache identity and no `E_49152`;
- only then attach cross-lane and packed-dot lowerings to real in-tile sites.

Do not do next:

- another metadata-only fusion;
- standalone score `fdot2` rerun as the main path;
- combine-only optimization;
- blind LDS staging without online-state reuse target.
