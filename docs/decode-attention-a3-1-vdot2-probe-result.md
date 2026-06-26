# Decode Attention A3.1 v_dot2 Probe Result

## Verdict

`A3_1_RENDERER_VDOT2_PROBE_PASS`

Generated tinygrad code can expose the AMD `fdot2` builtin through the existing opt-in lowering hook.

## Artifact

- `bench/qk-decode-attention-a3-1-vdot2/latest.json`
- Tool: `extra/qk_decode_attention_a3_1_vdot2_probe.py`

## Checks

| Check | Result |
|---|---:|
| lowering file exists | yes |
| `V_DOT2_LOWERING` codegen hook exists | yes |
| builtin template exists | yes |
| `CUSTOMI` renderer support exists | yes |
| matcher rewrites dot2 pair | yes |
| generated smoke kernel ran | yes |
| debug source contains `__builtin_amdgcn_fdot2` | yes |

## Generated smoke evidence

The smoke kernel rendered:

```c
__builtin_amdgcn_fdot2(val0, val1, 0.0f, false)
```

It returned the expected scalar dot result:

```text
1*3 + 2*4 = 11
```

## Interpretation

A3.1 is not blocked at the renderer-hook level.

The next task is to wire the existing opt-in lowering into the A2 whole-cache score path:

```text
DECODE_ATTN_SCORE_VDOT2=1
flash_score_whole_cache_vdot2_32_128
```

Required gate remains:

- generated route
- no owned flash
- no `E_49152`
- tokens match
- generated score source/ISA shows `fdot2` / `v_dot2`
- W==D improves vs A2
