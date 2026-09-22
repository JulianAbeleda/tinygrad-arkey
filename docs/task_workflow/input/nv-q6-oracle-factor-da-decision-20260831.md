# NV Q6 oracle dA-factor decision

## Test

The A/B implements the pinned llama arithmetic grouping while holding the
`128x128xK256` topology, canonical loads, 76-word Q6 shared layout, Q8 layout,
256 `IMMA`, 32 `LDSM`, four barriers and output mapping fixed.

Control:

```text
sum += dA * dB[p] * integer_dot[p]   for p=0..3
```

Candidate:

```text
tmp += dB[p] * integer_dot[p]        for p=0..3
sum += dA * tmp
```

This is the actual llama reuse axis.  `dA*dB[p]` cannot be commoned across
`p`, because `dB[p]` changes.

## R31 results

| population | control | factored dA | delta |
|---|---:|---:|---:|
| one CTA | `13.024 us` | `12.864 us` | `-0.160 us` (`1.23%`) |
| 170 CTAs | `18.816 us` | `18.304 us` | `-0.512 us` (`2.72%`) |

Both arms are bit-exact.  The 170-CTA comparison covers 2,785,280 FP32
outputs.  The candidate uses 255 registers, zero stack/local bytes and zero
`LDL/STL`.

| SASS family | control | candidate |
|---|---:|---:|
| total | 3,560 | 3,248 |
| `FMUL` | 512 | 0 |
| `FFMA` | 512 | 640 |
| `IMMA` | 256 | 256 |
| `LDSM` | 32 | 32 |
| `BAR` | 4 | 4 |
| `LDG / STS / LDS` | 49 / 42 / 172 | 49 / 42 / 172 |

The matched-control projection improves by `18.504 us`, below the required
`23.5 us`.  The candidate's screening projection remains `520.963 us`, versus
the current wide main at `318.8 us`.

## Decision

`NO_GO_FACTOR_DA_INTEGRATION`

The arithmetic grouping is proven and useful, but is not independently large
enough to invest in the full route.  Preserve it as a research arm and proceed
to the larger Q6 publisher topology test.

Evidence:

- `docs/task_workflow/evidence/nv-q6-oracle-factor-da-1-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-factor-da-170-20260831/result.json`

