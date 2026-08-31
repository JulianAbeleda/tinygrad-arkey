# NV Q6 oracle publisher decision

## Test

The candidate replaces only the generated Q6 publisher. The control uses one
sixteen-trip runtime loop that combines quant, D and scale publication. The
candidate directly expresses pinned llama's three compile-time ownership
loops:

- sixteen quant row bands: 64 Q6 `LDG`, 32 Q6 `STS`;
- one duplicated-D pass: one Q6 `LDG`, one Q6 `STS`; and
- two scale half-passes: four Q6 `LDG`, two Q6 `STS`.

The candidate therefore has the oracle's 69 Q6 loads and 35 Q6 stores. Q8,
the 76-word shared layout, arithmetic, 256 `IMMA`, 32 `LDSM`, four barriers and
output ownership are fixed.

## R31 publisher-only results

| population | control | oracle publisher | delta |
|---|---:|---:|---:|
| one CTA | `12.992 us` | `8.224 us` | `-4.768 us` (`36.70%`) |
| 170 CTAs | `18.880 us` | `14.208 us` | `-4.672 us` (`24.75%`) |

Both arms are bit-exact. The saturated comparison checks all 2,785,280 FP32
outputs. The candidate uses 255 registers, zero stack/local bytes and zero
`LDL/STL`.

| family | control static | oracle publisher |
|---|---:|---:|
| `LDG` | 49 inside a runtime loop | 105 straight-line |
| `STS` | 42 inside a runtime loop | 71 straight-line |
| `LOP3` | 27 | 206 |
| `BRA / BRX / BSSY / BSYNC` | 6 / 2 / 1 / 1 | 1 / 0 / 0 / 0 |
| `IMMA / LDSM / BAR` | 256 / 32 / 4 | 256 / 32 / 4 |

The candidate's larger static body is the correct comparison: the control's
Q6 body executes sixteen times. The candidate full-body counts now match the
pinned oracle's 105 `LDG`, 71 `STS`, and approximately 205 `LOP3`.

The matched-control screening recovery is `168.852 us`, which clears the
`23.5 us` causal investment bar. Its absolute full-main screening projection
is `372.929 us`, still `54.129 us` slower than the current wide route at
`318.8 us`.

## Composition with dA factoring

The exact composed arm reaches `7.968 us` at one CTA and `14.112 us` at 170
CTAs. At saturation, dA factoring adds only `0.096 us` over publisher-only.
The composed projection is `369.460 us`, still `50.660 us` behind the current
wide route.

## Decision

`CAUSAL_PASS_ORACLE_PUBLISHER`; `NO_GO_BROAD_ROUTE_INTEGRATION`.

The publisher is the largest proven oracle-derived lever. Preserve the exact
implementation and apply its three-loop ownership topology to the existing
wide Stream-K producer. Do not promote the broad screening CTA itself.

Evidence:

- `docs/task_workflow/evidence/nv-q6-oracle-publisher-1-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-publisher-170-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-publisher-factor-da-1-20260831/result.json`
- `docs/task_workflow/evidence/nv-q6-oracle-publisher-factor-da-170-20260831/result.json`
