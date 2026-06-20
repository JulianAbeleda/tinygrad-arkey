# Decode Native Renderer DNR-3C7D Confirmation Result - 2026-06-20

## Verdict

`BLOCKED_DNR3C7D_C7C_SIGNAL_NOT_REPRODUCED_PARK_NATIVE_ROUTE`

DNR-3C7D confirms that the C7C issue-order candidate is correct and moves PMC counters in the expected direction,
but it does not reproduce a material timing win. The native DNR-3C route should be parked unless new oracle
resource metadata or SQTT body attribution appears.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c7d_confirmation_probe.py --timing-warmups 4 --timing-iters 12 --pmc-warmups 1 --timeout-s 360
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7d_confirmation_result.json
```

## Timing

Same-process interleaved timing against native, best static, and the best C7C candidate:

| variant | correct | median us | delta vs native | delta vs best static |
|---|---:|---:|---:|---:|
| native DNR-2 | yes | `280.247` | `0.000` | `+9.613` |
| best static DNR-3C6 | yes | `270.635` | `-9.613` | `0.000` |
| C7C unpack-dot + dsload_b128 | yes | `264.628` | `-15.619` | `-6.006` |

Oracle reference:

| row | us |
|---|---:|
| hipcc/LLD oracle | `93.540` |
| C7C best | `264.628` |
| remaining gap | `171.088` |

## PMC Confirmation

The C7C best candidate moves issue counters in the expected direction, but not enough to justify promotion without
material timing:

| comparison | SQ busy delta | SQ wait delta | VALU delta |
|---|---:|---:|---:|
| C7C vs native | `-0.644` | `-75.412` | `-2.920` |
| C7C vs best static | `-0.705` | `-6.332` | `-1.070` |

LDS/memory pass versus best static:

| metric | delta |
|---|---:|
| LDS active | `-0.000930` |
| LDS inst | `-0.000434` |
| LDS bank conflict | `0.000000` |
| SQ busy | `+0.110356` |

## Gates

| gate | result |
|---|---:|
| C7C partial-signal input present | yes |
| all variants correct | yes |
| PMC runs OK | yes |
| timing material | no |
| PMC confirms wait/busy direction | yes |
| reaches <=110% oracle | no |
| renderer default changed | no |

## Interpretation

The C7C issue-order idea is real as a small local improvement: it reduces static VALU count from `138` to `124`
versus best static, and PMC sees lower SQ busy/wait in the issue pass.

It is not enough:

- confirmed timing gain versus best static is only `6.006us`;
- confirmed timing gain versus native is only `15.619us`;
- the candidate still sits `171.088us` behind the oracle;
- the material gates were `>=30us` versus native or `>=15us` versus best static.

So the native decode renderer path is no longer blocked on Q4_K addressing, q8 addressing, scale/min extraction,
dot4 selection, gate/up correctness, resource ledger, PMC capture, or a first issue-order experiment. It is blocked
on not having a remaining native schedule lever large enough to justify promotion.

## Decision

Park DNR-3C native schedule work. Continue only if one of these appears:

- oracle VGPR/SGPR/live-range metadata showing a different resource envelope;
- SQTT body timeline mapped to q8 kernel PCs;
- a new decode primitive route beyond local native q8 schedule rewrites;
- a promotion target that accepts a small local native win instead of oracle-class performance.

No renderer defaults changed.
