# Decode Native Renderer DNR-3B Compound Emitter Result - 2026-06-20

## Verdict

`BLOCKED_DNR3B_GENERIC_EMITTER_CORRECT_BUT_NOT_ORACLE_SHAPED`

DNR-3B tested whether the existing generic AMD schedule-action emitter can be applied to the correct DNR-2 q8/Q4_K
gate/up stream. It can launch and preserve correctness on a synthetic fixture, but it does not produce an oracle-shaped
decode schedule.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3b_compound_emitter_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3b_compound_emitter_result.json
```

## Result

| gate | result |
|---|---:|
| DNR-3A plan present | pass |
| instruction stream changed | pass |
| generic scheduled candidate launches | pass |
| synthetic gate/up correctness | pass |
| oracle shape reached | fail |

Correctness sample:

| output | max abs |
|---|---:|
| gate | `0.00048828125` |
| up | `0.000274658203125` |

## Why It Blocks

| grouped count | DNR-2 native | DNR-3B generic emitted | hipcc/LLD oracle |
|---|---:|---:|---:|
| dot4 | `16` | `16` | `16` |
| global load | `22` | `22` | `11` |
| ds | `10` | `10` | `7` |
| branch | `0` | `0` | `5` |
| waitcnt | `17` | `30` | `20` |
| `s_clause` | `0` | `14` | `3` |
| `s_delay_alu` | `0` | `157` | `30` |

The generic emitter proves the plumbing can modify and launch the DNR-2 stream. It also proves that the current generic
policy is not the decode solution: it inserts marker spam, does not rewrite global loads, does not introduce branch/exec
policy, and does not reduce LDS/reduction traffic.

## Next

DNR-3C must be decode-specific:

1. coalesced Q4_K/q8 global-load rewrite toward grouped global loads `11`;
2. marker policy targeting `s_clause=3`, `s_delay_alu=30`, not generic insertion after every VALU;
3. branch/exec policy derived from lane roles;
4. LDS/reduction policy reducing ds ops toward oracle `7`;
5. register/live-range ledger tied to the emitted instruction stream.

No performance claim is made from DNR-3B.
